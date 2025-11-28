from __future__ import annotations
import argparse
import os

def parse_args():
    p = argparse.ArgumentParser(description="Distributed CLIP embedding extraction with sharded outputs.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--model_name", type=str, default="/wy_data/bge_vl_large")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--shard_size", type=int, default=10_000)
    p.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 16) // 4))
    p.add_argument("--bf16", action="store_true", help="Enable BF16 autocast on CUDA.")
    return p.parse_args()

args = parse_args()



os.environ['CUDA_VISIBLE_DEVICES']=args.device
import math
from pathlib import Path
from typing import List, Tuple, Iterable, Optional
from glob import glob

import numpy as np
import csv
import pandas as pd
from PIL import Image, UnidentifiedImageError

import faiss
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
from transformers import AutoModel, CLIPModel, CLIPProcessor, AutoProcessor
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png",}

# -----------------------
# Utilities
# -----------------------
def collect_image_paths(root_dir: str | Path, extra_exts: Optional[List[str]] = None) -> List[Path]:
    """Recursively collect image paths and return a deterministic, cross-platform order."""
    root = Path(root_dir).expanduser().resolve()
    exts = set(IMG_EXTS)
    if extra_exts:
        exts |= {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extra_exts}
    paths = [p.resolve() for p in root.rglob("*") if p.suffix.lower() in exts]
    if not paths:
        raise RuntimeError(f"No images found under: {root}")
    return sorted(paths, key=lambda p: str(p))


def get_world_info_from_env():
    """Resolve rank, world_size, and local_rank from torchrun environment variables."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))  # fallback
    return rank, world_size, local_rank


# -----------------------
# Dataset
# -----------------------
class ImagePathDataset(Dataset):
    """Returns (PIL.Image|None, path:str, global_index:int)."""
    def __init__(self, paths: List[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return img, str(path), idx   # idx is the global (corpus) index
        except (UnidentifiedImageError, OSError):
            return None, str(path), idx


def make_collate(processor: CLIPProcessor):
    def collate(samples: List[Tuple[Image.Image | None, str, int]]):
        # Filter failed loads while preserving alignment
        samples = [(img, p, i) for img, p, i in samples if img is not None]
        if not samples:
            return {"pixel_values": None, "paths": [], "gidx": []}
        images, paths, gidx = zip(*samples)
        encoded = processor(images=list(images), return_tensors="pt")

        encoded.update({"paths": list(paths), "gidx": list(gidx)})
        return encoded
    return collate


# -----------------------
# Shard helpers
# -----------------------
def save_npz_shard(out_dir: Path, rank: int, shard_id: int,
                   emb_buf: List[np.ndarray], path_buf: List[str]) -> str:
    if not emb_buf:
        return ""
    embs = np.concatenate(emb_buf, axis=0)  # [M, D]
    shard_name = f"embeddings_rank{rank}_{shard_id:06d}.npz"
    shard_path = out_dir / shard_name
    np.savez_compressed(shard_path, embeddings=embs, paths=np.array(path_buf, dtype=object))
    return shard_name


def flush_if_needed(out_dir: Path, rank: int, shard_id: int, shard_size: int,
                    emb_buf: List[np.ndarray], path_buf: List[str], gidx_buf: List[int],
                    manifest_rows: List[tuple], global_counter: int) -> tuple[int, int]:
    """
    If buffers exceed shard_size, write a shard and update manifest rows.
    Returns: (new_shard_id, new_global_counter)
    """
    total = sum(arr.shape[0] for arr in emb_buf)
    while total >= shard_size:
        concat_emb = np.concatenate(emb_buf, axis=0)
        write_emb = concat_emb[:shard_size]
        rem_emb = concat_emb[shard_size:]

        write_paths = path_buf[:shard_size]
        rem_paths = path_buf[shard_size:]

        write_gidx = gidx_buf[:shard_size]
        rem_gidx = gidx_buf[shard_size:]

        shard_name = f"embeddings_rank{rank}_{shard_id:06d}.npz"
        np.savez_compressed(out_dir / shard_name,
                            embeddings=write_emb,
                            paths=np.array(write_paths, dtype=object))

        for row_in_shard, (gi, p) in enumerate(zip(write_gidx, write_paths)):
            # global_counter is not used for identity; we persist true global index 'gi'
            manifest_rows.append((int(gi), p, shard_name, row_in_shard))
            global_counter += 1

        # reset buffers with leftovers
        emb_buf = [rem_emb] if rem_emb.size > 0 else []
        path_buf = rem_paths
        gidx_buf = rem_gidx

        shard_id += 1
        total = sum(arr.shape[0] for arr in emb_buf)
    return shard_id, global_counter, emb_buf, path_buf, gidx_buf


# -----------------------
# Core distributed embedding
# -----------------------
def run(args):
    rank, world_size, local_rank = get_world_info_from_env()

    # Device binding
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # Prepare output
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Deterministic corpus and rank-local slice (no padding, no duplication)
    paths = collect_image_paths(args.data_root)
    ds_full = ImagePathDataset(paths)

    n = len(ds_full)
    chunk = math.ceil(n / world_size)
    start = rank * chunk
    end = min(start + chunk, n)
    if start >= end:
        # Degenerate rank with no data (can happen when world_size > number of batches)
        if rank == 0:
            print("Nothing to process; empty dataset or too many ranks.")
        # Still create an empty per-rank manifest so merge succeeds
        (out_dir / f"manifest_rank{rank}.csv").write_text("global_index,path,shard,row_in_shard\n", encoding="utf-8")
        return

    subset_indices = list(range(start, end))
    ds = Subset(ds_full, subset_indices)

    # 2) Model + processor

    # model:CLIPModel = AutoModel.from_pretrained("/wy_data/bge_vl_large", trust_remote_code=True, local_files_only=True, attn_implementation="flash_attention_2", dtype=torch.bfloat16, device_map="auto")
    # model = model.cuda()
    # model.set_processor("/wy_data/bge_vl_large")
    # model.eval()

    model_id = "/wy_data/siglip"   # try FixRes too: e.g., ...-patch16-384

    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16 if device=="cuda" else None).to(device).eval()
    proc  = AutoProcessor.from_pretrained(model_id)

    # 3) DataLoader
    #collate = make_collate(model.processor)
    collate = make_collate(proc)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate,
        persistent_workers=(args.num_workers > 0),
    )

    # 4) Inference loop with sharding
    dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else None
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if (dtype is not None)
        else torch.cuda.amp.autocast(enabled=False)
    )

    shard_id = 0
    emb_buf: List[np.ndarray] = []
    path_buf: List[str] = []
    gidx_buf: List[int] = []
    manifest_rows: List[tuple] = []
    global_counter = 0  # only for progress; identity is global_index from dataset

    with torch.no_grad(), autocast_ctx:
        for batch in tqdm(dl, total=len(dl), desc=f"[rank {rank}] Embedding"):
            px = batch["pixel_values"]
            bpaths = batch.pop("paths")
            gidx = batch.pop("gidx")
            if px is None or len(bpaths) == 0:
                continue

            batch = {k: v.to(model.device) for k,v in batch.items()}
            feats = model.get_image_features(**batch)  # [B, D]
            feats = F.normalize(feats, p=2, dim=-1)
            feats_np = feats.detach().cpu().half().numpy()

            emb_buf.append(feats_np)
            path_buf.extend(bpaths)
            gidx_buf.extend([int(i) for i in gidx])

            shard_id, global_counter, emb_buf, path_buf, gidx_buf = flush_if_needed(
                out_dir, rank, shard_id, args.shard_size,
                emb_buf, path_buf, gidx_buf, manifest_rows, global_counter
            )

    # Flush final partial shard
    if emb_buf:
        shard_name = save_npz_shard(out_dir, rank, shard_id, emb_buf, path_buf)
        if shard_name:
            for row_in_shard, (gi, p) in enumerate(zip(gidx_buf, path_buf)):
                manifest_rows.append((int(gi), p, shard_name, row_in_shard))

    # Per-rank manifest
    manifest_rank_path = out_dir / f"manifest_rank{rank}.csv"
    with open(manifest_rank_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["global_index", "path", "shard", "row_in_shard"])
        w.writerows(manifest_rows)

    # Rank-0 merge after all ranks are done
    # We avoid torch.distributed dependency for simplicity; a filesystem barrier is sufficient in most single-node runs.
    # If you run multi-node, ensure a proper cross-node barrier or merge manifests post hoc.
    if rank == 0:
        # Simple wait loop to ensure all manifests exist
        for r in range(world_size):
            path = out_dir / f"manifest_rank{r}.csv"
            while not path.exists():
                pass  # very brief spin; for robustness consider sleep()

        # Merge and sort by global_index
        frames = []
        for r in range(world_size):
            frames.append(pd.read_csv(out_dir / f"manifest_rank{r}.csv"))
        merged = pd.concat(frames, ignore_index=True)
        merged.sort_values("global_index", inplace=True)
        merged.to_csv(out_dir / "manifest.csv", index=False)
        print(f"Merged manifest written to: {out_dir / 'manifest.csv'}")




if __name__ == "__main__":
    
    run(args)