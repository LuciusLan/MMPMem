import os
os.environ['CUDA_VISIBLE_DEVICES']="1"
import math
import json
import re
from pathlib import Path
import hashlib
from collections import defaultdict
from typing import List, Tuple, Iterable, Optional, Iterator
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
from datasets import load_dataset, Dataset as HFDataset

from ret_util import evaluate_retrieval, aggregate_eval_results, print_agg_summary

_RANK_SHARD_RE = re.compile(r"embeddings_rank(\d+)_(\d+)\.npz$")

def _parse_rank_shard(name: str) -> Tuple[int, int]:
    m = _RANK_SHARD_RE.match(name)
    if not m:
        return (0, 0)
    return (int(m.group(1)), int(m.group(2)))

def list_subsets(emb_root: Path) -> List[Path]:
    emb_root = emb_root.expanduser().resolve()
    candidate_dirs = set()
    for p in emb_root.rglob("embeddings_*.npz"):
        candidate_dirs.add(p.parent)
    return sorted(candidate_dirs, key=lambda d: str(d.relative_to(emb_root)))


def list_shards_nested(emb_root: Path) -> List[Path]:
    """
    Recursively find shards under emb_root and return a deterministic order:
    (subset_relpath, rank, shard_id).
    """
    emb_root = emb_root.expanduser().resolve()
    files = []
    for p in emb_root.rglob("embeddings_*.npz"):
        rel_parent = str(p.parent.relative_to(emb_root))
        rank, shard = _parse_rank_shard(p.name)
        files.append((rel_parent, rank, shard, p))
    files.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in files]

def load_first_dim(shards: List[Path]) -> int:
    z = np.load(shards[0], allow_pickle=True)
    emb = z["embeddings"]
    return int(emb.shape[1])


def load_paths_from_shards(shards: List[Path]) -> np.ndarray:
    paths: List[str] = []
    for sf in tqdm(shards, desc="Collecting paths"):
        z = np.load(sf, allow_pickle=True)
        p = z["paths"].astype(object).tolist()
        paths.extend(p)
    return np.array(paths, dtype=object)

def resolve_paths(paths_path: Path, indices: np.ndarray) -> List[str]:
    paths = np.load(paths_path, allow_pickle=True)
    return [str(paths[i]) for i in indices]

def build_flat_index_shard_order(shards: List[Path], index_path: Path, paths_path: Path):
    dim = load_first_dim(shards)
    index = faiss.IndexFlatIP(dim)

    all_paths = []
    for sf in tqdm(shards, desc="Building IndexFlatIP (shard-order)"):
        with np.load(sf, allow_pickle=True) as z:
            emb = z["embeddings"].astype(np.float32, copy=False)
            index.add(emb)
            all_paths.extend(z["paths"].astype(object).tolist())

    faiss.write_index(index, str(index_path))
    np.save(paths_path, np.array(all_paths, dtype=object))
    print(f"[FlatIP] Wrote index: {index_path} | vectors={index.ntotal}")
    print(f"[FlatIP] Wrote paths: {paths_path} | count={len(all_paths)}")

def iter_manifest_runs(emb_root: Path) -> Iterator[Tuple[Path, np.ndarray]]:
    """
    Yields (shard_path, rows_array) in strict manifest order, but batched
    by *consecutive runs* belonging to the same shard. This eliminates
    Python-level per-row looping.
    """
    emb_root = emb_root.expanduser().resolve()
    subsets = list_subsets(emb_root)

    for subset_dir in tqdm(subsets):
        # Read manifest.csv or merge rank manifests
        manifest = subset_dir / "manifest.csv"
        if manifest.exists():
            df = pd.read_csv(manifest, usecols=["shard", "row_in_shard", "global_index"])
            df = df.sort_values("global_index")
        else:
            rank_parts = sorted(subset_dir.glob("manifest_rank*.csv"))
            if not rank_parts:
                continue
            frames = [pd.read_csv(p, usecols=["shard", "row_in_shard", "global_index"]) for p in rank_parts]
            df = pd.concat(frames, ignore_index=True).sort_values("global_index")

        # Convert to NumPy once
        shards = df["shard"].to_numpy()
        rows = df["row_in_shard"].to_numpy(dtype=np.int64)

        # Run-length encode by shard to respect order while batching loads
        if len(shards) == 0:
            continue
        # indices where shard value changes
        change = np.flatnonzero(shards[1:] != shards[:-1]) + 1
        # run boundaries
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [len(shards)]))

        for s, e in zip(starts, ends):
            shard_rel = shards[s]  # same for this run
            yield (subset_dir / shard_rel, rows[s:e])

def _image_md5(x) -> str:
    """
    Deterministic MD5 of image content.
    Accepts numpy array (H,W,C) or PIL.Image.Image.
    """
    if "PIL" in type(x).__module__:
        arr = np.asarray(x)
    else:
        arr = np.asarray(x)
    arr = np.ascontiguousarray(arr)
    return hashlib.md5(arr.tobytes()).hexdigest()

def build_index_manifest_order_flat(emb_root: Path, index_path: Path, paths_path: Path):
    runs = list(iter_manifest_runs(emb_root))
    if not runs:
        raise RuntimeError("No manifest entries found.")

    # Determine dim from first run
    dim = None
    index = None
    all_paths: List[str] = []

    # Use all CPU threads for FAISS
    faiss.omp_set_num_threads(os.cpu_count()//2 or 1)

    for shard_path, rows in tqdm(runs, desc="Loading embed shards and build index"):
        with np.load(shard_path, allow_pickle=True) as z:
            emb = z["embeddings"]
            if dim is None:
                dim = int(emb.shape[1])
                index = faiss.IndexFlatIP(dim)
            take = emb[rows].astype(np.float32, copy=False)
            index.add(take)
            all_paths.extend(z["paths"][rows].astype(object).tolist())

    faiss.write_index(index, str(index_path))
    np.save(paths_path, np.array(all_paths, dtype=object))
    print(f"[FlatIP/manifest] Wrote index: {index_path} | vectors={index.ntotal}")
    print(f"[FlatIP/manifest] Wrote paths: {paths_path} | count={len(all_paths)}")

def build_id_to_path_manifest_order(emb_root: str | Path) -> np.ndarray:
    emb_root = Path(emb_root).expanduser().resolve()
    # subsets are directories that contain shards
    subset_dirs = sorted({p.parent for p in emb_root.rglob("embeddings_*.npz")},
                         key=lambda d: str(d.relative_to(emb_root)))
    id_to_path = []
    for subset in subset_dirs:
        m = subset / "manifest.csv"
        if m.exists():
            df = pd.read_csv(m, usecols=["global_index", "path"])
        else:
            parts = sorted(subset.glob("manifest_rank*.csv"))
            if not parts:
                continue
            df = pd.concat([pd.read_csv(p, usecols=["global_index", "path"]) for p in parts],
                           ignore_index=True)
        df = df.sort_values("global_index", kind="mergesort")  # stable
        id_to_path.extend(df["path"].tolist())
    return np.array(id_to_path, dtype=object)

# skvqa = HFDataset.load_from_disk('/wy_data/SKVQA/test')

# emb_root = Path('/wy_data/COCO/embeds_siglip').expanduser().resolve()
# index_path = Path('/wy_data/COCO/siglip_flatip.idx').expanduser().resolve()
# paths_path = Path('/wy_data/COCO/paths_siglip.npy').expanduser().resolve()
# index_path.parent.mkdir(parents=True, exist_ok=True)
# paths_path.parent.mkdir(parents=True, exist_ok=True)
# build_index_manifest_order_flat(
#                 emb_root=emb_root,
#                 index_path=index_path,
#                 paths_path=paths_path,
#             )

id_to_path = build_id_to_path_manifest_order(emb_root="/wy_data/InfoSeek/embeds")  # or _shard_order(...)
index: faiss.IndexFlatIP = faiss.read_index('/wy_data/InfoSeek/index/bge_flatip.idx')

model:CLIPModel = AutoModel.from_pretrained("/wy_data/bge_vl_large", trust_remote_code=True, local_files_only=True, attn_implementation="flash_attention_2", dtype=torch.bfloat16, device_map="cuda")
model = model.cuda()
model.set_processor("/wy_data/bge_vl_large")
processor = model.processor
model.eval()

# id_to_path = build_id_to_path_manifest_order(emb_root="/wy_data/COCO/embeds")  # or _shard_order(...)
# index = faiss.read_index('/wy_data/COCO/bge_flatip.idx')
# with open('/wy_data/COCO/circo_val.json') as f:
#     ret_quries = json.load(f)

# ret_results = []
# for row in tqdm(ret_quries):
#     query_img_id = row['reference_img_id']
#     query_img_id = f'{query_img_id:012d}'
#     query_text = 'Find a picture showing '+ row['shared_concept'] + ', but ' + row['relative_caption']
#     input_ids = model.processor.tokenizer(query_text, return_tensors='pt')['input_ids'].to('cuda')
#     pixel_values = model.processor(images=f'/wy_data/COCO/unlabeled2017/{query_img_id}.jpg', return_tensors='pt')['pixel_values'].to('cuda')
#     qemb_t = model.get_text_features(input_ids).detach().cpu().float().numpy()
#     qemb_i = model.get_image_features(pixel_values).detach().cpu().float().numpy()
#     qemb = qemb_t + qemb_i

#     D, I = index.search(qemb, 20)
#     topk_paths = [id_to_path[i] for i in I][0].tolist()
#     topk_paths = [int(e.split('/')[-1].replace('.jpg', '')) for e in topk_paths]
#     ret_results.append(evaluate_retrieval(topk_paths, row['gt_img_ids'], ks=[1, 5, 10, 20, 30]))

# agg = aggregate_eval_results(ret_results, with_ci=False)  # ks inferred from inputs
# print_agg_summary(agg)

# model_id = "/wy_data/siglip"   # try FixRes too: e.g., ...-patch16-384
# model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='cuda:0').eval()
# processor  = AutoProcessor.from_pretrained(model_id)

ret_queries = HFDataset.load_from_disk('/wy_data/SKVQA/IRCAP/train')

ret_results = []
qimg_map: dict[str, dict[str]] = defaultdict(lambda: {"question_ids": [], "ret_images": None, "scores": None})
unique_md5s = []
unique_images = []

original = []
for rowid, row in tqdm(enumerate(ret_queries), total=len(ret_queries), desc="Dedup"):
    md5 = _image_md5(row["image"])
    if qimg_map[md5]["question_ids"] == []:
        unique_md5s.append(md5)
        unique_images.append(rowid)
    qimg_map[md5]["question_ids"].append(row["question_id"])
    original.append((row["question_id"], md5))

U = len(unique_images)
    
for s in tqdm(range(0, U, 300), desc="Embed+search (streaming)"):
    e = min(s + 300, U)
    batch_imgs = unique_images[s:e]
    batch_imgs = [ret_queries[e]['image'] for e in batch_imgs]

    inputs = processor(images=batch_imgs, return_tensors="pt").to(model.device)
    #px = inputs["pixel_values"].to(model.device, non_blocking=True)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,):
        feats = model.get_image_features(**inputs)
        feats = F.normalize(feats, p=2, dim=-1)

    Q = feats.float().cpu().numpy()  # FAISS expects float32, shape [B, D]
    D, I = index.search(Q, 30)     # batched FAISS call for this block

    # write results back, aligning j to md5 at index s+j
    for j in range(e - s):
        md5 = unique_md5s[s + j]
        paths = [str(id_to_path[idx]) for idx in I[j]]
        qimg_map[md5]["ret_images"] = paths
        qimg_map[md5]["scores"] = D[j].tolist()

# 3) Per-query list aligned to input order
per_query = [
    {"question_id": qid, "query_md5": md5,
        "ret_images": qimg_map[md5]["ret_images"],
        "scores": qimg_map[md5]["scores"]}
    for (qid, md5) in original
]
    #ret_results.append(evaluate_retrieval(topk_paths, row['gt_img_ids'], ks=[1, 5, 10, 20]))

torch.save(per_query, '/wy_data/SKVQA/IRCAP/query_ret_imgs.pt')
# agg = aggregate_eval_results(ret_results, with_ci=False)  # ks inferred from inputs
# print_agg_summary(agg)
