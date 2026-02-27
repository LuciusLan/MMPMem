import os
os.environ['CUDA_VISIBLE_DEVICES']="1"
os.environ['HF_HOME'] = '/data_external/hf_cache'
os.environ['HF_HUB_OFFLINE']="1"
import math
import json
import re
from pathlib import Path
import hashlib
from collections import defaultdict
from typing import List, Tuple, Iterable, Optional, Iterator
from glob import glob
import io

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
from datasets import load_dataset, Dataset as HFDataset, load_from_disk

from ret_util import evaluate_retrieval, aggregate_eval_results, print_agg_summary

# res = faiss.StandardGpuResources()
# co = faiss.GpuClonerOptions()
#co.use_cuvs = True               # request cuVS backend



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
    #return np.array(id_to_path, dtype=object)
    return id_to_path

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




# embed_paths = glob('/data_external/InfoSeek/embeds_qwen38b_100k/*.npy')
# embed_paths = sorted(embed_paths)

# embeds = [np.load(e) for e in embed_paths]
# embeds = np.concatenate(embeds, axis=0)

# embeds = np.load('/data_external/InfoSeek/wiki100k_titles.npz.npy')
# print('norm')
# faiss.normalize_L2(embeds)

# index = faiss.IndexFlatIP(768)
# print('train')
# index.add(embeds)

# print("train finished")
# faiss.write_index(index, "/data_external/InfoSeek/wiki100k_titleonly_index.idx")


# id_to_path = build_id_to_path_manifest_order(emb_root="/wy_data/wikiweb2m/embeds")  # or _shard_order(...)
# path_to_dir = {}
# for row in id_to_path:
#     img_id = row.split('/')[-1].split('.')[0]
#     path_to_dir[img_id] = row

evqa_kb = load_dataset('/data_external/InfoSeek/merged_kb')['train']

#with open("/data_external/evqa/image_kb/image_title_map.jsonl") as f:
    # image_title_map = f.readlines()
    # image_title_map = [json.loads(e) for e in image_title_map]

image_title_map = []
for rid, row in enumerate(evqa_kb):
    for i, img in enumerate(row['image_pixels']):
        if img is not None:
            image_title_map.append({'rowid': rid, 'title': row['title'], 'image_indice': i})

faiss.omp_set_num_threads(os.cpu_count()//2 or 1)

index: faiss.IndexFlatIP = faiss.read_index('/data_external/InfoSeek/wiki100k_bge_index.idx')
#index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu, co)

# model:CLIPModel = AutoModel.from_pretrained("/data_external/bge_vl_large", trust_remote_code=True, local_files_only=True, attn_implementation="flash_attention_2", dtype=torch.bfloat16, device_map="cuda")
# model = model.cuda()
# model.set_processor("/data_external/bge_vl_large")
# processor = model.processor
# model.eval()

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

#model_id = "/wy_data/siglip"   # try FixRes too: e.g., ...-patch16-384

model_id = "/data_external/bge_vl_large"
model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True, local_files_only=True).eval()
model.set_processor(model_id)
processor  =model.processor


from qwen3_vl_embedding import Qwen3VLEmbedder
#model = Qwen3VLEmbedder(model_name_or_path="/data_external/Qwen3-VL-Embedding-8B", attn_implementation="flash_attention_2", dtype=torch.bfloat16)

#ret_queries = HFDataset.load_from_disk('/wy_data/SKVQA/IRCAP/train')

pathss = np.load("/data_external/InfoSeek/paths_bge.npy", allow_pickle=True)

iid_to_path = {e.split('/')[-1].split('.')[0]: e for e in tqdm(pathss)}

with open('/data_external/InfoSeek/infoseek_val.jsonl') as f:
    ret_queries = [json.loads(e) for e in f.readlines()]

ret_queries = ret_queries[:10000]

#evqa_kb = load_from_disk('/data_external/evqa/image_kb')
kb_titles = evqa_kb['title']
kb_title_set = set(evqa_kb['title'])
kb_title_map = {t:i for i, t in enumerate(evqa_kb['title'])}
with open("/data_external/InfoSeek/infoseek_train.jsonl") as f:
    infoseek = f.readlines()
    infoseek = [json.loads(e) for e in infoseek]
    infoseek = infoseek[:10000]

with open("/data_external/InfoSeek/oven_entity_train.jsonl") as f:
    oven = f.readlines()
    oven = [json.loads(e) for e in oven]

with open("/data_external/InfoSeek/oven_entity_val.jsonl") as f:
    xx = f.readlines()
    xx = [json.loads(e) for e in xx]

oven.extend(xx)
    
oven_image_id_entity_map = {e['image_id']:e['entity_text'] for e in tqdm(oven)}
oven_reverse = defaultdict(list)
for x, y in oven_image_id_entity_map.items():
    oven_reverse[y].append(x)

ret_results = []
# qimg_map: dict[str, dict[str]] = defaultdict(lambda: {"question_ids": [], "ret_images": None, "scores": None})
# unique_md5s = []
# unique_images = []

# original = []
# for rowid, row in tqdm(enumerate(ret_queries), total=len(ret_queries), desc="Dedup"):
#     #md5 = _image_md5(row["image"])
#     md5 = row['image_id']
#     if qimg_map[md5]["question_ids"] == []:
#         unique_md5s.append(md5)
#         unique_images.append(rowid)
#     qimg_map[md5]["question_ids"].append(row["data_id"])
#     original.append((row["data_id"], md5))

# U = len(unique_images)

U = len(ret_queries)

def _dedupe_preserve_order(xs: list[str]) -> List[str]:
    """Remove duplicates while preserving first-occurrence order."""
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

hit = {1: 0, 5: 0, 10: 0, 20: 0, 30:0, 50:0, 100:0}
rerank_hit = {1: 0, 5: 0, 10: 0, 20: 0, 30:0, 50:0, 100:0}

count_filtered = 0

retrieved_collection = []
for s in tqdm(range(0, U, 300), desc="Embed+search (streaming)"):
    e = min(s + 300, U)
    batch_imgs = ret_queries[s:e]
    #batch_imgs = [ret_queries[e]['image_id'] for e in batch_imgs]
    batch_imgs_path = [iid_to_path[e['image_id']] for e in batch_imgs]
    #batch_imgs_path = [iid_to_path[e] for e in batch_imgs]
    batch_imgs_path= [e.replace('wy_data', 'data_external') for e in batch_imgs_path]

    
    #px = inputs["pixel_values"].to(model.device, non_blocking=True)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,):
        inputs = processor(images=batch_imgs_path, return_tensors="pt").to(model.device)
        feats = model.get_image_features(**inputs)
        # batch = [{"image":e} for e in batch_imgs_path]
        # feats = model.process(batch)

        feats = F.normalize(feats, p=2, dim=-1)

    Q = feats.float().cpu().numpy()  # FAISS expects float32, shape [B, D]
    D, I = index.search(Q, 200)     # batched FAISS call for this block

    #retrieved = [kb_titles[i] for i in I]
    retrieved = [[image_title_map[ii] for ii in row] for row in I]

    # top1_images= [r[0] for r in retrieved]
    # top1_images = [evqa_kb[e['rowid']]['image_pixels'][e['image_indice']] for e in top1_images]

    # top1_images = [Image.open(io.BytesIO(bytes(e['bytes']))).convert("RGB") for e in top1_images]
    # qimages = [Image.open(q) for q in batch_imgs_path]
    # qmd5 = [_image_md5(q) for q in qimages]
    # top1_images_md5 = [_image_md5(q) for q in top1_images]

    # retrieved_filtered = []
    # for i, (qq, rr) in enumerate(zip(qmd5, top1_images_md5)):
    #     if qq == rr:
    #         retrieved_filtered.append(retrieved[i][1:])
    #         count_filtered += 1
    #     else:
    #         retrieved_filtered.append(retrieved[i][:-1])

    gt = [e.split('/')[-1].split('.')[0] for e in batch_imgs_path]
    gt = [oven_image_id_entity_map[e] for e in gt]

    for ret_row, gt_tit, query, qimg in zip(retrieved, gt, ret_queries, batch_imgs_path):
        ret_titles = [e['title'] for e in ret_row]
        #ret_titles = ret_row
        
        ret_titles = _dedupe_preserve_order(ret_titles)
        
        retrieved_collection.append({'query': query['question'], 'qimg': qimg, 'retrieved_titles': ret_titles[:30], 'gt_title': gt_tit})
        if gt_tit == ret_titles[0]:
            hit[1] +=1
        if gt_tit in ret_titles[:5]:
            hit[5] += 1
        if gt_tit in ret_titles[:10]:
            hit[10] += 1
        if gt_tit in ret_titles[:20]:
            hit[20] += 1
        if gt_tit in ret_titles[:30]:
            hit[30] += 1
        if gt_tit in ret_titles[:50]:
            hit[50] += 1
        if gt_tit in ret_titles[:100]:
            hit[100] += 1

        # by image
        # bge {1: 2052, 5: 3966, 10: 4745, 20: 5515, 30: 5907, 50: 6391, 100: 7017}
        # qwen3_2b {1: 2470, 5: 4632, 10: 5452, 20: 6169, 30: 6554, 50: 6978, 100: 7480}
        # qwen3_8b {1: 2887, 5: 5039, 10: 5767, 20: 6466, 30: 6849, 50: 7308, 100: 7829}

        # By article (dedup)

    # write results back, aligning j to md5 at index s+j
    # for j in range(e - s):
    #     md5 = unique_md5s[s + j]
    #     paths = [str(id_to_path[idx]) for idx in I[j]]
    #     if paths[0] == batch_imgs_path[j]:
    #         paths = paths[1:]
    #         scores = D[j].tolist()[1:]
    #     else:
    #         paths = paths[:-1]
    #         scores = D[j].tolist()[:-1]
    #     qimg_map[md5]["ret_images"] = paths
    #     qimg_map[md5]["scores"] = scores


print(hit)

from qwen3_vl_reranker import Qwen3VLReranker

# Specify the model path
model_name_or_path = "/data_external/Qwen3-VL-Reranker-2B"

# Initialize the Qwen3VLEmbedder model
model = Qwen3VLReranker(model_name_or_path=model_name_or_path)


def reorganize_section(input_article):
    title = input_article['title']
    outputs = []
    for sec, text in zip(input_article['section_titles'], input_article['section_texts']):
        if len(text) < 10:
            continue
        if not sec:
            continue
        temp = f"{title}: {sec}: {text}"
        outputs.append({'section_text': temp, 'title': title})
    return outputs


import itertools
for row in tqdm(retrieved_collection):
    ret_docs = [kb_title_map[e] for e in row['retrieved_titles']]
    ret_docs = [evqa_kb[e] for e in ret_docs]

    sections = [reorganize_section(e) for e in ret_docs]
    sections = list(itertools.chain.from_iterable(sections))
    inputs = {
        "instruction": "Retrieve text paragraphs relevant to the user's query.",
            "query": {"text": row['query'], "image": row['qimg']},
        "documents": [
            {"text": x['section_text']} for x in sections
            ],
        "fps": 1.0
    }
    scores = model.process(inputs)

    scores = torch.tensor(scores, device='cpu')
    scores = scores.sort(descending=True)

    top500 = scores.indices[:500]
    top500_sections = [sections[e] for e in top500]
    top500_titles = [e['title'] for e in top500_sections]
    rerank_titles = _dedupe_preserve_order(top500_titles)
    gt_tit = row['gt_title']
    try:
        if gt_tit == rerank_titles[0]:
            rerank_hit[1] +=1
        if gt_tit in rerank_titles[:5]:
            rerank_hit[5] += 1
        if gt_tit in rerank_titles[:10]:
            rerank_hit[10] += 1
        if gt_tit in rerank_titles[:20]:
            rerank_hit[20] += 1
        if gt_tit in rerank_titles[:30]:
            rerank_hit[30] += 1
    except IndexError:
        pass
pass

# 3) Per-query list aligned to input order
per_query = [
    {"question_id": qid, "query_md5": md5,
        "ret_images": qimg_map[md5]["ret_images"],
        "scores": qimg_map[md5]["scores"]}
    for (qid, md5) in original
]
    #ret_results.append(evaluate_retrieval(topk_paths, row['gt_img_ids'], ks=[1, 5, 10, 20]))


with open('/wy_data/InfoSeek/query_ret_imgs.jsonl', 'w') as f:
    for row in per_query:
        f.write(json.dumps(row))
        f.write('\n')
#torch.save(per_query, '/wy_data/InfoSeek/query_ret_imgs.pt')
# agg = aggregate_eval_results(ret_results, with_ci=False)  # ks inferred from inputs
# print_agg_summary(agg)
