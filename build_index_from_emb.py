from pathlib import Path
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

res = faiss.StandardGpuResources()
co = faiss.GpuClonerOptions()
co.use_cuvs = True               # request cuVS backend

query_embeds = np.load('/data_external/InfoSeek/query_embeds.npz.npy')
with open('/data_external/InfoSeek/infoseek_train.jsonl') as f:
    ret_queries = [json.loads(e) for e in f.readlines()]

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

id_to_path = build_id_to_path_manifest_order(emb_root="/data_external/InfoSeek/corpus_emb_qwen3")  # or _shard_order(...)
path_to_dir = {}
for row in id_to_path:
    img_id = row.split('/')[-1].split('.')[0]
    path_to_dir[img_id] = row
index_cpu: faiss.IndexFlatIP = faiss.read_index('/data_external/InfoSeek/corpus_qwen3_flatip.idx')
index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu, co)

ret_results = []
qimg_map: dict[str, dict[str]] = defaultdict(lambda: {"question_ids": [], "ret_images": None, "scores": None})
unique_md5s = []
unique_images = []

original = []
for rowid, row in tqdm(enumerate(ret_queries), total=len(ret_queries), desc="Dedup"):
    #md5 = _image_md5(row["image"])
    md5 = row['image_id']
    if qimg_map[md5]["question_ids"] == []:
        unique_md5s.append(md5)
        unique_images.append(rowid)
    qimg_map[md5]["question_ids"].append(row["data_id"])
    original.append((row["data_id"], md5))

U = len(unique_images)

assert U == query_embeds.shape[0]
#U =2000
query_embs = []

top1_mutual = 0
for s in tqdm(range(0, U, 400), desc="Embed+search (streaming)"):
    e = min(s + 400, U)
    batch_imgs = unique_images[s:e]
    batch_imgs = [ret_queries[e]['image_id'] for e in batch_imgs]
    batch_imgs_path = [path_to_dir[e] for e in batch_imgs]
    
    #feats = F.normalize(feats, p=2, dim=-1).float().cpu().numpy()

    #Q = feats.float().cpu().numpy()  # FAISS expects float32, shape [B, D]

    Q = query_embeds[s:e, :]
    D, I = index_gpu.search(Q, 51)     # batched FAISS call for this block

    # write results back, aligning j to md5 at index s+j
    for j in range(e - s):
        md5 = unique_md5s[s + j]
        paths = [str(id_to_path[idx]) for idx in I[j]]
        if paths[0] == batch_imgs_path[j]:
            paths = paths[1:]
            scores = D[j].tolist()[1:]
            top1_mutual += 1
        else:
            paths = paths[:-1]
            scores = D[j].tolist()[:-1]
        qimg_map[md5]["ret_images"] = paths
        qimg_map[md5]["scores"] = scores


# 3) Per-query list aligned to input order
per_query = [
    {"question_id": qid, "query_md5": md5,
        "ret_images": qimg_map[md5]["ret_images"],
        "scores": qimg_map[md5]["scores"]}
    for (qid, md5) in original
]
    #ret_results.append(evaluate_retrieval(topk_paths, row['gt_img_ids'], ks=[1, 5, 10, 20]))


with open('/data_external/InfoSeek/query_ret_imgs_qwen.jsonl', 'w') as f:
    for row in per_query:
        f.write(json.dumps(row))
        f.write('\n')

pass