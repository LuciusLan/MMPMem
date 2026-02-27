from __future__ import annotations
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES']="2"
os.environ['HF_HOME'] = '/data_external/hf_cache'
os.environ['HF_HUB_OFFLINE']="1"
import math
from pathlib import Path
from typing import List, Tuple, Iterable, Optional
from glob import glob
import json
import io

import numpy as np
import csv
import pandas as pd
from PIL import Image, UnidentifiedImageError

#import faiss
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
from transformers import AutoModel, CLIPModel, CLIPProcessor, AutoProcessor
from tqdm import tqdm
import datasets
from datasets import load_dataset, Features, Sequence, Image as HFImage

from qwen3_vl_embedding import Qwen3VLEmbedder


# matched_files = sorted(glob("/data_external/evqa/merged_kb/matched_articles_parquet/*.parquet"))
# mismatched_files = sorted(glob("/data_external/evqa/merged_kb/mismatched_articles_parquet/*.parquet"))
with open("/data_external/InfoSeek/wiki_100_dict_v4.json") as f:
    infoseek_100k = json.load(f)

#evqa_kb = datasets.load_from_disk('/data_external/evqa/image_kb')
# with open("/data_external/evqa/image_kb/image_title_map.jsonl") as f:
#     image_title_map = f.readlines()
#     image_title_map = [json.loads(e) for e in image_title_map]

evqa_kb = datasets.load_dataset('/data_external/InfoSeek/merged_kb/matched_articles_parquet', split='train')

evqa_titles = evqa_kb['title']

# image_title_map = []
# for rid, row in enumerate(evqa_kb):
#     for i, img in enumerate(row['image_pixels']):
#         if img is not None:
#             image_title_map.append({'rowid': rid, 'title': row['title'], 'image_indice': i})


class ImageDS(Dataset):
    def __init__(self, data, source):
        super().__init__()
        self.data = data
        self.source = source
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_target = self.data[idx]
        image =  self.source[image_target['rowid']]['image_pixels'][image_target['image_indice']]
        b = image.get("bytes", None) if isinstance(image, dict) else None
        p = image.get("path", None) if isinstance(image, dict) else None

        if b is not None:
            return Image.open(io.BytesIO(bytes(b))).convert("RGB")
        if p:
            return Image.open(p).convert("RGB")
        
#image_ds = ImageDS(image_title_map, evqa_kb)

#image_dl = DataLoader(image_ds, batch_size=200, collate_fn=lambda batch: batch, num_workers=10, prefetch_factor=2)


model_id = "/data_external/bge_vl_large"   # try FixRes too: e.g., ...-patch16-384
model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True).to('cuda').eval()
model.set_processor(model_id)

#model = Qwen3VLEmbedder(model_name_or_path="/data_external/Qwen3-VL-Embedding-8B", attn_implementation="flash_attention_2", dtype=torch.bfloat16)

all_embeds=[]
for s in tqdm(range(0, len(evqa_titles), 300)):
    e = min(s + 300, len(evqa_titles))
    titles = evqa_titles[s:e]
    with torch.inference_mode():
        emb = model.encode(text=titles).cpu().float().numpy()
    all_embeds.append(emb)

embed_buffer = []
shard_id = 0
for step, batch in tqdm(enumerate(image_dl), total=len(image_dl)):
    embeds = model.encode(images=batch)
    # batch = [{"image":e} for e in batch]
    # embeds = model.process(batch)
    # embeds = embeds.detach().cpu().float().numpy()
    embed_buffer.append(embeds)
    if step == 0:
        print(embeds.shape)

    if len(embed_buffer) == 100:
        embed_buffer = np.concatenate(embed_buffer, axis=0)
        np.save(f"/data_external/InfoSeek/embeds_qwen38b_100k/shard{shard_id:03d}.npz", embed_buffer)
        embed_buffer = []
        shard_id += 1

if len(embed_buffer) > 0:
    embed_buffer = np.concatenate(embed_buffer, axis=0)
    np.save(f"/data_external/InfoSeek/embeds_qwen38b_100k/shard{shard_id:03d}.npz", embed_buffer)
    embed_buffer = []
    shard_id += 1





#kb_title = set(evqa_kb['title'])
# with open("/data_external/InfoSeek/infoseek_train.jsonl") as f:
#     infoseek = f.readlines()
#     infoseek = [json.loads(e) for e in infoseek]
#     infoseek = infoseek[:10000]

# with open("/data_external/InfoSeek/oven_entity_train.jsonl") as f:
#     oven = f.readlines()
#     oven = [json.loads(e) for e in oven]
    
# oven_image_id_entity_map = {e['image_id']:e['entity_text'] for e in tqdm(oven)}



pass