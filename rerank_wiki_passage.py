import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
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
import itertools

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
from chunk_wiki import chunk_article


from qwen3_vl_reranker import Qwen3VLReranker, softmax_top_p_filter

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
        outputs.append({'section_text': temp, 'num_words':len(temp.split(' ')),  'title': title})
    return outputs

with open('/data_external/InfoSeek/query_ret_imgs_qwen.jsonl') as f:
    retrieved_collection = [json.loads(e) for e in f.readlines()]

evqa_kb = load_from_disk('/data_external/evqa/image_kb')
kb_titles = evqa_kb['title']
kb_title_set = set(evqa_kb['title'])
kb_title_map = {t:i for i, t in enumerate(evqa_kb['title'])}

with open("/data_external/InfoSeek/oven_entity_train.jsonl") as f:
    oven = f.readlines()
    oven = [json.loads(e) for e in oven]

with open("/data_external/InfoSeek/oven_entity_val.jsonl") as f:
    temp = f.readlines()
    temp = [json.loads(e) for e in temp]

oven.extend(temp)

with open("/data_external/InfoSeek/oven_entity_test.jsonl") as f:
    oven_test = f.readlines()
    oven_test = [json.loads(e) for e in oven_test]

with open("/data_external/InfoSeek/infoseek_train.jsonl") as f:
    infoseek = f.readlines()
    infoseek = [json.loads(e) for e in infoseek]

oven_image_id_entity_map = {e['image_id']:e['entity_text'] for e in tqdm(oven)}


gather_outputs = open('/data_external/InfoSeek/ret_img_gt_docs_p1.jsonl', 'w')

rc =0

query_batch_buffer = []
output_buffer = []
for row, ques in tqdm(zip(retrieved_collection[:100000], infoseek[:100000]), total=len(infoseek[:100000])):
    assert row['question_id'] == ques['data_id']
    ret_images = row['ret_images']
    ret_docs = []
    ret_imgs_filtered = []
    for ret_img in ret_images:
        if len(ret_docs) == 30:
            break
        ovid = ret_img.split('/')[-1].split('.')[0]
        try:
            in_evqa = kb_title_map[oven_image_id_entity_map[ovid]]
            ret_docs.append(oven_image_id_entity_map[ovid])
            ret_imgs_filtered.append(ret_img)

        except KeyError:
            continue
    
    query_retdocs_map = defaultdict(list)
    for i, doc_tit in enumerate(ret_docs):
        query_retdocs_map[doc_tit].append(i)


    fulltext_ret_docs = [evqa_kb[kb_title_map[e]] for e in query_retdocs_map.keys()]

    section_dicts = [{'id': doc['url'], 'title': doc['title'], 'sections': [{'title': sec_tit,  'text': sec_txt} for sec_tit, sec_txt in zip(doc['section_titles'], doc['section_texts'])]} for doc in fulltext_ret_docs]
    ret_doc_chunks = [chunk_article(sec_dic, target_tokens=200, min_tokens=150, max_tokens=300, overlap_sents=2, prefix_headings_in_text=True) for sec_dic in section_dicts]
    ret_img_for_chunks = [ret_imgs_filtered[e[0]] for e in query_retdocs_map.values()]
    #sections = [reorganize_section(e) for e in ret_docs]
    #sections = list(itertools.chain.from_iterable(sections))

    top5_chunks_for_docs = []
    for ret_img, ret_doc in zip(ret_img_for_chunks, ret_doc_chunks):
        inputs = {
            "instruction": "Retrieve text paragraphs that contain answer to the user's image related query.",
            "query": {"text": ques['question'], "image": ret_img},
            "documents": [
                    {"text": x.text} for x in ret_doc
                ],
            "fps": 1.0
        }
    #     query_batch_buffer.append(inputs)

    # if len(query_batch_buffer) < 32:
    #     continue

    # batch_scores = model.process_cross_query_batch(query_batch_buffer, mini_batch_size=128)
        scores = model.process(inputs, batch_size=128).softmax(0)
        scores = scores.sort(descending=True)

        top5 = scores.indices[:5].tolist()
        top5_chunks_for_docs.append({'chunk_text': [ret_doc[i].text for i in top5], 'chunk_idx':top5})

    #query_batch_buffer.clear()
    temp = {k:v for k,v in ques.items()}
    temp['ret_images'] = row['ret_images']
    temp['ret_scores'] = row['scores']
    temp['ret_doc_chunks_top5'] = {title: {'indices': ix, 'doc_chunks': top5_chunks} for (title, ix), top5_chunks in zip(query_retdocs_map.items(), top5_chunks_for_docs)}
    gather_outputs.write(json.dumps(temp))
    gather_outputs.write('\n')
    rc +=1

    if rc%50 == 0:
        torch.cuda.empty_cache()
    
pass