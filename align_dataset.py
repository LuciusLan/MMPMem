import os
os.environ["HF_HOME"]='/data_external/hf_cache'
import json
from glob import glob
import datasets
datasets.disable_caching()
from datasets import Image as HFImage, Features, List, Value
from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import random
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained('/wyy/models/Qwen3-VL-8B-Instruct')

with open('/data_external/InfoSeek/infoseek_val.jsonl') as f:
    train_ds = [json.loads(e) for e in f.readlines()]

#train_ds = train_ds[:400000]
#train_ds = random.sample(train_ds, 1000)
gather = []

imageid_path_map = {}

for dir in ['01', '02', '03','04','06','07','08','09']:
    img_list = glob(f"/data_external/InfoSeek/{dir}/oven*")
    for img in tqdm(img_list):
        iid = img.split('/')[-1].split('.')[0]
        imageid_path_map[iid] = img


def gen():
    #for sid in shard_n:
        #current = input_ds[sid]
    for row in tqdm(train_ds):
        qid = row['data_id']
        # try:
        #     teacher_logits = torch.load(f'/data_external/InfoSeek/DataStore_rep_gen/{qid}.pt')
        # except FileNotFoundError:
        #     #print(f'qid{qid} teacher logit not found')
        #     continue
        #if row['image'].mode not in ['RGB', 'RGBA']:
        image = imageid_path_map[row['image_id']]
        image = Image.open(image).convert('RGB')
        row['image'] = image
        row['utype'] =  row['data_split']
        row.pop('data_split')
        # row['scores'] = teacher_logits['ret_scores']
        # row['per_ev_top_ids'] = pad_sequence(teacher_logits['per_ev_top_ids'], batch_first=True, padding_value=-1)
        # row['per_ev_top_logps'] =  pad_sequence(teacher_logits['per_ev_top_logps'], batch_first=True, padding_value=-999)
        # row['per_ev_tail'] =  pad_sequence(teacher_logits['per_ev_tail'], batch_first=True, padding_value=-999)
        yield row

new_feats = Features({
    "data_id": Value("string"),
    "image_id": Value("string"),
    "question": Value("string"),
    "answer": List(Value("string")),
    "answer_eval": List(Value("string")),
    "data_split": Value("string"),
    # "scores": List(Value("float16")),
    # "per_ev_top_ids": List(List(List(Value("int64")))),
    # "per_ev_top_logps": List(List(List(Value("float16")))),
    # "per_ev_tail": List(List(Value("float16"))),
    "image": HFImage(decode=True),
    "utype": Value("string")
})

shards = [
    train_ds[:9000],
    train_ds[9000:18000],
    train_ds[18000:27000],
    train_ds[27000:36000],
    train_ds[36000:45000],
    train_ds[45000:54000],
    train_ds[54000:63000],
    train_ds[63000:],
]
combined_ds = datasets.Dataset.from_generator(gen, features=new_feats)
print(len(combined_ds))
combined_ds.save_to_disk('/data_external/InfoSeek/val_full')

pass