import json
from glob import glob
import datasets
from datasets import Image as HFImage, Features, List, Value
from PIL import Image
import torch
from tqdm import tqdm
import random


with open('/data_external/InfoSeek/infoseek_val.jsonl') as f:
    train_ds = [json.loads(e) for e in f.readlines()]

#train_ds = train_ds[:30000]
train_ds = random.sample(train_ds, 1000)
gather = []

imageid_path_map = {}

for dir in ['01', '02', '03','04','06','07','08','09']:
    img_list = glob(f"/data_external/InfoSeek/{dir}/oven*")
    for img in img_list:
        iid = img.split('/')[-1].split('.')[0]
        imageid_path_map[iid] = img


def gen():
    for row in tqdm(train_ds):
        qid = row['data_id']
        #if row['image'].mode not in ['RGB', 'RGBA']:
        image = imageid_path_map[row['image_id']]
        image = Image.open(image).convert('RGB')
        #if image_loaded
        # try:
        #     teacher_logits = torch.load(f'/data_external/InfoSeek/DataStore_rep/{qid}.pt')
        # except FileNotFoundError:
        #     #print(f'qid{qid} teacher logit not found')
        #     continue
        # row['mix_idx'] = teacher_logits['mix_idx']
        # row['mix_logp'] = teacher_logits['mix_logp']
        # row['tail_logp'] = teacher_logits['tail_logp']
        row['image'] = image
        yield row

new_feats = Features({
    "data_id": Value("string"),
    "image_id": Value("string"),
    "question": Value("string"),
    "answer": List(Value("string")),
    "answer_eval": List(Value("string")),
    "data_split": Value("string"),
    # "mix_idx": List(List(Value("int64"))),
    # "mix_logp": List(List(Value("float16"))),
    # "tail_logp": List(Value("float16")),
    "image": HFImage(decode=True)
})
combined_ds = datasets.Dataset.from_generator(gen, features=new_feats)
print(len(combined_ds))
combined_ds.save_to_disk('/data_external/InfoSeek/val_combined')

pass