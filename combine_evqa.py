import json
import os
from glob import glob
from pathlib import Path

import pandas as pd
import datasets
from PIL import Image

gld_imgs = glob('/data_external/gldv2/val_images/images/*/*/*/*.jpg')
gld_img_map = {}
for img in gld_imgs:
    basename = os.path.basename(img)
    basename = basename.split('.')[0]
    gld_img_map.update({basename: img})

evqa_test_gld = pd.read_csv('/data_external/evqa/test.mapped.csv')


ds = []
for rid, row in evqa_test_gld.iterrows():
    temp = {}
    if row['num_resolved_images'] == 0:
        image_id = row['dataset_image_ids']
        try:
            image_file = gld_img_map[image_id]
        except KeyError:
            continue
        row['resolved_image_paths'] = image_file
        row['num_resolved_images'] = 1
    else:
        image_file = row['resolved_image_paths']
    image = Image.open(image_file)
    temp['image'] = image
    temp['data_id'] = row['dataset_image_ids']
    temp['question'] = row['question']
    temp['question_type'] = row['question_type']
    temp['answer'] = row['answer']
    temp['answer_eval'] = row['answer']
    ds.append(temp)




from datasets import Features, Sequence, Value, Image as HFImage
features = Features({
    "data_id": Value("string"),
    "question": Value("string"),
    "question_type": Value("string"),
    "answer": Value("string"),
    "answer_eval": Value("string"),
    "image": HFImage(decode=True)
})

print(len(ds))
dataset = datasets.Dataset.from_list(ds, features=features)

dataset.save_to_disk('/data_external/evqa/evqa_test_withimg')
pass