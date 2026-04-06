import json
from collections import Counter, defaultdict
from tqdm import tqdm

with open('/data_external/InfoSeek/infoseek_train.jsonl') as f:
    train = [json.loads(e) for e in f.readlines()]

train_questions = [e['question'] for e in train]
train_oven_ids = [e['image_id'] for e in train]
count_tq = Counter(train_questions)
coun_tq = {k: v for k,v in count_tq.items()}

with open('/data_external/InfoSeek/query_ret_imgs_qwen.jsonl') as f:
    ret_images = [json.loads(e) for e in f.readlines()]

with open("/data_external/InfoSeek/oven_entity_train.jsonl") as f:
    oven = f.readlines()
    oven = [json.loads(e) for e in tqdm(oven)]
with open("/data_external/InfoSeek/oven_entity_val.jsonl") as f:
    temp = f.readlines()
    temp = [json.loads(e) for e in temp]
oven.extend(temp)
oven_image_id_entity_map = {e['image_id']:e['entity_text'] for e in tqdm(oven)}


train_entities = [oven_image_id_entity_map[id] for id in train_oven_ids]
train_entities_unique = set(train_entities)
train_query_entity_map = defaultdict(list)
train_entity_query_map = defaultdict(list)

for row in tqdm(train):
    ent = oven_image_id_entity_map[row['image_id']]
    train_query_entity_map[ent].append(row['question'])
    train_entity_query_map[row['question']].append(ent)

notfound = 0
no_same_q = 0

total_ret = 0
for row_ret, row_train in tqdm(zip(ret_images, train),  total=len(train)):
    assert row_ret['question_id'] == row_train['data_id']
    top = row_ret['ret_images']
    total_ret+= len(top)
    top = [e.split('/')[-1].split('.')[0] for e in top]

    for id in top:
        try:
            ret_ent = oven_image_id_entity_map[id]
        except KeyError:
            notfound += 1
            continue
        
        if ret_ent not in train_entities_unique:
            notfound += 1
        else:
            train_q = row_train['question']
            train_q_ents = train_entity_query_map[train_q]
            if ret_ent not in train_q_ents:
                no_same_q += 1