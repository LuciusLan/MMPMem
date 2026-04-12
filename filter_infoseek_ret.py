import json
from collections import Counter, defaultdict
from tqdm import tqdm

import datasets

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
train_query_entity_map = defaultdict(set)
train_entity_query_map = defaultdict(set)
train_qe_combination_to_answer = defaultdict(str)

train_ent_query_combinations = []
for row in tqdm(train):
    ent = oven_image_id_entity_map[row['image_id']]
    train_query_entity_map[ent].add(row['question'])
    train_entity_query_map[row['question']].add(ent)
    train_ent_query_combinations.append(f'{ent}|{row["question"]}')
    if len(train_qe_combination_to_answer[f'{ent}|{row["question"]}']) == 0:
        train_qe_combination_to_answer[f'{ent}|{row["question"]}'] = row['answer']

count_ent_qs = Counter(train_ent_query_combinations)

# notfound = 0
# no_same_q = 0

# total_ret = 0
# for row_ret, row_train in tqdm(zip(ret_images, train),  total=len(train)):
#     assert row_ret['question_id'] == row_train['data_id']
#     top = row_ret['ret_images']
#     total_ret+= len(top)
#     top = [e.split('/')[-1].split('.')[0] for e in top]

#     for id in top:
#         try:
#             ret_ent = oven_image_id_entity_map[id]
#         except KeyError:
#             notfound += 1
#             continue
        
#         if ret_ent not in train_entities_unique:
#             notfound += 1
#         else:
#             train_q = row_train['question']
#             train_q_ents = train_entity_query_map[train_q]
#             if ret_ent not in train_q_ents:
#                 no_same_q += 1

distill_ds = datasets.load_from_disk('/data_external/InfoSeek/distill_train').with_format('torch')
distill_ids = distill_ds['data_id']
distill_id_map = {did: i for i, did in enumerate(distill_ids)}

train_ds = datasets.load_from_disk('/data_external/InfoSeek/train_gen_combined').with_format('torch')
train_id_map = {tid: i for i, tid in enumerate(train_ds['data_id'])}

for i, dis_row in tqdm(enumerate(distill_ds), total=len(distill_ds)):
    data_id = dis_row['data_id']
    data_id_int = int(data_id.split('_')[-1])
    train_idx = train_id_map[data_id]
    row_train = train_ds[train_idx]
    row_ret = ret_images[data_id_int]
    assert dis_row['data_id'] == row_train['data_id'] and dis_row['data_id'] == row_ret['question_id']
    
    dis_cand_count = dis_row['candidate_mask'].size(0)
    top = row_ret['ret_images'][:dis_cand_count]
    top = [e.split('/')[-1].split('.')[0] for e in top]
    top_entity = []
    identical_gt = []
    merged_scores = []
    merged_answers = []
    for j, (id, mask) in enumerate(zip(top, dis_row['candidate_mask'])):
        try:
            ret_ent = oven_image_id_entity_map[id]
        except KeyError:
            #notfound += 1
            ret_ent = f'unknown{j}'

        top_entity.append(ret_ent)
        if ret_ent in train_entities_unique:
            train_q = row_train['question']
            train_q_ents = train_entity_query_map[train_q]
            if ret_ent in train_q_ents:
                identical_gt.append(train_qe_combination_to_answer[f'{ret_ent}|{train_q}'])
            else:
                identical_gt.append(None)
        else:
            identical_gt.append(None)
    pass
pass