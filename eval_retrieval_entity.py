import os
os.environ['HF_HOME'] = '/data_external/hf_cache'

import json
import datasets
from tqdm import tqdm

with open('/data_external/InfoSeek/query_ret_imgs_qwen.jsonl') as f:
    prebuild_index = [json.loads(e) for e in f.readlines()]

train_ds = datasets.load_from_disk('/data_external/InfoSeek/train_gen_combined').with_format('torch')

with open("/data_external/InfoSeek/oven_entity_train.jsonl") as f:
    oven = f.readlines()
    oven = [json.loads(e) for e in oven]
with open("/data_external/InfoSeek/oven_entity_val.jsonl") as f:
    temp = f.readlines()
    temp = [json.loads(e) for e in temp]
oven.extend(temp)
oven_image_id_entity_map = {e['image_id']:e['entity_text'] for e in tqdm(oven)}

import math
from typing import Any, Dict, Iterable, List, Sequence


def evaluate_retrieval_mapped_targets(
    test_set: Sequence[Dict[str, Any]],
    ks: Iterable[int] = (1, 5, 10),
    require_num_relevant_for_ndcg: bool = False,
    return_per_example: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate retrieval when multiple database-A entries can map to the same
    database-B target, and every retrieved A-entry mapped to gt_id is counted
    as a correct hit.

    Each example in test_set must contain:
        {
            "retrieved_ids_processed": list[str],  # mapped B IDs of ranked A entries
            "gt_id": str
        }

    Optional:
        {
            "num_relevant": int  # total number of relevant A entries for this query
        }

    Metrics computed:
        - Hit@k
        - Precision@k
        - TP@k              (number of correct retrieved entries in top-k)
        - DCG@k
        - nDCG@k            only if num_relevant is available
    """
    ks = sorted(set(int(k) for k in ks))
    if not ks or min(ks) <= 0:
        raise ValueError("`ks` must contain positive integers.")
    if not test_set:
        raise ValueError("`test_set` must not be empty.")

    n = len(test_set)

    hit_sums = {k: 0.0 for k in ks}
    precision_sums = {k: 0.0 for k in ks}
    tp_sums = {k: 0.0 for k in ks}
    dcg_sums = {k: 0.0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}

    ndcg_count = 0
    per_example = []

    for idx, sample in tqdm(enumerate(test_set)):
        if "retrieved_ids_processed" not in sample or "gt_id" not in sample:
            raise KeyError(
                f"Example {idx} must contain 'retrieved_ids_processed' and 'gt_id'."
            )

        retrieved = sample["retrieved_ids_processed"]
        gt_id = sample["gt_id"]

        if not isinstance(retrieved, list):
            raise TypeError(f"Example {idx}: 'retrieved_ids_processed' must be a list.")
        if not isinstance(gt_id, str):
            raise TypeError(f"Example {idx}: 'gt_id' must be a string.")

        rel = [1 if x == gt_id else 0 for x in retrieved]
        num_relevant = sample.get("num_relevant", None)

        if num_relevant is not None:
            if not isinstance(num_relevant, int) or num_relevant < 0:
                raise TypeError(
                    f"Example {idx}: 'num_relevant' must be a non-negative integer."
                )

        example_metrics = {}

        for k in ks:
            top_rel = rel[:k]
            tp_at_k = sum(top_rel)
            hit_at_k = 1.0 if tp_at_k > 0 else 0.0
            precision_at_k = tp_at_k / k

            dcg_at_k = 0.0
            for rank, r in enumerate(top_rel, start=1):
                if r:
                    dcg_at_k += 1.0 / math.log2(rank + 1)

            hit_sums[k] += hit_at_k
            precision_sums[k] += precision_at_k
            tp_sums[k] += tp_at_k
            dcg_sums[k] += dcg_at_k

            example_metrics[f"tp@{k}"] = tp_at_k
            example_metrics[f"hit@{k}"] = hit_at_k
            example_metrics[f"precision@{k}"] = precision_at_k
            example_metrics[f"dcg@{k}"] = dcg_at_k

            if num_relevant is not None:
                ideal_hits = min(k, num_relevant)
                idcg_at_k = sum(
                    1.0 / math.log2(rank + 1)
                    for rank in range(1, ideal_hits + 1)
                )
                ndcg_at_k = dcg_at_k / idcg_at_k if idcg_at_k > 0 else 0.0
                ndcg_sums[k] += ndcg_at_k
                example_metrics[f"ndcg@{k}"] = ndcg_at_k

        if num_relevant is not None:
            ndcg_count += 1
        elif require_num_relevant_for_ndcg:
            raise ValueError(
                f"Example {idx} is missing 'num_relevant', which is required "
                f"for proper nDCG."
            )

        if return_per_example:
            per_example.append(example_metrics)

    results: Dict[str, Any] = {"num_examples": n}

    for k in ks:
        results[f"Hit@{k}"] = hit_sums[k] / n
        results[f"Precision@{k}"] = precision_sums[k] / n
        results[f"TP@{k}"] = tp_sums[k] / n
        results[f"DCG@{k}"] = dcg_sums[k] / n

    if ndcg_count > 0:
        for k in ks:
            results[f"nDCG@{k}"] = ndcg_sums[k] / ndcg_count
        results["num_examples_with_ndcg"] = ndcg_count
    else:
        results["nDCG"] = (
            "Not computed: proper nDCG requires per-query 'num_relevant'."
        )

    if return_per_example:
        results["per_example"] = per_example

    return results

err =0

step = 0
gather = []
for i, row in tqdm(enumerate(prebuild_index)):
    ret_images = row['ret_images']

    ret_images = [e.split('/')[-1].split('.')[0] for e in ret_images]

    ret_entities = []
    for e in ret_images:
        try:
            ret_entities.append(oven_image_id_entity_map[e])
        except KeyError:
            err +=1
    try:
        query_entity = oven_image_id_entity_map[row['query_md5']]
    except KeyError:
        err +=1
        continue
    temp = {'retrieved_ids_processed': ret_entities, 'gt_id': query_entity}
    gather.append(temp)
    step +=1
    if step == 1000000:
        break

results = evaluate_retrieval_mapped_targets(
    gather,
    ks=(1, 5, 10, 20, 30,50),
    return_per_example=False,
)

print(results)