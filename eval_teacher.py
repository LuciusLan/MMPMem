import os
import hashlib

import torch
import datasets
from PIL import Image
from datasets import Image as HFImage
from datasets import Sequence as HFSequence
import faiss
from transformers import AutoModel, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from typing import List, Sequence, Union, Dict, Any
from tqdm import tqdm

Hash = str
RetType = Union[List[Hash], List[List[Hash]]]
GtType = Union[List[Hash], List[List[Hash]]]

# model = teacher = Qwen2_5_VLForConditionalGeneration.from_pretrained("/wy_data/qwen2.5_vl_7b_instruct", local_files_only=True, attn_implementation="flash_attention_2", dtype=torch.bfloat16, device_map='cuda')
# #teacher = Qwen2_5_VLForConditionalGeneration.from_pretrained("/wy_data/qwen2.5_vl_7b_instruct", local_files_only=True, attn_implementation="flash_attention_2", dtype=torch.bfloat16).cuda()
# processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')

retriever = AutoModel.from_pretrained("/wy_data/bge_vl_large", trust_remote_code=True, local_files_only=True, attn_implementation="flash_attention_2", dtype=torch.bfloat16, device_map='cuda')
#retriever = retriever.cuda()
retriever.set_processor("BAAI/BGE-VL-large")
index = faiss.read_index('/wy_data/MRAG/bge.index')

master_dataset = datasets.load_from_disk('/wy_data/MRAG/train_test')

train_ds = master_dataset['train']
test_ds = master_dataset['test']
test_ds = test_ds.cast_column("image", HFImage(decode=True))
test_ds = test_ds.cast_column("gt_images", HFSequence(HFImage(decode=True)))

train_ds = train_ds.cast_column("image", HFImage(decode=True))
train_ds = train_ds.cast_column("gt_images", HFSequence(HFImage(decode=True)))

image_corpus = os.listdir('/wy_data/MRAG/image_corpus')
image_corpus = ['/wy_data/MRAG/image_corpus/'+e for e in image_corpus]

prompt_base = "Answer the image related question. Be concise, output the answer only, without any additional words."
prompt_teacher = "Answer the image related question. Be concise, output the answer only, without any additional words. An additional image is provided for your reference, but it is not guaranteed to be relevant to original question."

def process_input(question, image, ret_image=None):

    if not ret_image:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt_base + "\nQuestion: " +question},
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt_teacher + "\nQuestion: " +question + "Additional Image:\n"},
                    {
                        "type": "image",
                        "image": ret_image,
                    }
                ],
            }
        ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    return inputs



def _as_batched(x: Union[List[str], List[List[str]]]) -> List[List[str]]:
    """Ensure inputs are in batched form: List[List[str]]."""
    if not x:
        return []
    if isinstance(x[0], list):
        return x  # already batched
    return [x]   # wrap single query

def _dedup_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out

def evaluate_retrieval(
    ret_hash: RetType,
    gt_hash: GtType,
    ks: Sequence[int] = (1, 5, 10, 20, 30),
    *,
    case_insensitive: bool = True,
    deduplicate_retrieved: bool = True,
) -> Dict[str, Any]:
    """
    Compute recall@k and hit_rate@k for retrieval results against ground truth.

    Definitions:
      • For query i and cutoff k, R_i(k) = top-k retrieved doc hashes (postprocessing applied).
      • G_i = set of ground-truth doc hashes for query i.
      • recall@k = |R_i(k) ∩ G_i| / |G_i|.
      • hit_rate@k = 1{ |R_i(k) ∩ G_i| > 0 }.
      Aggregation:
      • macro_recall@k = mean_i recall@k_i
      • micro_recall@k = (Σ_i |R_i(k) ∩ G_i|) / (Σ_i |G_i|)
      • hit_rate@k = mean_i hit_rate@k_i

    Inputs can be a single query (list[str]) or batched (list[list[str]]).
    """
    # Normalize inputs to batched lists
    batched_ret = _as_batched(ret_hash)
    batched_gt = _as_batched(gt_hash)

    if len(batched_ret) != len(batched_gt):
        raise ValueError(f"ret_hash has {len(batched_ret)} queries but gt_hash has {len(batched_gt)}.")

    # Preprocess: normalization and optional deduplication
    def _norm(h: str) -> str:
        return h.lower() if case_insensitive else h

    processed_ret = []
    processed_gt = []
    for r, g in zip(batched_ret, batched_gt):
        r_proc = [_norm(x) for x in r]
        g_proc = [_norm(x) for x in g]
        if deduplicate_retrieved:
            r_proc = _dedup_preserve_order(r_proc)
        processed_ret.append(r_proc)
        processed_gt.append(g_proc)

    # Ensure ks are positive and increasing; cap at available retrieved size per query when computing
    ks = sorted(set(int(k) for k in ks if k > 0))

    n_queries = len(processed_ret)
    if n_queries == 0:
        return {"per_query": [], "per_k": {}, "ks": ks}

    # Per-query metrics and accumulators for micro/macro
    per_query_rows: List[Dict[str, Any]] = []
    # accumulators keyed by k
    sum_recall = {k: 0.0 for k in ks}
    sum_hits = {k: 0 for k in ks}               # numerator for micro recall (counts of matched items)
    sum_gt_total = {k: 0 for k in ks}           # denominator for micro recall (Σ|G_i|)
    sum_hit_ind = {k: 0 for k in ks}            # for hit rate

    for qi, (ret_list, gt_list) in enumerate(zip(processed_ret, processed_gt)):
        gt_set = set(gt_list)
        gt_size = len(gt_set)
        if gt_size == 0:
            # Skip queries with empty ground truth (undefined recall); record NaNs
            row: Dict[str, Any] = {"query_index": qi, "gt_size": 0}
            for k in ks:
                row[f"recall@{k}"] = float("nan")
                row[f"hit@{k}"] = float("nan")
            per_query_rows.append(row)
            continue

        row = {"query_index": qi, "gt_size": gt_size}
        for k in ks:
            k_eff = min(k, len(ret_list))
            Rk = ret_list[:k_eff]
            inter = gt_set.intersection(Rk)
            num_hits = len(inter)
            recall_k = num_hits / gt_size
            hit_indicator = 1 if num_hits > 0 else 0

            row[f"recall@{k}"] = recall_k
            row[f"hit@{k}"] = hit_indicator

            # macro accum
            sum_recall[k] += recall_k
            # micro accum
            sum_hits[k] += num_hits
            sum_gt_total[k] += gt_size
            # hit-rate accum
            sum_hit_ind[k] += hit_indicator

        per_query_rows.append(row)

    # Aggregate
    per_k_summary = {}
    valid_queries = sum(1 for row in per_query_rows if row["gt_size"] > 0)
    for k in ks:
        macro_recall = (sum_recall[k] / valid_queries) if valid_queries > 0 else float("nan")
        micro_recall = (sum_hits[k] / sum_gt_total[k]) if sum_gt_total[k] > 0 else float("nan")
        hit_rate = (sum_hit_ind[k] / valid_queries) if valid_queries > 0 else float("nan")
        per_k_summary[k] = {
            "macro_recall": macro_recall,
            "micro_recall": micro_recall,
            "hit_rate": hit_rate,
            "avg_hits_per_query": sum_hits[k] / valid_queries if valid_queries > 0 else float("nan"),
        }

    return {
        "ks": ks,
        "per_query": per_query_rows,   # list of dicts, one per query
        "per_k": per_k_summary,        # dict keyed by k with aggregates
        "config": {
            "case_insensitive": case_insensitive,
            "deduplicate_retrieved": deduplicate_retrieved,
        },
    }

#teacher.eval()
c = 0
ret_results = []
for step, row in tqdm(enumerate(test_ds), total=len(test_ds)):
    question = row['question']
    qimg = row['image']
    answer = row['answer']
    qid = row['id']
    qemb = retriever.encode(images=qimg,).detach().cpu().half().numpy()
    D, I = index.search(qemb, 30)
    ret_image = [image_corpus[i] for i in I[0]]
    # Compute MD5 hash
    ret_md5 = [hashlib.md5(Image.open(rimg).tobytes()).hexdigest() for rimg in ret_image]
    gt_md5 = [hashlib.md5(gimg.tobytes()).hexdigest() for gimg in row['gt_images']]
    ret_results.append(evaluate_retrieval(ret_md5, gt_md5, ks=[1, 5, 10, 20, 30]))

    # ret_image = image_corpus[I[0][0]]
    # teacher_input = process_input(question, qimg, ret_image)
    # answer = processor.tokenizer.encode(answer, return_tensors='pt').cuda()
    # with torch.inference_mode():
    #     pred_first = teacher(**teacher_input)
    # pred_first = torch.argmax(pred_first.logits,-1)
    # c += processor.tokenizer._convert_id_to_token(answer[0,0].item()).lower() == processor.tokenizer._convert_id_to_token(pred_first[:,-1].item()).lower()
for step, row in tqdm(enumerate(train_ds), total=len(train_ds)):
    question = row['question']
    qimg = row['image']
    answer = row['answer']
    qid = row['id']
    qemb = retriever.encode(images=qimg).detach().cpu().half().numpy()
    D, I = index.search(qemb, 30)
    ret_image = [image_corpus[i] for i in I[0]]

    # Compute MD5 hash
    ret_md5 = [hashlib.md5(Image.open(rimg).tobytes()).hexdigest() for rimg in ret_image]
    gt_md5 = [hashlib.md5(gimg.tobytes()).hexdigest() for gimg in row['gt_images']]
    ret_results.append(evaluate_retrieval(ret_md5, gt_md5, ks=[1, 5, 10, 20, 30]))

from ret_util import aggregate_eval_results, print_agg_summary
agg = aggregate_eval_results(ret_results, with_ci=False)  # ks inferred from inputs
print_agg_summary(agg)
