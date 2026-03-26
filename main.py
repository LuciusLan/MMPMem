import argparse

p = argparse.ArgumentParser()
p.add_argument("--ret_tau", type=float,  default=1.)
p.add_argument("--top_k", type=int, default=20)
p.add_argument("--ce_weight", type=float, default=1.)
p.add_argument("--kd_weight", type=float, default=0.4)
p.add_argument("--lr", type=float, default=2e-6)
args = p.parse_args()

import os
os.environ['HF_HOME'] = '/data_external/hf_cache'
os.environ['CUDA_VISIBLE_DEVICES']="0,1"
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
os.environ['HF_HUB_OFFLINE'] = "1" 

# os.environ['TORCH_DISTRIBUTED_DEBUG']="DETAIL"

# os.environ['NCCL_ASYNC_ERROR_HANDLING']="1"
from typing import List, Dict, Optional, Tuple, Literal, Callable, Any, Iterable
from transformers.feature_extraction_utils import BatchFeature
from glob import glob
import ast
import re
import json
import math
import random
from dataclasses import dataclass
import time

import numpy as np
#import pandas as pd
from PIL import Image
from datasets import Image as HFImage, Dataset as HFDataset

#import faiss
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, Qwen3VLForConditionalGeneration, CLIPModel, CLIPProcessor, AutoProcessor, Qwen3VLProcessor, Qwen3ForCausalLM
import datasets
import wandb
#from qwen_vl_utils import process_vision_info
from accelerate import dispatch_model, infer_auto_device_map, Accelerator
from accelerate.utils import tqdm
#from tqdm import tqdm

from modelling_memory import MemoryMLP, WrappedLM
from inference import generate_with_memory
from eval_util import score_infoseek
from seq_kd_utils import compute_seqkd_or_mml_loss, extract_top1_sequences_and_logp, compute_candidate_log_weights_for_cache, label_candidate

# ---------------------------
# Hyperparameters & utilities
# ---------------------------
ALPHA_CE = args.ce_weight            # weight for gold CE
BETA_DELTA = 0.5          # weight for delta residual distillation (optional)
KD_WEIGHT = args.kd_weight          # weight for teacher KL
KL_WEIGHT = 0.3
USE_RESIDUAL = False       # if True: memory learns residual logits; else memory outputs a dist directly
HID_LAYER_ID = -1
EPS = 1e-12
MACRO_BATCH_SIZE = 20
LR = args.lr

# =========================
# Optional: example train step wrapper
# =========================



# ---------------------------
# Single-sample training step
# ---------------------------
def train_step(model:WrappedLM, model_config, accelerator:Accelerator, processor, optimizer,scheduler, device, base_inputs: BatchFeature, input_lengths:torch.Tensor, answer_ids:torch.Tensor,  answer_mask:torch.Tensor, teacher_ids:torch.Tensor, teacher_logprob:torch.Tensor, tail_logprob:torch.Tensor, ret_scores:torch.Tensor):
    optimizer.zero_grad(set_to_none=True)
    base_inputs = {k :v.to(device) for k,v in base_inputs.items()}
    input_lengths= input_lengths.to(device)
    answer_ids = answer_ids.to(device)
    answer_mask = answer_mask.to(device)
    teacher_ids = [e.to(device) for e in teacher_ids]
    teacher_logprob = [e.to(device) for e in teacher_logprob]
    #tail_logprob = tail_logprob.to(device)
    ret_scores = [e.to(device) for e in ret_scores]

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # 1) Get base state & logits (frozen)
        B, L_full_max = base_inputs['input_ids'].shape
        _, La_max = answer_ids.shape

        # ----- base model forward, frozen -----
        out = model(
            **base_inputs,
            mix_mode='mem'
        )
        #logits = out["logits"].detach()             # [B, L_full_max, V]
        # hidden = out["hidden_states"][HID_LAYER_ID].detach()  # [B, L_full_max, H] (or choose another layer)

        # B, L_full_max, V = out["logits"].shape
        # H = hidden.size(-1)

        shift_logits = out["logits"]

        # ----- CE over gold answer tokens (masking padding) -----
        #mask_flat = torch.where(answer_mask.view(-1))                         # [B*La_max]
        shift_labels = base_inputs['input_ids'][:, 1:]
        shift_logits = shift_logits[:, :-1, :]
        shift_label_mask = answer_mask[:, 1:]
        gold_flat = shift_labels[shift_label_mask]               # [N_tokens]
        logits_flat = shift_logits[shift_label_mask]         # [N_tokens, V]

        ce_loss = F.cross_entropy(logits_flat, gold_flat)
        del shift_logits
        del out

        kd_loss = compute_seqkd_or_mml_loss(model=model,model_config=model_config, prompt_inputs=base_inputs, label_mask=answer_mask, retrieval_sims_list=ret_scores, answer_ids_top32_list=teacher_ids, answer_logp_top32_list=teacher_logprob, pad_id=processor.tokenizer.pad_token_id, eos_id=processor.tokenizer.eos_token_id, detach_prompt_cache=True)

        loss = kd_loss*KL_WEIGHT + ce_loss * ALPHA_CE
        #loss = ce_loss
        accelerator.backward(loss)
        params_to_clip = [p for g in optimizer.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        stats = {
            "loss": float(loss.item()),
            "kl": float(kd_loss.item()),
            #"kl": 0.,
            "ce": float(ce_loss.item()),
            #"delta": float(delta_loss.item()) if BETA_DELTA > 0 else 0.0,
            #"eta": float(eta.sigmoid().item()) if not USE_RESIDUAL else float(eta.item()),
        }
    return stats

def train_step_temperature(model:WrappedLM, model_config, accelerator:Accelerator, processor, optimizer,scheduler, device, base_inputs: BatchFeature, input_lengths:torch.Tensor, answer_ids:torch.Tensor,  answer_mask:torch.Tensor, batch_cand_tokens, ret_scores, sum_cand_logps,m_first_tok_id, m_first_tok_logp, m_first_tok_tail, candidate_mask, mode="seqkd"):
    optimizer.zero_grad(set_to_none=True)
    base_inputs = {k :v.to(device) for k,v in base_inputs.items()}
    input_lengths= input_lengths.to(device)
    answer_ids = answer_ids.to(device)
    answer_mask = answer_mask.to(device)
    # teacher_ids = [e.to(device) for e in teacher_ids]
    # teacher_logprob = [e.to(device) for e in teacher_logprob]
    # tail_logprob = tail_logprob.to(device)
    # ret_scores = [e.to(device) for e in ret_scores]
    # cand_tokens_batch = [e.to(device) for e in cand_tokens_batch]
    # logw_batch = [e.to(device) for e in logw_batch]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # 1) Get base state & logits (frozen)
        B, L_full_max = base_inputs['input_ids'].shape
        _, La_max = answer_ids.shape

        # ----- base model forward, frozen -----
        # out = model(
        #     **base_inputs,
        #     mix_mode='mem'
        # )
        # shift_logits = out["logits"]

        # ----- CE over gold answer tokens (masking padding) -----
        # shift_labels = base_inputs['input_ids'][:, 1:]
        # shift_logits = shift_logits[:, :-1, :]
        # shift_label_mask = answer_mask[:, 1:]
        # gold_flat = shift_labels[shift_label_mask]               # [N_tokens]
        # logits_flat = shift_logits[shift_label_mask]         # [N_tokens, V]

        kd_loss, ce_loss, ft_kl_loss = model(model_config=model_config, prompt_inputs=base_inputs, label_mask=answer_mask, answer_ids=answer_ids, batch_cand_tokens=batch_cand_tokens, ret_scores=ret_scores, sum_cand_logps=sum_cand_logps, m_first_tok_id=m_first_tok_id, m_first_tok_logp=m_first_tok_logp, m_first_tok_tail=m_first_tok_tail, candidate_mask=candidate_mask, pad_id=processor.tokenizer.pad_token_id, eos_id=processor.tokenizer.eos_token_id, detach_prompt_cache=True, tau_retrieval=args.ret_tau, top_k=args.top_k, branch="train", mode=mode, add_kl=True)

        loss = kd_loss*KD_WEIGHT + ce_loss * ALPHA_CE# + ft_kl_loss*KL_WEIGHT
        #loss = ce_loss
        accelerator.backward(loss)
        params_to_clip = [p for g in optimizer.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        stats = {
            "loss": float(loss.item()),
            "kl": float(kd_loss.item()),
            #"kl": 0.,
            "ce": float(ce_loss.item()),
            #"delta": float(delta_loss.item()) if BETA_DELTA > 0 else 0.0,
            #"eta": float(eta.sigmoid().item()) if not USE_RESIDUAL else float(eta.item()),
        }
    return stats

_ABSTENTION_PATTERNS = [
    r"\bi\s+don't\s+know\b",
    r"\bi\s+do\s+not\s+know\b",
    r"\bnot enough information\b",
    r"\binsufficient information\b",
    r"\bcannot be determined\b",
    r"\bcan't be determined\b",
    r"\bcannot answer\b",
    r"\bcan't answer\b",
    r"\bunable to answer\b",
    r"\bthe (given|provided) (evidence|context|information) is not enough\b",
    r"\bthere is no information\b",
]
def _contains_abstention(text: str, patterns: list[str]) -> bool:
    text = text.strip().lower()
    return any(re.search(p, text) is not None for p in patterns)


def get_stats_step(model:WrappedLM, processor, device, base_inputs: BatchFeature, input_lengths:torch.Tensor, answer_ids:torch.Tensor,  answer_mask:torch.Tensor, teacher_ids:torch.Tensor, teacher_logprob:torch.Tensor, tail_logprob:torch.Tensor, ret_scores:torch.Tensor, gt:str, score_func, tokenizer, tau=1.):
    base_inputs = {k :v.to(device) for k,v in base_inputs.items()}
    input_lengths= input_lengths.to(device)
    answer_ids = answer_ids.to(device)
    answer_mask = answer_mask.to(device)

    teacher_ids = [e.to(device)[:29, :, :] for e in teacher_ids]
    teacher_logprob = [e.to(device)[:29, :, :] for e in teacher_logprob]
    ret_scores = [e.to(device)[:29] for e in ret_scores]
    tail_logprob = [e.to(device)[:29, :] for e in tail_logprob]


    assert "input_ids" in base_inputs and "attention_mask" in base_inputs
    B = int(base_inputs["input_ids"].shape[0])
    #prompt_attention_mask = build_prompt_only_masks(model.device, prompt_inputs["input_ids"], prompt_inputs["attention_mask"], label_mask, pad_id=pad_id)
    #prompt_inputs['attention_mask'] = prompt_attention_mask
    assert len(ret_scores) == B
    assert len(teacher_ids) == B
    assert len(teacher_logprob) == B

    # Extract top-1 realized sequences (index 0) per example
    extracted = extract_top1_sequences_and_logp(teacher_ids, teacher_logprob)
    assert isinstance(extracted, list)

    for b in range(B):
        retrieval_sims = ret_scores[b].softmax(-1)         # [K_b]
        cand_tokens, teacher_token_logp = extracted[b]              # [K_b, L_b], [K_b, L_b]
        cand_tokens = cand_tokens
        teacher_token_logp = teacher_token_logp

        Kb = int(retrieval_sims.shape[0])
        if Kb == 0:
            return None

        candidate_mask = torch.isfinite(retrieval_sims)

        cand_texts = processor.batch_decode(cand_tokens, skip_special_tokens=True)

        counts_mask = (cand_tokens == 151643).sum(dim=1)
        cand_length = cand_tokens.size(1) - counts_mask
        has_abst = torch.tensor([_contains_abstention(e, _ABSTENTION_PATTERNS) for e in cand_texts], dtype=torch.bool)
        too_long = cand_length > 30

        invalid_cand = torch.logical_or(has_abst, too_long)
        if invalid_cand.any():
            ic = invalid_cand.tolist()
            c = ic.count(True)
            if c > Kb//2:
                return None

        retrieval_sims_raw, teacher_conf_raw, combined_score, candidate_mask = compute_candidate_log_weights_for_cache(
            retrieval_sims=retrieval_sims.unsqueeze(0),                 # [1, K]
            cand_tokens=cand_tokens.unsqueeze(0),                       # [1, K, L]
            teacher_token_logp=teacher_token_logp.unsqueeze(0),         # [1, K, L]
            pad_id=151643,
            eos_id=151645,
            candidate_mask=candidate_mask.unsqueeze(0),
            tau_retrieval=tau,
            tau_teacher=1.,
            teacher_confidence="mean",
            include_eos_in_teacher_conf=True,
            return_components=True
        )  # [K]

        candidate_mask = ~invalid_cand
        candidate_mask = candidate_mask.unsqueeze(0)

        gt_is_ranked_top, gt_rank, top_weight, merged_scores = label_candidate(
            cand_tokens=cand_tokens,
            gt=gt,
            retrieval_sims_raw=ret_scores[b],
            #teacher_conf_raw=None,
            teacher_conf_raw=teacher_conf_raw,
            candidate_mask=candidate_mask,
            pad_id=151643,
            eos_id=151645,
            score_func=score_func,
            tokenizer=tokenizer,
            tau_retrieval=tau,
        )
        # logw = logw.squeeze(0)
        # log_alpha = log_alpha.squeeze(0)

        # logw = logw.masked_fill(~candidate_mask, float("-inf"))

        # cand_tokens_m, logw_m, _, cand_mask_m = merge_duplicate_candidates(
        #     device=device,
        #     cand_tokens=cand_tokens.unsqueeze(0),
        #     logw=logw.unsqueeze(0),
        #     student_seq_logp=None,
        #     pad_id=151643,
        #     eos_id=151645,
        # )
        # cand_tokens = cand_tokens_m.squeeze(0)            # [K', L]
        # logw = logw_m.squeeze(0).exp()
        # merged_cand_tokens = processor.batch_decode(cand_tokens, skip_special_tokens=True)
        first_ids_list = teacher_ids[b][:, 0, :]
        first_logp_list = teacher_logprob[b][:, 0, :]
        first_tail_list = tail_logprob[b][:, 0]

        return cand_tokens, ret_scores[b], teacher_conf_raw,candidate_mask, first_ids_list, first_logp_list, first_tail_list, gt_is_ranked_top, gt_rank, top_weight, merged_scores


prompt_base = "Answer the image related question. For fact checking questions, the desired answer would be named entities instead of common concept, for example \"Mount Everest\" instead of \"mountain top\", \"River Thames\" instead of \"river\". Be concise, output the answer only, without any additional words."
prompt_teacher = "Answer the image related question. For fact checking questions, the desired answer would be named entities instead of common concept, for example \"Mount Everest\" instead of \"mountain top\", \"River Thames\" instead of \"river\". Be concise, output the answer only, without any additional words. An additional image is provided for your reference, but it is not guaranteed to be relevant to original question."

def process_input_test(processor, question_batch, qimg_batch, ret_image=None):
    messages = []
    for question, qimg in zip(question_batch, qimg_batch):
        if not ret_image:
            temp = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": qimg,
                        },
                        {"type": "text", "text": prompt_base + "\nQuestion: " +question},
                    ],
                }
                ]
        else:
            temp = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": qimg,
                        },
                        {"type": "text", "text": prompt_teacher + "\nQuestion: " +question + "Additional Image:\n"},
                        {
                            "type": "image",
                            "image": ret_image,
                        }
                    ],
                }
            ]
        messages.append(temp)

    # Preparation for inference
    text_inputs = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        #return_dict=True,
        #padding=True, 
        #return_tensors="pt"
        )
    
    inputs = processor(images=qimg_batch, text=text_inputs, padding=True, padding_side="left", return_tensors='pt')
    #inputs.pop("token_type_ids", None)   # per model card snippet
    #inputs = inputs.to(teacher.device)
    return inputs

def first_subsequence_pos(input_ids: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
    """
    input_ids: [B, L] long
    pattern:   [M] long
    returns:   [B] long, first start index of pattern, else -1
    """
    B, L = input_ids.shape
    M = pattern.numel()
    if L < M:
        return input_ids.new_full((B,), -1)

    pattern = pattern.to(device=input_ids.device, dtype=input_ids.dtype)

    windows = input_ids.unfold(dimension=1, size=M, step=1)   # [B, L-M+1, M] view
    match = (windows == pattern).all(dim=-1)                  # [B, L-M+1] bool

    idx = match.long().argmax(dim=1)
    idx = torch.where(match.any(dim=1), idx, input_ids.new_full(idx.shape, -1))
    idx +=3
    return idx

ASSISTANT_TOKENS = torch.tensor([151644,77091,198], dtype=torch.long)
IMAGE_END_TOKENS = torch.tensor([151655,151653], dtype=torch.long)
    
def process_input(processor:Qwen3VLProcessor, questions, qimgs, answers, ret_image=None):
    if not ret_image:
        messages = []
        for question, image in zip(questions, qimgs):
            messages.append([
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
            ])
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
    text_inputs = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        #return_dict=True,
        #padding=True, 
        #return_tensors="pt"
        )
    
    answer_lengths = []
    answer_ids_list = []

    B = len(answers)

    for i, ans in enumerate(answers):
        text_inputs[i]+=ans
        ans_id = processor.tokenizer.encode(ans, return_tensors='pt')[0]
        answer_lengths.append(len(ans_id))
        answer_ids_list.append(ans_id)
    La_max = max(answer_lengths)

    answer_ids = torch.full((B, La_max), processor.tokenizer.pad_token_id, dtype=torch.long)
    answer_mask = torch.zeros((B, La_max), dtype=torch.long)
    for i, (aid, La) in enumerate(zip(answer_ids_list, answer_lengths)):
        answer_ids[i, :La] = aid
        answer_mask[i, :La] = 1

    inputs = processor(images=qimgs, text=text_inputs, padding=True, return_tensors='pt', images_kwargs={
            "min_pixels": 64 * 32 * 32,
            "max_pixels": 1280 * 32 * 32,
        })
    #input_lengths = inputs['attention_mask'].sum(dim=1)
    input_lengths = first_subsequence_pos(inputs['input_ids'], ASSISTANT_TOKENS)
    assert -1 not in input_lengths

    B, T = inputs.input_ids.shape
    answer_lengths = torch.tensor(answer_lengths, dtype=torch.long)
    end = input_lengths + answer_lengths  # end-exclusive

    # Optional safety (recommended): clamp to valid range
    start = input_lengths.clamp(0, T)
    end = end.clamp(0, T)

    diff = torch.zeros((B, T + 1), dtype=torch.long)
    diff.scatter_add_(1, start[:, None], torch.ones((B, 1), dtype=diff.dtype))
    diff.scatter_add_(1, end[:, None], -torch.ones((B, 1), dtype=diff.dtype))
    mask = diff.cumsum(dim=1)[:, :T] > 0
    label_mask = torch.zeros_like(inputs.input_ids, dtype=torch.bool).masked_fill(mask, True)
    #inputs.pop("token_type_ids", None)   # per model card snippet
    #inputs = inputs.to(teacher.device)
    return inputs, input_lengths, answer_ids, label_mask, answer_lengths


def process_teacher_batch(mix_idx, mix_logp, tail_logp, answer_lengths:list[int]):
    B = len(answer_lengths)
    La_max = max(answer_lengths)
    K = mix_idx[0].size(1)
    #teacher_input = {k: [e[k] for e in teacher_buffer] for k in teacher_logits.keys()}

    teacher_ids_with = torch.zeros((B, La_max, K), dtype=torch.long)
    teacher_logprob_with = torch.full((B, La_max, K), float("-inf"), dtype=torch.float)
    tail_logprob = torch.full((B, La_max), float("-inf"), dtype=torch.float)

    for i, (idx, logp, taillogp, La) in enumerate(zip(mix_idx, mix_logp, tail_logp, answer_lengths)):
        teacher_ids_with[i, :La, :] = idx
        teacher_logprob_with[i, :La, :] = logp
        tail_logprob[i, :La] = taillogp
    return teacher_ids_with, teacher_logprob_with, tail_logprob


def gen(input_ds, processor, device):
    pbar = tqdm(total=len(input_ds))
    for step, batch in enumerate(input_ds):
        inputs, input_lengths, answer_ids, answer_mask, answer_lengths, teacher_ids, teacher_logps, teacher_tails, ret_scores, eval_answer, data_id = batch

        gt = eval_answer[0]
        if '{' in gt[0] and '}' in gt[0]:
            gt = ast.literal_eval(gt[0])
        #score = score_infoseek(cand_answer, gt)
        pbar.update()
        stats_out = get_stats_step(model=None, processor=processor, device='cpu', base_inputs=inputs, input_lengths=input_lengths, answer_ids=answer_ids, answer_mask=answer_mask, teacher_ids=teacher_ids, teacher_logprob=teacher_logps, tail_logprob=teacher_tails, ret_scores=ret_scores, gt=gt, score_func=score_infoseek, tokenizer=processor.tokenizer, tau=1.)
        if stats_out is not None:
            cand_tokens,retrieval_sims_raw, teacher_conf_raw,candidate_mask, first_ids_list, first_logp_list, first_tail_list, gt_is_ranked_top, gt_rank, top_weight, merged_score = stats_out
        else:
            continue
        
        if retrieval_sims_raw.size(0) == 1:
            continue
        elif retrieval_sims_raw.std().isnan():
            continue
        sample = {'data_id':data_id[0], 'cand_tokens': cand_tokens, 'ret_scores': retrieval_sims_raw, "sum_cand_logps": teacher_conf_raw.squeeze(0), "candidate_mask":candidate_mask.squeeze(0), 'm_first_tok_id': first_ids_list, 'm_first_tok_logp':first_logp_list, 'm_first_tok_tail': first_tail_list, "keep": gt_is_ranked_top, "candidate_gt_rank": gt_rank, "candidate_mass":top_weight, "gt": str(gt)}
        yield sample

def process_ds():
    train_ds = datasets.load_from_disk('/data_external/InfoSeek/train_gen_combined').with_format('torch')
    #train_ds = train_ds.remove_columns(["per_ev_top_ids", "per_ev_top_logps", "per_ev_tail"])
    processor = AutoProcessor.from_pretrained('/wyy/models/Qwen3-VL-8B-Instruct')
    def collate_train_raw(batch):
        row = {k: [e[k] for e in batch] for k in batch[0].keys()}
        question = row['question']
        qimg = row['image']
        answers = row['answer']
        longest_answers = [max(answer, key=len)+processor.tokenizer.eos_token+'\n' for answer in answers]
        #qid = row['data_id']

        inputs, input_lengths, answer_ids, answer_mask, answer_lengths = process_input(processor, question, qimg, longest_answers)        
        teacher_ids = row['per_ev_top_ids']
        teacher_logps = row['per_ev_top_logps']
        for i in range(len(teacher_ids)):
            pad_pos = teacher_ids[i] == -1
            teacher_ids[i][pad_pos] = processor.tokenizer.pad_token_id
        teacher_tails = row['per_ev_tail']
        ret_scores = row['scores']
        return inputs, input_lengths, answer_ids, answer_mask, answer_lengths, teacher_ids, teacher_logps, teacher_tails, ret_scores, row['answer_eval'], row['data_id']

    from datasets import Features, Sequence, Value 
    features = Features({
        "data_id": Value("string"),
        "cand_tokens": Sequence(Sequence(Value("int64"))),
        "ret_scores": Sequence(Value("float32")),
        "sum_cand_logps": Sequence(Value("float32")),
        "candidate_mask": Sequence(Value("bool")),
        "m_first_tok_id": Sequence(Sequence(Value("int64"))),
        "m_first_tok_logp": Sequence(Sequence(Value("float32"))),
        "m_first_tok_tail": Sequence(Value("float32")),
        "keep": Value("bool"),
        "candidate_gt_rank": Value("int64"),
        "candidate_mass": Value("float32"),
        "gt": Value("string"),
    })

    train_dl = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=collate_train_raw, prefetch_factor=2, num_workers=6)

    distil_dataset = HFDataset.from_generator(gen, features=features, gen_kwargs={'input_ds': train_dl, 'processor':processor, 'device': "cpu"})
    distil_dataset.save_to_disk('/data_external/InfoSeek/distill_train')
    return
    # k=30, t=1 avg_w=0.485 avg_topgt_w=0.647 avg_r=1.89
    # k=20, t=1 avg_w=0.514 avg_topgt_w=0.670 avg_r=1.72
    # k=10, t=1 avg_w=0.573 avg_topgt_w=0.699 avg_r=1.48

    # k=30, t=0.7 avg_w=0.489 avg_topgt_w=0.651
    # k=20, t=0.7 avg_w=0.518 avg_topgt_w=0.666
    # k=10, t=0.7 avg_w=0.590 avg_topgt_w=0.707

    # k=30, t=0.5 avg_w=0.481 avg_topgt_w=0.645
    # k=20, t=0.5 avg_w=0.514 avg_topgt_w=0.664
    # k=10, t=0.5 avg_w=0.587 avg_topgt_w=0.703

    # k=30, t=0.3 avg_w=0. avg_topgt_w=0.
    # k=20, t=0.3 avg_w=0. avg_topgt_w=0.
    # k=10, t=0.3 avg_w=0. avg_topgt_w=0.

def main():
    train_ds = datasets.load_from_disk('/data_external/InfoSeek/train_gen_combined').with_format('torch')
    train_distill = datasets.load_from_disk('/data_external/InfoSeek/distill_pos').with_format('torch')
    train_distill = train_distill.filter(lambda x: x['candidate_gt_rank'] == 0)
    #distill_pos = train_distill.filter(lambda x: x['keep'] == True)
    

    train_ds = train_ds.remove_columns(["per_ev_top_ids", "per_ev_top_logps", "per_ev_tail"])

    #train_ds_ids = train_ds['data_id']
    #train_ds_id_map = {e: i for i,e in tqdm(enumerate(train_ds_ids), total=len(train_ds_ids))}
    train_ds_id_map = torch.load('/data_external/InfoSeek/train_ds_id_map.pt')

    # train_select = train_distill['data_id']
    # train_select = [train_ds_id_map[e] for e in train_select]

    #train_ds = train_ds.cast_column("image", HFImage(decode=True))
    #train_ds = train_ds.select(range(100000))
    test_ds = datasets.load_from_disk('/data_external/InfoSeek/val_combined').with_format('torch')
    #test_ds = test_ds.filter(lambda x: x['utype'] == 'val_unseen_question') #val_unseen_entity
    #test_ds = test_ds.select(range(1000))

    base_model = Qwen3VLForConditionalGeneration.from_pretrained("/wyy/models/Qwen3-VL-8B-Instruct", local_files_only=True, attn_implementation="flash_attention_3", dtype=torch.bfloat16, device_map='cuda')

    # #base_model = Qwen3VLForConditionalGeneration.from_pretrained("/wyy/models/Qwen3-VL-8B-Instruct", local_files_only=True, attn_implementation="eager", dtype=torch.float16, device_map='cpu')

    for p in base_model.parameters():
       p.requires_grad_(False)
    processor = AutoProcessor.from_pretrained('/wyy/models/Qwen3-VL-8B-Instruct')

    memory = MemoryMLP()
    for p in memory.parameters():
        p.requires_grad_(True)

    print(next(memory.parameters()).device)
    print(sum(p.numel() for p in memory.parameters()) / 1e9, "B params")

    # saved_dict = torch.load('/data_external/MMPMem/checkpoints/t1.0_k20_skd_lr2e-6_LMLastLayer.pt')
    # memory.load_state_dict(saved_dict, strict=True)

    model = WrappedLM(base_model, memory, config=base_model.config, processor=processor, layer_idx_for_mem=HID_LAYER_ID)

    def collate_train_raw(batch):
        row = {k: [e[k] for e in batch] for k in batch[0].keys()}
        question = row['question']
        qimg = row['image']
        answers = row['answer']
        longest_answers = [max(answer, key=len)+processor.tokenizer.eos_token+'\n' for answer in answers]
        #qid = row['data_id']

        inputs, input_lengths, answer_ids, answer_mask, answer_lengths = process_input(processor, question, qimg, longest_answers)        
        teacher_ids = row['per_ev_top_ids']
        teacher_logps = row['per_ev_top_logps']
        for i in range(len(teacher_ids)):
            pad_pos = teacher_ids[i] == -1
            teacher_ids[i][pad_pos] = processor.tokenizer.pad_token_id
        teacher_tails = row['per_ev_tail']
        ret_scores = row['scores']
        return inputs, input_lengths, answer_ids, answer_mask, answer_lengths, teacher_ids, teacher_logps, teacher_tails, ret_scores, row['answer_eval'], row['data_id']

    def collate_train(batch):
        distill_row = {k: [e[k] for e in batch] for k in batch[0].keys()}
        base_batch = [train_ds[train_ds_id_map[e]] for e in distill_row['data_id']]
        row = {k: [e[k] for e in base_batch] for k in base_batch[0].keys()}
        question = row['question']
        qimg = row['image']
        answers = row['answer']
        longest_answers = [max(answer, key=len)+processor.tokenizer.eos_token+'\n' for answer in answers]
        #qid = row['data_id']

        inputs, input_lengths, answer_ids, answer_mask, answer_lengths = process_input(processor, question, qimg, longest_answers)
        #teacher_ids_with, teacher_logprob_with, tail_logprob = process_teacher_batch(row['mix_idx'], row['mix_logp'], row['tail_logp'], answer_lengths)
        
        # teacher_ids = row['per_ev_top_ids']
        # teacher_logps = row['per_ev_top_logps']
        # for i in range(len(teacher_ids)):
        #     pad_pos = teacher_ids[i] == -1
        #     teacher_ids[i][pad_pos] = processor.tokenizer.pad_token_id
        # teacher_tails = row['per_ev_tail']
        # ret_scores = row['scores']
        #return inputs, input_lengths, answer_ids, answer_mask, answer_lengths, teacher_ids, teacher_logps, teacher_tails, ret_scores, row['answer_eval'], row['data_id']
        return inputs, input_lengths, answer_ids, answer_mask, answer_lengths, row['answer_eval'], row['data_id'], distill_row['cand_tokens'], distill_row['ret_scores'], distill_row['sum_cand_logps'], distill_row['candidate_mask'], distill_row['m_first_tok_id'], distill_row['m_first_tok_logp'], distill_row['m_first_tok_tail'], 

    def collate_eval(batch):
        row = {k: [e[k] for e in batch] for k in batch[0].keys()}
        question = row['question']
        qimg = row['image']
        base_inputs = process_input_test(processor, question, qimg).to(model.device)
        return base_inputs, row['answer_eval'], row['data_id']

    train_dl = DataLoader(train_distill, batch_size=MACRO_BATCH_SIZE, shuffle=True, collate_fn=collate_train, num_workers=5, prefetch_factor=2,)
    
    #train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_train_raw)
    # import re
    # judge_model = Qwen3ForCausalLM.from_pretrained('/data_external/Qwen3-4B-Instruct-2507', local_files_only=True, attn_implementation="flash_attention_2", dtype=torch.bfloat16, device_map='cuda')
    # stats = []
    # for batch in tqdm(train_dl):
    #     inputs, input_lengths, answer_ids, answer_mask, answer_lengths, teacher_ids, teacher_logps, teacher_tails, ret_scores, eval_answer, data_id = batch
    #     cand_tokens, merged_cand_tokens, cand_mask_m, logw, first_ids_list, first_logp_list, first_tail_list = get_stats_step(None, processor, 'cpu', inputs, input_lengths, answer_ids, answer_mask,  teacher_ids, teacher_logps, teacher_tails, ret_scores)

    #     template = '<|im_start|>user\nPlease evaluate the student answers of following question, providing the reference answer. The question is regarding an image, though you are not able to view the image, you only need to judge if the semantic meaning of student answer aligns with the reference answer. You MUST output your final judgement wrapped in "\\boxed{}", labelled with either 0 or 1 score.\n'
    #     gt = processor.decode(answer_ids[0], skip_special_tokens=True).strip()
    #     q = processor.decode(inputs['input_ids'][0], skip_special_tokens=True)
    #     q = re.findall(r'Question: (.+)\nassistant', q)[0]
    #     has_correct=False
    #     rank = -1
    #     for i, cand in enumerate(cand_tokens):
    #         pred =  processor.decode(cand, skip_special_tokens=True).strip()
    #         prompt = template+"Question: "+q+"\nReference Answer: "+gt+"\nStudent Answer: "+pred+"<|im_end|>\n<|im_start|>assistant\n"
    #         input_ids = processor.tokenizer(prompt, return_tensors='pt')
    #         score = judge_model.generate(input_ids['input_ids'].to('cuda'), max_new_tokens=100)
    #         score = score[0, input_ids['input_ids'].size(1):]
    #         score_text = processor.decode(score)
    #         try:
    #             score = re.findall(r'\\boxed\{(\d)\}', score_text)[0]
    #         except:
    #             score = "0"
    #         if score == '1':
    #             has_correct = True
    #             rank = i
    #             break
    #     stats.append([has_correct, rank])
    #     pass
    #train_dl.__iter__().__next__()
    test_dl = DataLoader(test_ds, batch_size=1, collate_fn=collate_eval)

    
    num_warmup_steps = 200
    num_training_steps = len(train_dl)*4
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))

        progress = float(step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    optimizer = torch.optim.AdamW(list(memory.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    from accelerate.utils import DistributedDataParallelKwargs
    ddp = DistributedDataParallelKwargs(
        broadcast_buffers=True,
        find_unused_parameters=True,
        static_graph=False,
    )
    accel = Accelerator(kwargs_handlers=[ddp])
    model, optimizer,scheduler, train_dl, test_dl = accel.prepare(model, optimizer, scheduler, train_dl, test_dl)

    unw_model = accel.unwrap_model(model)
    _memory = unw_model.memory
    
    if accel.is_main_process:
       wandb_run = wandb.init(project="MMMem", name=f"CEKD_t{args.ret_tau}_k{args.top_k}_skd_last_layer_wide_lr{args.lr}")

    accel.wait_for_everyone()
    model.eval()
    correct = torch.tensor(0., device=model.device)
    #cb = torch.tensor(0., device=model.device)
    all_gt = []
    all_pred =  []
    all_pred_base = []
    gathered_results_base = []
    with torch.inference_mode(), accel.join_uneven_inputs([model], even_batches=False):
        for row in tqdm(test_dl):
            inputs, gts, data_ids = row
            output_ids = unw_model.generate(**inputs, mix_mode='base', mix_lambda=0.6, branch="generation")
            input_len = inputs.input_ids.size(1)
            generated_ids = output_ids[:, input_len:]
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)

            local_results = []
            for gt, pred, did in zip(gts, text, data_ids):
                if '{' in gt[0] and '}' in gt[0]:
                    gt = ast.literal_eval(gt[0])
                score_m = score_infoseek(pred, gt)
                correct += score_m['acc']
                local_results.append({"c": 1 if score_m['acc']>0 else 0, "data_id": did, 'gt': gt, 'pred': pred})
                all_gt.append(gt)
                all_pred.append(pred)

            gathered = accel.gather_for_metrics(
                local_results,
                use_gather_object=True,
            )
            if accel.is_main_process:
                # Normalize in case the gathered structure is nested by rank
                if len(gathered) > 0 and isinstance(gathered[0], list):
                    gathered = [x for chunk in gathered for x in chunk]
                gathered_results_base.extend(gathered)

    accel.wait_for_everyone()
    correct = accel.reduce(correct, reduction="sum")
    if accel.is_main_process:
        correct = correct.item()
        gathered_results_base.sort(key=lambda x: x["data_id"])
        print(correct)
        wandb_run.log({'eval/acc': correct/1000, 'eval/epoch': 0})
    accel.wait_for_everyone()

    # correct = torch.tensor(0., device=model.device)
    # gathered_results_mix = []
    # with torch.inference_mode(), accel.join_uneven_inputs([model], even_batches=False):
    #     for row in tqdm(test_dl):
    #         inputs, gts, data_ids = row
    #         output_ids = unw_model.generate(**inputs, mix_mode='mix', mix_lambda=0.6, branch="generation")
    #         input_len = inputs.input_ids.size(1)
    #         generated_ids = output_ids[:, input_len:]
    #         text = processor.batch_decode(generated_ids, skip_special_tokens=True)

    #         local_results = []
    #         for gt, pred, did in zip(gts, text, data_ids):
    #             if '{' in gt[0] and '}' in gt[0]:
    #                 gt = ast.literal_eval(gt[0])
    #             score_m = score_infoseek(pred, gt)
    #             correct += score_m['acc']
    #             local_results.append({"c": 1 if score_m['acc']>0 else 0, "data_id": did, 'gt': gt, 'pred': pred})

    #         gathered = accel.gather_for_metrics(
    #             local_results,
    #             use_gather_object=True,
    #         )
    #         if accel.is_main_process:
    #             # Normalize in case the gathered structure is nested by rank
    #             if len(gathered) > 0 and isinstance(gathered[0], list):
    #                 gathered = [x for chunk in gathered for x in chunk]
    #             gathered_results_mix.extend(gathered)

    # accel.wait_for_everyone()
    # correct = accel.reduce(correct, reduction="sum")
    # if accel.is_main_process:
    #     correct = correct.item()
    #     gathered_results_mix.sort(key=lambda x: x["data_id"])
    #     print(correct)
    #     #wandb_run.log({'eval/acc': correct/1000, 'eval/epoch': 0})
    # accel.wait_for_everyone()

    # torch.save([gathered_results_base, gathered_results_mix], "/latent_aug/MMPMem/result_analy_ue.pt")

    # unseen question 
    # base 21.92
    # mix 0.6 21.77
    # Mix 0.6 LMLast 22.24

    # 7512, 296
    # 264, 1928

    # unseen entity 
    # base  18.78
    # mix 0.6 18.69
    # Mix 0.6 LMLast 19.02
    #         # Top 30:
    #         # 4706/10000 has GT answer in candidates
    #         # average marginalized weight for GT answer in candidates: 45.54%
    #         # average rank of GT answer appearing in candidates: 2.38
    #         # Average number of correct answers (dataset provides alternative form) 1.36 (among 4706) / 0.72 (among 10000)
    #         # avg candidate num: 9.15

    #         # Top 10:
    #         # 4752/10000 has GT answer in candidates
    #         # average marginalized weight for GT answer in candidates: 55.27%
    #         # average rank of GT answer appearing in candidates: 1.66
    #         # avg candidate num: 4.08

    for ep in range(5):
        model.train()
        acc_loss = []
        pbar = tqdm(train_dl)
        for step, batch in enumerate(train_dl):
            #inputs, input_lengths, answer_ids, answer_mask, answer_lengths, teacher_ids, teacher_logps, teacher_tails, ret_scores = batch
            inputs, input_lengths, answer_ids, answer_mask, answer_lengths, answer_eval, data_id, cand_tokens, ret_scores, sum_cand_logps, candidate_mask, m_first_tok_id, m_first_tok_logp, m_first_tok_tail = batch
            #stats = train_step(model,unw_model.config, accel, processor, optimizer,scheduler, model.device, inputs, input_lengths, answer_ids, answer_mask,  teacher_ids, teacher_logps, teacher_tails, ret_scores)
            stats = train_step_temperature(model,unw_model.config, accel, processor, optimizer, scheduler, model.device, inputs, input_lengths, answer_ids, answer_mask,  cand_tokens, ret_scores, sum_cand_logps,m_first_tok_id, m_first_tok_logp, m_first_tok_tail, candidate_mask, mode="seqkd")
            
            acc_loss.append(stats['loss'])
            cur_loss = np.mean(acc_loss)
            pbar.update()
            pbar.set_postfix({"Loss": cur_loss})

            if accel.is_main_process:
                log_dict = {
                    "train/epoch": ep + 1,
                    "train/step": ep * len(train_dl) + step,
                    "train/loss": stats['loss'],
                    "train/ce_loss": stats['ce'],
                    "train/kd_loss": stats['kl'],
                    "train/current_lr": scheduler.get_last_lr()[0],
                }
                wandb_run.log(log_dict)
            if step%20 == 0:
                torch.cuda.empty_cache()

            if step%2000 == 0 and step != 0:
                accel.wait_for_everyone()
                state = accel.get_state_dict(_memory)         # gathered on rank 0, offloaded to CPU
                if accel.is_main_process:
                    torch.save(state, f"/data_external/MMPMem/checkpoints/ep{ep}step{step}_ce_kd.pt")

                accel.wait_for_everyone()
                model.eval()
                correct =torch.tensor(0., device=model.device)
                em = 0
                cb = torch.tensor(0., device=model.device)
                eb = 0
                for row in tqdm(test_dl):
                    inputs, gts, data_ids = row
                    with torch.inference_mode():
                        output_ids = unw_model.generate(**inputs, mix_mode='mix', branch='generation')
                        input_len = inputs.input_ids.size(1)
                        generated_ids = output_ids[:, input_len:]
                        text = processor.batch_decode(generated_ids, skip_special_tokens=True)

                    for gt, pred in zip(gts, text):
                        if '{' in gt[0] and '}' in gt[0]:
                            gt = ast.literal_eval(gt[0])
                        score_m = score_infoseek(pred, gt)
                        correct += score_m['acc']
                model.train()
                print(f"Ep {ep}: val acc: {correct/1000} | val em: {em/1000}")
                accel.wait_for_everyone()
                correct = accel.reduce(correct, reduction="sum").item()
                #cb = accel.reduce(cb, reduction="sum").item()
                if accel.is_main_process:
                    wandb_run.log({'eval/acc': correct/1000})

        accel.wait_for_everyone()
        state = accel.get_state_dict(_memory)         # gathered on rank 0, offloaded to CPU
        if accel.is_main_process:
            torch.save(state, f"/data_external/MMPMem/checkpoints/{ep}_ce_kd.pt")
        accel.wait_for_everyone()

        # saved_dict = torch.load('/data_external/MMPMem/checkpoints/0_ce_only.pt')
        # memory.load_state_dict(saved_dict, strict=True)
        model.eval()
        correct =torch.tensor(0., device=model.device)
        em = 0
        cb = torch.tensor(0., device=model.device)
        eb = 0
        for row in tqdm(test_dl):
            inputs, gts, data_ids = row
            with torch.inference_mode():
                output_ids = unw_model.generate(**inputs, mix_mode='mix', branch="generation")
                input_len = inputs.input_ids.size(1)
                generated_ids = output_ids[:, input_len:]
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for gt, pred in zip(gts, text):
                if '{' in gt[0] and '}' in gt[0]:
                    gt = ast.literal_eval(gt[0])
                score_m = score_infoseek(pred, gt)
                correct += score_m['acc']

        print(f"Ep {ep}: val acc: {correct/1000} | val em: {em/1000}")
        accel.wait_for_everyone()
        correct = accel.reduce(correct, reduction="sum").item()
        #cb = accel.reduce(cb, reduction="sum").item()
        if accel.is_main_process:
            wandb_run.log({'eval/acc': correct/1000})
            #wandb_run.log({'eval/em': em/1000})
            #wandb_run.log({'eval/acc_base': cb/1000})
            #wandb_run.log({'eval/em_base': eb/1000})
            wandb_run.log({'eval/epoch': ep+1})
        pass

def set_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    set_determinism(2026)
    main()
    #process_ds()