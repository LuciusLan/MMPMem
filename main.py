import argparse
import os
os.environ['HF_HOME'] = '/data_external/hf_cache'
os.environ['CUDA_VISIBLE_DEVICES']="0,1"
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
from typing import List, Dict, Optional, Tuple, Literal, Callable, Any, Iterable
from transformers.feature_extraction_utils import BatchFeature
from glob import glob
import ast
import json
import math
import random
from dataclasses import dataclass

import numpy as np
#import pandas as pd
from PIL import Image
from datasets import Image as HFImage

#import faiss
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, Qwen3VLForConditionalGeneration, CLIPModel, CLIPProcessor, AutoProcessor, Qwen3VLProcessor
import datasets
import wandb
#from qwen_vl_utils import process_vision_info
from accelerate import dispatch_model, infer_auto_device_map, Accelerator
from accelerate.utils import tqdm
#from tqdm import tqdm

from modelling_memory import MemoryMLP, WrappedLM
from inference import generate_with_memory
from eval_util import score_infoseek
from seq_kd_utils import compute_seqkd_or_mml_loss

# ---------------------------
# Hyperparameters & utilities
# ---------------------------
TEMPERATURE = 1.5         # softmax temperature for teacher
ALPHA_CE = 1.0            # weight for gold CE
BETA_DELTA = 0.5          # weight for delta residual distillation (optional)
KL_WEIGHT = 0.3           # weight for teacher KL
USE_RESIDUAL = False       # if True: memory learns residual logits; else memory outputs a dist directly
HID_LAYER_ID = 32
EPS = 1e-12
MACRO_BATCH_SIZE = 16
LR = 1e-4

# =========================
# Optional: example train step wrapper
# =========================

def softmax_T(logits, T=1.0):
    return F.softmax(logits / T, dim=-1)

def kl_divergence(p_logit, q_logit, T=1.0, eps=1e-8):
    # KL( p || q ); both inputs are logits
    p = softmax_T(p_logit, T)
    q = softmax_T(q_logit, T)
    return torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)), dim=-1)

def sparse_kl_with_tail(
    student_logits,   # [A, V] logits from base+memory at answer positions
    teacher_ids_with,          # [A, M] teacher union token ids (=-1 means padding)
    teacher_logprob_with,         # [A, M] teacher log-probs on union tokens
    tail_logprob         # [A]    teacher log-prob for OTHER bucket
):
    eps = 1e-8

    # log p_student over full vocab
    # [B, A, V]
    log_p_s = F.log_softmax(student_logits, dim=-1)

    # teacher top-K probs (optionally renormalized with tail)
    # [B, A, K]
    p_t_topk = torch.exp(teacher_logprob_with)
    p_t_tail = torch.exp(tail_logprob).unsqueeze(-1)  # [B, A, 1]

    # Optional: renormalize (robust if teacher probs are not perfectly normalized)
    mass_topk = p_t_topk.sum(-1, keepdim=True)        # [B, A, 1]
    Z = (mass_topk + p_t_tail).clamp_min(eps)         # [B, A, 1]
    p_t_topk = p_t_topk / Z
    p_t_tail = p_t_tail / Z

    log_p_t_topk = torch.log(p_t_topk + eps)          # [B, A, K]
    log_p_t_tail = torch.log(p_t_tail + eps).squeeze(-1)  # [B, A]

    # log p_student on teacher's top-K tokens
    # teacher_ids_with: [B, A, K]
    log_p_s_topk = log_p_s.gather(dim=-1, index=teacher_ids_with)  # [B, A, K]
    p_s_topk     = torch.exp(log_p_s_topk)                         # [B, A, K]

    # student tail mass = 1 - sum_{topK} p_s
    p_s_topk_sum = p_s_topk.sum(-1)               # [B, A]
    p_s_tail     = (1.0 - p_s_topk_sum).clamp_min(eps)   # [B, A]
    log_p_s_tail = torch.log(p_s_tail)            # [B, A]

    # KL = sum_i p_t(i) [log p_t(i)-log p_s(i)] + p_tail[log p_tail-log p_s_tail]
    term_topk = (p_t_topk * (log_p_t_topk - log_p_s_topk)).sum(-1)  # [B, A]
    term_tail = p_t_tail.squeeze(-1) * (log_p_t_tail - log_p_s_tail)  # [B, A]

    kl = term_topk + term_tail    # [B, A]
    return kl.mean()              # scalar

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


prompt_base = "Answer the image related question. For many fact checking questions, the desired answer would be named entities instead of common concept, for example \"Mount Everest\" instead of \"mountain top\", \"River Thames\" instead of \"river\". Be concise, output the answer only, without any additional words."
prompt_teacher = "Answer the image related question. For many fact checking questions, the desired answer would be named entities instead of common concept, for example \"Mount Everest\" instead of \"mountain top\", \"River Thames\" instead of \"river\". Be concise, output the answer only, without any additional words. An additional image is provided for your reference, but it is not guaranteed to be relevant to original question."

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


def main():
    train_ds = datasets.load_from_disk('/data_external/InfoSeek/train_gen_combined').with_format('torch')
    #train_ds = train_ds.cast_column("image", HFImage(decode=True))
    #train_ds = train_ds.select(range(100000))
    test_ds = datasets.load_from_disk('/data_external/InfoSeek/val_combined').with_format('torch')

    base_model = Qwen3VLForConditionalGeneration.from_pretrained("/wyy/models/Qwen3-VL-8B-Instruct", local_files_only=True, attn_implementation="flash_attention_2", dtype=torch.bfloat16, device_map='cpu')

    #base_model = Qwen3VLForConditionalGeneration.from_pretrained("/wyy/models/Qwen3-VL-8B-Instruct", local_files_only=True, attn_implementation="eager", dtype=torch.float16, device_map='cpu')

    for p in base_model.parameters():
        p.requires_grad_(False)
    processor = AutoProcessor.from_pretrained('/wyy/models/Qwen3-VL-8B-Instruct')

    memory = MemoryMLP()
    for p in memory.parameters():
        p.requires_grad_(True)

    # saved_dict = torch.load('/data_external/MMPMem/checkpoints/step5000_cekd.pt')
    # memory.load_state_dict(saved_dict, strict=True)

    model = WrappedLM(base_model, memory, config=base_model.config, processor=processor, layer_idx_for_mem=HID_LAYER_ID)


    def collate_train(batch):
        row = {k: [e[k] for e in batch] for k in batch[0].keys()}
        question = row['question']
        qimg = row['image']
        answers = row['answer']
        longest_answers = [max(answer, key=len)+processor.tokenizer.eos_token+'\n' for answer in answers]
        #qid = row['data_id']

        inputs, input_lengths, answer_ids, answer_mask, answer_lengths = process_input(processor, question, qimg, longest_answers)
        #teacher_ids_with, teacher_logprob_with, tail_logprob = process_teacher_batch(row['mix_idx'], row['mix_logp'], row['tail_logp'], answer_lengths)
        
        teacher_ids = row['per_ev_top_ids']
        teacher_logps = row['per_ev_top_logps']
        for i in range(len(teacher_ids)):
            pad_pos = teacher_ids[i] == -1
            teacher_ids[i][pad_pos] = processor.tokenizer.pad_token_id
        teacher_tails = row['per_ev_tail']
        ret_scores = row['scores']
        return inputs, input_lengths, answer_ids, answer_mask, answer_lengths, teacher_ids, teacher_logps, teacher_tails, ret_scores

    def collate_eval(batch):
        row = {k: [e[k] for e in batch] for k in batch[0].keys()}
        question = row['question']
        qimg = row['image']
        base_inputs = process_input_test(processor, question, qimg).to(model.device)
        return base_inputs, row['answer_eval']

    train_ds = DataLoader(train_ds, batch_size=MACRO_BATCH_SIZE, shuffle=True, collate_fn=collate_train, num_workers=6, prefetch_factor=2,)
    test_ds = DataLoader(test_ds, batch_size=MACRO_BATCH_SIZE, collate_fn=collate_eval)

    
    num_warmup_steps = 200
    num_training_steps = len(train_ds)*20
    def lr_lambda(step: int, m_start=1e-7, m_min=1e-6) -> float:
        # Clamp step to [0, num_training_steps] for safety
        step = min(max(step, 0), num_training_steps)

        if step < num_warmup_steps:
            # Linear warmup from m_start -> 1.0
            if num_warmup_steps == 0:
                return 1.0
            return m_start + (1.0 - m_start) * (step / num_warmup_steps)

        # Cosine decay from 1.0 -> m_min
        denom = max(1, num_training_steps - num_warmup_steps)
        progress = (step - num_warmup_steps) / denom  # in [0,1]
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
        return m_min + (1.0 - m_min) * cosine
    optimizer = torch.optim.AdamW(list(memory.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    from accelerate.utils import DistributedDataParallelKwargs
    ddp = DistributedDataParallelKwargs(
        broadcast_buffers=False        # critical
    )
    accel = Accelerator(kwargs_handlers=[ddp])
    model, optimizer,scheduler, train_ds, test_ds = accel.prepare(model, optimizer, scheduler, train_ds, test_ds)

    unw_model = accel.unwrap_model(model)
    _memory = unw_model.memory
    
    if accel.is_main_process:
        wandb_run = wandb.init(project="MMMem", name="CE+KD")

    # model.eval()
    # correct = torch.tensor(0., device=model.device)
    # cb = torch.tensor(0., device=model.device)
    # all_gt = []
    # all_pred =  []
    # all_pred_base = []
    # for row in tqdm(test_ds):
    #     inputs, gts = row
    #     with torch.inference_mode():
    #         output_ids = model.generate(**inputs, mix_mode='mem')
    #         input_len = inputs.input_ids.size(1)
    #         generated_ids = output_ids[:, input_len:]
    #         text = processor.batch_decode(generated_ids, skip_special_tokens=True)

    #     for gt, pred in zip(gts, text):
    #         if '{' in gt[0] and '}' in gt[0]:
    #             gt = ast.literal_eval(gt[0])
    #         score_m = score_infoseek(pred, gt)
    #         correct += score_m['acc']
    #         all_gt.append(gt)
    #         all_pred.append(pred)
    
    # if accel.is_main_process:
    #     wandb_run.log({'eval/acc': correct/1000, 'eval/acc_base':correct/1000, 'eval/epoch': 0})
    # 14.90


    for ep in range(20):
        model.train()
        acc_loss = []
        pbar = tqdm(train_ds)
        for step, batch in enumerate(train_ds):
            inputs, input_lengths, answer_ids, answer_mask, answer_lengths, teacher_ids, teacher_logps, teacher_tails, ret_scores = batch
            stats = train_step(model,unw_model.config, accel, processor, optimizer,scheduler, model.device, inputs, input_lengths, answer_ids, answer_mask,  teacher_ids, teacher_logps, teacher_tails, ret_scores)

            acc_loss.append(stats['loss'])
            cur_loss = np.mean(acc_loss)
            pbar.set_postfix({"Loss": cur_loss})
            pbar.update()
            if accel.is_main_process:
                log_dict = {
                    "train/epoch": ep + 1,
                    "train/step": ep * len(train_ds) + step,
                    "train/loss": stats['loss'],
                    "train/ce_loss": stats['ce'],
                    "train/kl_loss": stats['kl'],
                    "train/current_lr": scheduler.get_last_lr()[0],
                }
                wandb_run.log(log_dict)
            if step%5 == 0:
                torch.cuda.empty_cache()

            if step%1000 == 0:
                accel.wait_for_everyone()
                state = accel.get_state_dict(_memory)         # gathered on rank 0, offloaded to CPU
                if accel.is_main_process:
                    torch.save(state, f"/data_external/MMPMem/checkpoints/step{step}_cekd.pt")

        accel.wait_for_everyone()
        state = accel.get_state_dict(_memory)         # gathered on rank 0, offloaded to CPU
        if accel.is_main_process:
            torch.save(state, f"/data_external/MMPMem/checkpoints/{ep}_ce_only.pt")
        accel.wait_for_everyone()

        # saved_dict = torch.load('/data_external/MMPMem/checkpoints/0_ce_only.pt')
        # memory.load_state_dict(saved_dict, strict=True)
        model.eval()
        correct =torch.tensor(0., device=model.device)
        em = 0
        cb = torch.tensor(0., device=model.device)
        eb = 0
        for row in tqdm(test_ds):
            inputs, gts = row
            with torch.inference_mode():
                output_ids = model.module.generate(**inputs, mix_mode='mix')
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