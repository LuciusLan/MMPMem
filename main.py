import argparse
import os
os.environ['HF_HOME'] = '/data_external/hf_cache'
os.environ['CUDA_VISIBLE_DEVICES']="1,2"
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
from typing import List, Tuple, Iterable
from transformers.feature_extraction_utils import BatchFeature
from glob import glob
import ast
import json
import random

import numpy as np
#import pandas as pd
from PIL import Image
from datasets import Image as HFImage

#import faiss
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import AutoModel, Qwen3VLForConditionalGeneration, CLIPModel, CLIPProcessor, AutoProcessor, Qwen3VLProcessor
import datasets
import wandb
#from qwen_vl_utils import process_vision_info
from modelling_memory import MemoryMLP
from inference import generate_with_memory
from eval_util import score_infoseek

from accelerate import dispatch_model, infer_auto_device_map, Accelerator
from accelerate.utils import tqdm
#from tqdm import tqdm

# ---------------------------
# Hyperparameters & utilities
# ---------------------------
TEMPERATURE = 1.0         # softmax temperature for teacher
ALPHA_CE = 1.0            # weight for gold CE
BETA_DELTA = 0.5          # weight for delta residual distillation (optional)
KL_WEIGHT = 1.0           # weight for teacher KL
USE_RESIDUAL = False       # if True: memory learns residual logits; else memory outputs a dist directly
HID_LAYER_ID = 29
EPS = 1e-12
MACRO_BATCH_SIZE = 16




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


def concat_tf_inputs(sample, answer_ids) -> tuple[BatchFeature, int]:
    """
    prompt_ids: [B, P]
    answer_ids: [B, A]
    Returns:
      tf_ids: [B, P + A - 1]  (prompt + answer[:-1])
      tgt:    [B, A]          (full answer as labels)
    """
    # teacher-forcing: input = prompt + answer[:-1], targets = answer[:]
    sample['pixel_values'] = sample['pixel_values'].to(torch.bfloat16)


    tf_inp = torch.cat([sample['input_ids'], answer_ids[:, :-1]], dim=1)
    tgt    = answer_ids                                       # keep batch dimension
    return tf_inp, tgt

def memory_forward(memory, h):
    """
    By convention here:
    - If USE_RESIDUAL: memory outputs residual logits (same vocab size) to be *added* to base.
    - Else: memory outputs direct logits for a distribution to be mixed in prob-space.
    """
    mem_out = memory(h)  # adapt to your memory API (e.g., memory(h))
    # Expect mem_out.logits: [1, vocab]
    return mem_out

# ---------------------------
# Single-sample training step
# ---------------------------
def train_step(model:Qwen3VLForConditionalGeneration, memory:nn.Module, optimizer, device, base_inputs: BatchFeature, input_lengths:torch.Tensor, answer_ids:torch.Tensor,  answer_mask:torch.Tensor, teacher_ids:torch.Tensor, teacher_logprob:torch.Tensor, tail_logprob:torch.Tensor):
    memory.train()
    optimizer.zero_grad(set_to_none=True)

    base_inputs = {k :v.to(device) for k,v in base_inputs.items()}
    input_lengths= input_lengths.to(device)
    answer_ids = answer_ids.to(device)
    answer_mask = answer_mask.to(device)
    teacher_ids = teacher_ids.to(device)
    teacher_logprob = teacher_logprob.to(device)
    tail_logprob = tail_logprob.to(device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # 1) Get base state & logits (frozen)
        B, L_full_max = base_inputs['input_ids'].shape
        _, La_max = answer_ids.shape

        # ----- base model forward, frozen -----
        out = model.forward(
            **base_inputs,
            output_hidden_states=True,
        )
        #logits = out["logits"].detach()             # [B, L_full_max, V]
        hidden = out["hidden_states"][-1].detach()  # [B, L_full_max, H] (or choose another layer)

        B, L_full_max, V = out["logits"].shape
        H = hidden.size(-1)

        # ----- gather per-answer-step logits and hidden states -----
        # Answer token j (0-based) for sample i is predicted at time t_ij = start_i - 1 + j,
        # where start_i = |question_i| = input_lengths[i].
        j_pos = torch.arange(La_max, device=device).unsqueeze(0).expand(B, -1)          # [B, La_max]
        t_indices = (input_lengths.unsqueeze(1) - 1 + j_pos).clamp(0, L_full_max - 1)  # [B, La_max]

        #idx_logits = t_indices.unsqueeze(-1).expand(-1, -1, V)   # [B, La_max, V]
        idx_hidden = t_indices.unsqueeze(-1).expand(-1, -1, H)   # [B, La_max, H]

        #base_ans_logits = logits.gather(1, idx_logits)           # [B, La_max, V]
        h_ans = hidden.gather(1, idx_hidden)                     # [B, La_max, H]

        # ----- memory forward and final logits -----
        # memory: [B, La_max, H] -> [B, La_max, V]
        #h_ans = h_ans.to(memory.device)
        mem_logits = memory(h_ans)                               # [B, La_max, V]
        #final_logits = base_ans_logits + eta * mem_logits        # [B, La_max, V]
        #final_logits = base_ans_logits + mem_logits 
        final_logits = mem_logits

        # ----- CE over gold answer tokens (masking padding) -----
        mask_flat = answer_mask.view(-1)                         # [B*La_max]
        gold_flat = answer_ids.view(-1)[mask_flat]               # [N_tokens]
        logits_flat = final_logits.view(-1, V)[mask_flat]        # [N_tokens, V]

        ce_loss = F.cross_entropy(logits_flat, gold_flat)

        # ----- KL (teacher top-K + tail vs student) -----
        # Teacher probabilities
        p_t_topk = teacher_logprob.exp()                         # [B, La_max, K]
        p_t_tail = tail_logprob.exp()                            # [B, La_max]

        # Student probabilities
        p_s = F.softmax(final_logits / TEMPERATURE, dim=-1)      # [B, La_max, V]
        p_s_topk = p_s.gather(-1, teacher_ids)                   # [B, La_max, K]
        p_s_tail = (1.0 - p_s_topk.sum(dim=-1)).clamp_min(1e-8)  # [B, La_max]

        eps = 1e-8

        # KL(p_t || p_s) over K+1 events: K explicit tokens + tail
        kl_top = (p_t_topk * (teacher_logprob - torch.log(p_s_topk + eps))).sum(dim=-1)  # [B, La_max]
        kl_tail = p_t_tail * (tail_logprob - torch.log(p_s_tail + eps))                  # [B, La_max]
        kl = kl_top + kl_tail                                                            # [B, La_max]

        kl_flat = kl.view(-1)[mask_flat]
        kl_loss = kl_flat.mean()

        # ----- total loss and step -----
        loss = KL_WEIGHT * kl_loss + ALPHA_CE * ce_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(memory.parameters()), max_norm=1.0)
        optimizer.step()

    with torch.no_grad():
        stats = {
            "loss": float(loss.item()),
            "kl": float(kl_loss.item()),
            "ce": float(ce_loss.item()),
            #"delta": float(delta_loss.item()) if BETA_DELTA > 0 else 0.0,
            #"eta": float(eta.sigmoid().item()) if not USE_RESIDUAL else float(eta.item()),
        }
    return stats

prompt_base = "Answer the image related question. For many fact checking questions, the desired answer would be named entities instead of common concept, for example \"Mount Everest\" instead of \"mountain top\". Be concise, output the answer only, without any additional words."
prompt_teacher = "Answer the image related question. For many fact checking questions, the desired answer would be named entities instead of common concept, for example \"Mount Everest\" instead of \"mountain top\". Be concise, output the answer only, without any additional words. An additional image is provided for your reference, but it is not guaranteed to be relevant to original question."

def process_input_test(processor, question, qimg, ret_image=None):

    if not ret_image:
        messages = [
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
        messages = [
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

    # Preparation for inference
    text_inputs = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        #return_dict=True,
        #padding=True, 
        #return_tensors="pt"
        )
    
    inputs = processor(images=qimg, text=text_inputs, padding=True, return_tensors='pt')
    #inputs.pop("token_type_ids", None)   # per model card snippet
    #inputs = inputs.to(teacher.device)
    return inputs
    
def process_input(processor, questions, qimgs, answers, ret_image=None):
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

    inputs = processor(images=qimgs, text=text_inputs, padding=True, return_tensors='pt')
    input_lengths = inputs['attention_mask'].sum(dim=1)
    #inputs.pop("token_type_ids", None)   # per model card snippet
    #inputs = inputs.to(teacher.device)
    return inputs, input_lengths, answer_ids, answer_mask, answer_lengths


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
    

    train_ds = datasets.load_from_disk('/data_external/InfoSeek/train_combined').with_format('torch')
    #train_ds = train_ds.cast_column("image", HFImage(decode=True))
    #train_ds = train_ds.select(range(100000))
    test_ds = datasets.load_from_disk('/data_external/InfoSeek/val_combined').with_format('torch')


    model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", local_files_only=True, attn_implementation="flash_attention_2", dtype=torch.bfloat16, device_map='cpu')

    # device_map = infer_auto_device_map(
    #     model,
    #     max_memory={0: "10GiB", 1: "10GiB"},  # adapt to your hardware
    #     no_split_module_classes=["Qwen3VLTextDecoderLayer"],  # example
    # )
    # model = dispatch_model(model, device_map=device_map)
    for p in model.parameters():
        p.requires_grad_(False)
    #teacher = Qwen2_5_VLForConditionalGeneration.from_pretrained("/data_external/qwen2.5_vl_7b_instruct", local_files_only=True, attn_implementation="flash_attention_2", dtype=torch.bfloat16).cuda()
    processor = AutoProcessor.from_pretrained('Qwen/Qwen3-VL-8B-Instruct')

    #retriever = AutoModel.from_pretrained("/data_external/bge_vl_large", trust_remote_code=True, local_files_only=True, attn_implementation="flash_attention_2", dtype=torch.bfloat16, device_map='cuda')
    #retriever = retriever.cuda()
    #retriever.set_processor("/data_external/bge_vl_large")
    memory = MemoryMLP()
    # memory_device_map = {
    #     "blocks.0": "cuda:0",
    #     "blocks.1": "cuda:0",
    #     "blocks.2": "cuda:0",
    #     "blocks.3": "cuda:0",
    #     "blocks.4": "cuda:1",
    #     "head": "cuda:1"
    # }
    # memory = dispatch_model(memory, device_map=memory_device_map)
    #memory = memory.to('cuda:1')
    #total_params = sum(p.numel() for p in memory.parameters())
    #index = faiss.read_index('/data_external/MRAG/bge.index')

    # Optional: learnable global scale for how much to trust memory residual
    #eta = torch.nn.Parameter(torch.tensor(0.5, device=device))
    optimizer = torch.optim.AdamW(list(memory.parameters()), lr=1e-4)
    
    def collate(x):
        return {k: [e[k] for e in x] for k in x[0].keys()}

    train_ds = DataLoader(train_ds, batch_size=MACRO_BATCH_SIZE, shuffle=True, collate_fn=collate, num_workers=4, prefetch_factor=2,)
    test_ds = DataLoader(test_ds, batch_size=1, collate_fn=collate)

    accel = Accelerator()
    model, memory, optimizer, train_ds, test_ds = accel.prepare(model, memory, optimizer, train_ds, test_ds)

    #saved_dict = torch.load('/data_external/MMPMem/checkpoints/12.pt')
    #memory.load_state_dict(saved_dict, strict=True)

    if accel.is_main_process:
        wandb_run = wandb.init(project="MMMem", name="same_teacher_replace")

    model.eval()
    memory.eval()
    correct = torch.tensor(0., device=model.device)
    cb = torch.tensor(0., device=model.device)
    all_gt = []
    all_pred =  []
    all_pred_base = []
    for row in tqdm(test_ds):
        question = row['question'][0]
        qimg = row['image']
        #answer_id = row['answer']
        #answer_id = processor.tokenizer.encode(answer_id)
        base_inputs = process_input_test(processor, question, qimg).to(model.device)
        with torch.inference_mode():
            gid, text, text_base = generate_with_memory(model, memory, processor.tokenizer, base_inputs, eos_token_id=processor.tokenizer.eos_token_id, max_new_tokens=40, eta_or_lambda=1.)
        #answer_first = answer_id[0]
        #correct += processor.tokenizer._convert_id_to_token(answer_first).lower() == processor.tokenizer._convert_id_to_token(gid.item()).lower()

        # gt = ast.literal_eval(row['answer_eval'])
        gt = row['answer_eval'][0]
        if '{' in gt[0] and '}' in gt[0]:
            gt = ast.literal_eval(gt[0])
        #correct += gt.lower() == text.lower()
        score = score_infoseek(text_base, gt)
        cb += score['acc']
        #cb += gt.lower() == text_base.lower()
        all_gt.append(gt)
        all_pred.append(text)
        all_pred_base.append([question, gt, text_base])

    # print(f"Ep -1: val acc: {cb/1000} | val base: {cb/1000}")
    accel.wait_for_everyone()
    xx = accel.gather(cb)
    cb = accel.reduce(cb, reduction="sum").item()
    
    if accel.is_main_process:
        wandb_run.log({'eval/acc': cb/1000, 'eval/epoch': 0})
    # 14.90


    for ep in range(20):
        model.eval()
        memory.train()
        acc_loss = []
        pbar = tqdm(train_ds)
        batch_buffer = []
        teacher_buffer = []
        batch_step = 0
        for step, row in enumerate(train_ds):
            # if row['teacher_logits'] is None:
            #     continue
            # teacher_logits = row['teacher_logits']
            question = row['question']
            qimg = row['image']
            answers = row['answer']
            #answers = [ast.literal_eval(answer) for answer in answers]
            longest_answers = [max(answer, key=len)+processor.tokenizer.eos_token for answer in answers]
            qid = row['data_id']

            # try:
            #     teacher_logits = torch.load(f'/data_external/SKVQA/DataStore_rep/{qid}.pt')
            # except FileNotFoundError:
            #     #print(f'qid{qid} teacher logit not found')
            #     continue
            # batch_buffer.append([question, qimg, longest_answer])
            # teacher_buffer.append(teacher_logits)
            # if batch_step == MACRO_BATCH_SIZE-1 or step == len(train_ds)-1:
            #     inputs, input_lengths, answer_ids, answer_mask, answer_lengths = process_input(processor, question, qimg, longest_answers)#.to(model.device)
            #     teacher_ids_with, teacher_logprob_with, tail_logprob = process_teacher_batch(teacher_buffer, answer_lengths)

            #     batch_buffer.clear()
            #     teacher_buffer.clear()
            #     batch_step = 0
            # else:
            #     batch_step +=1
            #     continue

            inputs, input_lengths, answer_ids, answer_mask, answer_lengths = process_input(processor, question, qimg, longest_answers)#.to(model.device)
            teacher_ids_with, teacher_logprob_with, tail_logprob = process_teacher_batch(row['mix_idx'], row['mix_logp'], row['tail_logp'], answer_lengths)

            #teacher_logits = {k:v.cuda() if isinstance(v, torch.Tensor) else v for k,v in teacher_logits.items()}

            stats = train_step(model, memory, optimizer, model.device, inputs, input_lengths, answer_ids, answer_mask, teacher_ids_with, teacher_logprob_with, tail_logprob)

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
                }
                wandb_run.log(log_dict)
            if step%30 == 0:
                torch.cuda.empty_cache()

        accel.wait_for_everyone()
        state = accel.get_state_dict(memory)         # gathered on rank 0, offloaded to CPU
        if accel.is_main_process:
            torch.save(state, f"/data_external/MMPMem/checkpoints/{ep}.pt")
        accel.wait_for_everyone()

        model.eval()
        memory.eval()
        correct =torch.tensor(0., device=model.device)
        em = 0
        cb = torch.tensor(0., device=model.device)
        eb = 0
        for row in tqdm(test_ds):
            question = row['question'][0]
            qimg = row['image']
            #answer_id = row['answer']
            #answer_id = processor.tokenizer.encode(answer_id)
            base_inputs = process_input_test(processor, question, qimg).to(model.device)
            with torch.inference_mode():
                gid, text, text_base = generate_with_memory(model, memory, processor.tokenizer, base_inputs, eos_token_id=processor.tokenizer.eos_token_id, max_new_tokens=40, eta_or_lambda=0.5)
            #answer_first = answer_id[0]
            #correct += processor.tokenizer._convert_id_to_token(answer_first).lower() == processor.tokenizer._convert_id_to_token(gid.item()).lower()

            #gt = ast.literal_eval(row['answer'])
            gt = row['answer_eval'][0]
            if '{' in gt[0] and '}' in gt[0]:
                    gt = ast.literal_eval(gt[0])
            #correct += gt.lower() == text.lower()
            #cb += gt.lower() == text_base.lower()
            # all_gt.append(gt)
            # all_pred.append(text)
            # all_pred_base.append(text_base)
            score = score_infoseek(text, gt)
            sb = score_infoseek(text_base, gt)
            correct += score['acc']
            #em += score['em']
            cb += sb['acc']
            #eb += sb['em']
        

        print(f"Ep {ep}: val acc: {correct/1000} | val em: {em/1000}")
        accel.wait_for_everyone()
        correct = accel.reduce(correct, reduction="sum").item()
        cb = accel.reduce(cb, reduction="sum").item()
        if accel.is_main_process:
            wandb_run.log({'eval/acc': correct/1000})
            #wandb_run.log({'eval/em': em/1000})
            wandb_run.log({'eval/acc_base': cb/1000})
            #wandb_run.log({'eval/em_base': eb/1000})
            wandb_run.log({'eval/epoch': ep+1})


if __name__ == "__main__":
    main()