import os, json, math, argparse


#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="1"
os.environ["HF_HOME"]='/data_external/hf_cache'
#os.environ['VLLM_FLASH_ATTN_VERSION']="2"
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Literal
from dataclasses import dataclass
import ast
from collections import OrderedDict
import uuid, hashlib
import itertools
import time
import re

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen3VLProcessor, AutoConfig,Qwen3VLForConditionalGeneration
from PIL import Image
import datasets
from datasets import Image as HFImage, Dataset as HFDataset
from vllm import LLM, SamplingParams
from vllm.inputs.data import TextPrompt
from tqdm import tqdm

from modelling_memory import MemoryMLP
from inference import generate_with_memory
from eval_util import score_infoseek

mcq_keys = ['A', 'B', 'C', 'D']
test_ds = datasets.load_from_disk('/data_external/MRAG')

test_ds:HFDataset = datasets.concatenate_datasets([test_ds['train'], test_ds['test']])
test_ds = test_ds.cast_column('image', HFImage(decode=True))

processor= Qwen3VLProcessor.from_pretrained('/wyy/models/Qwen3-VL-8B-Instruct')
model = Qwen3VLForConditionalGeneration.from_pretrained("/wyy/models/Qwen3-VL-8B-Instruct", local_files_only=True, attn_implementation="flash_attention_2", dtype=torch.bfloat16, device_map='cpu')
model = model.to('cuda')
memory = MemoryMLP()
saved_dict = torch.load('/data_external/MMPMem/checkpoints/14.pt')
memory.load_state_dict(saved_dict, strict=True)
memory = memory.to('cuda')


def process_prompt(inputs):
    question = inputs['question']
    for key in mcq_keys:
        question+= f' {key}: {inputs[key]} |'
    question=question[:-2]+'\n'

    messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": inputs['image'],
                    },
                    {"type": "text", "text": "Please answer the image related multiple choice question. Be concise, output the answer only, without any additional words.\n" + "\nQuestion: " +question},
                    #{"type": "text", "text": "Please answer the image related multiple choice question.\nOutput the final answer, the option letter only, wrapped with \"\\boxed{}\"." + "\nQuestion: " +question},
                ],
            }
        ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        #padding=True, 
        return_tensors="pt"
        )

    # req = {
    #         "prompt": text_inputs,                                  # must contain two image placeholders per your model template
    #         "multi_modal_data": {"image": [inputs['image']]},       # always pass the images
    #         #"multi_modal_uuids": {"image": [img1_uuid, None]},
    # }
    return inputs

cm =0
cb=0
pbar = tqdm(total=len(test_ds))
pm_wrong= []
mm_corect= False
base_correct = False
for step,row in enumerate(test_ds):
    inputs = process_prompt(row)
    inputs = inputs.to('cuda')
    with torch.inference_mode():
        gid, text, _ = generate_with_memory(model, memory, processor.tokenizer, inputs, eos_token_id=processor.tokenizer.eos_token_id, max_new_tokens=40, eta_or_lambda=0.6)
        gid, _, text_base = generate_with_memory(model, memory, processor.tokenizer, inputs, eos_token_id=processor.tokenizer.eos_token_id, max_new_tokens=40, eta_or_lambda=1.)
    

    try:
        pred = re.search(r'\\boxed\{(.+)\}', text.lower(), re.DOTALL)
        pred = pred.groups()[0]
    except:
        pred = ""
    if row['answer_choice'].lower() in pred.lower() or row['answer'].lower() in text.lower():
        cm +=1
        mm_corect = True
    else:
        mm_correct = False
    
    try:
        pred_base = re.search(r'\\boxed\{(.+)\}', text_base.lower(), re.DOTALL)
        pred_base = pred_base.groups()[0]
    except:
        pred_base = ""
    if row['answer_choice'].lower() in pred_base.lower() or row['answer'].lower() in text_base.lower():
        cb +=1
        base_correct = True
    else:
        base_correct = False
    
    if base_correct and not mm_corect:
        pm_wrong.append([row['answer_choice'], text, text_base])

    pbar.update()
    pbar.set_postfix({'BaseLM ACC': 100*(cb/(1+step)), 'PMM ACC':  100*(cm/(1+step))})

pass

