import os, json, math, argparse
os.environ['CUDA_VISIBLE_DEVICES']="2"
os.environ["HF_HOME"]='/data_external/hf_cache'

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
import datasets
from datasets import Image as HFImage, Dataset as HFDataset
from vllm import LLM, SamplingParams
from vllm.inputs.data import TextPrompt
from tqdm import tqdm

#dpr = datasets.load_from_disk('/data_external/dpr_wiki_chunk')
train_ds = datasets.load_dataset('framolfese/2WikiMultihopQA', split='train')

llm = LLM(
    model='/data_external/Qwen3-4B-Thinking-2507',
    dtype='bfloat16',
    tensor_parallel_size=1,
    max_num_batched_tokens=55000,
    #max_num_seqs=150,
    max_model_len=16384,
    #max_logprobs=32,
    gpu_memory_utilization=0.8,
    trust_remote_code=True,
    #limit_mm_per_prompt={"image": 2, "video": 0},
)
prompt = "Given the factual knowledge required question, try your best to solve it step by step. If you are not sure about some factual information, you may use placeholder tokens \"<fact 1>\", \"<fact 2>\", etc. and \"<conclusion 1>\", \"<conclusion 2>\", etc. Follow the syntax of: Question: [question text]\nReasoning: [Your step by step reasoning]\nAnswer:\\boxed{answer text}. For example:\n\nQuestion: Where was the place of death of the director of film Ladies Courageous?\nReasoning: The director of film Ladies Courageous is John Rawlins.\nJohn Rawlins died on May 20 1997, in Arcadia, California.\n\nAnswer:\\boxed{Arcadia, California}\n\n\nQuestion: Which film has the director who died earlier, Budak Nafsu or The Bilingual Lover?\n\nReasoning: Director of Budak Nafsu is <fact 1>.\n<fact 1> died on <fact 2>.\nDirector of The Bilingual Lover is Vicente Aranda. Vicente Aranda died on 26 May 2015. Comparing <fact 2> and May 2015, the answer is <conclusion 1>.\n\nAnswer:\\boxed{<conclusion 1>}.\n\n\nAnswer the following question:\n\nQuestion: When did John V, Prince Of Anhalt-Zerbst's father die?"

LLM.generate(prompt)
pass