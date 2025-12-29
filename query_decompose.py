import os, json, math, argparse
os.environ['CUDA_VISIBLE_DEVICES']="2"
os.environ["HF_HOME"]='/data_external/hf_cache'
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import datasets
from datasets import Image as HFImage, Dataset as HFDataset
from vllm import LLM, SamplingParams
from vllm.inputs.data import TextPrompt
from tqdm import tqdm
MACRO_BATCH_SIZE = 24
#dpr = datasets.load_from_disk('/data_external/dpr_wiki_chunk')
train_ds = datasets.load_dataset('framolfese/2WikiMultihopQA', split='train')

llm = LLM(
    model='/data_external/Qwen3-4B-Thinking-2507',
    dtype='bfloat16',
    tensor_parallel_size=1,
    max_num_batched_tokens=110000,
    #max_num_seqs=150,
    max_model_len=32768,
    #max_logprobs=32,
    gpu_memory_utilization=0.95,
    trust_remote_code=True,
    #limit_mm_per_prompt={"image": 2, "video": 0},
)
tokenizer = AutoTokenizer.from_pretrained('/data_external/Qwen3-4B-Thinking-2507')
sp = SamplingParams(
        max_tokens=12000,
        temperature=0.6,
        top_k=20,
        top_p=0.95,
)


template = "Given the factual knowledge required question, try your best to solve it step by step. If you are not sure about some factual information, you may use placeholder tokens \"<fact 1>\", \"<fact 2>\", etc. and \"<conclusion 1>\", \"<conclusion 2>\", etc. Do NOT fabricate fact. If you chose to output placeholder tokens, do NOT state why (because you don't know, you are not sure, etc.), just complete the step by step template as a normal, fluent sentence, with replacing the factual information to the placeholder tokens. Follow the output syntax of: \"Question: question text\nReasoning: Your step by step reasoning\nAnswer:\\boxed{answer text}\". For example:\n\nQuestion: Where was the place of death of the director of film Ladies Courageous?\nReasoning: The director of film Ladies Courageous is John Rawlins.\nJohn Rawlins died on May 20 1997, in Arcadia, California.\n\nAnswer:\\boxed{Arcadia, California}\n\n\nQuestion: Which film has the director who died earlier, Budak Nafsu or The Bilingual Lover?\n\nReasoning: Director of Budak Nafsu is <fact 1>.\n<fact 1> died on <fact 2>.\nDirector of The Bilingual Lover is Vicente Aranda. Vicente Aranda died on 26 May 2015. Comparing <fact 2> and May 2015, the answer is <conclusion 1>.\n\nAnswer:\\boxed{<conclusion 1>}.\n\n\nFollowing passages are taken from Wikipedia, that might contain factual information needed, but DO NOT cite the passage when output the reasoning steps. That is, you should NEVER output sentences like \"evidence 1 stated fact 1\", \"passage 2 mentioned fact 2\", \"as stated by passages\", etc. You should always make the reasoning template as normal fluent sentences without citing the sources of facts.\n\nEvidence passages:"

input_buffer = []
output_buffer = []
batch_step = 0
outfile = open('/data_external/MMPMem/2wiki_cot.jsonl', 'w')
pbar = tqdm(total=len(train_ds)//MACRO_BATCH_SIZE)
for step, row in enumerate(train_ds):
    prompt = template + " "
    for i, title in enumerate(row['supporting_facts']['title']):
        ev_idx = row['context']['title'].index(title)
        evidence = row['context']['sentences'][ev_idx]
        evidence = ' '.join(evidence)
        prompt += f'\nPassage {i+1}: ' + evidence + '\n\n'
    prompt = prompt+'\n\nNow answer the following question:\n\nQuestion: '+row['question']

    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    if batch_step < MACRO_BATCH_SIZE:
        input_buffer.append(prompt)
        output_buffer.append({'question': row['question'], 'answer': row['answer'], 'cot': "", "pred": ""})
        batch_step += 1
    else:
        outs = llm.generate(prompts=input_buffer, sampling_params=sp,use_tqdm=False)
        batch_step = 0

        for i, out_text in enumerate(outs):
            try:
                cot = out_text.outputs[0].text.split('</think>')[1]
                cot_text = re.search(r'Reasoning:\s(.+)Answer:', cot, re.DOTALL)
                pred = re.search(r'\\boxed\{(.+)\}:', cot, re.DOTALL)
                output_buffer[i]['cot'] = cot_text.groups()[0].strip()
                output_buffer[i]['cot'] = pred.groups()[0].strip()
            except IndexError as e:
                pass
            except AttributeError as e:
                pass
        for xx in output_buffer:
            outfile.write(json.dumps(xx))
            outfile.write('\n')
        input_buffer.clear()
        output_buffer.clear()
        pbar.update()
