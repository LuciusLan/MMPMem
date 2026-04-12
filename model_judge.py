import os, argparse
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["HF_HOME"]='/data_external/hf_cache'
os.environ['CUDA_VISIBLE_DEVICES']="1" #args.device
#os.environ['VLLM_FLASH_ATTN_VERSION']="2"
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"]="spawn"

# os.environ["TORCHINDUCTOR_COMPILE_THREADS"]="1"
# os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"
#os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

import re
import logging
import json
from dataclasses import asdict
import random

import torch
from vllm import LLM, EngineArgs, SamplingParams
import datasets
from PIL import Image
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from internvl_utils import load_image
# from utils import combine_two_pil_images

def set_vllm_loglevel(level: int):
    for name in [
        "vllm",
        "vllm.engine",
        "vllm.engine.loggers",
    ]:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False  # don't bubble up to root


def main(questions, llm:LLM):
    q_collection = []
    c = 0
    #bar = tqdm(total=len(questions))
    step = 0
    set_vllm_loglevel(logging.WARNING)
    base_prompts = []
    aug_prompts = []

    #score_file = open(f'/wyy/DDLVLM/predictions/vision_module/scores_news3_{args.num_aug_tokens}tokens.txt', 'a')
    
    for row in questions:
        #image = row['image']
        template = '<|im_start|>user\nPlease evaluate the student answers of following question, providing the reference answer. The question is regarding an image, though the image is not provided here, you only need to judge if the semantic meaning of student answer aligns with the reference answer. For questions asking for a noun or noun phrase answer, the reference will be provided as a list of possible paraphrased versions of the ground truth. The student answer may still be a paraphrase form out of the list, if you think the answer is an acceptable paraphrase, mark it as correct. For questions requiring numeric answer, the reference answer will be formatted as a dictionary: {"wikidata": ground_truth_value, "range": [min, max]}. In this case, as long as student answer fall in the range inside [min, max], it is considered correct.\nYou MUST output your final judgement wrapped in "\\boxed{}", labelled binary score of either 1 or 0.\n'
        
        #base_prompts.append(f"{template}\nQuestion: {row['question']}\nReference Answer: {row['gt']}\nStudent Answer: {row['base_pred']}<|im_end|>\n<|im_start|>assistant\n")
        aug_prompts.append(f"{template}\nQuestion: {row['question']}\nReference Answer: {row['gt']}\nStudent Answer: {row['pred']}<|im_end|>\n<|im_start|>assistant\n")

        message = [{
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f'Please evaluate the student answers of following question, providing the reference answer. The question is regarding an image, though the image is not provided here, you only need to judge if the semantic meaning of student answer aligns with the reference answer. The student answer may render more information than the reference, but as long as all elements in the reference are mentioned, it is considered correct. Else, if the student is missing some elements, it is partially correct. You MUST output your final judgement wrapped in "\\boxed{{}}", labelled with either \"correct\", \"wrong\", or \"partial\".\n\nQuestion: {row["question"]}\nReference Answer: {row["gt"]}\nStudent Answer: {row["pred"]}'
            }
            ]
        }]

    request_outputs = llm.generate(
        #tokenized,
        prompts= aug_prompts,
        #"multi_modal_data": {"image": image},
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    c = 0
    step = 0
    errored = 0
    for row, out in zip(questions, request_outputs):
        answer = out.outputs[0].text
        answer = answer.split("</think>")[-1]
        try:
            score = re.findall(r'\\boxed\{(.+?)\}', answer, re.DOTALL)[-1]
            if score == '1':
                score = 1
            elif score == '0':
                score = 0
            else:
                score = 0
                errored+=1
        except AttributeError:
            score=0.
            errored+=1
        except IndexError:
            score = 0.
            errored+=1

        #q_collection.append({'question': row['question'], 'gt': row['gt'], 'pred': row['pred'], 'score': score})
        c+= score
        step += 1
    
    #print(f"EP: {ep}")
    print(c/step)
    print(f"Errored : {errored}")

    print()
    
    
    return q_collection


class VCDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index:int):
        return self.data[index]

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

    model_name = "/data_external/Qwen3-4B-Thinking-2507"
    #data.filter()
    engine_args = EngineArgs(
        model=model_name,
        #trust_remote_code=True,
        max_model_len=20000,
        #limit_mm_per_prompt={"image": 1, "video": 0},
        gpu_memory_utilization=0.4,
        max_num_batched_tokens=150000,
        tensor_parallel_size=1,
        seed=2026,
        #language_model_only=True,
        #enforce_eager=True,
        #attention_backend=""
    )
    engine_args = asdict(engine_args)

    llm = LLM(**engine_args)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_k=20,
        top_p=0.8,
        max_tokens=5000,
        seed=2026,
        #presence_penalty=1.5,
        #repetition_penalty=1.0
        #n=1,
        # ask vLLM to return logprobs for top TOP_LOGPROBS_REQUEST tokens
        #logprobs=20,
        # For reasoning distillation you often do NOT want beam search.
        # Stochastic sampling gives you diverse CoTs.
    )

    test_ds = datasets.load_from_disk('/data_external/InfoSeek/val_full').with_format('torch')
    #test_ds = test_ds.filter(lambda x: x['utype'] == 'val_unseen_entity') #val_unseen_question
    test_ds = test_ds.select(range(10000))
    test_ds_did_map = {e: i for i, e in enumerate(test_ds['data_id'])}
    gathered_results_base, gathered_results_mix = torch.load('/latent_aug/MMPMem/result_analy.pt')

    merged_data = []
    for row in tqdm(gathered_results_mix):
        question = test_ds[test_ds_did_map[row['data_id']]]
        gt = row['gt']
        pred = row['pred']
        
        sample = {'question': question['question'], 'gt': gt, 'pred': pred}
        merged_data.append(sample)

    print("mix")
    all_pred = main(merged_data, llm)

    # Mix: 2386
    # Base: 2369
    
    # with open(f'/wyy/DDLVLM/predictions/vision_module/ep{args.ep}.jsonl', 'w') as f:
    #     for row in all_pred:
    #         f.write(json.dumps(row))
    #         f.write('\n')
print()