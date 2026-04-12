from __future__ import annotations
import os
#os.environ['HF_HOME'] = '/data_external/hf_cache'
os.environ['CUDA_VISIBLE_DEVICES']="0"
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
os.environ['HF_HUB_OFFLINE'] = "1" 


import time
import json
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, Qwen3VLForConditionalGeneration, CLIPModel, CLIPProcessor, AutoProcessor, Qwen3VLProcessor, Qwen3ForCausalLM
import datasets
from modelling_memory import MemoryMLP, WrappedLM

@dataclass
class GenerationTiming:
    ttft_seconds: float
    tps_decode: float
    tps_end_to_end: float
    total_seconds: float
    generated_tokens: int
    output_ids: Optional[torch.Tensor] = None


class TimingStreamer:
    """
    Minimal streamer for timing GenerationMixin.generate().

    It records:
      - time of first generated token (TTFT anchor)
      - total number of generated tokens

    This avoids tokenizer decoding overhead, so the timing reflects model-side
    generation rather than text post-processing.

    Assumptions:
      - batch size = 1
      - generation is greedy or sampling (num_beams=1)
    """

    def __init__(self, prompt_length: int, device: Optional[torch.device] = None):
        self.prompt_tokens_to_skip = int(prompt_length)
        self.device = torch.device(device) if device is not None else None

        self.first_token_time: Optional[float] = None
        self.generated_token_count: int = 0
        self.generated_token_ids = []

        self.first_token_event = threading.Event()
        self.end_event = threading.Event()

    def put(self, value: Any) -> None:
        """
        Called by generate() whenever new token ids are available.

        `value` is usually a tensor shaped like:
          - [1, seq_len] for the initial prompt push
          - [1] or [1, 1] for generated tokens
        """
        if isinstance(value, torch.Tensor):
            ids = value.detach().reshape(-1).tolist()
        elif isinstance(value, (list, tuple)):
            flat = []
            for x in value:
                if isinstance(x, torch.Tensor):
                    flat.extend(x.detach().reshape(-1).tolist())
                elif isinstance(x, (list, tuple)):
                    flat.extend(list(x))
                else:
                    flat.append(int(x))
            ids = flat
        else:
            ids = [int(value)]

        # Skip prompt tokens so TTFT is measured on the first *generated* token.
        if self.prompt_tokens_to_skip > 0:
            n_skip = min(self.prompt_tokens_to_skip, len(ids))
            ids = ids[n_skip:]
            self.prompt_tokens_to_skip -= n_skip

        if not ids:
            return

        if self.first_token_time is None:
            if self.device is not None and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.first_token_time = time.perf_counter()
            self.first_token_event.set()

        self.generated_token_count += len(ids)
        self.generated_token_ids.extend(ids)

    def end(self) -> None:
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.end_event.set()


def _infer_prompt_length(model_inputs: Dict[str, Any]) -> int:
    """
    For multimodal GenerationMixin models, prompt length is still determined by input_ids.
    """
    if "input_ids" not in model_inputs:
        raise ValueError(
            "Could not infer prompt length because `input_ids` is missing. "
            "Pass `prompt_length=...` explicitly."
        )
    input_ids = model_inputs["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("model_inputs['input_ids'] must be a torch.Tensor.")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("This helper expects batch size 1.")
    return int(input_ids.shape[1])


def timed_generate(
    model,
    model_inputs: Dict[str, Any],
    *,
    generate_kwargs: Optional[Dict[str, Any]] = None,
    prompt_length: Optional[int] = None,
    device: Optional[torch.device] = None,
    return_output_ids: bool = True,
) -> GenerationTiming:
    """
    Measure TTFT and TPS for any GenerationMixin-compatible model.

    Parameters
    ----------
    model:
        Your wrapped model with `.generate(...)`.
    model_inputs:
        Dict passed directly into `model.generate(...)`, e.g.
        {
            "input_ids": ...,
            "attention_mask": ...,
            "pixel_values": ...,
            ...
        }
    generate_kwargs:
        Keyword args for `generate`, e.g.
        {
            "max_new_tokens": 32,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
        }
    prompt_length:
        Optional explicit prompt token length. If omitted, inferred from input_ids.
    device:
        Device for synchronization. If omitted, inferred from first tensor input.
    return_output_ids:
        Whether to return generated token ids.

    Returns
    -------
    GenerationTiming
    """
    generate_kwargs = dict(generate_kwargs or {})

    if prompt_length is None:
        prompt_length = _infer_prompt_length(model_inputs)

    if device is None:
        for v in model_inputs.values():
            if isinstance(v, torch.Tensor):
                device = v.device
                break
    device = torch.device(device) if device is not None else None

    # Streamers are intended for greedy/sampling generation, not beam search benchmarking.
    num_beams = int(generate_kwargs.get("num_beams", 1))
    if num_beams != 1:
        raise ValueError("This timing helper expects num_beams=1 for reliable TTFT/TPS measurement.")

    streamer = TimingStreamer(prompt_length=prompt_length, device=device)

    output_holder: Dict[str, Any] = {}
    error_holder: Dict[str, BaseException] = {}

    def _worker():
        try:
            with torch.inference_mode():
                output_holder["result"] = model.generate(
                    **model_inputs,
                    streamer=streamer,
                    return_dict_in_generate=True,
                    mix_mode='mix', mix_lambda=0.6, branch="generation",
                    **generate_kwargs,
                )
        except BaseException as e:
            error_holder["error"] = e
        finally:
            streamer.end()

    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    worker.join()

    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    t_end = time.perf_counter()

    if "error" in error_holder:
        raise error_holder["error"]

    if streamer.first_token_time is None:
        raise RuntimeError("No generated token was observed. Check max_new_tokens / stopping criteria.")

    ttft = streamer.first_token_time - t0
    total = t_end - t0
    n_gen = streamer.generated_token_count

    # Decode TPS: tokens/sec after the first generated token.
    if n_gen <= 1:
        tps_decode = 0.0
    else:
        decode_window = max(t_end - streamer.first_token_time, 1e-12)
        tps_decode = (n_gen - 1) / decode_window

    # End-to-end throughput, including prefill.
    tps_e2e = n_gen / max(total, 1e-12)

    output_ids = None
    if return_output_ids:
        result = output_holder["result"]
        output_ids = result.sequences if hasattr(result, "sequences") else result

    return GenerationTiming(
        ttft_seconds=ttft,
        tps_decode=tps_decode,
        tps_end_to_end=tps_e2e,
        total_seconds=total,
        generated_tokens=n_gen,
        output_ids=output_ids,
    )


def benchmark_generate(
    model,
    model_inputs: Dict[str, Any],
    *,
    generate_kwargs: Optional[Dict[str, Any]] = None,
    prompt_length: Optional[int] = None,
    device: Optional[torch.device] = None,
    warmup: int = 2,
    runs: int = 10,
) -> Dict[str, float]:
    """
    Run several timed generations and return mean/std metrics.
    """
    import statistics as stats

    for _ in range(warmup):
        _ = timed_generate(
            model,
            model_inputs,
            generate_kwargs=generate_kwargs,
            prompt_length=prompt_length,
            device=device,
            return_output_ids=False,
        )

    records = []
    for _ in range(runs):
        rec = timed_generate(
            model,
            model_inputs,
            generate_kwargs=generate_kwargs,
            prompt_length=prompt_length,
            device=device,
            return_output_ids=False,
        )
        records.append(rec)

    ttfts = [r.ttft_seconds for r in records]
    tps_dec = [r.tps_decode for r in records]
    tps_e2e = [r.tps_end_to_end for r in records]
    totals = [r.total_seconds for r in records]
    ntoks = [r.generated_tokens for r in records]

    return {
        "runs": runs,
        "ttft_mean": stats.mean(ttfts),
        "ttft_std": stats.pstdev(ttfts) if runs > 1 else 0.0,
        "tps_decode_mean": stats.mean(tps_dec),
        "tps_decode_std": stats.pstdev(tps_dec) if runs > 1 else 0.0,
        "tps_e2e_mean": stats.mean(tps_e2e),
        "tps_e2e_std": stats.pstdev(tps_e2e) if runs > 1 else 0.0,
        "total_mean": stats.mean(totals),
        "generated_tokens_mean": stats.mean(ntoks),
    }

prompt_base = "Answer the image related question. For fact checking questions, the desired answer would be named entities instead of common concept, for example \"Mount Everest\" instead of \"mountain top\", \"River Thames\" instead of \"river\". Be concise, output the answer only, without any additional words."
def process_input_test(processor,x, question_batch, qimg_batch, ret_image=None):
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
                        {"type": "text", "text": prompt_base + "Wikipedia Articles:\n" +x+ "\nQuestion: " +question},
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

from tqdm import tqdm
import torch.nn.functional as F
if __name__ == "__main__":
    test_ds = datasets.load_from_disk('/data_external/InfoSeek/val_combined').with_format('torch')
    #test_ds = datasets.load_from_disk('/data_external/InfoSeek/val_full').with_format('torch')
    #test_ds = test_ds.filter(lambda x: x['utype'] == 'val_unseen_entity') #val_unseen_question
    #test_ds = test_ds.select(range(5000))

    base_model = Qwen3VLForConditionalGeneration.from_pretrained("/wyy/models/Qwen3-VL-8B-Instruct", local_files_only=True, attn_implementation="flash_attention_3", dtype=torch.bfloat16, device_map='cuda')

    # #for p in base_model.parameters():
    # #   p.requires_grad_(False)
    processor = AutoProcessor.from_pretrained('/wyy/models/Qwen3-VL-8B-Instruct')

    memory = MemoryMLP()
    memory = memory.to('cuda')
    model = WrappedLM(base_model, memory, config=base_model.config, processor=processor, layer_idx_for_mem=-1)
    
    import faiss
    # with open("/data_external/InfoSeek/oven_entity_train.jsonl") as f:
    #     oven = f.readlines()
    #     oven = [json.loads(e) for e in oven]
    # with open("/data_external/InfoSeek/oven_entity_val.jsonl") as f:
    #     temp = f.readlines()
    #     temp = [json.loads(e) for e in temp]
    #oven.extend(temp)
    #oven_image_id_entity_map = {e['image_id']:e['entity_text'] for e in tqdm(oven)}

    # evqa_kb = datasets.load_from_disk('/data_external/evqa/image_kb')
    # kb_titles = evqa_kb['title']
    # kb_title_set = set(evqa_kb['title'])
    # kb_title_map = {t:i for i, t in enumerate(evqa_kb['title'])}

    with open('/data_external/InfoSeek/ret_img_gt_docs_p1.jsonl') as f:
        train_ref_docs = [json.loads(e) for e in f.readlines()]
    
    with open('/data_external/InfoSeek/ret_img_gt_docs_p2.jsonl') as f:
        temp = [json.loads(e) for e in f.readlines()]
    train_ref_docs.extend(temp)

    sample_chunks = '\n\n'.join(train_ref_docs[0]['top5_chunks'])

    sample_chunks5 = '\n\n'.join(train_ref_docs[0]['top5_chunks'] + train_ref_docs[1]['top5_chunks'] +train_ref_docs[2]['top5_chunks']+train_ref_docs[3]['top5_chunks']+train_ref_docs[4]['top5_chunks'])

    # index_cpu: faiss.IndexFlatIP = faiss.read_index('/data_external/InfoSeek/corpus_qwen3_flatip.idx')

    # from qwen3_vl_embedding import Qwen3VLEmbedder
    # retriever = Qwen3VLEmbedder(model_name_or_path="/data_external/Qwen3-VL-Embedding-2B", attn_implementation="flash_attention_2", dtype=torch.bfloat16)
    
    from torchvision.transforms.functional import to_pil_image
    def collate_eval(batch):
        row = {k: [e[k] for e in batch] for k in batch[0].keys()}
        question = row['question']
        qimg = row['image']
        base_inputs = process_input_test(processor,sample_chunks5, question, qimg).to('cuda')
        return base_inputs, row['answer_eval'], row['data_id'], qimg
    test_dl = DataLoader(test_ds, batch_size=1, collate_fn=collate_eval,shuffle=True)

    summarize = {
        'ttft': [],
        'tps': [],
    }

    testiter = test_dl.__iter__()
    ret_time = []
    for _ in range(100):
        testbatch = testiter.__next__()

        # t0 = time.perf_counter()
        # feats = retriever.process([{'image': to_pil_image(testbatch[3][0])}])
        # feats = F.normalize(feats, p=2, dim=-1).float().cpu().numpy()

        # Q = feats  # FAISS expects float32, shape [B, D]
        # D, I = index_cpu.search(Q, 5)     # batched FAISS call for this block
        # t1 = time.perf_counter()
        # ret_time.append(t1-t0)
        generate_kwargs = {
            "max_new_tokens": 10,
            "do_sample": False,
            #"num_beams": 1,
            "use_cache": True,
        }
        stats = benchmark_generate(
            model,
            testbatch[0],
            generate_kwargs=generate_kwargs,
            warmup=3,
            runs=10,
        )
        print(stats)
        summarize['ttft'].append(stats['ttft_mean'])
        summarize['tps'].append(stats['tps_decode_mean'])
    
    
    import numpy as np
    pass
    