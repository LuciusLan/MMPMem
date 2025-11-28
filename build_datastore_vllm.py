import os, json, math
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="0,1"
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:False'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Literal
from dataclasses import dataclass
import ast
from collections import OrderedDict
import uuid, hashlib
import itertools
import time

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
from PIL import Image
import datasets
from datasets import Image as HFImage
from vllm import LLM, SamplingParams
from vllm.inputs.data import TextPrompt
from tqdm import tqdm

# =========================
# Config
# =========================
MODEL_NAME              = '/wy_data/Qwen3-VL-8B-Instruct'
TP_SIZE                 = int(os.environ.get("TP_SIZE", "2"))        # tensor parallel degree
DTYPE                   = os.environ.get("DTYPE", "bfloat16")         # "float16" or "bfloat16"
MAX_MODEL_LEN           = int(os.environ.get("MAX_MODEL_LEN", "16384"))
GPU_MEM_UTIL            = float(os.environ.get("GPU_MEM_UTIL", "0.5"))
TEMPERATURE             = 1.0
TOPK_LOGPROB            = 32           # store top-K per position
STORE_PER_EVIDENCE      = False        # set True to also store per-evidence top-K (large!)
EVIDENCE_BATCH_SIZE     = 30           # vLLM batch over evidences per query (set to <= K)
MACRO_BATCH_SIZE        = 12
SAVE_DIR                = '/wy_data/SKVQA/DataStore_rep/'
USE_UNION_TOPK_FOR_MIX  = True         # union-of-topK across evidences to compute mixture
DEVICE                  = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Dataset adapter (replace)
# =========================
@dataclass
class Sample:
    qid: str
    prompt_input_ids: torch.LongTensor       # [1, P]
    prompt_attention_mask: torch.LongTensor  # [1, P]
    answer_input_ids: torch.LongTensor       # [1, A]  (includes <eoa>)
    image_path: str                          # path to main image
    retrieved_paths: List[str]               # list of K evidence image paths
    retrieved_ids: Optional[List[str]] = None
    evidence_weights: Optional[List[float]] = None

class MinimumDS(Dataset):
    def __init__(self, samples):
        super().__init__()
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, indices: Union[list, int]):
        if isinstance(indices, list):
            if isinstance(indices[0], int):
                return [load_image(self.samples[idx]) for idx in indices]
            else:
                raise TypeError("Wrong indices passed from dataloader")
        elif isinstance(indices, int):
            return load_image(self.samples[indices])
        
def pil_content_hash(img: Image.Image) -> str:
    """Deterministic hash of pixel data (ignores EXIF/metadata)."""
    # Hash pixel bytes + size + mode; resilient to filename/metadata differences.
    h = hashlib.sha256()
    h.update(img.tobytes())
    h.update(str(img.size).encode())
    return h.hexdigest()

def uuid_from_hash(h: str) -> str:
    """Turn a hex digest into a canonical UUID string (stable)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, h))

# ---------- Client-side LRU for image UUIDs ----------
class ImageUUIDLRU:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self._map = OrderedDict()  # key: content-hash, val: UUID string

    def get_or_create(self, img: Image.Image) -> str:
        key = pil_content_hash(img)
        if key in self._map:
            self._map.move_to_end(key)
            return self._map[key]
        u = uuid_from_hash(key)
        self._map[key] = u
        if len(self._map) > self.capacity:
            self._map.popitem(last=False)
        return u

    def forget_image(self, img: Image.Image) -> None:
        self._map.pop(pil_content_hash(img), None)

    def clear(self) -> None:
        self._map.clear()


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img

def dataset_iter_dummy():
    """
    Replace this with your own dataset iterator / dataloader.
    Must yield Sample objects as above.
    """
    raise NotImplementedError

# =========================
# Prompt construction
# =========================

def build_two_image_requests(prompt: str, image1_path: Union[str, Image.Image], batch_image2_paths: list[Union[str, Image.Image]], uuid_cache: ImageUUIDLRU):
    """
    Returns a list of vLLM inputs where each element includes:
      - the same image1 (with a stable UUID),
      - a varying image2 (optionally with its own UUID if you expect reuse).
    """
    # Prepare the shared image1 and its stable UUID
    #img1 = Image.open(image1_path).convert("RGB")
    img1_uuid = uuid_cache.get_or_create(image1_path)

    requests = []
    for img2_path in batch_image2_paths:
        #img2 = Image.open(img2_path).convert("RGB")
        # If img2 will repeat across batches, give it a stable UUID as well.
        # If not, set None to fall back to content hashing.
        #img2_uuid = uuid_cache.get_or_create(img2_path)  # or: None

        req = {
            "prompt": prompt,                                  # must contain two image placeholders per your model template
            "multi_modal_data": {"image": [image1_path, img2_path]},       # always pass the images
            "multi_modal_uuids": {"image": [img1_uuid, None]},
        }
        requests.append(req)
    return requests


def process_input(question, answer, image, ret_images, tokenizer=None, uuid_cache:ImageUUIDLRU=None, mode:Literal['augment', 'replace']='augment')->list[TextPrompt]:

    if mode == 'augment':
        prompt_teacher = "Answer the image related question. Be concise, output the answer only, without any additional words. An additional image is provided for your reference, but it is not guaranteed to be relevant to original question."
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
                        "image": ret_images[0],
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": answer
                }]
            }
        ]
    elif mode == 'replace':
        prompt_teacher = "Answer the image related question. Be concise, output the answer only, without any additional words."
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": ret_images[0],
                    },
                    {"type": "text", "text": prompt_teacher + "\nQuestion: " +question},
                ]
            },
            {
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": answer
                }]
            }
        ]
    # Preparation for inference

    # image_inputs, video_inputs = process_vision_info(messages)
    # inputs = processor(
    #     text=[text],
    #     images=image_inputs,
    #     videos=video_inputs,
    #     padding=True,
    #     return_tensors="pt",
    # )
    # inputs = inputs.to("cuda")
    
    prompt = tokenizer.apply_chat_template(messages,tokenize=False,)

    requests = []
    
    if mode == 'augment':
        img1_uuid = uuid_cache.get_or_create(image)

        for ret_image in ret_images:
            #img2 = Image.open(img2_path).convert("RGB")
            # If img2 will repeat across batches, give it a stable UUID as well.
            # If not, set None to fall back to content hashing.
            #img2_uuid = uuid_cache.get_or_create(img2_path)  # or: None

            req = {
                "prompt": prompt,                                  # must contain two image placeholders per your model template
                "multi_modal_data": {"image": [image, ret_image]},       # always pass the images
                "multi_modal_uuids": {"image": [img1_uuid, None]},
            }
            requests.append(req)
    elif mode == 'replace':
        for ret_image in ret_images:
            #img2 = Image.open(img2_path).convert("RGB")
            # If img2 will repeat across batches, give it a stable UUID as well.
            # If not, set None to fall back to content hashing.
            #img2_uuid = uuid_cache.get_or_create(img2_path)  # or: None

            req = {
                "prompt": prompt,                                  # must contain two image placeholders per your model template
                "multi_modal_data": {"image": [ret_image]},       # always pass the images
            }
            requests.append(req)
    return requests

def decode_without_last_token(tokenizer: AutoTokenizer, ids: torch.LongTensor) -> str:
    # ids: [1, A]; return text for ids[:, :-1]
    return tokenizer.decode(ids[0, :-1], skip_special_tokens=False, clean_up_tokenization_spaces=False)

# =========================
# vLLM engine
# =========================
def build_llm():
    llm = LLM(
        model=MODEL_NAME,
        dtype=DTYPE,
        tensor_parallel_size=TP_SIZE,
        max_model_len=MAX_MODEL_LEN,
        max_logprobs=32,
        gpu_memory_utilization=GPU_MEM_UTIL,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 2, "video": 0},
    )
    return llm

def build_sampling_params(max_new_tokens: int, topk_prompt: int) -> SamplingParams:
    # We need top-K prompt logprobs over the *prompt tokens*, and no stochasticity
    return SamplingParams(
        max_tokens=max_new_tokens,           # generate A tokens (we won't use them; we need prompt_logprobs)
        temperature=0.0,                     # deterministic
        top_p=1.0,
        logprobs=TOPK_LOGPROB,               # for generated tokens (unused here)
        prompt_logprobs=topk_prompt,         # <- crucial: top-K logprobs for each prompt token
        detokenize=False,
        allowed_token_ids=list(range(VOCAB_SIZE)),
        repetition_penalty=1.0,
        presence_penalty=1.5
    )

# =========================
# Utilities for top-K packs & tail
# =========================
def extract_prompt_topk_for_answer(outputs, answer_len: int, tokenizer) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    From vLLM outputs, extract for the *last answer_len* prompt positions:
      - top_k_ids:  [A, K]
      - top_k_logp: [A, K]
      - tail_mass:  [A]  (1 - sum(exp(top_k_logp)))
    NOTE: If the true token's logprob isn't in top-K, tail mass includes it (as desired).
    """
    # vLLM returns for each request: outputs[i].prompt_logprobs: List[Optional[Dict[token->LogProb]]]
    # Length == len(prompt_tokens). Take the last A entries.
    #out = outputs.outputs[0]  # single request
    plps = outputs.prompt_logprobs  # list length = prompt_len
    assert plps is not None and len(plps) >= answer_len, "Prompt logprobs not available or too short."
    ans_plps = plps[-answer_len:]  # list of dicts (or None) per answer position

    # Build dense tensors (CPU)
    top_ids_list, top_logp_list, tail_list = [], [], []
    for d in ans_plps:
        # Each d is expected to be a dict: key -> value
        # key: token id (int) or token string
        # value: LogProb object or { 'logprob': float, 'rank': int, 'decoded_token': str } or float
        if not isinstance(d, dict) or len(d) == 0:
            top_ids_list.append(torch.full((TOPK_LOGPROB,), -1, dtype=torch.int32))
            top_logp_list.append(torch.full((TOPK_LOGPROB,), float("-inf"), dtype=torch.float32))
            tail_list.append(torch.tensor(1.0, dtype=torch.float32))
            continue

        entries = []
        for k, v in d.items():
            # --- token id ---
            if isinstance(k, int):
                tid = k
            # else:
            #     # token string key; convert to id
            #     # Some Qwen token strings may map to multiple-merge pieces; accept direct conversion.
            #     tid = tokenizer.convert_tokens_to_ids(k)
            #     if tid is None:
            #         # fallback: try encoding then taking the *first* produced id if it’s a single-piece token
            #         enc = tokenizer.encode(k, add_special_tokens=False)
            #         tid = enc[0] if len(enc) == 1 else tokenizer.unk_token_id
            # Skip invalid ids
            if tid is None or tid < 0:
                continue

            # --- logprob & rank ---
            lp = None
            rnk = None
            if isinstance(v, (float, int)):
                lp = float(v)
            elif isinstance(v, dict):
                lp = float(v.get("logprob", float("-inf")))
                rnk = v.get("rank", None)
            else:
                # object with attributes (e.g., v.logprob, v.rank, v.decoded_token)
                lp = float(getattr(v, "logprob", float("-inf")))
                rnk = getattr(v, "rank", None)

            entries.append((rnk, lp, tid))

        if len(entries) == 0:
            top_ids_list.append(torch.full((TOPK_LOGPROB,), -1, dtype=torch.int32))
            top_logp_list.append(torch.full((TOPK_LOGPROB,), float("-inf"), dtype=torch.float32))
            tail_list.append(torch.tensor(1.0, dtype=torch.float32))
            continue

        # Prefer sorting by rank if available; otherwise by logprob descending
        if any(e[0] is not None for e in entries):
            entries = [e for e in entries if e[0] is not None]
            entries.sort(key=lambda x: x[0])  # ascending rank
        else:
            entries.sort(key=lambda x: x[1], reverse=True)  # by logprob

        # Truncate to topk
        entries = entries[:TOPK_LOGPROB]
        ids = torch.tensor([e[2] for e in entries], dtype=torch.int32)
        lps = torch.tensor([e[1] for e in entries], dtype=torch.float32)

        mass = torch.exp(lps).sum().clamp_(0.0, 1.0)
        tail = 1.0 - float(mass)

        # Pad if fewer than topk
        if ids.numel() < TOPK_LOGPROB:
            pad_n = TOPK_LOGPROB - ids.numel()
            ids = torch.cat([ids, torch.full((pad_n,), -1, dtype=torch.int32)], dim=0)
            lps = torch.cat([lps, torch.full((pad_n,), float("-inf"), dtype=torch.float32)], dim=0)

        top_ids_list.append(ids)
        top_logp_list.append(lps)
        tail_list.append(torch.tensor(tail, dtype=torch.float32))

    top_ids = torch.stack(top_ids_list, dim=0)  # [A,K]
    top_lps = torch.stack(top_logp_list, dim=0)  # [A,K]
    tail    = torch.stack(tail_list, dim=0)     # [A]
    return top_ids, top_lps, tail

def union_topk_mixture(
    per_ev_ids: List[torch.Tensor],        # K_elems of [A,K]
    per_ev_logp: List[torch.Tensor],       # K_elems of [A,K]
    weights: Optional[List[float]] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute mixture p_mix = sum_i w_i p_i over evidences, but only on the
    union of per-evidence top-K tokens at each position. Return:
      mix_ids:   [A, K_union]   (ragged union is padded/truncated to TOPK_LOGPROB)
      mix_logp:  [A, K_union]
      mix_tail:  [A]            (1 - sum_{tokens in union} p_mix(token))
    Implementation detail:
      - Build union per position (set of token_ids across evidences),
        truncate/sort by aggregate probability, and convert to log-probs.
    """
    K = len(per_ev_ids)
    assert K == len(per_ev_logp)
    if weights is None:
        weights = [1.0 / K] * K
    w = torch.tensor(weights, dtype=torch.float32).softmax(0)

    A, Kk = per_ev_ids[0].shape
    # Build union per position
    mix_ids_list, mix_logp_list, mix_tail_list = [], [], []
    for a in range(A):
        token_to_prob = {}
        # accumulate weighted probs across evidences
        for i in range(K):
            ids_row = per_ev_ids[i][a]        # [K]
            lps_row = per_ev_logp[i][a]       # [K]
            probs   = torch.exp(lps_row)      # [K]
            for tid, p in zip(ids_row.tolist(), probs.tolist()):
                if tid < 0:  # padding
                    continue
                token_to_prob[tid] = token_to_prob.get(tid, 0.0) + float(w[i]) * float(p)

        # sort tokens by aggregate prob
        if len(token_to_prob) == 0:
            # degenerate
            mix_ids_list.append(torch.full((TOPK_LOGPROB,), -1, dtype=torch.int32))
            mix_logp_list.append(torch.full((TOPK_LOGPROB,), float("-inf")))
            mix_tail_list.append(torch.tensor(1.0, dtype=torch.float32))
            continue

        sorted_items = sorted(token_to_prob.items(), key=lambda kv: kv[1], reverse=True)
        # truncate to TOPK_LOGPROB
        sorted_items = sorted_items[:TOPK_LOGPROB]
        ids = torch.tensor([tid for tid, _ in sorted_items], dtype=torch.int32)
        probs = torch.tensor([p for _, p in sorted_items], dtype=torch.float32)
        tail = max(0.0, 1.0 - float(probs.sum().clamp_(0.0, 1.0)))

        # convert to log-probs (avoid log(0))
        probs = probs.clamp_min(1e-12)
        lps = torch.log(probs)

        mix_ids_list.append(ids)
        mix_logp_list.append(lps)
        mix_tail_list.append(torch.tensor(tail, dtype=torch.float32))

    mix_ids  = torch.stack(mix_ids_list, dim=0)      # [A,K_u]
    mix_logp = torch.stack(mix_logp_list, dim=0)     # [A,K_u]
    mix_tail = torch.stack(mix_tail_list, dim=0)     # [A]
    return mix_ids, mix_logp, mix_tail

def collate_image(image_paths):
    return [load_image(image) for image in image_paths]

# =========================
# Cache saving
# =========================
def save_payload(save_dir: Path, qid: str, payload: Dict[str, Any]):
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, save_dir / f"{qid}.pt")

# =========================
# Main caching procedure
# =========================

def build_cache(dataset_iter, out_dir: str, prebuild_index:list[dict]):
    save_dir = Path(out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    llm = build_llm()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    uuid_cache = ImageUUIDLRU(capacity=10000)
    global VOCAB_SIZE 
    VOCAB_SIZE = len(tokenizer)
    meta = {
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "topk_prompt": TOPK_LOGPROB,
        "store_per_evidence": STORE_PER_EVIDENCE,
        "dtype": DTYPE,
        "tp_size": TP_SIZE,
    }
    (save_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    sample_buffer = []
    output_buffer = []
    payload_buffer = []
    buffer_step = 0
    for step, (sample, ret_obj) in tqdm(enumerate(zip(dataset_iter, prebuild_index)), total=len(dataset_iter)):
        assert sample['question_id'] == ret_obj['question_id']
        qid = sample['question_id']
        question_text = sample['question']
        question_image = sample['image']
        answer = ast.literal_eval(sample['answer'])
        answer = max(answer, key=len)
        answer_eos = answer + tokenizer.eos_token
        answer_ids = tokenizer.encode(answer_eos)

        A = len(answer_ids)


        # # NO-EVIDENCE request
        # prompt_noev = build_chat_prompt_qwen_vl(tokenizer, question_text, num_images=1)
        # request_noev = {
        #     "prompt": prompt_noev + answer_prefill_text,
        #     "multi_modal_data": {"image": [load_image(sample.image_path)]},
        # }

        # WITH-EVIDENCE requests (batched)
        evid_paths = ret_obj['ret_images']
        K = len(evid_paths)
        evid_ids = ret_obj['ret_images']
        weights = ret_obj['scores'] or [1.0 / K] * K

        # # ---- no-evidence run ----
        # outs_noev = llm.generate(
        #     prompts=[request_noev["prompt"]],
        #     sampling_params=sampling,
        #     use_tqdm=False,
        #     multi_modal_data=[request_noev["multi_modal_data"]],
        # )
        # noev_ids, noev_lps, noev_tail = extract_prompt_topk_for_answer(outs_noev, A, tokenizer)

        # ---- with-evidence runs (batched over evidences) ----
        evid_paths = MinimumDS(evid_paths)
        temp_dl = DataLoader(evid_paths, batch_size=30, num_workers=6, collate_fn=lambda batch: batch)
        for batch in temp_dl:
            chunk_prompts = []

            chunk_prompts = process_input(question_text, answer, question_image, batch, tokenizer, uuid_cache, mode='replace')

        if buffer_step < MACRO_BATCH_SIZE:
            payload_buffer.append([qid, A])
            sample_buffer.extend(chunk_prompts)
            buffer_step += 1
            continue

        max_ans_len = max([e[1] for e in payload_buffer])
        sampling = build_sampling_params(max_new_tokens=max_ans_len+1, topk_prompt=TOPK_LOGPROB)
        outs_chunk = llm.generate(
            prompts=sample_buffer,
            sampling_params=sampling,
            use_tqdm=False,
        )
        temp =  list(itertools.chain.from_iterable([e.outputs[0].token_ids for e in outs_chunk]))
        if any(x < 0 for x in temp):
            raise AttributeError("Generated token id out of bound")

        #output_buffer.extend(outs_chunk)
        # Parse requests in the chunk

        #for i in range(len(chunk_prompts)):
        for ii, (qq, aa) in enumerate(payload_buffer):
            # Build batched inputs (in chunks to fit memory)
            per_ev_top_ids: List[torch.Tensor]  = []
            per_ev_top_lps: List[torch.Tensor]  = []
            per_ev_tail:    List[torch.Tensor]  = []

            # Pack logprob per sample, iterate over top-k
            for i in range(K):
                out_id = ii*K + i
                ids_e, lps_e, tail_e = extract_prompt_topk_for_answer(outs_chunk[out_id], aa, tokenizer)
                per_ev_top_ids.append(ids_e)    # [A,K]
                per_ev_top_lps.append(lps_e)    # [A,K]
                per_ev_tail.append(tail_e)      # [A]

            # ---- mixture over evidences (prob-space average) ----
            if USE_UNION_TOPK_FOR_MIX:
                mix_ids, mix_logp, mix_tail = union_topk_mixture(per_ev_top_ids, per_ev_top_lps, weights=weights)
            else:
                # Fallback: simple average of tails and keep per-evidence results only
                mix_ids, mix_logp, mix_tail = noev_ids.clone(), torch.full_like(noev_lps, float("-inf")), noev_tail.clone()

            # ---- save payload ----

            payload = {
                "qid": qq,
                "answer_len": aa,
                "retrieved_ids": evid_ids,
                # no-evidence
                # "noev_top_idx":  noev_ids.to(torch.int32),
                # "noev_top_logp": noev_lps.to(torch.float16),
                # "noev_tail":     noev_tail.to(torch.float32),
                # with-evidence mixture
                "mix_idx":  mix_ids.cpu().to(torch.int32),
                "mix_logp": mix_logp.cpu().to(torch.float16),
                "tail_logp":     mix_tail.cpu().to(torch.float16),
                # optional per-evidence packs
                #"per_ev_stored": STORE_PER_EVIDENCE,
            }
            if STORE_PER_EVIDENCE:
                payload["per_evidence"] = [
                    {
                        "ev_id": evid_ids[i],
                        "top_idx":  per_ev_top_ids[i].to(torch.int32),
                        "top_logp": per_ev_top_lps[i].to(torch.float16),
                        "tail":     per_ev_tail[i].to(torch.float32),
                    }
                    for i in range(K)
                ]
            save_payload(Path(SAVE_DIR), qq, payload)
        if step % 30 == 0:
            torch.cuda.empty_cache()
        
        payload_buffer.clear()
        output_buffer.clear()
        sample_buffer.clear()
        buffer_step = 0

    print(f"[vLLM] finished. cache at: {Path(SAVE_DIR).resolve()}")


if __name__ == "__main__":
    # TODO: replace with your dataset loader
    # Example:
    # def dataset_iter():
    #     for row in your_rows:
    #         yield Sample(
    #             qid=row["qid"],
    #             prompt_input_ids=row["prompt_ids"],
    #             prompt_attention_mask=row["prompt_mask"],
    #             answer_input_ids=row["answer_ids"],  # includes <eoa>
    #             image_path=row["image_path"],
    #             retrieved_paths=row["retrieved_paths"],
    #             retrieved_ids=row.get("retrieved_ids"),
    #             evidence_weights=row.get("evidence_weights"),
    #         )
    # build_cache(dataset_iter(), SAVE_DIR)

    
    train_ds = datasets.load_from_disk('/wy_data/SKVQA/IRCAP/train')
    test_ds = datasets.load_from_disk('/wy_data/SKVQA/IRCAP/test')

    # train_ds = train_ds.cast_column("image", HFImage(decode=True))
    # test_ds = test_ds.cast_column("image", HFImage(decode=True))
    with open('/wy_data/SKVQA/IRCAP/query_ret_imgs.jsonl') as f:
        prebuild_index = [json.loads(e) for e in f.readlines()]
    train_ds = train_ds.select(range(50000,len(prebuild_index)))
    prebuild_index = prebuild_index[50000:]
    build_cache(dataset_iter=train_ds, out_dir=SAVE_DIR, prebuild_index=prebuild_index)