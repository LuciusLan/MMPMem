import os, json, math, argparse
os.environ['CUDA_VISIBLE_DEVICES']="2"
os.environ["HF_HOME"]='/data_external/hf_cache'
import re
import random
from dataclasses import dataclass
from typing import Optional, Any, Sequence

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import datasets
from datasets import Image as HFImage, Dataset as HFDataset
from vllm import LLM, SamplingParams
from vllm.inputs.data import TextPrompt
from tqdm import tqdm
MACRO_BATCH_SIZE = 24
STOPWORDS = {
    "the", "a", "an", "who", "which", "that", "and", "or", "of",
    "in", "on", "at", "for", "is", "are", "was", "were", "therefore",
    "so", "then", "as", "by", "to", "from", "they", "it", "this",
}

CURRENCY_CODES: set[str] = {"usd", "eur", "gbp", "jpy", "cny", "cad", "aud", "sgd", "hkd", "inr"}
MONTHS: set[str] = {
    "jan","january","feb","february","mar","march","apr","april","may","jun","june",
    "jul","july","aug","august","sep","sept","september","oct","october","nov","november","dec","december"
}


_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

_DATE_PATTERNS = [
    # 1) ISO-like: 2020-01-31
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),
    # 2) Slash: 01/31/2020 or 1/31/20
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    # 3) Month Day, Year: January 1, 2020 (comma optional; ordinal optional)
    re.compile(rf"\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*|\s+)\d{{4}}\b", re.IGNORECASE),
    # 4) Month Year: January 2020
    re.compile(rf"\b{_MONTH}\s+\d{{4}}\b", re.IGNORECASE),
]



@dataclass
class FactualTokenResult:
    # Raw CoT info
    cot: str
    cot_token_ids: list[int]
    cot_tokens: list[str]

    # Per-token scores and labels
    scores: list[float]                # KB-sensitivity score per CoT token
    factual_token_indices: list[int]   # positions (0-based) inside CoT
    # Optional: you can add other debug fields if needed

@dataclass
class Span:
    start: int
    end: int
    text: str

def _build_messages(question: str, kb_doc: Optional[str], cot: str) -> list[dict[str, str]]:
    """Builds chat messages for Qwen-style instruction models."""
    inst_template = "Given the factual knowledge required question, try your best to solve it step by step. If you are not sure about some factual information, you may use placeholder tokens \"<fact 1>\", \"<fact 2>\", etc. and \"<conclusion 1>\", \"<conclusion 2>\", etc. Do NOT fabricate fact. If you chose to output placeholder tokens, do NOT state why (because you don't know, you are not sure, etc.), just complete the step by step template as a normal, fluent sentence, with replacing the factual information to the placeholder tokens. Follow the output syntax as the examples:\n\n(user) Question: Where was the place of death of the director of film Ladies Courageous?\n(model answer)The director of film Ladies Courageous is John Rawlins.\nJohn Rawlins died on May 20 1997, in Arcadia, California.\n\nAnswer:\\boxed{Arcadia, California}\n\n(user)Question: Which film has the director who died earlier, Budak Nafsu or The Bilingual Lover?\n\n(model answer)Director of Budak Nafsu is <fact 1>.\n<fact 1> died on <fact 2>.\nDirector of The Bilingual Lover is Vicente Aranda. Vicente Aranda died on 26 May 2015. Comparing <fact 2> and May 2015, the answer is <conclusion 1>."
    if kb_doc and kb_doc.strip():
        user_content = f"{inst_template}\nQuestion: {question}\n\nFollowing Wikipedia passages might contain useful information. Do NOT cite the passage when output your step by step reasoning.\n{kb_doc}"
    else:
        user_content = f"{inst_template}\nQuestion: {question}"

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": cot},
    ]


def _apply_chat_template(
    tokenizer: AutoTokenizer,
    question: str,
    kb_doc: Optional[str],
    cot: str,
    chat_template_kwargs: Optional[dict[str, Any]] = None,
) -> str:
    """Convert (question, KB, CoT) into a single prompt string via HF chat template.

    For Qwen3 this uses tokenizer.apply_chat_template. :contentReference[oaicite:2]{index=2}
    """
    messages = _build_messages(question, kb_doc, cot)
    kwargs = dict(
        tokenize=False,
        add_generation_prompt=False,  # CoT is part of the prompt, not to be generated
    )
    if chat_template_kwargs:
        kwargs.update(chat_template_kwargs)

    text = tokenizer.apply_chat_template(messages, **kwargs)
    return text


def _run_vllm_with_prompt_logprobs(
    llm: LLM,
    prompts: list[str],
    prompt_logprobs_k: int = 1,
) -> list[dict[str, Any]]:
    """Run vLLM and return prompt token IDs and logprobs for each prompt.

    Uses prompt_logprobs in SamplingParams. :contentReference[oaicite:3]{index=3}
    """
    sampling_params = SamplingParams(
        temperature=0.0,       # deterministic; we only care about scores
        top_p=1.0,
        top_k=-1,
        max_tokens=1,          # we do not really need generation; just prefill
        prompt_logprobs=prompt_logprobs_k,
        detokenize=False,
    )

    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=False)

    results = []
    for out in outputs:
        prompt_ids = list(out.prompt_token_ids)
        plp = out.prompt_logprobs  # list[None | dict[token_id, Logprob]] :contentReference[oaicite:4]{index=4}

        # Map each prompt token to its own logprob (or NaN if unavailable).
        token_logprobs: list[float] = []
        for tid, lp_dict in zip(prompt_ids, plp):
            if lp_dict is None:
                token_logprobs.append(float("nan"))
                continue

            # lp_dict[token_id] is a Logprob dataclass with .logprob field in recent vLLM. :contentReference[oaicite:5]{index=5}
            obj = lp_dict.get(tid, None)
            if obj is None:
                token_logprobs.append(float("nan"))
            else:
                # Be robust to different shapes across versions.
                lp = getattr(obj, "logprob", obj)
                token_logprobs.append(float(lp))

        results.append(
            {
                "prompt_token_ids": prompt_ids,
                "prompt_token_logprobs": token_logprobs,
            }
        )

    return results


def _find_subsequence_end_aligned(haystack: list[int], needle: list[int]) -> int:
    """Return the start index where `needle` appears in `haystack`, searching from the end.

    This biases toward the last occurrence, which is usually the last assistant (CoT) message.
    Returns -1 if not found.
    """
    if not needle:
        return -1
    for start in range(len(haystack) - len(needle), -1, -1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return -1


def _z_score_thresholding(scores: list[float], z_threshold: float) -> list[int]:
    """Return indices with z-score >= z_threshold, ignoring NaNs."""
    valid = [(i, s) for i, s in enumerate(scores) if not math.isnan(s)]
    if not valid:
        return []

    values = [s for _, s in valid]
    mean = sum(values) / len(values)
    var = sum((s - mean) ** 2 for s in values) / max(len(values) - 1, 1)
    std = math.sqrt(var) if var > 0 else 0.0

    factual_indices: list[int] = []
    for i, s in valid:
        if std == 0.0:
            z = 0.0
        else:
            z = (s - mean) / std
        if z >= z_threshold:
            factual_indices.append(i)
    return factual_indices

def extract_question_words(question: str,
                           stopwords: set[str],
                           domain_stopwords: Optional[set[str]] = None) -> set[str]:
    domain_stopwords = domain_stopwords or set()
    q_words = set()
    for w in re.findall(r"\w+", question):
        wn = normalize_word(w)
        if not wn:
            continue
        if wn.isdigit():
            continue
        if wn in stopwords or wn in domain_stopwords:
            continue
        q_words.add(wn)
    return q_words

def token_indices_overlapping_span(
    offsets: list[tuple[int, int]],
    span: tuple[int, int],
) -> list[int]:
    ss, se = span
    out = []
    for i, (a, b) in enumerate(offsets):
        if b <= ss or a >= se:
            continue
        out.append(i)
    return out

def _whitespace_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        j = i
        while j < n and not text[j].isspace():
            j += 1
        spans.append(Span(i, j, text[i:j]))
        i = j
    return spans

def _is_month(tok: str) -> bool:
    t = tok.strip().lower()
    t = re.sub(r"^[^\w]+|[^\w]+$", "", t)  # trim outer punctuation
    return t in MONTHS

def _looks_like_day(tok: str) -> bool:
    t = tok.strip()
    # allow "1", "1,", "01", "01,"
    t = re.sub(r"[^\d]", "", t)
    if not t:
        return False
    try:
        d = int(t)
        return 1 <= d <= 31
    except:
        return False

def _looks_like_year(tok: str) -> bool:
    t = re.sub(r"[^\d]", "", tok.strip())
    return len(t) == 4 and t.isdigit()

def _has_digit(tok: str) -> bool:
    return any(c.isdigit() for c in tok)

def _merge_spans(spans: list[Span], text: str) -> list[Span]:
    """Merge common multi-chunk factual patterns: 'Jan 1, 2020' and '980,000 USD'."""
    out: list[Span] = []
    i = 0
    while i < len(spans):
        # Pattern 1: Month Day Year  => "Jan 1, 2020"
        if i + 2 < len(spans) and _is_month(spans[i].text) and _looks_like_day(spans[i+1].text) and _looks_like_year(spans[i+2].text):
            start = spans[i].start
            end = spans[i+2].end
            out.append(Span(start, end, text[start:end]))
            i += 3
            continue

        # Pattern 2: number + currency code => "980,000 USD"
        if i + 1 < len(spans) and _has_digit(spans[i].text):
            nxt = re.sub(r"^[^\w]+|[^\w]+$", "", spans[i+1].text.strip().lower())
            if nxt in CURRENCY_CODES:
                start = spans[i].start
                end = spans[i+1].end
                out.append(Span(start, end, text[start:end]))
                i += 2
                continue

        out.append(spans[i])
        i += 1
    return out

def _normalize_alpha(s: str) -> str:
    # for stopwords and “question word” checks; keeps letters only
    return re.sub(r"[^a-z]+", "", s.lower())

def _span_variants(s: str) -> set[str]:
    """
    Variants for KB/question matching, designed to catch:
      - 8,848 vs 8848
      - $123,456 vs 123456
      - 2020-01-01 preserved
      - "Jan 1, 2020" normalized
    """
    raw = s.strip()
    lower = raw.lower()
    # trim outer punctuation but keep internal punctuation
    trimmed = re.sub(r"^[^\w$]+|[^\w]+$", "", lower)

    # numeric canonical: keep digits only
    digits_only = re.sub(r"[^\d]", "", trimmed)

    # alnum canonical: letters+digits only
    alnum_only = re.sub(r"[^a-z0-9]", "", trimmed)

    # keep dash dates: remove spaces and commas but keep dashes
    dash_preserve = re.sub(r"[,\s]+", "", trimmed)

    out = {trimmed}
    if digits_only:
        out.add(digits_only)
    if alnum_only:
        out.add(alnum_only)
    if dash_preserve:
        out.add(dash_preserve)

    # also add a version with common currency symbols stripped
    no_currency = re.sub(r"[$€£¥]", "", trimmed)
    out.add(no_currency)
    no_currency_digits = re.sub(r"[^\d]", "", no_currency)
    if no_currency_digits:
        out.add(no_currency_digits)

    # remove empty
    return {v for v in out if v}

def _build_match_set(doc: str, max_chars: Optional[int] = None) -> set[str]:
    """Build a set of normalized variants for all spans in a doc (optionally truncated)."""
    if max_chars is not None:
        doc = doc[:max_chars]
    spans = _merge_spans(_whitespace_spans(doc), doc)
    m: set[str] = set()
    for sp in spans:
        m |= _span_variants(sp.text)
    return m

def _map_tokens_to_spans(token_offsets: list[tuple[int,int]], spans: list[Span]) -> list[Optional[int]]:
    """
    Map each token (by char offsets) to a span index by containment.
    Assumes spans are non-overlapping and sorted.
    """
    res: list[Optional[int]] = [None] * len(token_offsets)
    si = 0
    for ti, (ts, te) in enumerate(token_offsets):
        if ts is None or te is None:
            continue
        while si < len(spans) and spans[si].end <= ts:
            si += 1
        if si < len(spans) and spans[si].start <= ts and te <= spans[si].end:
            res[ti] = si
        else:
            # fallback: try intersection (rare with odd offset behavior)
            for sj in range(max(si-1, 0), min(si+2, len(spans))):
                if not (te <= spans[sj].start or ts >= spans[sj].end):
                    res[ti] = sj
                    break
    return res

def select_factual_tokens_by_spans_from_top_tokens(
    cot_text: str,
    cot_token_ids: list[int],
    token_scores: list[float],
    question: str,
    gt_kb_doc: str,
    tokenizer,
    max_spans: int = 10,
    restrict_to_kb: bool = True,
    exclude_question_spans: bool = True,
    skip_tokens_below_best_question_span: bool = True,
    question_span_margin: float = 0.0,
    stopwords: set[str] = STOPWORDS,
    kb_max_chars: Optional[int] = None,
) -> tuple[list[int], list[tuple[str, list[int]]]]:
    """
    Rank tokens by token_scores, lift to whitespace/merged spans, deduplicate spans,
    filter spans by stopwords + KB overlap + question exclusion, and optionally
    stop when score < best_question_span_score + margin.
    """

    # 1) Build spans for CoT and question with the same logic
    cot_spans = _merge_spans(_whitespace_spans(cot_text), cot_text)
    q_spans = _merge_spans(_whitespace_spans(question), question)

    # 2) Build match sets for KB and question
    kb_match = _build_match_set(gt_kb_doc, max_chars=kb_max_chars) if restrict_to_kb else set()
    q_match = _build_match_set(question)  # small anyway

    # 3) Encode CoT to get token offsets (no word_ids)
    enc = tokenizer(
        cot_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    if len(input_ids) != len(cot_token_ids) or any(a != b for a, b in zip(input_ids, cot_token_ids)):
        raise ValueError(
            "Tokenization mismatch: ensure cot_token_ids were produced by "
            "tokenizer(cot_text, add_special_tokens=False)."
        )

    # 4) Map tokens -> span indices
    tok2span = _map_tokens_to_spans(offsets, cot_spans)

    # 5) Compute cutoff = best score among tokens that lie inside a question-span match
    best_qspan_score = float("-inf")
    if skip_tokens_below_best_question_span:
        for ti, s in enumerate(token_scores):
            if math.isnan(s):
                continue
            if ti<len(token_scores)//4:
                continue
            sp_i = tok2span[ti]
            if sp_i is None:
                continue
            sp_text = cot_spans[sp_i].text
            # if this span’s variants overlap the question match set, treat as question-given
            if _span_variants(sp_text) & q_match:
                if s > best_qspan_score:
                    best_qspan_score = s

    cutoff = best_qspan_score + question_span_margin if best_qspan_score != float("-inf") else float("-inf")
    #cutoff*=0.2

    # 6) Prepare span -> token indices map (for expansion)
    span_to_tokens: dict[int, list[int]] = {}
    for ti, sp_i in enumerate(tok2span):
        if sp_i is None:
            continue
        span_to_tokens.setdefault(sp_i, []).append(ti)

    # 7) Rank tokens and select spans
    scored_tokens = [(i, s) for i, s in enumerate(token_scores) if not math.isnan(s)]
    scored_tokens.sort(key=lambda x: x[1], reverse=True)

    selected_spans: list[int] = []
    selected_set: set[int] = set()

    for ti, s in scored_tokens:
        if skip_tokens_below_best_question_span and s < cutoff:
            break
        if len(selected_spans) >= max_spans:
            break

        sp_i = tok2span[ti]
        if sp_i is None or sp_i in selected_set:
            continue

        sp_text = cot_spans[sp_i].text.strip()
        if not sp_text:
            continue

        # stopword filter: only apply to “pure alphabetic” spans
        alpha_norm = _normalize_alpha(sp_text)
        if alpha_norm and alpha_norm in stopwords:
            continue

        # restrict_to_kb: require any variant overlap
        if restrict_to_kb:
            if not (_span_variants(sp_text) & kb_match):
                continue

        # exclude question spans: require no overlap with question match set
        if exclude_question_spans:
            if _span_variants(sp_text) & q_match:
                continue

        selected_spans.append(sp_i)
        selected_set.add(sp_i)

    # 8) Expand spans back to token indices
    factual_token_indices = sorted(
        ti for sp_i in selected_spans for ti in span_to_tokens.get(sp_i, [])
    )
    factual_spans = [(cot_spans[sp_i].text, span_to_tokens.get(sp_i, [])) for sp_i in selected_spans]
    return factual_token_indices, factual_spans


def normalize_word(w: str) -> str:
    return re.sub(r"\W+", "", w).lower()

def identify_factual_tokens_vllm(
    llm: LLM,
    tokenizer: AutoTokenizer,
    batch: Sequence[dict[str, str]],
    z_threshold: float = 1.0,
    chat_template_kwargs: Optional[dict[str, Any]] = None,
) -> list[FactualTokenResult]:
    """Identify 'factual' CoT tokens using logprob deltas across KB conditions.

    Args
    ----
    llm:
        vLLM LLM instance already loaded with Qwen3-8B-Instruct (or similar). :contentReference[oaicite:6]{index=6}
    tokenizer:
        HuggingFace tokenizer corresponding to the same model.
    batch:
        Iterable of dicts with keys:
            - "question": str
            - "cot": str
            - "gt_kb_doc": str (ground-truth KB doc)
            - "irrel_kb_doc": str (irrelevant doc)
            - "corrupt_kb_doc": str (corrupted doc)
    z_threshold:
        Z-score threshold on the KB-sensitivity score. Larger => fewer tokens marked factual.
    chat_template_kwargs:
        Extra kwargs passed to tokenizer.apply_chat_template, e.g.
        {"enable_thinking": False} for Qwen3 thinking models. :contentReference[oaicite:7]{index=7}

    Returns
    -------
    list[FactualTokenResult]
        One result per example in `batch`. `factual_token_indices` are positions in the
        tokenized CoT (0-based).
    """
    # 1) Pre-tokenize CoTs once.
    cot_ids_batch: list[list[int]] = []
    cot_tokens_batch: list[list[str]] = []
    for example in batch:
        cot_text = example["cot"]
        # No special tokens: we want raw CoT tokenization.
        cot_ids = tokenizer.encode(cot_text, add_special_tokens=False)
        cot_tokens = tokenizer.convert_ids_to_tokens(cot_ids)
        cot_ids_batch.append(cot_ids)
        cot_tokens_batch.append(cot_tokens)

    # 2) Build prompts for four conditions.
    prompts_base: list[str] = []
    prompts_gt: list[str] = []
    prompts_irrel: list[str] = []
    prompts_corr: list[str] = []

    for example in batch:
        q = example["question"]
        cot = example["cot"]
        gt = example.get("gt_kb_doc", "")
        irrel = example.get("irrel_kb_doc", "")

        prompts_base.append(
            _apply_chat_template(tokenizer, question=q, kb_doc=None, cot=cot,
                                 chat_template_kwargs=chat_template_kwargs)
        )
        prompts_gt.append(
            _apply_chat_template(tokenizer, question=q, kb_doc=gt, cot=cot,
                                 chat_template_kwargs=chat_template_kwargs)
        )
        prompts_irrel.append(
            _apply_chat_template(tokenizer, question=q, kb_doc=irrel, cot=cot,
                                 chat_template_kwargs=chat_template_kwargs)
        )

    # 3) Score all prompts with prompt_logprobs.
    base_res = _run_vllm_with_prompt_logprobs(llm, prompts_base)
    gt_res = _run_vllm_with_prompt_logprobs(llm, prompts_gt)
    irrel_res = _run_vllm_with_prompt_logprobs(llm, prompts_irrel)

    results: list[FactualTokenResult] = []

    # 4) For each example, align CoT segment and compute KB-sensitivity scores.
    for idx, example in enumerate(batch):
        cot_text = example["cot"]
        cot_ids = cot_ids_batch[idx]
        cot_tokens = cot_tokens_batch[idx]

        def extract_segment(res: dict[str, Any]) -> list[float]:
            prompt_ids = res["prompt_token_ids"]
            prompt_lps = res["prompt_token_logprobs"]

            start = _find_subsequence_end_aligned(prompt_ids, cot_ids)
            if start < 0:
                raise ValueError(
                    f"Could not locate CoT tokens inside prompt for example {idx}. "
                    "Check that tokenizer and chat template are consistent."
                )
            end = start + len(cot_ids)
            return prompt_lps[start:end]

        base_lp_cot = extract_segment(base_res[idx])
        gt_lp_cot = extract_segment(gt_res[idx])
        irrel_lp_cot = extract_segment(irrel_res[idx])

        # 5) Compute per-token KB-sensitivity score.
        scores: list[float] = []
        for b, g, r in zip(base_lp_cot, gt_lp_cot, irrel_lp_cot):
            pieces: list[float] = []
            if not math.isnan(g):
                if not math.isnan(b):
                    pieces.append(g - b)
                if not math.isnan(r):
                    pieces.append(g - r)
            if not pieces:
                scores.append(float("nan"))
            else:
                # Average of available contrasts; positive => KB raises the probability.
                scores.append(sum(pieces) / len(pieces))

        # 6) Threshold scores within this CoT.
        factual_indices = _z_score_thresholding(scores, z_threshold=z_threshold)
        score_fact_idc = [scores[fi] for fi in factual_indices]
        results.append(
            FactualTokenResult(
                cot=cot_text,
                cot_token_ids=cot_ids,
                cot_tokens=cot_tokens,
                scores=scores,
                factual_token_indices=factual_indices,
            )
        )

    return results

def main():
    llm = LLM(
        model='/data_external/Qwen3-4B-Instruct-2507',
        dtype='bfloat16',
        tensor_parallel_size=1,
        max_num_batched_tokens=80000,
        #max_num_seqs=150,
        max_model_len=16384,
        max_logprobs=32,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        #limit_mm_per_prompt={"image": 2, "video": 0},
    )
    tokenizer = AutoTokenizer.from_pretrained('/data_external/Qwen3-4B-Instruct-2507')
    sp = SamplingParams(
            max_tokens=8000,
            temperature=0.6,
            top_k=20,
            top_p=0.95,
            prompt_logprobs=1
    )


    train_ds = datasets.load_dataset('framolfese/2WikiMultihopQA', split='train')
    with open('/data_external/MMPMem/2wiki_cot.jsonl') as f:
        cot_ds = [json.loads(e) for e in f.readlines()]

    template = "Given the factual knowledge required question, try your best to solve it step by step."
    train_ds = train_ds.select(range(len(cot_ds)))

    batch_buffer = []
    batch_step = 0

    outfile = open('/data_external/MMPMem/2wiki_cot_facttok_label.jsonl', 'w')
    step = 0
    for row in tqdm(train_ds, total=len(train_ds)):
        cot = cot_ds[step]
        if row['question'] == cot['question']:
            step += 1
        else:
            continue
        if re.search(r'(?:(?:evidence)|(?:passage))\s?\d', cot['cot'], re.IGNORECASE):
            continue
        evidence_doc = ""
        irrel_doc = ""
        evids= []
        irrel = []
        for i, title in enumerate(row['supporting_facts']['title']):
            ev_idx = row['context']['title'].index(title)
            evids.append(ev_idx)
            evidence = row['context']['sentences'][ev_idx]
            evidence = ' '.join(evidence)
            evidence_doc += f'\nEvidence {i+1}: ' + evidence + '\n\n'
        for i, sent in enumerate(row['context']['sentences']):
            if i not in evids:
                irrel.append(' '.join(sent))
        
        irrel = random.sample(irrel, len(evids))
        for i, doc in enumerate(irrel):
            irrel_doc += f'\nEvidence {i+1}: ' + doc + '\n\n'
        
        if batch_step < 16:
            batch_buffer.append({"question": row['question'], "cot": cot['cot'], "gt_kb_doc":evidence_doc, "irrel_kb_doc":irrel_doc})
            batch_step += 1
        else:
            results = identify_factual_tokens_vllm(
                llm=llm,
                tokenizer=tokenizer,
                batch=batch_buffer,
                z_threshold=0,
            )
            for ex, res in zip(batch_buffer, results):
                factual_token_indices, factual_words = select_factual_tokens_by_spans_from_top_tokens(
                    cot_text=res.cot,
                    cot_token_ids=res.cot_token_ids,
                    token_scores=res.scores,
                    question=ex["question"],
                    gt_kb_doc=ex["gt_kb_doc"],
                    tokenizer=tokenizer,
                    max_spans=12,
                    restrict_to_kb=True,
                    exclude_question_spans=True,
                    skip_tokens_below_best_question_span=False,
                    question_span_margin=0.0,
                )
                res.factual_token_indices = factual_token_indices
                temp = {'question': ex['question'], 'cot':ex['cot'], 'cot_token_ids':res.cot_token_ids, 'factual_token_indices': res.factual_token_indices, 'cot_token_scores': res.scores}
                outfile.write(json.dumps(temp))
                outfile.write('\n')
            batch_buffer.clear()
            batch_step = 0


if __name__ == "__main__":
    main()