import argparse
import time
import os
import re
import json
import hashlib
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Sequence, Iterator, Union

os.environ["OMP_NUM_THREADS"] = "32"
os.environ["HF_HOME"]='/data_external/hf_cache'
os.environ['CUDA_VISIBLE_DEVICES']="1" #args.device
#os.environ['VLLM_FLASH_ATTN_VERSION']="3"
os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"]="spawn"

# For debugger
# os.environ["TORCHINDUCTOR_COMPILE_THREADS"]="1"
# os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"


from pathlib import Path
from typing import Dict, Any
from vllm import LLM, EngineArgs
import numpy as np
from datasets import load_from_disk, Dataset as HFDataset,  Features, Value, Sequence as HFSequence
from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler, get_worker_info
#from vllm.entrypoints.pooling.score.protocol import ScoreMultiModalParam
from tqdm import tqdm

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

MACRO_BATCH_SIZE = 50000

# ----------------------------- Data structures ----------------------------- #

@dataclass
class Chunk:
    article_id: Optional[str]
    article_title: str
    section_path: str
    section_index: int
    chunk_index: int
    text: str
    token_count: int
    start_sentence_idx: int
    end_sentence_idx: int  # exclusive


# ----------------------------- Tokenization utils -------------------------- #

def ensure_nltk() -> None:
    """
    Ensure required NLTK models are available.
    """
    # punkt is used by sent_tokenize; punkt_tab may be required by newer NLTK versions.
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(resource)
        except LookupError:
            try:
                nltk.download(resource.split("/")[-1], quiet=True)
            except Exception:
                # If punkt_tab fails (older NLTK), punkt is usually sufficient.
                pass


def count_tokens(text: str) -> int:
    # NLTK's word_tokenize counts punctuation as tokens; this is typically acceptable as a "generic" tokenizer.
    return len(word_tokenize(text))


def split_into_sentences(text: str) -> List[str]:
    # Normalise trivial whitespace without altering content.
    text = " ".join(text.split())
    if not text:
        return []
    return sent_tokenize(text)


def indices_to_drop_endmatter_sections(section_titles: Sequence[str]) -> List[int]:
    """
    Return 0-based indices of section titles that are likely Wikipedia end-matter
    (references/links/bibliography/navigation appendices) rather than substantive content.

    Heuristics (English-centric):
      - Drop canonical appendix headings and common variants:
        'References', 'Notes', 'Footnotes', 'Citations', 'Bibliography', 'Works cited',
        'Further reading', 'External links', 'See also', and combinations thereof.
      - Avoid false positives such as 'References in popular culture' or 'Notes on X'
        by requiring that 'references/notes' appear as standalone headings or in
        known end-matter combinations.
    """
    def norm(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"\s*\([^)]*\)\s*", " ", s)  # drop parenthetical disambiguators
        s = s.replace("&", " and ")
        s = re.sub(r"[^a-z0-9\s]", " ", s)      # remove punctuation
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # Exact matches: very likely non-expository end matter.
    exact_meta = {
        "references",
        "reference",
        "notes",
        "footnotes",
        "citations",
        "citation",
        "sources",
        "bibliography",
        "selected bibliography",
        "works cited",
        "works cited and references",
        "general references",
        "further reading",
        "external links",
        "external link",
        "see also",
        "related topics",          # older/variant heading sometimes used like "See also"
        "acknowledgements",
        "acknowledgments",
    }

    # Regexes for common end-matter combinations. Anchored to the whole heading to
    # reduce false positives like "references in popular culture".
    combo_patterns = [
        r"^(notes|footnotes)\s+and\s+(references|citations)$",
        r"^(references|citations)\s+and\s+(notes|footnotes)$",
        r"^(references|citations)\s+and\s+external\s+links$",
        r"^external\s+links\s+and\s+(references|citations)$",
        r"^(references|citations)\s+and\s+further\s+reading$",
        r"^further\s+reading\s+and\s+(references|citations)$",
        r"^(bibliography|selected\s+bibliography)\s+and\s+(references|citations)$",
        r"^(references|citations)\s+and\s+(bibliography|selected\s+bibliography)$",
        r"^(external\s+links)\s+and\s+(further\s+reading)$",
        r"^(further\s+reading)\s+and\s+(external\s+links)$",
        r"^(notes|footnotes)\s+(references|citations)$",  # e.g., "Notes references" after normalization
        r"^(references|citations)\s+(notes|footnotes)$",
    ]
    combo_re = re.compile("|".join(f"(?:{p})" for p in combo_patterns))

    # Substring triggers that are rarely expository when present in a heading.
    # (More permissive than exact matches, but still focused.)
    def has_strong_meta_substring(n: str) -> bool:
        if "external links" in n:
            return True
        if "further reading" in n:
            return True
        if n.startswith("see also"):
            return True
        if n == "works cited" or n.startswith("works cited "):
            return True
        return False

    drop = []
    for i, title in enumerate(section_titles):
        n = norm(title)
        if not n:
            continue

        if n in exact_meta:
            drop.append(i)
            continue

        if combo_re.match(n):
            drop.append(i)
            continue

        if has_strong_meta_substring(n):
            drop.append(i)
            continue

        # Conservative handling for headings beginning with "references"/"notes":
        # drop only if they are not followed by "in"/"on"/"for" etc.
        # This keeps e.g. "References in popular culture" and "Notes on terminology".
        if n.startswith("references"):
            if re.fullmatch(r"references(\s+and\s+\w+)*", n):
                drop.append(i)
                continue
        if n.startswith("notes") and n != "notes":
            # keep "notes on ..." etc.
            pass

    return drop


# ----------------------------- Chunking logic ------------------------------ #

def build_sentence_chunks(
    sentences: List[str],
    *,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
    overlap_sents: int,
) -> List[Tuple[int, int, str, int]]:
    """
    Build chunks as (start_idx, end_idx, chunk_text, chunk_token_count),
    where indices refer to sentence indices and end_idx is exclusive.

    Greedy strategy:
      - Accumulate sentences until reaching target_tokens, but do not exceed max_tokens.
      - If next sentence would exceed max_tokens and current >= min_tokens, finalize.
      - Handle very long single sentences by placing them alone in a chunk.

    Overlap:
      - Next chunk starts overlap_sents sentences before previous end.
      - Guaranteed to advance at least one sentence to avoid infinite loops.
    """
    if not sentences:
        return []

    sent_token_lens = [count_tokens(s) for s in sentences]
    n = len(sentences)

    chunks: List[Tuple[int, int, str, int]] = []
    start = 0

    while start < n:
        end = start
        tok_sum = 0

        # If a single sentence is longer than max_tokens, emit it alone.
        if sent_token_lens[start] > max_tokens:
            end = start + 1
            chunk_text = sentences[start]
            chunks.append((start, end, chunk_text, sent_token_lens[start]))
            # Overlap does not make sense here; just advance by one.
            start = end
            continue

        while end < n:
            next_len = sent_token_lens[end]
            if tok_sum + next_len > max_tokens:
                # Stop if we already have enough content; otherwise accept the stop and possibly merge later.
                break

            tok_sum += next_len
            end += 1

            # Prefer stopping once we've reached target and are above minimum.
            if tok_sum >= target_tokens and tok_sum >= min_tokens:
                break

        # Safety: ensure progress even if something odd happens.
        if end == start:
            end = min(start + 1, n)
            tok_sum = sent_token_lens[start]

        chunk_text = " ".join(sentences[start:end]).strip()
        chunks.append((start, end, chunk_text, tok_sum))

        # Compute next start with sentence-aware overlap.
        next_start = max(end - overlap_sents, start + 1)  # must advance by >= 1 sentence
        start = next_start

    return chunks


def merge_short_chunks(
    chunks: List[Tuple[int, int, str, int]],
    *,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
) -> List[Tuple[int, int, str, int]]:
    """
    Merge adjacent chunks (within the same section) using two triggers:

    1) Hard trigger: chunk token_count < min_tokens
    2) Soft trigger: chunk token_count < target_tokens AND chunk+next <= target_tokens

    Merge is allowed only if merged size <= max_tokens.
    """
    merged: List[Tuple[int, int, str, int]] = []
    i = 0
    while i < len(chunks):
        s0, e0, t0, n0 = chunks[i]

        if i + 1 < len(chunks):
            s1, e1, t1, n1 = chunks[i + 1]
            should_merge = (n0 < min_tokens) or (n0 < target_tokens and (n0 + n1) <= target_tokens)
            can_merge = (n0 + n1) <= max_tokens

            if should_merge and can_merge:
                merged_text = (t0 + " " + t1).strip()
                merged.append((s0, e1, merged_text, n0 + n1))
                i += 2
                continue

        merged.append((s0, e0, t0, n0))
        i += 1

    return merged


def format_section_path(section: Dict[str, Any]) -> str:
    """
    Convert a section dict into a stable "path" string.
    """
    path = section.get("path")
    if isinstance(path, list) and path:
        return " > ".join(str(x) for x in path)
    title = section.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return ""


def chunk_article(
    article: Dict[str, Any],
    *,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
    overlap_sents: int,
    prefix_headings_in_text: bool,
) -> List[Chunk]:
    """
    Chunk all sections of an article independently.
    """
    article_id = article.get("id")
    title = str(article.get("title", "")).strip()
    sections = article.get("sections", [])
    if not isinstance(sections, list):
        return []
    
    sec_to_drop = indices_to_drop_endmatter_sections([e['title'] for e in sections])

    out: List[Chunk] = []
    global_chunk_index = 0

    for sec_idx, section in enumerate(sections):
        if not isinstance(section, dict):
            continue
        if sec_idx in sec_to_drop:
            continue
        section_text = str(section.get("text", "")).strip()
        if not section_text:
            continue

        section_path = format_section_path(section)
        sentences = split_into_sentences(section_text)

        raw_chunks = build_sentence_chunks(
            sentences,
            target_tokens=target_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            overlap_sents=overlap_sents,
        )
        final_chunks = merge_short_chunks(
            raw_chunks,
            target_tokens=target_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )

        for local_chunk_idx, (s, e, chunk_text, tok_ct) in enumerate(final_chunks):
            if prefix_headings_in_text:
                header_lines = []
                if title:
                    header_lines.append(f"Article: {title}")
                if section_path:
                    header_lines.append(f"Section: {section_path}")
                if header_lines:
                    chunk_text = " ".join(header_lines) + "\n" + chunk_text

            out.append(
                Chunk(
                    article_id=str(article_id) if article_id is not None else None,
                    article_title=title,
                    section_path=section_path,
                    section_index=sec_idx,
                    chunk_index=global_chunk_index,
                    text=chunk_text,
                    token_count=tok_ct,
                    start_sentence_idx=s,
                    end_sentence_idx=e,
                )
            )
            global_chunk_index += 1

    return out

def _to_image_url(x: str) -> str:
    if x.startswith(("http://", "https://", "file://")):
        return x
    return "file://" + os.path.abspath(x)


def _stable_image_uuid(image_ref: str) -> str:
    # stable ID so repeated query image can reuse mm cache\
    global UUID_CACHE
    try:
        return UUID_CACHE[image_ref]
    except KeyError:
        UUID_CACHE[image_ref] = "img-" + hashlib.sha1(image_ref.encode("utf-8")).hexdigest()
        return UUID_CACHE[image_ref]

def format_query_to_score_param(q: Dict[str, Any]) -> Dict[str, Any]:
    """
    q example:
    {"text": "...", "image": "/path/to/q.png"}   # image optional
    """
    content = [{"type": "text", "text": q.get("text", "")}]
    if q.get("image"):
        img_url = _to_image_url(q["image"])
        content.append({
            "type": "image_url",
            "image_url": {"url": img_url},
            # Optional but recommended for cross-request/media caching
            "uuid": _stable_image_uuid(img_url),
        })
    return {"content": content}
    #return content

def format_doc_to_score_param(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    text-only doc in your setup
    d example: {"text": "..."}
    """
    return {"content": [{"type": "text", "text": d.get("text", "")}]}
    #return {"type": "text", "text": d.get("text", "")}

def reorganize_section(input_article):
    title = input_article['title']
    outputs = []
    for sec, text in zip(input_article['section_titles'], input_article['section_texts']):
        if len(text) < 10:
            continue
        if not sec:
            continue
        temp = f"{title}: {sec}: {text}"
        outputs.append({'section_text': temp, 'num_words':len(temp.split(' ')),  'title': title})
    return outputs

class TDataset(IterableDataset):
    def __init__(self, retrieved_collection, questions):
        super().__init__()
        self.retrieved_collection = retrieved_collection
        self.questions = questions
        self.drop_last = False
    
    def __len__(self):
        return len(self.questions)

    def __iter__(self):
        worker = get_worker_info()

        # Worker-local row partitioning (avoids duplicated rows across workers)
        # strided split: worker k takes k, k+W, k+2W, ...
        if worker is None:
            row_indices = range(len(self.retrieved_collection))
        else:
            row_indices = range(worker.id, len(self.retrieved_collection), worker.num_workers)

        query_batch_buffer = []
        doc_batch_buffer=[]
        output_buffer=[]
        buffer_step= 0
        for i in row_indices:
            ret_img_for_chunks, query_retdocs_map, ret_doc_chunks = self.process_row_fn(self.retrieved_collection[i])
            ques = self.questions[i]
            # Expand one row -> many sub-samples


            for rerank_qimg, (title, ret_img_idx), chunk_obj in zip(ret_img_for_chunks, query_retdocs_map.items(), ret_doc_chunks):
                output_buffer.append({'infoseek_qid': ques['data_id'], 'ret_img_wiki_title': title, 'ret_img_idx': ret_img_idx, 'rerank_buffer_idx': [], 'chunks': chunk_obj})
                for chunk in chunk_obj:
                    query_batch_buffer.append({'text': ques['question'], 'image': rerank_qimg})
                    doc_batch_buffer.append({'text': chunk.text,})
                    output_buffer[-1]['rerank_buffer_idx'].append(buffer_step)
                    buffer_step += 1

            if buffer_step>MACRO_BATCH_SIZE:
                flat_queries = []
                flat_docs = []
                for qq, dd in zip(query_batch_buffer, doc_batch_buffer):
                    flat_queries.append(format_query_to_score_param(qq))                  # repeated reference is fine
                    flat_docs.append(format_doc_to_score_param(dd))
                yield flat_queries, flat_docs, output_buffer
                query_batch_buffer.clear()
                doc_batch_buffer.clear()
                output_buffer.clear()
                buffer_step = 0

        if len(doc_batch_buffer)>0 and not self.drop_last:
            flat_queries = []
            flat_docs = []
            for qq, dd in zip(query_batch_buffer, doc_batch_buffer):
                flat_queries.append(format_query_to_score_param(qq))                  # repeated reference is fine
                flat_docs.append(format_doc_to_score_param(dd))
            yield flat_queries, flat_docs, output_buffer
    
    def process_row_fn(self, row):
        ret_images = row['ret_images']
        ret_docs = []
        ret_imgs_filtered = []
        for ret_img in ret_images:
            if len(ret_docs) == 30:
                break
            ovid = ret_img.split('/')[-1].split('.')[0]
            try:
                in_evqa = kb_title_map[oven_image_id_entity_map[ovid]]
                ret_docs.append(oven_image_id_entity_map[ovid])
                ret_imgs_filtered.append(ret_img)

            except KeyError:
                continue
        query_retdocs_map = defaultdict(list)
        for i, doc_tit in enumerate(ret_docs):
            query_retdocs_map[doc_tit].append(i)
            
        fulltext_ret_docs = [evqa_kb[kb_title_map[e]] for e in query_retdocs_map.keys()]

        section_dicts = [{'id': doc['url'], 'title': doc['title'], 'sections': [{'title': sec_tit,  'text': sec_txt} for sec_tit, sec_txt in zip(doc['section_titles'], doc['section_texts'])]} for doc in fulltext_ret_docs]
        ret_doc_chunks = [chunk_article(sec_dic, target_tokens=200, min_tokens=150, max_tokens=300, overlap_sents=2, prefix_headings_in_text=True) for sec_dic in section_dicts]
        ret_img_for_chunks = [ret_imgs_filtered[e[0]] for e in query_retdocs_map.values()]

        return ret_img_for_chunks, query_retdocs_map, ret_doc_chunks


def build_batch_plan(
    group_lengths: Sequence[int],
    max_batch_items: int,
    drop_last: bool = False,
    allow_oversize_group: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    group_lengths: lengths of contiguous groups, e.g. [5,10,15, ...]
    returns:
      batch_starts: sample-level start index for each batch
      batch_ends:   sample-level end index (exclusive)
      batch_group_spans: [(g0,g1), ...] so per-batch lengths are group_lengths[g0:g1]
    """
    counts = np.asarray(group_lengths, dtype=np.int64)
    if counts.ndim != 1:
        raise ValueError("group_lengths must be 1D")
    if counts.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), []
    if (counts <= 0).any():
        raise ValueError("group_lengths must be positive")
    if max_batch_items <= 0:
        raise ValueError("max_batch_items must be > 0")

    prefix = np.cumsum(counts, dtype=np.int64)  # end idx of each group
    n = counts.size

    starts, ends = [], []
    batch_group_spans: List[Tuple[int, int]] = []

    g = 0
    while g < n:
        if counts[g] > max_batch_items:
            if not allow_oversize_group:
                raise ValueError(
                    f"group {g} size {counts[g]} > max_batch_items={max_batch_items}"
                )
            g_next = g + 1
        else:
            prev_total = 0 if g == 0 else int(prefix[g - 1])
            limit = prev_total + max_batch_items
            g_next = int(np.searchsorted(prefix, limit, side="right"))
            g_next = max(g_next, g + 1)

        s = 0 if g == 0 else int(prefix[g - 1])
        e = int(prefix[g_next - 1])

        # optional drop_last behavior for final underfilled batch
        if drop_last and g_next == n and (e - s) < max_batch_items:
            break

        starts.append(s)
        ends.append(e)
        batch_group_spans.append((g, g_next))
        g = g_next

    return (
        np.asarray(starts, dtype=np.int64),
        np.asarray(ends, dtype=np.int64),
        batch_group_spans,
    )

def resolve_skip_to_group_boundary(
    group_lengths: Sequence[int],
    skip_rows: int,
    mode: str = "strict",  # "strict" | "ceil" | "floor"
) -> Tuple[int, int, np.ndarray]:
    """
    Returns:
      g_skip:      number of whole groups to skip
      skip_aligned:actual rows skipped at boundary
      boundaries:  cumulative boundaries (len = n_groups + 1)
    """
    gl = np.asarray(group_lengths, dtype=np.int64)
    if gl.ndim != 1:
        raise ValueError("group_lengths must be 1D")
    if gl.size > 0 and np.any(gl <= 0):
        raise ValueError("group_lengths must be positive")
    if skip_rows < 0:
        raise ValueError("skip_rows must be >= 0")

    boundaries = np.empty(gl.size + 1, dtype=np.int64)
    boundaries[0] = 0
    if gl.size:
        boundaries[1:] = np.cumsum(gl, dtype=np.int64)

    total = int(boundaries[-1])
    if skip_rows > total:
        raise ValueError(f"skip_rows={skip_rows} exceeds total rows={total}")

    if mode == "strict":
        g_skip = int(np.searchsorted(boundaries, skip_rows, side="left"))
        if boundaries[g_skip] != skip_rows:
            raise ValueError(
                f"skip_rows={skip_rows} is not a group boundary. "
                f"Use mode='ceil' or mode='floor', or pass one of boundaries."
            )
    elif mode == "ceil":
        g_skip = int(np.searchsorted(boundaries, skip_rows, side="left"))
    elif mode == "floor":
        g_skip = int(np.searchsorted(boundaries, skip_rows, side="right")) - 1
    else:
        raise ValueError("mode must be 'strict', 'ceil', or 'floor'")

    skip_aligned = int(boundaries[g_skip])
    return g_skip, skip_aligned, boundaries

def build_plan_with_skip(
    group_lengths: Sequence[int],
    max_batch_items: int,
    skip_rows: int,
    skip_mode: str = "strict",
    drop_last: bool = False,
    allow_oversize_group: bool = True,
):
    g_skip, skip_aligned, _ = resolve_skip_to_group_boundary(
        group_lengths, skip_rows, mode=skip_mode
    )

    # Rebuild on remaining groups only
    rem = np.asarray(group_lengths, dtype=np.int64)[g_skip:]
    local_starts, local_ends, local_group_spans = build_batch_plan(
        rem,
        max_batch_items=max_batch_items,
        drop_last=drop_last,
        allow_oversize_group=allow_oversize_group,
    )

    # Convert local sample indices back to global dataset indices
    starts = local_starts + skip_aligned
    ends = local_ends + skip_aligned

    # Convert local group spans back to global group indices
    group_spans = [(a + g_skip, b + g_skip) for (a, b) in local_group_spans]

    return starts, ends, group_spans, skip_aligned, g_skip

class BoundaryBatchSampler(Sampler[range]):
    def __init__(self, batch_starts, batch_ends):
        self.batch_starts = np.asarray(batch_starts, dtype=np.int64)
        self.batch_ends = np.asarray(batch_ends, dtype=np.int64)
        if self.batch_starts.shape != self.batch_ends.shape:
            raise ValueError("starts/ends shape mismatch")
        if np.any(self.batch_ends <= self.batch_starts):
            raise ValueError("invalid span: end must be > start")

    def __len__(self):
        return int(self.batch_starts.size)

    def __iter__(self):
        for s, e in zip(self.batch_starts, self.batch_ends):
            yield range(int(s), int(e))   # contiguous indices, memory-efficient

class LoaderWithGroupLengths:
    def __init__(self, loader, group_lengths, batch_group_spans):
        self.loader = loader
        self.group_lengths = np.asarray(group_lengths, dtype=np.int64)
        self.batch_group_spans = batch_group_spans
        if len(self.batch_group_spans) != len(self.loader):
            raise ValueError("metadata length must match number of loader batches")

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        # Requires FIFO batch return order.
        for bidx, batch in enumerate(self.loader):
            g0, g1 = self.batch_group_spans[bidx]
            lens = self.group_lengths[g0:g1]   # e.g. [5, 10, 15]
            yield batch, lens

if __name__ == "__main__":

    chat_template = Path("/root/vllm/examples/pooling/score/template/qwen3_vl_reranker.jinja").read_text()


    UUID_CACHE = {}
    pair_map: list[tuple[int, int]] = []


    with open("/data_external/InfoSeek/infoseek_train.jsonl") as f:
        infoseek = f.readlines()
        infoseek = [json.loads(e) for e in infoseek]

    def make_ranges(n, chunk):
        return [(i, min(i + chunk, n)) for i in range(0, n, chunk)]
    ranges = make_ranges(len(infoseek)-200_000, 80_000)

    with open('/data_external/InfoSeek/query_ret_imgs_qwen.jsonl') as f:
        retrieved_collection = [json.loads(e) for e in f.readlines()]

    evqa_kb = load_from_disk('/data_external/evqa/image_kb')
    kb_titles = evqa_kb['title']
    kb_title_set = set(evqa_kb['title'])
    kb_title_map = {t:i for i, t in enumerate(evqa_kb['title'])}

    with open("/data_external/InfoSeek/oven_entity_train.jsonl") as f:
        oven = f.readlines()
        oven = [json.loads(e) for e in oven]

    with open("/data_external/InfoSeek/oven_entity_val.jsonl") as f:
        temp = f.readlines()
        temp = [json.loads(e) for e in temp]

    oven.extend(temp)

    with open("/data_external/InfoSeek/oven_entity_test.jsonl") as f:
        oven_test = f.readlines()
        oven_test = [json.loads(e) for e in oven_test]

    oven_image_id_entity_map = {e['image_id']:e['entity_text'] for e in tqdm(oven)}


    #

    rc =0
    
    #rerank_ds[1234]
    
    # def gen(shards):
    #     global UUID_CACHE
    #     for start, end in shards:
    #         ret_c = retrieved_collection[start:end]
    #         questions = infoseek[start:end]
    #         for row, ques in tqdm(zip(ret_c, questions), total=len(questions)):
    #             assert row['question_id'] == ques['data_id']
    #             if len(UUID_CACHE)>10000:
    #                 UUID_CACHE = {}
    #             ret_images = row['ret_images']
    #             ret_docs = []
    #             ret_imgs_filtered = []
    #             for ret_img in ret_images:
    #                 if len(ret_docs) == 30:
    #                     break
    #                 ovid = ret_img.split('/')[-1].split('.')[0]
    #                 try:
    #                     in_evqa = kb_title_map[oven_image_id_entity_map[ovid]]
    #                     ret_docs.append(oven_image_id_entity_map[ovid])
    #                     ret_imgs_filtered.append(ret_img)
    #                 except KeyError:
    #                     continue
    #             query_retdocs_map = defaultdict(list)
    #             for i, doc_tit in enumerate(ret_docs):
    #                 query_retdocs_map[doc_tit].append(i)
    #             fulltext_ret_docs = [evqa_kb[kb_title_map[e]] for e in query_retdocs_map.keys()]

    #             section_dicts = [{'id': doc['url'], 'title': doc['title'], 'sections': [{'title': sec_tit,  'text': sec_txt} for sec_tit, sec_txt in zip(doc['section_titles'], doc['section_texts'])]} for doc in fulltext_ret_docs]
    #             ret_doc_chunks = [chunk_article(sec_dic, target_tokens=200, min_tokens=150, max_tokens=300, overlap_sents=2, prefix_headings_in_text=True) for sec_dic in section_dicts]
    #             ret_img_for_chunks = [ret_imgs_filtered[e[0]] for e in query_retdocs_map.values()]
    #             for rerank_qimg, (title, ret_img_idx), chunk_obj in zip(ret_img_for_chunks, query_retdocs_map.items(), ret_doc_chunks):
    #                 #output_buffer.append({'infoseek_qid': ques['data_id'], 'ret_img_wiki_title': title, 'ret_img_idx': ret_img_idx, 'rerank_buffer_idx': [], 'chunks': chunk_obj})
    #                 for chunk in chunk_obj:
    #                     yield {'qid': ques['data_id'],'question': ques['question'], 'ret_img': rerank_qimg, 'doc_chunk':chunk.text, 'ret_img_wiki_title': title, 'ret_img_idx': ret_img_idx,}

    # features = Features({
    #     "qid": Value("string"),
    #     "question": Value("string"),
    #     "ret_img": Value("string"),
    #     "doc_chunk": Value("string"),
    #     "ret_img_wiki_title": Value("string"),
    #     "ret_img_idx": HFSequence(Value("int64")),
    # })
    # t0 = time.perf_counter()
    # hf_ds = HFDataset.from_generator(
    #     gen,
    #     gen_kwargs={"shards": ranges},
    #     features=features,
    #     keep_in_memory=False,
    #     num_proc=10
    # )

    #hf_ds.save_to_disk("/data_external/InfoSeek/ds_for_rerank", max_shard_size="1GB",)

    # query_batch_buffer = []
    # doc_batch_buffer = []
    # output_buffer = []
    # buffer_step = 0
    # num_chunks=[]
    # for row, ques in tqdm(zip(retrieved_collection, infoseek), total=len(infoseek)):
    #     assert row['question_id'] == ques['data_id']

    #     if len(UUID_CACHE)>10000:
    #         UUID_CACHE = {}
    #     ret_images = row['ret_images']
    #     ret_docs = []
    #     ret_imgs_filtered = []
    #     for ret_img in ret_images:
    #         if len(ret_docs) == 30:
    #             break
    #         ovid = ret_img.split('/')[-1].split('.')[0]
    #         try:
    #             in_evqa = kb_title_map[oven_image_id_entity_map[ovid]]
    #             ret_docs.append(oven_image_id_entity_map[ovid])
    #             ret_imgs_filtered.append(ret_img)

    #         except KeyError:
    #             continue
        
    #     query_retdocs_map = defaultdict(list)
    #     for i, doc_tit in enumerate(ret_docs):
    #         query_retdocs_map[doc_tit].append(i)


    #     fulltext_ret_docs = [evqa_kb[kb_title_map[e]] for e in query_retdocs_map.keys()]

    #     section_dicts = [{'id': doc['url'], 'title': doc['title'], 'sections': [{'title': sec_tit,  'text': sec_txt} for sec_tit, sec_txt in zip(doc['section_titles'], doc['section_texts'])]} for doc in fulltext_ret_docs]
    #     ret_doc_chunks = [chunk_article(sec_dic, target_tokens=200, min_tokens=150, max_tokens=300, overlap_sents=2, prefix_headings_in_text=True) for sec_dic in section_dicts]
    #     ret_img_for_chunks = [ret_imgs_filtered[e[0]] for e in query_retdocs_map.values()]


    #     for rerank_qimg, (title, ret_img_idx), chunk_obj in zip(ret_img_for_chunks, query_retdocs_map.items(), ret_doc_chunks):
    #         output_buffer.append({'infoseek_qid': ques['data_id'], 'ret_img_wiki_title': title, 'ret_img_idx': ret_img_idx, 'rerank_buffer_idx': [], 'chunks': chunk_obj})
    #         for chunk in chunk_obj:
    #             query_batch_buffer.append({'text': ques['question'], 'image': rerank_qimg})
    #             doc_batch_buffer.append({'text': chunk.text,})
    #             output_buffer[-1]['rerank_buffer_idx'].append(buffer_step)
    #             buffer_step += 1

    #     if buffer_step < MACRO_BATCH_SIZE:
    #         continue

    #     flat_queries = []
    #     flat_docs = []
    #     for qq, dd in zip(query_batch_buffer, doc_batch_buffer):
    #         flat_queries.append(format_query_to_score_param(qq))                  # repeated reference is fine
    #         flat_docs.append(format_doc_to_score_param(dd))

    # llm = LLM(
    #     model="/data_external/Qwen3-VL-Reranker-2B",
    #     runner="pooling",
    #     hf_overrides={
    #         "architectures": ["Qwen3VLForSequenceClassification"],
    #         "classifier_from_token": ["no", "yes"],
    #         "is_original_qwen3_reranker": True,
    #     },
    #     gpu_memory_utilization=0.9,
    #     tensor_parallel_size=1,
    #     max_num_batched_tokens=200_000,
    #     enable_prefix_caching=True,
    #     limit_mm_per_prompt={"image": 1},
    #     allowed_local_media_path="/data_external/InfoSeek"
    # )

    flattened_ds = load_from_disk('/data_external/InfoSeek/ds_for_rerank')

    group_lengths= np.load('/data_external/InfoSeek/rr_group_id.npz.npy')


    group_lengths = group_lengths[:2711381]
    flattened_ds = flattened_ds.select(range(80000091))
    starts, ends, batch_group_spans = build_batch_plan(
        group_lengths=group_lengths,
        max_batch_items=50000,
        drop_last=False,
        allow_oversize_group=True,
    )

    # starts, ends, batch_group_spans, x_used, g_used = build_plan_with_skip(
    #     group_lengths=group_lengths,
    #     max_batch_items=50000,
    #     skip_rows=80_000_000,
    #     skip_mode="ceil",
    # )
    batch_sampler = BoundaryBatchSampler(starts, ends)

    def collate(inputs):
        for i in range(len(inputs)):
            temp = inputs[i]['doc_chunk']
            temp = temp.split(' Section:')
            temp = temp[0]+', Section:'+temp[1]
            inputs[i]['doc_chunk'] = temp
        return inputs

    loader = DataLoader(
        flattened_ds,
        batch_sampler=batch_sampler,    # do NOT set batch_size/shuffle/sampler/drop_last
        num_workers=16,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=True,
        in_order=True,
    )

    loader_with_meta = LoaderWithGroupLengths(loader, group_lengths, batch_group_spans)
    gather_outputs = open('/data_external/InfoSeek/ret_img_gt_docs_p3.jsonl', 'w')
    for batch, groups in tqdm(loader_with_meta):
        assert len(batch) == int(groups.sum())
        if len(UUID_CACHE)>10000:
            UUID_CACHE = {}
        flat_queries = [format_query_to_score_param({'text': e['question'], 'image': e['ret_img']}) for e in batch]
        flat_docs = [format_doc_to_score_param({'text': e['doc_chunk']}) for e in batch]

        # if step == 100:
        #     break
        outs = llm.score(flat_queries, flat_docs, chat_template=chat_template, use_tqdm=False)
        flat_scores = [o.outputs.score for o in outs]

        group_start = 0
        for group in groups:
            current_subq_scores = flat_scores[group_start: group_start+group]
            top5_idx = np.argsort(current_subq_scores)[::-1][:5]
            top5_chunks = [batch[e+group_start]['doc_chunk'] for e in top5_idx]
            payload = {'qid': batch[group_start]['qid'], 'ret_img_idx': batch[group_start]['ret_img_idx'], 'top5_chunks': top5_chunks}
            gather_outputs.write(json.dumps(payload))
            gather_outputs.write('\n')
            group_start+=group
            
