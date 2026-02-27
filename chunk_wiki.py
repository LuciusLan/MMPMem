#!/usr/bin/env python3
"""
Section-bounded, sentence-aware chunking for (cleaned) Wikipedia dumps.

Assumed input (JSONL):
  One JSON object per line, representing an article:
  {
    "id": "optional",
    "title": "Article title",
    "sections": [
      {
        "path": ["History", "Early life"],   # hierarchical headings (optional)
        "title": "Early life",              # optional; may duplicate path[-1]
        "text": "Section text ..."
      },
      ...
    ]
  }

Output (JSONL):
  One JSON object per line, representing a chunk with metadata.

Tokenization:
  - Sentences: nltk.sent_tokenize
  - Tokens:    nltk.word_tokenize  (generic, non-LLM tokenizer)

Defaults:
  target_tokens = 150
  min_tokens    = 100
  max_tokens    = 240
  overlap_sents = 2
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Sequence

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


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
                    chunk_text = "\n".join(header_lines) + "\n\n" + chunk_text

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


# ----------------------------- I/O ----------------------------------------- #

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------- Main ---------------------------------------- #

def main() -> None:
    ensure_nltk()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSONL path")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")

    parser.add_argument("--target_tokens", type=int, default=150)
    parser.add_argument("--min_tokens", type=int, default=100)
    parser.add_argument("--max_tokens", type=int, default=240)
    parser.add_argument("--overlap_sents", type=int, default=2)

    parser.add_argument(
        "--prefix_headings_in_text",
        action="store_true",
        help="If set, prepend article/section headings into chunk text (useful for embedding).",
    )

    args = parser.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)

    def gen_rows() -> Iterable[Dict[str, Any]]:
        for article in read_jsonl(inp):
            chunks = chunk_article(
                article,
                target_tokens=args.target_tokens,
                min_tokens=args.min_tokens,
                max_tokens=args.max_tokens,
                overlap_sents=args.overlap_sents,
                prefix_headings_in_text=args.prefix_headings_in_text,
            )
            for c in chunks:
                yield asdict(c)

    write_jsonl(outp, gen_rows())


if __name__ == "__main__":
    main()
