from typing import Union, Iterable, Optional
import unicodedata
import re
import string

def levenshtein(a: str, b: str) -> int:
    """Standard Levenshtein distance."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,         # deletion
                curr[j - 1] + 1,     # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev
    return prev[m]


def anls_single(pred: str, golds: Union[Iterable[str], str], tau: float = 0.5) -> float:
    """
    ANLS for a single QA item with possibly multiple references.
    Follows DocVQA/InfographicVQA definition. :contentReference[oaicite:5]{index=5}
    """
    if isinstance(golds, str):
        golds = [golds]

    pred = pred.strip()
    if len(pred) == 0:
        return 0.0

    best = 0.0
    for g in golds:
        g = g.strip()
        if len(g) == 0:
            continue
        d = levenshtein(pred.lower(), g.lower())
        nl = d / max(len(pred), len(g))
        score = 1.0 - nl if nl < tau else 0.0
        if score > best:
            best = score
    return best


def normalize_docvqa_string(s: str) -> str:
    """
    Normalization for strict EM in DocVQA/InfographicVQA:
    - strip leading/trailing spaces
    - lowercase
    """
    return s.strip().lower()


def exact_match_doc_like(pred: str, golds: Union[Iterable[str], str]) -> bool:
    if isinstance(golds, str):
        golds = [golds]
    np = normalize_docvqa_string(pred)
    return any(np == normalize_docvqa_string(g) for g in golds)


def score_docvqa_like(
    pred: str,
    golds: Union[Iterable[str], str],
) -> dict[str, float]:
    """
    For DocVQA and InfographicVQA: ANLS + strict EM. :contentReference[oaicite:9]{index=9}
    """
    if isinstance(golds, str):
        golds_list = [golds]
    else:
        golds_list = list(golds)

    anls = anls_single(pred, golds_list)
    em = 1.0 if exact_match_doc_like(pred, golds_list) else 0.0
    return {"anls": anls, "em": em}


def normalize_infoseek_answer(text: str) -> str:
    """
    Normalization for InfoSeek STRING/TIME answers:

    - Unicode NFKC
    - lowercase
    - remove English articles (a/an/the)
    - strip ASCII punctuation
    - collapse whitespace
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = " ".join(text.split())
    return text


def infoseek_exact_match(pred: str, golds: Union[str, list[str]]) -> bool:
    """
    Correct if normalized prediction matches ANY normalized reference.
    """
    if isinstance(golds, str):
        golds = [golds]
    np = normalize_infoseek_answer(pred)
    return any(np == normalize_infoseek_answer(g) for g in golds)

_NUMERIC_TOKEN_RE = re.compile(
    r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?"
)


def _clean_range_hyphens(text: str) -> str:
    """
    Heuristic: split range hyphens from minus signs.
    Example: '9-10' -> '9 - 10', but '-5' remains '-5'.
    """
    out = []
    for i, ch in enumerate(text):
        if ch == "-" and i > 0 and text[i - 1].isdigit():
            out.append(" - ")
        else:
            out.append(ch)
    return "".join(out)


def parse_infoseek_numeric_pred(s: str) -> Optional[Union[float, list[float]]]:
    """
    Parse a numeric prediction string into either:

      - a scalar float, or
      - a [v1, v2] range (first two numbers found).

    Returns None if no numeric token can be parsed.
    """
    s = unicodedata.normalize("NFKC", s)
    s = _clean_range_hyphens(s)

    raw_tokens = _NUMERIC_TOKEN_RE.findall(s)
    cleaned: list[float] = []
    for token in raw_tokens:
        token = token.replace(",", "").strip(".")
        # Avoid malformed numbers with multiple decimal points
        if token.count(".") > 1:
            token = token.split(".")[0]
        try:
            cleaned.append(float(token))
        except ValueError:
            continue

    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    # Only the first two numbers are used for a predicted range
    return cleaned[:2]

def range_iou(x: list[float], y: list[float]) -> float:
    """
    Intersection-over-Union for 1D ranges [x_min, x_max] and [y_min, y_max].
    """
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    overlap = max(0.0, min(x_max, y_max) - max(x_min, y_min))
    len_x = (x_max - x_min) + 1e-12
    len_y = (y_max - y_min) + 1e-12
    denom = len_x + len_y - overlap
    return overlap / denom if denom > 0 else 0.0


def infoseek_numeric_correct(
    pred: Optional[Union[float, list[float]]],
    gold_dict: dict,
    tolerance: float = 0.10,
) -> bool:
    """
    Numerical relaxed accuracy for InfoSeek (Appendix A.3):

      - Ground truth format: {"wikidata": float, "range": [ref_min, ref_max]}.
        * 'wikidata' is the canonical value.
        * 'range' is the acceptable reference range (often +/-10% around wikidata).

      - If 'range' is missing, construct it as [v*(1-tol), v*(1+tol)] from 'wikidata'.

      - Scalar prediction 'p' is correct if: ref_min <= p <= ref_max.

      - Range prediction [p_min, p_max] is correct if:
          IoU([p_min, p_max], [ref_min, ref_max]) >= 0.5.
    """
    if pred is None:
        return False

    if "wikidata" not in gold_dict:
        raise ValueError("InfoSeek numerical gold must contain 'wikidata' field.")

    v = float(gold_dict["wikidata"])
    ref_range = gold_dict.get("range", None)

    if ref_range is not None and len(ref_range) >= 2:
        ref_min = float(ref_range[0])
        ref_max = float(ref_range[1])
        ref_min, ref_max = min(ref_min, ref_max), max(ref_min, ref_max)
    else:
        # Fallback: construct ±10% tolerance around wikidata value
        ref_min = v * (1 - tolerance)
        ref_max = v * (1 + tolerance)
        ref_min, ref_max = min(ref_min, ref_max), max(ref_min, ref_max)

    if isinstance(pred, list):
        if len(pred) == 0:
            return False
        if len(pred) == 1:
            p = float(pred[0])
            return ref_min <= p <= ref_max
        # Use first two values for a predicted range
        p1, p2 = float(pred[0]), float(pred[1])
        p_min, p_max = min(p1, p2), max(p1, p2)
        iou = range_iou([p_min, p_max], [ref_min, ref_max])
        return iou >= 0.5 - 1e-12

    # Scalar prediction
    p = float(pred)
    return ref_min <= p <= ref_max

def score_infoseek(
    pred_raw: str,
    gold,

) -> dict[str, float]:
    """
    Per-item InfoSeek scoring.

    Parameters
    ----------
    pred_raw : str
        Full model output (may include reasoning); final answer must be in \\boxed{...}.
    gold :
        - For STRING/TIME: list[str] (acceptable paraphrases).
        - For NUMERICAL: dict with keys {"wikidata": float, "range": [min, max]}.
    question_type : str
        One of {"string", "time", "numerical"} (case-insensitive).

    Returns
    -------
    dict
        - For STRING: {"accuracy": 0/1, "string_correct": 0/1}
        - For TIME:   {"accuracy": 0/1, "time_correct": 0/1}
        - For NUMERICAL: {"accuracy": 0/1, "num_correct": 0/1}
    """
    #qtype = question_type.strip().lower()
    pred_text = pred_raw

    if isinstance(gold, dict):
        pred_val = parse_infoseek_numeric_pred(pred_text)
        correct = infoseek_numeric_correct(pred_val, gold)
        return {
            "acc": 1.0 if correct else 0.0,
            "num_correct": 1.0 if correct else 0.0,
        }
    else:
        # STRING / TIME: gold is a list of acceptable paraphrases
        if isinstance(gold, str):
            golds = [gold]
        else:
            golds = list(gold)

        correct = infoseek_exact_match(pred_text, golds)
        result = {"acc": 1.0 if correct else 0.0}
    return result