from typing import Union, Iterable

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
