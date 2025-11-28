from typing import Dict, Any, List, Sequence, Optional, Tuple, Union
import math

def _mean_ci95(xs: Sequence[float]) -> Tuple[float, float, Tuple[float, float]]:
    """
    Normal-approx 95% CI for mean (useful for large Q). Returns (mean, se, (lo, hi)).
    If n<2, se and CI are NaN.
    """
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), (float("nan"), float("nan"))
    mu = sum(xs) / n
    if n == 1:
        return mu, float("nan"), (float("nan"), float("nan"))
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    se = math.sqrt(var / n)
    z = 1.96
    return mu, se, (mu - z * se, mu + z * se)

def aggregate_eval_results(
    results_list: Sequence[Dict[str, Any]],
    ks: Optional[Sequence[int]] = None,
    with_ci: bool = False,
) -> Dict[str, Any]:
    if ks is None:
        k_union = set()
        for res in results_list:
            k_union.update(res.get("ks", []))
        ks = sorted(int(k) for k in k_union)
    else:
        ks = sorted(set(int(k) for k in ks))

    all_rows: List[Dict[str, Any]] = []
    gidx = 0
    for run_idx, res in enumerate(results_list):
        for row in res.get("per_query", []):
            norm = dict(row)
            norm["run_index"] = run_idx
            norm["global_query_index"] = gidx
            norm["gt_size"] = int(row.get("gt_size", 0) or 0)
            all_rows.append(norm)
            gidx += 1

    per_k_summary: Dict[int, Dict[str, Any]] = {}
    for k in ks:
        rkey, hkey, akey = f"recall@{k}", f"hit@{k}", f"AP@{k}"
        n_q = 0
        sum_r = 0.0
        sum_h = 0.0
        sum_hits_cnt = 0.0
        sum_gt = 0
        sum_ap = 0.0

        r_vec: List[float] =[] 
        h_vec: List[float] = [] 
        ap_vec: List[float] = []

        for row in all_rows:
            gt_size = row["gt_size"]
            if gt_size <= 0:
                continue
            r = row.get(rkey); h = row.get(hkey); ap = row.get(akey)
            if r is None or h is None or ap is None or any(isinstance(x, float) and math.isnan(x) for x in (r, h, ap)):
                continue

            n_q += 1
            sum_r += r
            sum_h += float(h)
            sum_ap += ap
            sum_hits_cnt += r * gt_size
            sum_gt += gt_size
            r_vec.append(r); h_vec.append(float(h)); ap_vec.append(ap)

        if n_q == 0 or sum_gt == 0:
            per_k_summary[k] = {
                "macro_recall": float("nan"),
                "micro_recall": float("nan"),
                "hit_rate": float("nan"),
                "avg_hits_per_query": float("nan"),
                "mAP": float("nan"),
                "n_queries": 0,
            }
            continue

        out = {
            "macro_recall": sum_r / n_q,
            "micro_recall": sum_hits_cnt / sum_gt,
            "hit_rate": sum_h / n_q,
            "avg_hits_per_query": sum_hits_cnt / n_q,
            "mAP": sum_ap / n_q,
            "n_queries": n_q,
        }

        if with_ci:
            _, se_r, ci_r = _mean_ci95(r_vec)
            _, se_h, ci_h = _mean_ci95(h_vec)
            _, se_a, ci_a = _mean_ci95(ap_vec)
            out.update({
                "macro_recall_se": se_r, "macro_recall_ci95": ci_r,
                "hit_rate_se": se_h,     "hit_rate_ci95": ci_h,
                "mAP_se": se_a,          "mAP_ci95": ci_a,
            })

        per_k_summary[k] = out

    return {"ks": ks, "per_k": per_k_summary, "n_total_runs": len(results_list), "all_per_query": all_rows}

def print_agg_summary(agg: Dict[str, Any]) -> None:
    header = f"{'k':>3} | {'macro_recall':>12} {'micro_recall':>12} {'hit_rate':>10} {'mAP':>8} {'avg_hits/q':>11} {'n_q':>6}"
    print(header)
    print("-" * len(header))
    for k in agg["ks"]:
        s = agg["per_k"][k]
        print(
            f"{k:>3} | {s['macro_recall']:.4f}     {s['micro_recall']:.4f}     "
            f"{s['hit_rate']:.4f}    {s['mAP']:.4f}   {s['avg_hits_per_query']:.4f}   {s['n_queries']:>6}"
        )


def _as_batched(x: Union[List[str], List[List[str]]]) -> List[List[str]]:
    """Ensure inputs are in batched form: List[List[str]]."""
    if not x:
        return []
    if isinstance(x[0], list):
        return x  # already batched
    if isinstance(x[0], int):
        x = [str(e) for e in x]
    return [x]   # wrap single query

def _dedup_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out

def _ap_at_k(ret_list: Sequence[str], gt_set: set, k: int, denom_mode: str) -> float:
    """
    Average Precision at k.
    denom_mode in {"|G|", "min(|G|,k)"}:
      - "|G|" (default in IR) normalizes by |G|; AP@k <= 1 if k >= |G|.
      - "min(|G|,k)" normalizes by min(|G|,k); keeps AP@k in [0,1] even when k < |G|.
    """
    if not gt_set or k <= 0:
        return float("nan")
    k_eff = min(k, len(ret_list))
    if k_eff == 0:
        return 0.0
    num_hits = 0
    prec_sum = 0.0
    for rank, h in enumerate(ret_list[:k_eff], start=1):
        if h in gt_set:
            num_hits += 1
            prec_sum += num_hits / rank
    denom = len(gt_set) if denom_mode == "|G|" else min(len(gt_set), k)
    return prec_sum / denom if denom > 0 else float("nan")

def evaluate_retrieval(
    ret_hash: list,
    gt_hash: list,
    ks: Sequence[int] = (1, 5, 10, 20, 30),
    *,
    case_insensitive: bool = True,
    deduplicate_retrieved: bool = True,
    ap_denom_mode: str = "|G|",  # or "min(|G|,k)"
) -> Dict[str, Any]:
    """
    Adds AP@k per query and mAP@k aggregates to recall@k and hit@k.
    recall@k_i = |R_i(k) ∩ G_i| / |G_i|
    hit@k_i    = 1[|R_i(k) ∩ G_i| > 0]
    AP@k_i     = (Σ_{r=1..k} Precision@r * rel_r) / denom, denom per `ap_denom_mode`
    mAP@k      = mean_i AP@k_i over queries with |G_i|>0
    """
    batched_ret = _as_batched(ret_hash)
    batched_gt  = _as_batched(gt_hash)
    if len(batched_ret) != len(batched_gt):
        raise ValueError(f"ret_hash has {len(batched_ret)} queries but gt_hash has {len(batched_gt)}.")

    def _norm(h: str) -> str:
        return h.lower() if case_insensitive else h

    processed_ret, processed_gt = [], []
    for r, g in zip(batched_ret, batched_gt):
        r_proc = [_norm(x) for x in r]
        g_proc = [_norm(x) for x in g]
        if deduplicate_retrieved:
            r_proc = _dedup_preserve_order(r_proc)
        processed_ret.append(r_proc)
        processed_gt.append(g_proc)

    ks = sorted(set(int(k) for k in ks if k > 0))
    n_queries = len(processed_ret)
    if n_queries == 0:
        return {"per_query": [], "per_k": {}, "ks": ks}

    per_query_rows: List[Dict[str, Any]] = []
    sum_recall = {k: 0.0 for k in ks}
    sum_hits   = {k: 0   for k in ks}
    sum_gt_tot = {k: 0   for k in ks}
    sum_hitind = {k: 0   for k in ks}
    sum_ap     = {k: 0.0 for k in ks}

    for qi, (ret_list, gt_list) in enumerate(zip(processed_ret, processed_gt)):
        gt_set = set(gt_list)
        gt_size = len(gt_set)
        row: Dict[str, Any] = {"query_index": qi, "gt_size": gt_size}

        if gt_size == 0:
            for k in ks:
                row[f"recall@{k}"] = float("nan")
                row[f"hit@{k}"]    = float("nan")
                row[f"AP@{k}"]     = float("nan")
            per_query_rows.append(row)
            continue

        for k in ks:
            k_eff = min(k, len(ret_list))
            Rk = ret_list[:k_eff]
            inter = gt_set.intersection(Rk)
            num_hits_k = len(inter)

            recall_k = num_hits_k / gt_size
            hit_ind  = 1 if num_hits_k > 0 else 0
            ap_k     = _ap_at_k(ret_list, gt_set, k, ap_denom_mode)

            row[f"recall@{k}"] = recall_k
            row[f"hit@{k}"]    = hit_ind
            row[f"AP@{k}"]     = ap_k

            sum_recall[k] += recall_k
            sum_hits[k]   += num_hits_k
            sum_gt_tot[k] += gt_size
            sum_hitind[k] += hit_ind
            sum_ap[k]     += ap_k

        per_query_rows.append(row)

    valid_q = sum(1 for r in per_query_rows if r["gt_size"] > 0)
    per_k_summary = {}
    for k in ks:
        macro_recall = (sum_recall[k] / valid_q) if valid_q > 0 else float("nan")
        micro_recall = (sum_hits[k]   / sum_gt_tot[k]) if sum_gt_tot[k] > 0 else float("nan")
        hit_rate     = (sum_hitind[k] / valid_q) if valid_q > 0 else float("nan")
        mAP_k        = (sum_ap[k]     / valid_q) if valid_q > 0 else float("nan")
        per_k_summary[k] = {
            "macro_recall": macro_recall,
            "micro_recall": micro_recall,
            "hit_rate": hit_rate,
            "avg_hits_per_query": (sum_hits[k] / valid_q) if valid_q > 0 else float("nan"),
            "mAP": mAP_k,
        }

    return {
        "ks": ks,
        "per_query": per_query_rows,
        "per_k": per_k_summary,
        "config": {
            "case_insensitive": case_insensitive,
            "deduplicate_retrieved": deduplicate_retrieved,
            "ap_denom_mode": ap_denom_mode,
        },
    }

import math
import random
from typing import List, Iterable, Iterator, Sequence, Optional
from torch.utils.data import Dataset

class MinimumDS(Dataset):
    def __init__(self, samples):
        super().__init__()
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, indices: list):
        return [self.samples[idx] for idx in indices]

class MaxCostBatchSampler:
    """
    Yield batches of indices such that:
      (i)  sum(cost[idx]) <= max_cost_per_batch, and
      (ii) len(batch)     <= max_items_per_batch (if set).

    This produces variable-size batches up to the item cap, with approximately
    constant memory per batch governed by max_cost_per_batch.

    Args:
        indices: Dataset indices to sample from.
        costs: Per-sample nonnegative costs aligned with `indices`.
        max_cost_per_batch: Upper bound on the sum of costs per batch.
        max_items_per_batch: Upper bound on the number of items per batch (optional).
        shuffle: Shuffle indices each epoch before packing.
        seed: RNG seed used only if shuffle=True.
        sort_desc_within_epoch: Sort by descending cost after shuffling (first-fit-decreasing).
        drop_last: Drop the final incomplete batch at epoch end.

    Notes:
        - If a single sample's cost exceeds `max_cost_per_batch`, it is yielded alone.
        - `__len__` returns a conservative estimate (useful for schedulers).
    """
    def __init__(
        self,
        indices: Sequence[int],
        costs: Sequence[float],
        max_cost_per_batch: float,
        max_items_per_batch: Optional[int] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
        sort_desc_within_epoch: bool = True,
        drop_last: bool = False,
    ):
        assert len(indices) == len(costs), "indices and costs must align"
        assert max_cost_per_batch > 0, "max_cost_per_batch must be positive"
        if max_items_per_batch is not None:
            assert max_items_per_batch >= 1, "max_items_per_batch must be >= 1"

        self.indices = list(indices)
        self.costs = list(costs)
        self.max_cost = float(max_cost_per_batch)
        self.max_items = max_items_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.sort_desc = sort_desc_within_epoch
        self.drop_last = drop_last

        self._cost = self.costs.__getitem__

    def __len__(self) -> int:
        # Conservative lower-bound/upper-bound synthesis for epoch length:
        # - cost-based batches (lower bound): floor(total_cost / max_cost)
        # - item-cap batches (lower bound):  ceil(N / max_items) if capped
        total_cost = sum(self.costs[i] for i in self.indices)
        cost_based = max(1, math.floor(total_cost / self.max_cost))
        if self.max_items is None:
            return cost_based
        item_based = math.ceil(len(self.indices) / self.max_items)
        return max(cost_based, item_based)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed) if self.seed is not None else random
        order = list(self.indices)
        if self.shuffle:
            rng.shuffle(order)
        if self.sort_desc:
            order.sort(key=lambda i: self._cost(i), reverse=True)

        batch, acc = [], 0.0
        for idx in order:
            c = self._cost(idx)

            # Oversize singleton: emit current batch (if any), then the singleton.
            if c > self.max_cost:
                if batch:
                    yield batch
                yield [idx]
                batch, acc = [], 0.0
                continue

            # If adding this item would violate cost OR item cap, flush current batch first.
            would_violate_cost = (acc + c) > self.max_cost
            would_violate_cap = (self.max_items is not None) and (len(batch) >= self.max_items)
            if would_violate_cost or would_violate_cap:
                if batch or not self.drop_last:
                    yield batch
                batch, acc = [], 0.0

            # Start new batch with current item
            batch.append(idx)
            acc += c

            # If we exactly reach the item cap after adding, flush now.
            if (self.max_items is not None) and (len(batch) >= self.max_items):
                yield batch
                batch, acc = [], 0.0

        if batch and not self.drop_last:
            yield batch