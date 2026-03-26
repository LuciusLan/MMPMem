from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict, List, Any,Union
import copy

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache


# ----------------------------
# Masking utilities
# ----------------------------

def token_mask_right_padded(
    tokens: torch.Tensor,
    pad_id: int,
    eos_id: Optional[int] = None,
    include_eos: bool = True,
) -> torch.Tensor:
    """
    Build a boolean mask for right-padded sequences.

    tokens: [..., L]
    Returns: [..., L] bool, True for positions counted in sequence likelihood.

    - Always masks PAD.
    - If eos_id is provided, masks tokens strictly after the first EOS.
      Optionally includes the EOS token itself.
    """
    # positions that are not pad
    not_pad = tokens.ne(pad_id)

    if eos_id is None:
        return not_pad

    is_eos = tokens.eq(eos_id)
    # eos_cum[..., t] = number of EOS seen up to position t
    eos_cum = is_eos.cumsum(dim=-1)

    if include_eos:
        # keep positions up to and including first EOS (if EOS exists)
        before_or_at_eos = eos_cum.le(1)
    else:
        # keep positions strictly before first EOS
        before_or_at_eos = eos_cum.eq(0)

    return not_pad & before_or_at_eos


# ----------------------------
# Candidate extraction (from your stored top-32 tensors)
# ----------------------------

def extract_top1_sequences_and_logp(
    answer_ids_top32: torch.Tensor | List[torch.Tensor],
    answer_logp_top32: torch.Tensor | List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor] | List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    If tensors:
      answer_ids_top32:  [B, K, L, 32] or [K, L, 32]
      answer_logp_top32: same
      Returns:
        cand_tokens:        [B, K, L] (or [K, L])
        teacher_token_logp: [B, K, L] (or [K, L])

    If lists (preferred for your sparse setting):
      answer_ids_top32[b]:  [K_b, L_b, 32]
      answer_logp_top32[b]: [K_b, L_b, 32]
      Returns list of length B, each element:
        (cand_tokens_b [K_b, L_b], teacher_token_logp_b [K_b, L_b])

    Assumption (important):
      The realized teacher answer token at each position is stored in index 0:
        cand_tokens[..., t]        = answer_ids_top32[..., t, 0]
        teacher_token_logp[..., t] = answer_logp_top32[..., t, 0]
    If your storage differs (e.g., top-32 are purely "top tokens" and the sampled token may not be included),
    you must store the realized sampled token ids separately and use those instead.
    """
    if isinstance(answer_ids_top32, list):
        assert isinstance(answer_logp_top32, list)
        out: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for ids, lp in zip(answer_ids_top32, answer_logp_top32):
            out.append((ids[..., 0], lp[..., 0]))
        return out

    # tensor path
    cand_tokens = answer_ids_top32[..., 0]
    teacher_token_logp = answer_logp_top32[..., 0]
    return cand_tokens, teacher_token_logp

# ----------------------------
# Helper: repeat/copy DynamicCache across a batch
# ----------------------------

def _repeat_dynamic_cache(base_cache: Union[Any,DynamicCache], repeat: int) -> Any:
    """
    Create a new DynamicCache with batch dimension repeated.
    We MUST allocate separate storage (repeat/clone) because the cache will be updated during forward.

    This assumes the cache exposes attributes key_cache/value_cache lists of tensors, as in Transformers DynamicCache.
    """
    if repeat == 1:
        return base_cache

    if DynamicCache is None:
        raise RuntimeError("DynamicCache is not available; install a recent transformers version.")

    new_cache = copy.deepcopy(base_cache)
    new_cache.batch_repeat_interleave(repeat)
    #new_cache = None

    return new_cache

def _get_qwen3vl_image_nums_from_input_ids(
    input_ids: torch.Tensor,
    *,
    image_token_id: int,
    vision_start_token_id: int,
) -> torch.Tensor:
    """
    Replicates Qwen3-VL logic: count images per sample by detecting <vision_start> followed by <image>.
    input_ids: [B, S]
    returns: image_nums [B] (LongTensor)
    """
    vision_start_mask = input_ids.eq(vision_start_token_id)
    image_mask = input_ids.eq(image_token_id)
    vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
    image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
    return image_nums.to(dtype=torch.long)


def _split_visual_inputs_qwen3vl_by_sample(
    *,
    input_ids_batch: torch.Tensor,                         # [B, S]
    pixel_values: Optional[torch.Tensor],                  # [sum(image_token_len), ...] or None
    image_grid_thw: Optional[torch.Tensor],                # [sum(num_images), 3] or None
    model_config: Any,
) -> Tuple[List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
    """
    Returns per-sample lists:
      pixel_values_list[b]: [image_token_len_b, ...] or None
      image_grid_thw_list[b]: [num_images_b, 3] or None

    Qwen3-VL convention (Transformers): pixel_values is concatenated across samples, and the split sizes are
    derived from image_grid_thw + image_nums. For each sample, image_token_len_b = sum_i prod(grid_thw_i).
    """
    B = input_ids_batch.shape[0]
    if pixel_values is None or image_grid_thw is None:
        return [None] * B, [None] * B

    image_nums = _get_qwen3vl_image_nums_from_input_ids(
        input_ids_batch,
        image_token_id=int(model_config.image_token_id),
        vision_start_token_id=int(model_config.vision_start_token_id),
    )  # [B]

    # Split image_grid_thw into per-sample chunks by number of images
    img_grid_chunks = list(torch.split(image_grid_thw, image_nums.tolist(), dim=0))

    # For each sample, compute how many vision tokens belong to its images
    # per-image token length = prod(t, h, w)
    pixel_lens: List[int] = []
    for chunk in img_grid_chunks:
        if chunk.numel() == 0:
            pixel_lens.append(0)
        else:
            per_image = torch.prod(chunk.to(torch.long), dim=1)  # [num_images]
            pixel_lens.append(int(per_image.sum().item()))

    # Split pixel_values along dim0 by these token-lengths
    pv_chunks = list(torch.split(pixel_values, pixel_lens, dim=0))

    pv_list: List[Optional[torch.Tensor]] = []
    grid_list: List[Optional[torch.Tensor]] = []
    for b in range(B):
        pv_b = pv_chunks[b]
        grid_b = img_grid_chunks[b]
        pv_list.append(pv_b if pv_b.numel() > 0 else None)
        grid_list.append(grid_b if grid_b.numel() > 0 else None)

    return pv_list, grid_list


def _slice_text_batch_to_single(
    batch: Dict[str, Any],
    b: int,
) -> Dict[str, Any]:
    """
    Slice batch tensors with batch dimension -> single example (batch=1).
    Does not touch non-batched visual tensors.
    """
    single: Dict[str, Any] = {}
    for k, v in batch.items():
        if v is None:
            continue
        if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == batch["input_ids"].shape[0]:
            single[k] = v[b : b + 1]
        else:
            # keep as-is (will be overridden for visual keys later if needed)
            single[k] = v
    return single


def _compact_right_pad_or_left_pad(
    input_ids_1: torch.Tensor,       # [1, S]
    attention_mask_1: torch.Tensor,  # [1, S]
    other_seq_tensors: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Remove pure padding columns by slicing to [start:end] where attention_mask==1.

    Works for both left-pad and right-pad.
    """
    assert input_ids_1.shape[0] == 1 and attention_mask_1.shape[0] == 1
    mask = attention_mask_1[0].to(dtype=torch.bool)
    if mask.any():
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        start = int(idx[0].item())
        end = int(idx[-1].item()) + 1
    else:
        # degenerate: all pad
        start, end = 0, 1

    input_ids_1 = input_ids_1[:, start:end]
    attention_mask_1 = attention_mask_1[:, start:end]

    sliced_other: Dict[str, torch.Tensor] = {}
    for k, t in other_seq_tensors.items():
        if t.dim() >= 2 and t.shape[:2] == attention_mask_1.shape[:2] and t.shape[1] >= end:
            sliced_other[k] = t[:, start:end]
        else:
            sliced_other[k] = t
    return input_ids_1, attention_mask_1, sliced_other

# ----------------------------
# Candidate weighting (retrieval similarity + optional teacher confidence)
# ----------------------------

def compute_candidate_log_weights(
    retrieval_sims: torch.Tensor,
    cand_tokens: torch.Tensor,
    teacher_token_logp: Optional[torch.Tensor],
    pad_id: int,
    eos_id: Optional[int],
    *,
    candidate_mask: Optional[torch.Tensor] = None,
    tau_retrieval: float = 1.0,
    tau_teacher: float = 1.0,
    teacher_confidence: Literal["none", "sum", "mean"] = "sum",
    include_eos_in_teacher_conf: bool = True,
) -> torch.Tensor:
    """
    Compute per-candidate log-weights log w_{b,k}.

    retrieval_sims:      [B, K] (raw similarities; may contain padding candidates)
    cand_tokens:         [B, K, L]
    teacher_token_logp:  [B, K, L] or None
    candidate_mask:      [B, K] bool, True for valid candidates. If None, inferred from finite retrieval_sims.

    teacher_confidence:
      - "none":  logw = log_softmax(retrieval_sims / tau_retrieval)
      - "sum":   logw = log_softmax(retrieval_sims / tau_retrieval) + (sum token logp)/tau_teacher
      - "mean":  same but uses mean token logp (length-normalized), reduces length bias in weights

    Returns:
      logw: [B, K] (invalid candidates set to -inf)
    """
    if retrieval_sims.dim() == 1:
        retrieval_sims = retrieval_sims.unsqueeze(0)
        cand_tokens = cand_tokens.unsqueeze(0)
        if teacher_token_logp is not None:
            teacher_token_logp = teacher_token_logp.unsqueeze(0)
        if candidate_mask is not None:
            candidate_mask = candidate_mask.unsqueeze(0)

    B, K = retrieval_sims.shape[:2]

    if candidate_mask is None:
        candidate_mask = torch.isfinite(retrieval_sims)
    else:
        candidate_mask = candidate_mask.to(dtype=torch.bool)

    # Mask invalid candidates before log_softmax
    sims = retrieval_sims / float(tau_retrieval)
    sims = sims.masked_fill(~candidate_mask, float("-inf"))
    log_alpha = F.log_softmax(sims, dim=1)  # sums to 1 over valid candidates

    if teacher_confidence == "none":
        return log_alpha

    if teacher_token_logp is None:
        raise ValueError("teacher_token_logp must be provided when teacher_confidence != 'none'.")

    # token mask for teacher confidence aggregation
    tok_mask = token_mask_right_padded(
        cand_tokens,
        pad_id=pad_id,
        eos_id=eos_id,
        include_eos=include_eos_in_teacher_conf,
    ).to(dtype=teacher_token_logp.dtype)  # [B, K, L]

    # sum logp over valid positions
    sum_logp = (teacher_token_logp * tok_mask).sum(dim=-1)  # [B, K]
    if teacher_confidence == "mean":
        denom = tok_mask.sum(dim=-1).clamp_min(1.0)          # [B, K]
        agg_logp = sum_logp / denom
    elif teacher_confidence == "sum":
        agg_logp = sum_logp
    else:
        raise ValueError(f"Unknown teacher_confidence: {teacher_confidence}")

    logw = log_alpha + (agg_logp / float(tau_teacher))
    logw = logw.masked_fill(~candidate_mask, float("-inf"))
    return logw, log_alpha

def compute_candidate_log_weights_for_cache(
    retrieval_sims: torch.Tensor,
    cand_tokens: torch.Tensor,
    teacher_token_logp: Optional[torch.Tensor],
    pad_id: int,
    eos_id: Optional[int],
    *,
    candidate_mask: Optional[torch.Tensor] = None,
    tau_retrieval: float = 1.0,
    tau_teacher: float = 1.0,
    teacher_confidence: Literal["none", "sum", "mean"] = "sum",
    include_eos_in_teacher_conf: bool = True,
    # NEW:
    normalize: bool = True,
    return_components: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Compute per-candidate log-weights.

    Default behavior (normalize=True, return_components=False):
      identical to the previous version:
        logw_i = log_softmax(retrieval_sims / tau_retrieval)_i + agg_logp_i / tau_teacher

    For caching / later temperature tuning:
      set normalize=False and return_components=True.
      Then the function returns:
        retrieval_sims_raw: [B, K]
        teacher_conf_raw:   [B, K] or None
        combined_score:     [B, K]
        candidate_mask:     [B, K]
      where
        combined_score_i = retrieval_sims_raw_i / tau_retrieval + teacher_conf_raw_i / tau_teacher
      and NO log_softmax is applied.

    Why this is the right quantity to cache:
      - For both seq-KD and MML, only relative candidate weights matter.
      - The log_softmax normalization over retrieval scores can be applied later at training runtime
        after choosing top-K and temperatures.
      - Duplicate-answer merging should be done with logaddexp over the *unnormalized* combined_score,
        not over the normalized retrieval term, so that you can re-normalize after top-K / tau changes.

    Shapes:
      retrieval_sims:      [B, K] or [K]
      cand_tokens:         [B, K, L] or [K, L]
      teacher_token_logp:  [B, K, L] or [K, L] or None
      candidate_mask:      [B, K] or [K] or None

    Returns:
      if return_components=False:
        logw: [B, K] (or [1, K] if input was [K])

      if return_components=True:
        retrieval_sims_raw: [B, K]
        teacher_conf_raw:   [B, K] or None
        combined_score:     [B, K]
        candidate_mask:     [B, K]
    """
    squeeze_batch = False
    if retrieval_sims.dim() == 1:
        retrieval_sims = retrieval_sims.unsqueeze(0)
        cand_tokens = cand_tokens.unsqueeze(0)
        if teacher_token_logp is not None:
            teacher_token_logp = teacher_token_logp.unsqueeze(0)
        if candidate_mask is not None:
            candidate_mask = candidate_mask.unsqueeze(0)
        squeeze_batch = True

    if candidate_mask is None:
        candidate_mask = torch.isfinite(retrieval_sims)
    else:
        candidate_mask = candidate_mask.to(dtype=torch.bool, device=retrieval_sims.device)

    retrieval_sims_raw = retrieval_sims
    teacher_conf_raw: Optional[torch.Tensor] = None

    if teacher_confidence != "none":
        if teacher_token_logp is None:
            raise ValueError("teacher_token_logp must be provided when teacher_confidence != 'none'.")

        tok_mask = token_mask_right_padded(
            cand_tokens,
            pad_id=pad_id,
            eos_id=eos_id,
            include_eos=include_eos_in_teacher_conf,
        ).to(dtype=teacher_token_logp.dtype)  # [B, K, L]

        sum_logp = (teacher_token_logp * tok_mask).sum(dim=-1)  # [B, K]

        if teacher_confidence == "mean":
            denom = tok_mask.sum(dim=-1).clamp_min(1.0)
            teacher_conf_raw = sum_logp / denom
        elif teacher_confidence == "sum":
            teacher_conf_raw = sum_logp
        else:
            raise ValueError(f"Unknown teacher_confidence: {teacher_confidence}")

    # Unnormalized combined score: this is what you should merge/cache if you want to tune tau / top-K later.
    combined_score = retrieval_sims_raw / float(tau_retrieval)
    if teacher_conf_raw is not None:
        combined_score = combined_score + teacher_conf_raw / float(tau_teacher)

    combined_score = combined_score.masked_fill(~candidate_mask, float("-inf"))

    if return_components:
        if squeeze_batch:
            return (
                retrieval_sims_raw.squeeze(0),
                None if teacher_conf_raw is None else teacher_conf_raw.squeeze(0),
                combined_score.squeeze(0),
                candidate_mask.squeeze(0),
            )
        return retrieval_sims_raw, teacher_conf_raw, combined_score, candidate_mask

    if not normalize:
        return combined_score.squeeze(0) if squeeze_batch else combined_score

    # Backward-compatible normalized behavior.
    # Note: adding/subtracting a constant over candidates does not change seq-KD/MML after renormalization,
    # but we keep the previous semantics here.
    sims = (retrieval_sims_raw / float(tau_retrieval)).masked_fill(~candidate_mask, float("-inf"))
    log_alpha = F.log_softmax(sims, dim=1)

    logw = log_alpha
    if teacher_conf_raw is not None:
        logw = logw + teacher_conf_raw / float(tau_teacher)

    logw = logw.masked_fill(~candidate_mask, float("-inf"))
    return logw.squeeze(0) if squeeze_batch else logw

def label_candidate(
    cand_tokens: torch.Tensor,                 # [K, L] or [B, K, L]
    gt: list[str]|dict,                   # [L_gt] or [B, L_gt]
    retrieval_sims_raw: torch.Tensor,         # [K] or [B, K]
    teacher_conf_raw: Optional[torch.Tensor], # [K] or [B, K] or None
    candidate_mask: torch.Tensor,             # [K] or [B, K]
    pad_id: int,
    eos_id: Optional[int],
    score_func,
    tokenizer,
    *,
    tau_retrieval: float = 1.0,
    tau_teacher: float = 1.0,
    include_eos: bool = True,
    return_debug: bool = False,
):
    from typing import Any, Dict, List, Optional, Tuple

    squeeze_batch = False

    # Normalize shapes independently
    if cand_tokens.dim() == 2:   # [K, L] -> [1, K, L]
        cand_tokens = cand_tokens.unsqueeze(0)
        squeeze_batch = True
    elif cand_tokens.dim() != 3:
        raise ValueError(f"cand_tokens must be [K,L] or [B,K,L], got {cand_tokens.shape}")

    if retrieval_sims_raw.dim() == 1:   # [K] -> [1, K]
        retrieval_sims_raw = retrieval_sims_raw.unsqueeze(0)
    elif retrieval_sims_raw.dim() != 2:
        raise ValueError(f"retrieval_sims_raw must be [K] or [B,K], got {retrieval_sims_raw.shape}")

    if candidate_mask.dim() == 1:       # [K] -> [1, K]
        candidate_mask = candidate_mask.unsqueeze(0)
    elif candidate_mask.dim() != 2:
        raise ValueError(f"candidate_mask must be [K] or [B,K], got {candidate_mask.shape}")

    if teacher_conf_raw is not None:
        if teacher_conf_raw.dim() == 1:  # [K] -> [1, K]
            teacher_conf_raw = teacher_conf_raw.unsqueeze(0)
        elif teacher_conf_raw.dim() != 2:
            raise ValueError(f"teacher_conf_raw must be [K] or [B,K], got {teacher_conf_raw.shape}")

    B, K, L = cand_tokens.shape

    assert retrieval_sims_raw.shape[0] == B, f"retrieval_sims_raw batch mismatch: {retrieval_sims_raw.shape[0]} vs {B}"
    assert candidate_mask.shape[0] == B, f"candidate_mask batch mismatch: {candidate_mask.shape[0]} vs {B}"
    if teacher_conf_raw is not None:
        assert teacher_conf_raw.shape[0] == B, f"teacher_conf_raw batch mismatch: {teacher_conf_raw.shape[0]} vs {B}"

    keep = False
    debug: List[Dict[str, Any]] = []

    #K = min(K, 10)
    for b in range(B):
        score = retrieval_sims_raw[b] / float(tau_retrieval)  # [K]
        if teacher_conf_raw is not None:
            score = score + teacher_conf_raw[b] / float(tau_teacher)
        score = score.masked_fill(~candidate_mask[b], float("-inf"))

        merged = {}
        num_valid = 0

        for k in range(K):
            if not torch.isfinite(score[k]).item():
                continue
            num_valid += 1

            cand_mask_k = token_mask_right_padded(
                cand_tokens[b, k],
                pad_id=pad_id,
                eos_id=eos_id,
                include_eos=include_eos,
            )
            seq_k = tuple(cand_tokens[b, k][cand_mask_k].tolist())

            if seq_k in merged:
                merged[seq_k] = torch.logaddexp(merged[seq_k], score[k])
            else:
                merged[seq_k] = score[k]

        if len(merged) == 0:
            keep = False
            if return_debug:
                debug.append(
                    {
                        "top_seq": None,
                        "top_score": float("-inf"),
                        "gt_found": False,
                        "gt_score": float("-inf"),
                        "gt_rank": None,
                        "top1_is_gt": False,
                        "num_valid_candidates": 0,
                        "num_merged_candidates": 0,
                    }
                )
            continue

        # Sort merged candidates by score descending
        ranked = sorted(merged.items(), key=lambda kv: float(kv[1].item()), reverse=True)
        norm_weights  = torch.tensor([e for e in merged.values()]).softmax(dim=0)
        gt_rank = -1
        gt_weight = -1
        for idx, (seq, _) in enumerate(ranked):
            seq_text = tokenizer.decode(seq, skip_special_tokens=True)
            score =score_func(seq_text, gt)
            if score['acc']>0:
                if idx == 0 or idx ==1:
                    keep = True
                gt_rank = idx
                gt_weight = norm_weights[idx].item()
                break

    return keep, gt_rank, gt_weight, torch.stack([e[1] for e in ranked], dim=0)

def softmax_T(logits, T=1.0):
    return F.softmax(logits / T, dim=-1)

def kl_divergence(p_logit, q_logit, T=1.0, eps=1e-8):
    # KL( p || q ); both inputs are logits
    p = softmax_T(p_logit, T)
    q = softmax_T(q_logit, T)
    return torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)), dim=-1)

def sparse_kl_with_tail(
    student_logits,   # [A, V] logits from base+memory at answer positions
    teacher_ids_with,          # [A, M] teacher union token ids (=-1 means padding)
    teacher_logprob_with,         # [A, M] teacher log-probs on union tokens
    tail_logprob         # [A]    teacher log-prob for OTHER bucket
):
    eps = 1e-8

    # log p_student over full vocab
    # [B, A, V]
    log_p_s = F.log_softmax(student_logits, dim=-1)

    # teacher top-K probs (optionally renormalized with tail)
    # [B, A, K]
    p_t_topk = torch.exp(teacher_logprob_with)
    p_t_tail = torch.exp(tail_logprob).unsqueeze(-1)  # [B, A, 1]

    # Optional: renormalize (robust if teacher probs are not perfectly normalized)
    mass_topk = p_t_topk.sum(-1, keepdim=True)        # [B, A, 1]
    Z = (mass_topk + p_t_tail).clamp_min(eps)         # [B, A, 1]
    p_t_topk = p_t_topk / Z
    p_t_tail = p_t_tail / Z

    log_p_t_topk = torch.log(p_t_topk + eps)          # [B, A, K]
    log_p_t_tail = torch.log(p_t_tail + eps).squeeze(-1)  # [B, A]

    # log p_student on teacher's top-K tokens
    # teacher_ids_with: [B, A, K]
    log_p_s_topk = log_p_s.gather(dim=-1, index=teacher_ids_with)  # [B, A, K]
    p_s_topk     = torch.exp(log_p_s_topk)                         # [B, A, K]

    # student tail mass = 1 - sum_{topK} p_s
    p_s_topk_sum = p_s_topk.sum(-1)               # [B, A]
    p_s_tail     = (1.0 - p_s_topk_sum).clamp_min(eps)   # [B, A]
    log_p_s_tail = torch.log(p_s_tail)            # [B, A]

    # KL = sum_i p_t(i) [log p_t(i)-log p_s(i)] + p_tail[log p_tail-log p_s_tail]
    term_topk = (p_t_topk * (log_p_t_topk - log_p_s_topk)).sum(-1)  # [B, A]
    term_tail = p_t_tail.squeeze(-1) * (log_p_t_tail - log_p_s_tail)  # [B, A]

    kl = term_topk + term_tail    # [B, A]
    return kl.mean()              # scalar

# ----------------------------
# Optional: merge duplicate candidate sequences (within each original query)
# ----------------------------

def merge_duplicate_candidates(
    device,
    cand_tokens: torch.Tensor,        # [B, K, L]
    logw: torch.Tensor,               # [B, K]
    student_seq_logp: Optional[torch.Tensor] = None,  # [B, K], if you want to reduce compute downstream
    pad_id: int = -1,
    eos_id: Optional[int] = 151645,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Merge candidates that yield identical realized answer sequences.

    Mathematically safe if duplicate sequences have identical student_seq_logp
    (e.g., deterministic student forward; or you compute logp once per unique sequence).

    Returns:
      merged_tokens: [B, K', L] padded to original L (K' <= K, but returned as dense with padding candidates)
      merged_logw:   [B, K'] with -inf for padded candidate slots
      merged_student_seq_logp: [B, K'] if provided, else None
      merged_mask:   [B, K'] bool
    """
    if cand_tokens.dim() == 2:
        cand_tokens = cand_tokens.unsqueeze(0)
        logw = logw.unsqueeze(0)
        if student_seq_logp is not None:
            student_seq_logp = student_seq_logp.unsqueeze(0)

    B, K, L = cand_tokens.shape
    dtype = logw.dtype

    merged_tokens_list: List[torch.Tensor] = []
    merged_logw_list: List[torch.Tensor] = []
    merged_logp_list: List[torch.Tensor] = []
    merged_mask_list: List[torch.Tensor] = []

    for b in range(B):
        # dict: seq_tuple -> (logw_merged, representative_index)
        acc: Dict[Tuple[int, ...], Tuple[torch.Tensor, int]] = {}
        for k in range(K):
            if not torch.isfinite(logw[b, k]):
                continue

            seq = cand_tokens[b, k]
            # truncate at EOS and drop PAD
            mask = token_mask_right_padded(seq, pad_id=pad_id, eos_id=eos_id, include_eos=True)
            seq_trim = seq[mask].tolist()
            key = tuple(seq_trim)

            if key not in acc:
                acc[key] = (logw[b, k], k)
            else:
                prev_logw, rep_k = acc[key]
                acc[key] = (torch.logaddexp(prev_logw, logw[b, k]), rep_k)
                prev_logfirst, rep_k2 = acc[key]
        # Pack back
        keys = list(acc.keys())
        Kp = len(keys)
        if Kp == 0:
            merged_tokens = torch.full((1, L), pad_id, device=device, dtype=cand_tokens.dtype)
            merged_logw = torch.full((1,), float("-inf"), device=device, dtype=dtype)
            merged_mask = torch.zeros((1,), device=device, dtype=torch.bool)
            merged_logp = None if student_seq_logp is None else torch.zeros((1,), device=device, dtype=student_seq_logp.dtype)
        else:
            merged_tokens = torch.full((Kp, L), pad_id, device=device, dtype=cand_tokens.dtype)
            merged_logw = torch.empty((Kp,), device=device, dtype=dtype)
            merged_mask = torch.ones((Kp,), device=device, dtype=torch.bool)
            merged_logp = None if student_seq_logp is None else torch.empty((Kp,), device=device, dtype=student_seq_logp.dtype)

            for j, key in enumerate(keys):
                lw, rep_k = acc[key]
                merged_logw[j] = lw
                # rebuild padded tokens
                seq_j = torch.tensor(list(key), device=device, dtype=cand_tokens.dtype)
                take = min(L, seq_j.numel())
                merged_tokens[j, :take] = seq_j[:take]
                if student_seq_logp is not None:
                    merged_logp[j] = student_seq_logp[b, rep_k]

        # pad to fixed K for batching by adding -inf slots
        merged_tokens_list.append(merged_tokens)
        merged_logw_list.append(merged_logw)
        if student_seq_logp is not None:
            merged_logp_list.append(merged_logp)  # type: ignore[arg-type]
        merged_mask_list.append(merged_mask)

    # batch to max K' across batch
    maxKp = max(x.shape[0] for x in merged_tokens_list)
    out_tokens = torch.full((B, maxKp, L), pad_id, device=device, dtype=cand_tokens.dtype)
    out_logw = torch.full((B, maxKp), float("-inf"), device=device, dtype=dtype)
    out_mask = torch.zeros((B, maxKp), device=device, dtype=torch.bool)
    out_logp = None if student_seq_logp is None else torch.zeros((B, maxKp), device=device, dtype=student_seq_logp.dtype)

    for b in range(B):
        Kp = merged_tokens_list[b].shape[0]
        out_tokens[b, :Kp] = merged_tokens_list[b]
        out_logw[b, :Kp] = merged_logw_list[b]
        out_mask[b, :Kp] = merged_mask_list[b]
        if student_seq_logp is not None:
            out_logp[b, :Kp] = merged_logp_list[b]  # type: ignore[index]

    return out_tokens, out_logw, out_logp, out_mask


# ----------------------------
# Core losses: seq-KD vs MML
# ----------------------------

def seq_kd_loss_from_seq_logp(
    student_seq_logp: torch.Tensor,  # [B, K] log p_S(y_k | x0)
    logw: torch.Tensor,              # [B, K] candidate log-weights
    *,
    candidate_mask: Optional[torch.Tensor] = None,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    seq-KD in sequence form (sum-of-logs expectation):
      L = - E_{k ~ w}[ log p_S(y_k | x0) ].

    Uses normalized weights:
      w_norm = softmax(logw over valid candidates).
    """
    if student_seq_logp.dim() == 1:
        student_seq_logp = student_seq_logp.unsqueeze(0)
        logw = logw.unsqueeze(0)
        if candidate_mask is not None:
            candidate_mask = candidate_mask.unsqueeze(0)

    if candidate_mask is None:
        candidate_mask = torch.isfinite(logw)
    else:
        candidate_mask = candidate_mask.to(dtype=torch.bool)

    lw = logw.masked_fill(~candidate_mask, float("-inf"))
    w_norm = torch.softmax(lw, dim=1)  # [B, K]

    per_ex = -(w_norm * student_seq_logp).sum(dim=1)  # [B]

    if reduction == "none":
        return per_ex
    if reduction == "sum":
        return per_ex.sum()
    if reduction == "mean":
        return per_ex.mean()
    raise ValueError(f"Unknown reduction: {reduction}")


def mml_loss_from_seq_logp(
    student_seq_logp: torch.Tensor,  # [B, K] log p_S(y_k | x0)
    logw: torch.Tensor,              # [B, K] candidate log-weights
    *,
    candidate_mask: Optional[torch.Tensor] = None,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    MML-like candidate marginal (log-of-sum):
      L = -log sum_k w_k * p_S(y_k | x0).

    Implemented stably as:
      L = -( logsumexp(logw + student_seq_logp) - logsumexp(logw) )
    which corresponds to normalized weights without explicitly normalizing.
    """
    if student_seq_logp.dim() == 1:
        student_seq_logp = student_seq_logp.unsqueeze(0)
        logw = logw.unsqueeze(0)
        if candidate_mask is not None:
            candidate_mask = candidate_mask.unsqueeze(0)

    if candidate_mask is None:
        candidate_mask = torch.isfinite(logw)
    else:
        pass
        #candidate_mask = candidate_mask.to(dtype=torch.bool)

    lw = logw.masked_fill(~candidate_mask, float("-inf"))

    logZ = torch.logsumexp(lw, dim=1)  # [B]
    log_num = torch.logsumexp(lw + student_seq_logp, dim=1)  # [B]

    per_ex = -(log_num - logZ)  # [B]

    if reduction == "none":
        return per_ex
    if reduction == "sum":
        return per_ex.sum()
    if reduction == "mean":
        return per_ex.mean()
    raise ValueError(f"Unknown reduction: {reduction}")

def trim_excess_right_padding(x, pad_value=0):
    """
    Remove trailing columns that are entirely padding.

    Parameters
    ----------
    x : torch.Tensor
        2D right-padded tensor of shape [N, T].
    pad_value : scalar
        Padding value.

    Returns
    -------
    torch.Tensor
        Tensor with redundant right-padding columns removed.
    """
    if x.dim() != 2:
        raise ValueError("x must be a 2D tensor")

    # True for columns that contain at least one non-pad value
    keep_col = (x != pad_value).any(dim=0)

    # If everything is padding, return an empty-width tensor
    if not keep_col.any():
        return x[:, :0]

    last_valid_col = keep_col.nonzero(as_tuple=True)[0].max().item()
    return x[:, :last_valid_col + 1]

# ----------------------------
# Convenience wrapper: compute logw from your saved tensors, then compute loss
# ----------------------------

def compute_student_seq_logp_qwen3vl(
    *,
    model: torch.nn.Module,
    model_config,
    prompt_inputs: Dict[str, Any],     # BatchFeature-like dict, batched tensors
    sample_index: int,                # which item in the batch
    cand_tokens: torch.Tensor,        # [K, L] right-padded token ids (may include EOS)
    pad_id: int,
    eos_id: Optional[int],
    chunk_size: int = 16,
    include_eos: bool = True,
    length_norm: Literal["none", "mean"] = "none",
    detach_prompt_cache: bool = False,
) -> tuple[torch.Tensor, Cache, torch.Tensor, torch.Tensor] :
    """
    Computes log p_S(y | original prompt for sample_index) for each candidate sequence y.

    Critical Qwen3-VL detail handled:
      - In batch, pixel_values is concatenated across samples and split sizes are derived from image_grid_thw
        (see Qwen3-VL generation expansion logic).
      - We slice pixel_values/image_grid_thw to the *single* sample before the prompt forward.

    The student never sees amortized/retrieved images here; it conditions only on the original sample prompt.
    """
    if DynamicCache is None:
        raise RuntimeError("DynamicCache is required for Qwen3-VL cache reuse; please install a recent transformers.")

    device = model.device
    cand_tokens = cand_tokens.to(device)

    K, L = cand_tokens.shape
    if K == 0:
        return cand_tokens.new_zeros((0,), dtype=torch.float32)

    # ---- 0) Build a SINGLE-example prompt dict, including the correct visual slices ----
    # Slice batched text tensors first
    single = _slice_text_batch_to_single(prompt_inputs, sample_index)

    # Split concatenated visual tensors by sample
    pv_list, grid_list = _split_visual_inputs_qwen3vl_by_sample(
        input_ids_batch=prompt_inputs["input_ids"],
        pixel_values=prompt_inputs.get("pixel_values", None),
        image_grid_thw=prompt_inputs.get("image_grid_thw", None),
        model_config=model_config,
    )

    # Override visual keys with single-sample slices (or remove if None)
    pv_b = pv_list[sample_index]
    grid_b = grid_list[sample_index]
    if pv_b is not None and grid_b is not None:
        single["pixel_values"] = pv_b
        single["image_grid_thw"] = grid_b
    else:
        single.pop("pixel_values", None)
        single.pop("image_grid_thw", None)

    # Remove sequence padding columns to avoid "last token is PAD" issues
    # Keep other seq-like tensors (e.g., token_type_ids) aligned if present
    other_seq = {}
    for k, v in list(single.items()):
        if isinstance(v, torch.Tensor) and v.dim() == 2 and k not in ("input_ids", "attention_mask"):
            other_seq[k] = v
            single.pop(k)

    input_ids_1, attn_1, other_seq = _compact_right_pad_or_left_pad(
        single["input_ids"], single["attention_mask"], other_seq
    )
    single["input_ids"] = input_ids_1
    single["attention_mask"] = attn_1
    single.update(other_seq)

    # ---- 1) Prompt forward pass (single example) to get last-token logits and a DynamicCache ----
    base_cache = DynamicCache()

    try:
        out_prompt = model(
            **single,
            past_key_values=base_cache,
            use_cache=True,
            return_dict=True,
            mix_mode='mem'
        )
    except TypeError:
        # Some versions initialize DynamicCache internally if not passed
        out_prompt = model(
            **single,
            use_cache=True,
            return_dict=True,
            mix_mode='mem'
        )

    prompt_logits_last = out_prompt.logits[:, -1, :]  # [1, V]
    prompt_cache = out_prompt.past_key_values         # DynamicCache

    if detach_prompt_cache:
        try:
            for i in range(len(prompt_cache.key_cache)):
                prompt_cache.key_cache[i] = prompt_cache.key_cache[i].detach()
                prompt_cache.value_cache[i] = prompt_cache.value_cache[i].detach()
            prompt_logits_last = prompt_logits_last.detach()
        except Exception:
            pass

    # ---- 2) Score candidates in chunks, reusing the prompt cache ----
    prompt_attn = single["attention_mask"]  # [1, P]
    logp_first_dist = F.log_softmax(prompt_logits_last, dim=-1)  # [1, V]
    prompt_position_ids = out_prompt.position_ids

    mask_all = token_mask_right_padded(
        cand_tokens,
        pad_id=pad_id,
        eos_id=eos_id,
        include_eos=include_eos,
    )  # [K, L] bool
    msk_all = mask_all.to(torch.float32)

    student_seq_logp = cand_tokens.new_zeros((K,), dtype=torch.float32)
    prefix_len = prompt_cache.get_seq_length()

    prompt_cache_clone = copy.deepcopy(prompt_cache)
    prompt_cache_return =  copy.deepcopy(prompt_cache)
    for start in range(0, K, chunk_size):
        end = min(K, start + chunk_size)
        n = end - start

        tok = cand_tokens[start:end]            # [n, L]
        msk = msk_all[start:end]                # [n, L]
        if int(msk.sum().item()) == 0:
            continue

        cache_n = _repeat_dynamic_cache(prompt_cache_clone, n)

        ans_attn = msk.to(dtype=prompt_attn.dtype)  # [n, L]
        
        full_attn = torch.cat([prompt_attn.repeat(n, 1), ans_attn], dim=1)  # [n, P+L]

        # Answer forward (no pixel_values required because prompt cache already encodes vision)
        # try:
        #     out_ans = model(
        #         input_ids=tok,
        #         attention_mask=full_attn,
        #         past_key_values=cache_n,
        #         use_cache=True,
        #         return_dict=True,
        #     )
        #except Exception:
            # Fallback: some variants accept only the unprocessed attention mask
        L = tok.size(1)
        suffix_position_ids = prompt_position_ids[:,:,-1].unsqueeze(-1) +1
        suffix_position_ids = suffix_position_ids + torch.arange(L, device=device).view(1, -1)
        suffix_cache_position = torch.arange(prefix_len, prefix_len + L, device=device)
        out_ans = model(
            input_ids=tok,
            attention_mask=full_attn,
            past_key_values=cache_n,
            position_ids=suffix_position_ids,
            cache_position=suffix_cache_position,
            use_cache=False,
            return_dict=True,
            mix_mode='mem'
        )

        logits = out_ans.logits  # [n, L, V]

        # token 1 prob: from prompt-last distribution
        tok0 = tok[:, 0].clamp_min(0)
        logp0 = logp_first_dist.repeat(n, 1).gather(1, tok0.unsqueeze(1)).squeeze(1)  # [n]
        contrib0 = logp0 * msk[:, 0]

        # tokens 2..L prob: logits[:, :-1] predict tok[:, 1:]
        if L > 1:
            logp_rest_dist = F.log_softmax(logits[:, :-1, :], dim=-1)                 # [n, L-1, V]
            labels_rest = tok[:, 1:].unsqueeze(-1)                                    # [n, L-1, 1]
            logp_rest = logp_rest_dist.gather(-1, labels_rest).squeeze(-1)            # [n, L-1]
            contrib_rest = (logp_rest * msk[:, 1:]).sum(dim=-1)                       # [n]
        else:
            contrib_rest = tok.new_zeros((n,), dtype=torch.float32)

        seq_logp = contrib0 + contrib_rest  # [n]

        if length_norm == "mean":
            denom = msk.sum(dim=-1).clamp_min(1.0)
            seq_logp = seq_logp / denom

        student_seq_logp[start:end] = seq_logp
        del out_ans
        del cache_n
        del logits
        del logp_rest_dist
        del seq_logp

    return student_seq_logp, prompt_cache_return, logp_first_dist, prompt_position_ids

def build_prompt_only_masks(device, input_ids, attention_mask, label_mask, pad_id: int):
    """
    Returns:
      prompt_attention_mask: [B, S] (1 only on prompt tokens, 0 on answer+pad)
    """
    #device = input_ids.device
    B, S = input_ids.shape

    is_answer = label_mask & attention_mask.bool()          # [B, S]
    has_answer = is_answer.any(dim=1)                            # [B]

    # first index of answer tokens (only valid if has_answer)
    first_ans = is_answer.float().argmax(dim=1)                  # [B]

    # fallback: if no answer region, treat full non-pad as prompt
    nonpad_len = attention_mask.sum(dim=1)                       # [B]
    prefix_len = torch.where(has_answer, first_ans, nonpad_len)  # [B]

    pos = torch.arange(S, device=device).unsqueeze(0)            # [1, S]
    prompt_mask = (pos < prefix_len.unsqueeze(1))                # [B, S]

    # ensure we never include PAD
    prompt_attention_mask = (prompt_mask & attention_mask.bool()).long()
    return prompt_attention_mask


    
#prompt_attention_mask = build_prompt_only_masks(prefix_inputs["input_ids"], prefix_inputs["attention_mask"], label_mask, pad_id=pad_id)

def compute_seqkd_or_mml_loss(
    *,
    model: torch.nn.Module,
    model_config,
    prompt_inputs: Dict[str, torch.Tensor],                    # BatchFeature-like dict, tensors with batch dim [B, ...]
    label_mask:torch.Tensor,
    retrieval_sims_list: List[torch.Tensor],                   # len B, each [K_b]
    answer_ids_top32_list: List[torch.Tensor],                 # len B, each [K_b, L_b, 32]
    answer_logp_top32_list: List[torch.Tensor],                # len B, each [K_b, L_b, 32]
    pad_id: int,
    eos_id: Optional[int],
    mode: Literal["seqkd", "mml"] = "mml",
    merge_duplicates: bool = True,
    # weighting hyperparameters
    tau_retrieval: float = 1.0,
    tau_teacher: float = 1.0,
    teacher_confidence: Literal["none", "sum", "mean"] = "sum",
    include_eos_in_teacher_conf: bool = True,
    # student scoring
    chunk_size: int = 8,
    student_length_norm: Literal["none", "mean"] = "none",
    detach_prompt_cache: bool = False,
    # reduction
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    Same semantics as the previous version, but:
      - prompt_inputs is a batched BatchFeature-like dict (tensors with leading batch dim B).
      - teacher candidates (ids/logps) remain as per-example lists to avoid over-padding.

    Student is always conditioned on the ORIGINAL query image+question in prompt_inputs[b].
    Candidate answers come from amortized queries (retrieved images), weighted by retrieval similarity
    (and optionally teacher confidence). Amortized images are never fed to the student.

    Requires helper functions (unchanged from earlier):
      - extract_top1_sequences_and_logp (list-supporting)
      - compute_candidate_log_weights
      - merge_duplicate_candidates
      - compute_student_seq_logp_qwen3vl
      - seq_kd_loss_from_seq_logp / mml_loss_from_seq_logp
    """
    # Infer batch size from input_ids
    assert "input_ids" in prompt_inputs and "attention_mask" in prompt_inputs
    B = int(prompt_inputs["input_ids"].shape[0])
    prompt_attention_mask = build_prompt_only_masks(model.device, prompt_inputs["input_ids"], prompt_inputs["attention_mask"], label_mask, pad_id=pad_id)
    prompt_inputs['attention_mask'] = prompt_attention_mask
    assert len(retrieval_sims_list) == B
    assert len(answer_ids_top32_list) == B
    assert len(answer_logp_top32_list) == B

    # Extract top-1 realized sequences (index 0) per example
    extracted = extract_top1_sequences_and_logp(answer_ids_top32_list, answer_logp_top32_list)
    assert isinstance(extracted, list)

    losses: List[torch.Tensor] = []

    for b in range(B):
        retrieval_sims = retrieval_sims_list[b].softmax(-1)         # [K_b]
        cand_tokens, teacher_token_logp = extracted[b]              # [K_b, L_b], [K_b, L_b]
        cand_tokens = cand_tokens
        teacher_token_logp = teacher_token_logp

        Kb = int(retrieval_sims.shape[0])
        if Kb == 0:
            print('KB == 0')
            losses.append(prompt_inputs["input_ids"].new_zeros((), dtype=torch.float32))
            continue

        candidate_mask = torch.isfinite(retrieval_sims)

        logw = compute_candidate_log_weights(
            retrieval_sims=retrieval_sims.unsqueeze(0),                 # [1, K]
            cand_tokens=cand_tokens.unsqueeze(0),                       # [1, K, L]
            teacher_token_logp=teacher_token_logp.unsqueeze(0),         # [1, K, L]
            pad_id=pad_id,
            eos_id=eos_id,
            candidate_mask=candidate_mask.unsqueeze(0),
            tau_retrieval=tau_retrieval,
            tau_teacher=tau_teacher,
            teacher_confidence=teacher_confidence,
            include_eos_in_teacher_conf=include_eos_in_teacher_conf,
        ).squeeze(0)  # [K]

        logw = logw.masked_fill(~candidate_mask, float("-inf"))

        if merge_duplicates:
            cand_tokens_m, logw_m, _, cand_mask_m = merge_duplicate_candidates(
                device=model.device,
                cand_tokens=cand_tokens.unsqueeze(0),
                logw=logw.unsqueeze(0),
                student_seq_logp=None,
                pad_id=pad_id,
                eos_id=eos_id,
            )
            cand_tokens = cand_tokens_m.squeeze(0)            # [K', L]
            logw = logw_m.squeeze(0)                          # [K']
            candidate_mask = cand_mask_m.squeeze(0)            # [K']
        else:
            candidate_mask = torch.isfinite(logw)

        if int(candidate_mask.sum().item()) == 0:
            losses.append(prompt_inputs["input_ids"].new_zeros((), dtype=torch.float32))
            print("Empty candidate batch")
            continue

        # Student conditioned on ORIGINAL image+question for sample b
        student_seq_logp = compute_student_seq_logp_qwen3vl(
            model=model,
            model_config=model_config,
            prompt_inputs=prompt_inputs,
            sample_index=b,
            cand_tokens=cand_tokens,
            pad_id=pad_id,
            eos_id=eos_id,
            chunk_size=chunk_size,
            include_eos=True,
            length_norm=student_length_norm,
            detach_prompt_cache=detach_prompt_cache,
        )  # [K']

        if mode == "seqkd":
            loss_b = seq_kd_loss_from_seq_logp(
                student_seq_logp=student_seq_logp.unsqueeze(0),
                logw=logw.unsqueeze(0),
                candidate_mask=candidate_mask.unsqueeze(0),
                reduction="none",
            ).squeeze(0)
        elif mode == "mml":
            loss_b = mml_loss_from_seq_logp(
                student_seq_logp=student_seq_logp.unsqueeze(0),
                logw=logw.unsqueeze(0),
                candidate_mask=candidate_mask.unsqueeze(0),
                reduction="none",
            ).squeeze(0)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        losses.append(loss_b)

    loss_vec = torch.stack(losses, dim=0)  # [B]
    if reduction == "none":
        return loss_vec
    if reduction == "sum":
        return loss_vec.sum()
    if reduction == "mean":
        return loss_vec.mean()
    raise ValueError(f"Unknown reduction: {reduction}")

def marginalize_first_token_logp_topk(
    answer_ids_top32: torch.Tensor,         # [K, L, 32] or [K, 32]  (token ids)
    answer_logp_top32: torch.Tensor,        # [K, L, 32] or [K, 32]  (log-probs)
    candidate_log_weights: torch.Tensor,    # [K]
    candidate_mask: torch.Tensor,           # [K] bool
    top_k: Optional[int] = None,
    tail_mass: Optional[torch.Tensor] = None,  # [K, L] or [K], probability mass (NOT log) outside top-32
    sort_desc: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Marginalize the first-token teacher distribution over the top-k valid candidates.

    Steps:
      1) apply candidate_mask
      2) keep top-k candidates by candidate_log_weights
      3) normalize candidate weights over the selected candidates
      4) form mixture over each candidate's first-token top-32 distribution
      5) aggregate duplicate token ids by logaddexp

    Returns:
      merged_token_ids:   [U] unique first-token ids after aggregation
      merged_log_probs:   [U] log p(token) after marginalization
      merged_tail_logp:   scalar tensor or None
                          = log of total tail mass after marginalization
                          (mass outside the returned merged_token_ids)

    Notes:
      - This function uses ONLY the first decoding position.
      - If `answer_ids_top32` / `answer_logp_top32` are [K, L, 32], position 0 is used.
      - `tail_mass` should be probability mass, not log-mass.
      - If you do not want tail handling, pass `tail_mass=None`.
    """
    device = candidate_log_weights.device
    candidate_mask = candidate_mask.to(torch.bool)

    if answer_ids_top32.dim() == 3:
        first_ids = answer_ids_top32[:, 0, :]      # [K, 32]
        first_logp = answer_logp_top32[:, 0, :]    # [K, 32]
        first_tail = tail_mass[:, 0] if tail_mass is not None else None  # [K]
    elif answer_ids_top32.dim() == 2:
        first_ids = answer_ids_top32               # [K, 32]
        first_logp = answer_logp_top32             # [K, 32]
        first_tail = tail_mass if tail_mass is not None else None        # [K]
    else:
        raise ValueError("answer_ids_top32 must have shape [K, L, 32] or [K, 32].")

    valid_idx = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        empty_ids = torch.empty(0, dtype=first_ids.dtype, device=device)
        empty_logp = torch.empty(0, dtype=first_logp.dtype, device=device)
        empty_tail = torch.tensor(float("-inf"), dtype=first_logp.dtype, device=device) if tail_mass is not None else None
        return empty_ids, empty_logp, empty_tail

    # top-k AFTER applying candidate_mask
    valid_logw = candidate_log_weights[valid_idx]
    if top_k is not None and top_k < valid_idx.numel():
        _, top_pos = torch.topk(valid_logw, k=top_k, dim=0)
        sel_idx = valid_idx[top_pos]
    else:
        sel_idx = valid_idx

    sel_logw = candidate_log_weights[sel_idx]          # [K']
    sel_logw = sel_logw - torch.logsumexp(sel_logw, dim=0)  # normalize over selected candidates

    sel_ids = first_ids[sel_idx]                       # [K', 32]
    sel_first_logp = first_logp[sel_idx]               # [K', 32]

    # mixture contribution for each sparse token entry
    # log p_mix(token entry) = log w(candidate) + log p_teacher(token | candidate)
    contrib_logp = sel_logw[:, None] + sel_first_logp  # [K', 32]

    flat_ids = sel_ids.reshape(-1)                     # [K'*32]
    flat_logp = contrib_logp.reshape(-1)               # [K'*32]

    # aggregate duplicate token ids by logaddexp
    uniq_ids, inverse = torch.unique(flat_ids, sorted=False, return_inverse=True)
    merged_logp = torch.full(
        (uniq_ids.numel(),),
        float("-inf"),
        dtype=flat_logp.dtype,
        device=device,
    )
    for j in range(flat_logp.numel()):
        merged_logp[inverse[j]] = torch.logaddexp(merged_logp[inverse[j]], flat_logp[j])

    merged_tail_logp = None
    if first_tail is not None:
        sel_tail = first_tail[sel_idx].clamp_min(0.0)  # [K']
        if torch.any(sel_tail > 0):
            merged_tail_logp = torch.logsumexp(
                sel_logw + torch.log(sel_tail.clamp_min(torch.finfo(sel_tail.dtype).tiny)),
                dim=0,
            )
        else:
            merged_tail_logp = torch.tensor(float("-inf"), dtype=flat_logp.dtype, device=device)

    if sort_desc and merged_logp.numel() > 0:
        order = torch.argsort(merged_logp, descending=True)
        uniq_ids = uniq_ids[order]
        merged_logp = merged_logp[order]

    return uniq_ids, merged_logp, merged_tail_logp

def merge_candidates_with_temperature(
    cand_tokens: torch.Tensor,              # [K, L]
    retrieval_sims_raw: torch.Tensor,       # [K]
    teacher_conf_raw: Optional[torch.Tensor],  # [K] or None
    candidate_mask: torch.Tensor,           # [K] bool
    pad_id: int,
    eos_id: Optional[int],
    *,
    tau_retrieval: float = 1.0,
    tau_teacher: float = 1.0,
    top_k: Optional[int] = None,
    include_eos: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-example merge function.

    Steps:
      1) recompute unnormalized score under current temperatures
      2) optional top-k on that score
      3) merge duplicate answer sequences with logaddexp
      4) normalize to logw

    Returns:
      merged_tokens: [K_uniq, L]
      logw:          [K_uniq]   (normalized log-weights)
      merged_score:  [K_uniq]   (unnormalized merged scores, useful for debugging / re-normalization)
    """
    device = cand_tokens.device
    dtype = retrieval_sims_raw.dtype
    K, L = cand_tokens.shape

    candidate_mask = candidate_mask.to(device=device, dtype=torch.bool)

    # 1) recompute unnormalized score under current temperatures
    score = retrieval_sims_raw.to(device=device, dtype=dtype) / float(tau_retrieval)
    if teacher_conf_raw is not None:
        score = score + teacher_conf_raw.to(device=device, dtype=dtype) / float(tau_teacher)

    score = score.masked_fill(~candidate_mask, float("-inf"))

    # 2) optional top-k BEFORE merging/normalization
    valid_idx = torch.nonzero(torch.isfinite(score), as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        return (
            torch.full((0, L), pad_id, device=device, dtype=cand_tokens.dtype),
            torch.empty((0,), device=device, dtype=dtype),
            torch.empty((0,), device=device, dtype=dtype),
        )

    if top_k is not None and valid_idx.numel() > top_k:
        topk_local = torch.topk(score[valid_idx], k=top_k, dim=0, largest=True, sorted=False).indices
        keep_idx = valid_idx[topk_local]
    else:
        keep_idx = valid_idx

    kept_tokens = cand_tokens[keep_idx]   # [K_keep, L]
    kept_score = score[keep_idx]          # [K_keep]

    # 3) merge duplicates with logaddexp on UNNORMALIZED scores
    seq_mask = token_mask_right_padded(
        kept_tokens,
        pad_id=pad_id,
        eos_id=eos_id,
        include_eos=include_eos,
    )  # [K_keep, L] bool

    merged_map: Dict[Tuple[int, ...], Tuple[torch.Tensor, torch.Tensor]] = {}
    # value = (merged_score, representative_padded_tokens)

    if not candidate_mask.all():
        pass
    for i in range(kept_tokens.shape[0]):
        seq_i = kept_tokens[i]
        mask_i = seq_mask[i]
        seq_trim = seq_i[mask_i]
        key = tuple(seq_trim.tolist())

        if key not in merged_map:
            rep = torch.full((L,), pad_id, device=device, dtype=cand_tokens.dtype)
            if seq_trim.numel() > 0:
                rep[: seq_trim.numel()] = seq_trim
            merged_map[key] = (kept_score[i], rep)
        else:
            prev_score, rep = merged_map[key]
            merged_map[key] = (torch.logaddexp(prev_score, kept_score[i]), rep)

    K_uniq = len(merged_map)
    merged_tokens = torch.full((K_uniq, L), pad_id, device=device, dtype=cand_tokens.dtype)
    merged_score = torch.empty((K_uniq,), device=device, dtype=dtype)

    for j, (_, (s, rep)) in enumerate(merged_map.items()):
        merged_tokens[j] = rep
        merged_score[j] = s

    # 4) normalize after top-k + merge
    logZ = torch.logsumexp(merged_score, dim=0)
    logw = merged_score - logZ

    return merged_tokens, logw, merged_score

def compute_loss_premerged_with_ce(
    *,
    model: torch.nn.Module,
    model_config,
    prompt_inputs: Dict[str, torch.Tensor],                    # BatchFeature-like dict, tensors with batch dim [B, ...]
    label_mask:torch.Tensor,
    answer_ids,
    batch_cand_tokens, ret_scores, sum_cand_logps, candidate_mask,
    pad_id: int,
    eos_id: Optional[int],
    mode: Literal["seqkd", "mml"] = "mml",
    merge_duplicates: bool = True,
    # weighting hyperparameters
    tau_retrieval: float = 1.0,
    tau_teacher: float = 1.0,
    top_k=10,
    teacher_confidence: Literal["none", "sum", "mean"] = "sum",
    include_eos_in_teacher_conf: bool = True,
    # student scoring
    chunk_size: int = 8,
    student_length_norm: Literal["none", "mean"] = "none",
    detach_prompt_cache: bool = False,
    # reduction
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    Same semantics as the previous version, but:
      - prompt_inputs is a batched BatchFeature-like dict (tensors with leading batch dim B).
      - teacher candidates (ids/logps) remain as per-example lists to avoid over-padding.

    Student is always conditioned on the ORIGINAL query image+question in prompt_inputs[b].
    Candidate answers come from amortized queries (retrieved images), weighted by retrieval similarity
    (and optionally teacher confidence). Amortized images are never fed to the student.

    Requires helper functions (unchanged from earlier):
      - extract_top1_sequences_and_logp (list-supporting)
      - compute_candidate_log_weights
      - merge_duplicate_candidates
      - compute_student_seq_logp_qwen3vl
      - seq_kd_loss_from_seq_logp / mml_loss_from_seq_logp
    """
    # Infer batch size from input_ids
    assert "input_ids" in prompt_inputs and "attention_mask" in prompt_inputs
    device = model.device
    B = int(prompt_inputs["input_ids"].shape[0])
    prompt_attention_mask = build_prompt_only_masks(model.device, prompt_inputs["input_ids"], prompt_inputs["attention_mask"], label_mask, pad_id=pad_id)
    #full_attn_mask = prompt_inputs['attention_mask']
    prompt_inputs['attention_mask'] = prompt_attention_mask


    assert len(batch_cand_tokens) == B
    assert len(sum_cand_logps) == B

    # Extract top-1 realized sequences (index 0) per example
    kd_losses: List[torch.Tensor] = []
    ce_losses: List[torch.Tensor] = []

    for b in range(B):
        sample_cand_token = batch_cand_tokens[b]
        sample_ret_score = ret_scores[b]
        sample_candidate_mask = candidate_mask[b]
        sample_teacher_conf = sum_cand_logps[b]

        cand_tokens, logw, merged_score = merge_candidates_with_temperature(
            cand_tokens=sample_cand_token,
            retrieval_sims_raw=sample_ret_score,
            teacher_conf_raw=sample_teacher_conf,
            candidate_mask=sample_candidate_mask,
            pad_id=pad_id,
            eos_id=eos_id,
            tau_retrieval=tau_retrieval,
            tau_teacher=tau_teacher,
            top_k=top_k
        )
        #candidate_mask = None
        if len(cand_tokens) == 0:
            kd_losses.append(prompt_inputs["input_ids"].new_zeros((), dtype=torch.float32))
            ce_losses.append(prompt_inputs["input_ids"].new_zeros((), dtype=torch.float32))
            print("Empty candidate batch")
            continue

        cand_tokens = trim_excess_right_padding(cand_tokens, pad_id)

        # Student conditioned on ORIGINAL image+question for sample b
        student_seq_logp, prompt_cache_return, logp_first_dist, prompt_position_ids = compute_student_seq_logp_qwen3vl(
            model=model,
            model_config=model_config,
            prompt_inputs=prompt_inputs,
            sample_index=b,
            cand_tokens=cand_tokens,
            pad_id=pad_id,
            eos_id=eos_id,
            chunk_size=chunk_size,
            include_eos=True,
            length_norm=student_length_norm,
            detach_prompt_cache=detach_prompt_cache,
        )  # [K']

        prefix_len = prompt_cache_return.get_seq_length()
        current_gt = answer_ids[b]
        current_gt = trim_excess_right_padding(current_gt.unsqueeze(0), pad_id)
        #single = _slice_text_batch_to_single(prompt_inputs, b)

        L = current_gt.size(1)
        suffix_position_ids = prompt_position_ids[:,:,-1].unsqueeze(-1) +1
        suffix_position_ids = suffix_position_ids + torch.arange(L, device=device).view(1, -1)
        suffix_cache_position = torch.arange(prefix_len, prefix_len + L, device=device)

        ce_loss = compute_ce_loss_from_prefix_cache(
            model=model,
            ans_ids_1=current_gt,
            attention_mask_1=torch.ones_like(current_gt),
            ans_labels_1=current_gt,
            suffix_position_ids=suffix_position_ids,
            suffix_cache_position=suffix_cache_position,
            prefix_cache=prompt_cache_return,
            logp_first_dist=logp_first_dist,
            )

        if mode == "seqkd":
            loss_b = seq_kd_loss_from_seq_logp(
                student_seq_logp=student_seq_logp.unsqueeze(0),
                logw=logw.unsqueeze(0),
                candidate_mask=candidate_mask,
                reduction="none",
            ).squeeze(0)
        elif mode == "mml":
            loss_b = mml_loss_from_seq_logp(
                student_seq_logp=student_seq_logp.unsqueeze(0),
                logw=logw.unsqueeze(0),
                candidate_mask=candidate_mask,
                reduction="none",
            ).squeeze(0)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        kd_losses.append(loss_b)
        ce_losses.append(ce_loss)
        del prompt_cache_return

    kd_losses = torch.stack(kd_losses, dim=0)  # [B]
    ce_losses = torch.stack(ce_losses, dim=0)
    if reduction == "none":
        return kd_losses, ce_losses
    if reduction == "sum":
        return kd_losses.sum(), ce_losses.sum()
    if reduction == "mean":
        return kd_losses.mean(), ce_losses.mean()
    raise ValueError(f"Unknown reduction: {reduction}")


def compute_ce_loss_from_prefix_cache(
    *,
    model: torch.nn.Module,
    # single-sample (batch=1) tensors for FULL sequence including answer, already compacted to remove pure padding columns
    ans_ids_1: torch.Tensor,        # [1, S]
    attention_mask_1: torch.Tensor,   # [1, S]
    ans_labels_1: torch.Tensor,           # [1, S] with -100 for ignored tokens
    # prefix cache built from tokens input_ids_1[:, :prefix_len]
    suffix_position_ids,
    suffix_cache_position,
    prefix_cache: DynamicCache,
    logp_first_dist: torch.Tensor, # [1, V] logits for the last prefix position (i.e., position prefix_len-1)
    prefix_len: Optional[int] = None, # if None, inferred as first target idx
    # options
    reduction: Literal["mean", "sum"] = "mean",
    length_norm: Literal["none", "mean"] = "mean",
) -> torch.Tensor:
    """
    Computes teacher-forced CE over target (labels != -100) tokens using cached prefix.
    The model never re-encodes the prefix (including vision); it only runs the answer segment.

    Important alignment requirement:
      prefix_cache and prefix_last_logits MUST correspond to the prefix up to first target token.
    """
    device = model.device
    assert ans_ids_1.shape[0] == 1

    L = ans_ids_1.shape[1]

    # 1) Score the FIRST supervised token using prefix_last_logits (predicts token at position t_start)
    first_token = ans_ids_1[:, 0]                                  # [1]
    first_logp = logp_first_dist.gather(1, first_token.unsqueeze(1)).squeeze(1)  # [1]
    nll_first = -(first_logp)  # [1]

    # If L==1, we're done
    if L == 1:
        denom = attention_mask_1.sum() if length_norm == "mean" else 1.0
        denom = torch.clamp(denom, min=1.0)
        loss = nll_first.sum() / denom
        return loss if reduction == "mean" else nll_first.sum()

    # 2) Score remaining tokens by continuing from prefix_cache
    # Note: passing past_key_values may mutate it; clone defensively.
    cache = copy.deepcopy(prefix_cache)

    # Build attention mask for the continuation. With past, most HF models accept either:
    #   - full mask of length prefix_len + L
    #   - or just the new tokens mask [1, L]
    # Qwen3-VL is generally safest with the full concatenated mask.
    if prefix_len is None:
        prefix_len = cache.get_seq_length()
    prefix_attn = torch.ones((1, prefix_len), device=device, dtype=attention_mask_1.dtype)
    ans_attn = attention_mask_1
    full_attn = torch.cat([prefix_attn, ans_attn], dim=1)  # [1, prefix_len + L]

    # Forward on the full answer segment tokens. We will use logits[:, :-1] to score ans_ids[:, 1:].
    out = model(
        input_ids=ans_ids_1,
        attention_mask=full_attn,
        position_ids=suffix_position_ids,
        cache_position=suffix_cache_position,
        past_key_values=cache,
        mix_mode="mem",
        use_cache=True,
        return_dict=True,
    )
    logits = out.logits  # [1, L, V]

    # logits at index j predicts token at index j+1 in ans_ids
    logp_dist = F.log_softmax(logits[:, :-1, :], dim=-1)          # [1, L-1, V]
    next_tokens = ans_ids_1[:, 1:].unsqueeze(-1)                    # [1, L-1, 1]
    next_logp = logp_dist.gather(-1, next_tokens).squeeze(-1)     # [1, L-1]
    nll_rest = -(next_logp * attention_mask_1[:, 1:])                     # [1, L-1]

    nll_total = nll_first.sum() + nll_rest.sum()

    if length_norm == "mean":
        denom = attention_mask_1.sum().clamp_min(1.0)
        loss = nll_total / denom
    else:
        loss = nll_total

    return loss
