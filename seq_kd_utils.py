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

    #new_cache = copy.deepcopy(base_cache)
    #new_cache.batch_repeat_interleave(repeat)
    new_cache = None

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
    return logw


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
        candidate_mask = candidate_mask.to(dtype=torch.bool)

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
    chunk_size: int = 8,
    include_eos: bool = True,
    length_norm: Literal["none", "mean"] = "none",
    detach_prompt_cache: bool = False,
) -> torch.Tensor:
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

    mask_all = token_mask_right_padded(
        cand_tokens,
        pad_id=pad_id,
        eos_id=eos_id,
        include_eos=include_eos,
    )  # [K, L] bool
    msk_all = mask_all.to(torch.float32)

    student_seq_logp = cand_tokens.new_zeros((K,), dtype=torch.float32)

    prompt_cache_clone = copy.deepcopy(prompt_cache)
    for start in range(0, K, chunk_size):
        end = min(K, start + chunk_size)
        n = end - start

        tok = cand_tokens[start:end]            # [n, L]
        msk = msk_all[start:end]                # [n, L]
        if int(msk.sum().item()) == 0:
            continue

        cache_n = _repeat_dynamic_cache(prompt_cache_clone, n)

        ans_attn = msk.to(dtype=prompt_attn.dtype)  # [n, L]
        #full_attn = torch.cat([prompt_attn.repeat(n, 1), ans_attn], dim=1)  # [n, P+L]

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
        out_ans = model(
            input_ids=tok,
            attention_mask=ans_attn,
            past_key_values=cache_n,
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

    return student_seq_logp

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