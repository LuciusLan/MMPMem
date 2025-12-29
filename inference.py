from copy import deepcopy
import torch
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration

# Choose where to read hidden states for memory
LAYER_IDX = 29  # e.g., around 70% depth

def softmax_T(logits, T=1.0):
    return F.softmax(logits / T, dim=-1)

def combine_logits(base_logits, mem_logits, mode="residual", eta_or_lambda=0.5, temperature=1.0):
    """
    base_logits, mem_logits: [B, V]
    mode: "residual" or "mixture"
    eta_or_lambda: if residual -> eta (scale for memory); if mixture -> lambda (weight on base)
    returns: final logits [B, V]
    """
    if mode == "residual":
        eta = eta_or_lambda
        return base_logits + eta * mem_logits
    elif mode == "mixture":
        lam = eta_or_lambda
        p_base = softmax_T(base_logits, temperature)
        p_mem  = softmax_T(mem_logits,  temperature)
        p_final = lam * p_base + (1.0 - lam) * p_mem
        return torch.log(p_final.clamp_min(1e-12))
    else:
        raise ValueError("mode must be 'residual' or 'mixture'")
    
def prob_mix(base_logits, mem_logits, lam=0.6):  # lam = weight on base
    pb = F.softmax(base_logits, dim=-1)
    pm = F.softmax(mem_logits, dim=-1)
    p  = lam * pb + (1.0 - lam) * pm
    return torch.log(p.clamp_min(1e-32))  # return logits for sampling

def top_k_top_p_filtering(logits, top_k=None, top_p=None):
    """ Inplace filtering of logits for top-k / nucleus sampling. """
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        cutoff = values[..., -1, None]
        logits = torch.where(logits < cutoff, torch.full_like(logits, float('-inf')), logits)
    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = probs.cumsum(dim=-1)
        mask = cumprobs > top_p
        # Shift mask to keep at least one token
        mask[..., 0] = False
        filtered = torch.full_like(sorted_logits, float('-inf'))
        filtered[~mask] = sorted_logits[~mask]
        # map back
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(dim=-1, index=sorted_idx, src=filtered)
    return logits

@torch.no_grad()
def generate_with_memory(
    model: Qwen3VLForConditionalGeneration,
    memory,
    tokenizer,
    # input_ids,                 # [1, T] prompt that includes the question formatting
    # attention_mask,            # [1, T]
    # image,                     # preprocessed image tensor for the base MLLM (first step only)
    inputs,
    max_new_tokens=16,
    eos_token_id=None,
    temperature=1.0,
    top_k=None,
    top_p=None,
    mode="residual",           # "residual" or "mixture"
    eta_or_lambda=0.5,         # eta if residual; lambda if mixture
    gate_module=None,          # optional: small gate(h) -> scalar in [0,1] to replace eta/lambda
):
    """
    Greedy/sampling decode with memory fusion at every step.
    Returns: generated token ids [1, N_new] and the decoded string.
    """
    device = model.device
    past_kv = None
    first = True
    generated = []


    cur_input_ids = inputs['input_ids']
    cur_attn_mask = inputs['attention_mask']
    cur_grid = inputs['image_grid_thw']
    cur_pixel = inputs['pixel_values']

    out = model(
        **inputs,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True
    )
    kv_snapshot = deepcopy(out.past_key_values)
    last_base_logits = out.logits[:, -1, :]               # [1,V]
    h_last = out.hidden_states[LAYER_IDX][:, -1, :]  # [1,H]
    mem_logits = memory(h_last)              # [1,V]
    #fused_logits = combine_logits(last_base_logits, mem_logits, eta_or_lambda=eta_or_lambda)

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    # ---- First token ----

    #next_token_mem = torch.argmax(mem_logits, dim=-1)  # greedy by default; or self._sample_from_logits(fused_logits)
    next_token_base = torch.argmax(last_base_logits, -1)
    mixed_prob = prob_mix(last_base_logits, mem_logits, eta_or_lambda)
    next_token_mixed = torch.argmax(mixed_prob, dim=-1)

    generated = [next_token_mixed.item()]
    baselm_gen = [next_token_base.item()]

    # Update for step decoding
    step_input_ids = next_token_mixed.unsqueeze(0)  # [1,1]
    step_input_ids_base = next_token_base.unsqueeze(0)
    step_attn = torch.ones_like(step_input_ids)

    # Some MLLMs need images every step; otherwise pass None
    #step_images = None if self.images_first_pass_only else images_first
    past_kv = deepcopy(kv_snapshot)
    # ---- Decode loop ----
    for _ in range(max_new_tokens - 1):
        out = model.forward(
            input_ids=step_input_ids,
            attention_mask=step_attn,
            use_cache=True,
            past_key_values=past_kv,
            output_hidden_states=True,
            return_dict=True
        )
        past_kv = out.past_key_values
        base_logits = out.logits[:, -1, :]                      # [1,V]
        h_last = out.hidden_states[LAYER_IDX][:, -1, :]    # [1,H]
        mem_logits = memory(h_last)                # [1,V]
        #fused_logits = combine_logits(base_logits, mem_logits, eta_or_lambda=eta_or_lambda)      # [1,V]

        # sample or greedy
        # next_token = self._sample_from_logits(fused_logits)
        next_token_base = torch.argmax(base_logits, -1)
        mixed_prob = prob_mix(base_logits, mem_logits, eta_or_lambda)
        next_token_mixed = torch.argmax(mixed_prob, dim=-1)

        generated.append(next_token_mixed.item())
        #baselm_gen.append(next_token_base.item())
        if next_token_mixed.item() == eos_token_id:
            break

        step_input_ids = next_token_mixed.unsqueeze(0)  # [1,1]
        step_attn = torch.ones_like(step_input_ids)

    past_kv = deepcopy(kv_snapshot)
    for _ in range(max_new_tokens - 1):
        out = model.forward(
            input_ids=step_input_ids_base,
            attention_mask=step_attn,
            use_cache=True,
            past_key_values=past_kv,
            output_hidden_states=True,
            return_dict=True
        )
        past_kv = out.past_key_values
        base_logits = out.logits[:, -1, :]                      # [1,V]
        #h_last = out.hidden_states[LAYER_IDX][:, -1, :]    # [1,H]
        #mem_logits = memory(h_last)                # [1,V]
        #fused_logits = combine_logits(base_logits, mem_logits, eta_or_lambda=eta_or_lambda)      # [1,V]

        # sample or greedy
        # next_token = self._sample_from_logits(fused_logits)
        next_token_base = torch.argmax(base_logits, -1)
        #mixed_prob = prob_mix(base_logits, mem_logits, eta_or_lambda)
        #next_token_mixed = torch.argmax(mixed_prob, dim=-1)

        #generated.append(next_token_mixed.item())
        baselm_gen.append(next_token_base.item())
        if next_token_base.item() == eos_token_id:
            break

        step_input_ids_base = next_token_base.unsqueeze(0)  # [1,1]
        step_attn = torch.ones_like(step_input_ids_base)


    #gen_ids = torch.stack(generated, dim=1) if generated else torch.empty(1, 0, dtype=torch.long, device=device)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    text_baselm = tokenizer.decode(baselm_gen, skip_special_tokens=True)
    return generated, text, text_baselm
