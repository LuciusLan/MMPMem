from typing import List, Dict, Optional, Tuple, Literal, Callable, Any, Iterable, Union
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, is_torchdynamo_compiling
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast, Qwen3VLModelOutputWithPast

from seq_kd_utils import _slice_text_batch_to_single, _split_visual_inputs_qwen3vl_by_sample, _compact_right_pad_or_left_pad, token_mask_right_padded, _repeat_dynamic_cache, build_prompt_only_masks, merge_candidates_with_temperature, marginalize_first_token_logp_topk, trim_excess_right_padding, seq_kd_loss_from_seq_logp, mml_loss_from_seq_logp, sparse_kl_with_tail

class Qwen3VLCausalLMOutputWithPast_Pos(Qwen3VLCausalLMOutputWithPast):
    def __init__(self, *args, **kwargs):
        position_ids =  kwargs.pop("position_ids", None)
        super().__init__(*args, **kwargs)
        self["position_ids"] = position_ids

class SwiGLU(nn.Module):
    def __init__(self, d_in, d_up):
        super().__init__()
        # in -> 2*up for gate/value, up -> in
        self.w_in = nn.Linear(d_in, 2 * d_up, bias=False)
        self.w_out = nn.Linear(d_up, d_in, bias=False)

        nn.init.kaiming_uniform_(self.w_in.weight, nonlinearity="relu")
        #nn.init.zeros_(self.w_in.bias)
        nn.init.xavier_uniform_(self.w_out.weight)
        #nn.init.zeros_(self.w_out.bias)

    def forward(self, x):
        x = self.w_in(x)                     # [B, 2u]
        v, g = x.chunk(2, dim=-1)            # value, gate
        return self.w_out(F.silu(g) * v)     # [B, d_in]

class MemBlock(nn.Module):
    def __init__(self, d_in, d_up, norm='rms'):
        super().__init__()
        self.norm = nn.LayerNorm(d_in) if norm == 'ln' else RMSNorm(d_in)
        self.ff = SwiGLU(d_in, d_up)

    def forward(self, x):
        h = self.norm(x)
        h = self.ff(h)
        return x + h

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return self.scale * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

class MemoryMLP(nn.Module):
    """
    Residual MLP memory that takes a hidden state h [B, d]
    and outputs residual logits [B, V] to be ADDED to base logits.
    """
    # TODO:
    # Bind LM head with base LM head

    def __init__(self, d_in=4096, d_up=8192, num_blocks=6, vocab_size=151936,
                 head='dense', head_rank=4096, norm='rms'):
        super().__init__()
        self.blocks = nn.ModuleList([MemBlock(d_in, d_up, norm=norm) for _ in range(num_blocks)])
        self.out_norm = RMSNorm(head_rank)
        if head == 'dense':
            self.head = nn.Linear(d_in, vocab_size, bias=False)
            #nn.init.zeros_(self.head.bias)
            nn.init.zeros_(self.head.weight)
        elif head == 'factorized':
            self.proj = nn.Linear(d_in, head_rank, bias=False)
            self.out = nn.Linear(head_rank, vocab_size, bias=True)
        else:
            raise ValueError("head must be 'dense' or 'factorized'")
        self.head_type = head



    def forward(self, h):
        # h: [B, d_in] (e.g., last-token hidden at chosen layer)
        for blk in self.blocks:
            h = blk(h)
        h = self.out_norm(h)
        if self.head_type == 'dense':
            return self.head(h)              # residual logits
        else:
            z = self.proj(h)
            return self.out(z)               # residual logits

class WrappedLM(nn.Module, GenerationMixin):
    _is_stateful = False

    def __init__(self, base_model: Qwen3VLForConditionalGeneration, memory: MemoryMLP, config, processor, layer_idx_for_mem=32):
        super().__init__()
        self.base_lm = base_model
        self.base_model = base_model.model
        self.lm_head = base_model.lm_head
        self.memory = memory
        self.layer_idx_for_mem = layer_idx_for_mem

        self.config = config
        self.processor = processor
        self.generation_config = getattr(self.base_lm, "generation_config", None)
        self.main_input_name = getattr(self.base_lm, "main_input_name", "input_ids")
        self.rope_deltas = None
        self.pad_token_id = processor.tokenizer.pad_token_id
        self.eos_token_id = processor.tokenizer.eos_token_id


    @property
    def device(self) -> torch.device:
        return self.base_model.device
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_lm.prepare_inputs_for_generation(*args, **kwargs)

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder=False,
        num_new_tokens=1,
    ):
        # keep HF default behavior: cache, attention_mask, position_ids, cache_position, etc.
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )

        # add your custom state
        if getattr(outputs, "position_ids", None) is not None:
            #model_kwargs["position_ids"] = outputs.position_ids
            next_position_id =  outputs.position_ids.clone()
            next_position_id = next_position_id[:,:,-1] +1
            next_position_id = next_position_id.unsqueeze(1)
            model_kwargs["position_ids"] = next_position_id

        return model_kwargs
    
    def forward_basic(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep=0,
        mix_mode='base',
        mix_lambda=0.4,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        image_embeds_return = None

        additional_emb = kwargs.pop("additional_emb", None)
        latent_token_mask = kwargs.pop("latent_token_mask", None)

        if pixel_values is not None:
            image_outputs = self.base_model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
            image_embeds = image_outputs.pooler_output
            deepstack_image_embeds = image_outputs.deepstack_features
            #image_embeds_return = [e.clone() for e in image_embeds]
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.base_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.base_model.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.base_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            position_ids = self.base_lm.model.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            if position_ids.size(0) ==3:
                # [3, B, seq_len]
                #text_pos_ids = torch.arange(position_ids.size(-1), dtype=torch.long, device=position_ids.device).unsqueeze(0).unsqueeze(0)
                L = position_ids.size(-1)
                prefix_lens = attention_mask.sum(dim=1).to(torch.long).unsqueeze(1)
                text_pos = torch.arange(L, device=self.device, dtype=torch.long).unsqueeze(0).repeat(prefix_lens.size(0), 1)              # [B, K]
                text_pos = (text_pos - (L - prefix_lens)).clamp_min(0).unsqueeze(0)
                position_ids = torch.cat([text_pos,position_ids], dim=0)
        
        outputs:Qwen3VLModelOutputWithPast = self.base_model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            output_hidden_states=True,
            **kwargs,
        )

        last_hidden_states = outputs.last_hidden_state
        hidden_state_for_mem = outputs.hidden_states[self.layer_idx_for_mem]
        
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        if mix_mode == 'base':
            base_logits = self.lm_head(last_hidden_states[:, slice_indices, :])
            logits = base_logits
        elif mix_mode == 'mem':
            memory_logits = self.memory(hidden_state_for_mem[:, slice_indices, :])
            logits = memory_logits
        elif mix_mode == 'mix':
            log_lam = math.log(mix_lambda)
            log_1m = math.log(1.0 - mix_lambda)
            base_logits = self.lm_head(last_hidden_states[:, slice_indices, :])
            memory_logits = self.memory(hidden_state_for_mem[:, slice_indices, :])
            base_logp = torch.log_softmax(base_logits, dim=-1)
            mem_logp = torch.log_softmax(memory_logits, dim=-1)

            logits = torch.logsumexp(
                torch.stack([base_logp + log_lam, mem_logp + log_1m], dim=0),
                dim=0,
            )
            pass
        else:
            raise AttributeError("Mix mode has to be either 'base', 'mem' or 'mix'.")


        return Qwen3VLCausalLMOutputWithPast_Pos(
            loss=None,
            logits=logits,
            #hidden_states=outputs.hidden_states,
            past_key_values=outputs.past_key_values,
            position_ids=position_ids
            #past_key_values=None,
        )
    
    def compute_student_seq_logp_qwen3vl(
        self,
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

        device = self.device
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
            out_prompt = self.forward_basic(
                **single,
                past_key_values=base_cache,
                use_cache=True,
                return_dict=True,
                mix_mode='mem'
            )
        except TypeError:
            # Some versions initialize DynamicCache internally if not passed
            out_prompt = self.forward_basic(
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
            B_cand = tok.size(0)
            suffix_position_ids = prompt_position_ids[:,:,-1].unsqueeze(-1) +1
            suffix_position_ids = suffix_position_ids + torch.arange(L, device=device).view(1, -1)
            suffix_position_ids = suffix_position_ids.expand(-1,B_cand,-1)
            suffix_cache_position = torch.arange(prefix_len, prefix_len + L, device=device)
            out_ans = self.forward_basic(
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

    def compute_ce_loss_from_prefix_cache(
        self,
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
        device = self.device
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
        out = self.forward_basic(
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

        return loss, logits

    def compute_loss_premerged_with_ce(
        self,
        model_config,
        prompt_inputs: dict[str, torch.Tensor],                    # BatchFeature-like dict, tensors with batch dim [B, ...]
        label_mask:torch.Tensor,
        answer_ids,
        batch_cand_tokens, ret_scores, sum_cand_logps, candidate_mask,m_first_tok_id, m_first_tok_logp, m_first_tok_tail,
        pad_id: int,
        eos_id: Optional[int],
        mode: Literal["seqkd", "mml"] = "seqkd",
        add_kl = False,
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
        device = self.device
        B = int(prompt_inputs["input_ids"].shape[0])
        prompt_attention_mask = build_prompt_only_masks(self.device, prompt_inputs["input_ids"], prompt_inputs["attention_mask"], label_mask, pad_id=pad_id)
        #full_attn_mask = prompt_inputs['attention_mask']
        prompt_inputs['attention_mask'] = prompt_attention_mask


        assert len(batch_cand_tokens) == B
        assert len(sum_cand_logps) == B

        # Extract top-1 realized sequences (index 0) per example
        kd_losses: List[torch.Tensor] = []
        ce_losses: List[torch.Tensor] = []
        ft_kl_losses: List[torch.Tensor] = []

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

            candidate_log_weights=sample_ret_score/float(tau_retrieval)
            uniq_ids, merged_logp, merged_tail_logp = marginalize_first_token_logp_topk(answer_ids_top32=m_first_tok_id[b], answer_logp_top32=m_first_tok_logp[b], tail_mass=m_first_tok_tail[b], candidate_mask=sample_candidate_mask, candidate_log_weights=candidate_log_weights, top_k=top_k)
            
            #candidate_mask = None
            if len(cand_tokens) == 0:
                kd_losses.append(prompt_inputs["input_ids"].new_zeros((), dtype=torch.float32))
                ce_losses.append(prompt_inputs["input_ids"].new_zeros((), dtype=torch.float32))
                print("Empty candidate batch")
                continue

            cand_tokens = trim_excess_right_padding(cand_tokens, pad_id)

            # Student conditioned on ORIGINAL image+question for sample b
            student_seq_logp, prompt_cache_return, logp_first_dist, prompt_position_ids = self.compute_student_seq_logp_qwen3vl(
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

            ce_loss, logits = self.compute_ce_loss_from_prefix_cache(
                ans_ids_1=current_gt,
                attention_mask_1=torch.ones_like(current_gt),
                ans_labels_1=current_gt,
                suffix_position_ids=suffix_position_ids,
                suffix_cache_position=suffix_cache_position,
                prefix_cache=prompt_cache_return,
                logp_first_dist=logp_first_dist,
                )
            
            if add_kl:
                first_tok_kl_loss = sparse_kl_with_tail(student_logits=logits[:, 0, :], teacher_ids_with=uniq_ids.unsqueeze(0), teacher_logprob_with=merged_logp.unsqueeze(0), tail_logprob=merged_tail_logp.unsqueeze(0))
                ft_kl_losses.append(first_tok_kl_loss)

            if mode == "seqkd":
                loss_b = seq_kd_loss_from_seq_logp(
                    student_seq_logp=student_seq_logp.unsqueeze(0),
                    logw=logw.unsqueeze(0),
                    candidate_mask=None,
                    reduction="none",
                ).squeeze(0)
            elif mode == "mml":
                loss_b = mml_loss_from_seq_logp(
                    student_seq_logp=student_seq_logp.unsqueeze(0),
                    logw=logw.unsqueeze(0),
                    candidate_mask=None,
                    reduction="none",
                ).squeeze(0)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            kd_losses.append(loss_b)
            ce_losses.append(ce_loss)
            del prompt_cache_return

        kd_losses = torch.stack(kd_losses, dim=0)  # [B]
        ce_losses = torch.stack(ce_losses, dim=0)
        ft_kl_losses= torch.stack(ft_kl_losses, dim=0)
        if reduction == "none":
            if add_kl:
                return kd_losses, ce_losses, ft_kl_losses
            else:
                return kd_losses, ce_losses
        if reduction == "sum":
            if add_kl:
                return kd_losses.sum(), ce_losses.sum(), ft_kl_losses.sum()
            else:
                return kd_losses.sum(), ce_losses.sum()
        if reduction == "mean":
            if add_kl:
                return kd_losses.mean(), ce_losses.mean(), ft_kl_losses.mean()
            else:
                return kd_losses.mean(), ce_losses.mean()
        raise ValueError(f"Unknown reduction: {reduction}")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep=0,
        mix_mode='base',
        mix_lambda=0.6,

        model_config=None,
        prompt_inputs=None, 
        label_mask=None, 
        answer_ids=None, 
        batch_cand_tokens=None, 
        ret_scores=None, 
        sum_cand_logps=None,
        m_first_tok_id=None, 
        m_first_tok_logp=None, 
        m_first_tok_tail=None,
        candidate_mask=None, 
        pad_id=None, 
        eos_id=None, 
        detach_prompt_cache=True, 
        tau_retrieval=None, 
        top_k=None,
        mode=None,
        add_kl=False,

        branch=None,
        **kwargs: Unpack[TransformersKwargs],):
        if branch == "generation":
            if position_ids.size(0) != 4 or len(position_ids.shape) != 3:
                position_ids = None
            out = self.forward_basic(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                past_key_values=past_key_values,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                mix_mode=mix_mode,
                mix_lambda=mix_lambda,
                **kwargs,)
        elif branch == "train":
            if mode is None:
                mode = "mml"
            out = self.compute_loss_premerged_with_ce(
            model_config=model_config, prompt_inputs=prompt_inputs, label_mask=label_mask, answer_ids=answer_ids, batch_cand_tokens=batch_cand_tokens, ret_scores=ret_scores, sum_cand_logps=sum_cand_logps, m_first_tok_id=m_first_tok_id,m_first_tok_logp=m_first_tok_logp, 
            m_first_tok_tail=m_first_tok_tail, candidate_mask=candidate_mask, pad_id=self.pad_token_id, eos_id=self.eos_token_id, detach_prompt_cache=detach_prompt_cache, tau_retrieval=tau_retrieval, top_k=top_k, chunk_size=20, mode=mode, add_kl=add_kl
        )
        
        return out