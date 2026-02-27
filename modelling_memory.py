from typing import Union, Optional, Literal
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, is_torchdynamo_compiling
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast, Qwen3VLModelOutputWithPast

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


    @property
    def device(self) -> torch.device:
        return self.base_model.device
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_lm.prepare_inputs_for_generation(*args, **kwargs)

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()
    
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
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.base_model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        
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


        return Qwen3VLCausalLMOutputWithPast(
            loss=None,
            logits=logits,
            #hidden_states=outputs.hidden_states,
            past_key_values=outputs.past_key_values,
            #past_key_values=None,
        )