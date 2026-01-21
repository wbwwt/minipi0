import logging
import math
import os
import re
import json
from collections import deque
from pathlib import Path
from typing import Literal, TypedDict, Optional, List, Union, Tuple
from typing_extensions import Unpack

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# --- 依赖库: 仅保留 transformers 和 torch ---
from transformers import PreTrainedModel, PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
from transformers.utils import cached_file
from safetensors.torch import load_file

# --- 本地依赖: 稍后我会给你 config.py ---
# try:
#     from .config import PI0Config
# except ImportError:
#     PI0Config = None # 占位符
    
from model.config import PI0Config

# --- 常量定义 (原 lerobot.utils.constants) ---
ACTION = "action"
OBS_STATE = "observation.state"
OBS_LANGUAGE_TOKENS = "observation.language_instruction.input_ids"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language_instruction.attention_mask"
OBS_IMAGES = "observation.images"
OPENPI_ATTENTION_MASK_VALUE = -1e9
DEFAULT_IMAGE_SIZE = 224

# --- 辅助类与函数 ---

class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None

def get_safe_dtype(target_dtype, device_type):
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype

# def create_sinusoidal_pos_embedding(time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu") -> Tensor:
#     if dimension % 2 != 0: raise ValueError(f"dimension ({dimension}) must be divisible by 2")
#     if time.ndim != 1: raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
#     dtype = get_safe_dtype(torch.float64, device.type)
#     fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
#     period = min_period * (max_period / min_period) ** fraction
#     scaling_factor = 1.0 / period * 2 * math.pi
#     sin_input = scaling_factor[None, :] * time[:, None]
#     return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)

def create_sinusoidal_pos_embedding(time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu") -> torch.Tensor:
    if dimension % 2 != 0: raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1: raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
    
    # 1. 【修改点】保存目标设备，但强制当前计算使用 CPU
    target_device = device
    calc_device = "cpu" 
    
    # 2. 【修改点】把输入 time 也搬到 CPU，防止后续乘法报错
    time_cpu = time.to(calc_device)

    # 3. 【修改点】dtype 直接用 float32 (CPU上很安全)，device 用 calc_device
    # 原代码: dtype = get_safe_dtype(torch.float64, device.type) 
    # 修改为:
    fraction = torch.linspace(0.0, 1.0, dimension // 2, device=calc_device)
    
    # --- 下面这几行逻辑完全不用动 ---
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time_cpu[:, None] # 注意这里用了 time_cpu
    
    # 4. 【修改点】计算结果搬回目标 GPU
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1).to(target_device)

def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)

def make_att_2d_masks(pad_masks, att_masks):
    if att_masks.ndim != 2: raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2: raise ValueError(pad_masks.ndim)
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks

def pad_vector(vector, new_dim):
    if vector.shape[-1] >= new_dim: return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))

def resize_with_pad_torch(images: torch.Tensor, height: int, width: int, mode: str = "bilinear") -> torch.Tensor:
    if images.shape[-1] <= 4:
        channels_last = True
        if images.dim() == 3: images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
    else:
        channels_last = False
        if images.dim() == 3: images = images.unsqueeze(0)
    
    batch_size, channels, cur_height, cur_width = images.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    
    resized_images = F.interpolate(
        images, size=(resized_height, resized_width), 
        mode=mode, align_corners=False if mode == "bilinear" else None
    )
    
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    
    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images, (pad_w0, pad_w1, pad_h0, pad_h1), 
        mode="constant", value=constant_value
    )
    
    if channels_last: padded_images = padded_images.permute(0, 2, 3, 1)
    return padded_images

def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, paligemma, gemma_expert):
    models = [paligemma.language_model, gemma_expert.model]
    query_states, key_states, value_states, gates = [], [], [], []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_states.append(layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2))
        key_states.append(layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2))
        value_states.append(layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2))

    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    
    dummy_tensor = torch.zeros(query_states.shape[0], query_states.shape[2], query_states.shape[-1], device=query_states.device, dtype=query_states.dtype)
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)
    
    scaling = paligemma.language_model.layers[layer_idx].self_attn.scaling
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.language_model.layers[layer_idx].self_attn, 
        query_states, key_states, value_states, attention_mask, scaling
    )
    
    head_dim = paligemma.language_model.layers[layer_idx].self_attn.head_dim
    att_output = att_output.reshape(query_states.shape[0], -1, 1 * 8 * head_dim)
    
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds

class GemmaConfig:
    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width, self.depth, self.mlp_dim = width, depth, mlp_dim
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim

def get_gemma_config(variant: str) -> GemmaConfig:
    if variant == "gemma_300m":
        return GemmaConfig(1024, 18, 4096, 8, 1, 256)
    elif variant == "gemma_2b":
        return GemmaConfig(2048, 18, 16_384, 8, 1, 256)
    raise ValueError(f"Unknown variant: {variant}")


# --- 核心模型组件 ---

class PaliGemmaWithExpertModel(nn.Module):
    def __init__(self, vlm_config, action_expert_config, use_adarms=None, precision: Literal["bfloat16", "float32"] = "bfloat16", image_size: int = DEFAULT_IMAGE_SIZE, freeze_vision_encoder: bool = False, train_expert_only: bool = False):
        if use_adarms is None: use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None
        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    def to_bfloat16_for_selected_params(self, precision="bfloat16"):
        if precision == "bfloat16": self.to(dtype=torch.bfloat16)
        elif precision == "float32": self.to(dtype=torch.float32); return
        
        params_to_keep_float32 = ["vision_tower.vision_model.embeddings", "input_layernorm", "post_attention_layernorm", "model.norm"]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
            for p in self.paligemma.vision_tower.parameters(): p.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for p in self.paligemma.parameters(): p.requires_grad = False
            
    def train(self, mode=True):
        super().train(mode)
        if self.freeze_vision_encoder: self.paligemma.vision_tower.eval()
        if self.train_expert_only: self.paligemma.eval()

    def embed_image(self, image): return self.paligemma.model.get_image_features(image)
    def embed_language_tokens(self, tokens): return self.paligemma.language_model.embed_tokens(tokens)

    def forward(self, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=None, adarms_cond=None):
        if adarms_cond is None: adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(inputs_embeds=inputs_embeds[0], attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, adarms_cond=adarms_cond[0] if adarms_cond else None)
            return [prefix_output.last_hidden_state, None], prefix_output.past_key_values
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(inputs_embeds=inputs_embeds[1], attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, adarms_cond=adarms_cond[1] if adarms_cond else None)
            return [None, suffix_output.last_hidden_state], None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers
            
            # 使用梯度检查点的简化逻辑 (这里直接前向传播)
            for layer_idx in range(num_layers):
                 inputs_embeds = compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, self.paligemma, self.gemma_expert)
            
            outputs_embeds = []
            for i, hidden_states in enumerate(inputs_embeds):
                out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                outputs_embeds.append(out_emb)
            return outputs_embeds, None


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)
        
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config, action_expert_config, use_adarms=[False, False],
            precision=config.dtype, image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )
        
        # 动作头
        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)
        self.state_proj = nn.Linear(config.max_state_dim, action_expert_config.width)
        self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
        self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        
        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)
        return func(*args, **kwargs)

    # def _prepare_attention_masks_4d(self, att_2d_masks):
    #     att_2d_masks_4d = att_2d_masks[:, None, :, :]
    #     return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)
    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        # 👇 加上 .to(torch.bool) 即可
        return torch.where(att_2d_masks_4d.to(torch.bool), 0.0, OPENPI_ATTENTION_MASK_VALUE)

    # def sample_noise(self, shape, device):
    #     return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)
    def sample_noise(self, shape, device):
    # 强制在 CPU 生成，然后移动到 device，避开 CUDA kernel 兼容性问题
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device="cpu").to(device)

    def sample_time(self, bsize, device):
        time_beta = sample_beta(self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device)
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        embs, pad_masks, att_masks = [], [], []
        for img, img_mask in zip(images, img_masks):
             img_emb = self._apply_checkpoint(self.paligemma_with_expert.embed_image, img)
             bsize, num_img_embs = img_emb.shape[:2]
             embs.append(img_emb)
             pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
             att_masks += [0] * num_img_embs
        
        def lang_embed_func(lang_tokens):
             lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
             return lang_emb * math.sqrt(lang_emb.shape[-1])
        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]
        
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(embs.shape[0], len(att_masks))
        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        embs, pad_masks, att_masks = [], [], []
        if self.state_proj.weight.dtype == torch.float32: state = state.to(torch.float32)
        state_emb = self._apply_checkpoint(self.state_proj, state)
        embs.append(state_emb[:, None, :])
        pad_masks.append(torch.ones(state_emb.shape[0], 1, dtype=torch.bool, device=state_emb.device))
        att_masks += [1]
        
        time_emb = create_sinusoidal_pos_embedding(timestep, self.action_in_proj.out_features, self.config.min_period, self.config.max_period, device=timestep.device).type(timestep.dtype)
        action_emb = self._apply_checkpoint(self.action_in_proj, noisy_actions)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        
        def mlp_func(x): return self.action_time_mlp_out(F.silu(self.action_time_mlp_in(x)))
        action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
        
        embs.append(action_time_emb)
        pad_masks.append(torch.ones(action_time_emb.shape[0], action_time_emb.shape[1], dtype=torch.bool, device=timestep.device))
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))
        
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)[None, :].expand(embs.shape[0], len(att_masks))
        return embs, pad_masks, att_masks, None

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
        if noise is None: noise = self.sample_noise(actions.shape, actions.device)
        if time is None: time = self.sample_time(actions.shape[0], actions.device)
        
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        
        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d, position_ids=position_ids, past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs], use_cache=False, adarms_cond=[None, adarms_cond]
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond)
        suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)
        v_t = self._apply_checkpoint(self.action_out_proj, suffix_out)
        
        return F.mse_loss(u_t, v_t, reduction="none")

    # @torch.no_grad()
    # def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, num_steps=None, **kwargs):
    #     if num_steps is None: num_steps = self.config.num_inference_steps
    #     bsize, device = state.shape[0], state.device
    #     print(device)
        
    #     if noise is None:
    #         actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
    #         noise = self.sample_noise(actions_shape, device)
    #     # ****************************************************
    #     # 1. 智能获取设备 (以图像为基准)
    #     if isinstance(images, (list, tuple)):
    #         target_device = images[0].device
    #     else:
    #         target_device = images.device
        
    #     # 2. 强制搬运语言输入
    #     if lang_tokens is not None:
    #         lang_tokens = lang_tokens.to(target_device)
    #     if lang_masks is not None:
    #         lang_masks = lang_masks.to(target_device)

    #     # 3. [新增] 强制搬运 State (机械臂状态)
    #     # 这就是解决本次报错的关键
    #     if state is not None:
    #         state = state.to(target_device)
    #     # ****************************************************
    #     prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    #     prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    #     prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    #     prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        
    #     self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    #     _, past_key_values = self.paligemma_with_expert.forward(
    #         attention_mask=prefix_att_2d_masks_4d, position_ids=prefix_position_ids, past_key_values=None,
    #         inputs_embeds=[prefix_embs, None], use_cache=True
    #     )
        
    #     dt = -1.0 / num_steps
    #     x_t = noise
    #     for step in range(num_steps):
    #         time = 1.0 + step * dt
    #         time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
            
    #         suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time_tensor)
            
    #         suffix_len = suffix_pad_masks.shape[1]
    #         prefix_len = prefix_pad_masks.shape[1]
    #         prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
    #         suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    #         full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
            
    #         prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    #         position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    #         full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
    #         self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
            
    #         outputs_embeds, _ = self.paligemma_with_expert.forward(
    #             attention_mask=full_att_2d_masks_4d, position_ids=position_ids, past_key_values=past_key_values,
    #             inputs_embeds=[None, suffix_embs], use_cache=False, adarms_cond=[None, adarms_cond]
    #         )
    #         suffix_out = outputs_embeds[1][:, -self.config.chunk_size :].to(dtype=torch.float32)
    #         v_t = self.action_out_proj(suffix_out)
    #         x_t = x_t + dt * v_t
            
    #     return x_t

    @torch.no_grad()
    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, num_steps=None, **kwargs):
        if num_steps is None: num_steps = self.config.num_inference_steps
        
        # =========================================================================
        # 1. 【核心修正】先确定目标设备 (以 images 为绝对标准)
        # =========================================================================
        if isinstance(images, (list, tuple)):
            target_device = images[0].device
        else:
            target_device = images.device

        # =========================================================================
        # 2. 【核心修正】有了目标设备后，立刻把 State 搬过去
        # =========================================================================
        if state is not None:
            state = state.to(target_device)
        
        bsize = state.shape[0]
        # 注意：此时 state.device 已经是 GPU 了，但为了保险，我们后续全部使用 target_device

        # =========================================================================
        # 3. 【核心修正】基于目标设备生成 Noise
        #    这样 x_t 就一定是在 GPU 上
        # =========================================================================
        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            # 使用 target_device (GPU)，你的 sample_noise 函数内部会在 CPU 生成后搬运过来
            noise = self.sample_noise(actions_shape, target_device) 
        
        # 4. 搬运语言 Token
        if lang_tokens is not None:
            lang_tokens = lang_tokens.to(target_device)
        if lang_masks is not None:
            lang_masks = lang_masks.to(target_device)

        # -------------------------------------------------------------------------
        # 下面的逻辑保持不变，但注意 device 变量的使用
        # -------------------------------------------------------------------------
        
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d, position_ids=prefix_position_ids, past_key_values=None,
            inputs_embeds=[prefix_embs, None], use_cache=True
        )
        
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            # 【重要】时间张量也要用 target_device
            time_tensor = torch.tensor(time, dtype=torch.float32, device=target_device).expand(bsize)
            
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time_tensor)
            
            suffix_len = suffix_pad_masks.shape[1]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
            
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
            self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
            
            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d, position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs], use_cache=False, adarms_cond=[None, adarms_cond]
            )
            suffix_out = outputs_embeds[1][:, -self.config.chunk_size :].to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)
            x_t = x_t + dt * v_t
            
        return x_t


# --- 顶级 Policy 类 (继承 nn.Module) ---

class PI0Policy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = PI0Pytorch(config)
        self._action_queue = deque(maxlen=config.n_action_steps)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str | Path, config=None, **kwargs):
        # 1. 加载配置
        if config is None:
            # 假设 config.json 存在
            config_path = Path(pretrained_name_or_path) / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = PI0Config(**config_dict)
            else:
                raise ValueError(f"No config.json found in {pretrained_name_or_path}")

        # 2. 初始化模型
        model = cls(config)

        # 3. 加载权重
        print(f"Loading model from: {pretrained_name_or_path}")
        try:
            # 尝试加载 safetensors
            model_file = Path(pretrained_name_or_path) / "model.safetensors"
            if model_file.exists():
                state_dict = load_file(model_file)
            else:
                # 尝试 bin
                model_file = Path(pretrained_name_or_path) / "pytorch_model.bin"
                if model_file.exists():
                    state_dict = torch.load(model_file, map_location="cpu")
                else:
                    raise FileNotFoundError("No model weights found")

            # 4. 执行 Key Remapping (核心逻辑)
            fixed_state_dict = model._fix_pytorch_state_dict_keys(state_dict)
            
            # 添加 model. 前缀
            remapped_state_dict = {}
            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    remapped_state_dict[f"model.{key}"] = value
                else:
                    remapped_state_dict[key] = value

            # 5. 加载状态字典
            msg = model.load_state_dict(remapped_state_dict, strict=False)
            print(f"Load result: {msg}")
            
        except Exception as e:
            print(f"Error loading weights: {e}")
        
        return model

    def _fix_pytorch_state_dict_keys(self, state_dict):
        # 保持原有的映射逻辑，这对加载权重至关重要
        fixed_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            # 处理 expert layers
            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight", key):
                expert_uses_adarms = getattr(self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False)
                if expert_uses_adarms: continue
            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                expert_uses_adarms = getattr(self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False)
                if expert_uses_adarms: continue

            # 处理 MLP 命名差异
            if key.startswith("time_mlp_in."): new_key = key.replace("time_mlp_in.", "action_time_mlp_in.")
            elif key.startswith("time_mlp_out."): new_key = key.replace("time_mlp_out.", "action_time_mlp_out.")
            
            fixed_state_dict[new_key] = value
        return fixed_state_dict

    def _preprocess_images(self, batch):
        images, img_masks = [], []
        device = next(self.parameters()).device
        
        # 假设 batch 里有 observation.images.cam_high 这样的 key
        # 这里简化处理，寻找包含 'image' 的 key
        img_keys = [k for k in batch.keys() if "image" in k]
        
        for key in img_keys:
            img = batch[key].to(device).float()
            # [B, C, H, W] -> [-1, 1]
            if img.dim() == 4:
                # Resize logic here if needed, or assume input is already resized
                if img.shape[-2:] != (self.config.image_resolution[0], self.config.image_resolution[1]):
                    img = resize_with_pad_torch(img, self.config.image_resolution[0], self.config.image_resolution[1])
                
                # Normalize [0,1] -> [-1,1]
                if img.min() >= 0:
                    img = img * 2.0 - 1.0
                    
            images.append(img)
            img_masks.append(torch.ones(img.shape[0], dtype=torch.bool, device=device))
            
        return images, img_masks

    def prepare_state(self, batch):
        state = batch.get(OBS_STATE)
        if state is None: return torch.zeros(1, self.config.max_state_dim).to(next(self.parameters()).device)
        return pad_vector(state, self.config.max_state_dim)

    def prepare_action(self, batch):
        actions = batch.get(ACTION)
        return pad_vector(actions, self.config.max_action_dim)

    def forward(self, batch):
        images, img_masks = self._preprocess_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)
        
        losses = self.model(images, img_masks, lang_tokens, lang_masks, state, actions)
        
        # Truncate to action dim
        original_action_dim = self.config.output_features[ACTION]["shape"][0]
        losses = losses[:, :, :original_action_dim]
        return losses.mean()

    @torch.no_grad()
    def select_action(self, batch):
        self.eval()
        if len(self._action_queue) == 0:
            images, img_masks = self._preprocess_images(batch)
            lang_tokens = batch[OBS_LANGUAGE_TOKENS]
            lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
            state = self.prepare_state(batch)
            
            actions = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
            
            original_action_dim = self.config.output_features[ACTION]["shape"][0]
            actions = actions[:, :, :original_action_dim]
            
            # 转置并存入队列 [B, Chunk, Dim] -> [Chunk, B, Dim]
            actions = actions.transpose(0, 1)
            self._action_queue.extend(actions[:self.config.n_action_steps])
            
        return self._action_queue.popleft()

