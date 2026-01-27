#!/usr/bin/env python3
"""
GLM-4.7-Flash Full Inference Engine for 8GB GPUs

Complete implementation with:
- INT4 dequantization
- MLA (Multi-head Latent Attention) with KV cache
- MoE with expert offloading to CPU
- Expert LRU cache on GPU

Usage:
    python3 scripts/glm_inference_full.py \
        --model ./models/glm-4.7-flash-int4 \
        --prompt "def fibonacci(n):"

Copyright (c) 2025 Distillix. MIT License.
"""

import argparse
import gc
import json
import math
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Try to import transformers for tokenizer
try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False
    print("[WARN] transformers not installed. Using simple tokenizer fallback.")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GLMConfig:
    """Model configuration loaded from config.json."""
    hidden_size: int = 2048
    num_hidden_layers: int = 47
    vocab_size: int = 154880
    num_attention_heads: int = 20
    num_key_value_heads: int = 20
    
    # MLA dimensions
    q_lora_rank: int = 768
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    
    # MoE
    n_routed_experts: int = 64
    n_shared_experts: int = 1
    num_experts_per_tok: int = 4
    moe_intermediate_size: int = 1536
    intermediate_size: int = 10240  # Dense MLP
    first_k_dense_replace: int = 1
    routed_scaling_factor: float = 1.8
    
    # Other
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 202752
    
    # EOS tokens
    eos_token_id: List[int] = field(default_factory=lambda: [154820, 154827, 154829])
    pad_token_id: int = 154820
    
    @classmethod
    def from_dict(cls, d: dict) -> "GLMConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass  
class InferenceConfig:
    """Inference settings."""
    model_path: str = "./models/glm-4.7-flash-int4"
    device: str = "cuda"
    expert_device: str = "cpu"
    
    # Memory management
    expert_cache_size: int = 8  # LRU cache for experts on GPU
    use_pinned_memory: bool = True
    offload_experts: bool = True
    
    # Quantization
    group_size: int = 128
    
    # Generation
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


# =============================================================================
# Safetensors Loading Utilities
# =============================================================================

class SafetensorsLoader:
    """Efficient loader for safetensors files."""
    
    def __init__(self, path: str):
        self.path = path
        self._header_size = None
        self.meta = self._load_metadata()
        self._file = None
        
    def _load_metadata(self) -> dict:
        with open(self.path, 'rb') as f:
            header_size = struct.unpack('<Q', f.read(8))[0]
            self._header_size = header_size
            header = f.read(header_size).decode('utf-8')
        return json.loads(header)
    
    def keys(self) -> List[str]:
        return list(self.meta.keys())
    
    def has_key(self, key: str) -> bool:
        return key in self.meta
    
    def get_tensor(self, key: str, device: str = "cpu") -> Tensor:
        """Load a tensor by key."""
        if key not in self.meta:
            raise KeyError(f"Key {key} not found")
        
        info = self.meta[key]
        dtype_map = {
            'F16': (np.float16, torch.float16),
            'F32': (np.float32, torch.float32),
            'BF16': (np.float16, torch.bfloat16),  # Read as fp16, convert
            'I64': (np.int64, torch.int64),
            'I32': (np.int32, torch.int32),
            'U8': (np.uint8, torch.uint8),
            'I8': (np.int8, torch.int8),
        }
        
        np_dtype, torch_dtype = dtype_map.get(info['dtype'], (np.float32, torch.float32))
        shape = info['shape']
        start, end = info['data_offsets']
        
        # Data section starts after 8-byte header size + header content
        data_offset = 8 + self._header_size
        
        with open(self.path, 'rb') as f:
            # Check if offset is valid
            f.seek(0, 2)
            file_size = f.tell()
            if data_offset + end > file_size:
                raise RuntimeError(f"Tensor {key} offset {end} exceeds file size")
            
            f.seek(data_offset + start)
            raw_data = f.read(end - start)
            if len(raw_data) == 0:
                raise RuntimeError(f"Failed to read data for {key}: got 0 bytes")
            data = np.frombuffer(raw_data, dtype=np_dtype)
        
        tensor = torch.from_numpy(data.copy()).reshape(shape)
        
        if info['dtype'] == 'BF16':
            tensor = tensor.to(torch.bfloat16)
        
        return tensor.to(device) if device != "cpu" else tensor
    
    def is_tensor_valid(self, key: str) -> bool:
        """Check if tensor can be loaded (offset is within file)."""
        if key not in self.meta:
            return False
        
        info = self.meta[key]
        _, end = info['data_offsets']
        data_offset = 8 + self._header_size
        
        with open(self.path, 'rb') as f:
            f.seek(0, 2)
            file_size = f.tell()
        
        return data_offset + end <= file_size


# =============================================================================
# INT4 Dequantization
# =============================================================================

def unpack_int4(packed: Tensor, num_elements: int) -> Tensor:
    """Unpack INT4 from uint8 (2 values per byte)."""
    packed = packed.flatten()
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    unpacked = torch.stack([low, high], dim=-1).flatten()
    return (unpacked[:num_elements].to(torch.int8) - 8)


def dequantize_int4(
    qweight: Tensor,
    scales: Tensor,
    shape: Tensor,
    group_size: int = 128,
) -> Tensor:
    """Dequantize INT4 weight to float."""
    out_features, in_features = shape.tolist()
    
    # Unpack INT4
    q = unpack_int4(qweight, out_features * in_features)
    q = q.reshape(out_features, in_features).float()
    
    # Apply scales per group
    num_groups = scales.shape[1]
    group_size_actual = (in_features + num_groups - 1) // num_groups
    
    # Reshape for group-wise scaling
    q = q.reshape(out_features, num_groups, -1)
    weight = q * scales.unsqueeze(-1).float()
    
    return weight.reshape(out_features, -1)[:, :in_features]


# =============================================================================
# Expert LRU Cache
# =============================================================================

class ExpertCache:
    """LRU cache for keeping hot experts on GPU."""
    
    def __init__(self, max_size: int, device: str = "cuda"):
        self.max_size = max_size
        self.device = device
        self.cache: OrderedDict[str, Dict[str, Tensor]] = OrderedDict()
        self.lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0}
    
    def get(self, key: str) -> Optional[Dict[str, Tensor]]:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.stats["hits"] += 1
                return self.cache[key]
            self.stats["misses"] += 1
            return None
    
    def put(self, key: str, weights: Dict[str, Tensor]):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return
            
            # Evict if full
            while len(self.cache) >= self.max_size:
                _, evicted = self.cache.popitem(last=False)
                del evicted
                torch.cuda.empty_cache()
            
            # Store on GPU
            self.cache[key] = {
                k: v.to(self.device, non_blocking=True) 
                for k, v in weights.items()
            }
    
    def get_stats(self) -> dict:
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {**self.stats, "size": len(self.cache), "hit_rate": f"{hit_rate:.1%}"}


# =============================================================================
# RMSNorm
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


# =============================================================================
# Rotary Position Embedding (RoPE)
# =============================================================================

class RotaryEmbedding:
    """RoPE implementation."""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 1000000.0, device: str = "cuda"):
        self.dim = dim
        self.theta = theta
        self.device = device
        
        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer = inv_freq
        
        # Precompute cos/sin for common sequence lengths
        self._cos_cache = {}
        self._sin_cache = {}
    
    def _get_cos_sin(self, seq_len: int, device: str) -> Tuple[Tensor, Tensor]:
        if seq_len not in self._cos_cache:
            inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
            t = torch.arange(seq_len, device=device).float()
            freqs = torch.outer(t, inv_freq)
            self._cos_cache[seq_len] = freqs.cos()
            self._sin_cache[seq_len] = freqs.sin()
        return self._cos_cache[seq_len], self._sin_cache[seq_len]
    
    def apply(self, x: Tensor, positions: Tensor) -> Tensor:
        """Apply RoPE to tensor x at given positions."""
        seq_len = int(positions.max().item()) + 1
        cos, sin = self._get_cos_sin(seq_len, x.device)
        
        # x shape: [batch, seq, heads, head_dim] or [batch, heads, seq, head_dim]
        # We apply to the rope dimensions only
        
        cos = cos[positions]  # [batch, seq, dim//2]
        sin = sin[positions]
        
        # Reshape for broadcasting
        if x.dim() == 4:
            cos = cos.unsqueeze(2)  # [batch, seq, 1, dim//2]
            sin = sin.unsqueeze(2)
        
        # Split into pairs
        x1, x2 = x[..., ::2], x[..., 1::2]
        
        # Apply rotation
        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], dim=-1)
        
        return x_rotated


# =============================================================================
# MLA Attention (Multi-head Latent Attention)
# =============================================================================

class MLAAttention(nn.Module):
    """
    Multi-head Latent Attention from GLM-4.
    Uses LoRA-style compression for Q and KV projections.
    """
    
    def __init__(self, config: GLMConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        
        self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Weights will be loaded externally
        self.q_a_proj_weight = None
        self.q_b_proj_weight = None
        self.kv_a_proj_weight = None
        self.kv_b_proj_weight = None
        self.o_proj_weight = None
        
        # Layer norms
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, config.rms_norm_eps)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
        
        # RoPE
        self.rope = RotaryEmbedding(self.qk_rope_head_dim, theta=config.rope_theta)
    
    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_kv: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        batch, seq_len, _ = x.shape
        
        # Convert input to float32 for computation (weights are float32 from dequant)
        x = x.float()
        
        # Q projection: hidden -> q_lora -> heads * head_dim
        q = F.linear(x, self.q_a_proj_weight)
        q = self.q_a_layernorm(q)
        q = F.linear(q, self.q_b_proj_weight)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        
        # KV projection with MQA: hidden -> (kv_lora + rope_dim)
        # kv_a_proj_with_mqa outputs [kv_lora_rank + qk_rope_head_dim]
        # The rope part is applied separately, the kv_lora part goes through layernorm
        kv_compressed = F.linear(x, self.kv_a_proj_weight)
        
        # Split compressed KV into the normalized part and rope part
        # kv_lora_rank (512) + qk_rope_head_dim (64) = 576
        kv_lora, k_rope_pe = kv_compressed.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        
        # Normalize only the kv_lora part
        kv_lora = self.kv_a_layernorm(kv_lora)
        
        # Expand: kv_lora -> heads * (qk_nope_head_dim + v_head_dim)
        kv_expanded = F.linear(kv_lora, self.kv_b_proj_weight)
        kv_expanded = kv_expanded.view(batch, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        
        k_nope, v = kv_expanded.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        
        # For K, combine nope (from expanded) with rope (from compressed, broadcasted)
        # The rope PE is shared across heads (MQA style)
        k_rope = k_rope_pe.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        
        # Apply RoPE to the rope dimensions
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        q_rope = self.rope.apply(q_rope, positions)
        k_rope = self.rope.apply(k_rope, positions)
        
        # Combine into full Q and K
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)
        
        # KV cache
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        
        new_kv = (k, v)
        
        # Attention: [batch, heads, seq_q, head_dim] x [batch, heads, head_dim, seq_kv]
        q = q.transpose(1, 2)  # [batch, heads, seq_q, head_dim]
        k = k.transpose(1, 2)  # [batch, heads, seq_kv, head_dim]
        v = v.transpose(1, 2)  # [batch, heads, seq_kv, v_head_dim]
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        
        # [batch, seq, heads * v_head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, -1)
        
        # Output projection
        output = F.linear(attn_output, self.o_proj_weight)
        
        return output, new_kv


# =============================================================================
# MoE Layer with Expert Offloading
# =============================================================================

class MoELayer(nn.Module):
    """
    Mixture of Experts layer with CPU offloading.
    - Router and shared expert stay on GPU
    - Routed experts stay on CPU, loaded on-demand with LRU cache
    """
    
    def __init__(self, config: GLMConfig, layer_idx: int, inf_config: InferenceConfig):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.inf_config = inf_config
        
        self.hidden_size = config.hidden_size
        self.moe_intermediate = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.num_active = config.num_experts_per_tok
        self.scaling_factor = config.routed_scaling_factor
        
        # Router (small, stays on GPU)
        self.router_weight = None  # [num_experts, hidden_size]
        self.router_bias = None    # Optional correction bias
        
        # Shared expert (stays on GPU)
        self.shared_gate_weight = None
        self.shared_up_weight = None
        self.shared_down_weight = None
        
        # Routed experts (on CPU, loaded on-demand)
        # Stored as: {expert_idx: {"gate": Tensor, "up": Tensor, "down": Tensor}}
        self.expert_weights: Dict[int, Dict[str, Tensor]] = {}
        
        # Expert cache (on GPU)
        self.cache: Optional[ExpertCache] = None
    
    def set_cache(self, cache: ExpertCache):
        self.cache = cache
    
    def load_expert(self, expert_idx: int, gate: Tensor, up: Tensor, down: Tensor):
        """Load expert weights to CPU."""
        self.expert_weights[expert_idx] = {
            "gate": gate.to(self.inf_config.expert_device),
            "up": up.to(self.inf_config.expert_device),
            "down": down.to(self.inf_config.expert_device),
        }
        # Pin memory for faster transfer
        if self.inf_config.use_pinned_memory and self.inf_config.expert_device == "cpu":
            self.expert_weights[expert_idx] = {
                k: v.pin_memory() for k, v in self.expert_weights[expert_idx].items()
            }
    
    def _get_expert(self, expert_idx: int) -> Optional[Dict[str, Tensor]]:
        """Get expert weights from cache or CPU. Returns None if expert not available."""
        if expert_idx not in self.expert_weights:
            return None
        
        cache_key = f"L{self.layer_idx}_E{expert_idx}"
        
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        
        # Load from CPU
        weights = self.expert_weights[expert_idx]
        gpu_weights = {
            k: v.to(self.inf_config.device, non_blocking=True)
            for k, v in weights.items()
        }
        
        if self.cache:
            self.cache.put(cache_key, gpu_weights)
        
        return gpu_weights
    
    def _expert_forward(self, x: Tensor, weights: Dict[str, Tensor]) -> Tensor:
        """Forward through single expert."""
        gate = F.silu(F.linear(x, weights["gate"]))
        up = F.linear(x, weights["up"])
        hidden = gate * up
        return F.linear(hidden, weights["down"])
    
    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, hidden = x.shape
        x = x.float()  # Weights are float32 from dequant
        x_flat = x.view(-1, hidden)
        num_tokens = x_flat.shape[0]
        
        # Route
        router_logits = F.linear(x_flat, self.router_weight)
        if self.router_bias is not None:
            router_logits = router_logits + self.router_bias
        
        # Top-k selection
        routing_weights, selected_experts = torch.topk(
            router_logits, self.num_active, dim=-1
        )
        routing_weights = F.softmax(routing_weights, dim=-1) * self.scaling_factor
        
        # Output accumulator
        output = torch.zeros_like(x_flat)
        
        # Process each unique expert
        unique_experts = selected_experts.unique().tolist()
        
        for expert_idx in unique_experts:
            expert_weights = self._get_expert(expert_idx)
            
            # Skip if expert not available (file truncated)
            if expert_weights is None:
                continue
            
            # Find tokens using this expert
            expert_mask = (selected_experts == expert_idx)
            if not expert_mask.any():
                continue
            
            token_indices = expert_mask.any(dim=-1).nonzero(as_tuple=True)[0]
            expert_input = x_flat[token_indices]
            
            # Compute
            expert_output = self._expert_forward(expert_input, expert_weights)
            
            # Weight and accumulate
            weights_for_expert = (routing_weights * expert_mask.float()).sum(dim=-1)
            output[token_indices] += expert_output * weights_for_expert[token_indices].unsqueeze(-1)
        
        # Add shared expert
        if self.shared_gate_weight is not None:
            shared_gate = F.silu(F.linear(x_flat, self.shared_gate_weight))
            shared_up = F.linear(x_flat, self.shared_up_weight)
            shared_out = F.linear(shared_gate * shared_up, self.shared_down_weight)
            output = output + shared_out
        
        return output.view(batch, seq_len, hidden)


# =============================================================================
# Dense MLP (for first layer)
# =============================================================================

class DenseMLP(nn.Module):
    """Dense MLP for layers without MoE."""
    
    def __init__(self, config: GLMConfig):
        super().__init__()
        self.gate_weight = None
        self.up_weight = None
        self.down_weight = None
    
    def forward(self, x: Tensor) -> Tensor:
        x = x.float()  # Weights are float32 from dequant
        gate = F.silu(F.linear(x, self.gate_weight))
        up = F.linear(x, self.up_weight)
        return F.linear(gate * up, self.down_weight)


# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock(nn.Module):
    """Single transformer block."""
    
    def __init__(self, config: GLMConfig, layer_idx: int, inf_config: InferenceConfig):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        
        # Norms
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # Attention
        self.self_attn = MLAAttention(config, layer_idx)
        
        # MLP (MoE for most layers, dense for first)
        self.is_moe = layer_idx >= config.first_k_dense_replace
        if self.is_moe:
            self.mlp = MoELayer(config, layer_idx, inf_config)
        else:
            self.mlp = DenseMLP(config)
    
    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_kv: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # Attention
        residual = x
        x = self.input_layernorm(x)
        x, new_kv = self.self_attn(x, positions, attention_mask, past_kv)
        x = residual + x
        
        # MLP
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x, new_kv


# =============================================================================
# Full Model
# =============================================================================

class GLM47Flash(nn.Module):
    """Complete GLM-4.7-Flash model."""
    
    def __init__(self, config: GLMConfig, inf_config: InferenceConfig):
        super().__init__()
        self.config = config
        self.inf_config = inf_config
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, i, inf_config)
            for i in range(config.num_hidden_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # LM head
        self.lm_head_weight = None  # Loaded separately
        
        # Expert cache shared across layers
        self.expert_cache = ExpertCache(
            max_size=inf_config.expert_cache_size,
            device=inf_config.device,
        )
        
        # Set cache for MoE layers
        for layer in self.layers:
            if hasattr(layer.mlp, 'set_cache'):
                layer.mlp.set_cache(self.expert_cache)
    
    def forward(
        self,
        input_ids: Tensor,
        positions: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        batch, seq_len = input_ids.shape
        
        # Positions
        if positions is None:
            start_pos = 0 if past_key_values is None else past_key_values[0][0].shape[1]
            positions = torch.arange(start_pos, start_pos + seq_len, device=input_ids.device)
            positions = positions.unsqueeze(0).expand(batch, -1)
        
        # Embeddings
        x = self.embed_tokens(input_ids)
        
        # Causal mask
        if attention_mask is None and seq_len > 1:
            attention_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=x.device),
                diagonal=1,
            )
        
        # Process layers
        new_kv = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            x, kv = layer(x, positions, attention_mask, past_kv)
            new_kv.append(kv)
        
        # Output
        x = self.norm(x)
        # LM head is FP16, convert x to FP16 (saves memory)
        logits = F.linear(x.half(), self.lm_head_weight)
        
        return logits, new_kv
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        eos_token_ids: Optional[List[int]] = None,
    ) -> Tensor:
        """Generate tokens autoregressively."""
        if eos_token_ids is None:
            eos_token_ids = self.config.eos_token_id
        
        past_kv = None
        generated = input_ids
        
        for step in range(max_new_tokens):
            # Forward pass
            if past_kv is not None:
                curr_input = generated[:, -1:]
            else:
                curr_input = generated
            
            logits, past_kv = self(curr_input, past_key_values=past_kv)
            next_logits = logits[:, -1, :]
            
            # Sampling
            if temperature > 0:
                next_logits = next_logits / temperature
                
                # Top-k
                if top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = float("-inf")
                
                # Top-p
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float("-inf")
                
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Check EOS
            if next_token.item() in eos_token_ids:
                break
        
        return generated


# =============================================================================
# Model Loading
# =============================================================================

def load_model(inf_config: InferenceConfig) -> Tuple[GLM47Flash, Any]:
    """Load quantized model from disk."""
    model_path = Path(inf_config.model_path)
    device = inf_config.device
    
    # Load configs
    with open(model_path / "config.json") as f:
        config_dict = json.load(f)
    config = GLMConfig.from_dict(config_dict)
    
    with open(model_path / "quantize_config.json") as f:
        quant_config = json.load(f)
    group_size = quant_config.get("group_size", 128)
    inf_config.group_size = group_size
    
    # Check how many layers are actually valid in the file
    st_path = model_path / "model.safetensors"
    loader = SafetensorsLoader(str(st_path))
    
    valid_layers = 0
    for layer_idx in range(config.num_hidden_layers):
        key = f"model.layers.{layer_idx}.self_attn.q_a_proj.weight.qweight"
        if loader.is_tensor_valid(key):
            valid_layers += 1
        else:
            break
    
    if valid_layers < config.num_hidden_layers:
        print(f"[WARN] File truncated! Only {valid_layers}/{config.num_hidden_layers} layers available")
        config.num_hidden_layers = valid_layers
    
    print(f"Loading GLM-4.7-Flash INT4")
    print(f"  Hidden: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers} (of 47)")
    print(f"  Experts: {config.n_routed_experts} routed + {config.n_shared_experts} shared")
    print(f"  Group size: {group_size}")
    
    # Create model
    model = GLM47Flash(config, inf_config)
    
    # Load weights
    st_path = model_path / "model.safetensors"
    loader = SafetensorsLoader(str(st_path))
    
    print("\nLoading weights...")
    start_time = time.time()
    
    # Embeddings (FP16, on GPU)
    print("  Loading embeddings...")
    embed_weight = loader.get_tensor("model.embed_tokens.weight", device)
    model.embed_tokens.weight = nn.Parameter(embed_weight, requires_grad=False)
    
    # LM head (FP16, on GPU)
    print("  Loading LM head...")
    model.lm_head_weight = loader.get_tensor("lm_head.weight", device)
    
    # Final norm
    model.norm.weight = nn.Parameter(
        loader.get_tensor("model.norm.weight", device), requires_grad=False
    )
    
    # Load layers
    for layer_idx in range(config.num_hidden_layers):
        if layer_idx % 10 == 0:
            print(f"  Loading layer {layer_idx}/{config.num_hidden_layers}...")
        
        prefix = f"model.layers.{layer_idx}"
        layer = model.layers[layer_idx]
        
        # Layer norms
        layer.input_layernorm.weight = nn.Parameter(
            loader.get_tensor(f"{prefix}.input_layernorm.weight", device),
            requires_grad=False
        )
        layer.post_attention_layernorm.weight = nn.Parameter(
            loader.get_tensor(f"{prefix}.post_attention_layernorm.weight", device),
            requires_grad=False
        )
        
        # Attention weights (dequantized on GPU)
        attn = layer.self_attn
        
        # Q projections
        attn.q_a_layernorm.weight = nn.Parameter(
            loader.get_tensor(f"{prefix}.self_attn.q_a_layernorm.weight", device),
            requires_grad=False
        )
        attn.q_a_proj_weight = dequantize_int4(
            loader.get_tensor(f"{prefix}.self_attn.q_a_proj.weight.qweight", device),
            loader.get_tensor(f"{prefix}.self_attn.q_a_proj.weight.scales", device),
            loader.get_tensor(f"{prefix}.self_attn.q_a_proj.weight.shape"),
            group_size,
        ).to(device)
        
        attn.q_b_proj_weight = dequantize_int4(
            loader.get_tensor(f"{prefix}.self_attn.q_b_proj.weight.qweight", device),
            loader.get_tensor(f"{prefix}.self_attn.q_b_proj.weight.scales", device),
            loader.get_tensor(f"{prefix}.self_attn.q_b_proj.weight.shape"),
            group_size,
        ).to(device)
        
        # KV projections
        attn.kv_a_layernorm.weight = nn.Parameter(
            loader.get_tensor(f"{prefix}.self_attn.kv_a_layernorm.weight", device),
            requires_grad=False
        )
        attn.kv_a_proj_weight = dequantize_int4(
            loader.get_tensor(f"{prefix}.self_attn.kv_a_proj_with_mqa.weight.qweight", device),
            loader.get_tensor(f"{prefix}.self_attn.kv_a_proj_with_mqa.weight.scales", device),
            loader.get_tensor(f"{prefix}.self_attn.kv_a_proj_with_mqa.weight.shape"),
            group_size,
        ).to(device)
        
        attn.kv_b_proj_weight = dequantize_int4(
            loader.get_tensor(f"{prefix}.self_attn.kv_b_proj.weight.qweight", device),
            loader.get_tensor(f"{prefix}.self_attn.kv_b_proj.weight.scales", device),
            loader.get_tensor(f"{prefix}.self_attn.kv_b_proj.weight.shape"),
            group_size,
        ).to(device)
        
        # Output projection
        attn.o_proj_weight = dequantize_int4(
            loader.get_tensor(f"{prefix}.self_attn.o_proj.weight.qweight", device),
            loader.get_tensor(f"{prefix}.self_attn.o_proj.weight.scales", device),
            loader.get_tensor(f"{prefix}.self_attn.o_proj.weight.shape"),
            group_size,
        ).to(device)
        
        # MLP
        mlp = layer.mlp
        
        if layer.is_moe:
            # Router
            mlp.router_weight = dequantize_int4(
                loader.get_tensor(f"{prefix}.mlp.gate.weight.qweight", device),
                loader.get_tensor(f"{prefix}.mlp.gate.weight.scales", device),
                loader.get_tensor(f"{prefix}.mlp.gate.weight.shape"),
                group_size,
            ).to(device)
            
            if loader.has_key(f"{prefix}.mlp.gate.e_score_correction_bias"):
                mlp.router_bias = loader.get_tensor(
                    f"{prefix}.mlp.gate.e_score_correction_bias", device
                )
            
            # Shared expert (on GPU)
            mlp.shared_gate_weight = dequantize_int4(
                loader.get_tensor(f"{prefix}.mlp.shared_experts.gate_proj.weight.qweight", device),
                loader.get_tensor(f"{prefix}.mlp.shared_experts.gate_proj.weight.scales", device),
                loader.get_tensor(f"{prefix}.mlp.shared_experts.gate_proj.weight.shape"),
                group_size,
            ).to(device)
            
            mlp.shared_up_weight = dequantize_int4(
                loader.get_tensor(f"{prefix}.mlp.shared_experts.up_proj.weight.qweight", device),
                loader.get_tensor(f"{prefix}.mlp.shared_experts.up_proj.weight.scales", device),
                loader.get_tensor(f"{prefix}.mlp.shared_experts.up_proj.weight.shape"),
                group_size,
            ).to(device)
            
            mlp.shared_down_weight = dequantize_int4(
                loader.get_tensor(f"{prefix}.mlp.shared_experts.down_proj.weight.qweight", device),
                loader.get_tensor(f"{prefix}.mlp.shared_experts.down_proj.weight.scales", device),
                loader.get_tensor(f"{prefix}.mlp.shared_experts.down_proj.weight.shape"),
                group_size,
            ).to(device)
            
            # Routed experts (on CPU, dequantized lazily)
            loaded_experts = 0
            for expert_idx in range(config.n_routed_experts):
                exp_prefix = f"{prefix}.mlp.experts.{expert_idx}"
                
                # Check if this expert's tensors are valid
                qweight_key = f"{exp_prefix}.gate_proj.weight.qweight"
                if not loader.is_tensor_valid(qweight_key):
                    # Skip invalid experts (file truncated)
                    continue
                
                try:
                    gate = dequantize_int4(
                        loader.get_tensor(f"{exp_prefix}.gate_proj.weight.qweight"),
                        loader.get_tensor(f"{exp_prefix}.gate_proj.weight.scales"),
                        loader.get_tensor(f"{exp_prefix}.gate_proj.weight.shape"),
                        group_size,
                    )
                    up = dequantize_int4(
                        loader.get_tensor(f"{exp_prefix}.up_proj.weight.qweight"),
                        loader.get_tensor(f"{exp_prefix}.up_proj.weight.scales"),
                        loader.get_tensor(f"{exp_prefix}.up_proj.weight.shape"),
                        group_size,
                    )
                    down = dequantize_int4(
                        loader.get_tensor(f"{exp_prefix}.down_proj.weight.qweight"),
                        loader.get_tensor(f"{exp_prefix}.down_proj.weight.scales"),
                        loader.get_tensor(f"{exp_prefix}.down_proj.weight.shape"),
                        group_size,
                    )
                    
                    mlp.load_expert(expert_idx, gate, up, down)
                    loaded_experts += 1
                except Exception as e:
                    print(f"    [WARN] Failed to load expert {expert_idx}: {e}")
                    continue
            
            if loaded_experts < config.n_routed_experts:
                print(f"    Layer {layer_idx}: loaded {loaded_experts}/{config.n_routed_experts} experts")
        else:
            # Dense MLP
            mlp.gate_weight = dequantize_int4(
                loader.get_tensor(f"{prefix}.mlp.gate_proj.weight.qweight", device),
                loader.get_tensor(f"{prefix}.mlp.gate_proj.weight.scales", device),
                loader.get_tensor(f"{prefix}.mlp.gate_proj.weight.shape"),
                group_size,
            ).to(device)
            
            mlp.up_weight = dequantize_int4(
                loader.get_tensor(f"{prefix}.mlp.up_proj.weight.qweight", device),
                loader.get_tensor(f"{prefix}.mlp.up_proj.weight.scales", device),
                loader.get_tensor(f"{prefix}.mlp.up_proj.weight.shape"),
                group_size,
            ).to(device)
            
            mlp.down_weight = dequantize_int4(
                loader.get_tensor(f"{prefix}.mlp.down_proj.weight.qweight", device),
                loader.get_tensor(f"{prefix}.mlp.down_proj.weight.scales", device),
                loader.get_tensor(f"{prefix}.mlp.down_proj.weight.shape"),
                group_size,
            ).to(device)
    
    elapsed = time.time() - start_time
    print(f"\nModel loaded in {elapsed:.1f}s")
    
    # Memory stats
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    # Load tokenizer
    tokenizer = None
    if HAS_TOKENIZER:
        try:
            print("\nLoading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                "THUDM/glm-4-9b",
                trust_remote_code=True
            )
            print("Tokenizer loaded")
        except Exception as e:
            print(f"Could not load tokenizer: {e}")
    
    return model, tokenizer


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GLM-4.7-Flash INT4 Inference")
    parser.add_argument("--model", type=str, default="./models/glm-4.7-flash-int4")
    parser.add_argument("--prompt", type=str, default="def fibonacci(n):")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--expert-cache", type=int, default=8, help="Expert LRU cache size")
    args = parser.parse_args()
    
    print("="*60)
    print("GLM-4.7-Flash INT4 Inference Engine")
    print("="*60)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu} ({mem:.1f} GB)")
    else:
        print("No GPU available, using CPU")
        args.device = "cpu"
    
    # Config
    inf_config = InferenceConfig(
        model_path=args.model,
        device=args.device,
        expert_cache_size=args.expert_cache,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    
    # Load model
    model, tokenizer = load_model(inf_config)
    model.eval()
    
    if tokenizer is None:
        print("\n[ERROR] Tokenizer not available. Cannot run inference.")
        print("Install transformers: pip install transformers")
        return
    
    # Generate
    print("\n" + "="*60)
    print("GENERATION")
    print("="*60)
    print(f"Prompt: {args.prompt}")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}, Top-k: {args.top_k}")
    print("-"*60)
    
    # Tokenize
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(args.device)
    
    # Generate
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    elapsed = time.time() - start_time
    
    # Decode
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    new_tokens = output_ids.shape[1] - input_ids.shape[1]
    
    print(output_text)
    print("-"*60)
    print(f"Generated {new_tokens} tokens in {elapsed:.1f}s ({new_tokens/elapsed:.1f} tok/s)")
    
    # Cache stats
    print(f"\nExpert cache stats: {model.expert_cache.get_stats()}")


if __name__ == "__main__":
    main()
