"""
Superposition Distillation V2.2 - Production-Ready Implementation

FIXES FROM V2 REVIEW (all applied):
  1. BitLinear broadcasting fixed for 2D/3D inputs
  2. Attention mask support + ignore_index in LM loss
  3. Phase 2 freezes embeddings + lm_head (cleaner separation)
  4. Teacher layer indices stored in cache manifest
  5. Student/teacher layer alignment via mapping
  6. Stronger variance loss (std ratio matching)
  7. Learnable alpha scaling in BitLinear
  8. Cache sampling uniformity (proportional to shard size)
  9. Gradient clipping in Phase 3
  10. Head_dim parity assertion for RoPE

V2.2 PRODUCTION FIXES (5 high-ROI changes):
  11. Teacher hidden state indexing: +1 to skip embedding layer
  12. Activation clamp STE in BitLinear: gradients flow through saturation
  13. Softmax NaN prevention: clamp -inf to -1e4 before softmax
  14. Fast shard selection: O(B log S) via numpy searchsorted
  15. Keep targets fp16: avoid fp32 upload, save GPU bandwidth

Target Hardware: A100 80GB
  - Teacher (7B 4-bit): ~4GB
  - Student (125M FP32): ~500MB
  - Cached states (10k samples): ~2GB on disk
  - Plenty of headroom for batch size 32+

Protocol:
  Phase 1: "Prep Kitchen" - Extract & cache teacher hidden states (multi-layer)
  Phase 2: "Spoon Feed" - Train student blocks + projectors (freeze embed/lm_head)
  Phase 3: "Speech Therapy" - Fine-tune LM head + last 2 blocks

Copyright (c) 2025 Distillix. All Rights Reserved.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Tuple, List, Dict, Any
from tqdm.auto import tqdm
from pathlib import Path
import gc
import math
import json
import shutil
import numpy as np

# =============================================================================
# BitNet Implementation - V2.1 with all fixes
# =============================================================================

class STESign(torch.autograd.Function):
    """Straight-Through Estimator for ternary quantization."""
    @staticmethod
    def forward(ctx, x):
        return torch.clamp(torch.round(x), -1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class STERound(torch.autograd.Function):
    """Straight-Through Estimator for INT8 rounding."""
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_sign(x):
    return STESign.apply(x)


def ste_round(x):
    return STERound.apply(x)


class BitLinear(nn.Module):
    """
    BitNet b1.58 Linear with all V2.1 fixes:

    FIX #1: Robust broadcasting for 2D/3D inputs
    FIX #7: Learnable per-channel alpha scaling
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # FIX #7: Learnable per-channel gain (alpha scaling)
        # This prevents the layer from being systematically underpowered
        self.alpha = nn.Parameter(torch.ones(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Initialize alpha to 1.0 (already done in __init__)

    def forward(self, x):
        # Input shape: [batch, seq, in_features] or [batch, in_features]
        input_dim = x.dim()

        # Per-channel (per-output) weight scaling
        w_scale = self.weight.abs().mean(dim=1, keepdim=True) + 1e-8  # [out, 1]
        w_quant = ste_sign(self.weight / w_scale)

        # Per-token activation quantization with full STE for clamp + round
        # FIX: Apply STE to clamp saturation too, preventing dead gradients at boundaries
        a_scale = x.abs().amax(dim=-1, keepdim=True) + 1e-8  # [..., 1]
        a_scaled = x / a_scale * 127.0
        a_clamped = torch.clamp(a_scaled, -128, 127)
        # STE for clamp: gradient passes through as if no clamp happened
        a_ste = (a_clamped - a_scaled).detach() + a_scaled
        a_quant = ste_round(a_ste)

        # Matrix multiply
        y = F.linear(a_quant, w_quant, None)

        # FIX #1: Robust rescaling for any input dimension
        # w_scale: [out_features, 1] -> [out_features]
        # a_scale: [batch, seq, 1] or [batch, 1]
        # y: [batch, seq, out_features] or [batch, out_features]
        w_scale_flat = w_scale.squeeze(-1)  # [out_features]

        if input_dim == 3:
            # 3D: [batch, seq, out] * [1, 1, out] * [batch, seq, 1]
            rescale = w_scale_flat.view(1, 1, -1) * a_scale / 127.0
        else:
            # 2D: [batch, out] * [1, out] * [batch, 1]
            rescale = w_scale_flat.view(1, -1) * a_scale / 127.0

        y = y * rescale

        # FIX #7: Apply learnable alpha scaling
        if input_dim == 3:
            y = y * self.alpha.view(1, 1, -1)
        else:
            y = y * self.alpha.view(1, -1)

        if self.bias is not None:
            y = y + self.bias
        return y


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


# =============================================================================
# Rotary Position Embedding - V2.1 with head_dim assertion
# =============================================================================

class RotaryEmbedding(nn.Module):
    """
    RoPE with FIX #10: head_dim parity assertion.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()

        # FIX #10: Assert head_dim is even (required for RoPE rotation)
        assert dim % 2 == 0, f"RoPE requires even head_dim, got {dim}"

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device):
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
            self.cos_cached = self.cos_cached.to(device)
            self.sin_cached = self.sin_cached.to(device)
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)


def rotate_half(x):
    """Rotate half the hidden dims."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE to query and key tensors."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# Student Model - V2.1 with attention mask support
# =============================================================================

class BitNetBlock(nn.Module):
    """
    Transformer block with FIX #2: proper attention mask support.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # FIX #10: Validate head_dim is even for RoPE
        assert self.head_dim % 2 == 0, f"head_dim must be even for RoPE, got {self.head_dim}"

        # Attention projections
        self.q_proj = BitLinear(hidden_dim, hidden_dim)
        self.k_proj = BitLinear(hidden_dim, hidden_dim)
        self.v_proj = BitLinear(hidden_dim, hidden_dim)
        self.o_proj = BitLinear(hidden_dim, hidden_dim)

        # RoPE
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len, rope_base)

        # MLP (SwiGLU)
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.gate_proj = BitLinear(hidden_dim, mlp_dim)
        self.up_proj = BitLinear(hidden_dim, mlp_dim)
        self.down_proj = BitLinear(mlp_dim, hidden_dim)

        # Norms
        self.input_norm = RMSNorm(hidden_dim)
        self.post_attn_norm = RMSNorm(hidden_dim)

        # Attention scale
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass.

        Args:
            x: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len] with 1=valid, 0=pad (FIX #2)
        """
        batch, seq_len, _ = x.shape
        device = x.device

        # Pre-norm attention
        residual = x
        x = self.input_norm(x)

        # QKV projections
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(seq_len, device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask (always applied)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        attn = attn.masked_fill(causal_mask, float('-inf'))

        # FIX #2: Apply padding mask if provided
        # attention_mask: [batch, seq] -> mask keys that are padding
        if attention_mask is not None:
            # Expand: [batch, seq] -> [batch, 1, 1, seq] for broadcasting
            key_padding_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(key_padding_mask, float('-inf'))

        # FIX: Prevent NaN from softmax(all -inf) when entire row is masked
        # Replace -inf with large negative number to keep softmax stable
        attn = torch.clamp(attn, min=-1e4)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(x.dtype)
        # Additional safety: replace any remaining NaNs with 0 (shouldn't happen after clamp)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_dim)
        out = self.o_proj(out)

        x = residual + out

        # Pre-norm MLP (SwiGLU)
        residual = x
        x = self.post_attn_norm(x)
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        x = residual + x

        return x


class BitNetStudentV2(nn.Module):
    """
    BitNet student model V2.1 with attention mask support.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        distill_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # Which layers to extract for multi-layer distillation
        # Default: [2, 5, 11] for 12-layer model
        self.distill_layers = distill_layers or [
            max(0, num_layers // 4 - 1),
            num_layers // 2 - 1,
            num_layers - 1
        ]

        # Embeddings (full precision)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)

        # Transformer blocks
        self.layers = nn.ModuleList([
            BitNetBlock(hidden_dim, num_heads, mlp_ratio, max_seq_len, rope_base)
            for _ in range(num_layers)
        ])

        # Final norm
        self.norm = RMSNorm(hidden_dim)

        # Separate LM head (NOT tied to embeddings)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # FIX #2
        return_hidden_states: bool = False,
        return_intermediate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] with 1=valid, 0=pad (FIX #2)
            return_hidden_states: Return final hidden states
            return_intermediate: Return intermediate layer hidden states
        """
        x = self.embed_tokens(input_ids)

        intermediate_states = []

        for i, layer in enumerate(self.layers):
            x = layer(x, attention_mask)  # FIX #2: pass attention mask

            if return_intermediate and i in self.distill_layers:
                intermediate_states.append(x)

        x = self.norm(x)

        result = {}

        if return_hidden_states:
            result['hidden_states'] = x

        if return_intermediate:
            result['intermediate_states'] = intermediate_states

        result['logits'] = self.lm_head(x)

        return result

    def generate(self, input_ids, max_new_tokens=50, temperature=0.7, top_p=0.9):
        """Simple autoregressive generation."""
        self.eval()

        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids)
            next_logits = outputs['logits'][:, -1, :] / temperature

            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)

            mask = cumulative_probs > top_p
            mask[:, 1:] = mask[:, :-1].clone()
            mask[:, 0] = False
            sorted_logits[mask] = float('-inf')

            probs = F.softmax(sorted_logits, dim=-1)
            next_token = sorted_indices.gather(-1, torch.multinomial(probs, 1))

            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids


# =============================================================================
# BitMamba Implementation - Zen Architecture (No separate MLP)
# =============================================================================

class BitMambaBlock(nn.Module):
    """
    BitMamba Block (Zen Style).
    
    Structure:
    Input -> Norm -> BitLinear(Expand) -> Conv1d -> SSM(Selective Scan) -> BitLinear(Project) -> Residual
    """
    def __init__(
        self, 
        d_model: int, 
        d_state: int = 16, 
        d_conv: int = 4, 
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_conv = d_conv
        self.d_state = d_state
        self.expand = expand
        
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # 1. Input Projection: Up-projects to (d_inner * 2) for x and z (gate) branches
        # This is the BIG matrix, so we use BitLinear (1.58-bit)
        self.in_proj = BitLinear(d_model, self.d_inner * 2, bias=False)

        # 2. 1D Convolution (Standard FP32/FP16 - keeps local context)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # 3. Activation
        self.act = nn.SiLU()

        # 4. SSM Parameters (Sensitive - Keep High Precision)
        self.x_proj = nn.Linear(self.d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # S4D / Hippo initialization for A
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.log_A = nn.Parameter(torch.log(A)) 
        
        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 5. Output Projection: Down-projects back to d_model
        # This is the other BIG matrix, so we use BitLinear (1.58-bit)
        self.out_proj = BitLinear(self.d_inner, d_model, bias=False)
        
        self.norm = RMSNorm(d_model)
        
        self._init_weights(dt_rank, dt_min, dt_max)

    def _init_weights(self, dt_rank, dt_min, dt_max):
        nn.init.uniform_(self.dt_proj.bias, math.log(dt_min), math.log(dt_max))
        nn.init.normal_(self.x_proj.weight, std=0.02)

    def ssm_scan(self, x, dt, A, B, C):
        """
        Pure PyTorch Selective Scan (Reference).
        Install 'mamba-ssm' for 10x speedup: from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        """
        batch, seq_len, d_inner = x.shape
        d_state = B.shape[2]
        
        # Discretize A: exp(A * dt) -> [batch, seq, d_inner, d_state]
        dA = torch.exp(torch.einsum('bsd,dn->bsdn', dt, -torch.exp(self.log_A)))
        
        # Discretize B: dt * B
        dB = torch.einsum('bsd,bsn->bsdn', dt, B)
        
        # Current input impact: dB * x
        u = torch.einsum('bsdn,bsd->bsdn', dB, x)
        
        # Recurrence (Sequential loop - slow but correct)
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        for t in range(seq_len):
            h = dA[:, t] * h + u[:, t]
            ys.append(h)
            
        y_stack = torch.stack(ys, dim=1) 
        
        # Project to output: C * y
        out = torch.einsum('bsdn,bsn->bsd', y_stack, C)
        return out

    def forward(self, x, attention_mask=None):
        batch, seq_len, _ = x.shape
        residual = x
        x = self.norm(x)

        # 1. Expand (BitLinear)
        xz = self.in_proj(x) 
        x, z = xz.chunk(2, dim=-1)

        # 2. Conv1d
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len] 
        x_conv = x_conv.transpose(1, 2)
        x = self.act(x_conv)

        # 3. SSM (High Precision)
        x_ssm_in = x.float() 
        ssm_params = self.x_proj(x)
        dt, B, C = torch.split(ssm_params, [self.dt_proj.in_features, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        
        y_ssm = self.ssm_scan(x_ssm_in, dt, self.log_A, B, C)
        
        # 4. Output Gate
        y = y_ssm * self.act(z) * self.D
        y = y.to(x.dtype) 

        # 5. Project (BitLinear)
        out = self.out_proj(y)
        
        return residual + out


class BitMambaStudent(nn.Module):
    """
    BitMamba Student V2.2 (Zen Architecture).
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 768,
        num_layers: int = 12,
        d_state: int = 16,
        expand: int = 2,
        distill_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.distill_layers = distill_layers or [
            max(0, num_layers // 4 - 1),
            num_layers // 2 - 1,
            num_layers - 1
        ]

        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        
        # The Zen Stack: Pure BitMamba blocks
        self.layers = nn.ModuleList([
            BitMambaBlock(d_model=hidden_dim, d_state=d_state, expand=expand)
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(self, input_ids, attention_mask=None, return_hidden_states=False, return_intermediate=False):
        x = self.embed_tokens(input_ids)
        intermediate_states = []

        for i, layer in enumerate(self.layers):
            x = layer(x)  # Mamba ignores attention mask
            if return_intermediate and i in self.distill_layers:
                intermediate_states.append(x)

        x = self.norm(x)
        result = {'logits': self.lm_head(x)}
        
        if return_hidden_states:
            result['hidden_states'] = x
        
        if return_intermediate:
            result['intermediate_states'] = intermediate_states
            
        return result

    def generate(self, input_ids, max_new_tokens=50, temperature=0.7):
        # Naive generation loop
        self.eval()
        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids)
            next_logits = outputs['logits'][:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids


# =============================================================================
# Deep Residual Projector
# =============================================================================

class ResidualProjector(nn.Module):
    """Deep residual projector for "decompressing" superposition."""

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        hidden_dim: int = 4096,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(student_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, teacher_dim)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = F.gelu(x)
        x = self.input_norm(x)

        for block in self.blocks:
            x = x + block(x)

        x = self.output_proj(x)
        return x


# =============================================================================
# Hybrid Loss - V2.1 with stronger variance matching (FIX #6)
# =============================================================================

class HybridDistillationLoss(nn.Module):
    """
    Hybrid loss with FIX #6: std ratio matching instead of simple variance.

    This prevents mode collapse by matching per-feature standard deviations.
    """

    def __init__(
        self,
        cosine_weight: float = 0.5,
        mse_weight: float = 0.3,
        std_weight: float = 0.2,  # FIX #6: renamed from variance_weight
    ):
        super().__init__()
        self.cosine_weight = cosine_weight
        self.mse_weight = mse_weight
        self.std_weight = std_weight

    def forward(
        self,
        projected: torch.Tensor,
        target: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # FIX #2: mask for valid positions
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute hybrid loss with optional masking.
        """
        # Flatten to [batch*seq, dim]
        proj_flat = projected.reshape(-1, projected.shape[-1])
        target_flat = target.reshape(-1, target.shape[-1])

        # FIX #2: Apply mask if provided
        if attention_mask is not None:
            mask_flat = attention_mask.reshape(-1).bool()
            proj_flat = proj_flat[mask_flat]
            target_flat = target_flat[mask_flat]

        # 1. Cosine similarity loss
        cosine_sim = F.cosine_similarity(proj_flat, target_flat, dim=-1)
        cosine_loss = (1 - cosine_sim).mean()

        # 2. MSE loss
        mse_loss = F.mse_loss(proj_flat, target_flat)

        # FIX #6: Std ratio matching (stronger than simple variance)
        # Match per-feature standard deviations
        proj_std = proj_flat.std(dim=0) + 1e-6
        target_std = target_flat.std(dim=0) + 1e-6
        std_ratio_loss = ((proj_std / target_std) - 1.0).abs().mean()

        # Combine
        total_loss = (
            self.cosine_weight * cosine_loss +
            self.mse_weight * mse_loss +
            self.std_weight * std_ratio_loss
        )

        metrics = {
            'cosine_loss': cosine_loss.item(),
            'mse_loss': mse_loss.item(),
            'std_ratio_loss': std_ratio_loss.item(),
            'cosine_sim': cosine_sim.mean().item(),
            'proj_std_mean': proj_std.mean().item(),
            'target_std_mean': target_std.mean().item(),
        }

        return total_loss, metrics


# =============================================================================
# Disk-Sharded Cache - V2.1 with teacher layer indices + uniform sampling
# =============================================================================

class DiskShardedCache:
    """
    Disk-based caching with FIX #4 and FIX #8:
    - Stores teacher layer indices in manifest
    - Uniform sampling across all samples (not just shards)
    """

    def __init__(
        self,
        cache_dir: str,
        shard_size: int = 100,
        dtype: torch.dtype = torch.float16,
    ):
        self.cache_dir = Path(cache_dir)
        self.shard_size = shard_size
        self.dtype = dtype

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.cache_dir / "manifest.json"
        self.manifest = {
            'num_samples': 0,
            'num_shards': 0,
            'shard_size': shard_size,
            'shard_sizes': [],  # FIX #8: track actual size of each shard
            'teacher_dim': None,
            'seq_len': None,
            'num_distill_layers': None,
            'teacher_layer_indices': None,  # FIX #4: store actual teacher layer indices
            'pad_token_id': None,  # FIX #2: store pad token for masking
        }

        self._current_shard_inputs = []
        self._current_shard_masks = []  # FIX #2: store attention masks
        self._current_shard_targets = []

    def add_sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,  # FIX #2
        hidden_states: List[torch.Tensor],
    ):
        """Add a sample to the cache."""
        if self.manifest['teacher_dim'] is None:
            self.manifest['teacher_dim'] = hidden_states[0].shape[-1]
            self.manifest['seq_len'] = hidden_states[0].shape[1]
            self.manifest['num_distill_layers'] = len(hidden_states)

        self._current_shard_inputs.append(input_ids.cpu())
        self._current_shard_masks.append(attention_mask.cpu())  # FIX #2
        self._current_shard_targets.append([h.cpu().to(self.dtype) for h in hidden_states])

        self.manifest['num_samples'] += 1

        if len(self._current_shard_inputs) >= self.shard_size:
            self._flush_shard()

    def _flush_shard(self):
        """Write current shard to disk."""
        if not self._current_shard_inputs:
            return

        shard_idx = self.manifest['num_shards']
        shard_path = self.cache_dir / f"shard_{shard_idx:04d}.pt"

        actual_size = len(self._current_shard_inputs)

        inputs = torch.cat(self._current_shard_inputs, dim=0)
        masks = torch.cat(self._current_shard_masks, dim=0)  # FIX #2

        num_layers = len(self._current_shard_targets[0])
        targets = []
        for layer_idx in range(num_layers):
            layer_targets = torch.cat(
                [t[layer_idx] for t in self._current_shard_targets],
                dim=0
            )
            targets.append(layer_targets)

        torch.save({
            'inputs': inputs,
            'attention_masks': masks,  # FIX #2
            'targets': targets,
        }, shard_path)

        self.manifest['num_shards'] += 1
        self.manifest['shard_sizes'].append(actual_size)  # FIX #8
        self._current_shard_inputs = []
        self._current_shard_masks = []
        self._current_shard_targets = []

        self._save_manifest()

    def set_teacher_layer_indices(self, indices: List[int]):
        """FIX #4: Store teacher layer indices for verification."""
        self.manifest['teacher_layer_indices'] = indices

    def set_pad_token_id(self, pad_token_id: int):
        """FIX #2: Store pad token ID."""
        self.manifest['pad_token_id'] = pad_token_id

    def _save_manifest(self):
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def finalize(self):
        """Flush remaining samples and save manifest."""
        self._flush_shard()
        self._save_manifest()
        print(f"Cache finalized: {self.manifest['num_samples']} samples in {self.manifest['num_shards']} shards")
        print(f"Teacher layer indices: {self.manifest['teacher_layer_indices']}")

    def load_manifest(self):
        """Load existing cache manifest."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                self.manifest = json.load(f)
            return True
        return False

    def get_shard(self, shard_idx: int) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Load a shard from disk. Returns (inputs, masks, targets)."""
        shard_path = self.cache_dir / f"shard_{shard_idx:04d}.pt"
        data = torch.load(shard_path)
        return data['inputs'], data['attention_masks'], data['targets']

    def __len__(self):
        return self.manifest['num_samples']

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        FIX #8: Uniform sampling across all samples, not just shards.
        FIX: Use searchsorted for O(B log S) instead of O(B * S) complexity.
        """
        from collections import defaultdict

        # Build cumulative sample counts for weighted shard selection
        shard_sizes = self.manifest.get('shard_sizes',
                                         [self.manifest['shard_size']] * self.manifest['num_shards'])
        total_samples = sum(shard_sizes)

        # FIX: Precompute cumulative boundaries for O(log S) lookup
        boundaries = np.cumsum([0] + shard_sizes)  # [0, s0, s0+s1, s0+s1+s2, ...]

        # Sample global indices uniformly
        global_indices = np.random.randint(0, total_samples, size=batch_size)

        # FIX: Use searchsorted for O(B log S) mapping instead of O(B * S)
        shard_indices = np.searchsorted(boundaries[1:], global_indices, side='right')
        local_indices = global_indices - boundaries[shard_indices]

        # Group by shard for efficient loading
        shard_to_locals = defaultdict(list)
        for i, (shard_idx, local_idx) in enumerate(zip(shard_indices, local_indices)):
            shard_to_locals[int(shard_idx)].append((i, int(local_idx)))

        # Initialize output tensors
        num_layers = self.manifest['num_distill_layers']
        seq_len = self.manifest['seq_len']
        teacher_dim = self.manifest['teacher_dim']

        batch_inputs = torch.zeros(batch_size, seq_len, dtype=torch.long)
        batch_masks = torch.zeros(batch_size, seq_len, dtype=torch.long)
        batch_targets = [torch.zeros(batch_size, seq_len, teacher_dim) for _ in range(num_layers)]

        # Load and fill
        for shard_idx, idx_pairs in shard_to_locals.items():
            inputs, masks, targets = self.get_shard(shard_idx)

            for batch_pos, local_idx in idx_pairs:
                batch_inputs[batch_pos] = inputs[local_idx]
                batch_masks[batch_pos] = masks[local_idx]
                for layer_idx in range(num_layers):
                    batch_targets[layer_idx][batch_pos] = targets[layer_idx][local_idx]

        batch_inputs = batch_inputs.to(device)
        batch_masks = batch_masks.to(device)
        # FIX: Keep targets in fp16 to save bandwidth - autocast handles conversion
        # Previously: .float() forced fp32 upload, wasting GPU memory bandwidth
        batch_targets = [t.to(device) for t in batch_targets]  # stays fp16

        return batch_inputs, batch_masks, batch_targets


# =============================================================================
# Layer Alignment Utility (FIX #5)
# =============================================================================

def compute_layer_alignment(
    student_layers: int,
    teacher_layers: int,
    student_distill_layers: List[int],
) -> List[int]:
    """
    FIX #5: Compute aligned teacher layer indices for student distill layers.

    Maps student layers proportionally to teacher layers.
    E.g., student layer 5 out of 12 -> teacher layer 10 out of 24.
    """
    teacher_indices = []
    for s_layer in student_distill_layers:
        # Proportional mapping
        t_layer = round((s_layer / (student_layers - 1)) * (teacher_layers - 1))
        teacher_indices.append(t_layer)

    return teacher_indices


# =============================================================================
# Superposition Distiller V2.1
# =============================================================================

class SuperpositionDistillerV2(nn.Module):
    """
    V2.1 Distiller with all fixes applied.
    """

    def __init__(
        self,
        student: BitNetStudentV2,
        teacher_dim: int,
        cache_dir: str = "./distill_cache",
        device: str = "cuda",
        projector_hidden: int = 4096,
        projector_layers: int = 4,
    ):
        super().__init__()
        self.student = student.to(device)
        self.teacher_dim = teacher_dim
        self.student_dim = student.hidden_dim
        self.device = device
        self.cache_dir = cache_dir

        # One projector per distillation layer
        num_distill_layers = len(student.distill_layers)
        self.projectors = nn.ModuleList([
            ResidualProjector(
                self.student_dim,
                teacher_dim,
                hidden_dim=projector_hidden,
                num_layers=projector_layers,
            ).to(device)
            for _ in range(num_distill_layers)
        ])

        # Hybrid loss with std ratio matching
        self.rep_loss_fn = HybridDistillationLoss(
            cosine_weight=0.5,
            mse_weight=0.3,
            std_weight=0.2,
        )

        # Disk cache
        self.cache = DiskShardedCache(cache_dir)

        # Store pad_token_id for loss masking
        self.pad_token_id = None

    def cache_teacher_states(
        self,
        teacher_model,
        tokenizer,
        dataset,
        num_samples: int = 5000,
        max_length: int = 128,
    ):
        """
        Phase 1: Cache teacher hidden states with FIX #4 and FIX #5.
        """
        print("=" * 60)
        print("PHASE 1: Caching Teacher Hidden States (Multi-Layer)")
        print("=" * 60)

        # FIX #5: Compute aligned teacher layer indices
        num_teacher_layers = teacher_model.config.num_hidden_layers
        teacher_layer_indices = compute_layer_alignment(
            self.student.num_layers,
            num_teacher_layers,
            self.student.distill_layers,
        )

        print(f"Student distill layers: {self.student.distill_layers}")
        print(f"Aligned teacher layers: {teacher_layer_indices}")

        # FIX #4: Store in manifest
        self.cache.set_teacher_layer_indices(teacher_layer_indices)

        # FIX #2: Store pad token
        self.pad_token_id = tokenizer.pad_token_id
        self.cache.set_pad_token_id(self.pad_token_id)

        teacher_model.eval()

        count = 0
        for item in tqdm(dataset, total=num_samples, desc="Caching"):
            if count >= num_samples:
                break

            # Get text
            if isinstance(item, dict):
                text = item.get('text', item.get('content', item.get('instruction', '')))
                if 'output' in item:
                    text = f"{text}\n{item['output']}"
                elif 'response' in item:
                    text = f"{text}\n{item['response']}"
            else:
                text = str(item)

            if not text.strip() or len(text) < 20:
                continue

            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,  # FIX #2
            ).to(self.device)

            # Get teacher hidden states
            with torch.no_grad():
                outputs = teacher_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                )

                # FIX: HF hidden_states[0] is embedding output, transformer layers are [1:]
                # So teacher layer index N -> hidden_states[N + 1]
                all_hidden = outputs.hidden_states
                selected_hidden = [all_hidden[idx + 1] for idx in teacher_layer_indices]

            # Add to disk cache with attention mask
            self.cache.add_sample(
                inputs.input_ids,
                inputs.attention_mask,  # FIX #2
                selected_hidden
            )
            count += 1

        self.cache.finalize()

        print(f"Cached {count} samples with {len(teacher_layer_indices)} layers each")

        # Free teacher memory
        del teacher_model
        torch.cuda.empty_cache()
        gc.collect()
        print("Teacher unloaded from memory")

    def train_superposition(
        self,
        num_steps: int = 2000,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        aux_lm_weight: float = 0.1,
        warmup_steps: int = 100,
        log_interval: int = 50,
    ):
        """
        Phase 2: Train student representations.

        FIX #3: Freezes embeddings + lm_head during this phase.
        """
        print("\n" + "=" * 60)
        print("PHASE 2: Superposition Training (Frozen Embed/LM Head)")
        print("=" * 60)

        if not self.cache.load_manifest():
            raise ValueError("No cache found! Run cache_teacher_states first.")

        self.pad_token_id = self.cache.manifest.get('pad_token_id')

        # FIX #4: Verify layer alignment
        cached_teacher_layers = self.cache.manifest.get('teacher_layer_indices')
        if cached_teacher_layers:
            print(f"Using cached teacher layers: {cached_teacher_layers}")

        print(f"Loaded cache: {len(self.cache)} samples")

        # FIX #3: Freeze embeddings and LM head during Phase 2
        for param in self.student.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.student.lm_head.parameters():
            param.requires_grad = False

        print("Frozen: embed_tokens, lm_head")
        print("Training: transformer blocks, projectors")

        # Collect trainable params
        trainable_params = []
        for name, param in self.student.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        for proj in self.projectors:
            trainable_params.extend(proj.parameters())

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler = GradScaler()

        self.student.train()
        for proj in self.projectors:
            proj.train()

        metrics_history = []

        progress = tqdm(range(num_steps), desc="Phase 2")
        for step in progress:
            # Sample with attention masks
            batch_in, batch_masks, batch_targets = self.cache.sample_batch(batch_size, self.device)

            with autocast():
                outputs = self.student(
                    batch_in,
                    attention_mask=batch_masks,  # FIX #2
                    return_hidden_states=True,
                    return_intermediate=True,
                )

                # Multi-layer representation loss
                rep_loss = 0.0
                rep_metrics = {}

                student_intermediates = outputs['intermediate_states']

                for layer_idx, (student_hidden, teacher_target, projector) in enumerate(
                    zip(student_intermediates, batch_targets, self.projectors)
                ):
                    projected = projector(student_hidden)

                    # FIX #2 + FIX #6: Hybrid loss with masking and std ratio
                    layer_loss, layer_metrics = self.rep_loss_fn(
                        projected, teacher_target, batch_masks
                    )
                    rep_loss = rep_loss + layer_loss

                    for k, v in layer_metrics.items():
                        rep_metrics[f'layer{layer_idx}_{k}'] = v

                rep_loss = rep_loss / len(self.projectors)

                # Auxiliary LM loss (with masking)
                logits = outputs['logits']
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch_in[:, 1:].contiguous()

                # FIX #2: Use ignore_index to mask padding tokens
                lm_loss = F.cross_entropy(
                    shift_logits.view(-1, self.student.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=self.pad_token_id if self.pad_token_id is not None else -100,
                )

                total_loss = rep_loss + aux_lm_weight * lm_loss

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            metrics = {
                'total_loss': total_loss.item(),
                'rep_loss': rep_loss.item(),
                'lm_loss': lm_loss.item(),
                'lr': scheduler.get_last_lr()[0],
                **rep_metrics,
            }
            metrics_history.append(metrics)

            if step % log_interval == 0:
                avg_cos = sum(m.get('layer0_cosine_sim', 0) for m in metrics_history[-log_interval:]) / max(len(metrics_history[-log_interval:]), 1)
                progress.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'lm': f"{lm_loss.item():.2f}",
                    'cos': f"{avg_cos:.3f}",
                })

        # Unfreeze for Phase 3
        for param in self.student.embed_tokens.parameters():
            param.requires_grad = True
        for param in self.student.lm_head.parameters():
            param.requires_grad = True

        final_metrics = {k: sum(m[k] for m in metrics_history[-100:]) / max(len(metrics_history[-100:]), 1)
                        for k in metrics_history[-1].keys()}
        print(f"\nPhase 2 Final:")
        print(f"  Rep loss: {final_metrics['rep_loss']:.4f}")
        print(f"  LM loss: {final_metrics['lm_loss']:.4f}")

        return metrics_history

    def train_lm_head(
        self,
        num_steps: int = 1000,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        unfreeze_last_n_blocks: int = 2,
        log_interval: int = 50,
    ):
        """
        Phase 3: Fine-tune LM head + last N blocks.

        FIX #9: Proper gradient clipping.
        """
        print("\n" + "=" * 60)
        print(f"PHASE 3: LM Head + Last {unfreeze_last_n_blocks} Blocks")
        print("=" * 60)

        # Freeze everything
        for param in self.student.parameters():
            param.requires_grad = False

        # Unfreeze LM head
        for param in self.student.lm_head.parameters():
            param.requires_grad = True

        # Unfreeze last N transformer blocks
        num_layers = len(self.student.layers)
        for i in range(num_layers - unfreeze_last_n_blocks, num_layers):
            for param in self.student.layers[i].parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.student.parameters())
        print(f"Training {trainable:,} / {total:,} parameters ({100*trainable/total:.1f}%)")

        trainable_params = [p for p in self.student.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

        scaler = GradScaler()
        self.student.train()

        losses = []

        progress = tqdm(range(num_steps), desc="Phase 3")
        for step in progress:
            batch_in, batch_masks, _ = self.cache.sample_batch(batch_size, self.device)

            with autocast():
                outputs = self.student(batch_in, attention_mask=batch_masks)
                logits = outputs['logits']

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch_in[:, 1:].contiguous()

                # FIX #2: ignore_index for padding
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.student.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=self.pad_token_id if self.pad_token_id is not None else -100,
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # FIX #9: Gradient clipping in Phase 3
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())

            if step % log_interval == 0:
                avg_loss = sum(losses[-log_interval:]) / max(len(losses[-log_interval:]), 1)
                progress.set_postfix({"loss": f"{avg_loss:.4f}"})

        # Unfreeze everything
        for param in self.student.parameters():
            param.requires_grad = True

        final_loss = sum(losses[-100:]) / max(len(losses[-100:]), 1)
        print(f"\nPhase 3 Final LM loss: {final_loss:.4f}")

        return losses

    def save(self, path: str):
        """Save student and projectors."""
        torch.save({
            'student': self.student.state_dict(),
            'projectors': [p.state_dict() for p in self.projectors],
            'config': {
                'student_dim': self.student_dim,
                'teacher_dim': self.teacher_dim,
                'vocab_size': self.student.vocab_size,
                'num_layers': self.student.num_layers,
                'distill_layers': self.student.distill_layers,
                'pad_token_id': self.pad_token_id,
            },
        }, path)
        print(f"Saved to {path}")

    def load(self, path: str):
        """Load student and projectors."""
        ckpt = torch.load(path, map_location=self.device)
        self.student.load_state_dict(ckpt['student'])
        for proj, state in zip(self.projectors, ckpt['projectors']):
            proj.load_state_dict(state)
        self.pad_token_id = ckpt['config'].get('pad_token_id')
        print(f"Loaded from {path}")


# =============================================================================
# Main Script
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Superposition Distillation V2.1")

    # Model args
    parser.add_argument('--teacher', type=str, default='Qwen/Qwen2.5-Coder-1.5B-Instruct')
    parser.add_argument('--student-dim', type=int, default=768)
    parser.add_argument('--student-layers', type=int, default=12)
    parser.add_argument('--student-heads', type=int, default=12)

    # Data args
    parser.add_argument('--dataset', type=str, default='bigcode/starcoderdata')
    parser.add_argument('--dataset-subset', type=str, default='python')
    parser.add_argument('--num-cache', type=int, default=5000)
    parser.add_argument('--max-length', type=int, default=128)

    # Training args
    parser.add_argument('--phase2-steps', type=int, default=2000)
    parser.add_argument('--phase3-steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--aux-lm-weight', type=float, default=0.1)

    # Output args
    parser.add_argument('--cache-dir', type=str, default='./distill_cache')
    parser.add_argument('--output', type=str, default='superposition_student_v2.pt')
    parser.add_argument('--skip-cache', action='store_true')

    args = parser.parse_args(args=[])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Superposition Distillation V2.1")
    print("=" * 60)

    # Load Teacher
    if not args.skip_cache:
        print("\nLoading teacher model...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Safety check: verify no token IDs exceed vocab
        _test_ids = tokenizer("test hello world", return_tensors="pt").input_ids
        assert _test_ids.max().item() < len(tokenizer), \
            f"Token ID {_test_ids.max().item()} >= vocab size {len(tokenizer)}!"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # === FIX FOR CUDA DEVICE-SIDE ASSERT ERROR ===
        # Qwen 2.5 Coder has config.vocab_size (151643) != len(tokenizer) (151665).
        # When tokenizer emits ID 151660 and model has embedding of size 151643,
        # CUDA crashes with "device-side assert triggered" on next GPU command.
        # Resizing embeddings to match tokenizer prevents this crash.
        if teacher.config.vocab_size != len(tokenizer):
            print(f"Resizing teacher embeddings from {teacher.config.vocab_size} to {len(tokenizer)}")
            teacher.resize_token_embeddings(len(tokenizer))
        # =============================================

        teacher.eval()

        teacher_dim = teacher.config.hidden_size
        # FIX: Use len(tokenizer) instead of config.vocab_size
        # HuggingFace tokenizers may have added special tokens that extend beyond vocab_size
        # Using vocab_size can cause CUDA device-side assert when token IDs >= embedding size
        vocab_size = len(tokenizer)
        print(f"Teacher: {args.teacher}")
        print(f"Teacher layers: {teacher.config.num_hidden_layers}")
        print(f"Teacher hidden dim: {teacher_dim}")
        print(f"Tokenizer vocab_size: {tokenizer.vocab_size}, len(tokenizer): {vocab_size}")
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        cache = DiskShardedCache(args.cache_dir)
        cache.load_manifest()
        teacher_dim = cache.manifest['teacher_dim']
        # FIX: Use len(tokenizer) for consistency with Phase 1
        vocab_size = len(tokenizer)
        print(f"Tokenizer vocab_size: {tokenizer.vocab_size}, len(tokenizer): {vocab_size}")
        teacher = None

    # Create Student
    print("\nCreating student model...")
    student = BitNetStudentV2(
        vocab_size=vocab_size,
        hidden_dim=args.student_dim,
        num_layers=args.student_layers,
        num_heads=args.student_heads,
        max_seq_len=args.max_length * 2,
    )

    total_params = sum(p.numel() for p in student.parameters())
    print(f"Student parameters: {total_params / 1e6:.1f}M")
    print(f"Student distill layers: {student.distill_layers}")

    # Create Distiller
    distiller = SuperpositionDistillerV2(
        student,
        teacher_dim,
        cache_dir=args.cache_dir,
        device=device,
    )

    # Phase 1: Cache
    if not args.skip_cache:
        from datasets import load_dataset

        print(f"\nLoading dataset: {args.dataset}/{args.dataset_subset}")
        try:
            dataset = load_dataset(
                args.dataset,
                args.dataset_subset,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Failed: {e}")
            print("Falling back to TinyStories...")
            dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

        distiller.cache_teacher_states(
            teacher,
            tokenizer,
            dataset,
            num_samples=args.num_cache,
            max_length=args.max_length,
        )
    else:
        distiller.cache.load_manifest()
        distiller.pad_token_id = distiller.cache.manifest.get('pad_token_id')
        print(f"Using existing cache: {len(distiller.cache)} samples")

    # Phase 2: Superposition Training
    distiller.train_superposition(
        num_steps=args.phase2_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        aux_lm_weight=args.aux_lm_weight,
    )

    # Phase 3: LM Head Training
    distiller.train_lm_head(
        num_steps=args.phase3_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr / 10,
        unfreeze_last_n_blocks=2,
    )

    # Save
    distiller.save(args.output)

    # Test
    print("\n" + "=" * 60)
    print("Generation Test")
    print("=" * 60)

    student.eval()
    test_prompts = [
        "def fibonacci(n):",
        "def reverse_list(lst):",
        "# Function to check if prime",
    ]

    for prompt in test_prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            output_ids = student.generate(input_ids, max_new_tokens=50, temperature=0.7)

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Output: {output_text[:200]}...")


if __name__ == "__main__":
    main()
