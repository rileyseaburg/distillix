#!/usr/bin/env python3
"""
Superposition Distillation V2.1 - Production-Ready with All Corrections

V2.1 FIXES (from second review):
  1. BitLinear broadcasting fix for 2D/3D inputs
  2. Attention mask support + ignore_index in LM loss  
  3. Freeze embeddings + lm_head in Phase 2
  4. Store teacher layer indices in cache manifest
  5. Align student/teacher distill layers via mapping
  6. Stronger variance loss (std ratio matching)
  7. Learnable alpha scaling in BitLinear
  8. Uniform cache sampling (weighted by shard size)
  9. Gradient clipping in Phase 3
  10. RoPE head_dim parity assertion

Target Hardware: A100 80GB

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

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# BitNet Implementation V2.1 - Fixed Broadcasting + Learnable Alpha
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
    BitNet b1.58 Linear V2.1 with:
      - Per-channel weight scaling
      - Robust 2D/3D broadcasting (FIX #1)
      - Learnable alpha gain per channel (FIX #7)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        # FIX #7: Learnable per-channel gain (alpha)
        # This allows the layer to learn optimal scaling for ternary weights
        self.alpha = nn.Parameter(torch.ones(out_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Initialize alpha to 1.0 (identity scaling)
        nn.init.ones_(self.alpha)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Per-channel weight scaling
        w_scale = self.weight.abs().mean(dim=1, keepdim=True) + 1e-8  # [out, 1]
        w_quant = ste_sign(self.weight / w_scale)
        
        # Apply learnable alpha to quantized weights
        # This compensates for quantization error
        w_scaled = w_quant * self.alpha.unsqueeze(1)  # [out, in]
        
        # Per-token activation quantization
        a_scale = x.abs().amax(dim=-1, keepdim=True) + 1e-8  # [..., 1]
        a_quant = ste_round(torch.clamp(x / a_scale * 127.0, -128, 127))
        
        # Matrix multiply
        y = F.linear(a_quant, w_scaled, None)
        
        # FIX #1: Robust rescaling that handles both 2D and 3D inputs
        # w_scale: [out_features, 1] -> need [out_features] for broadcast
        # a_scale: [batch, seq, 1] (3D) or [batch, 1] (2D)
        # y: [batch, seq, out] (3D) or [batch, out] (2D)
        
        w_scale_flat = w_scale.squeeze(-1)  # [out_features]
        
        if x.dim() == 3:
            # 3D: [batch, seq, hidden] -> w_scale needs [1, 1, out]
            rescale = w_scale_flat.view(1, 1, -1) * a_scale / 127.0
        elif x.dim() == 2:
            # 2D: [batch, hidden] -> w_scale needs [1, out]
            rescale = w_scale_flat.view(1, -1) * a_scale / 127.0
        else:
            # Fallback for other dims
            rescale = w_scale_flat * a_scale / 127.0
        
        y = y * rescale
        
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
# Rotary Position Embedding with Safety Check
# =============================================================================

class RotaryEmbedding(nn.Module):
    """RoPE with head_dim parity assertion (FIX #10)."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        
        # FIX #10: Assert head_dim is even (RoPE requires this)
        assert dim % 2 == 0, f"RoPE requires even head_dim, got {dim}"
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
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
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# Student Model V2.1 with Attention Mask Support
# =============================================================================

class BitNetBlock(nn.Module):
    """Transformer block with proper attention masking (FIX #2)."""
    
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
        
        # FIX #10: Validate head_dim for RoPE
        assert self.head_dim % 2 == 0, f"head_dim must be even for RoPE, got {self.head_dim}"
        
        self.q_proj = BitLinear(hidden_dim, hidden_dim)
        self.k_proj = BitLinear(hidden_dim, hidden_dim)
        self.v_proj = BitLinear(hidden_dim, hidden_dim)
        self.o_proj = BitLinear(hidden_dim, hidden_dim)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len, rope_base)
        
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.gate_proj = BitLinear(hidden_dim, mlp_dim)
        self.up_proj = BitLinear(hidden_dim, mlp_dim)
        self.down_proj = BitLinear(mlp_dim, hidden_dim)
        
        self.input_norm = RMSNorm(hidden_dim)
        self.post_attn_norm = RMSNorm(hidden_dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Forward with proper attention masking.
        
        Args:
            x: [batch, seq, hidden]
            attention_mask: [batch, seq] where 1=valid, 0=pad (FIX #2)
        """
        batch, seq_len, _ = x.shape
        device = x.device
        
        residual = x
        x = self.input_norm(x)
        
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(seq_len, device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), 
            diagonal=1
        )
        attn = attn.masked_fill(causal_mask, float('-inf'))
        
        # FIX #2: Apply padding mask to keys
        if attention_mask is not None:
            # attention_mask: [batch, seq] -> [batch, 1, 1, seq] for key masking
            key_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(key_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(x.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_dim)
        out = self.o_proj(out)
        
        x = residual + out
        
        residual = x
        x = self.post_attn_norm(x)
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        x = residual + x
        
        return x


class BitNetStudentV2_1(nn.Module):
    """BitNet student V2.1 with attention mask propagation."""
    
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
        
        self.distill_layers = distill_layers or [
            num_layers // 4 - 1, 
            num_layers // 2 - 1, 
            num_layers - 1
        ]
        
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        
        self.layers = nn.ModuleList([
            BitNetBlock(hidden_dim, num_heads, mlp_ratio, max_seq_len, rope_base)
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(hidden_dim)
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
        Forward with attention mask support.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] where 1=valid, 0=pad
        """
        x = self.embed_tokens(input_ids)
        
        intermediate_states = []
        
        for i, layer in enumerate(self.layers):
            x = layer(x, attention_mask)  # FIX #2: Pass mask to each layer
            
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
        self.eval()
        
        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids)
            next_logits = outputs['logits'][:, -1, :] / temperature
            
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
# Deep Residual Projector (unchanged from V2)
# =============================================================================

class ResidualProjector(nn.Module):
    """Deep residual projector for representation decompression."""
    
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
        
        return self.output_proj(x)


# =============================================================================
# Hybrid Loss V2.1 with Std Ratio Matching (FIX #6)
# =============================================================================

class HybridDistillationLossV2_1(nn.Module):
    """
    Improved hybrid loss with std ratio matching (FIX #6).
    
    The original variance loss (relu(target_var - proj_var)) was weak.
    Std ratio matching ensures projected features have similar spread.
    """
    
    def __init__(
        self,
        cosine_weight: float = 0.4,
        mse_weight: float = 0.3,
        std_ratio_weight: float = 0.3,  # FIX #6: Replace variance with std ratio
    ):
        super().__init__()
        self.cosine_weight = cosine_weight
        self.mse_weight = mse_weight
        self.std_ratio_weight = std_ratio_weight
    
    def forward(
        self, 
        projected: torch.Tensor,
        target: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # FIX #2
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute hybrid loss, optionally masking padding.
        
        Args:
            projected: [batch, seq, teacher_dim]
            target: [batch, seq, teacher_dim]
            attention_mask: [batch, seq] where 1=valid, 0=pad
        """
        # Flatten to [batch*seq, dim]
        proj_flat = projected.reshape(-1, projected.shape[-1])
        target_flat = target.reshape(-1, target.shape[-1])
        
        # FIX #2: Mask out padding positions
        if attention_mask is not None:
            mask_flat = attention_mask.reshape(-1).bool()
            proj_flat = proj_flat[mask_flat]
            target_flat = target_flat[mask_flat]
        
        # Skip if all masked
        if proj_flat.shape[0] == 0:
            zero = torch.tensor(0.0, device=projected.device)
            return zero, {'cosine_loss': 0, 'mse_loss': 0, 'std_ratio_loss': 0}
        
        # 1. Cosine similarity loss
        cosine_sim = F.cosine_similarity(proj_flat, target_flat, dim=-1)
        cosine_loss = (1 - cosine_sim).mean()
        
        # 2. MSE loss
        mse_loss = F.mse_loss(proj_flat, target_flat)
        
        # 3. FIX #6: Std ratio matching (stronger than simple variance)
        # Match per-feature standard deviation ratio
        proj_std = proj_flat.std(dim=0) + 1e-6
        target_std = target_flat.std(dim=0) + 1e-6
        std_ratio_loss = ((proj_std / target_std) - 1).abs().mean()
        
        # Combine
        total_loss = (
            self.cosine_weight * cosine_loss +
            self.mse_weight * mse_loss +
            self.std_ratio_weight * std_ratio_loss
        )
        
        metrics = {
            'cosine_loss': cosine_loss.item(),
            'mse_loss': mse_loss.item(),
            'std_ratio_loss': std_ratio_loss.item(),
            'cosine_sim': cosine_sim.mean().item(),
        }
        
        return total_loss, metrics


# =============================================================================
# Disk Cache V2.1 with Uniform Sampling (FIX #8) and Teacher Indices (FIX #4)
# =============================================================================

class DiskShardedCacheV2_1:
    """
    Disk cache with:
      - Teacher layer indices stored (FIX #4)
      - Attention masks stored
      - Uniform sampling weighted by shard size (FIX #8)
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
            'shard_sizes': [],  # FIX #8: Track actual size of each shard
            'teacher_dim': None,
            'seq_len': None,
            'num_distill_layers': None,
            'teacher_layer_indices': None,  # FIX #4: Store actual teacher layer indices
            'pad_token_id': None,
        }
        
        self._current_shard_inputs = []
        self._current_shard_masks = []  # FIX #2: Store attention masks
        self._current_shard_targets = []
    
    def add_sample(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,  # FIX #2
        hidden_states: List[torch.Tensor],
        teacher_layer_indices: Optional[List[int]] = None,  # FIX #4
    ):
        """Add a sample with attention mask."""
        if self.manifest['teacher_dim'] is None:
            self.manifest['teacher_dim'] = hidden_states[0].shape[-1]
            self.manifest['seq_len'] = hidden_states[0].shape[1]
            self.manifest['num_distill_layers'] = len(hidden_states)
            if teacher_layer_indices:
                self.manifest['teacher_layer_indices'] = teacher_layer_indices  # FIX #4
        
        self._current_shard_inputs.append(input_ids.cpu())
        self._current_shard_masks.append(attention_mask.cpu())
        self._current_shard_targets.append([h.cpu().to(self.dtype) for h in hidden_states])
        
        self.manifest['num_samples'] += 1
        
        if len(self._current_shard_inputs) >= self.shard_size:
            self._flush_shard()
    
    def _flush_shard(self):
        if not self._current_shard_inputs:
            return
        
        shard_idx = self.manifest['num_shards']
        shard_path = self.cache_dir / f"shard_{shard_idx:04d}.pt"
        
        actual_size = len(self._current_shard_inputs)
        
        inputs = torch.cat(self._current_shard_inputs, dim=0)
        masks = torch.cat(self._current_shard_masks, dim=0)
        
        num_layers = len(self._current_shard_targets[0])
        targets = []
        for layer_idx in range(num_layers):
            layer_targets = torch.cat([t[layer_idx] for t in self._current_shard_targets], dim=0)
            targets.append(layer_targets)
        
        torch.save({
            'inputs': inputs,
            'attention_masks': masks,
            'targets': targets,
        }, shard_path)
        
        self.manifest['num_shards'] += 1
        self.manifest['shard_sizes'].append(actual_size)  # FIX #8
        self._current_shard_inputs = []
        self._current_shard_masks = []
        self._current_shard_targets = []
        
        self._save_manifest()
    
    def _save_manifest(self):
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def finalize(self):
        self._flush_shard()
        self._save_manifest()
        print(f"Cache finalized: {self.manifest['num_samples']} samples in {self.manifest['num_shards']} shards")
    
    def load_manifest(self):
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                self.manifest = json.load(f)
            return True
        return False
    
    def get_shard(self, shard_idx: int) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
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
        FIX #8: Sample uniformly over samples, not shards.
        
        Uses shard_sizes to weight shard selection.
        """
        shard_sizes = self.manifest.get('shard_sizes', [self.shard_size] * self.manifest['num_shards'])
        total_samples = sum(shard_sizes)
        
        # Weight by shard size for uniform sampling
        weights = torch.tensor([s / total_samples for s in shard_sizes])
        shard_idx = torch.multinomial(weights, 1).item()
        
        inputs, masks, targets = self.get_shard(shard_idx)
        
        shard_size = inputs.shape[0]
        indices = torch.randint(0, shard_size, (min(batch_size, shard_size),))
        
        batch_inputs = inputs[indices].to(device)
        batch_masks = masks[indices].to(device)
        batch_targets = [t[indices].to(device).float() for t in targets]
        
        return batch_inputs, batch_masks, batch_targets


# =============================================================================
# Distiller V2.1 with All Fixes
# =============================================================================

class SuperpositionDistillerV2_1:
    """
    V2.1 Distiller with all production fixes:
      - FIX #2: Attention mask support
      - FIX #3: Freeze embeddings + lm_head in Phase 2
      - FIX #4: Store/verify teacher layer indices
      - FIX #5: Align student/teacher layers via mapping
      - FIX #6: Std ratio loss
      - FIX #9: Gradient clipping in Phase 3
    """
    
    def __init__(
        self,
        student: BitNetStudentV2_1,
        teacher_dim: int,
        cache_dir: str = "./distill_cache",
        device: str = "cuda",
        projector_hidden: int = 4096,
        projector_layers: int = 4,
        pad_token_id: int = 0,
    ):
        self.student = student.to(device)
        self.teacher_dim = teacher_dim
        self.student_dim = student.hidden_dim
        self.device = device
        self.cache_dir = cache_dir
        self.pad_token_id = pad_token_id
        
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
        
        self.rep_loss_fn = HybridDistillationLossV2_1()
        self.cache = DiskShardedCacheV2_1(cache_dir)
    
    def compute_layer_mapping(
        self,
        num_teacher_layers: int,
    ) -> List[int]:
        """
        FIX #5: Compute aligned teacher layer indices.
        
        Maps student distill layers to proportionally equivalent teacher layers.
        E.g., student layer 3/12 -> teacher layer 6/24
        """
        student_layers = self.student.num_layers
        mapping = []
        
        for s_layer in self.student.distill_layers:
            # Proportional mapping
            t_layer = round((s_layer / (student_layers - 1)) * (num_teacher_layers - 1))
            mapping.append(t_layer)
        
        return mapping
    
    def cache_teacher_states(
        self,
        teacher_model,
        tokenizer,
        dataset,
        num_samples: int = 5000,
        max_length: int = 128,
    ):
        """Phase 1: Cache with attention masks and aligned layer indices."""
        print("=" * 60)
        print("PHASE 1: Caching Teacher Hidden States (V2.1)")
        print("=" * 60)
        
        # FIX #5: Compute aligned layer mapping
        num_teacher_layers = teacher_model.config.num_hidden_layers
        teacher_layer_indices = self.compute_layer_mapping(num_teacher_layers)
        
        print(f"Student distill layers: {self.student.distill_layers}")
        print(f"Mapped teacher layers: {teacher_layer_indices}")
        
        self.cache.manifest['pad_token_id'] = tokenizer.pad_token_id
        
        teacher_model.eval()
        count = 0
        
        for item in tqdm(dataset, total=num_samples, desc="Caching"):
            if count >= num_samples:
                break
            
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
            
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding='max_length',
            ).to(self.device)
            
            with torch.no_grad():
                outputs = teacher_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                )
                
                all_hidden = outputs.hidden_states
                # FIX #5: Use aligned teacher layer indices
                selected_hidden = [all_hidden[idx] for idx in teacher_layer_indices]
            
            # FIX #2: Store attention mask
            self.cache.add_sample(
                inputs.input_ids, 
                inputs.attention_mask,
                selected_hidden,
                teacher_layer_indices,
            )
            count += 1
        
        self.cache.finalize()
        
        print(f"Cached {count} samples")
        
        del teacher_model
        torch.cuda.empty_cache()
        gc.collect()
        print("Teacher unloaded")
    
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
        Phase 2 with:
          - FIX #2: Attention mask in loss
          - FIX #3: Freeze embeddings + lm_head
        """
        print("\n" + "=" * 60)
        print("PHASE 2: Superposition Training (V2.1)")
        print("=" * 60)
        
        if not self.cache.load_manifest():
            raise ValueError("No cache found!")
        
        # FIX #4: Verify layer count matches
        cached_layers = self.cache.manifest['num_distill_layers']
        student_layers = len(self.student.distill_layers)
        assert cached_layers == student_layers, \
            f"Cache has {cached_layers} layers but student expects {student_layers}"
        
        print(f"Loaded cache: {len(self.cache)} samples")
        print(f"Teacher layers used: {self.cache.manifest.get('teacher_layer_indices', 'unknown')}")
        
        # FIX #3: Freeze embeddings and lm_head in Phase 2
        for param in self.student.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.student.lm_head.parameters():
            param.requires_grad = False
        
        # Trainable: transformer blocks + projectors
        trainable_params = []
        for layer in self.student.layers:
            trainable_params.extend(layer.parameters())
        for proj in self.projectors:
            trainable_params.extend(proj.parameters())
        
        trainable_count = sum(p.numel() for p in trainable_params)
        total_count = sum(p.numel() for p in self.student.parameters())
        print(f"Training {trainable_count:,} / {total_count:,} student params ({100*trainable_count/total_count:.1f}%)")
        print("(Embeddings and LM head frozen)")
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler = GradScaler()
        
        self.student.train()
        for proj in self.projectors:
            proj.train()
        
        pad_token_id = self.cache.manifest.get('pad_token_id', 0)
        metrics_history = []
        
        progress = tqdm(range(num_steps), desc="Training")
        for step in progress:
            batch_in, batch_mask, batch_targets = self.cache.sample_batch(batch_size, self.device)
            
            with autocast():
                outputs = self.student(
                    batch_in,
                    attention_mask=batch_mask,
                    return_hidden_states=True,
                    return_intermediate=True,
                )
                
                rep_loss = 0.0
                rep_metrics = {}
                
                student_intermediates = outputs['intermediate_states']
                
                for layer_idx, (student_hidden, teacher_target, projector) in enumerate(
                    zip(student_intermediates, batch_targets, self.projectors)
                ):
                    projected = projector(student_hidden)
                    
                    # FIX #2: Pass attention mask to loss
                    layer_loss, layer_metrics = self.rep_loss_fn(
                        projected, teacher_target, attention_mask=batch_mask
                    )
                    rep_loss = rep_loss + layer_loss
                    
                    for k, v in layer_metrics.items():
                        rep_metrics[f'layer{layer_idx}_{k}'] = v
                
                rep_loss = rep_loss / len(self.projectors)
                
                # Auxiliary LM loss with ignore_index (FIX #2)
                logits = outputs['logits']
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch_in[:, 1:].contiguous()
                
                lm_loss = F.cross_entropy(
                    shift_logits.view(-1, self.student.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=pad_token_id,  # FIX #2
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
                avg_cos = sum(m.get('layer0_cosine_sim', 0) for m in metrics_history[-log_interval:]) / max(log_interval, 1)
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
        
        return metrics_history
    
    def train_lm_head(
        self,
        num_steps: int = 1000,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        unfreeze_last_n_blocks: int = 2,
        log_interval: int = 50,
    ):
        """Phase 3 with gradient clipping (FIX #9)."""
        print("\n" + "=" * 60)
        print(f"PHASE 3: LM Head + Last {unfreeze_last_n_blocks} Blocks (V2.1)")
        print("=" * 60)
        
        # Freeze everything
        for param in self.student.parameters():
            param.requires_grad = False
        
        # Unfreeze LM head
        for param in self.student.lm_head.parameters():
            param.requires_grad = True
        
        # Unfreeze last N blocks
        num_layers = len(self.student.layers)
        for i in range(num_layers - unfreeze_last_n_blocks, num_layers):
            for param in self.student.layers[i].parameters():
                param.requires_grad = True
        
        trainable_params = [p for p in self.student.parameters() if p.requires_grad]
        trainable = sum(p.numel() for p in trainable_params)
        total = sum(p.numel() for p in self.student.parameters())
        print(f"Training {trainable:,} / {total:,} parameters ({100*trainable/total:.1f}%)")
        
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        scaler = GradScaler()
        self.student.train()
        
        pad_token_id = self.cache.manifest.get('pad_token_id', 0)
        losses = []
        
        progress = tqdm(range(num_steps), desc="LM Head Training")
        for step in progress:
            batch_in, batch_mask, _ = self.cache.sample_batch(batch_size, self.device)
            
            with autocast():
                outputs = self.student(batch_in, attention_mask=batch_mask)
                logits = outputs['logits']
                
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch_in[:, 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.student.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=pad_token_id,
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
                avg_loss = sum(losses[-log_interval:]) / len(losses[-log_interval:])
                progress.set_postfix({"loss": f"{avg_loss:.4f}"})
        
        # Unfreeze everything
        for param in self.student.parameters():
            param.requires_grad = True
        
        final_loss = sum(losses[-100:]) / len(losses[-100:])
        print(f"\nFinal LM loss: {final_loss:.4f}")
        
        return losses
    
    def save(self, path: str):
        torch.save({
            'student': self.student.state_dict(),
            'projectors': [p.state_dict() for p in self.projectors],
            'config': {
                'student_dim': self.student_dim,
                'teacher_dim': self.teacher_dim,
                'vocab_size': self.student.vocab_size,
                'num_layers': self.student.num_layers,
                'distill_layers': self.student.distill_layers,
                'teacher_layer_indices': self.cache.manifest.get('teacher_layer_indices'),
            },
        }, path)
        print(f"Saved to {path}")
    
    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.student.load_state_dict(ckpt['student'])
        for proj, state in zip(self.projectors, ckpt['projectors']):
            proj.load_state_dict(state)
        print(f"Loaded from {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Superposition Distillation V2.1")
    
    parser.add_argument('--teacher', type=str, default='Qwen/Qwen2.5-Coder-1.5B-Instruct')
    parser.add_argument('--student-dim', type=int, default=768)
    parser.add_argument('--student-layers', type=int, default=12)
    parser.add_argument('--student-heads', type=int, default=12)
    
    parser.add_argument('--dataset', type=str, default='bigcode/starcoderdata')
    parser.add_argument('--dataset-subset', type=str, default='python')
    parser.add_argument('--num-cache', type=int, default=5000)
    parser.add_argument('--max-length', type=int, default=128)
    
    parser.add_argument('--phase2-steps', type=int, default=2000)
    parser.add_argument('--phase3-steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--aux-lm-weight', type=float, default=0.1)
    
    parser.add_argument('--cache-dir', type=str, default='./distill_cache_v2_1')
    parser.add_argument('--output', type=str, default='superposition_student_v2_1.pt')
    parser.add_argument('--skip-cache', action='store_true')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if not args.skip_cache:
        print("\nLoading teacher...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
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
        teacher.eval()
        
        teacher_dim = teacher.config.hidden_size
        vocab_size = teacher.config.vocab_size
        pad_token_id = tokenizer.pad_token_id
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        cache = DiskShardedCacheV2_1(args.cache_dir)
        cache.load_manifest()
        teacher_dim = cache.manifest['teacher_dim']
        vocab_size = tokenizer.vocab_size
        pad_token_id = cache.manifest.get('pad_token_id', tokenizer.pad_token_id)
        teacher = None
    
    print(f"\nCreating student...")
    student = BitNetStudentV2_1(
        vocab_size=vocab_size,
        hidden_dim=args.student_dim,
        num_layers=args.student_layers,
        num_heads=args.student_heads,
        max_seq_len=args.max_length * 2,
    )
    
    total_params = sum(p.numel() for p in student.parameters())
    print(f"Student: {total_params / 1e6:.1f}M params")
    print(f"Distill layers: {student.distill_layers}")
    
    distiller = SuperpositionDistillerV2_1(
        student, 
        teacher_dim, 
        cache_dir=args.cache_dir,
        device=device,
        pad_token_id=pad_token_id,
    )
    
    if not args.skip_cache:
        from datasets import load_dataset
        
        try:
            dataset = load_dataset(
                args.dataset, 
                args.dataset_subset,
                split="train", 
                streaming=True,
                trust_remote_code=True,
            )
        except:
            print("Falling back to TinyStories")
            dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        
        distiller.cache_teacher_states(
            teacher, tokenizer, dataset,
            num_samples=args.num_cache,
            max_length=args.max_length,
        )
    else:
        distiller.cache.load_manifest()
    
    distiller.train_superposition(
        num_steps=args.phase2_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        aux_lm_weight=args.aux_lm_weight,
    )
    
    distiller.train_lm_head(
        num_steps=args.phase3_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr / 10,
        unfreeze_last_n_blocks=2,
    )
    
    distiller.save(args.output)
    
    # Test
    print("\n" + "=" * 60)
    print("Generation Test")
    print("=" * 60)
    
    student.eval()
    prompts = ["def fibonacci(n):", "def reverse_list(lst):"]
    
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            output_ids = student.generate(input_ids, max_new_tokens=50)
        
        print(f"\nPrompt: {prompt}")
        print(f"Output: {tokenizer.decode(output_ids[0], skip_special_tokens=True)[:200]}")


if __name__ == "__main__":
    main()
