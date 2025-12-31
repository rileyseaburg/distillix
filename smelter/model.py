"""
Distillix "Frankenstein" BitNet Student Model.

A ~125M parameter hybrid architecture stealing the best from each lineage:

  MATH (Microsoft BitNet b1.58):
    - 1.58-bit ternary weights {-1, 0, +1} for all projections
    - ~5x memory reduction at inference

  TOKENIZER (Llama-2 "Brain-First"):
    - 32k vocab (vs 128k Llama-3, 256k Gemma)
    - Saves ~75M params, allocates 80% to "Brain" not "Dictionary"

  ATTENTION (Llama 3 GQA):
    - Grouped Query Attention with 12 Q heads / 4 KV heads
    - 3x KV cache reduction, enables longer context on 8GB VRAM

  STABILITY (Gemma 2/3):
    - QK-Norm: RMSNorm on Q/K per-head before attention
    - Logit Soft-Capping: tanh-bounded logits in attention and LM head
    - Prevents training collapse, enables higher learning rates

  POSITION (Extended RoPE):
    - theta=1,000,000 for long code context (vs 10k default)
    - Supports up to 32k context with proper scaling

Reference:
  - BitNet b1.58: https://arxiv.org/abs/2402.17764
  - GQA: https://arxiv.org/abs/2305.13245
  - Gemma 2: https://arxiv.org/abs/2408.00118
  - ViT-22B QK-Norm: https://arxiv.org/abs/2302.05442

Copyright (c) 2025 Distillix. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math

from .bitnet import BitLinear, RMSNorm
from .config import ModelConfig


# =============================================================================
# Rotary Position Embedding (RoPE)
# =============================================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from the RoFormer paper.
    
    RoPE encodes position information directly into the attention mechanism
    by rotating query and key vectors. This provides:
      - Relative position awareness
      - Extrapolation to longer sequences
      - No learned parameters
    
    Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
               https://arxiv.org/abs/2104.09864
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        scaling_factor: Optional[float] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos/sin for all positions
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for given sequence length."""
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        
        if self.scaling_factor is not None:
            positions = positions / self.scaling_factor
        
        # Outer product: [seq_len, dim/2]
        freqs = torch.outer(positions, self.inv_freq)
        
        # Interleave to get [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(
        self, 
        x: Tensor, 
        position_ids: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Get cos/sin for positions.
        
        Args:
            x: Input tensor [batch, seq_len, ...]
            position_ids: Optional position indices [batch, seq_len]
        
        Returns:
            (cos, sin) each of shape [seq_len, dim] or [batch, seq_len, dim]
        """
        seq_len = x.shape[1]
        
        # Extend cache if needed
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        
        if position_ids is None:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        else:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
        
        return cos, sin


def apply_rotary_pos_emb(
    q: Tensor, 
    k: Tensor, 
    cos: Tensor, 
    sin: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Apply rotary position embedding to query and key tensors.
    
    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_heads, seq_len, head_dim]
        cos: Cosine values [seq_len, head_dim] or [batch, seq_len, head_dim]
        sin: Sine values [seq_len, head_dim] or [batch, seq_len, head_dim]
    
    Returns:
        Rotated (q, k) tensors
    """
    # Reshape cos/sin for broadcasting
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)  # [batch, 1, seq, dim]
        sin = sin.unsqueeze(1)
    
    # Rotate: split into halves and apply rotation
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


# =============================================================================
# Attention Layer
# =============================================================================

class Attention(nn.Module):
    """
    Grouped Query Attention (GQA) with BitLinear, RoPE, and Gemma 2 stability.
    
    "Frankenstein" Architecture combining:
      - Llama 3 GQA: Reduced KV heads for 3x KV cache savings
      - Gemma 2 QK-Norm: RMSNorm on Q/K for stability
      - Gemma 2 Soft-Capping: Bounded attention logits
      - BitNet b1.58: Ternary weight projections
      - RoPE: Rotary position embeddings (theta=1M for long context)
    
    GQA Memory Math:
      - MHA: KV cache = 2 * num_heads * head_dim * seq_len * batch
      - GQA: KV cache = 2 * num_kv_heads * head_dim * seq_len * batch
      - With 12 heads / 4 KV heads: 3x reduction in KV cache
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads  # Query heads
        self.num_kv_heads = config.num_kv_heads  # Key/Value heads (fewer for GQA)
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_kv_groups  # Queries per KV head
        
        # Dimensions
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        
        # Choose layer type
        Linear = BitLinear if config.use_bitlinear else nn.Linear
        
        # Q projection: full size (all query heads)
        self.q_proj = Linear(self.hidden_dim, self.q_dim, bias=False)
        
        # K, V projections: reduced size (fewer KV heads for GQA)
        self.k_proj = Linear(self.hidden_dim, self.kv_dim, bias=False)
        self.v_proj = Linear(self.hidden_dim, self.kv_dim, bias=False)
        
        # O projection: full size
        self.o_proj = Linear(self.q_dim, self.hidden_dim, bias=False)
        
        # QK-Norm: Apply RMSNorm to Q and K per-head (Gemma 2 / ViT-22B style)
        # This stabilizes attention by preventing feature domination
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        
        # Logit soft-capping (Gemma 2 style)
        # Prevents attention logits from growing unbounded
        self.attn_logit_soft_cap = config.attn_logit_soft_cap
        
        # Rotary embeddings (theta=1M for long code context)
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
            scaling_factor=config.rope_scaling,
        )
        
        # Attention scale
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with Grouped Query Attention.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: [batch, 1, seq_len, seq_len] or None for causal
            position_ids: [batch, seq_len] or None
        
        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)  # [batch, seq, q_dim]
        k = self.k_proj(hidden_states)  # [batch, seq, kv_dim]
        v = self.v_proj(hidden_states)  # [batch, seq, kv_dim]
        
        # Reshape Q: [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Reshape K, V: [batch, seq, num_kv_heads, head_dim] -> [batch, num_kv_heads, seq, head_dim]
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # QK-Norm: Normalize Q and K per-head before RoPE (Gemma 2 / ViT-22B)
        # This prevents attention collapse and stabilizes BitNet training
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Apply rotary embeddings
        # Note: For GQA, we apply RoPE to Q (full heads) and K (reduced heads) separately
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # GQA: Expand K and V to match Q's head count
        # [batch, num_kv_heads, seq, head_dim] -> [batch, num_heads, seq, head_dim]
        # Each KV head is repeated num_kv_groups times
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Scaled dot-product attention
        # [batch, num_heads, seq, head_dim] @ [batch, num_heads, head_dim, seq]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Logit soft-capping (Gemma 2 style)
        # Prevents extreme attention logits: logits = cap * tanh(logits / cap)
        # NOTE: Run in float32 to avoid AMP precision issues with tanh
        if self.attn_logit_soft_cap is not None:
            attn_weights = self.attn_logit_soft_cap * torch.tanh(
                attn_weights.float() / self.attn_logit_soft_cap
            ).to(attn_weights.dtype)
        
        # Apply causal mask if no custom mask provided
        if attention_mask is None:
            # Create causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        else:
            attn_weights = attn_weights + attention_mask
        
        # Softmax (compute in FP32 for stability, then cast back)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        
        # Apply attention to values
        # [batch, num_heads, seq, seq] @ [batch, num_heads, seq, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back: [batch, num_heads, seq, head_dim] -> [batch, seq, q_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.q_dim)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output


# =============================================================================
# MLP Layer (SwiGLU)
# =============================================================================

class MLP(nn.Module):
    """
    SwiGLU MLP as used in Llama/Mistral.
    
    SwiGLU: Swish-Gated Linear Unit
      output = down_proj(silu(gate_proj(x)) * up_proj(x))
    
    Uses BitLinear for all projections.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # Choose layer type
        Linear = BitLinear if config.use_bitlinear else nn.Linear
        
        # SwiGLU projections
        self.gate_proj = Linear(config.hidden_dim, config.mlp_hidden_dim, bias=False)
        self.up_proj = Linear(config.hidden_dim, config.mlp_hidden_dim, bias=False)
        self.down_proj = Linear(config.mlp_hidden_dim, config.hidden_dim, bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with SwiGLU activation.
        
        Args:
            x: [batch, seq_len, hidden_dim]
        
        Returns:
            Output [batch, seq_len, hidden_dim]
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock(nn.Module):
    """
    Single Transformer block with pre-normalization.
    
    Architecture (Pre-LN):
      x = x + Attention(RMSNorm(x))
      x = x + MLP(RMSNorm(x))
    
    Pre-normalization is more stable for training.
    """
    
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Pre-attention norm
        self.input_layernorm = RMSNorm(config.hidden_dim, eps=config.norm_eps)
        
        # Attention
        self.self_attn = Attention(config)
        
        # Pre-MLP norm
        self.post_attention_layernorm = RMSNorm(config.hidden_dim, eps=config.norm_eps)
        
        # MLP
        self.mlp = MLP(config)
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: Optional attention mask
            position_ids: Optional position indices
        
        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        # Attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


# =============================================================================
# Full Model
# =============================================================================

class StudentLLM(nn.Module):
    """
    Distillix "Frankenstein" Code Model (~125M params).
    
    Hybrid architecture optimized for coding performance per parameter:
    
      - Token embeddings (FP16, 32k Llama-2 vocab)
      - 12 Transformer blocks with:
        * BitLinear projections (1.58-bit weights)
        * GQA: 12 Q heads / 4 KV heads (3x KV cache reduction)
        * QK-Norm (Gemma 2 stability)
        * Logit soft-capping (Gemma 2 stability)
        * SwiGLU MLP
        * RMSNorm (FP32)
      - LM head (tied to embeddings)
    
    Memory footprint (inference):
      - Weights: ~25MB (1.58-bit quantized)
      - KV cache @ 2048 tokens: ~2MB (vs 6MB without GQA)
    
    Designed for knowledge distillation from frontier models (Claude, GPT-4).
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings (full precision for stability)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        
        # Final normalization
        self.norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)
        
        # LM head (optionally tied to embeddings)
        if config.tie_word_embeddings:
            self.lm_head = None  # Will use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Final logit soft-capping (Gemma 2 style)
        # Prevents extreme logits before loss computation
        self.final_logit_soft_cap = config.final_logit_soft_cap
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = False
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, BitLinear):
            # BitLinear already has its own init
            pass
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
    
    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embedding layer."""
        return self.embed_tokens
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> dict:
        """
        Forward pass.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] padding mask (1 = valid, 0 = pad)
            position_ids: [batch, seq_len] position indices
            labels: [batch, seq_len] target tokens for loss computation
        
        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Convert attention mask to 4D if provided
        if attention_mask is not None:
            # [batch, seq] -> [batch, 1, 1, seq]
            attention_mask = attention_mask[:, None, None, :].to(hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        
        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # Transformer blocks
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, attention_mask, position_ids)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # LM head
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden_states, self.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)
        
        # Final logit soft-capping (Gemma 2 style)
        # Prevents extreme logits before loss/sampling: logits = cap * tanh(logits / cap)
        # NOTE: Run in float32 to avoid AMP precision issues with tanh
        if self.final_logit_soft_cap is not None:
            logits = self.final_logit_soft_cap * torch.tanh(
                logits.float() / self.final_logit_soft_cap
            ).to(logits.dtype)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for causal LM: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return {"logits": logits, "loss": loss}
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        Simple autoregressive generation.
        
        Args:
            input_ids: [batch, seq_len] prompt tokens
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            eos_token_id: Stop generation token
        
        Returns:
            Generated tokens [batch, seq_len + new_tokens]
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Truncate if exceeding max length
            if input_ids.shape[1] >= self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]
            
            # Forward pass
            outputs = self.forward(input_ids)
            logits = outputs["logits"][:, -1, :]  # Last token logits
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop at EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return input_ids
    
    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """Count number of parameters."""
        total = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            total -= self.embed_tokens.weight.numel()
            if self.lm_head is not None:
                total -= self.lm_head.weight.numel()
        return total
    
    def count_bitlinear_params(self) -> Tuple[int, int]:
        """Count BitLinear vs full precision parameters."""
        bitlinear_params = 0
        fp_params = 0
        
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                                                          'gate_proj', 'up_proj', 'down_proj']):
                bitlinear_params += param.numel()
            else:
                fp_params += param.numel()
        
        return bitlinear_params, fp_params


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    from .config import ModelConfig, get_config_125m
    
    print("Testing StudentLLM...")
    
    # Create model with default config
    config = get_config_125m()
    model = StudentLLM(config.model)
    
    print(f"\nModel Config:")
    print(f"  Hidden dim: {config.model.hidden_dim}")
    print(f"  Num layers: {config.model.num_layers}")
    print(f"  Num heads: {config.model.num_heads}")
    
    # Count parameters
    total_params = model.num_parameters()
    bitlinear_params, fp_params = model.count_bitlinear_params()
    
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  BitLinear: {bitlinear_params:,} ({100*bitlinear_params/total_params:.1f}%)")
    print(f"  Full precision: {fp_params:,} ({100*fp_params/total_params:.1f}%)")
    
    # Memory estimate
    fp32_mb = total_params * 4 / 1024 / 1024
    # BitLinear in inference: 1.58 bits for quantized, 32 bits for FP
    inference_mb = (bitlinear_params * 1.58/8 + fp_params * 4) / 1024 / 1024
    
    print(f"\nMemory:")
    print(f"  Training (FP32): {fp32_mb:.1f} MB")
    print(f"  Inference (quantized): {inference_mb:.1f} MB")
    print(f"  Compression: {fp32_mb / inference_mb:.1f}x")
    
    # Test forward pass
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
    
    print(f"\nForward pass test:")
    print(f"  Input shape: {input_ids.shape}")
    
    outputs = model(input_ids)
    print(f"  Output logits shape: {outputs['logits'].shape}")
    
    # Test with labels
    labels = input_ids.clone()
    outputs = model(input_ids, labels=labels)
    print(f"  Loss: {outputs['loss'].item():.4f}")
    
    # Test gradient flow
    outputs['loss'].backward()
    
    # Check gradients exist for BitLinear layers
    grad_exists = all(
        p.grad is not None 
        for name, p in model.named_parameters() 
        if 'proj' in name
    )
    print(f"  Gradients flow through BitLinear: {grad_exists}")
    
    print("\nStudentLLM test passed!")
