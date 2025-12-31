"""
BitNet b1.58 - Ternary Weight Quantization with Straight-Through Estimator

This module implements the core BitLinear layer from the BitNet b1.58 paper.
Key innovations:
  1. Weights quantized to {-1, 0, +1} (1.58 bits per weight)
  2. Activations quantized to INT8 (8 bits)
  3. Straight-Through Estimator (STE) for gradient flow through quantization

Reference: "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
           https://arxiv.org/abs/2402.17764

Copyright (c) 2025 Distillix. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math


# =============================================================================
# Quantization Functions with STE
# =============================================================================

class STESign(torch.autograd.Function):
    """
    Straight-Through Estimator for sign function.
    
    Forward: sign(x) -> {-1, 0, +1}
    Backward: gradient passes through unchanged (identity)
    
    This is the critical trick that makes training possible.
    Without STE, gradients would be zero almost everywhere (sign has zero gradient).
    """
    
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        # Round to nearest integer in [-1, 0, 1]
        # Using round() gives us ternary: -1, 0, +1
        return torch.clamp(torch.round(x), -1, 1)
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        # STE: pass gradient through unchanged
        # This is the "straight-through" part
        return grad_output


class STERound(torch.autograd.Function):
    """
    Straight-Through Estimator for rounding to integers.
    Used for INT8 activation quantization.
    
    Forward: round(x) to nearest integer
    Backward: gradient passes through unchanged
    """
    
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        return torch.round(x)
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output


def ste_sign(x: Tensor) -> Tensor:
    """Apply STE sign function."""
    return STESign.apply(x)


def ste_round(x: Tensor) -> Tensor:
    """Apply STE round function."""
    return STERound.apply(x)


# =============================================================================
# Weight Quantization: FP32 -> Ternary {-1, 0, +1}
# =============================================================================

def quantize_weights_ternary(weights: Tensor, eps: float = 1e-8) -> Tuple[Tensor, Tensor]:
    """
    Quantize weights to ternary values {-1, 0, +1} using absmax scaling.
    
    Algorithm (from BitNet b1.58):
        1. Compute scale: gamma = mean(|W|)  [per-tensor]
        2. Normalize: W_norm = W / (gamma + eps)
        3. Quantize: W_q = round(clip(W_norm, -1, 1)) -> {-1, 0, +1}
    
    The use of mean(|W|) instead of max(|W|) gives better dynamic range.
    
    Args:
        weights: FP32 latent weights [out_features, in_features]
        eps: Small constant for numerical stability
    
    Returns:
        w_quant: Ternary weights {-1, 0, +1}
        scale: Scaling factor (gamma) for dequantization
    """
    # Per-tensor absmax scaling using mean absolute value
    # This is the BitNet b1.58 approach (better than max for ternary)
    scale = weights.abs().mean() + eps
    
    # Normalize to [-1, 1] range
    w_normalized = weights / scale
    
    # Quantize to {-1, 0, +1} with STE
    w_quant = ste_sign(w_normalized)
    
    return w_quant, scale


def quantize_weights_ternary_per_channel(
    weights: Tensor, 
    eps: float = 1e-8
) -> Tuple[Tensor, Tensor]:
    """
    Per-channel (per-output) weight quantization.
    Better accuracy but slightly more complex.
    
    Args:
        weights: [out_features, in_features]
    
    Returns:
        w_quant: Ternary weights
        scale: Per-channel scales [out_features, 1]
    """
    # Scale per output channel
    scale = weights.abs().mean(dim=1, keepdim=True) + eps
    w_normalized = weights / scale
    w_quant = ste_sign(w_normalized)
    
    return w_quant, scale


# =============================================================================
# Activation Quantization: FP32 -> INT8
# =============================================================================

def quantize_activations_int8(
    activations: Tensor, 
    eps: float = 1e-8
) -> Tuple[Tensor, Tensor]:
    """
    Quantize activations to INT8 range [-127, 127] using absmax scaling.
    
    Algorithm:
        1. Compute scale: alpha = max(|X|) per token (row-wise)
        2. Quantize: X_q = round(clip(X / alpha * 127, -128, 127))
    
    Per-token quantization is better than per-tensor for activations
    because different tokens can have very different magnitudes.
    
    Args:
        activations: Input activations [..., features]
        eps: Numerical stability constant
    
    Returns:
        a_quant: INT8 activations (stored as float for computation)
        scale: Per-token scales [..., 1]
    """
    # Per-token (row-wise) absmax
    scale = activations.abs().amax(dim=-1, keepdim=True) + eps
    
    # Scale to INT8 range and round
    a_normalized = activations / scale * 127.0
    a_quant = ste_round(torch.clamp(a_normalized, -128, 127))
    
    return a_quant, scale


def quantize_activations_int8_per_tensor(
    activations: Tensor,
    eps: float = 1e-8
) -> Tuple[Tensor, Tensor]:
    """
    Per-tensor activation quantization (simpler, slightly less accurate).
    """
    scale = activations.abs().max() + eps
    a_normalized = activations / scale * 127.0
    a_quant = ste_round(torch.clamp(a_normalized, -128, 127))
    
    return a_quant, scale


# =============================================================================
# BitLinear Layer - Drop-in replacement for nn.Linear
# =============================================================================

class BitLinear(nn.Module):
    """
    BitNet b1.58 Linear Layer.
    
    This is a drop-in replacement for nn.Linear that:
      - Stores latent weights in FP32 (for gradient updates)
      - Quantizes weights to {-1, 0, +1} during forward pass
      - Quantizes activations to INT8 during forward pass
      - Uses STE for backpropagation through quantization
    
    Memory during training:
      - Same as FP32 (latent weights must be FP32 for gradients)
    
    Memory during inference:
      - 1.58 bits per weight (vs 32 bits for FP32)
      - ~20x compression
    
    Compute during inference:
      - Matmul becomes: accumulate = sum(+input, -input, or 0)
      - No floating point multiplication needed
      - Massive speedup on specialized hardware
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to include bias (default: False, as in most modern LLMs)
        per_channel: Use per-channel weight quantization (more accurate)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        per_channel: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.per_channel = per_channel
        self.eps = eps
        
        # Latent weights in FP32 - these are what the optimizer updates
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Initialize weights using Kaiming uniform.
        
        Note: Some BitNet papers suggest special initialization,
        but Kaiming works well in practice for training from scratch.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with quantization.
        
        1. Quantize weights: FP32 -> {-1, 0, +1}
        2. Quantize activations: FP32 -> INT8
        3. Compute: Y = X_q @ W_q^T
        4. Rescale: Y_out = Y * (w_scale * a_scale / 127)
        
        The rescaling factor corrects for the quantization scales.
        """
        # Quantize weights to ternary
        if self.per_channel:
            w_quant, w_scale = quantize_weights_ternary_per_channel(self.weight, self.eps)
        else:
            w_quant, w_scale = quantize_weights_ternary(self.weight, self.eps)
        
        # Quantize activations to INT8 (per-token)
        a_quant, a_scale = quantize_activations_int8(x, self.eps)
        
        # Matrix multiplication with quantized values
        # Note: In actual deployment, this would use specialized int8/ternary kernels
        # During training, we simulate with float
        y = F.linear(a_quant, w_quant, None)
        
        # Rescale output
        # For per-channel: w_scale is [out_features, 1], broadcasts correctly
        # For per-tensor: w_scale is scalar
        rescale = (w_scale * a_scale) / 127.0
        y = y * rescale
        
        # Add bias if present
        if self.bias is not None:
            y = y + self.bias
        
        return y
    
    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'bias={self.bias is not None}, '
            f'per_channel={self.per_channel}'
        )


# =============================================================================
# BitLinear with RMSNorm Fusion (as in BitNet b1.58 paper)
# =============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    More efficient than LayerNorm (no mean subtraction).
    Used in Llama, Mistral, and BitNet architectures.
    
    Formula: x_norm = x / sqrt(mean(x^2) + eps) * weight
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: Tensor) -> Tensor:
        # Compute RMS
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return (x / rms) * self.weight


class BitLinearWithNorm(nn.Module):
    """
    BitLinear with fused RMSNorm (as recommended in BitNet b1.58).
    
    The paper suggests applying normalization before quantization
    to improve the dynamic range of quantized activations.
    
    Architecture:
        Input -> RMSNorm -> Quantize -> BitLinear -> Output
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        
        self.norm = RMSNorm(in_features)
        self.linear = BitLinear(in_features, out_features, bias=bias, eps=eps)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.norm(x))


# =============================================================================
# Utilities for Model Conversion
# =============================================================================

def convert_linear_to_bitlinear(
    module: nn.Module,
    exclude_names: Optional[list] = None,
) -> nn.Module:
    """
    Convert all nn.Linear layers to BitLinear.
    
    Use this to convert a pre-trained model to BitNet architecture.
    Note: For best results, fine-tune after conversion.
    
    Args:
        module: PyTorch module to convert
        exclude_names: List of layer names to exclude (e.g., ['lm_head', 'embed'])
    
    Returns:
        Converted module (modified in-place)
    """
    exclude_names = exclude_names or []
    
    for name, child in module.named_children():
        if name in exclude_names:
            continue
        
        if isinstance(child, nn.Linear):
            # Create BitLinear with same dimensions
            bitlinear = BitLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
            )
            # Copy weights
            bitlinear.weight.data = child.weight.data.clone()
            if child.bias is not None:
                bitlinear.bias.data = child.bias.data.clone()
            
            # Replace
            setattr(module, name, bitlinear)
        else:
            # Recurse
            convert_linear_to_bitlinear(child, exclude_names)
    
    return module


def count_ternary_params(module: nn.Module) -> Tuple[int, int]:
    """
    Count parameters that will be ternary-quantized vs full precision.
    
    Returns:
        (ternary_params, full_precision_params)
    """
    ternary = 0
    full_prec = 0
    
    for name, param in module.named_parameters():
        if 'bitlinear' in name.lower() or isinstance(
            dict(module.named_modules()).get(name.rsplit('.', 1)[0]), 
            BitLinear
        ):
            ternary += param.numel()
        else:
            full_prec += param.numel()
    
    return ternary, full_prec


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Quick sanity check
    print("Testing BitLinear layer...")
    
    # Create layer
    layer = BitLinear(768, 768)
    
    # Create input
    x = torch.randn(2, 16, 768)  # [batch, seq, hidden]
    
    # Forward pass
    y = layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Check gradients flow
    loss = y.sum()
    loss.backward()
    print(f"Weight grad exists: {layer.weight.grad is not None}")
    print(f"Weight grad norm: {layer.weight.grad.norm().item():.4f}")
    
    # Check weight statistics
    w_quant, w_scale = quantize_weights_ternary(layer.weight.data)
    unique_vals = torch.unique(w_quant)
    print(f"Quantized weight unique values: {unique_vals.tolist()}")
    print(f"Weight scale: {w_scale.item():.4f}")
    
    # Memory calculation
    params = sum(p.numel() for p in layer.parameters())
    fp32_mb = params * 4 / 1024 / 1024
    ternary_mb = params * 1.58 / 8 / 1024 / 1024  # 1.58 bits per param
    print(f"Parameters: {params:,}")
    print(f"FP32 memory: {fp32_mb:.2f} MB")
    print(f"Ternary memory (inference): {ternary_mb:.2f} MB")
    print(f"Compression ratio: {fp32_mb / ternary_mb:.1f}x")
    
    print("\nBitLinear layer test passed!")
