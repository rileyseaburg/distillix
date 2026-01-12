#!/usr/bin/env python3
"""
Distillix-300M: BitNet Knowledge Distillation

This script distills knowledge from a large teacher (Llama-3-8B or TinyLlama)
into a 300M parameter BitNet student model.

Key fixes from the original notebook:
1. Proper BitLinear with activation quantization and rescaling
2. Correct vocab size matching between teacher and student
3. PolarizedOptimizer to prevent weight collapse
4. Proper logit alignment when vocab sizes differ
5. Mixed precision training for stability

Usage:
    python distillix_300m_distillation.py --steps 5000 --batch_size 8

Copyright (c) 2025 Distillix. All Rights Reserved.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Tuple
import math

# =============================================================================
# BitNet Implementation (Proper version with STE)
# =============================================================================

class STESign(torch.autograd.Function):
    """Straight-Through Estimator for ternary quantization."""
    @staticmethod
    def forward(ctx, x):
        return torch.clamp(torch.round(x), -1, 1)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Pass gradient through unchanged


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
    BitNet b1.58 Linear Layer with proper quantization.
    
    Key differences from naive implementation:
    1. Uses mean(|W|) scaling (not max) - better for ternary
    2. Quantizes activations to INT8 with per-token scaling
    3. Proper rescaling of outputs
    4. STE for gradient flow
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Latent weights in FP32 for gradient updates
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        # 1. Quantize weights to ternary {-1, 0, +1}
        w_scale = self.weight.abs().mean() + 1e-8
        w_normalized = self.weight / w_scale
        w_quant = ste_sign(w_normalized)
        
        # 2. Quantize activations to INT8 (per-token scaling)
        a_scale = x.abs().amax(dim=-1, keepdim=True) + 1e-8
        a_normalized = x / a_scale * 127.0
        a_quant = ste_round(torch.clamp(a_normalized, -128, 127))
        
        # 3. Compute with quantized values
        y = F.linear(a_quant, w_quant, None)
        
        # 4. Rescale output
        y = y * (w_scale * a_scale / 127.0)
        
        # 5. Add bias
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
# Polarized Optimizer (Prevents Weight Collapse)
# =============================================================================

class PolarizedOptimizer:
    """
    Wrapper that prevents BitNet weight collapse.
    
    The Problem: Standard weight decay + ternary quantization causes weights
    to collapse to 0 (the "death zone").
    
    The Solution: Replace L2 regularization's convex bowl with a double-well
    potential that REPELS weights from 0.
    """
    
    DEFAULT_PATTERNS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                        'gate_proj', 'up_proj', 'down_proj']
    
    def __init__(
        self,
        optimizer,
        model: nn.Module,
        target_scale: float = 0.01,
        polarization_strength: float = 0.1,
        apply_to_patterns: Optional[list] = None,
    ):
        self.optimizer = optimizer
        self.model = model
        self.target_scale = target_scale
        self.polarization_strength = polarization_strength
        self.patterns = apply_to_patterns or self.DEFAULT_PATTERNS
        
        # Cache parameters to polarize
        self._polarized_params = []
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in self.patterns):
                self._polarized_params.append((name, param))
    
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._apply_polarization()
        return loss
    
    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    @torch.no_grad()
    def _apply_polarization(self):
        """Apply double-well potential to prevent collapse."""
        target = self.target_scale
        strength = self.polarization_strength
        
        for name, param in self._polarized_params:
            w = param.data
            
            # Dead zone: weights too close to 0
            dead_zone = w.abs() < (target * 0.5)
            danger_zone = (w.abs() >= target * 0.5) & (w.abs() < target)
            
            # Strong push in dead zone
            if dead_zone.any():
                push_direction = w[dead_zone].sign()
                zero_mask = push_direction == 0
                if zero_mask.any():
                    push_direction[zero_mask] = (
                        torch.randint_like(push_direction[zero_mask], 0, 2) * 2 - 1
                    ).float()
                param.data[dead_zone] += push_direction * strength * 2.0
            
            # Moderate push in danger zone
            if danger_zone.any():
                push_direction = w[danger_zone].sign()
                deficit = target - w[danger_zone].abs()
                param.data[danger_zone] += push_direction * deficit * strength
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups


# =============================================================================
# Model Conversion Utilities
# =============================================================================

def convert_to_bitlinear(model: nn.Module, exclude_patterns: list = None):
    """
    Convert nn.Linear layers to BitLinear, preserving weights.
    
    Args:
        model: Model to convert
        exclude_patterns: Layer name patterns to exclude (e.g., ['lm_head', 'embed'])
    """
    exclude_patterns = exclude_patterns or ['lm_head', 'embed', 'norm']
    
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Check if should be excluded
            if any(pat in name for pat in exclude_patterns):
                continue
            
            # Get parent module and attribute name
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            
            # Create BitLinear and copy weights
            bit_layer = BitLinear(
                module.in_features, 
                module.out_features, 
                bias=module.bias is not None
            )
            bit_layer.weight.data = module.weight.data.clone()
            if module.bias is not None:
                bit_layer.bias.data = module.bias.data.clone()
            
            setattr(parent, attr_name, bit_layer)
    
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =============================================================================
# Distillation Loss Functions
# =============================================================================

def kl_divergence_loss(student_logits, teacher_logits, temperature: float = 2.0):
    """
    KL Divergence loss for knowledge distillation.
    
    Uses temperature scaling to soften probability distributions,
    making it easier for the student to learn from teacher.
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # KL(teacher || student) = sum(teacher * log(teacher/student))
    loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    
    # Scale by temperature^2 as per Hinton et al.
    return loss * (temperature ** 2)


def align_logits(student_logits, teacher_logits, student_tokenizer, teacher_tokenizer):
    """
    Align logits when student and teacher have different vocab sizes.
    
    This is critical when distilling across different tokenizers.
    We project student logits to teacher vocab space or vice versa.
    """
    s_vocab = student_logits.size(-1)
    t_vocab = teacher_logits.size(-1)
    
    if s_vocab == t_vocab:
        return student_logits, teacher_logits
    
    # Use minimum vocab size
    min_vocab = min(s_vocab, t_vocab)
    return student_logits[..., :min_vocab], teacher_logits[..., :min_vocab]


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Distillix-300M Knowledge Distillation')
    parser.add_argument('--steps', type=int, default=5000, help='Training steps')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--teacher', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                        help='Teacher model (use smaller for testing)')
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoint every N steps')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--use_polarization', action='store_true', default=True,
                        help='Use polarized optimizer to prevent collapse')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("DISTILLIX-300M KNOWLEDGE DISTILLATION")
    print("=" * 60)
    
    # ==========================================================================
    # 1. Load Teacher Model
    # ==========================================================================
    print(f"\n[1/5] Loading Teacher: {args.teacher}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    # Try 4-bit quantization for larger teachers
    try:
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
    except Exception as e:
        print(f"  4-bit loading failed: {e}")
        print("  Falling back to standard loading...")
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    teacher_vocab_size = teacher.config.vocab_size
    print(f"  Teacher vocab size: {teacher_vocab_size}")
    print(f"  Teacher parameters: {sum(p.numel() for p in teacher.parameters()) / 1e9:.2f}B")
    
    # ==========================================================================
    # 2. Create Student Model
    # ==========================================================================
    print("\n[2/5] Creating Student (Distillix-300M)")
    
    from transformers import LlamaConfig, LlamaForCausalLM
    
    # Student config - use SAME vocab as teacher!
    student_config = LlamaConfig(
        vocab_size=teacher_vocab_size,  # CRITICAL: Match teacher vocab
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,  # GQA
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        use_cache=False,  # Disable for training
    )
    
    student = LlamaForCausalLM(student_config)
    
    # Convert to BitLinear (exclude embedding and head)
    print("  Injecting BitLinear layers...")
    student = convert_to_bitlinear(student, exclude_patterns=['embed', 'lm_head', 'norm'])
    
    student = student.to(device)
    total_params, trainable_params = count_parameters(student)
    print(f"  Student parameters: {total_params / 1e6:.1f}M")
    
    # ==========================================================================
    # 3. Setup Optimizer
    # ==========================================================================
    print("\n[3/5] Setting up Optimizer")
    
    # Use AdamW with NO weight decay (polarization handles regularization)
    base_optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,  # CRITICAL: No weight decay for BitNet
    )
    
    if args.use_polarization:
        print("  Using PolarizedOptimizer (anti-collapse)")
        optimizer = PolarizedOptimizer(
            base_optimizer,
            student,
            target_scale=0.01,
            polarization_strength=0.1,
        )
    else:
        optimizer = base_optimizer
    
    # Mixed precision
    scaler = GradScaler()
    
    # ==========================================================================
    # 4. Load Dataset
    # ==========================================================================
    print("\n[4/5] Loading Dataset (TinyStories)")
    
    from datasets import load_dataset
    
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    def collate_fn(batch):
        texts = [item['text'][:args.max_length * 4] for item in batch]  # Rough char limit
        return tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )
    
    # ==========================================================================
    # 5. Training Loop
    # ==========================================================================
    print(f"\n[5/5] Starting Training ({args.steps} steps)")
    print("=" * 60)
    
    student.train()
    step = 0
    running_loss = 0.0
    
    for batch in dataloader:
        if step >= args.steps:
            break
        
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Skip empty batches
        if input_ids.size(0) == 0:
            continue
        
        try:
            # Teacher forward (no grad, FP16)
            with torch.no_grad():
                with autocast():
                    teacher_outputs = teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    teacher_logits = teacher_outputs.logits
            
            # Student forward (with grad, mixed precision)
            with autocast():
                student_outputs = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                student_logits = student_outputs.logits
                
                # Align vocab sizes if needed
                s_logits, t_logits = align_logits(
                    student_logits, teacher_logits, None, None
                )
                
                # KL divergence loss
                loss = kl_divergence_loss(s_logits, t_logits, temperature=2.0)
            
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer.optimizer if hasattr(optimizer, 'optimizer') else optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Logging
            running_loss += loss.item()
            step += 1
            
            if step % 50 == 0:
                avg_loss = running_loss / 50
                print(f"Step {step:5d} | Loss: {avg_loss:.4f}")
                running_loss = 0.0
            
            # Checkpointing
            if step % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f"distillix-300m-step{step}.pt")
                torch.save({
                    'step': step,
                    'model_state_dict': student.state_dict(),
                    'config': student_config,
                }, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  OOM at step {step}, skipping batch...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    # ==========================================================================
    # 6. Final Save & Test
    # ==========================================================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    # Save final model
    final_path = os.path.join(args.output_dir, "distillix-300m-final.pt")
    torch.save({
        'step': step,
        'model_state_dict': student.state_dict(),
        'config': student_config,
    }, final_path)
    print(f"Final model saved: {final_path}")
    
    # Quick generation test
    print("\nGeneration Test:")
    student.eval()
    prompt = "Once upon a time"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        output = student.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Output: {generated_text}")


if __name__ == "__main__":
    main()
