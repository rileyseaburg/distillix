#!/usr/bin/env python3
"""
Superposition Distillation: The "Spoon Feed" Protocol

This implements the key insight: force a small BitNet model to reconstruct
the high-dimensional hidden states of a large teacher model.

The Theory:
  - Teacher (e.g., Qwen-7B): Thinks in 3584-dimensional FP16 space
  - Student (e.g., 125M BitNet): Thinks in 768-dimensional 1.58-bit space
  - Projector: Learns to "unzip" student's compressed thoughts to teacher's space

Why this works:
  1. 1.58-bit weights are naturally sparse/orthogonal
  2. Sparse vectors can encode MORE features via superposition
  3. The projector acts as a "decompressor" that unpacks the student's thoughts

Protocol:
  Phase 1: "Prep Kitchen" - Cache teacher hidden states (teacher then deleted)
  Phase 2: "Spoon Feed" - Train student to match cached states (fast, no teacher)
  Phase 3: "Speech Therapy" - Train LM head to decode the new representations

Copyright (c) 2025 Distillix. All Rights Reserved.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Tuple, List
from tqdm.auto import tqdm
import gc
import math

# =============================================================================
# BitNet Implementation
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
    """BitNet b1.58 Linear with proper quantization."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, x):
        # Quantize weights
        w_scale = self.weight.abs().mean() + 1e-8
        w_quant = ste_sign(self.weight / w_scale)
        
        # Quantize activations  
        a_scale = x.abs().amax(dim=-1, keepdim=True) + 1e-8
        a_quant = ste_round(torch.clamp(x / a_scale * 127.0, -128, 127))
        
        # Compute and rescale
        y = F.linear(a_quant, w_quant, None)
        y = y * (w_scale * a_scale / 127.0)
        
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
# Superposition Projector (The "Decompressor")
# =============================================================================

class SuperpositionProjector(nn.Module):
    """
    Projects student's compressed representations to teacher's high-dim space.
    
    This is the "unzip" algorithm - it learns to decode the student's
    superposition-encoded features back into the teacher's representation space.
    
    Architecture: MLP with expansion then projection
        student_dim -> 2*student_dim (expand) -> teacher_dim (project)
    """
    
    def __init__(self, student_dim: int, teacher_dim: int, expansion_factor: int = 2):
        super().__init__()
        hidden_dim = student_dim * expansion_factor
        
        self.net = nn.Sequential(
            nn.Linear(student_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, teacher_dim),
        )
        
        # Initialize for stable training
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.net(x)


# =============================================================================
# Student Model (BitNet Transformer or Mamba)
# =============================================================================

class BitNetBlock(nn.Module):
    """Single transformer block with BitLinear layers."""
    
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Attention
        self.q_proj = BitLinear(hidden_dim, hidden_dim)
        self.k_proj = BitLinear(hidden_dim, hidden_dim)
        self.v_proj = BitLinear(hidden_dim, hidden_dim)
        self.o_proj = BitLinear(hidden_dim, hidden_dim)
        
        # MLP
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.gate_proj = BitLinear(hidden_dim, mlp_dim)
        self.up_proj = BitLinear(hidden_dim, mlp_dim)
        self.down_proj = BitLinear(mlp_dim, hidden_dim)
        
        # Norms
        self.input_norm = RMSNorm(hidden_dim)
        self.post_attn_norm = RMSNorm(hidden_dim)
    
    def forward(self, x, attention_mask=None):
        batch, seq_len, _ = x.shape
        
        # Pre-norm attention
        residual = x
        x = self.input_norm(x)
        
        # QKV
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
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


class BitNetStudent(nn.Module):
    """
    BitNet student model for superposition distillation.
    
    This model outputs hidden states (not logits) during training,
    which are then projected to match the teacher's hidden states.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Embeddings (full precision)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            BitNetBlock(hidden_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(hidden_dim)
        
        # LM head (tied to embeddings by default)
        self.lm_head = None  # Will use embed_tokens.weight
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
    
    def forward(self, input_ids, return_hidden_states: bool = True):
        """
        Forward pass.
        
        Args:
            input_ids: [batch, seq_len]
            return_hidden_states: If True, return hidden states instead of logits
        
        Returns:
            If return_hidden_states: [batch, seq_len, hidden_dim]
            Else: [batch, seq_len, vocab_size]
        """
        x = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        if return_hidden_states:
            return x
        else:
            # LM head (tied weights)
            logits = F.linear(x, self.embed_tokens.weight)
            return logits
    
    def generate(self, input_ids, max_new_tokens=50, temperature=0.7, top_p=0.9):
        """Simple autoregressive generation."""
        self.eval()
        
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids, return_hidden_states=False)
            next_logits = logits[:, -1, :] / temperature
            
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
# Superposition Distillation Trainer
# =============================================================================

class SuperpositionDistiller:
    """
    Implements the "Spoon Feed" protocol for superposition distillation.
    
    Phase 1: Cache teacher hidden states
    Phase 2: Train student to project to teacher's space
    Phase 3: Fine-tune LM head for language generation
    """
    
    def __init__(
        self,
        student: BitNetStudent,
        teacher_dim: int,
        device: str = "cuda",
    ):
        self.student = student.to(device)
        self.teacher_dim = teacher_dim
        self.student_dim = student.hidden_dim
        self.device = device
        
        # Create projector
        self.projector = SuperpositionProjector(
            self.student_dim, 
            teacher_dim,
            expansion_factor=2
        ).to(device)
        
        # Cached data
        self.cached_inputs: List[torch.Tensor] = []
        self.cached_targets: List[torch.Tensor] = []
    
    def cache_teacher_states(
        self,
        teacher_model,
        tokenizer,
        dataset,
        num_samples: int = 200,
        max_length: int = 128,
    ):
        """
        Phase 1: Cache teacher hidden states.
        
        This extracts the "thoughts" from the teacher so we can
        train the student without keeping the teacher in memory.
        """
        print("=" * 60)
        print("PHASE 1: Caching Teacher Hidden States")
        print("=" * 60)
        
        teacher_model.eval()
        self.cached_inputs = []
        self.cached_targets = []
        
        count = 0
        for item in tqdm(dataset, total=num_samples, desc="Caching"):
            if count >= num_samples:
                break
            
            # Get text
            if isinstance(item, dict):
                text = item.get('text', item.get('instruction', ''))
                if 'response' in item:
                    text = f"{item['instruction']}\n{item['response']}"
            else:
                text = str(item)
            
            if not text.strip():
                continue
            
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding='max_length',
            ).to(self.device)
            
            # Get teacher hidden states
            with torch.no_grad():
                outputs = teacher_model(
                    input_ids=inputs.input_ids,
                    output_hidden_states=True,
                )
                # Get last layer hidden states
                hidden = outputs.hidden_states[-1].cpu()
            
            self.cached_inputs.append(inputs.input_ids.cpu())
            self.cached_targets.append(hidden)
            count += 1
        
        print(f"Cached {len(self.cached_inputs)} samples")
        print(f"Input shape: {self.cached_inputs[0].shape}")
        print(f"Target shape: {self.cached_targets[0].shape}")
        
        # Free teacher memory
        del teacher_model
        torch.cuda.empty_cache()
        gc.collect()
        print("Teacher unloaded from memory")
    
    def train_superposition(
        self,
        num_steps: int = 500,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        use_cosine_loss: bool = True,
    ):
        """
        Phase 2: Train student to match teacher hidden states.
        
        Uses cosine similarity loss (more stable than MSE) to align
        the student's projected representations with the teacher's.
        """
        print("\n" + "=" * 60)
        print("PHASE 2: Superposition Training")
        print("=" * 60)
        
        if not self.cached_inputs:
            raise ValueError("No cached data! Run cache_teacher_states first.")
        
        # Optimizer for both student and projector
        optimizer = torch.optim.AdamW(
            list(self.student.parameters()) + list(self.projector.parameters()),
            lr=learning_rate,
            weight_decay=0.0,  # No weight decay for BitNet
        )
        
        self.student.train()
        self.projector.train()
        
        num_samples = len(self.cached_inputs)
        losses = []
        
        progress = tqdm(range(num_steps), desc="Training")
        for step in progress:
            # Random batch
            indices = torch.randint(0, num_samples, (batch_size,))
            
            # Get batch (squeeze extra dimensions)
            batch_in = torch.stack([self.cached_inputs[i] for i in indices])
            batch_target = torch.stack([self.cached_targets[i] for i in indices])
            
            # Handle shape: might be [B, 1, S] -> [B, S]
            if batch_in.dim() == 3 and batch_in.size(1) == 1:
                batch_in = batch_in.squeeze(1)
            if batch_target.dim() == 4 and batch_target.size(1) == 1:
                batch_target = batch_target.squeeze(1)
            
            batch_in = batch_in.to(self.device)
            batch_target = batch_target.to(self.device).float()
            
            # Forward: student -> projector
            student_hidden = self.student(batch_in, return_hidden_states=True)
            projected = self.projector(student_hidden)
            
            # Loss
            if use_cosine_loss:
                # Cosine similarity loss (more stable)
                # Flatten to [B*S, D]
                proj_flat = projected.view(-1, self.teacher_dim)
                target_flat = batch_target.view(-1, self.teacher_dim)
                
                # Cosine embedding loss expects target of 1s (same direction)
                ones = torch.ones(proj_flat.size(0), device=self.device)
                loss = F.cosine_embedding_loss(proj_flat, target_flat, ones)
            else:
                # MSE loss (can be unstable)
                loss = F.mse_loss(projected, batch_target)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            optimizer.step()
            
            losses.append(loss.item())
            
            if step % 10 == 0:
                avg_loss = sum(losses[-10:]) / len(losses[-10:])
                progress.set_postfix({"loss": f"{avg_loss:.4f}"})
        
        final_loss = sum(losses[-50:]) / len(losses[-50:])
        print(f"\nFinal loss: {final_loss:.4f}")
        
        return losses
    
    def train_lm_head(
        self,
        num_steps: int = 200,
        batch_size: int = 16,
        learning_rate: float = 5e-3,
    ):
        """
        Phase 3: Train LM head to decode representations.
        
        After superposition training, the student's internal representations
        have moved. We need to retrain the output head to map these new
        representations back to vocabulary tokens.
        """
        print("\n" + "=" * 60)
        print("PHASE 3: LM Head Training")
        print("=" * 60)
        
        # Freeze everything except embeddings (which are tied to LM head)
        for param in self.student.parameters():
            param.requires_grad = False
        
        # Unfreeze embedding (tied to LM head)
        self.student.embed_tokens.weight.requires_grad = True
        
        optimizer = torch.optim.AdamW(
            [self.student.embed_tokens.weight],
            lr=learning_rate,
        )
        
        self.student.train()
        num_samples = len(self.cached_inputs)
        losses = []
        
        progress = tqdm(range(num_steps), desc="LM Head Training")
        for step in progress:
            indices = torch.randint(0, num_samples, (batch_size,))
            
            batch_in = torch.stack([self.cached_inputs[i] for i in indices])
            if batch_in.dim() == 3 and batch_in.size(1) == 1:
                batch_in = batch_in.squeeze(1)
            batch_in = batch_in.to(self.device)
            
            # Forward with logits
            logits = self.student(batch_in, return_hidden_states=False)
            
            # Next token prediction loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch_in[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.student.vocab_size),
                shift_labels.view(-1),
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if step % 10 == 0:
                avg_loss = sum(losses[-10:]) / len(losses[-10:])
                progress.set_postfix({"loss": f"{avg_loss:.4f}"})
        
        # Unfreeze all parameters for future training
        for param in self.student.parameters():
            param.requires_grad = True
        
        final_loss = sum(losses[-20:]) / len(losses[-20:])
        print(f"\nFinal LM loss: {final_loss:.4f}")
        
        return losses
    
    def save(self, path: str):
        """Save student and projector."""
        torch.save({
            'student': self.student.state_dict(),
            'projector': self.projector.state_dict(),
        }, path)
        print(f"Saved to {path}")
    
    def load(self, path: str):
        """Load student and projector."""
        ckpt = torch.load(path)
        self.student.load_state_dict(ckpt['student'])
        self.projector.load_state_dict(ckpt['projector'])
        print(f"Loaded from {path}")


# =============================================================================
# Main Script
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', type=str, default='Qwen/Qwen2.5-Coder-7B-Instruct')
    parser.add_argument('--dataset', type=str, default='roneneldan/TinyStories')
    parser.add_argument('--num_cache', type=int, default=500)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--superposition_steps', type=int, default=500)
    parser.add_argument('--lm_head_steps', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output', type=str, default='superposition_student.pt')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # ==========================================================================
    # Load Teacher (temporarily)
    # ==========================================================================
    print("\nLoading teacher model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from datasets import load_dataset
    
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load in 4-bit to save memory
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
    print(f"Teacher hidden dim: {teacher_dim}")
    print(f"Vocab size: {vocab_size}")
    
    # ==========================================================================
    # Create Student
    # ==========================================================================
    print("\nCreating student model...")
    student = BitNetStudent(
        vocab_size=vocab_size,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
    )
    
    total_params = sum(p.numel() for p in student.parameters())
    print(f"Student parameters: {total_params / 1e6:.1f}M")
    
    # ==========================================================================
    # Create Distiller
    # ==========================================================================
    distiller = SuperpositionDistiller(student, teacher_dim, device)
    
    # ==========================================================================
    # Phase 1: Cache Teacher States
    # ==========================================================================
    dataset = load_dataset(args.dataset, split="train", streaming=True)
    
    distiller.cache_teacher_states(
        teacher,
        tokenizer,
        dataset,
        num_samples=args.num_cache,
        max_length=args.max_length,
    )
    
    # ==========================================================================
    # Phase 2: Superposition Training
    # ==========================================================================
    distiller.train_superposition(
        num_steps=args.superposition_steps,
        batch_size=args.batch_size,
        learning_rate=1e-3,
        use_cosine_loss=True,
    )
    
    # ==========================================================================
    # Phase 3: LM Head Training
    # ==========================================================================
    distiller.train_lm_head(
        num_steps=args.lm_head_steps,
        batch_size=args.batch_size,
        learning_rate=5e-3,
    )
    
    # ==========================================================================
    # Save & Test
    # ==========================================================================
    distiller.save(args.output)
    
    # Quick test
    print("\n" + "=" * 60)
    print("Generation Test")
    print("=" * 60)
    
    student.eval()
    prompt = "Once upon a time"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        output_ids = student.generate(input_ids, max_new_tokens=50)
    
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Output: {output_text}")


if __name__ == "__main__":
    main()
