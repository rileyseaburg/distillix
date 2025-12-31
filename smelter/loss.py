"""
Distillation Loss Functions for BitNet Training.

This module implements loss functions for knowledge distillation:
  1. Task Loss: Cross-entropy against ground truth tokens
  2. Knowledge Loss: KL divergence between student and teacher distributions
  3. Combined Distillation Loss: Weighted combination

For API-based teachers (no logits available), we use:
  - Text-based distillation (teacher response as ground truth)
  - Label smoothing to soften the distribution

Copyright (c) 2025 Distillix. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


# =============================================================================
# Core Loss Functions
# =============================================================================

def cross_entropy_loss(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> Tensor:
    """
    Standard cross-entropy loss for language modeling.
    
    Args:
        logits: Student logits [batch, seq_len, vocab_size]
        labels: Target token IDs [batch, seq_len]
        ignore_index: Token ID to ignore (padding)
        label_smoothing: Label smoothing factor (0 = none)
    
    Returns:
        Scalar loss tensor
    """
    # Shift for causal LM: predict token t+1 from position t
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten
    vocab_size = shift_logits.size(-1)
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    
    return F.cross_entropy(
        shift_logits,
        shift_labels,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )


def kl_divergence_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 2.0,
    reduction: str = "batchmean",
) -> Tensor:
    """
    KL Divergence loss between student and teacher distributions.
    
    This is the "knowledge" part of knowledge distillation - learning from
    the teacher's soft probability distribution rather than hard labels.
    
    The temperature parameter softens the distributions:
      - T=1: Standard softmax
      - T>1: Softer distributions, more probability mass on less likely tokens
      - Higher T reveals more of the "dark knowledge" in the teacher
    
    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_logits: [batch, seq_len, vocab_size]
        temperature: Softening temperature (typically 2-4)
        reduction: How to reduce ('batchmean', 'sum', 'none')
    
    Returns:
        KL divergence loss (scaled by T^2 to maintain gradient magnitude)
    """
    # Soften with temperature
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # KL divergence: sum over vocab, mean over batch/seq
    kl_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction,
    )
    
    # Scale by T^2 to balance with CE loss
    # (KL gradients are proportional to 1/T^2, so we multiply to compensate)
    return kl_loss * (temperature ** 2)


def kl_divergence_loss_with_mask(
    student_logits: Tensor,
    teacher_logits: Tensor,
    attention_mask: Tensor,
    temperature: float = 2.0,
) -> Tensor:
    """
    KL divergence with attention mask for variable-length sequences.
    
    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_logits: [batch, seq_len, vocab_size]
        attention_mask: [batch, seq_len] (1 = valid, 0 = pad)
        temperature: Softening temperature
    
    Returns:
        Masked KL divergence loss
    """
    # Shift for causal LM
    student_logits = student_logits[..., :-1, :].contiguous()
    teacher_logits = teacher_logits[..., :-1, :].contiguous()
    mask = attention_mask[..., 1:].contiguous()
    
    # Softened distributions
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # Per-token KL divergence
    kl_per_token = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction='none'
    ).sum(dim=-1)  # Sum over vocab
    
    # Mask and average
    kl_per_token = kl_per_token * mask
    loss = kl_per_token.sum() / mask.sum().clamp(min=1)
    
    return loss * (temperature ** 2)


# =============================================================================
# Combined Distillation Loss
# =============================================================================

@dataclass
class DistillationLossOutput:
    """Container for distillation loss components."""
    total_loss: Tensor
    task_loss: Tensor
    knowledge_loss: Optional[Tensor]
    alpha: float
    temperature: float


class DistillationLoss(nn.Module):
    """
    Combined distillation loss for training student models.
    
    Loss = alpha * TaskLoss + (1 - alpha) * KnowledgeLoss
    
    Where:
      - TaskLoss: Cross-entropy against ground truth tokens
      - KnowledgeLoss: KL divergence against teacher soft labels
    
    For API-based teachers (no logits), only TaskLoss is used
    with label smoothing to approximate soft labels.
    
    Args:
        temperature: Softening temperature for KL loss
        alpha: Weight for task loss (1 = only task, 0 = only KL)
        label_smoothing: Label smoothing for task loss
        ignore_index: Token ID to ignore (padding)
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        label_smoothing: float = 0.1,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
    
    def forward(
        self,
        student_logits: Tensor,
        labels: Tensor,
        teacher_logits: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> DistillationLossOutput:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Student model outputs [batch, seq, vocab]
            labels: Ground truth token IDs [batch, seq]
            teacher_logits: Optional teacher logits [batch, seq, vocab]
            attention_mask: Optional mask for variable-length sequences
        
        Returns:
            DistillationLossOutput with all loss components
        """
        # Task loss (always computed)
        task_loss = cross_entropy_loss(
            student_logits,
            labels,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
        
        # Knowledge loss (only if teacher logits available)
        knowledge_loss = None
        if teacher_logits is not None:
            if attention_mask is not None:
                knowledge_loss = kl_divergence_loss_with_mask(
                    student_logits,
                    teacher_logits,
                    attention_mask,
                    temperature=self.temperature,
                )
            else:
                knowledge_loss = kl_divergence_loss(
                    student_logits[..., :-1, :],
                    teacher_logits[..., :-1, :],
                    temperature=self.temperature,
                )
        
        # Combine losses
        if knowledge_loss is not None:
            total_loss = self.alpha * task_loss + (1 - self.alpha) * knowledge_loss
        else:
            total_loss = task_loss
        
        return DistillationLossOutput(
            total_loss=total_loss,
            task_loss=task_loss,
            knowledge_loss=knowledge_loss,
            alpha=self.alpha,
            temperature=self.temperature,
        )


# =============================================================================
# Text-Only Distillation (for API-based teachers)
# =============================================================================

class TextDistillationLoss(nn.Module):
    """
    Distillation loss for API-based teachers (no logits available).
    
    When we only have teacher text responses (not probability distributions),
    we train the student to predict the teacher's text. This is essentially
    supervised fine-tuning on teacher outputs.
    
    To approximate soft labels, we use:
      - Label smoothing (redistributes probability mass)
      - Higher temperature during generation (softer sampling)
    
    This is less information-rich than logit-based distillation,
    but works with any teacher model accessible via API.
    """
    
    def __init__(
        self,
        label_smoothing: float = 0.1,
        ignore_index: int = -100,
        focal_gamma: float = 0.0,  # Focal loss (0 = disabled)
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.focal_gamma = focal_gamma
    
    def forward(
        self,
        student_logits: Tensor,
        teacher_tokens: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute text-based distillation loss.
        
        Args:
            student_logits: [batch, seq_len, vocab_size]
            teacher_tokens: [batch, seq_len] teacher's output tokens
            attention_mask: [batch, seq_len] mask for valid tokens
        
        Returns:
            Loss tensor
        """
        # Shift for causal LM
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = teacher_tokens[..., 1:].contiguous()
        
        vocab_size = shift_logits.size(-1)
        
        # Flatten
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        
        # Standard CE with label smoothing
        loss = F.cross_entropy(
            shift_logits,
            shift_labels,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction='none' if self.focal_gamma > 0 else 'mean',
        )
        
        # Optional focal loss weighting
        if self.focal_gamma > 0:
            # Focal loss: (1 - p_t)^gamma * CE
            # Helps focus on hard examples
            probs = F.softmax(shift_logits, dim=-1)
            p_t = probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - p_t) ** self.focal_gamma
            
            # Mask padding
            valid_mask = shift_labels != self.ignore_index
            loss = (loss * focal_weight * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        
        return loss


# =============================================================================
# Multi-Teacher Distillation
# =============================================================================

class MultiTeacherDistillationLoss(nn.Module):
    """
    Distillation from multiple teachers.
    
    Strategies:
      - 'average': Average teacher logits before computing loss
      - 'weighted': Weighted average based on teacher quality scores
      - 'mixture': Sample from teachers during training (data augmentation)
      - 'ensemble': Learn from all teachers simultaneously (multiple loss terms)
    
    For API-based teachers, we use the 'mixture' strategy with text outputs.
    """
    
    def __init__(
        self,
        strategy: str = "average",
        temperature: float = 2.0,
        alpha: float = 0.5,
        teacher_weights: Optional[Dict[str, float]] = None,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.strategy = strategy
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_weights = teacher_weights or {}
        self.label_smoothing = label_smoothing
        
        self.base_loss = DistillationLoss(
            temperature=temperature,
            alpha=alpha,
            label_smoothing=label_smoothing,
        )
    
    def forward(
        self,
        student_logits: Tensor,
        labels: Tensor,
        teacher_outputs: Dict[str, Tensor],
        output_type: str = "logits",  # or "tokens"
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute multi-teacher distillation loss.
        
        Args:
            student_logits: [batch, seq, vocab]
            labels: Ground truth tokens
            teacher_outputs: Dict mapping teacher name to outputs
            output_type: Whether teacher outputs are logits or tokens
        
        Returns:
            (total_loss, per_teacher_losses)
        """
        if not teacher_outputs:
            # No teacher outputs, just use labels
            result = self.base_loss(student_logits, labels)
            return result.total_loss, {}
        
        if output_type == "tokens":
            # Text-based distillation
            return self._text_distillation(student_logits, labels, teacher_outputs)
        else:
            # Logit-based distillation
            return self._logit_distillation(student_logits, labels, teacher_outputs)
    
    def _text_distillation(
        self,
        student_logits: Tensor,
        labels: Tensor,
        teacher_tokens: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Text-based distillation from multiple teachers."""
        text_loss_fn = TextDistillationLoss(label_smoothing=self.label_smoothing)
        
        per_teacher_losses = {}
        total_loss = torch.tensor(0.0, device=student_logits.device)
        total_weight = 0.0
        
        for name, tokens in teacher_tokens.items():
            weight = self.teacher_weights.get(name, 1.0)
            loss = text_loss_fn(student_logits, tokens)
            per_teacher_losses[name] = loss
            total_loss = total_loss + weight * loss
            total_weight += weight
        
        if total_weight > 0:
            total_loss = total_loss / total_weight
        
        return total_loss, per_teacher_losses
    
    def _logit_distillation(
        self,
        student_logits: Tensor,
        labels: Tensor,
        teacher_logits: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Logit-based distillation from multiple teachers."""
        if self.strategy == "average":
            # Average teacher logits
            avg_logits = torch.stack(list(teacher_logits.values())).mean(dim=0)
            result = self.base_loss(student_logits, labels, avg_logits)
            return result.total_loss, {"averaged": result.knowledge_loss}
        
        elif self.strategy == "ensemble":
            # Learn from all teachers
            per_teacher_losses = {}
            total_loss = torch.tensor(0.0, device=student_logits.device)
            total_weight = 0.0
            
            for name, logits in teacher_logits.items():
                weight = self.teacher_weights.get(name, 1.0)
                result = self.base_loss(student_logits, labels, logits)
                per_teacher_losses[name] = result.knowledge_loss
                total_loss = total_loss + weight * result.total_loss
                total_weight += weight
            
            return total_loss / total_weight, per_teacher_losses
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


# =============================================================================
# Auxiliary Losses
# =============================================================================

def representation_loss(
    student_hidden: Tensor,
    teacher_hidden: Tensor,
    loss_type: str = "mse",
) -> Tensor:
    """
    Loss on intermediate representations.
    
    Can help student learn internal features, not just output distribution.
    Requires dimension matching (projection layer) if dims differ.
    
    Args:
        student_hidden: Student's hidden states [batch, seq, hidden]
        teacher_hidden: Teacher's hidden states [batch, seq, hidden]
        loss_type: 'mse', 'cosine', or 'huber'
    
    Returns:
        Representation loss
    """
    if loss_type == "mse":
        return F.mse_loss(student_hidden, teacher_hidden)
    elif loss_type == "cosine":
        # 1 - cosine_similarity (so 0 = perfect match)
        return 1 - F.cosine_similarity(
            student_hidden.view(-1, student_hidden.size(-1)),
            teacher_hidden.view(-1, teacher_hidden.size(-1)),
        ).mean()
    elif loss_type == "huber":
        return F.smooth_l1_loss(student_hidden, teacher_hidden)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing distillation loss functions...")
    
    batch_size = 2
    seq_len = 16
    vocab_size = 32000
    
    # Create fake data
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Test CE loss
    ce_loss = cross_entropy_loss(student_logits, labels)
    print(f"Cross-entropy loss: {ce_loss.item():.4f}")
    
    # Test KL loss
    kl_loss = kl_divergence_loss(student_logits, teacher_logits, temperature=2.0)
    print(f"KL divergence loss (T=2): {kl_loss.item():.4f}")
    
    # Test combined loss
    distill_loss = DistillationLoss(temperature=2.0, alpha=0.5)
    result = distill_loss(student_logits, labels, teacher_logits)
    print(f"Combined distillation loss: {result.total_loss.item():.4f}")
    print(f"  Task loss: {result.task_loss.item():.4f}")
    print(f"  Knowledge loss: {result.knowledge_loss.item():.4f}")
    
    # Test text-only distillation
    text_loss = TextDistillationLoss(label_smoothing=0.1)
    teacher_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    t_loss = text_loss(student_logits, teacher_tokens)
    print(f"Text distillation loss: {t_loss.item():.4f}")
    
    # Test multi-teacher
    multi_loss = MultiTeacherDistillationLoss(strategy="average", alpha=0.5)
    teacher_outputs = {
        "teacher_1": teacher_logits,
        "teacher_2": torch.randn_like(teacher_logits),
    }
    total, per_teacher = multi_loss(student_logits, labels, teacher_outputs, "logits")
    print(f"Multi-teacher loss: {total.item():.4f}")
    
    print("\nAll distillation loss tests passed!")
