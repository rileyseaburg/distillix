"""
BitNet Distillation Training Loop.

Features:
  - Automatic Mixed Precision (AMP) for memory efficiency
  - Gradient checkpointing for large models
  - Gradient accumulation for effective larger batch sizes
  - Learning rate warmup and cosine decay
  - Checkpoint saving and resumption
  - Logging to console, TensorBoard, and W&B

Copyright (c) 2025 Distillix. All Rights Reserved.
"""

import os
import sys
import math
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import StudentLLM
from .loss import DistillationLoss, TextDistillationLoss
from .data import create_dataloader, TokenizerWrapper
from .config import Config, ModelConfig, TrainingConfig, get_config_125m


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: str, rank: int = 0) -> logging.Logger:
    """Set up logging to file and console."""
    logger = logging.getLogger("distillix")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARN)
    
    if rank == 0:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(output_dir, "training.log")
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# Learning Rate Scheduler
# =============================================================================

def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Learning rate scheduler with warmup and cosine decay.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of max LR
    
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        
        # Cosine decay phase
        progress = (current_step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        
        # Scale to [min_lr_ratio, 1.0]
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Training State
# =============================================================================

@dataclass
class TrainingState:
    """Holds training state for checkpointing."""
    step: int = 0
    epoch: int = 0
    best_loss: float = float('inf')
    total_tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingState":
        return cls(**d)


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """
    BitNet Distillation Trainer.
    
    Handles the complete training pipeline:
      - Model and optimizer setup
      - Training loop with AMP and gradient accumulation
      - Checkpointing and logging
      - Evaluation
    """
    
    def __init__(
        self,
        config: Config,
        model: Optional[StudentLLM] = None,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        tokenizer: Optional[TokenizerWrapper] = None,
    ):
        self.config = config
        self.training_config = config.training
        self.model_config = config.model
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if config.training.amp_dtype == "float16" else torch.bfloat16
        
        # Logging
        self.logger = setup_logging(config.training.output_dir)
        self.logger.info(f"Using device: {self.device}")
        
        # Model
        if model is None:
            self.logger.info("Creating model from config...")
            model = StudentLLM(config.model)
        
        self.model = model.to(self.device)
        
        # Gradient checkpointing
        if config.training.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()
            self.logger.info("Gradient checkpointing enabled")
        
        # Log model stats
        total_params = self.model.num_parameters()
        bitlinear_params, fp_params = self.model.count_bitlinear_params()
        self.logger.info(f"Model parameters: {total_params:,}")
        self.logger.info(f"  BitLinear: {bitlinear_params:,} ({100*bitlinear_params/total_params:.1f}%)")
        self.logger.info(f"  Full precision: {fp_params:,} ({100*fp_params/total_params:.1f}%)")
        
        # Tokenizer
        self.tokenizer = tokenizer
        
        # Data loaders
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            warmup_steps=config.training.warmup_steps,
            max_steps=config.training.max_steps,
            min_lr_ratio=config.training.min_learning_rate / config.training.learning_rate,
        )
        
        # Loss function
        self.loss_fn = TextDistillationLoss(
            label_smoothing=config.distillation.label_smoothing,
        )
        
        # AMP scaler
        self.scaler = GradScaler() if config.training.use_amp else None
        
        # Training state
        self.state = TrainingState()
        
        # Output directories
        self.output_dir = Path(config.training.output_dir)
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # W&B setup
        self.wandb_run = None
        if config.training.use_wandb:
            self._setup_wandb()
    
    def _create_optimizer(self) -> AdamW:
        """Create optimizer with weight decay handling."""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # No weight decay for biases and normalization weights
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.training_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        return AdamW(
            param_groups,
            lr=self.training_config.learning_rate,
            betas=(self.training_config.adam_beta1, self.training_config.adam_beta2),
            eps=self.training_config.adam_eps,
        )
    
    def _setup_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            import wandb
            
            self.wandb_run = wandb.init(
                project=self.training_config.wandb_project,
                name=self.training_config.wandb_run_name,
                config={
                    "model": self.config.model.to_dict(),
                    "training": self.config.training.to_dict(),
                    "distillation": {
                        "temperature": self.config.distillation.temperature,
                        "alpha": self.config.distillation.alpha,
                        "teachers": self.config.distillation.teachers,
                    },
                },
            )
            self.logger.info("Weights & Biases initialized")
        except ImportError:
            self.logger.warning("wandb not installed, skipping W&B logging")
            self.training_config.use_wandb = False
    
    def save_checkpoint(self, name: str = "checkpoint"):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{name}_step{self.state.step}.pt"
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_state": self.state.to_dict(),
            "config": {
                "model": self.config.model.to_dict(),
                "training": self.config.training.to_dict(),
            },
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save model config separately
        self.config.model.save(self.output_dir / "model_config.json")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.state = TrainingState.from_dict(checkpoint["training_state"])
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.logger.info(f"Resumed from step {self.state.step}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute single training step."""
        self.model.train()
        
        # Move to device
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        labels = batch.labels.to(self.device)
        
        # Mixed precision context
        amp_context = autocast(dtype=self.dtype) if self.training_config.use_amp else nullcontext()
        
        with amp_context:
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            # Compute loss
            loss = self.loss_fn(outputs["logits"], labels)
            
            # Scale for gradient accumulation
            loss = loss / self.training_config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Count tokens
        num_tokens = (labels != -100).sum().item()
        
        return {
            "loss": loss.item() * self.training_config.gradient_accumulation_steps,
            "tokens": num_tokens,
        }
    
    def optimizer_step(self):
        """Execute optimizer step with gradient clipping."""
        if self.scaler is not None:
            # Unscale gradients
            self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.training_config.max_grad_norm,
        )
        
        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Scheduler step
        self.scheduler.step()
        
        # Zero gradients
        self.optimizer.zero_grad(set_to_none=True)
        
        return grad_norm.item()
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on validation set."""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
            input_ids = batch.input_ids.to(self.device)
            attention_mask = batch.attention_mask.to(self.device)
            labels = batch.labels.to(self.device)
            
            amp_context = autocast(dtype=self.dtype) if self.training_config.use_amp else nullcontext()
            
            with amp_context:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = self.loss_fn(outputs["logits"], labels)
            
            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
        
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow
        
        return {
            "val_loss": avg_loss,
            "val_perplexity": perplexity,
        }
    
    def train(self):
        """Main training loop."""
        if self.train_dataloader is None:
            raise ValueError("train_dataloader is required for training")
        
        self.logger.info("Starting training...")
        self.logger.info(f"  Max steps: {self.training_config.max_steps}")
        self.logger.info(f"  Batch size: {self.training_config.micro_batch_size}")
        self.logger.info(f"  Gradient accumulation: {self.training_config.gradient_accumulation_steps}")
        self.logger.info(f"  Effective batch size: {self.training_config.effective_batch_size}")
        self.logger.info(f"  Learning rate: {self.training_config.learning_rate}")
        
        # Training loop
        accumulation_loss = 0.0
        accumulation_tokens = 0
        step_start_time = time.time()
        
        data_iter = iter(self.train_dataloader)
        
        pbar = tqdm(
            range(self.state.step, self.training_config.max_steps),
            desc="Training",
            initial=self.state.step,
            total=self.training_config.max_steps,
        )
        
        for step in pbar:
            # Accumulation loop
            for accum_step in range(self.training_config.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # Restart data iterator (new epoch)
                    self.state.epoch += 1
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)
                
                step_metrics = self.train_step(batch)
                accumulation_loss += step_metrics["loss"]
                accumulation_tokens += step_metrics["tokens"]
            
            # Optimizer step
            grad_norm = self.optimizer_step()
            
            # Update state
            self.state.step = step + 1
            self.state.total_tokens += accumulation_tokens
            
            # Calculate metrics
            step_time = time.time() - step_start_time
            tokens_per_sec = accumulation_tokens / step_time
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Logging
            if (step + 1) % self.training_config.logging_steps == 0:
                pbar.set_postfix({
                    "loss": f"{accumulation_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "tok/s": f"{tokens_per_sec:.0f}",
                })
                
                if self.wandb_run:
                    import wandb
                    wandb.log({
                        "train/loss": accumulation_loss,
                        "train/learning_rate": current_lr,
                        "train/grad_norm": grad_norm,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/total_tokens": self.state.total_tokens,
                        "train/epoch": self.state.epoch,
                    }, step=step + 1)
            
            # Evaluation
            if (step + 1) % self.training_config.eval_steps == 0:
                eval_metrics = self.evaluate()
                if eval_metrics:
                    self.logger.info(
                        f"Step {step + 1}: val_loss={eval_metrics['val_loss']:.4f}, "
                        f"val_ppl={eval_metrics['val_perplexity']:.2f}"
                    )
                    
                    if self.wandb_run:
                        import wandb
                        wandb.log(eval_metrics, step=step + 1)
                    
                    # Save best model
                    if eval_metrics['val_loss'] < self.state.best_loss:
                        self.state.best_loss = eval_metrics['val_loss']
                        self.save_checkpoint("best")
            
            # Checkpointing
            if (step + 1) % self.training_config.save_steps == 0:
                self.save_checkpoint("checkpoint")
            
            # Reset accumulators
            accumulation_loss = 0.0
            accumulation_tokens = 0
            step_start_time = time.time()
        
        # Final checkpoint
        self.save_checkpoint("final")
        self.logger.info("Training complete!")
        
        if self.wandb_run:
            self.wandb_run.finish()
    
    def save_model(self, path: str):
        """Save model weights in a deployment-ready format."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save(self.model.state_dict(), save_path / "model.pt")
        
        # Save config
        self.config.model.save(save_path / "config.json")
        
        # Save tokenizer if available
        if self.tokenizer is not None:
            self.tokenizer.tokenizer.save_pretrained(save_path / "tokenizer")
        
        self.logger.info(f"Model saved to {save_path}")


# =============================================================================
# Training Script Entry Point
# =============================================================================

def train_from_config(config_path: Optional[str] = None):
    """Train model from config file or defaults."""
    # Load or create config
    if config_path:
        config = Config.load(config_path)
    else:
        config = get_config_125m()
    
    # Create tokenizer
    tokenizer = TokenizerWrapper(
        tokenizer_name_or_path=config.data.tokenizer_name,
        max_length=config.data.max_length,
    )
    
    # Update vocab size if needed
    if config.model.vocab_size != tokenizer.vocab_size:
        print(f"Updating vocab_size from {config.model.vocab_size} to {tokenizer.vocab_size}")
        config.model.vocab_size = tokenizer.vocab_size
    
    # Create data loaders
    train_dataloader = create_dataloader(
        data_path=config.data.train_data,
        tokenizer=tokenizer,
        batch_size=config.training.micro_batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        teacher_strategy=config.distillation.teacher_strategy,
    )
    
    val_dataloader = None
    if Path(config.data.val_data).exists():
        val_dataloader = create_dataloader(
            data_path=config.data.val_data,
            tokenizer=tokenizer,
            batch_size=config.training.micro_batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
    
    # Create trainer
    trainer = Trainer(
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model(config.training.output_dir)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train BitNet distillation model")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    train_from_config(args.config)
