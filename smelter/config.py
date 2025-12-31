"""
Configuration for Distillix BitNet Student Model and Training.

This module defines all hyperparameters for:
  - Model architecture (dimensions, layers, etc.)
  - Training (learning rate, batch size, etc.)
  - Hardware optimization (gradient checkpointing, AMP, etc.)

Tuned for RTX 2080 Super (8GB VRAM).

Copyright (c) 2025 Distillix. All Rights Reserved.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """
    Student model architecture configuration.
    
    Default: ~125M parameter model optimized for 8GB VRAM.
    """
    
    # Vocabulary
    vocab_size: int = 32000  # Match Llama tokenizer
    
    # Transformer dimensions
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    head_dim: int = 64  # hidden_dim // num_heads
    
    # MLP dimensions (SwiGLU uses 8/3 * hidden_dim, rounded)
    mlp_hidden_dim: int = 2048  # ~2.7x hidden_dim
    
    # Sequence length
    max_seq_len: int = 1024  # Conservative for 8GB VRAM
    
    # Regularization
    dropout: float = 0.0  # BitNet papers suggest no dropout
    
    # Positional encoding
    rope_base: float = 10000.0
    rope_scaling: Optional[float] = None  # For extended context
    
    # Quantization settings
    use_bitlinear: bool = True
    bitlinear_per_channel: bool = False  # Per-tensor is faster
    quantization_eps: float = 1e-8
    
    # Precision for non-quantized layers
    norm_eps: float = 1e-6
    
    # Tie embeddings to output projection
    tie_word_embeddings: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
        assert self.head_dim == self.hidden_dim // self.num_heads, \
            f"head_dim should be {self.hidden_dim // self.num_heads}"
    
    @property
    def num_params_estimate(self) -> int:
        """Estimate total parameters (approximate)."""
        # Embeddings
        embed = self.vocab_size * self.hidden_dim
        
        # Per layer
        # Attention: Q, K, V, O projections
        attn = 4 * self.hidden_dim * self.hidden_dim
        # MLP: gate, up, down (SwiGLU)
        mlp = 3 * self.hidden_dim * self.mlp_hidden_dim
        # Norms (negligible but included)
        norms = 2 * self.hidden_dim
        
        per_layer = attn + mlp + norms
        
        # Total
        total = embed + (self.num_layers * per_layer)
        if not self.tie_word_embeddings:
            total += self.vocab_size * self.hidden_dim  # LM head
        
        return total
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "rope_base": self.rope_base,
            "rope_scaling": self.rope_scaling,
            "use_bitlinear": self.use_bitlinear,
            "bitlinear_per_channel": self.bitlinear_per_channel,
            "quantization_eps": self.quantization_eps,
            "norm_eps": self.norm_eps,
            "tie_word_embeddings": self.tie_word_embeddings,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """Create from dictionary."""
        return cls(**d)
    
    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        """Load config from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class TrainingConfig:
    """
    Training hyperparameters.
    
    Optimized for RTX 2080 Super (8GB VRAM).
    """
    
    # Batch size
    micro_batch_size: int = 4  # Per-GPU batch size
    gradient_accumulation_steps: int = 8  # Effective batch = 32
    
    # Learning rate
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5  # For cosine decay
    weight_decay: float = 0.1
    
    # Schedule
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Optimizer
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Memory optimization
    use_amp: bool = True  # Automatic Mixed Precision
    amp_dtype: Literal["float16", "bfloat16"] = "float16"  # bfloat16 better if supported
    gradient_checkpointing: bool = True  # Trade compute for memory
    
    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 10
    
    # Paths
    output_dir: str = "artifacts/students"
    checkpoint_dir: str = "artifacts/checkpoints"
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "distillix"
    wandb_run_name: Optional[str] = None
    
    # Reproducibility
    seed: int = 42
    
    @property
    def effective_batch_size(self) -> int:
        """Total batch size across accumulation steps."""
        return self.micro_batch_size * self.gradient_accumulation_steps
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "optimizer": self.optimizer,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_eps": self.adam_eps,
            "max_grad_norm": self.max_grad_norm,
            "use_amp": self.use_amp,
            "amp_dtype": self.amp_dtype,
            "gradient_checkpointing": self.gradient_checkpointing,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps,
            "output_dir": self.output_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "use_wandb": self.use_wandb,
            "wandb_project": self.wandb_project,
            "wandb_run_name": self.wandb_run_name,
            "seed": self.seed,
        }


@dataclass
class DistillationConfig:
    """
    Knowledge distillation configuration.
    """
    
    # Temperature for softening distributions
    temperature: float = 2.0
    
    # Balance between task loss and distillation loss
    # alpha * CE_loss + (1 - alpha) * KL_loss
    alpha: float = 0.5
    
    # Label smoothing for task loss
    label_smoothing: float = 0.1
    
    # Teacher models (via OpenCode server)
    teachers: List[str] = field(default_factory=lambda: [
        "azure/claude-sonnet-4-5",
        "zai-coding-plan/glm-4.7",
        "minimax/MiniMax-M2.1",
    ])
    
    # Teacher selection strategy
    teacher_strategy: Literal["random", "round_robin", "ensemble", "best"] = "random"
    
    # OpenCode server
    opencode_host: str = "127.0.0.1"
    opencode_port: int = 4096


@dataclass
class DataConfig:
    """
    Data pipeline configuration.
    """
    
    # Paths
    train_data: str = "data/distillation/train.jsonl"
    val_data: str = "data/distillation/val.jsonl"
    prompts_dir: str = "data/prompts"
    
    # Tokenizer
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf"  # Or path to local tokenizer
    
    # Processing
    max_length: int = 1024
    num_workers: int = 4
    prefetch_factor: int = 2
    
    # Data generation
    num_samples_per_prompt: int = 1
    max_concurrent_requests: int = 5


@dataclass
class Config:
    """
    Master configuration combining all sub-configs.
    """
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def save(self, path: str):
        """Save full config to JSON."""
        config_dict = {
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "distillation": {
                "temperature": self.distillation.temperature,
                "alpha": self.distillation.alpha,
                "label_smoothing": self.distillation.label_smoothing,
                "teachers": self.distillation.teachers,
                "teacher_strategy": self.distillation.teacher_strategy,
                "opencode_host": self.distillation.opencode_host,
                "opencode_port": self.distillation.opencode_port,
            },
            "data": {
                "train_data": self.data.train_data,
                "val_data": self.data.val_data,
                "prompts_dir": self.data.prompts_dir,
                "tokenizer_name": self.data.tokenizer_name,
                "max_length": self.data.max_length,
                "num_workers": self.data.num_workers,
            },
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "Config":
        """Load full config from JSON."""
        with open(path, 'r') as f:
            d = json.load(f)
        
        return cls(
            model=ModelConfig.from_dict(d.get("model", {})),
            training=TrainingConfig(**d.get("training", {})),
            distillation=DistillationConfig(**d.get("distillation", {})),
            data=DataConfig(**d.get("data", {})),
        )


# =============================================================================
# Preset Configurations
# =============================================================================

def get_config_125m() -> Config:
    """~125M parameter student model (default for 8GB VRAM)."""
    return Config()


def get_config_50m() -> Config:
    """~50M parameter student model (very small, for testing)."""
    return Config(
        model=ModelConfig(
            hidden_dim=512,
            num_layers=8,
            num_heads=8,
            head_dim=64,
            mlp_hidden_dim=1408,
            max_seq_len=512,
        )
    )


def get_config_300m() -> Config:
    """~300M parameter student model (needs 16GB+ VRAM)."""
    return Config(
        model=ModelConfig(
            hidden_dim=1024,
            num_layers=16,
            num_heads=16,
            head_dim=64,
            mlp_hidden_dim=2816,
            max_seq_len=2048,
        ),
        training=TrainingConfig(
            micro_batch_size=2,
            gradient_accumulation_steps=16,
        )
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    # Print default config
    config = get_config_125m()
    
    print("=" * 60)
    print("Distillix Default Configuration (~125M params)")
    print("=" * 60)
    
    print(f"\nModel:")
    print(f"  Vocab size: {config.model.vocab_size:,}")
    print(f"  Hidden dim: {config.model.hidden_dim}")
    print(f"  Num layers: {config.model.num_layers}")
    print(f"  Num heads: {config.model.num_heads}")
    print(f"  MLP hidden: {config.model.mlp_hidden_dim}")
    print(f"  Max seq len: {config.model.max_seq_len}")
    print(f"  Estimated params: {config.model.num_params_estimate:,}")
    
    print(f"\nTraining:")
    print(f"  Micro batch: {config.training.micro_batch_size}")
    print(f"  Grad accum: {config.training.gradient_accumulation_steps}")
    print(f"  Effective batch: {config.training.effective_batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Max steps: {config.training.max_steps:,}")
    print(f"  AMP: {config.training.use_amp}")
    print(f"  Grad checkpointing: {config.training.gradient_checkpointing}")
    
    print(f"\nDistillation:")
    print(f"  Temperature: {config.distillation.temperature}")
    print(f"  Alpha: {config.distillation.alpha}")
    print(f"  Teachers: {config.distillation.teachers}")
    
    print(f"\nData:")
    print(f"  Tokenizer: {config.data.tokenizer_name}")
    print(f"  Max length: {config.data.max_length}")
    
    # Save example config
    config.save("config_example.json")
    print(f"\nSaved example config to config_example.json")
