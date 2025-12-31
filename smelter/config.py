"""
Configuration for Distillix "Frankenstein" BitNet Student Model.

A hybrid architecture stealing the best from each lineage:
  - Math: BitNet b1.58 (Microsoft) - 1.58-bit weights
  - Tokenizer: Llama-2 32k vocab - "Brain-First" parameter allocation
  - Stability: Gemma 2/3 - Logit soft-capping, QK-Norm
  - Attention: Llama 3 GQA - 3x KV cache reduction
  - Position: RoPE with theta=1M for long code context

Target: 125M params on RTX 2080 Super (8GB VRAM)
Goal: Maximum coding performance per parameter

Copyright (c) 2025 Distillix. All Rights Reserved.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """
    "Frankenstein" Student Model Architecture.
    
    Hybrid design stealing the best from each lineage:
      - BitNet b1.58 core (1.58-bit weights)
      - Llama-2 tokenizer (32k vocab, "Brain-First")
      - Gemma 2/3 stability (soft-capping, QK-Norm)
      - Llama 3 attention (GQA for 3x KV cache reduction)
      - Extended RoPE (theta=1M for long code context)
    
    Default: ~125M parameter model optimized for 8GB VRAM.
    """
    
    # ==========================================================================
    # Vocabulary (Llama-2 "Brain-First" Strategy)
    # ==========================================================================
    # CRITICAL: 32k vocab saves ~75M params vs Llama-3/Gemma (128k/256k)
    # This allocates 80% of params to "Brain" instead of "Dictionary"
    vocab_size: int = 32000  # Llama-2 tokenizer (meta-llama/Llama-2-7b-hf)
    
    # ==========================================================================
    # Transformer Dimensions
    # ==========================================================================
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12  # Query heads
    num_kv_heads: int = 4  # GQA: Key/Value heads (3:1 ratio = 3x KV cache reduction)
    head_dim: int = 64  # hidden_dim // num_heads
    
    # MLP dimensions (SwiGLU uses 8/3 * hidden_dim, rounded)
    mlp_hidden_dim: int = 2048  # ~2.7x hidden_dim
    
    # ==========================================================================
    # Sequence Length
    # ==========================================================================
    max_seq_len: int = 2048  # Extended for code context (GQA enables this)
    
    # ==========================================================================
    # Regularization
    # ==========================================================================
    dropout: float = 0.0  # BitNet papers suggest no dropout
    
    # ==========================================================================
    # Positional Encoding (Extended RoPE for Long Code)
    # ==========================================================================
    rope_base: float = 1000000.0  # theta=1M for long context (vs 10k default)
    rope_scaling: Optional[float] = None  # For extended context beyond training
    
    # ==========================================================================
    # BitNet Quantization (Microsoft)
    # ==========================================================================
    use_bitlinear: bool = True
    bitlinear_per_channel: bool = False  # Per-tensor is faster
    quantization_eps: float = 1e-8
    
    # ==========================================================================
    # Precision
    # ==========================================================================
    norm_eps: float = 1e-6
    
    # Tie embeddings to output projection (saves params)
    tie_word_embeddings: bool = True
    
    # ==========================================================================
    # Gemma 2/3 Stability Features
    # ==========================================================================
    use_qk_norm: bool = True  # Apply RMSNorm to Q and K before attention
    attn_logit_soft_cap: Optional[float] = 50.0  # Soft-cap attention logits
    final_logit_soft_cap: Optional[float] = 30.0  # Soft-cap LM head logits
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
        assert self.head_dim == self.hidden_dim // self.num_heads, \
            f"head_dim should be {self.hidden_dim // self.num_heads}"
        # GQA validation
        assert self.num_heads % self.num_kv_heads == 0, \
            f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
    
    @property
    def num_kv_groups(self) -> int:
        """Number of query heads per KV head (GQA group size)."""
        return self.num_heads // self.num_kv_heads
    
    @property
    def num_params_estimate(self) -> int:
        """Estimate total parameters (approximate)."""
        # Embeddings
        embed = self.vocab_size * self.hidden_dim
        
        # Per layer (with GQA)
        # Attention: Q full, K/V reduced by GQA ratio, O full
        kv_dim = self.num_kv_heads * self.head_dim
        attn_q = self.hidden_dim * self.hidden_dim  # Q projection
        attn_k = self.hidden_dim * kv_dim  # K projection (reduced)
        attn_v = self.hidden_dim * kv_dim  # V projection (reduced)
        attn_o = self.hidden_dim * self.hidden_dim  # O projection
        attn = attn_q + attn_k + attn_v + attn_o
        
        # MLP: gate, up, down (SwiGLU)
        mlp = 3 * self.hidden_dim * self.mlp_hidden_dim
        
        # Norms (negligible but included)
        # With QK-Norm: input_norm + post_attn_norm + q_norm + k_norm
        norms = 2 * self.hidden_dim + 2 * self.head_dim if self.use_qk_norm else 2 * self.hidden_dim
        
        per_layer = attn + mlp + norms
        
        # Total
        total = embed + (self.num_layers * per_layer)
        if not self.tie_word_embeddings:
            total += self.vocab_size * self.hidden_dim  # LM head
        
        return total
    
    @property
    def kv_cache_size_per_token(self) -> int:
        """KV cache bytes per token (for memory estimation)."""
        # 2 (K+V) * num_kv_heads * head_dim * 2 bytes (fp16)
        return 2 * self.num_kv_heads * self.head_dim * 2
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
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
            "use_qk_norm": self.use_qk_norm,
            "attn_logit_soft_cap": self.attn_logit_soft_cap,
            "final_logit_soft_cap": self.final_logit_soft_cap,
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
    Training hyperparameters with Stanford Muon optimizer.
    
    Optimizer Strategy (Percy Liang et al., Sept 2025):
      - Matrix params (2D): Muon optimizer (30-40% faster convergence)
      - Vector params (1D): AdamW optimizer
    
    Optimized for RTX 2080 Super (8GB VRAM).
    """
    
    # Batch size
    micro_batch_size: int = 4  # Per-GPU batch size
    gradient_accumulation_steps: int = 8  # Effective batch = 32
    
    # ==========================================================================
    # Optimizer: Hybrid Muon + AdamW (Stanford "Fantastic Optimizers" paper)
    # ==========================================================================
    optimizer: Literal["muon", "adamw", "adam", "sgd"] = "muon"  # Default to Muon
    
    # Muon settings (for 2D matrix parameters - BitLinear weights)
    # NOTE: Muon uses MUCH higher LR due to gradient orthogonalization
    muon_lr: float = 0.02  # High LR is correct for Muon
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.0  # Usually no weight decay for Muon
    
    # AdamW settings (for 1D vector parameters - norms, biases, embeddings)
    adamw_lr: float = 3e-4  # Standard LR for vectors
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.95
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 0.01  # Light weight decay for vectors
    
    # Legacy (for backward compat when optimizer != "muon")
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5  # For cosine decay
    weight_decay: float = 0.1
    
    # Schedule
    # NOTE: Muon converges faster, so shorter warmup is recommended
    warmup_steps: int = 500  # Shorter for Muon (was 1000 for AdamW)
    max_steps: int = 100000
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Memory optimization
    use_amp: bool = True  # Automatic Mixed Precision
    amp_dtype: Literal["float16", "bfloat16"] = "float16"  # bfloat16 better if supported
    gradient_checkpointing: bool = True  # Trade compute for memory
    use_torch_compile: bool = True  # Use torch.compile() for Triton fused kernels (PyTorch 2.0+)
    
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
            "optimizer": self.optimizer,
            # Muon params
            "muon_lr": self.muon_lr,
            "muon_momentum": self.muon_momentum,
            "muon_weight_decay": self.muon_weight_decay,
            # AdamW params
            "adamw_lr": self.adamw_lr,
            "adamw_beta1": self.adamw_beta1,
            "adamw_beta2": self.adamw_beta2,
            "adamw_eps": self.adamw_eps,
            "adamw_weight_decay": self.adamw_weight_decay,
            # Legacy
            "learning_rate": self.learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "weight_decay": self.weight_decay,
            # Schedule
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "max_grad_norm": self.max_grad_norm,
            # Memory
            "use_amp": self.use_amp,
            "amp_dtype": self.amp_dtype,
            "gradient_checkpointing": self.gradient_checkpointing,
            "use_torch_compile": self.use_torch_compile,
            # Checkpointing
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps,
            "output_dir": self.output_dir,
            "checkpoint_dir": self.checkpoint_dir,
            # Logging
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
    Data pipeline configuration with "Phi-Style" textbook quality focus.
    """
    
    # Paths
    train_data: str = "data/distillation/train.jsonl"
    val_data: str = "data/distillation/val.jsonl"
    prompts_dir: str = "data/prompts"
    
    # Tokenizer (Llama-2 32k vocab - "Brain-First" strategy)
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf"
    
    # Fill-In-Middle (FIM) sentinel tokens for code completion
    # These enable code infilling: <PRE>prefix<SUF>suffix<MID>middle
    fim_prefix_token: str = "<|fim_prefix|>"
    fim_middle_token: str = "<|fim_middle|>"
    fim_suffix_token: str = "<|fim_suffix|>"
    fim_pad_token: str = "<|fim_pad|>"
    fim_rate: float = 0.5  # 50% of samples use FIM format
    
    # Processing
    max_length: int = 2048  # Extended for code (matches model max_seq_len)
    num_workers: int = 4
    prefetch_factor: int = 2
    
    # Data generation ("Phi Protocol" - Textbook Quality)
    num_samples_per_prompt: int = 1
    max_concurrent_requests: int = 5
    use_textbook_format: bool = True  # Generate "Concept -> Explanation -> Example"


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
            num_kv_heads=2,  # GQA 4:1 ratio
            head_dim=64,
            mlp_hidden_dim=1408,
            max_seq_len=1024,
        )
    )


def get_config_300m() -> Config:
    """~300M parameter student model (needs 16GB+ VRAM)."""
    return Config(
        model=ModelConfig(
            hidden_dim=1024,
            num_layers=16,
            num_heads=16,
            num_kv_heads=4,  # GQA 4:1 ratio
            head_dim=64,
            mlp_hidden_dim=2816,
            max_seq_len=4096,  # Longer context with GQA
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
    print("Distillix 'Frankenstein' Configuration (~125M params)")
    print("=" * 60)
    
    print(f"\nModel Architecture:")
    print(f"  Vocab size: {config.model.vocab_size:,} (Llama-2 'Brain-First')")
    print(f"  Hidden dim: {config.model.hidden_dim}")
    print(f"  Num layers: {config.model.num_layers}")
    print(f"  Query heads: {config.model.num_heads}")
    print(f"  KV heads: {config.model.num_kv_heads} (GQA {config.model.num_kv_groups}:1)")
    print(f"  MLP hidden: {config.model.mlp_hidden_dim}")
    print(f"  Max seq len: {config.model.max_seq_len}")
    print(f"  RoPE theta: {config.model.rope_base:,.0f} (long context)")
    print(f"  Estimated params: {config.model.num_params_estimate:,}")
    print(f"  KV cache/token: {config.model.kv_cache_size_per_token} bytes")
    
    print(f"\nStability Features (Gemma 2/3):")
    print(f"  QK-Norm: {config.model.use_qk_norm}")
    print(f"  Attn soft-cap: {config.model.attn_logit_soft_cap}")
    print(f"  Final soft-cap: {config.model.final_logit_soft_cap}")
    
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
