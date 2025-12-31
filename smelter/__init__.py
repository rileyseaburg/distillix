"""
Smelter: BitNet Training Pipeline for Distillix.

Components:
  - model: StudentLLM "Frankenstein" architecture
  - bitnet: BitLinear layer with 1.58-bit quantization
  - muon: Stanford Muon optimizer for matrix-based training
  - train: Trainer class with hybrid Muon+AdamW
  - config: Configuration dataclasses
  - data: Dataset and DataLoader utilities
  - loss: Distillation loss functions

Copyright (c) 2025 Distillix. All Rights Reserved.
"""

from .config import (
    ModelConfig,
    TrainingConfig,
    DistillationConfig,
    DataConfig,
    Config,
    get_config_125m,
    get_config_50m,
    get_config_300m,
)

__all__ = [
    "ModelConfig",
    "TrainingConfig", 
    "DistillationConfig",
    "DataConfig",
    "Config",
    "get_config_125m",
    "get_config_50m",
    "get_config_300m",
]
