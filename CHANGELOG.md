# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-12-31

### Added

- **Smelter**: BitNet b1.58 training pipeline
  - `bitnet.py`: BitLinear layer with ternary weight quantization ({-1, 0, +1}) and STE
  - `model.py`: Student Transformer (~125M params) with RoPE, SwiGLU, RMSNorm
  - `loss.py`: Distillation loss functions (CE, KL divergence, multi-teacher)
  - `data.py`: Dataset and DataLoader for JSONL training data
  - `config.py`: Model and training configuration with presets (50M, 125M, 300M)
  - `train.py`: Training loop with AMP, gradient checkpointing, and W&B logging

- **Foundry**: Data generation from teacher models
  - `opencode_client.py`: HTTP client for OpenCode server API
  - `teacher.py`: Multi-teacher ensemble with configurable strategies
  - `generate.py`: Dataset generation pipeline with built-in seed prompts

- **Scripts**: Utility scripts for setup and training
  - `setup_cuda.sh`: NVIDIA driver and PyTorch installation
  - `start_server.sh`: Launch OpenCode server
  - `train.sh`: Run training pipeline

- **Documentation**
  - MIT License
  - README with installation and usage instructions

### Technical Details

- Target hardware: RTX 2080 Super (8GB VRAM), Threadripper 3960X, 30GB RAM
- Default model: ~125M parameters, 12 layers, 768 hidden dim
- Sequence length: 1024 tokens (conservative for 8GB VRAM)
- Teachers: Azure Claude, GLM-4.7, MiniMax M2.1 via OpenCode server
