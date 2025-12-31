# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-12-31

### Added

- **"Frankenstein" Hybrid Architecture**: Combined best innovations from multiple sources
  - Llama 3 GQA (Grouped Query Attention) with 12Q/4KV heads for 3x KV cache reduction
  - Gemma 2/3 stability features: QK-Norm and Logit Soft-Capping
  - Extended RoPE with theta=1,000,000 for long code context
  - Fill-In-Middle (FIM) token support for code completion

- **Stanford Muon Optimizer** (`smelter/muon.py`)
  - Newton-Schulz orthogonalization for momentum
  - `Muon` class for 2D matrix parameters
  - `MuonAdamW` hybrid class splitting params by dimensionality
  - 30-40% faster convergence vs AdamW for small models
  - Based on "Fantastic Pretraining Optimizers" (Percy Liang et al., Sept 2025)

- **GQA Implementation** (`smelter/model.py`)
  - Reduced K/V projections: `kv_dim = num_kv_heads * head_dim`
  - `repeat_interleave` expansion for attention computation
  - Configurable `num_kv_heads` (default: 4 for 3:1 ratio)

- **Gemma 2 Stability** (`smelter/model.py`)
  - QK-Norm: RMSNorm applied to Q and K per-head before attention
  - Attention logit soft-capping: `50.0 * tanh(logits / 50.0)`
  - Final logit soft-capping: `30.0 * tanh(logits / 30.0)`
  - Prevents gradient explosion in BitNet training

- **New Configuration Options** (`smelter/config.py`)
  - `num_kv_heads`: Number of key/value heads for GQA
  - `use_qk_norm`: Enable/disable QK normalization
  - `attn_logit_soft_cap`: Attention soft-cap value (None to disable)
  - `final_logit_soft_cap`: LM head soft-cap value (None to disable)
  - `muon_lr`, `muon_momentum`: Muon optimizer settings
  - `adamw_lr`, `adamw_weight_decay`: AdamW settings for vectors
  - FIM tokens: `fim_prefix_token`, `fim_middle_token`, `fim_suffix_token`

- **torch.compile() Integration** (`smelter/train.py`)
  - Automatic Triton kernel fusion for PyTorch 2.0+
  - Configurable via `use_torch_compile` flag

### Changed

- **Default Configuration**
  - `vocab_size`: 32,000 (was 32,000, confirmed Llama-2 "Brain-First")
  - `max_seq_len`: 2,048 (was 1,024, enabled by GQA memory savings)
  - `rope_base`: 1,000,000 (was 10,000, for long code context)
  - `optimizer`: "muon" (was "adamw")

- **Parameter Estimation**
  - `num_params_estimate` now accounts for GQA reduced K/V projections
  - Added `kv_cache_size_per_token` property

- **Training Loop**
  - Hybrid optimizer step handling for Muon + AdamW
  - Separate AMP scaler handling for each optimizer
  - LR scheduler applied only to AdamW (Muon uses fixed high LR)

### Technical Details

- GQA memory savings: 3x reduction in KV cache
- Muon learning rate: 0.02 (vs 3e-4 for AdamW)
- Estimated parameters: ~100M (down from ~125M due to GQA)
- KV cache per token: 1,024 bytes (vs 3,072 without GQA)

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
