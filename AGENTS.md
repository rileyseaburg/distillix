# AGENTS.md - Distillix Codebase Guide

This document provides critical context for AI agents working with the Distillix codebase.

## Project Overview

Distillix is a ~125M parameter BitNet-based language model designed for code completion on edge devices. The goal is to fit within L2 cache (~100MB) for instant inference.

## Architecture

### Model: StudentLLM (`smelter/model.py`)
- **125M parameters** (target: fit in L2 cache when quantized)
- **12 transformer layers**
- **768 hidden dim**, 2048 MLP hidden dim
- **12 query heads, 4 KV heads** (GQA - 3x KV cache reduction)
- **32k vocab** (Llama-2 tokenizer)
- **RoPE** with theta=1M for long context
- **QK-Norm** (Gemma 2 style stability)
- **Logit soft-capping** (Gemma 2 style)

### BitNet 1.58-bit Quantization (`smelter/bitnet.py`)
- Ternary weights: {-1, 0, +1}
- ~5x memory reduction at inference
- Uses `BitLinear` and `BitLinearFFN` layers

## Critical State Dict Conventions

### Key Naming Pattern
When loading packed checkpoints, be aware of the naming convention:

```
layers.0.self_attn.q_proj
                     ├── .weight      (The Tensor - PyTorch Parameter)
                     ├── _shape       (Metadata Buffer - note underscore, not dot)
                     └── _scale       (Metadata Buffer - note underscore, not dot)
```

- **Weights**: Use `.` separator (e.g., `layers.0.mlp.gate_proj.weight`)
- **Metadata**: Use `_` separator (e.g., `layers.0.mlp.gate_proj_shape`)

### Unpacking i2s-Packed Weights

The exports use int2-storage (i2s) packing where 4 ternary weights are stored per byte:

```python
def unpack_i2s(packed, shape_tensor, scale):
    """Unpack int2-stored ternary weights."""
    shape = tuple(shape_tensor.tolist())
    numel = 1
    for s in shape:
        numel *= s
    
    packed = packed.to(torch.uint8)
    
    # Unpack 4 weights from each byte
    w0 = ((packed >> 6) & 0x03).to(torch.int8)
    w1 = ((packed >> 4) & 0x03).to(torch.int8)
    w2 = ((packed >> 2) & 0x03).to(torch.int8)
    w3 = (packed & 0x03).to(torch.int8)
    
    # Interleave
    unpacked = torch.stack([w0, w1, w2, w3], dim=1).flatten()
    
    # Map back: 0->-1, 1->0, 2->+1
    unpacked = unpacked - 1
    
    # Truncate and reshape
    unpacked = unpacked[:numel].view(shape).to(torch.float32)
    
    return unpacked * scale.item()
```

## Model Config Flags

When loading checkpoints, ensure these flags match the checkpoint:

```python
config = get_config_125m()
config.model.use_bitlinear_ffn = False   # True if using BitLinearFFN wrapper
config.model.use_bitlinear_attn = False  # True if attention uses BitLinear
config.model.use_qk_norm = True          # QK normalization (check checkpoint)
config.model.tie_word_embeddings = True  # lm_head tied to embed_tokens
```

## Known Issues

### 1. BitNet Weight Collapse (CRITICAL)
**Status**: All current checkpoints have collapsed MLP weights.

**Symptoms**:
- MLP weights (gate_proj, up_proj, down_proj) have Std ≈ 0.0
- Model outputs gibberish despite attention weights being healthy
- Attention weights in middle layers (4-5) also collapsed

**Root Cause**: Ternary quantization during training drove weights to zero.

**Affected Checkpoints**:
- `distillix-codetether-*.pt` - ALL MLP weights are zero (dead from step 1000)
- `distillix-grok-*.pt` - ALL MLP weights are zero  
- `distillix-v04-*.pt` - Collapse observed over training:
  - Step 2000: Std=4.5e-5 (barely alive)
  - Step 10000: Std=6.2e-6 (dying)
  - Step 20000: Std=5.1e-7 (dead)
- `exports/*.pt` - Packed versions of collapsed weights

**Healthy Checkpoints** (MLP Std > 0.001):
- `model_500steps.pt` - Std=0.0084 (HEALTHY but undertrained)
- `model_1500steps.pt` - Std=0.0051 (HEALTHY but undertrained)

**Root Cause**: Weight decay + ternary quantization. The optimizer shrinks weights
faster than gradients can grow them back, and once weights fall into the "0 bucket"
of ternary quantization, they become sticky.

**The Math**: In BitNet, quantization maps weights to {-1, 0, +1} using a threshold.
Standard L2 regularization creates a "bowl" pulling all weights toward 0:
  E_reg = λ||w||² → gradient always points to origin

Once |w| < threshold, the forward pass becomes w_q = 0, disconnecting the neuron.
Weight decay provides constant pressure toward 0, while noisy gradients average out.
Result: weights get trapped in the "zero bucket" and can't escape.

**The Fix - Double-Well Potential**: Replace the convex bowl with a "W-shaped" 
(sombrero) potential that REPELS weights from zero:
  E_new = λ(|w| - target)² 
  
This creates:
- Repulsion force when |w| < target (pushes away from 0)
- Attraction force when |w| > target (pulls toward ±target)

**Implementation** (Polarized Optimizer):
```python
def polarized_step(model, optimizer, target_scale=0.01, resurrection_strength=1e-2):
    optimizer.step()
    optimizer.zero_grad()
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if any(x in name for x in ['gate_proj', 'up_proj', 'down_proj']):
                w = param.data
                death_zone = (w.abs() < target_scale * 0.5) & (w.abs() > 1e-9)
                repulsion = w.sign() * resurrection_strength
                param.data.add_(repulsion * death_zone.float())
```

**Target Scale**: From healthy checkpoints, MLP abs_mean ≈ 0.005-0.007.
Use `target_scale = 0.01` for the polarizer.

**Next Steps**: Need to retrain with:
1. Disable standard weight decay (`weight_decay=0.0` in AdamW)
2. Use polarized_step() instead of optimizer.step()
3. Resume from `model_500steps.pt` or `model_1500steps.pt`

**NEW: Anti-Collapse Optimizers** (`smelter/spin_solver.py`):

Two approaches are now available:

1. **BitNetSpinSolver** - Full "Quantum-Discrete" optimizer:
   - Treats weights as spins on a hypercube
   - Uses gradient SIGN (direction), not magnitude
   - Hysteresis buffer prevents premature bit flips
   - Dead zone tunneling forces escape from 0
   
   ```python
   from smelter.spin_solver import BitNetSpinSolver
   optimizer = BitNetSpinSolver(model.parameters(), lr=1e-3, hysteresis_threshold=0.05)
   ```

2. **PolarizedOptimizer** - Wrapper for any base optimizer:
   - Works with AdamW, Muon, SGD, etc.
   - Adds post-step polarization to prevent collapse
   - Simpler to integrate, preserves base optimizer behavior
   
   ```python
   from smelter.spin_solver import create_polarized_adamw
   optimizer = create_polarized_adamw(model, lr=3e-4, target_scale=0.01)
   ```

**Training Script**:
```bash
# Resume from healthy checkpoint with polarization
python scripts/train_polarized.py --resume artifacts/model_500steps.pt --steps 50000

# Key flags:
#   --target-scale 0.01      # Target weight magnitude
#   --polarization-strength 0.1  # How hard to push from 0
```

**The Physics Behind It** (Spin Glass / Ising Model):
- BitNet is analogous to a spin glass: weights are "spins" trying to be ±1
- Standard weight decay creates a convex "bowl" potential pulling spins to 0
- Ternary quantization creates "sticky" zones around {-1, 0, +1}
- Once |w| < threshold, the spin gets trapped at 0 (paramagnetic phase)
- Solution: Double-well ("sombrero") potential that REPELS from 0
- This enforces "ferromagnetic" order where spins align to ±target

### 2. No KV-Cache Support
**Status**: Missing feature.

The `StudentLLM.forward()` does not accept `past_key_values`, causing O(n^2) complexity during generation. Each new token requires reprocessing the entire sequence.

**Impact**: Slow inference, especially on CPU.

**Fix Required**: Add KV-cache support to `Attention` and `StudentLLM` classes.

### 3. Inference Speed on CPU
Without KV-cache, a 125M model generates ~1 token/second on CPU due to redundant computation. GPU masks this issue but it's critical for edge deployment.

## File Structure

```
distillix/
├── smelter/
│   ├── model.py          # StudentLLM model definition
│   ├── bitnet.py         # BitLinear, BitLinearFFN, RMSNorm
│   ├── config.py         # Model configs (get_config_125m)
│   ├── train.py          # Training loop (Trainer class)
│   ├── muon.py           # Stanford Muon optimizer
│   ├── spin_solver.py    # Anti-collapse optimizers (BitNetSpinSolver, PolarizedOptimizer)
│   ├── loss.py           # Distillation loss functions
│   ├── data.py           # Dataset utilities
│   └── export.py         # GGUF/SafeTensors export
├── scripts/
│   ├── pack_weights.py   # i2s packing for ternary weights
│   ├── train_polarized.py # Training with anti-collapse (RECOMMENDED)
│   ├── train_codetether.py
│   └── train_500m.py     # Larger model training
├── artifacts/            # Training checkpoints (400MB each, FP32)
│   ├── model_500steps.pt   # HEALTHY early checkpoint
│   ├── model_1500steps.pt  # HEALTHY early checkpoint
│   └── distillix-v*.pt     # Later checkpoints (may have collapsed weights)
├── exports/              # Packed/exported models
│   ├── *.pt              # i2s-packed PyTorch (68MB)
│   ├── *.gguf            # llama.cpp format
│   └── *.safetensors     # SafeTensors format
├── chat.py               # Interactive chat script
└── data/                 # Training data
    ├── distillation/     # Distillation datasets
    └── training/         # Direct training data
```

## Diagnostic Protocol

When testing a checkpoint, use this 3-step protocol:

### Step 1: Greedy Test (Syntax Check)
```bash
# Temperature 0.0 (or 0.1 with top_k=1)
# Prompt: "def fibonacci(n):"
# Pass: Completes with valid Python syntax
# Fail (Collapse): "return return return..."
# Fail (Gibberish): Random symbols
```

### Step 2: Logic Test (Intelligence Check)
```bash
# Temperature 0.1
# Prompt: "Write a function to reverse a list without using .reverse()"
# Pass: Uses loop or slicing [::-1]
# Fail: Hallucinates fake methods
```

### Step 3: Weight Sanity Check
```python
import torch
ckpt = torch.load('checkpoint.pt', map_location='cpu')
state = ckpt.get('model_state_dict', ckpt)

for key in state:
    w = state[key]
    if 'proj' in key:  # Check projection weights
        print(f"{key}: Std={w.std():.6f}")
        # Std should be 0.01-0.05 for healthy weights
        # Std ≈ 0.0 means collapsed
```

## Physics Constraints (Why 100M)

| Model Size | Memory | Where it Lives | Speed |
|------------|--------|----------------|-------|
| 100M | ~112MB | L2 Cache | Instant (2ns access) |
| 300M | ~450MB | RAM | Fast (100ns access) |
| 1B | ~2GB | RAM + Swap | Slow, fans spin |

The 100M target is physics-driven: fit in L2 cache for instant inference on laptops.

## Contact

For issues with this codebase, check:
1. This document for known issues
2. `CHANGELOG.md` for recent changes
3. `BENCHMARKS.md` for performance data
