# Distillix

A "Frankenstein" BitNet b1.58 knowledge distillation framework combining the best architecture innovations from Microsoft, Meta, and Google for training efficient 1.58-bit coding models.

## The "Royal Flush" Architecture

Distillix steals the best ideas from each lineage:

| Component | Source | Benefit |
|-----------|--------|---------|
| **Math** | BitNet b1.58 (Microsoft) | 1.58-bit weights, ~5x smaller |
| **Tokenizer** | Llama-2 32k | "Brain-First" - 80% params for logic |
| **Attention** | Llama 3 GQA | 3x smaller KV cache |
| **Stability** | Gemma 2/3 | QK-Norm + Soft-Capping |
| **Optimizer** | Stanford Muon | 30-40% faster convergence |
| **Position** | Extended RoPE | theta=1M for long code context |

### Why This Architecture?

**The Vocabulary Trap**: Using Gemma's 256k vocab on a 125M model would consume 100% of parameters on the "dictionary" alone. Llama-2's 32k vocab allocates 80% to the "brain."

**The Stability Problem**: BitNet's ternary weights are notoriously unstable. Gemma 2's soft-capping prevents gradient explosion and enables higher learning rates.

**The Memory Problem**: Standard Multi-Head Attention has massive KV cache. GQA with 12Q/4KV ratio gives 3x reduction, enabling longer code contexts on 8GB VRAM.

**The Optimizer Edge**: Stanford's Sept 2025 "Fantastic Optimizers" paper showed Muon beats AdamW by 30-40% for small models. We use Muon for matrices, AdamW for vectors.

## Architecture

```
distillix/
├── foundry/                 # Data generation from teacher models
│   ├── opencode_client.py   # HTTP client for OpenCode server
│   ├── teacher.py           # Multi-teacher ensemble orchestration
│   └── generate.py          # Dataset generation pipeline
│
├── smelter/                 # BitNet training pipeline
│   ├── bitnet.py            # BitLinear layer with STE
│   ├── model.py             # "Frankenstein" StudentLLM with GQA
│   ├── muon.py              # Stanford Muon optimizer
│   ├── loss.py              # Distillation loss functions
│   ├── data.py              # Dataset and DataLoader
│   ├── config.py            # Full configuration system
│   └── train.py             # Training loop with hybrid optimizer
│
├── scripts/                 # Utility scripts
│   ├── setup_cuda.sh
│   ├── start_server.sh
│   └── train.sh
│
└── data/
    └── prompts/             # Input prompts for data generation
```

## Model Specifications

### Default Configuration (~100M params)

```
Vocabulary:     32,000 (Llama-2 "Brain-First")
Hidden dim:     768
Layers:         12
Query heads:    12
KV heads:       4 (GQA 3:1 ratio)
Head dim:       64
MLP hidden:     2,048
Max seq len:    2,048
RoPE theta:     1,000,000

Stability:
  QK-Norm:      True (Gemma 2)
  Attn cap:     50.0
  Final cap:    30.0

Optimizer:
  Type:         Muon + AdamW hybrid
  Muon LR:      0.02 (matrices)
  AdamW LR:     3e-4 (vectors)
```

### Memory Footprint

| Component | Size |
|-----------|------|
| Weights (1.58-bit) | ~25 MB |
| KV cache @ 2048 tokens | ~2 MB |
| Training (FP32 + optimizer) | ~1.5 GB |

## Requirements

- Python 3.10+
- PyTorch 2.1+
- CUDA 11.8+ (for GPU training)
- 8GB+ VRAM recommended

## Installation

```bash
# Clone the repository
git clone https://github.com/rileyseaburg/distillix.git
cd distillix

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Training Data

First, start the OpenCode server, then generate distillation data:

```bash
# Start OpenCode server (in separate terminal)
opencode serve --port 4096

# Generate training data from teacher models
python -m foundry.generate --builtin --output data/distillation/train.jsonl
```

### 2. Train the Student Model

```bash
python -m smelter.train --config config.json
```

### 3. Run Inference

```python
from smelter.model import StudentLLM
from smelter.config import get_config_125m

config = get_config_125m()
model = StudentLLM(config.model)
model.load_state_dict(torch.load("artifacts/students/model.pt"))

# Generate text
output = model.generate(input_ids, max_new_tokens=100)
```

## Model Configurations

| Config | Parameters | VRAM | GQA Ratio | Description |
|--------|-----------|------|-----------|-------------|
| 50M    | ~50M      | 4GB  | 4:1       | Testing and experimentation |
| 125M   | ~100M     | 8GB  | 3:1       | Default, fits RTX 2080/3060 |
| 300M   | ~300M     | 16GB | 4:1       | Larger model, needs RTX 3090+ |

## Key Innovations

### 1. BitNet b1.58 Core

Weights quantized to ternary values {-1, 0, +1}:
- **Weight Quantization**: `W_q = round(clip(W / mean(|W|), -1, 1))`
- **STE**: Gradients pass through quantization unchanged
- **Result**: ~20x compression vs FP32

### 2. Grouped Query Attention (GQA)

```python
# Standard MHA: 12 Q heads, 12 KV heads
# GQA:          12 Q heads, 4 KV heads (3:1)
k = k.repeat_interleave(num_kv_groups, dim=1)  # Expand 4 -> 12
v = v.repeat_interleave(num_kv_groups, dim=1)
```

### 3. Gemma 2 Stability

```python
# QK-Norm: Normalize Q and K before attention
q = self.q_norm(q)
k = self.k_norm(k)

# Soft-Capping: Bound logits to prevent explosion
logits = 50.0 * torch.tanh(logits / 50.0)
```

### 4. Stanford Muon Optimizer

```python
# Newton-Schulz orthogonalization on momentum
buf_ortho = newton_schulz_orthogonalize(momentum_buffer)

# Split by parameter dimension
# 2D matrices: Muon @ lr=0.02
# 1D vectors:  AdamW @ lr=3e-4
```

## Teacher Models

The framework supports any models accessible via OpenCode server:

- Azure AI Foundry (Claude)
- ZAI Coding Plan (GLM-4)
- MiniMax (M2.1)
- And any other configured providers

## Fill-In-Middle (FIM) Support

Distillix supports code completion via FIM tokens:

```python
# Sentinel tokens
<|fim_prefix|>  # Code before cursor
<|fim_suffix|>  # Code after cursor
<|fim_middle|>  # Model fills this

# 50% of training samples use FIM format
```

## Configuration

See `smelter/config.py` for all available options:

```python
from smelter.config import Config, get_config_125m

config = get_config_125m()

# Adjust optimizer
config.training.muon_lr = 0.01
config.training.adamw_lr = 1e-4

# Adjust stability
config.model.attn_logit_soft_cap = 30.0

config.save("my_config.json")
```

## References

- [BitNet b1.58](https://arxiv.org/abs/2402.17764) - Microsoft's 1.58-bit quantization
- [GQA](https://arxiv.org/abs/2305.13245) - Grouped Query Attention
- [Gemma 2](https://arxiv.org/abs/2408.00118) - Soft-capping and QK-Norm
- [ViT-22B](https://arxiv.org/abs/2302.05442) - QK-Norm for large models
- [RoFormer](https://arxiv.org/abs/2104.09864) - Rotary Position Embeddings
- [Fantastic Optimizers](https://arxiv.org/) - Stanford Muon optimizer (Sept 2025)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{distillix2025,
  author = {Seaburg, Riley},
  title = {Distillix: Frankenstein BitNet Knowledge Distillation},
  year = {2025},
  url = {https://github.com/rileyseaburg/distillix}
}
```
