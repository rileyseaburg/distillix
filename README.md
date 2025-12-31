# Distillix

A BitNet b1.58 knowledge distillation framework for training efficient 1.58-bit language models from frontier teacher models.

## Overview

Distillix enables training small, efficient language models using knowledge distillation from large teacher models. The framework implements BitNet b1.58 quantization, where weights are constrained to {-1, 0, +1} (1.58 bits per weight), enabling massive compression and efficient inference.

### Key Features

- **BitNet b1.58 Implementation**: Ternary weight quantization with Straight-Through Estimator (STE) for gradient flow
- **Multi-Teacher Ensemble**: Query multiple frontier models via OpenCode server and aggregate responses
- **Memory Efficient**: Gradient checkpointing, AMP, and optimizations for consumer GPUs (8GB+ VRAM)
- **Modern Architecture**: RoPE, SwiGLU, RMSNorm following Llama/Mistral design patterns

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
│   ├── model.py             # Student Transformer (~125M params)
│   ├── loss.py              # Distillation loss functions
│   ├── data.py              # Dataset and DataLoader
│   ├── config.py            # Model and training configuration
│   └── train.py             # Training loop with AMP
│
├── scripts/                 # Utility scripts
│   ├── setup_cuda.sh
│   ├── start_server.sh
│   └── train.sh
│
└── data/
    └── prompts/             # Input prompts for data generation
```

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

| Config | Parameters | VRAM | Description |
|--------|-----------|------|-------------|
| 50M    | ~50M      | 4GB  | Testing and experimentation |
| 125M   | ~125M     | 8GB  | Default, fits RTX 2080/3060 |
| 300M   | ~300M     | 16GB | Larger model, needs RTX 3090+ |

## BitNet b1.58 Details

BitNet b1.58 quantizes weights to ternary values {-1, 0, +1}:

- **Weight Quantization**: `W_q = round(clip(W / mean(|W|), -1, 1))`
- **Activation Quantization**: INT8 per-token using absmax scaling
- **STE**: Gradients pass through quantization unchanged during backprop

This achieves ~20x compression vs FP32 weights while maintaining competitive accuracy.

## Teacher Models

The framework supports any models accessible via OpenCode server:

- Azure AI Foundry (Claude)
- ZAI Coding Plan (GLM-4)
- MiniMax (M2.1)
- And any other configured providers

## Configuration

See `smelter/config.py` for all available options:

```python
from smelter.config import Config, get_config_125m

config = get_config_125m()
config.training.learning_rate = 1e-4
config.training.max_steps = 50000
config.save("my_config.json")
```

## References

- [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) - BitNet b1.58 paper
- [RoFormer](https://arxiv.org/abs/2104.09864) - Rotary Position Embeddings
- [Distilling the Knowledge](https://arxiv.org/abs/1503.02531) - Knowledge Distillation

## License

MIT License - see [LICENSE](LICENSE) for details.
