# Distillix Benchmarks

## Model Versions

| Version | Samples | Steps | Seq Len | Final Loss | Status |
|---------|---------|-------|---------|------------|--------|
| v0.3-burnin | 861 | 10,000 | 256 | 0.28 | ✅ Complete |
| v0.4 | 10,861 | 20,000 | 1024 | - | 🔄 Training |

---

## v0.3 Burn-in (Stability Test)

**Purpose:** Verify architecture stability and training dynamics

### Training Curve
```
Step   500 | Loss: 5.87
Step  1000 | Loss: 2.00
Step  2000 | Loss: 1.00
Step  5000 | Loss: 0.30
Step 10000 | Loss: 0.28
```

### LM-Eval Harness Results (0-shot, n=100)

| Benchmark | Accuracy | Random Baseline |
|-----------|----------|-----------------|
| **HellaSwag** | 28.0% | 25% |
| **ARC-Easy** | 27.0% | 25% |
| **PIQA** | 53.0% | 50% |
| **BoolQ** | 48.0% | 50% |

⚠️ **Note:** Model trained on only 861 samples for stability testing. 
Performance near random baseline is expected - real capability requires 
training on the full 10k+ dataset.

### Inference Benchmarks

| Metric | Value |
|--------|-------|
| Parameters | 100.1M |
| Perplexity (val) | 26,079* |
| Throughput (512 seq) | 14,984 tokens/sec |
| Generation Speed | 19.7 tokens/sec |
| Peak Memory | 988 MB |

*High perplexity expected - model memorized small dataset

### Inference Speed by Sequence Length

| Seq Len | Time (ms) | Tokens/sec | Memory (MB) |
|---------|-----------|------------|-------------|
| 64 | 36 | 1,777 | 822 |
| 128 | 34 | 3,750 | 846 |
| 256 | 36 | 7,117 | 895 |
| 512 | 34 | 14,984 | 988 |

---

## v0.4 Real Dataset (Planned)

**Purpose:** Train on real-world coding data for actual capability

### Configuration
```
Dataset:      10,861 samples (SWE-rebench trajectories)
Seq Length:   1024 tokens (4x v0.3)
Batch Size:   2 x 8 = 16 effective
Optimizer:    Muon + AdamW hybrid
Checkpointing: Enabled (2.27 GB VRAM)
Target Steps: 20,000
```

### Expected Improvements
- Training on 12x more data (10,861 vs 861 samples)
- 4x longer context (1024 vs 256 tokens)
- Real-world coding trajectories from SWE-rebench
- Should show meaningful improvement over random baseline

---

## Hardware

- **GPU:** NVIDIA RTX 2080 Super (8GB VRAM)
- **CPU:** AMD Threadripper 3960X
- **RAM:** 30GB

---

## Architecture

| Component | Specification |
|-----------|--------------|
| Base | BitNet b1.58 (1.58-bit ternary weights) |
| Attention | GQA (12 Q-heads / 4 KV-heads) |
| Stability | QK-Norm + Logit Soft-Capping |
| Position | RoPE (theta=1M) |
| Vocab | 32,000 (Llama-2 tokenizer) |
| Hidden Dim | 768 |
| Layers | 12 |
| MLP Dim | 2,048 |
| Max Seq | 2,048 |

---

## Files

| File | Description | Size |
|------|-------------|------|
| `distillix-v0.3-burnin-861.pt` | Stability test checkpoint | 400 MB |
| `distillix-v0.safetensors` | SafeTensors export | 382 MB |
| `distillix-v0.3.gguf` | GGUF for llama.cpp | 191 MB |

---

## Checkpoints

HuggingFace: https://huggingface.co/rileyseaburg/distillix-100m-v0.3

GitHub: https://github.com/rileyseaburg/distillix
