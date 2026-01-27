# LEFTOFF.md - Kimi-K2.5 Deployment

**Last Updated:** 2026-01-27 12:15 UTC

## Current Status: DEPLOYING KIMI-K2.5 ON VAST.AI

Setting up Kimi-K2.5 (1T param MoE model) for inference on Vast.ai with KTransformers + SGLang.

---

## Active Instance

| Field | Value |
|-------|-------|
| **Instance ID** | 30622181 |
| **GPU** | 2x RTX PRO 6000 WS (98GB VRAM each) |
| **CPU** | AMD EPYC 9534 64-Core (AVX512) |
| **RAM** | **619GB** (required for expert offloading) |
| **Disk** | 700GB allocated |
| **Cost** | $1.74/hr |
| **Status** | Starting up... |

### SSH Access
```bash
# Wait for instance to be ready, then:
ssh -p <PORT> root@ssh<N>.vast.ai
```

Check instance status:
```bash
curl -s "https://console.vast.ai/api/v0/instances/30622181/?api_key=9b698b6991d82f28d7404a53c70267cef1769bcb3c339e7391c9f718dd755725"
```

---

## Setup Commands (once instance is ready)

### 1. Install Dependencies
```bash
pip install huggingface-hub transformers accelerate safetensors einops

cd /workspace
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers && git checkout kimi_k2.5
git submodule update --init --recursive
cd kt-kernel && ./install.sh

cd /workspace
git clone https://github.com/kvcache-ai/sglang.git
cd sglang && git checkout kimi_k2.5
pip install -e "python[all]"
pip install nvidia-cudnn-cu12==9.16.0.29
```

### 2. Download Model (595GB, ~30 min)
```bash
mkdir -p /workspace/models
huggingface-cli download moonshotai/Kimi-K2.5 --local-dir /workspace/models/kimi-k2.5
```

### 3. Launch Server
```bash
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 31245 \
  --model /workspace/models/kimi-k2.5 \
  --kt-weight-path /workspace/models/kimi-k2.5 \
  --kt-cpuinfer 32 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 30 \
  --kt-method RAWINT4 \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --served-model-name Kimi-K2.5 \
  --tensor-parallel-size 2 \
  --max-total-tokens 30000
```

### 4. Test Inference
```bash
curl -s http://localhost:31245/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Kimi-K2.5", "messages": [{"role": "user", "content": "Hello!"}]}'
```

---

## Kimi-K2.5 Architecture

| Spec | Value |
|------|-------|
| Total Parameters | 1T |
| Activated Parameters | 32B |
| Layers | 61 (1 dense + 60 MoE) |
| Experts | 384 routed + 1 shared |
| Experts per Token | 8 |
| Context Length | 256K |
| Attention | MLA |
| Quantization | Native INT4 |
| Model Size | 595GB |

---

## Requirements

- **GPU**: 48GB+ VRAM (2x 4090 or 1x H100)
- **CPU**: AVX512 support (Intel Xeon or AMD EPYC 9xxx)
- **RAM**: **600GB+** for expert offloading
- **Disk**: 700GB+ for model

---

## Previous Attempts

### Attempt 1: 2x RTX PRO 6000 WS (1TB RAM)
- Instance ID: 30608437
- Issue: AMD EPYC 7262 doesn't have AVX512
- Resolution: Destroyed, found new instance

### Attempt 2: H100 NVL (322GB RAM)
- Instance ID: 30613566
- Issue: Only 314GB RAM, OOM when loading experts to CPU
- Resolution: Destroyed, need 600GB+ RAM

### Attempt 3: 2x RTX PRO 6000 WS (619GB RAM) - CURRENT
- Instance ID: 30622181
- CPU: AMD EPYC 9534 (AVX512)
- Status: Starting up...

---

## Vast.ai API Key
```
9b698b6991d82f28d7404a53c70267cef1769bcb3c339e7391c9f718dd755725
```

## Cost Summary

| Item | Cost |
|------|------|
| Instance 30608437 (no AVX512) | ~$0.50 |
| Instance 30613566 (not enough RAM) | ~$2.50 |
| Instance 30622181 (current) | $1.74/hr |
| **Total so far** | ~$3 + running |
| **Remaining balance** | ~$9.78 |

---

## Code Pushed to GitHub

Commit: `a871354` - Add Kimi-K2.5 and GLM inference scripts for Vast.ai

Files added:
- `scripts/kimi_k25_vast_deploy.sh` - One-click deployment script
- `scripts/glm_inference_full.py` - GLM-4.7-Flash inference engine
- `scripts/glm_inference_int4.py` - Simpler INT4 inference
- `scripts/vast_quantize_glm.sh` - GLM quantization script
- `scripts/cleanup_vast.sh` - Instance cleanup utility
- `AGENTS.md` - Project documentation

---

## Next Steps

1. Wait for instance 30622181 to be ready
2. Run setup commands above
3. Download Kimi-K2.5 model (~30 min)
4. Launch SGLang server
5. Test inference
6. If successful, document final setup

---

## Alternative: Smaller Models

If Kimi-K2.5 proves too difficult, consider:
- **DeepSeek-V2-Lite (14B)** - Fits in 6GB VRAM
- **Kimi-K2-Instruct (smaller)** - May have GGUF versions
- **GLM-4.7-Flash** - Already have inference engine (but model was truncated)
