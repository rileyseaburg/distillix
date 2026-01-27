#!/bin/bash
# Kimi-K2.5 Deployment Script for Vast.ai
# Requires: H100/H200 GPU with 96GB+ VRAM, AMD EPYC 9xxx or Intel Xeon (AVX512), 300GB+ RAM, 700GB+ disk
#
# Usage:
#   1. Rent a suitable instance on Vast.ai (see requirements above)
#   2. Upload this script to the instance
#   3. Run: bash kimi_k25_vast_deploy.sh
#
# Estimated cost: ~$2-3 for download + testing (at $1.07/hr for H100 NVL)

set -e

echo "========================================"
echo "Kimi-K2.5 Deployment Script for Vast.ai"
echo "========================================"
echo ""

# Check prerequisites
echo "[1/6] Checking prerequisites..."

# Check GPU
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEM" -lt 90000 ]; then
    echo "ERROR: Need 90GB+ VRAM, found ${GPU_MEM}MB"
    exit 1
fi
echo "  GPU Memory: ${GPU_MEM}MB ✓"

# Check AVX512
if ! grep -q "avx512f" /proc/cpuinfo; then
    echo "ERROR: AVX512F not found. Need Intel Xeon or AMD EPYC 9xxx CPU."
    exit 1
fi
echo "  AVX512 support: ✓"

# Check RAM
RAM_GB=$(free -g | grep Mem | awk '{print $2}')
if [ "$RAM_GB" -lt 200 ]; then
    echo "ERROR: Need 200GB+ RAM, found ${RAM_GB}GB"
    exit 1
fi
echo "  RAM: ${RAM_GB}GB ✓"

# Check disk
DISK_GB=$(df -BG / | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "$DISK_GB" -lt 600 ]; then
    echo "ERROR: Need 600GB+ free disk, found ${DISK_GB}GB"
    exit 1
fi
echo "  Free Disk: ${DISK_GB}GB ✓"

echo ""
echo "[2/6] Installing dependencies..."
pip install -q huggingface-hub transformers accelerate safetensors einops

echo ""
echo "[3/6] Setting up KTransformers..."
mkdir -p /workspace
cd /workspace

if [ ! -d "ktransformers" ]; then
    git clone https://github.com/kvcache-ai/ktransformers.git
    cd ktransformers
    git checkout kimi_k2.5
    git submodule update --init --recursive
    cd kt-kernel && ./install.sh
    cd /workspace
fi

echo ""
echo "[4/6] Setting up SGLang..."
if [ ! -d "sglang" ]; then
    git clone https://github.com/kvcache-ai/sglang.git
    cd sglang
    git checkout kimi_k2.5
    pip install -e "python[all]" -q
    pip install nvidia-cudnn-cu12==9.16.0.29 -q
    cd /workspace
fi

echo ""
echo "[5/6] Downloading Kimi-K2.5 model (595GB)..."
echo "This will take approximately 25-30 minutes..."
mkdir -p /workspace/models
huggingface-cli download moonshotai/Kimi-K2.5 --local-dir /workspace/models/kimi-k2.5

echo ""
echo "[6/6] Creating launch script..."
cat > /workspace/launch_kimi.sh << 'LAUNCH'
#!/bin/bash
MODEL_PATH="/workspace/models/kimi-k2.5"

echo "Launching Kimi-K2.5 SGLang Server..."
echo "Model: $MODEL_PATH"

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPU(s)"

# Calculate experts per GPU (384 total, keep ~60 hot on GPU)
GPU_EXPERTS=$((60 / NUM_GPUS))

python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 31245 \
  --model $MODEL_PATH \
  --kt-weight-path $MODEL_PATH \
  --kt-cpuinfer 32 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts $GPU_EXPERTS \
  --kt-method RAWINT4 \
  --trust-remote-code \
  --mem-fraction-static 0.90 \
  --served-model-name Kimi-K2.5 \
  --tensor-parallel-size $NUM_GPUS \
  --max-total-tokens 50000
LAUNCH
chmod +x /workspace/launch_kimi.sh

echo ""
echo "========================================"
echo "Deployment Complete!"
echo "========================================"
echo ""
echo "To start the server:"
echo "  cd /workspace && ./launch_kimi.sh"
echo ""
echo "To test inference:"
echo '  curl -s http://localhost:31245/v1/chat/completions \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"model": "Kimi-K2.5", "messages": [{"role": "user", "content": "Hello!"}]}'"'"
echo ""
