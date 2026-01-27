#!/bin/bash
# =============================================================================
# GLM-4.7-Flash Quantization on Vast.ai
# =============================================================================
#
# This script is designed to run on a Vast.ai A100 instance.
# It quantizes GLM-4.7-Flash to 2-bit for RTX 2080 SUPER (8GB) inference.
#
# Usage:
#   1. Rent an A100 on Vast.ai (40GB or 80GB)
#   2. Upload this script to the instance
#   3. Run: bash vast_quantize_glm.sh
#   4. Download the quantized model (~8GB)
#
# Estimated time: 30-60 minutes
# Estimated cost: ~$1-2 (A100 @ $1.50/hr)
# =============================================================================

set -e

echo "============================================================"
echo "GLM-4.7-Flash 2-bit Quantization for 8GB GPUs"
echo "============================================================"

# Check GPU
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Install dependencies
echo "[1/5] Installing dependencies..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers accelerate safetensors huggingface_hub tqdm

# Download model
echo "[2/5] Downloading GLM-4.7-Flash..."
python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

# Download to local cache
local_dir = snapshot_download(
    "zai-org/GLM-4.7-Flash",
    local_dir="./glm-4.7-flash-bf16",
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
)
print(f"Downloaded to: {local_dir}")
EOF

# Create quantization script
echo "[3/5] Running layer-by-layer 2-bit quantization..."
python3 << 'QUANTIZE_EOF'
import gc
import json
import os
from pathlib import Path
from typing import Dict, Tuple
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# Config
BITS = 2
GROUP_SIZE = 128
INPUT_DIR = Path("./glm-4.7-flash-bf16")
OUTPUT_DIR = Path("./glm-4.7-flash-2bit")
OUTPUT_DIR.mkdir(exist_ok=True)

# Skip quantization for these (keep fp16)
SKIP_PATTERNS = [
    "embed_tokens", "lm_head",  # Embeddings
    "layernorm", "norm",  # Norms
    "router",  # MoE router (important for routing quality)
]

def quantize_symmetric(weight: torch.Tensor, bits: int, group_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-group quantization."""
    out_features, in_features = weight.shape
    
    # Pad if needed
    if in_features % group_size != 0:
        pad = group_size - (in_features % group_size)
        weight = torch.nn.functional.pad(weight, (0, pad))
        in_features = weight.shape[1]
    
    # Reshape for groupwise
    weight = weight.reshape(out_features, -1, group_size)
    
    # Per-group scales
    scales = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    
    # Quantize
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    w_scaled = weight / scales * qmax
    q_weight = torch.clamp(torch.round(w_scaled), qmin, qmax).to(torch.int8)
    
    return q_weight.reshape(out_features, in_features), scales.reshape(out_features, -1)

def pack_int2(q_weight: torch.Tensor) -> torch.Tensor:
    """Pack 4 x int2 into uint8."""
    out_f, in_f = q_weight.shape
    assert in_f % 4 == 0
    q_weight = q_weight.reshape(out_f, -1, 4)
    q_weight = (q_weight + 2).to(torch.uint8)  # Shift to unsigned
    packed = (q_weight[:,:,0] << 6) | (q_weight[:,:,1] << 4) | (q_weight[:,:,2] << 2) | q_weight[:,:,3]
    return packed

# Find safetensor shards
shards = sorted(INPUT_DIR.glob("model*.safetensors"))
if not shards:
    shards = list(INPUT_DIR.glob("*.safetensors"))
print(f"Found {len(shards)} shard(s)")

# Copy config
config_path = INPUT_DIR / "config.json"
if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

# Process each shard
output_tensors = {}
quant_info = {}
total_original = 0
total_quantized = 0

for shard_path in tqdm(shards, desc="Processing shards"):
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for name in tqdm(f.keys(), desc=f"  {shard_path.name}", leave=False):
            tensor = f.get_tensor(name)
            original_size = tensor.numel() * tensor.element_size()
            total_original += original_size
            
            # Check if should skip
            should_skip = any(p in name.lower() for p in SKIP_PATTERNS)
            is_weight = "weight" in name and len(tensor.shape) == 2
            
            if should_skip or not is_weight:
                # Keep as fp16
                output_tensors[name] = tensor.to(torch.float16)
                total_quantized += tensor.numel() * 2
            else:
                # Quantize
                out_f, in_f = tensor.shape
                
                # Pad in_features to multiple of 4 for int2 packing
                pad_in = (4 - (in_f % 4)) % 4
                if pad_in:
                    tensor = torch.nn.functional.pad(tensor, (0, pad_in))
                
                # Also pad to group_size
                padded_in = tensor.shape[1]
                pad_group = (GROUP_SIZE - (padded_in % GROUP_SIZE)) % GROUP_SIZE
                if pad_group:
                    tensor = torch.nn.functional.pad(tensor, (0, pad_group))
                
                # Quantize on GPU if available
                if torch.cuda.is_available():
                    tensor = tensor.cuda()
                
                q_weight, scales = quantize_symmetric(tensor.float(), BITS, GROUP_SIZE)
                
                # Pack to uint8
                packed = pack_int2(q_weight.cpu())
                
                # Store
                output_tensors[f"{name}.qweight"] = packed
                output_tensors[f"{name}.scales"] = scales.cpu().to(torch.float16)
                output_tensors[f"{name}.shape"] = torch.tensor([out_f, in_f])  # Original shape
                
                total_quantized += packed.numel() + scales.numel() * 2 + 8
                
                quant_info[name] = {"bits": BITS, "group_size": GROUP_SIZE, "shape": [out_f, in_f]}
                
                del tensor, q_weight, scales, packed
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            gc.collect()

# Save
print("\nSaving quantized model...")
save_file(output_tensors, OUTPUT_DIR / "model.safetensors")

# Save quant config
with open(OUTPUT_DIR / "quantize_config.json", "w") as f:
    json.dump({
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "quantized_tensors": quant_info,
    }, f, indent=2)

# Summary
compression = total_original / total_quantized
print(f"\n{'='*60}")
print(f"QUANTIZATION COMPLETE")
print(f"{'='*60}")
print(f"Original:   {total_original / 1e9:.2f} GB")
print(f"Quantized:  {total_quantized / 1e9:.2f} GB")
print(f"Compression: {compression:.1f}x")
print(f"Output:     {OUTPUT_DIR}")
QUANTIZE_EOF

# Package for download
echo "[4/5] Creating archive..."
cd glm-4.7-flash-2bit
tar -cvzf ../glm-4.7-flash-2bit.tar.gz .
cd ..

# Show results
echo "[5/5] Done!"
echo ""
echo "============================================================"
echo "RESULTS"
echo "============================================================"
ls -lh glm-4.7-flash-2bit/
echo ""
ls -lh glm-4.7-flash-2bit.tar.gz
echo ""
echo "Download the model:"
echo "  scp user@vast-instance:~/glm-4.7-flash-2bit.tar.gz ."
echo ""
echo "Or use vast.ai file transfer"
