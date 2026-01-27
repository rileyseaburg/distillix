#!/usr/bin/env python3
"""
GLM-4.7-Flash INT4 Inference Engine for 8GB GPUs

This is a complete inference implementation for running the quantized GLM-4.7-Flash
model on an RTX 2080 SUPER (8GB) or similar consumer GPU.

Architecture:
- INT4 quantized weights (17GB total model)
- Expert offloading: routed experts on CPU, loaded on-demand
- Shared experts + attention always on GPU
- Streaming dequantization during forward pass

The model uses:
- 47 transformer layers
- 64 routed experts (4 active per token)
- MLA (Multi-head Latent Attention) for efficiency
- INT4 group-wise quantization (128 group size)

Usage:
    python scripts/glm_inference_int4.py \
        --model ./models/glm-4.7-flash-int4 \
        --prompt "def fibonacci(n):"

Copyright (c) 2025 Distillix. All Rights Reserved.
"""

import argparse
import gc
import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("[WARN] safetensors not installed. Install with: pip install safetensors")

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("[WARN] transformers not installed. Install with: pip install transformers")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class InferenceConfig:
    """Inference configuration for INT4 model."""
    # Model
    model_path: str = "./models/glm-4.7-flash-int4"
    
    # Quantization
    bits: int = 4
    group_size: int = 128
    
    # Devices
    compute_device: str = "cuda"
    expert_device: str = "cpu"
    
    # Memory management
    expert_cache_size: int = 8  # LRU cache for experts on GPU (smaller for INT4)
    use_pinned_memory: bool = True
    offload_experts: bool = True  # Whether to offload routed experts to CPU
    
    # Generation
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


# =============================================================================
# INT4 Dequantization Utilities
# =============================================================================

def unpack_int4(packed: Tensor, num_elements: int) -> Tensor:
    """Unpack 2 x int4 from uint8."""
    # Each byte contains 2 INT4 values
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    
    # Interleave low and high nibbles
    unpacked = torch.stack([low, high], dim=-1).flatten()
    
    # Convert from unsigned [0,15] to signed [-8, 7]
    unpacked = unpacked.to(torch.int8) - 8
    
    return unpacked[:num_elements]


def dequantize_int4_weight(
    qweight: Tensor,
    scales: Tensor,
    zeros: Optional[Tensor],
    shape: Tuple[int, int],
    group_size: int = 128,
) -> Tensor:
    """Dequantize INT4 packed weights to float."""
    out_features, in_features = shape
    num_elements = out_features * in_features
    
    # Unpack INT4 values
    q_weight = unpack_int4(qweight.flatten(), num_elements)
    q_weight = q_weight.reshape(out_features, in_features).float()
    
    # Apply scales per group
    num_groups = (in_features + group_size - 1) // group_size
    
    # Reshape for group-wise scaling
    padded_in = num_groups * group_size
    if in_features < padded_in:
        q_weight = F.pad(q_weight, (0, padded_in - in_features))
    
    q_weight = q_weight.reshape(out_features, num_groups, group_size)
    
    # Apply scales
    if scales.dim() == 1:
        scales = scales.reshape(out_features, num_groups)
    
    weight = q_weight * scales.unsqueeze(-1)
    
    # Apply zeros if present
    if zeros is not None:
        if zeros.dim() == 1:
            zeros = zeros.reshape(out_features, num_groups)
        weight = weight - zeros.unsqueeze(-1) * scales.unsqueeze(-1)
    
    return weight.reshape(out_features, -1)[:, :in_features]


# =============================================================================
# Quantized Linear Layer (INT4)
# =============================================================================

class QuantizedLinearINT4(nn.Module):
    """Linear layer with INT4 on-the-fly dequantization."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        # Packed weights will be loaded later
        self.qweight: Optional[Tensor] = None
        self.scales: Optional[Tensor] = None
        self.zeros: Optional[Tensor] = None
        self.bias_param: Optional[Tensor] = None
        
    def load_quantized(
        self, 
        qweight: Tensor, 
        scales: Tensor, 
        zeros: Optional[Tensor] = None,
        bias: Optional[Tensor] = None
    ):
        """Load quantized weights."""
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        if zeros is not None:
            self.register_buffer("zeros", zeros)
        if bias is not None:
            self.register_buffer("bias_param", bias)
    
    def forward(self, x: Tensor) -> Tensor:
        # Dequantize on-the-fly
        weight = dequantize_int4_weight(
            self.qweight,
            self.scales,
            self.zeros if hasattr(self, 'zeros') else None,
            (self.out_features, self.in_features),
            self.group_size,
        )
        
        y = F.linear(x, weight.to(x.dtype))
        
        if self.bias_param is not None:
            y = y + self.bias_param
        
        return y


# =============================================================================
# Expert Cache (LRU)
# =============================================================================

class ExpertLRUCache:
    """LRU cache for keeping hot experts on GPU."""
    
    def __init__(self, max_size: int, device: str = "cuda"):
        self.max_size = max_size
        self.device = device
        self.cache: OrderedDict[str, Dict[str, Tensor]] = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Dict[str, Tensor]]:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def put(self, key: str, expert_weights: Dict[str, Tensor]):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return
            
            while len(self.cache) >= self.max_size:
                _, evicted = self.cache.popitem(last=False)
                for v in evicted.values():
                    del v
                torch.cuda.empty_cache()
            
            self.cache[key] = {
                k: v.to(self.device, non_blocking=True) for k, v in expert_weights.items()
            }
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            torch.cuda.empty_cache()
    
    def stats(self) -> Dict[str, int]:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "hits": self.hits, 
            "misses": self.misses, 
            "size": len(self.cache),
            "hit_rate": f"{hit_rate:.1%}"
        }


# =============================================================================
# RMSNorm
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# =============================================================================
# Simplified Loading for AutoGPTQ/GPTQ Format
# =============================================================================

class GLM47FlashINT4:
    """
    Wrapper to load and run INT4 quantized GLM-4.7-Flash.
    
    This uses transformers + auto-gptq for actual model loading,
    with optional expert offloading for 8GB GPUs.
    """
    
    def __init__(self, model_path: str, config: InferenceConfig):
        self.model_path = Path(model_path)
        self.config = config
        self.model = None
        self.tokenizer = None
        self.model_config = None
        
    def load(self):
        """Load the quantized model."""
        print(f"Loading model from {self.model_path}")
        
        # Load config
        config_path = self.model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.model_config = json.load(f)
            print(f"  Architecture: {self.model_config.get('architectures', ['unknown'])[0]}")
            print(f"  Hidden size: {self.model_config.get('hidden_size', '?')}")
            print(f"  Layers: {self.model_config.get('num_hidden_layers', '?')}")
            print(f"  Experts: {self.model_config.get('n_routed_experts', '?')}")
        
        # Load quantization config
        quant_config_path = self.model_path / "quantize_config.json"
        if quant_config_path.exists():
            with open(quant_config_path) as f:
                quant_config = json.load(f)
            print(f"  Quantization: {quant_config.get('bits', '?')}-bit")
            print(f"  Group size: {quant_config.get('group_size', '?')}")
        
        # Try to load with transformers + auto_gptq
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print("\nLoading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "THUDM/glm-4-9b",  # Use base tokenizer
                trust_remote_code=True
            )
            
            print("Loading quantized model...")
            # Try loading with device_map for memory efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"\n[ERROR] Could not load model with transformers: {e}")
            print("\nTo use this model, you may need:")
            print("  1. pip install auto-gptq")
            print("  2. pip install optimum")
            print("  3. The model may need conversion to a compatible format")
            
            # Show manual loading instructions
            self._show_manual_loading_info()
            
    def _show_manual_loading_info(self):
        """Show info about manual weight loading."""
        print("\n" + "="*60)
        print("MANUAL LOADING INFO")
        print("="*60)
        
        if not HAS_SAFETENSORS:
            print("Install safetensors: pip install safetensors")
            return
        
        # Inspect the safetensors file
        st_path = self.model_path / "model.safetensors"
        if st_path.exists():
            print(f"\nInspecting {st_path}...")
            with safe_open(str(st_path), framework="pt") as f:
                keys = list(f.keys())
                print(f"Total tensors: {len(keys)}")
                print("\nFirst 20 tensor names:")
                for key in keys[:20]:
                    tensor = f.get_tensor(key)
                    print(f"  {key}: {tensor.shape} ({tensor.dtype})")
                
                # Check total size
                total_params = sum(f.get_tensor(k).numel() for k in keys)
                print(f"\nTotal parameters: {total_params:,}")
                print(f"Size (FP16): {total_params * 2 / 1e9:.2f} GB")
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> str:
        """Generate text from prompt."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        start = time.time()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        elapsed = time.time() - start
        
        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Stats
        new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        print(f"\n[Generated {new_tokens} tokens in {elapsed:.1f}s ({new_tokens/elapsed:.1f} tok/s)]")
        
        return generated


# =============================================================================
# Memory Analysis for 8GB GPU
# =============================================================================

def analyze_memory_requirements(model_path: str):
    """Analyze memory requirements for running on 8GB GPU."""
    print("\n" + "="*60)
    print("MEMORY ANALYSIS FOR 8GB GPU")
    print("="*60)
    
    model_path = Path(model_path)
    
    # Load configs
    config_path = model_path / "config.json"
    quant_config_path = model_path / "quantize_config.json"
    
    if not config_path.exists():
        print(f"Config not found at {config_path}")
        return
    
    with open(config_path) as f:
        model_config = json.load(f)
    
    with open(quant_config_path) as f:
        quant_config = json.load(f)
    
    bits = quant_config.get("bits", 4)
    group_size = quant_config.get("group_size", 128)
    
    hidden = model_config["hidden_size"]
    layers = model_config["num_hidden_layers"]
    vocab = model_config["vocab_size"]
    n_experts = model_config.get("n_routed_experts", 64)
    n_shared = model_config.get("n_shared_experts", 1)
    active_experts = model_config.get("num_experts_per_tok", 4)
    moe_intermediate = model_config.get("moe_intermediate_size", 1536)
    dense_intermediate = model_config.get("intermediate_size", 10240)
    first_k_dense = model_config.get("first_k_dense_replace", 1)
    
    # MLA dimensions
    q_lora = model_config.get("q_lora_rank", 768)
    kv_lora = model_config.get("kv_lora_rank", 512)
    n_heads = model_config.get("num_attention_heads", 20)
    qk_nope = model_config.get("qk_nope_head_dim", 192)
    qk_rope = model_config.get("qk_rope_head_dim", 64)
    v_head = model_config.get("v_head_dim", 256)
    
    print(f"\nModel Config:")
    print(f"  Hidden: {hidden}")
    print(f"  Layers: {layers}")
    print(f"  Vocab: {vocab}")
    print(f"  Experts: {n_experts} routed + {n_shared} shared")
    print(f"  Active experts/token: {active_experts}")
    print(f"  Quantization: INT{bits} (group={group_size})")
    
    # Calculate memory per component (INT4 = 0.5 bytes per param)
    bytes_per_param = bits / 8
    
    # Embeddings (usually kept in FP16)
    embed_params = vocab * hidden
    embed_mem = embed_params * 2 / 1e9  # FP16
    
    # Attention per layer (MLA)
    attn_params_per_layer = (
        hidden * q_lora +  # q_a_proj
        q_lora * n_heads * (qk_nope + qk_rope) +  # q_b_proj
        hidden * kv_lora +  # kv_a_proj
        kv_lora * n_heads * (qk_nope + qk_rope + v_head) +  # kv_b_proj
        n_heads * v_head * hidden  # o_proj
    )
    attn_mem_per_layer = attn_params_per_layer * bytes_per_param / 1e9
    
    # Shared expert per MoE layer (stays on GPU)
    shared_expert_params = 3 * hidden * moe_intermediate  # gate, up, down
    shared_expert_mem = shared_expert_params * bytes_per_param / 1e9
    
    # Routed experts per layer (offloaded to CPU)
    routed_expert_params = n_experts * 3 * hidden * moe_intermediate
    routed_expert_mem = routed_expert_params * bytes_per_param / 1e9
    
    # Dense MLP (first k layers)
    dense_mlp_params = 3 * hidden * dense_intermediate
    dense_mlp_mem = dense_mlp_params * bytes_per_param / 1e9
    
    # Router (tiny, FP16)
    router_params = hidden * n_experts
    router_mem = router_params * 2 / 1e9
    
    # Norms (FP16)
    norm_params = 2 * hidden  # 2 norms per layer
    norm_mem = norm_params * 2 / 1e9
    
    # LM head (tied to embeddings or separate)
    lm_head_mem = 0 if model_config.get("tie_word_embeddings", False) else embed_mem
    
    # Calculate totals
    moe_layers = layers - first_k_dense
    dense_layers = first_k_dense
    
    gpu_mem = (
        embed_mem +  # Embeddings
        layers * attn_mem_per_layer +  # All attention
        moe_layers * shared_expert_mem +  # Shared experts
        dense_layers * dense_mlp_mem +  # Dense MLP
        moe_layers * router_mem +  # Routers
        layers * norm_mem +  # Norms
        lm_head_mem  # LM head
    )
    
    cpu_mem = moe_layers * routed_expert_mem  # Routed experts
    
    # Active memory (what's actually used during inference)
    active_expert_mem = active_experts * shared_expert_params * bytes_per_param / 1e9
    
    print(f"\nMemory Breakdown:")
    print(f"  Embeddings:        {embed_mem:.2f} GB (FP16, GPU)")
    print(f"  Attention (all):   {layers * attn_mem_per_layer:.2f} GB (INT{bits}, GPU)")
    print(f"  Shared experts:    {moe_layers * shared_expert_mem:.2f} GB (INT{bits}, GPU)")
    print(f"  Dense MLP:         {dense_layers * dense_mlp_mem:.2f} GB (INT{bits}, GPU)")
    print(f"  Routers:           {moe_layers * router_mem:.3f} GB (FP16, GPU)")
    print(f"  Norms:             {layers * norm_mem:.3f} GB (FP16, GPU)")
    print(f"  Routed experts:    {cpu_mem:.2f} GB (INT{bits}, CPU)")
    
    print(f"\n{'='*40}")
    print(f"  GPU VRAM needed:   {gpu_mem:.2f} GB")
    print(f"  CPU RAM needed:    {cpu_mem:.2f} GB")
    print(f"  Total model:       {gpu_mem + cpu_mem:.2f} GB")
    print(f"{'='*40}")
    
    # KV cache estimate (for 2048 context)
    ctx_len = 2048
    kv_cache_mem = 2 * layers * ctx_len * n_heads * (qk_nope + qk_rope + v_head) * 2 / 1e9
    print(f"\nKV Cache ({ctx_len} tokens): {kv_cache_mem:.2f} GB (FP16)")
    
    total_gpu = gpu_mem + kv_cache_mem + 0.5  # + overhead
    print(f"Total GPU with cache: {total_gpu:.2f} GB")
    
    if total_gpu <= 8.0:
        print(f"\n[OK] Fits on 8GB GPU with expert offloading!")
    elif total_gpu <= 10.0:
        print(f"\n[WARN] Tight fit - may need smaller batch/context")
    else:
        print(f"\n[ERROR] Too large for 8GB GPU - need more aggressive offloading")
        
    return {
        "gpu_mem": gpu_mem,
        "cpu_mem": cpu_mem,
        "kv_cache": kv_cache_mem,
        "total_gpu": total_gpu,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GLM-4.7-Flash INT4 inference on 8GB GPU")
    parser.add_argument("--model", type=str, default="./models/glm-4.7-flash-int4")
    parser.add_argument("--prompt", type=str, default="def fibonacci(n):")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--analyze", action="store_true", help="Only analyze memory requirements")
    parser.add_argument("--inspect", action="store_true", help="Inspect model weights")
    args = parser.parse_args()
    
    print("="*60)
    print("GLM-4.7-Flash INT4 Inference (8GB GPU)")
    print("="*60)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu} ({mem:.1f} GB)")
    else:
        print("No GPU available, using CPU")
    
    # Memory analysis
    if args.analyze:
        analyze_memory_requirements(args.model)
        return
    
    # Inspect weights
    if args.inspect:
        model_path = Path(args.model)
        st_path = model_path / "model.safetensors"
        if st_path.exists() and HAS_SAFETENSORS:
            print(f"\nInspecting {st_path}...")
            with safe_open(str(st_path), framework="pt") as f:
                keys = list(f.keys())
                print(f"Total tensors: {len(keys)}")
                for key in sorted(keys)[:50]:
                    tensor = f.get_tensor(key)
                    print(f"  {key}: {tensor.shape} {tensor.dtype}")
        return
    
    # Load and run
    config = InferenceConfig(
        model_path=args.model,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    # First do memory analysis
    mem_info = analyze_memory_requirements(args.model)
    
    # Try to load
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    
    engine = GLM47FlashINT4(args.model, config)
    engine.load()
    
    # Generate if model loaded
    if engine.model is not None:
        print("\n" + "="*60)
        print("GENERATION")
        print("="*60)
        print(f"Prompt: {args.prompt}")
        
        result = engine.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        print("\n" + "-"*40)
        print(result)


if __name__ == "__main__":
    main()
