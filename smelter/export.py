"""
Model Export Utilities for Distillix.

Supports exporting trained models to various formats:
- SafeTensors: Fast, secure serialization (default)
- GGUF: llama.cpp format for CPU/edge inference
- HuggingFace: Compatible with transformers library

GGUF Export:
  BitNet models require special handling since GGUF doesn't natively
  support 1.58-bit quantization. We export as FP16 and let llama.cpp
  re-quantize, or use the bitnet.cpp project for native 1.58-bit.

Usage:
    python -m smelter.export --checkpoint artifacts/model.pt --format gguf

Copyright (c) 2025 Distillix. All Rights Reserved.
"""

import argparse
import json
import struct
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import sys

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("PyTorch required. Install with: pip install torch")
    sys.exit(1)

try:
    from safetensors.torch import save_file, load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

from .config import ModelConfig, get_config_125m
from .model import StudentLLM


# =============================================================================
# GGUF Constants and Structures
# =============================================================================

# GGUF Magic number
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian

# GGUF Versions
GGUF_VERSION = 3

# GGUF Data Types
class GGUFType:
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12

# GGML Tensor Types
class GGMLType:
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    BF16 = 29


@dataclass
class GGUFMetadata:
    """GGUF file metadata."""
    architecture: str = "llama"
    name: str = "distillix"
    version: str = "0.3.0"
    
    # Model architecture
    context_length: int = 2048
    embedding_length: int = 768
    block_count: int = 12
    
    # Attention
    attention_head_count: int = 12
    attention_head_count_kv: int = 4
    attention_layer_norm_rms_epsilon: float = 1e-6
    
    # RoPE
    rope_dimension_count: int = 64
    rope_freq_base: float = 1000000.0
    
    # Vocabulary
    vocab_size: int = 32000
    
    # Tokenizer
    tokenizer_model: str = "llama"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to GGUF metadata dictionary."""
        arch = self.architecture
        return {
            "general.architecture": self.architecture,
            "general.name": self.name,
            "general.version": self.version,
            f"{arch}.context_length": self.context_length,
            f"{arch}.embedding_length": self.embedding_length,
            f"{arch}.block_count": self.block_count,
            f"{arch}.attention.head_count": self.attention_head_count,
            f"{arch}.attention.head_count_kv": self.attention_head_count_kv,
            f"{arch}.attention.layer_norm_rms_epsilon": self.attention_layer_norm_rms_epsilon,
            f"{arch}.rope.dimension_count": self.rope_dimension_count,
            f"{arch}.rope.freq_base": self.rope_freq_base,
            f"{arch}.vocab_size": self.vocab_size,
            "tokenizer.ggml.model": self.tokenizer_model,
        }


# =============================================================================
# Export Functions
# =============================================================================

def export_safetensors(
    model: nn.Module,
    output_path: str,
    config: Optional[ModelConfig] = None,
) -> str:
    """
    Export model to SafeTensors format.
    
    Args:
        model: PyTorch model to export
        output_path: Output file path
        config: Model config (saved alongside)
    
    Returns:
        Path to saved file
    """
    if not HAS_SAFETENSORS:
        raise ImportError("safetensors required. Install with: pip install safetensors")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get state dict
    state_dict = model.state_dict()
    
    # Save model
    model_path = output_path.with_suffix('.safetensors')
    save_file(state_dict, str(model_path))
    
    # Save config
    if config:
        config_path = output_path.with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
    
    print(f"Saved SafeTensors to {model_path}")
    return str(model_path)


def export_gguf(
    model: nn.Module,
    output_path: str,
    config: ModelConfig,
    quantization: str = "f16",
) -> str:
    """
    Export model to GGUF format for llama.cpp.
    
    NOTE: This exports as FP16. For true 1.58-bit inference,
    use bitnet.cpp or similar native BitNet runtime.
    
    Args:
        model: PyTorch model to export
        output_path: Output file path
        config: Model configuration
        quantization: Target quantization (f16, q8_0, q4_0)
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not str(output_path).endswith('.gguf'):
        output_path = output_path.with_suffix('.gguf')
    
    print(f"Exporting to GGUF: {output_path}")
    print(f"  Quantization: {quantization}")
    print(f"  Note: BitNet weights exported as FP16, re-quantize with llama.cpp")
    
    # Build metadata
    metadata = GGUFMetadata(
        name="distillix",
        context_length=config.max_seq_len,
        embedding_length=config.hidden_dim,
        block_count=config.num_layers,
        attention_head_count=config.num_heads,
        attention_head_count_kv=config.num_kv_heads,
        attention_layer_norm_rms_epsilon=config.norm_eps,
        rope_dimension_count=config.head_dim,
        rope_freq_base=config.rope_base,
        vocab_size=config.vocab_size,
    )
    
    # Get state dict and convert to FP16
    state_dict = model.state_dict()
    
    # Map our layer names to GGUF/llama names
    tensor_map = _map_tensor_names(state_dict, config)
    
    # Write GGUF file
    with open(output_path, 'wb') as f:
        _write_gguf_header(f, metadata, tensor_map)
        _write_gguf_tensors(f, tensor_map, quantization)
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved GGUF to {output_path} ({file_size:.1f} MB)")
    
    return str(output_path)


def _map_tensor_names(
    state_dict: Dict[str, torch.Tensor],
    config: ModelConfig,
) -> Dict[str, torch.Tensor]:
    """Map Distillix tensor names to GGUF/llama names."""
    
    mapped = {}
    
    for name, tensor in state_dict.items():
        # Convert to FP16 for export
        tensor = tensor.to(torch.float16).contiguous()
        
        # Embeddings
        if name == "embed_tokens.weight":
            mapped["token_embd.weight"] = tensor
            continue
        
        # Final norm
        if name == "norm.weight":
            mapped["output_norm.weight"] = tensor
            continue
        
        # LM head (if not tied)
        if name == "lm_head.weight":
            mapped["output.weight"] = tensor
            continue
        
        # Transformer layers
        if name.startswith("layers."):
            parts = name.split(".")
            layer_idx = parts[1]
            rest = ".".join(parts[2:])
            
            # Attention
            if rest == "self_attn.q_proj.weight":
                mapped[f"blk.{layer_idx}.attn_q.weight"] = tensor
            elif rest == "self_attn.k_proj.weight":
                mapped[f"blk.{layer_idx}.attn_k.weight"] = tensor
            elif rest == "self_attn.v_proj.weight":
                mapped[f"blk.{layer_idx}.attn_v.weight"] = tensor
            elif rest == "self_attn.o_proj.weight":
                mapped[f"blk.{layer_idx}.attn_output.weight"] = tensor
            
            # QK Norm
            elif rest == "self_attn.q_norm.weight":
                mapped[f"blk.{layer_idx}.attn_q_norm.weight"] = tensor
            elif rest == "self_attn.k_norm.weight":
                mapped[f"blk.{layer_idx}.attn_k_norm.weight"] = tensor
            
            # MLP
            elif rest == "mlp.gate_proj.weight":
                mapped[f"blk.{layer_idx}.ffn_gate.weight"] = tensor
            elif rest == "mlp.up_proj.weight":
                mapped[f"blk.{layer_idx}.ffn_up.weight"] = tensor
            elif rest == "mlp.down_proj.weight":
                mapped[f"blk.{layer_idx}.ffn_down.weight"] = tensor
            
            # Norms
            elif rest == "input_layernorm.weight":
                mapped[f"blk.{layer_idx}.attn_norm.weight"] = tensor
            elif rest == "post_attention_layernorm.weight":
                mapped[f"blk.{layer_idx}.ffn_norm.weight"] = tensor
            
            else:
                print(f"  Warning: Unmapped tensor {name}")
        else:
            print(f"  Warning: Unmapped tensor {name}")
    
    return mapped


def _write_gguf_header(
    f,
    metadata: GGUFMetadata,
    tensors: Dict[str, torch.Tensor],
):
    """Write GGUF file header."""
    
    meta_dict = metadata.to_dict()
    
    # Magic
    f.write(struct.pack('<I', GGUF_MAGIC))
    
    # Version
    f.write(struct.pack('<I', GGUF_VERSION))
    
    # Tensor count
    f.write(struct.pack('<Q', len(tensors)))
    
    # Metadata KV count
    f.write(struct.pack('<Q', len(meta_dict)))
    
    # Write metadata
    for key, value in meta_dict.items():
        _write_gguf_string(f, key)
        
        if isinstance(value, str):
            f.write(struct.pack('<I', GGUFType.STRING))
            _write_gguf_string(f, value)
        elif isinstance(value, int):
            f.write(struct.pack('<I', GGUFType.UINT32))
            f.write(struct.pack('<I', value))
        elif isinstance(value, float):
            f.write(struct.pack('<I', GGUFType.FLOAT32))
            f.write(struct.pack('<f', value))
        else:
            raise ValueError(f"Unsupported metadata type: {type(value)}")


def _write_gguf_string(f, s: str):
    """Write a GGUF string."""
    encoded = s.encode('utf-8')
    f.write(struct.pack('<Q', len(encoded)))
    f.write(encoded)


def _write_gguf_tensors(
    f,
    tensors: Dict[str, torch.Tensor],
    quantization: str,
):
    """Write GGUF tensor data."""
    
    # Determine quantization type
    if quantization == "f16":
        ggml_type = GGMLType.F16
    elif quantization == "f32":
        ggml_type = GGMLType.F32
    elif quantization == "q8_0":
        ggml_type = GGMLType.Q8_0
    elif quantization == "q4_0":
        ggml_type = GGMLType.Q4_0
    else:
        raise ValueError(f"Unsupported quantization: {quantization}")
    
    # Calculate offsets
    offset = 0
    tensor_info = []
    
    for name, tensor in tensors.items():
        # Get tensor shape
        shape = list(tensor.shape)
        n_dims = len(shape)
        
        # Calculate size
        n_elements = tensor.numel()
        if quantization == "f16":
            size = n_elements * 2
        elif quantization == "f32":
            size = n_elements * 4
        else:
            # For quantized, approximate
            size = n_elements * 2  # Placeholder
        
        tensor_info.append({
            'name': name,
            'tensor': tensor,
            'n_dims': n_dims,
            'shape': shape,
            'type': ggml_type,
            'offset': offset,
        })
        
        offset += size
    
    # Write tensor info
    for info in tensor_info:
        _write_gguf_string(f, info['name'])
        f.write(struct.pack('<I', info['n_dims']))
        for dim in info['shape']:
            f.write(struct.pack('<Q', dim))
        f.write(struct.pack('<I', info['type']))
        f.write(struct.pack('<Q', info['offset']))
    
    # Align to 32 bytes
    current_pos = f.tell()
    padding = (32 - (current_pos % 32)) % 32
    f.write(b'\x00' * padding)
    
    # Write tensor data
    for info in tensor_info:
        tensor = info['tensor']
        
        if quantization in ["f16", "f32"]:
            # Direct export
            if quantization == "f16":
                data = tensor.to(torch.float16).numpy().tobytes()
            else:
                data = tensor.to(torch.float32).numpy().tobytes()
            f.write(data)
        else:
            # Quantization would go here
            # For now, export as f16 and let llama.cpp quantize
            data = tensor.to(torch.float16).numpy().tobytes()
            f.write(data)


def export_huggingface(
    model: nn.Module,
    output_path: str,
    config: ModelConfig,
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
) -> str:
    """
    Export model to HuggingFace format.
    
    Args:
        model: PyTorch model to export
        output_path: Output directory
        config: Model configuration
        tokenizer_name: Tokenizer to copy
    
    Returns:
        Path to saved directory
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    if HAS_SAFETENSORS:
        save_file(model.state_dict(), output_path / "model.safetensors")
    else:
        torch.save(model.state_dict(), output_path / "pytorch_model.bin")
    
    # Save config
    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_dim,
        "intermediate_size": config.mlp_hidden_dim,
        "num_hidden_layers": config.num_layers,
        "num_attention_heads": config.num_heads,
        "num_key_value_heads": config.num_kv_heads,
        "max_position_embeddings": config.max_seq_len,
        "rms_norm_eps": config.norm_eps,
        "rope_theta": config.rope_base,
        "tie_word_embeddings": config.tie_word_embeddings,
        "torch_dtype": "float16",
    }
    
    with open(output_path / "config.json", 'w') as f:
        json.dump(hf_config, f, indent=2)
    
    # Copy tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.save_pretrained(output_path)
    except Exception as e:
        print(f"Warning: Could not save tokenizer: {e}")
    
    print(f"Saved HuggingFace format to {output_path}")
    return str(output_path)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Export Distillix model to various formats")
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                        help="Path to model checkpoint (.pt or .safetensors)")
    parser.add_argument("--output", "-o", type=str, default="exports/model",
                        help="Output path")
    parser.add_argument("--format", "-f", type=str, default="safetensors",
                        choices=["safetensors", "gguf", "huggingface"],
                        help="Export format")
    parser.add_argument("--quantization", "-q", type=str, default="f16",
                        choices=["f16", "f32", "q8_0", "q4_0"],
                        help="Quantization for GGUF")
    args = parser.parse_args()
    
    print("="*60)
    print("DISTILLIX MODEL EXPORT")
    print("="*60)
    
    # Load config
    config = get_config_125m()
    
    # Create model
    print(f"Creating model...")
    model = StudentLLM(config.model)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint_path = Path(args.checkpoint)
    
    if checkpoint_path.suffix == '.safetensors':
        state_dict = load_file(str(checkpoint_path))
    else:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Export
    print(f"Exporting as {args.format}...")
    
    if args.format == "safetensors":
        export_safetensors(model, args.output, config.model)
    elif args.format == "gguf":
        export_gguf(model, args.output, config.model, args.quantization)
    elif args.format == "huggingface":
        export_huggingface(model, args.output, config.model)
    
    print("="*60)
    print("EXPORT COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
