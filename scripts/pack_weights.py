#!/usr/bin/env python3
"""
BitNet Weight Packer

Packs ternary weights {-1, 0, +1} into 2-bit format.
Achieves ~4x compression: 200MB -> 50MB

Usage: python scripts/pack_weights.py artifacts/model.safetensors
"""
import torch
from safetensors.torch import load_file, save_file
import sys
import os

def pack_ternary(tensor):
    """
    Packs a ternary tensor {-1, 0, 1} into uint8 (4 weights per byte).
    
    Mapping:
      -1 -> 00 (0)
       0 -> 01 (1)
      +1 -> 10 (2)
    """
    # 1. Quantize to ternary {-1, 0, 1}
    weights = torch.sign(tensor).to(torch.int8)
    
    # 2. Map to 2-bit integers: -1->0, 0->1, 1->2
    weights = weights + 1
    
    # 3. Flatten and pad to multiple of 4
    flat = weights.flatten()
    padding = (4 - (flat.numel() % 4)) % 4
    if padding > 0:
        flat = torch.cat([flat, torch.ones(padding, dtype=torch.int8)])  # Pad with 0s (mapped to 1)
    
    # 4. Reshape into groups of 4
    groups = flat.view(-1, 4)
    
    # 5. Pack 4 weights into 1 byte: (w0 << 6) | (w1 << 4) | (w2 << 2) | w3
    packed = (groups[:, 0].to(torch.int32) << 6) | \
             (groups[:, 1].to(torch.int32) << 4) | \
             (groups[:, 2].to(torch.int32) << 2) | \
              groups[:, 3].to(torch.int32)
    
    return packed.to(torch.uint8)


def unpack_ternary(packed, original_shape):
    """
    Unpacks uint8 back to ternary tensor.
    """
    # Unpack 4 weights from each byte
    w0 = (packed >> 6) & 0x03
    w1 = (packed >> 4) & 0x03
    w2 = (packed >> 2) & 0x03
    w3 = packed & 0x03
    
    # Interleave
    unpacked = torch.stack([w0, w1, w2, w3], dim=1).flatten()
    
    # Map back: 0->-1, 1->0, 2->+1
    unpacked = unpacked.to(torch.int8) - 1
    
    # Reshape to original
    numel = original_shape.numel() if hasattr(original_shape, 'numel') else torch.prod(torch.tensor(original_shape)).item()
    return unpacked[:numel].view(original_shape).to(torch.float16)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/pack_weights.py <model.safetensors|model.pt>")
        sys.exit(1)

    input_path = sys.argv[1]
    filename = os.path.basename(input_path).replace(".safetensors", "").replace(".pt", "")
    output_path = f"artifacts/{filename}_packed.safetensors"

    print("="*50)
    print("BITNET WEIGHT PACKER")
    print("="*50)
    print(f"\nLoading {input_path}...")
    
    # Load model
    if input_path.endswith('.safetensors'):
        state_dict = load_file(input_path)
    else:
        checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    packed_dict = {}
    metadata = {}
    
    total_original = 0
    total_packed = 0
    layers_packed = 0

    print("\nPacking weights...")
    for key, tensor in state_dict.items():
        orig_bytes = tensor.numel() * tensor.element_size()
        total_original += orig_bytes
        
        # Pack 2D weight matrices from attention/MLP (BitLinear layers)
        is_bitlinear = any(x in key for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                                               'gate_proj', 'up_proj', 'down_proj'])
        
        if is_bitlinear and 'weight' in key and tensor.dim() == 2:
            packed = pack_ternary(tensor)
            packed_dict[key] = packed
            
            # Store original shape for unpacking
            metadata[f"{key}_shape"] = list(tensor.shape)
            
            pack_bytes = packed.numel()
            total_packed += pack_bytes
            layers_packed += 1
            
            ratio = orig_bytes / pack_bytes
            print(f"  {key}: {orig_bytes/1024:.1f}KB -> {pack_bytes/1024:.1f}KB ({ratio:.1f}x)")
        else:
            # Keep non-BitLinear weights as-is (embeddings, norms, etc.)
            packed_dict[key] = tensor
            total_packed += orig_bytes

    # Save packed model
    print(f"\nSaving to {output_path}...")
    
    # Store shapes as a tensor for safetensors compatibility
    import json
    shapes_json = json.dumps(metadata)
    packed_dict['__metadata__'] = torch.tensor([ord(c) for c in shapes_json], dtype=torch.uint8)
    
    save_file(packed_dict, output_path)
    
    # Summary
    print("\n" + "="*50)
    print("PACKING COMPLETE")
    print("="*50)
    print(f"Layers packed:     {layers_packed}")
    print(f"Original size:     {total_original / 1024**2:.2f} MB")
    print(f"Packed size:       {total_packed / 1024**2:.2f} MB")
    print(f"Compression:       {total_original / total_packed:.2f}x")
    print(f"Output:            {output_path}")
    print("="*50)


if __name__ == "__main__":
    main()
