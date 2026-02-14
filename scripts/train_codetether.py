#!/usr/bin/env python3
"""
Distillix CodeTether Fine-tuning - Train on Real Tool Usage Data

THE SNOWBALL EFFECT:
- Base model learned Python syntax and reasoning
- This script adds REAL tool usage from 1,500+ CodeTether sessions

Usage:
    python scripts/train_codetether.py --base artifacts/distillix-v04-final.pt
    python scripts/train_codetether.py --base artifacts/distillix-grok-45k.pt
    
The model will learn:
- How to debug code
- When and how to use tools (bash, read, write, edit, etc.)
- Multi-turn conversation patterns
- Reason before acting
"""
import os
import sys
import json
import torch
import time
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Make imports work from any location (Colab, local checkout, etc.)
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from smelter.config import ModelConfig, Config
from smelter.model import StudentLLM
from smelter.muon import MuonAdamW
from smelter.spin_solver import create_polarization_hook


def load_codetether_data(data_path: str) -> list:
    """Load training data from CodeTether JSONL export or unified format"""
    samples = []
    
    with open(data_path) as f:
        for line in f:
            try:
                item = json.loads(line)
                
                # Handle unified format (prompt/response)
                if 'prompt' in item and 'response' in item:
                    prompt = item['prompt']
                    response = item['response']
                    text = f"<|user|>\n{prompt}\n<|assistant|>\n{response}"
                    samples.append(text)
                    continue
                
                # Handle messages format
                messages = item.get('messages', [])
                
                # Convert to conversation format
                text_parts = []
                for msg in messages:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    
                    if role == 'system':
                        text_parts.append(f"<|system|>\n{content}")
                    elif role == 'user':
                        text_parts.append(f"<|user|>\n{content}")
                    elif role == 'assistant':
                        # Include tool calls if present
                        tool_calls = msg.get('tool_calls', [])
                        if tool_calls:
                            tools_str = "\n".join([
                                f"<tool>{tc['function']['name']}</tool>\n<params>{tc['function']['arguments']}</params>"
                                for tc in tool_calls
                            ])
                            content = f"{content}\n{tools_str}" if content else tools_str
                        text_parts.append(f"<|assistant|>\n{content}")
                
                if text_parts:
                    samples.append("\n".join(text_parts))
                    
            except Exception as e:
                continue
    
    return samples


class ConversationDataset(Dataset):
    """Dataset for conversation training"""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        ids = enc['input_ids'].squeeze(0)
        return {'input_ids': ids, 'labels': ids.clone()}


def main():
    parser = argparse.ArgumentParser(description="Train Distillix on CodeTether data")
    parser.add_argument("--base", type=str, default="artifacts/distillix-v04-final.pt",
                        help="Base checkpoint to continue from")
    parser.add_argument("--data", type=str, default="data/training/combined_full.jsonl",
                        help="Training data path")
    parser.add_argument("--output", type=str, default="distillix-codetether",
                        help="Output prefix")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Max training steps")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--muon-lr", type=float, default=0.005,
                        help="Muon learning rate")
    parser.add_argument("--adamw-lr", type=float, default=0.0001,
                        help="AdamW learning rate")

    # BitNet stability: avoid weight decay collapse by default.
    parser.add_argument("--muon-weight-decay", type=float, default=0.0,
                        help="Weight decay for Muon (2D params). Default 0.0 for BitNet stability")
    parser.add_argument("--adamw-weight-decay", type=float, default=0.0,
                        help="Weight decay for AdamW (1D params). Default 0.0 for BitNet stability")
    parser.add_argument("--polarize", action="store_true",
                        help="Apply BitNet anti-collapse polarization post-step")
    parser.add_argument("--target-scale", type=float, default=0.01,
                        help="Polarization target magnitude")
    parser.add_argument("--polarization-strength", type=float, default=0.1,
                        help="Polarization strength")
    args = parser.parse_args()
    
    # ========================================
    # Setup
    # ========================================
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print("=" * 60)
    print("DISTILLIX CODETETHER FINE-TUNING")
    print("=" * 60)
    print()
    print(">>> THE SNOWBALL EFFECT <<<")
    print("Loading existing brain and teaching it tool usage...")
    print()
    
    # Create model
    config = Config(model=ModelConfig(use_bitlinear=True, use_bitlinear_ffn=False))
    model = StudentLLM(config.model).cuda()
    
    # ========================================
    # Load checkpoint (not random init!)
    # ========================================
    
    OUTPUT_DIR = 'artifacts'
    resume_path = f'{OUTPUT_DIR}/{args.output}-latest.pt'
    
    if os.path.exists(resume_path):
        print(f"Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location='cuda', weights_only=False)
        start_step = ckpt.get('step', 0)
    elif os.path.exists(args.base):
        print(f"Loading base checkpoint: {args.base}")
        ckpt = torch.load(args.base, map_location='cuda', weights_only=False)
        start_step = 0
    else:
        raise FileNotFoundError(f"No checkpoint found! Need {args.base}")
    
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    
    model.enable_gradient_checkpointing()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count:,}")
    
    # ========================================
    # Load tokenizer and data
    # ========================================
    
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading CodeTether data from: {args.data}")
    samples = load_codetether_data(args.data)
    print(f"Loaded {len(samples)} training samples")
    
    if len(samples) == 0:
        raise ValueError("No training samples loaded!")
    
    # Show sample
    print(f"\nSample (first 300 chars):")
    print("-" * 40)
    print(samples[0][:300])
    print("-" * 40)
    
    # ========================================
    # Create dataloader
    # ========================================
    
    dataset = ConversationDataset(samples, tokenizer, max_length=768)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    
    # ========================================
    # Optimizer (Muon + AdamW)
    # ========================================
    
    optimizer = MuonAdamW(
        model.named_parameters(),
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        weight_decay=args.muon_weight_decay,
        adamw_weight_decay=args.adamw_weight_decay,
    )

    polarization_hook = None
    if args.polarize:
        polarization_hook = create_polarization_hook(
            model,
            target_scale=args.target_scale,
            polarization_strength=args.polarization_strength,
        )
    
    print(f"\nOptimizer: Muon LR={args.muon_lr}, AdamW LR={args.adamw_lr}")
    print(f"Effective batch: {args.batch * args.accum}")
    
    import sys
    sys.stdout.flush()
    
    # ========================================
    # Training loop (FP32 for BitNet stability)
    # ========================================
    
    model.train()
    step = start_step
    running_loss = 0.0
    t0 = time.time()
    
    LOG_EVERY = 50
    SAVE_EVERY = 1000
    
    print(f"\nTraining for {args.steps} steps (from step {start_step})...")
    print()
    
    while step < args.steps:
        for batch in loader:
            ids = batch['input_ids'].cuda()
            labels = batch['labels'].cuda()
            
            # Forward (FP32)
            loss = model(ids, labels=labels)['loss'] / args.accum
            loss.backward()
            
            running_loss += loss.item()
            
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if polarization_hook is not None:
                    polarization_hook()
                optimizer.zero_grad()
            
            step += 1
            
            if step % LOG_EVERY == 0:
                avg_loss = running_loss / LOG_EVERY
                elapsed = time.time() - t0
                steps_per_sec = LOG_EVERY / elapsed
                eta_min = (args.steps - step) / steps_per_sec / 60
                
                print(f"Step {step:>5}/{args.steps} | Loss: {avg_loss:.4f} | {steps_per_sec:.2f} steps/s | ETA: {eta_min:.1f}m", flush=True)
                
                running_loss = 0.0
                t0 = time.time()
            
            if step % SAVE_EVERY == 0:
                ckpt_path = f'{OUTPUT_DIR}/{args.output}-{step}steps.pt'
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                }, ckpt_path)
                print(f">>> Saved: {ckpt_path}")
                
                latest_path = f'{OUTPUT_DIR}/{args.output}-latest.pt'
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                }, latest_path)
            
            if step >= args.steps:
                break
    
    # Save final
    final_path = f'{OUTPUT_DIR}/{args.output}-final.pt'
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'config': {
            'base': args.base,
            'data': args.data,
            'samples': len(samples),
        }
    }, final_path)
    
    print()
    print("=" * 60)
    print(f"TRAINING COMPLETE!")
    print(f"Final: {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
