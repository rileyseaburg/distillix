"""
Distillix v0.5 Training - Cognitive Kernel Fine-tuning
- Start from v0.4 checkpoint (base coder)
- Train on Cognitive Kernel reasoning data
- Target: 2000 steps (quick fine-tune)
- Result: "Engineer" model with System 2 thinking
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import time
import sys
import gc

sys.path.insert(0, '/root/distillix')

from smelter.config import get_config_125m
from smelter.model import StudentLLM
from smelter.muon import MuonAdamW
from transformers import AutoTokenizer

gc.collect()
torch.cuda.empty_cache()

print("="*60)
print("DISTILLIX v0.5 - COGNITIVE KERNEL FINE-TUNING")
print("="*60)

config = get_config_125m()
device = torch.device('cuda')

# Load v0.4 as base
print("\nLoading v0.4 checkpoint as base...")
model = StudentLLM(config.model).to(device)
checkpoint = torch.load('artifacts/distillix-v04-final.pt', map_location=device, weights_only=False)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
print("Loaded v0.4 weights")

model.enable_gradient_checkpointing()
print("Gradient checkpointing: ENABLED")

total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params/1e6:.1f}M params")

# Tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Dataset
class CognitiveKernelDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=1024):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Format: prompt + response with reasoning
                text = f"Problem: {data['prompt']}\n\nSolution:\n{data['response']}"
                self.samples.append(text)
        
        print(f"Loaded {len(self.samples)} Cognitive Kernel samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        return tokens['input_ids'].squeeze(0)

# Load Cognitive Kernel data
print("\nLoading Cognitive Kernel training data...")
dataset = CognitiveKernelDataset('data/distillation/cognitive_kernel_v2.jsonl', tokenizer, max_len=1024)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)

# Lower learning rate for fine-tuning
print("\nSetting up optimizer (lower LR for fine-tuning)...")
optimizer = MuonAdamW(
    model.named_parameters(),
    muon_lr=0.005,      # 4x lower than v0.4
    adamw_lr=0.0001,    # 3x lower than v0.4
    weight_decay=config.training.weight_decay,
)

# Training settings - short fine-tune
total_steps = 2000
log_interval = 50
save_interval = 500
grad_accum = 4

print(f"\nTraining config:")
print(f"  Total steps: {total_steps}")
print(f"  Batch size: 2 x {grad_accum} = {2 * grad_accum}")
print(f"  Seq length: 1024")
print(f"  Data samples: {len(dataset)}")
print(f"  Muon LR: 0.005 (fine-tune)")
print(f"  AdamW LR: 0.0001 (fine-tune)")

print("\n" + "="*60)
print("FINE-TUNING ON COGNITIVE KERNEL DATA")
print("="*60)

model.train()
scaler = torch.amp.GradScaler('cuda')

step = 0
epoch = 0
total_loss = 0
start_time = time.time()
avg_loss = 10.0

while step < total_steps:
    epoch += 1
    for batch in dataloader:
        if step >= total_steps:
            break
        
        input_ids = batch.to(device)
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
            output = model(input_ids, labels=input_ids)
            loss = output['loss'] / grad_accum
        
        scaler.scale(loss).backward()
        
        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum
        step += 1
        
        if step % log_interval == 0:
            avg_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            eta = (total_steps - step) / steps_per_sec / 60
            mem = torch.cuda.max_memory_allocated() / 1024**3
            
            print(f"Step {step:4d}/{total_steps} | Loss: {avg_loss:.4f} | "
                  f"{steps_per_sec:.1f} steps/s | ETA: {eta:.1f}m | Mem: {mem:.2f}GB")
            total_loss = 0
        
        if step % save_interval == 0:
            ckpt_path = f'artifacts/distillix-v05-{step}steps.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'step': step,
                'loss': avg_loss,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

elapsed = time.time() - start_time
print(f"\nFine-tuning complete in {elapsed/60:.1f} minutes")

final_path = 'artifacts/distillix-v05-cognitive.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'step': step,
}, final_path)
print(f"Saved final model: {final_path}")

print("\n" + "="*60)
print("v0.5 COGNITIVE KERNEL TRAINING COMPLETE")
print("="*60)
