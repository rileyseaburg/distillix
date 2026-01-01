"""
Distillix v0.4 Training - Real Dataset Run
- 10k+ samples from SWE-rebench trajectories
- 1024 token sequences (4x longer)
- Gradient checkpointing enabled
- Target: 20k steps
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
print("DISTILLIX v0.4 TRAINING - REAL DATASET")
print("="*60)

# Config
config = get_config_125m()
device = torch.device('cuda')

# Create fresh model
print("\nInitializing fresh model...")
model = StudentLLM(config.model).to(device)

# ENABLE GRADIENT CHECKPOINTING - Key for longer sequences!
model.enable_gradient_checkpointing()
print("Gradient checkpointing: ENABLED")

total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params/1e6:.1f}M params")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Dataset with LONGER sequences
class TextDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=1024):  # 4x longer!
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'prompt' in data and 'response' in data:
                    text = f"{data['prompt']}\n\n{data['response']}"
                    self.samples.append(text)
                elif 'response' in data:
                    self.samples.append(data['response'])
        
        print(f"Loaded {len(self.samples)} samples")
    
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

# Load FULL 10k dataset
print("\nLoading training data...")
dataset = TextDataset('data/distillation/train_full_10k.jsonl', tokenizer, max_len=1024)

# Smaller batch due to longer sequences, but grad accumulation compensates
# With checkpointing: 1024 tokens @ batch 2 x accum 8 = effective 16
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)

# Optimizer
print("\nSetting up Muon optimizer...")
optimizer = MuonAdamW(
    model.named_parameters(),
    muon_lr=config.training.muon_lr,
    adamw_lr=config.training.adamw_lr,
    weight_decay=config.training.weight_decay,
)

# Training settings
total_steps = 20000
log_interval = 100
save_interval = 2000
grad_accum = 8  # 2 * 8 = 16 effective batch

print(f"\nTraining config:")
print(f"  Total steps: {total_steps}")
print(f"  Batch size: 2 x {grad_accum} = {2 * grad_accum}")
print(f"  Seq length: 1024 (4x previous)")
print(f"  Data samples: {len(dataset)}")
print(f"  Gradient checkpointing: ON")
print(f"  Muon LR: {config.training.muon_lr}")
print(f"  AdamW LR: {config.training.adamw_lr}")

# Training loop
print("\n" + "="*60)
print("TRAINING")
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
            
            print(f"Step {step:5d}/{total_steps} | Loss: {avg_loss:.4f} | "
                  f"{steps_per_sec:.1f} steps/s | ETA: {eta:.1f}m | Mem: {mem:.2f}GB")
            total_loss = 0
        
        if step % save_interval == 0:
            ckpt_path = f'artifacts/distillix-v04-{step}steps.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'step': step,
                'loss': avg_loss,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

# Final save
elapsed = time.time() - start_time
print(f"\nTraining complete in {elapsed/60:.1f} minutes")

final_path = f'artifacts/distillix-v04-final.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'step': step,
}, final_path)
print(f"Saved final model: {final_path}")

print("\n" + "="*60)
print("v0.4 TRAINING COMPLETE")
print("="*60)
