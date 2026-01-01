"""
Continue training Distillix from checkpoint.
Run with: nohup python3 scripts/continue_train.py > train.log 2>&1 &
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

# Clear any cached memory
gc.collect()
torch.cuda.empty_cache()

print("="*60)
print("DISTILLIX CONTINUED TRAINING")
print("="*60)

# Config
config = get_config_125m()
device = torch.device('cuda')

# Find latest checkpoint
import glob
checkpoints = sorted(glob.glob('artifacts/model_*steps.pt'))
if checkpoints:
    latest = checkpoints[-1]
    # Extract step number
    import re
    match = re.search(r'model_(\d+)steps', latest)
    start_step = int(match.group(1)) if match else 0
else:
    latest = None
    start_step = 0

print(f"\nLoading from: {latest or 'scratch'}")
print(f"Starting at step: {start_step}")

# Load model
model = StudentLLM(config.model).to(device)
if latest:
    checkpoint = torch.load(latest, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params/1e6:.1f}M params")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Dataset
class TextDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=256):
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

# Load data
print("\nLoading training data...")
dataset = TextDataset('data/distillation/train_all.jsonl', tokenizer, max_len=256)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True)

# Optimizer
print("\nSetting up Muon optimizer...")
optimizer = MuonAdamW(
    model.named_parameters(),
    muon_lr=config.training.muon_lr,
    adamw_lr=config.training.adamw_lr,
    weight_decay=config.training.weight_decay,
)

# Training settings
target_step = 10000
total_steps = target_step - start_step
log_interval = 100
save_interval = 1000
grad_accum = 4

print(f"\nTraining config:")
print(f"  Target step: {target_step}")
print(f"  Steps to run: {total_steps}")
print(f"  Batch size: 4 x {grad_accum} = 16")
print(f"  Seq length: 256")
print(f"  Data samples: {len(dataset)}")

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
        global_step = start_step + step
        
        if step % log_interval == 0:
            avg_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            eta = (total_steps - step) / steps_per_sec / 60
            mem = torch.cuda.max_memory_allocated() / 1024**3
            
            print(f"Step {global_step:5d}/{target_step} | Loss: {avg_loss:.4f} | "
                  f"{steps_per_sec:.1f} steps/s | ETA: {eta:.1f}m | Mem: {mem:.2f}GB")
            total_loss = 0
        
        if step % save_interval == 0:
            ckpt_path = f'artifacts/model_{global_step}steps.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'step': global_step,
                'loss': avg_loss,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

# Final save
elapsed = time.time() - start_time
print(f"\nTraining complete in {elapsed/60:.1f} minutes")

final_path = f'artifacts/model_{target_step}steps.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'step': target_step,
}, final_path)
print(f"Saved final model: {final_path}")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
