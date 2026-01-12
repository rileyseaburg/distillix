#!/usr/bin/env python3
"""
RESUME TITAN 3B TRAINING - Google Colab Script

Run this in Google Colab with GPU runtime (T4 or better).

Instructions:
1. Go to https://colab.research.google.com
2. Create new notebook
3. Runtime > Change runtime type > GPU (T4 or A100)
4. Copy and paste this entire script into a cell
5. Run it

This will:
- Download the titan-3b-2500.pt checkpoint
- Resume training for 500 more steps (to reach 3000 total)
- Save the final checkpoint
"""

# Install dependencies
!pip install -q torch transformers datasets huggingface_hub

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download, HfApi
import time
import math
import os

# ============================================================
# CONFIG
# ============================================================
HF_TOKEN = "YOUR_HF_TOKEN_HERE"  # Your token
CHECKPOINT = "titan/titan-3b-2500.pt"
REPO_ID = "rileyseaburg/distillix"
STEPS = 500  # Remaining steps to reach 3000
BATCH_SIZE = 1  # Small for T4 (16GB)
LR = 5e-5

# ============================================================
# MODEL DEFINITION (Simplified for Colab)
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=1000000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len):
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x, position_ids=None):
        seq_len = x.shape[1]
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Attention(nn.Module):
    def __init__(self, hidden_dim=3200, num_heads=32, num_kv_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, x):
        B, L, _ = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        cos, sin = self.rotary_emb(x)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        attn = attn.masked_fill(causal_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)

class MLP(nn.Module):
    def __init__(self, hidden_dim=3200, intermediate_dim=8640):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
    
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim=3200):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_dim)
        self.self_attn = Attention(hidden_dim)
        self.post_attention_layernorm = RMSNorm(hidden_dim)
        self.mlp = MLP(hidden_dim)
    
    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class StudentLLM(nn.Module):
    def __init__(self, vocab_size=32000, hidden_dim=3200, num_layers=26):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([TransformerBlock(hidden_dim) for _ in range(num_layers)])
        self.norm = RMSNorm(hidden_dim)
        self.vocab_size = vocab_size
    
    def forward(self, input_ids, labels=None):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = F.linear(x, self.embed_tokens.weight)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1), ignore_index=-100)
        
        return {"logits": logits, "loss": loss}
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

# ============================================================
# SGLD OPTIMIZER
# ============================================================
class AnnealedSGLD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, initial_temp=0.01, final_temp=1e-7, anneal_steps=500):
        defaults = dict(lr=lr, initial_temp=initial_temp, final_temp=final_temp, anneal_steps=anneal_steps)
        super().__init__(params, defaults)
        self.step_count = 0
    
    def get_temperature(self):
        group = self.param_groups[0]
        progress = min(self.step_count / group['anneal_steps'], 1.0)
        t0, t1 = group['initial_temp'], group['final_temp']
        return t1 + 0.5 * (t0 - t1) * (1 + math.cos(math.pi * progress))
    
    @torch.no_grad()
    def step(self):
        temp = self.get_temperature()
        for group in self.param_groups:
            noise_scale = math.sqrt(2.0 * group['lr'] * temp)
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.add_(p.grad, alpha=-group['lr'])
                if temp > 0:
                    p.data.add_(torch.randn_like(p) * noise_scale)
        self.step_count += 1

# ============================================================
# DATASET
# ============================================================
class SimpleDataset(IterableDataset):
    def __init__(self, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __iter__(self):
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        for item in ds:
            text = item.get("text", "")[:1500]
            if len(text) < 50:
                continue
            tokens = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                     return_tensors="pt", padding="max_length")
            yield tokens["input_ids"].squeeze(0)

# ============================================================
# MAIN TRAINING
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Download checkpoint
    print(f"\nDownloading {CHECKPOINT}...")
    ckpt_path = hf_hub_download(repo_id=REPO_ID, filename=CHECKPOINT, token=HF_TOKEN)
    
    # Create model
    print("Creating 3B model...")
    model = StudentLLM()
    
    # Load weights
    print("Loading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    
    # Move to GPU with FP16
    model = model.half().to(device)
    model.train()
    
    print(f"Parameters: {model.num_parameters():,}")
    print(f"Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    
    # Optimizer
    optimizer = AnnealedSGLD(model.parameters(), lr=LR, anneal_steps=STEPS)
    
    # Data
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = SimpleDataset(tokenizer, max_length=256)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    data_iter = iter(dataloader)
    
    # Training loop
    print(f"\nStarting training ({STEPS} steps)...")
    print("="*60)
    
    start = time.time()
    for step in range(1, STEPS + 1):
        try:
            input_ids = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            input_ids = next(data_iter)
        
        input_ids = input_ids.to(device)
        
        outputs = model(input_ids, labels=input_ids)
        loss = outputs["loss"]
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 50 == 0:
            elapsed = time.time() - start
            eta = (STEPS - step) / (step / elapsed) / 60
            temp = optimizer.get_temperature()
            mem = torch.cuda.memory_allocated() / 1e9
            print(f"Step {step}/{STEPS} | Loss: {loss.item():.3f} | Temp: {temp:.2e} | Mem: {mem:.1f}GB | ETA: {eta:.1f}m")
    
    # Save final checkpoint
    print("\nSaving final checkpoint...")
    final_path = "titan-3b-3000.pt"
    torch.save({"model_state_dict": model.float().state_dict(), "mode": "titan_3b", "step": 3000}, final_path)
    
    # Upload to HF
    print("Uploading to HuggingFace...")
    api = HfApi(token=HF_TOKEN)
    api.upload_file(path_or_fileobj=final_path, path_in_repo="titan/titan-3b-3000.pt",
                    repo_id=REPO_ID, token=HF_TOKEN)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print(f"Final checkpoint: titan/titan-3b-3000.pt")
    print("="*60)

if __name__ == "__main__":
    main()
