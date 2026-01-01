#!/usr/bin/env python3
"""
Distillix BitNet Training Script for Vertex AI
FP32 + Muon optimizer for stability
"""
import os
import sys
import json
import torch
import time

# Install dependencies
os.system('pip install -q transformers huggingface_hub')

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download, HfApi


# ============================================================================
# Muon Optimizer (inline to avoid dependency issues)
# ============================================================================
from torch.optim import Optimizer

def newton_schulz_orthogonalize(M, num_iters=5, eps=1e-7):
    norm = torch.norm(M, p='fro')
    if norm < eps:
        return M
    X = M / (norm + eps)
    for _ in range(num_iters):
        A = X.T @ X
        X = X @ (1.5 * torch.eye(A.shape[0], device=A.device, dtype=A.dtype) - 0.5 * A)
    return X * norm


class Muon(Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_iters=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_iters=ns_iters, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr, mom, nesterov, ns_iters, wd = group['lr'], group['momentum'], group['nesterov'], group['ns_iters'], group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0:
                    p.mul_(1 - lr * wd)
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                buf = state['momentum_buffer']
                buf.mul_(mom).add_(grad)
                buf_ortho = newton_schulz_orthogonalize(buf, num_iters=ns_iters)
                update = grad + mom * buf_ortho if nesterov else buf_ortho
                update = newton_schulz_orthogonalize(update, num_iters=ns_iters)
                p.add_(update, alpha=-lr)
        return loss


class MuonAdamW:
    def __init__(self, named_params, muon_lr=0.02, adamw_lr=3e-4, weight_decay=0.01):
        muon_params, adamw_params = [], []
        for name, p in named_params:
            if not p.requires_grad:
                continue
            is_embedding = 'embed' in name.lower() or 'lm_head' in name.lower()
            if p.ndim == 2 and not is_embedding:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        self.optims = []
        if muon_params:
            self.optims.append(Muon(muon_params, lr=muon_lr, weight_decay=weight_decay))
        if adamw_params:
            self.optims.append(torch.optim.AdamW(adamw_params, lr=adamw_lr, weight_decay=weight_decay))
        self.num_matrix = sum(p.numel() for p in muon_params)
        self.num_vector = sum(p.numel() for p in adamw_params)

    def step(self):
        for opt in self.optims:
            opt.step()
    
    def zero_grad(self, set_to_none=True):
        for opt in self.optims:
            opt.zero_grad(set_to_none=set_to_none)


# ============================================================================
# Model Definition (inline)
# ============================================================================
import torch.nn as nn
import torch.nn.functional as F
import math


class STESign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.register_buffer('weight_scale', torch.ones(1))
    
    def forward(self, x):
        w = self.weight
        alpha = w.abs().mean()
        w_quant = STESign.apply(w) * alpha
        self.weight_scale = alpha
        out = F.linear(x, w_quant, self.bias)
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
    
    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, use_bitlinear=True):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        
        Linear = BitLinear if use_bitlinear else nn.Linear
        self.q_proj = Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = Linear(n_heads * self.head_dim, dim, bias=False)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
    
    def forward(self, x, cos, sin, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.tanh(attn / 50.0) * 50.0  # Soft capping
        
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, use_bitlinear=True):
        super().__init__()
        Linear = BitLinear if use_bitlinear else nn.Linear
        self.gate_proj = Linear(dim, hidden_dim, bias=False)
        self.up_proj = Linear(dim, hidden_dim, bias=False)
        self.down_proj = Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, ffn_dim, use_bitlinear=True):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, n_kv_heads, use_bitlinear)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FFN(dim, ffn_dim, use_bitlinear)
    
    def forward(self, x, cos, sin, mask=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class StudentLLM(nn.Module):
    def __init__(self, vocab_size=32000, dim=768, n_layers=12, n_heads=12, n_kv_heads=4,
                 ffn_dim=2048, max_seq_len=2048, use_bitlinear=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        
        self.embed = nn.Embedding(vocab_size, dim)
        self.rotary = RotaryEmbedding(dim // n_heads, max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, ffn_dim, use_bitlinear)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.embed.weight = self.lm_head.weight  # Weight tying
        
        self._gradient_checkpointing = False
    
    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True
    
    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        x = self.embed(input_ids)
        
        cos, sin = self.rotary(x, T)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        mask = mask.float().masked_fill(mask, float('-inf'))
        
        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, cos, sin, mask, use_reentrant=False)
            else:
                x = layer(x, cos, sin, mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        logits = torch.tanh(logits / 30.0) * 30.0  # Soft capping
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
        
        return {'loss': loss, 'logits': logits}


# ============================================================================
# Dataset
# ============================================================================
class DistillationDataset(Dataset):
    def __init__(self, paths, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        for path in paths:
            if not os.path.exists(path):
                try:
                    path = hf_hub_download(
                        repo_id="rileyseaburg/distillix-100m-v0.3",
                        filename=os.path.basename(path),
                        repo_type="model"
                    )
                except:
                    continue
            
            with open(path, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        if 'text' in item:
                            self.samples.append(item['text'])
                        elif 'prompt' in item and 'response' in item:
                            self.samples.append(f"{item['prompt']}\n{item['response']}")
                    except:
                        continue
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding='max_length', return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        return {'input_ids': input_ids, 'labels': input_ids.clone()}


# ============================================================================
# Main Training
# ============================================================================
def main():
    max_steps = int(os.environ.get('MAX_STEPS', '10000'))
    batch_size = int(os.environ.get('BATCH_SIZE', '16'))
    muon_lr = float(os.environ.get('MUON_LR', '0.02'))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print("=" * 60)
    print("DISTILLIX BITNET TRAINING - VERTEX AI")
    print("=" * 60)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Model
    model = StudentLLM(
        vocab_size=32000, dim=768, n_layers=12, n_heads=12, n_kv_heads=4,
        ffn_dim=2048, max_seq_len=2048, use_bitlinear=True
    ).to(device)
    
    # Load checkpoint
    try:
        ckpt_path = hf_hub_download(
            repo_id="rileyseaburg/distillix-100m-v0.3",
            filename="distillix-v05-cognitive.pt"
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state, strict=False)
        print("Loaded checkpoint from HuggingFace")
    except Exception as e:
        print(f"Starting fresh: {e}")
    
    model.enable_gradient_checkpointing()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    except:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Data
    data_path = hf_hub_download(
        repo_id="rileyseaburg/distillix-100m-v0.3",
        filename="train_full_10k.jsonl"
    )
    dataset = DistillationDataset([data_path], tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Optimizer
    optimizer = MuonAdamW(
        model.named_parameters(),
        muon_lr=muon_lr,
        adamw_lr=muon_lr / 66,
        weight_decay=0.01,
    )
    print(f"Optimizer: Muon ({optimizer.num_matrix:,}) + AdamW ({optimizer.num_vector:,})")
    
    grad_accum = 2
    print(f"Training: {max_steps} steps, batch={batch_size}, accum={grad_accum}, FP32")
    print()
    
    # Train
    model.train()
    step = 0
    running_loss = 0.0
    t0 = time.time()
    
    while step < max_steps:
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, labels=labels)
            loss = outputs['loss'] / grad_accum
            loss.backward()
            running_loss += loss.item()
            
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            step += 1
            
            if step % 100 == 0:
                avg = running_loss / 100 * grad_accum
                running_loss = 0.0
                elapsed = time.time() - t0
                sps = step / elapsed
                eta = (max_steps - step) / sps / 60
                
                if avg != avg:
                    print(f"Step {step} | Loss: NaN - FAILED")
                    return
                
                print(f"Step {step:6d}/{max_steps} | Loss: {avg:.4f} | {sps:.1f} steps/s | ETA: {eta:.1f}m")
            
            if step % 2000 == 0:
                try:
                    path = f"distillix-vertex-{step//1000}k.pt"
                    torch.save({'model_state_dict': model.state_dict(), 'step': step}, path)
                    api = HfApi()
                    api.upload_file(path_or_fileobj=path, path_in_repo=path, repo_id="rileyseaburg/distillix-100m-v0.3")
                    print(f"  Uploaded {path} to HuggingFace")
                except Exception as e:
                    print(f"  Upload failed: {e}")
            
            if step >= max_steps:
                break
    
    # Final save
    print("\nSaving final model...")
    path = "distillix-vertex-final.pt"
    torch.save({'model_state_dict': model.state_dict(), 'step': step}, path)
    try:
        api = HfApi()
        api.upload_file(path_or_fileobj=path, path_in_repo=path, repo_id="rileyseaburg/distillix-100m-v0.3")
        print("Uploaded to HuggingFace!")
    except Exception as e:
        print(f"Upload failed: {e}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
