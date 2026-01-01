#!/usr/bin/env python3
"""
Distillix Grokking Run - Muon + FP32

Fixes BitNet NaN instability by:
1. Pure FP32 training (no AMP/Autocast) - prevents numerical underflow/overflow
2. MuonAdamW optimizer - orthogonalized updates prevent gradient explosions
"""
import os, sys, json, torch, time
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, '/root/distillix')
from smelter.config import ModelConfig, Config
from smelter.model import StudentLLM
from smelter.optim import MuonAdamW

def main():
    # Enforce TF32 for speed (Ampere+ GPUs)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    config = Config(model=ModelConfig(use_bitlinear=True, use_bitlinear_ffn=False))
    model = StudentLLM(config.model).cuda()
    
    # Load checkpoint (Cognitive base or Resume)
    ckpt_path = 'artifacts/distillix-v05-cognitive.pt'
    if os.path.exists('artifacts/distillix-grok-latest.pt'):
        ckpt_path = 'artifacts/distillix-grok-latest.pt'
        print("Resuming from latest grok checkpoint...")
        
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
        
    model.enable_gradient_checkpointing()

    print("="*60)
    print("DISTILLIX GROKKING RUN - MUON + FP32")
    print("="*60)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    tokenizer.pad_token = tokenizer.eos_token

    samples = []
    for path in ['data/distillation/train_full_10k.jsonl', 'data/distillation/cognitive_kernel_v2.jsonl']:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        text = item.get('text') or f"{item.get('prompt','')}\n{item.get('response','')}"
                        if text.strip(): samples.append(text)
                    except: pass
    print(f"Samples: {len(samples)}")

    class DS(Dataset):
        def __init__(s, texts, tok, maxlen): s.texts, s.tok, s.maxlen = texts, tok, maxlen
        def __len__(s): return len(s.texts)
        def __getitem__(s, i):
            enc = s.tok(s.texts[i], truncation=True, max_length=s.maxlen, padding='max_length', return_tensors='pt')
            ids = enc['input_ids'].squeeze(0)
            return {'input_ids': ids, 'labels': ids.clone()}

    # Batch 16 fits easily in FP32 for 125M model
    loader = DataLoader(DS(samples, tokenizer, 512), batch_size=16, shuffle=True, num_workers=2, drop_last=True)
    
    # Initialize MuonAdamW
    # Muon LR 0.02 (Matrices), AdamW LR 3e-4 (Vectors/Embeds)
    optimizer = MuonAdamW(
        model.named_parameters(), 
        muon_lr=0.02,
        adamw_lr=0.0003,
        weight_decay=0.01
    )
    
    print(f"Optimizer: Muon ({optimizer.num_matrix:,} params) + AdamW ({optimizer.num_vector:,} params)")

    model.train()
    step, running_loss, t0 = 0, 0, time.time()
    max_steps = 50000
    grad_accum = 2

    print(f"Training for {max_steps} steps (FP32 Mode)...")
    print()

    while step < max_steps:
        for batch in loader:
            ids, labels = batch['input_ids'].cuda(), batch['labels'].cuda()
            
            # Forward pass in FP32 (No AMP/Autocast)
            # This prevents BitNet numerical instability
            loss = model(ids, labels=labels)['loss'] / grad_accum
            loss.backward()
            
            running_loss += loss.item()
            
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            step += 1
            
            if step % 100 == 0:
                # running_loss tracks scaled loss, multiply by grad_accum to get actual
                avg = running_loss / 100 * grad_accum
                running_loss = 0
                elapsed = time.time() - t0
                if elapsed > 0:
                    eta_hr = (max_steps - step) / (step / elapsed) / 3600
                    print(f'Step {step:6d}/{max_steps} | Loss: {avg:.4f} | ETA: {eta_hr:.1f}h', flush=True)
            
            if step % 5000 == 0:
                torch.save({'model_state_dict': model.state_dict(), 'step': step}, f'artifacts/distillix-grok-{step//1000}k.pt')
                torch.save({'model_state_dict': model.state_dict()}, 'artifacts/distillix-grok-latest.pt')
                print(f'  Saved checkpoint', flush=True)
            
            if step >= max_steps: break

    torch.save({'model_state_dict': model.state_dict()}, 'artifacts/distillix-grok-final.pt')
    print('Done!')

if __name__ == '__main__':
    main()
