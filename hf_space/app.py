#!/usr/bin/env python3
"""
Distillix Training Space - HuggingFace A10G
FP32 + Muon for BitNet stability
"""
import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download, upload_file, HfApi
import gradio as gr
import time

# Clone the model code
if not os.path.exists('smelter'):
    os.system('git clone https://github.com/rileyseaburg/distillix.git /tmp/distillix')
    os.system('cp -r /tmp/distillix/smelter .')
    os.system('cp -r /tmp/distillix/data .')

sys.path.insert(0, '.')

from smelter.model import StudentLLM
from smelter.config import get_config_125m
from smelter.optim import MuonAdamW  # Updated import


class DistillationDataset(Dataset):
    def __init__(self, paths: list, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        for path in paths:
            if not os.path.exists(path):
                # Try downloading from HF
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


def train_model(max_steps=5000, learning_rate=0.02, batch_size=16, progress=gr.Progress()):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Enable TF32 for speed on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    yield f"Starting training on {device} (FP32 mode)...\n"
    
    # Config & Model
    config = get_config_125m()
    model = StudentLLM(config.model).to(device)
    
    # Try to load checkpoint
    try:
        ckpt_path = hf_hub_download(
            repo_id="rileyseaburg/distillix-100m-v0.3",
            filename="distillix-v05-cognitive.pt"
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Handle architecture mismatch by loading what we can
        model_state = model.state_dict()
        for k, v in ckpt.get('model_state_dict', ckpt).items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
        model.load_state_dict(model_state, strict=False)
        yield f"Loaded checkpoint from HuggingFace\n"
    except Exception as e:
        yield f"Starting from scratch: {e}\n"
    
    model.enable_gradient_checkpointing()
    total_params = sum(p.numel() for p in model.parameters())
    yield f"Parameters: {total_params:,}\n"
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Data - download from HF
    try:
        data_path = hf_hub_download(
            repo_id="rileyseaburg/distillix-100m-v0.3",
            filename="train_full_10k.jsonl"
        )
        dataset = DistillationDataset([data_path], tokenizer, max_length=512)
    except:
        yield "Error: Could not load training data\n"
        return
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Optimizer - MuonAdamW (Muon for matrices, AdamW for embeddings)
    optimizer = MuonAdamW(
        named_params=model.named_parameters(),
        muon_lr=learning_rate,
        adamw_lr=learning_rate / 66,  # 0.02 / 66 ≈ 0.0003
        weight_decay=0.01,
    )
    
    yield f"Optimizer: Muon ({optimizer.num_matrix:,} params) + AdamW ({optimizer.num_vector:,} params)\n"
    
    grad_accum = 2
    
    yield f"Training for {max_steps} steps (batch={batch_size}, accum={grad_accum}, FP32)...\n\n"
    
    model.train()
    step = 0
    running_loss = 0.0
    start_time = time.time()
    log_output = ""
    
    while step < max_steps:
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # FP32 forward pass (no AMP/autocast) - prevents BitNet NaN
            outputs = model(input_ids, labels=labels)
            loss = outputs['loss'] / grad_accum
            
            loss.backward()
            running_loss += loss.item()
            
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            step += 1
            progress(step / max_steps)
            
            if step % 50 == 0:
                avg_loss = running_loss / 50 * grad_accum
                running_loss = 0.0
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
                eta_min = (max_steps - step) / steps_per_sec / 60
                
                # Check for NaN
                if avg_loss != avg_loss:  # NaN check
                    msg = f"Step {step}/{max_steps} | Loss: NaN - TRAINING FAILED\n"
                    log_output += msg
                    yield log_output
                    return
                
                msg = f"Step {step}/{max_steps} | Loss: {avg_loss:.4f} | {steps_per_sec:.1f} steps/s | ETA: {eta_min:.1f}m\n"
                log_output += msg
                yield log_output
            
            if step % 1000 == 0:
                # Save checkpoint
                save_path = f"distillix-grok-{step//1000}k.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config.model.to_dict(),
                    'step': step,
                }, save_path)
                log_output += f"  Saved checkpoint: {save_path}\n"
                yield log_output
            
            if step >= max_steps:
                break
    
    # Save final model
    yield log_output + "\nSaving final model...\n"
    
    save_path = "distillix-grok-final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.model.to_dict(),
        'step': step,
    }, save_path)
    
    # Upload to HF
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=save_path,
            path_in_repo="distillix-grok-final.pt",
            repo_id="rileyseaburg/distillix-100m-v0.3",
        )
        yield log_output + f"\nUploaded to HuggingFace!\n"
    except Exception as e:
        yield log_output + f"\nUpload failed: {e}\n"
    
    yield log_output + "\nTraining complete!"


# Gradio UI
with gr.Blocks(title="Distillix Training") as demo:
    gr.Markdown("# Distillix 100M BitNet Training")
    gr.Markdown("Train BitNet with **FP32 + Muon** for stability (no AMP)")
    
    with gr.Row():
        max_steps = gr.Slider(1000, 20000, value=5000, step=1000, label="Max Steps")
        lr = gr.Slider(0.005, 0.05, value=0.02, step=0.005, label="Muon LR")
        batch_size = gr.Slider(4, 32, value=16, step=4, label="Batch Size")
    
    train_btn = gr.Button("Start Training", variant="primary")
    output = gr.Textbox(label="Training Log", lines=20, max_lines=30)
    
    train_btn.click(
        fn=train_model,
        inputs=[max_steps, lr, batch_size],
        outputs=output,
    )

if __name__ == "__main__":
    demo.queue().launch()
