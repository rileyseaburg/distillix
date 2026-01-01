#!/usr/bin/env python3
"""Distillix Training - Debug version"""
import gradio as gr
import traceback
import sys

# Capture startup errors
startup_log = []

def log(msg):
    startup_log.append(str(msg))
    print(msg, flush=True)

try:
    log("=== STARTUP ===")
    
    log("Importing torch...")
    import torch
    log(f"  torch {torch.__version__}")
    log(f"  CUDA available: {torch.cuda.is_available()}")
    log(f"  GPU count: {torch.cuda.device_count()}")
    
    log("Importing transformers...")
    from transformers import AutoTokenizer
    log("  OK")
    
    log("Importing huggingface_hub...")
    from huggingface_hub import hf_hub_download, HfApi
    log("  OK")
    
    log("Importing smelter.bitnet...")
    sys.path.insert(0, '.')
    from smelter.bitnet import BitLinear, RMSNorm
    log("  OK")
    
    log("Importing smelter.config...")
    from smelter.config import get_config_125m
    log("  OK")
    
    log("Checking get_config_630m...")
    try:
        from smelter.config import get_config_630m
        log("  get_config_630m exists!")
    except ImportError as e:
        log(f"  get_config_630m missing: {e}")
    
    log("Importing smelter.model...")
    from smelter.model import StudentLLM
    log("  OK")
    
    log("Importing smelter.muon...")
    from smelter.muon import MuonAdamW
    log("  OK")
    
    log("")
    log("=== ALL IMPORTS OK ===")
    IMPORTS_OK = True

except Exception as e:
    log(f"IMPORT ERROR: {e}")
    log(traceback.format_exc())
    IMPORTS_OK = False


def get_startup_log():
    return "\n".join(startup_log)


def run_test():
    results = []
    results.append("=== TEST RUN ===")
    
    if not IMPORTS_OK:
        results.append("Imports failed - see startup log")
        return "\n".join(results)
    
    try:
        results.append("Creating 100M config...")
        config = get_config_125m()
        results.append(f"  hidden_dim: {config.model.hidden_dim}")
        
        results.append("Creating model...")
        model = StudentLLM(config.model)
        params = sum(p.numel() for p in model.parameters())
        results.append(f"  params: {params:,}")
        
        results.append("Moving to GPU...")
        model = model.cuda()
        results.append("  OK")
        
        results.append("Test forward pass...")
        x = torch.randint(0, 1000, (1, 32)).cuda()
        out = model(x)
        results.append(f"  output shape: {out['logits'].shape}")
        
        results.append("")
        results.append("=== SUCCESS ===")
        
    except Exception as e:
        results.append(f"ERROR: {e}")
        results.append(traceback.format_exc())
    
    return "\n".join(results)


with gr.Blocks(title="Distillix Debug") as demo:
    gr.Markdown("# Distillix Debug")
    
    startup_box = gr.Textbox(label="Startup Log", value=get_startup_log, lines=20)
    
    test_btn = gr.Button("Run Test")
    test_out = gr.Textbox(label="Test Output", lines=15)
    
    test_btn.click(run_test, outputs=test_out)

demo.launch()
