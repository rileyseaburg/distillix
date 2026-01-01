#!/usr/bin/env python3
"""
Distillix Chat Interface
Usage: python chat.py [--checkpoint PATH]

Features:
- Streaming token output
- Repetition penalty to prevent loops
- Top-k + Top-p sampling
- Conversation history
"""
import torch
import torch.nn.functional as F
import sys
import argparse
import readline  # For arrow key history

sys.path.insert(0, '/root/distillix')

from smelter.model import StudentLLM
from smelter.bitnet import BitLinear  # Load BitNet module
from smelter.config import get_config_125m
from transformers import AutoTokenizer


def load_model(checkpoint_path, device='cuda'):
    config = get_config_125m()
    model = StudentLLM(config.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def generate(model, tokenizer, prompt, max_tokens=200, temperature=0.7, 
             top_k=50, top_p=0.9, repetition_penalty=1.2):
    """
    Generate with repetition penalty and top-k/top-p sampling.
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
    generated = input_ids.clone()
    
    # Track generated tokens for repetition penalty
    generated_tokens = set()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            if generated.shape[1] >= 2048:
                break
            
            output = model(generated)
            logits = output['logits'][:, -1, :].clone()
            
            # Apply repetition penalty
            for token_id in generated_tokens:
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Track for repetition penalty
            generated_tokens.add(next_token.item())
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stream token
            token_str = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(token_str, end='', flush=True)
            
            # Stop conditions
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Stop on newline after reasonable output
            if '\n' in token_str and generated.shape[1] > input_ids.shape[1] + 20:
                break
    
    print()  # Newline
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description='Distillix Chat')
    parser.add_argument('--checkpoint', '-c', type=str, 
                        default='artifacts/distillix-v0.3-burnin-861.pt',
                        help='Model checkpoint path')
    parser.add_argument('--temperature', '-t', type=float, default=0.7,
                        help='Sampling temperature (lower = more focused)')
    parser.add_argument('--max-tokens', '-m', type=int, default=150,
                        help='Maximum tokens to generate')
    parser.add_argument('--rep-penalty', '-r', type=float, default=1.2,
                        help='Repetition penalty (1.0 = off, 1.2 = moderate)')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*55)
    print("  DISTILLIX CHAT - 100M BitNet b1.58")
    print("="*55)
    
    if 'burnin' in args.checkpoint:
        print("""
╔═══════════════════════════════════════════════════════╗
║  ⚠️  WARNING: BURN-IN MODEL (861 samples)              ║
║                                                        ║
║  This model demonstrates architecture stability only.  ║
║  Expect:                                               ║
║    - Missing spaces and broken grammar                 ║
║    - Keyword soup without coherent sentences           ║
║    - Repetition (mitigated by rep_penalty=1.2)         ║
║                                                        ║
║  For coherent output: train on 10k+ dataset            ║
╚═══════════════════════════════════════════════════════╝
""")
    
    print(f"Loading {args.checkpoint}...")
    print(f"Device: {device}")
    
    model = load_model(args.checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"\nSettings: temp={args.temperature}, rep_penalty={args.rep_penalty}")
    print("Commands: 'quit' to exit, 'clear' to reset history\n")
    
    history = []
    
    while True:
        try:
            user_input = input("\033[92mYou:\033[0m ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Bye!")
            break
        
        if user_input.lower() == 'clear':
            history = []
            print("History cleared.\n")
            continue
        
        if not user_input.strip():
            continue
        
        # Build prompt - use Phi-style for concepts
        if any(kw in user_input.lower() for kw in ['what is', 'explain', 'how', 'define']):
            # Concept explanation format
            prompt = f"Concept: {user_input}\n\nExplanation:"
        else:
            # Conversational format with history
            prompt = ""
            for h in history[-2:]:  # Last 2 turns only
                prompt += f"User: {h['user']}\nAssistant: {h['assistant']}\n"
            prompt += f"User: {user_input}\nAssistant:"
        
        print("\033[94mDistillix:\033[0m ", end='')
        full_response = generate(
            model, tokenizer, prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            repetition_penalty=args.rep_penalty
        )
        
        # Extract response
        if "Explanation:" in full_response:
            response = full_response.split("Explanation:")[-1].strip()
        else:
            response = full_response.split("Assistant:")[-1].strip()
        
        history.append({'user': user_input, 'assistant': response})
        print()


if __name__ == '__main__':
    main()
