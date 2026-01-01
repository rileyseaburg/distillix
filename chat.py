#!/usr/bin/env python3
"""
Distillix Chat Interface
Usage: python chat.py [--checkpoint PATH]
"""
import torch
import torch.nn.functional as F
import sys
import argparse
import readline  # For arrow key history

sys.path.insert(0, '/root/distillix')

from smelter.model import StudentLLM
from smelter.config import get_config_125m
from transformers import AutoTokenizer

def load_model(checkpoint_path):
    config = get_config_125m()
    model = StudentLLM(config.model).cuda()
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def generate(model, tokenizer, prompt, max_tokens=200, temperature=0.7, top_p=0.9):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            if generated.shape[1] >= 2048:
                break
            
            output = model(generated)
            logits = output['logits'][:, -1, :] / temperature
            
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Print token as it's generated
            token_str = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(token_str, end='', flush=True)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    print()  # Newline after generation
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description='Distillix Chat')
    parser.add_argument('--checkpoint', '-c', type=str, 
                        default='/root/distillix/artifacts/distillix-v0.3-burnin-861.pt',
                        help='Model checkpoint path')
    parser.add_argument('--temperature', '-t', type=float, default=0.7)
    parser.add_argument('--max-tokens', '-m', type=int, default=200)
    args = parser.parse_args()
    
    print("="*50)
    print("  DISTILLIX CHAT")
    print("  100M BitNet b1.58 Language Model")
    print("="*50)
    
    if 'burnin' in args.checkpoint:
        print("\n⚠️  WARNING: This is the burn-in model (861 samples only)")
        print("   Output will be largely incoherent - this is expected!")
        print("   Train on full 10k+ dataset for coherent responses.")
    
    print(f"\nLoading {args.checkpoint}...")
    
    model = load_model(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Ready! Type 'quit' to exit, 'clear' to reset.\n")
    
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
        
        # Build prompt with history
        prompt = ""
        for h in history[-4:]:  # Last 4 turns
            prompt += f"User: {h['user']}\nAssistant: {h['assistant']}\n"
        prompt += f"User: {user_input}\nAssistant:"
        
        print("\033[94mDistillix:\033[0m ", end='')
        full_response = generate(model, tokenizer, prompt, 
                                  max_tokens=args.max_tokens,
                                  temperature=args.temperature)
        
        # Extract just the assistant response
        response = full_response.split("Assistant:")[-1].strip()
        history.append({'user': user_input, 'assistant': response})
        print()

if __name__ == '__main__':
    main()
