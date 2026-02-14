"""
Data Pipeline for Distillix - Unified Data Processing

Converts all data sources into a consistent training format.
Combines CodeTether sessions, MiniMax generations, and synthetic data.

Usage:
    python -m foundry.data_pipeline --combine --output data/training/unified.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional
from dataclasses import dataclass
import random


@dataclass
class TrainingExample:
    """A single training example."""
    prompt: str
    response: str
    source: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "source": self.source,
            "metadata": self.metadata or {},
        }
    
    def to_chat_format(self) -> Dict[str, Any]:
        """Convert to OpenAI-style chat format."""
        return {
            "messages": [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": self.response},
            ],
            "source": self.source,
        }


def load_codetether_data(path: str) -> Iterator[TrainingExample]:
    """Load data from CodeTether extraction format (messages array)."""
    with open(path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                messages = data.get('messages', [])
                
                # Extract user and assistant messages
                user_content = ""
                assistant_content = ""
                
                for msg in messages:
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    
                    if role == 'user':
                        user_content += content + "\n"
                    elif role == 'assistant':
                        assistant_content += content + "\n"
                
                if user_content.strip() and assistant_content.strip():
                    yield TrainingExample(
                        prompt=user_content.strip(),
                        response=assistant_content.strip(),
                        source="codetether",
                        metadata=data.get('metadata', {}),
                    )
            except json.JSONDecodeError:
                continue


def load_minimax_data(path: str) -> Iterator[TrainingExample]:
    """Load data from MiniMax generation format."""
    with open(path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                
                prompt = data.get('prompt', '')
                response = data.get('response', '')
                
                # Include thinking if available (Chain of Thought)
                thinking = data.get('thinking', '')
                if thinking:
                    # Prefix response with thinking in a visible way
                    response = f"<thinking>\n{thinking}\n</thinking>\n\n{response}"
                
                if prompt and response:
                    yield TrainingExample(
                        prompt=prompt,
                        response=response,
                        source="minimax",
                        metadata=data.get('metadata', {}),
                    )
            except json.JSONDecodeError:
                continue


def load_generic_jsonl(path: str) -> Iterator[TrainingExample]:
    """Load generic JSONL with prompt/response or messages format."""
    with open(path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                
                # Check for messages format
                if 'messages' in data:
                    yield from load_codetether_data_single(data)
                # Check for prompt/response format
                elif 'prompt' in data and 'response' in data:
                    yield TrainingExample(
                        prompt=data['prompt'],
                        response=data['response'],
                        source=data.get('source', 'generic'),
                        metadata=data.get('metadata', {}),
                    )
                # Check for input/output format
                elif 'input' in data and 'output' in data:
                    yield TrainingExample(
                        prompt=data['input'],
                        response=data['output'],
                        source=data.get('source', 'generic'),
                        metadata=data.get('metadata', {}),
                    )
            except json.JSONDecodeError:
                continue


def load_codetether_data_single(data: Dict) -> Iterator[TrainingExample]:
    """Process single CodeTether entry."""
    messages = data.get('messages', [])
    
    user_content = ""
    assistant_content = ""
    
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if role == 'user':
            user_content += content + "\n"
        elif role == 'assistant':
            assistant_content += content + "\n"
    
    if user_content.strip() and assistant_content.strip():
        yield TrainingExample(
            prompt=user_content.strip(),
            response=assistant_content.strip(),
            source="codetether",
            metadata=data.get('metadata', {}),
        )


def combine_datasets(
    output_path: str,
    codetether_paths: List[str] = None,
    minimax_paths: List[str] = None,
    generic_paths: List[str] = None,
    shuffle: bool = True,
    deduplicate: bool = True,
) -> int:
    """
    Combine multiple datasets into a unified training file.
    
    Returns the number of examples written.
    """
    all_examples = []
    seen_prompts = set()
    
    # Load CodeTether data
    for path in (codetether_paths or []):
        if Path(path).exists():
            print(f"Loading CodeTether: {path}")
            for ex in load_codetether_data(path):
                if deduplicate:
                    prompt_hash = hash(ex.prompt[:200])
                    if prompt_hash in seen_prompts:
                        continue
                    seen_prompts.add(prompt_hash)
                all_examples.append(ex)
    
    # Load MiniMax data
    for path in (minimax_paths or []):
        if Path(path).exists():
            print(f"Loading MiniMax: {path}")
            for ex in load_minimax_data(path):
                if deduplicate:
                    prompt_hash = hash(ex.prompt[:200])
                    if prompt_hash in seen_prompts:
                        continue
                    seen_prompts.add(prompt_hash)
                all_examples.append(ex)
    
    # Load generic data
    for path in (generic_paths or []):
        if Path(path).exists():
            print(f"Loading generic: {path}")
            for ex in load_generic_jsonl(path):
                if deduplicate:
                    prompt_hash = hash(ex.prompt[:200])
                    if prompt_hash in seen_prompts:
                        continue
                    seen_prompts.add(prompt_hash)
                all_examples.append(ex)
    
    print(f"\nTotal examples: {len(all_examples)}")
    
    # Shuffle
    if shuffle:
        random.shuffle(all_examples)
        print("Shuffled.")
    
    # Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    total_chars = 0
    with open(output_path, 'w') as f:
        for ex in all_examples:
            f.write(json.dumps(ex.to_dict()) + "\n")
            total_chars += len(ex.prompt) + len(ex.response)
    
    print(f"Written to: {output_path}")
    print(f"Total chars: {total_chars:,}")
    print(f"Avg chars/example: {total_chars / len(all_examples):.0f}")
    
    return len(all_examples)


def main():
    parser = argparse.ArgumentParser(description="Distillix Data Pipeline")
    parser.add_argument("--combine", action="store_true", help="Combine datasets")
    parser.add_argument("--output", "-o", type=str, default="data/training/unified.jsonl")
    parser.add_argument("--codetether", nargs="*", default=["data/training/combined_full.jsonl"])
    parser.add_argument("--minimax", nargs="*", default=["data/training/synthetic_500.jsonl"])
    parser.add_argument("--generic", nargs="*", default=[])
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--no-dedup", action="store_true")
    args = parser.parse_args()
    
    if args.combine:
        count = combine_datasets(
            output_path=args.output,
            codetether_paths=args.codetether,
            minimax_paths=args.minimax,
            generic_paths=args.generic,
            shuffle=not args.no_shuffle,
            deduplicate=not args.no_dedup,
        )
        print(f"\nDone! {count} examples combined.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
