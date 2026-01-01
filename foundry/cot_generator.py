#!/usr/bin/env python3
"""
Cognitive Kernel Data Generator (v0.5)

Generates training data with structured reasoning traces:
- DECOMPOSE: Break problem into atomic steps
- CONSTRAINTS: List requirements and edge cases  
- DRAFT: Quick naive implementation
- CRITIQUE: Find flaws in the draft
- REFINE: Final optimized solution

This teaches the model to ENGINEER, not just CODE.

Usage:
    python -m foundry.cot_generator --count 1000 --output data/cot_train.jsonl
"""

import os
import json
import asyncio
import aiohttp
import argparse
import random
from typing import Optional

# MiniMax API config (Anthropic-compatible endpoint)
MINIMAX_BASE_URL = "https://api.minimax.io/anthropic/v1/messages"
MINIMAX_API_KEY = "sk-api-vli90ZHzgEZxnG_JVzeDZD_1-nOQiI2lhb1PKghH1yotqk6RE05maodQD4ogYcmb19aV-DDI-LkhuTJHRPqfcFXTk_L5f4MOq3OsLDys0HNZpktQjmMB8go"
MODEL = "MiniMax-M2.1"

# The Cognitive Kernel System Prompt
COGNITIVE_KERNEL_PROMPT = """You are a Senior Principal Engineer. Show your complete reasoning process.

Structure EVERY response with these 4 sections using the EXACT XML tags:

<DECOMPOSE>
Break the problem into numbered steps:
1. First step
2. Second step
...
</DECOMPOSE>

<DRAFT>
Write your first implementation:
```python
# Your initial code here
```
</DRAFT>

<CRITIQUE>
Find at least ONE flaw in your draft:
- Bug or edge case missed?
- Performance issue?
- Code smell?
Be specific about what's wrong.
</CRITIQUE>

<REFINE>
Write improved code fixing the CRITIQUE issue:
```python
# Your improved code here
```
</REFINE>

IMPORTANT: All 4 XML tags (<DECOMPOSE>, <DRAFT>, <CRITIQUE>, <REFINE>) are REQUIRED."""

# Problem templates that require deep reasoning
ALGORITHMS = [
    "binary search with duplicates handling",
    "merge sort with optimization for nearly-sorted arrays", 
    "quick sort with three-way partitioning",
    "BFS with path reconstruction",
    "DFS with cycle detection",
    "Dijkstra's algorithm with early termination",
    "dynamic programming for longest common subsequence",
    "LRU cache with O(1) operations",
    "trie with prefix counting",
    "union-find with path compression and rank",
    "topological sort with cycle detection",
    "sliding window maximum using deque",
    "two pointers for three-sum",
    "interval merging with optimal complexity",
    "binary indexed tree (Fenwick tree)",
]

PROBLEMS = [
    "find the longest palindromic substring efficiently",
    "detect a cycle in a linked list and return the start node",
    "serialize and deserialize a binary tree to string",
    "find all unique permutations handling duplicates",
    "implement a thread-safe rate limiter",
    "find the median of two sorted arrays in O(log(min(m,n)))",
    "validate a binary search tree handling edge cases",
    "merge k sorted lists in O(n log k) time",
    "find the longest increasing subsequence with O(n log n)",
    "implement a min stack with O(1) getMin",
    "design a data structure for O(1) insert, delete, getRandom",
    "find the kth largest element without full sorting",
    "implement a trie with wildcard search",
    "detect if a graph is bipartite",
    "find strongly connected components",
]

CONCEPTS = [
    "Python decorators with arguments and functools.wraps",
    "generators vs iterators and memory efficiency",
    "context managers and the with statement internals",
    "metaclasses and when you'd actually use them",
    "descriptors and the descriptor protocol",
    "async/await and the event loop",
    "the GIL and its implications for threading",
    "Python memory management and reference counting",
    "multiple inheritance and Method Resolution Order",
    "closures, nonlocal, and scope chain",
    "slots and memory optimization",
    "__new__ vs __init__ and singleton pattern",
]

SYSTEM_DESIGN = [
    "a URL shortener handling billions of URLs",
    "a rate limiter for an API gateway",
    "a distributed cache with consistency",
    "a message queue with guaranteed delivery",
    "a search autocomplete system",
    "a real-time leaderboard for millions of users",
    "a file storage system like Dropbox",
    "a notification system handling millions of pushes",
]

DEBUGGING = [
    '''def binary_search(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) / 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return -1''',
    
    '''def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    while left and right:
        if left[0] < right[0]:
            result.append(left.pop(0))  # Bug here
        else:
            result.append(right.pop(0))
    return result + left + right''',
    
    '''class LRUCache:
    def __init__(self, capacity):
        self.cache = {}
        self.capacity = capacity
    
    def get(self, key):
        if key in self.cache:
            return self.cache[key]
        return -1
    
    def put(self, key, value):
        if len(self.cache) >= self.capacity:
            del self.cache[next(iter(self.cache))]  # Bug: not LRU
        self.cache[key] = value''',
]


def generate_prompt() -> str:
    """Generate a random problem requiring deep reasoning."""
    template_type = random.choice([
        'algorithm', 'algorithm', 'problem', 'problem',  # Weight towards code
        'concept', 'debug', 'system'
    ])
    
    if template_type == 'algorithm':
        algo = random.choice(ALGORITHMS)
        return f"Implement {algo} in Python."
    
    elif template_type == 'problem':
        problem = random.choice(PROBLEMS)
        return f"Write a Python function to {problem}."
    
    elif template_type == 'concept':
        concept = random.choice(CONCEPTS)
        return f"Explain and implement {concept} with a practical example."
    
    elif template_type == 'debug':
        code = random.choice(DEBUGGING)
        return f"This code has bugs. Find them, explain why they're wrong, and fix them:\n```python\n{code}\n```"
    
    else:  # system
        system = random.choice(SYSTEM_DESIGN)
        return f"Design {system}. Focus on the core data structures and algorithms needed."


async def call_minimax(session: aiohttp.ClientSession, prompt: str) -> Optional[str]:
    """Call MiniMax API (Anthropic-compatible) for CoT response."""
    headers = {
        "x-api-key": MINIMAX_API_KEY,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    payload = {
        "model": MODEL,
        "max_tokens": 3000,
        "system": COGNITIVE_KERNEL_PROMPT,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        async with session.post(MINIMAX_BASE_URL, json=payload, headers=headers, timeout=60) as resp:
            if resp.status == 200:
                data = await resp.json()
                # MiniMax Anthropic-compatible format
                content = data.get("content", [])
                if content and isinstance(content, list):
                    # Find the text block (skip thinking block)
                    for block in content:
                        if block.get("type") == "text":
                            return block.get("text", "")
                return ""
            else:
                text = await resp.text()
                print(f"API error {resp.status}: {text[:200]}")
                return None
    except asyncio.TimeoutError:
        print("Request timeout")
        return None
    except Exception as e:
        print(f"Request error: {e}")
        return None


def validate_cognitive_response(response: str) -> bool:
    """Check if response has proper Cognitive Kernel structure."""
    required_sections = ["<DECOMPOSE>", "<DRAFT>", "<CRITIQUE>", "<REFINE>"]
    has_all = all(section in response for section in required_sections)
    has_code = "```python" in response or "def " in response
    return has_all and has_code


async def generate_batch(count: int, output_path: str, workers: int = 10):
    """Generate batch of Cognitive Kernel samples."""
    
    print("="*60)
    print("COGNITIVE KERNEL DATA GENERATOR (v0.5)")
    print("="*60)
    print(f"Target: {count} samples")
    print(f"Workers: {workers}")
    print(f"Output: {output_path}")
    print(f"Model: {MODEL}")
    print()
    
    samples = []
    valid_count = 0
    failed_count = 0
    
    connector = aiohttp.TCPConnector(limit=workers)
    timeout = aiohttp.ClientTimeout(total=120)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for i in range(0, count, workers):
            batch_size = min(workers, count - i)
            prompts = [generate_prompt() for _ in range(batch_size)]
            
            tasks = [call_minimax(session, p) for p in prompts]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for prompt, response in zip(prompts, responses):
                if isinstance(response, Exception):
                    failed_count += 1
                    continue
                    
                if response and validate_cognitive_response(response):
                    samples.append({
                        "prompt": prompt,
                        "response": response,
                        "type": "cognitive_kernel"
                    })
                    valid_count += 1
                else:
                    failed_count += 1
            
            pct = valid_count / (valid_count + failed_count) * 100 if (valid_count + failed_count) > 0 else 0
            print(f"[{i + batch_size}/{count}] Valid: {valid_count} ({pct:.0f}%) | Failed: {failed_count}")
            
            # Rate limiting
            await asyncio.sleep(0.2)
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print()
    print("="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"Valid samples: {valid_count}")
    print(f"Failed: {failed_count}")
    print(f"Success rate: {valid_count/(valid_count+failed_count)*100:.1f}%")
    print(f"Output: {output_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Generate Cognitive Kernel training data')
    parser.add_argument('--count', '-c', type=int, default=100, help='Number of samples')
    parser.add_argument('--output', '-o', type=str, default='data/distillation/cognitive_kernel.jsonl')
    parser.add_argument('--workers', '-w', type=int, default=5, help='Concurrent requests')
    args = parser.parse_args()
    
    asyncio.run(generate_batch(args.count, args.output, args.workers))


if __name__ == '__main__':
    main()
