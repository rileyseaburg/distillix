"""
Dataset Generation Pipeline for Knowledge Distillation.

Generates training data by querying teacher models through OpenCode server.

Usage:
    # Start OpenCode server first:
    opencode serve --port 4096
    
    # Run generation:
    python -m foundry.generate \
        --prompts data/prompts/seed.jsonl \
        --output data/distillation/train.jsonl \
        --samples-per-prompt 1

Copyright (c) 2025 Distillix. All Rights Reserved.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator
from dataclasses import dataclass
from datetime import datetime
import argparse

from tqdm import tqdm

from .opencode_client import OpenCodeClient
from .teacher import (
    TeacherEnsemble,
    TeacherEnsembleContext,
    EnsembleConfig,
    TeacherConfig,
    TeacherModel,
    EnsembleResponse,
)


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("distillix.generate")


# =============================================================================
# Prompt Sources
# =============================================================================

def load_prompts_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load prompts from JSONL file.
    
    Expected format:
    {"prompt": "...", "category": "...", "metadata": {...}}
    
    Or simple:
    {"prompt": "..."}
    """
    prompts = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if isinstance(data, str):
                        data = {"prompt": data}
                    prompts.append(data)
                except json.JSONDecodeError:
                    # Plain text line
                    prompts.append({"prompt": line})
    
    return prompts


def load_prompts_txt(path: str) -> List[Dict[str, Any]]:
    """Load prompts from plain text file (one per line)."""
    prompts = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                prompts.append({"prompt": line})
    
    return prompts


def load_prompts(path: str) -> List[Dict[str, Any]]:
    """Load prompts from file (auto-detect format)."""
    path = Path(path)
    
    if path.suffix == '.jsonl':
        return load_prompts_jsonl(str(path))
    elif path.suffix in ['.txt', '.md']:
        return load_prompts_txt(str(path))
    else:
        # Try JSONL first
        try:
            return load_prompts_jsonl(str(path))
        except json.JSONDecodeError:
            return load_prompts_txt(str(path))


# =============================================================================
# Built-in Prompt Sets
# =============================================================================

SEED_PROMPTS_REASONING = [
    "Explain the concept of recursion in programming with a simple example.",
    "What is the difference between machine learning and deep learning?",
    "How does a binary search algorithm work? Explain step by step.",
    "What are the SOLID principles in software engineering?",
    "Explain how a hash table works and when to use one.",
    "What is the difference between SQL and NoSQL databases?",
    "How does garbage collection work in programming languages?",
    "Explain the concept of Big O notation with examples.",
    "What is the difference between concurrency and parallelism?",
    "How do neural networks learn? Explain backpropagation.",
]

SEED_PROMPTS_CODE = [
    "Write a Python function to check if a string is a palindrome.",
    "Implement a binary search tree in Python with insert and search methods.",
    "Write a function to find the nth Fibonacci number using dynamic programming.",
    "Implement a simple LRU cache in Python.",
    "Write a function to merge two sorted arrays into one sorted array.",
    "Implement a queue using two stacks.",
    "Write a Python decorator that measures function execution time.",
    "Implement depth-first search for a graph.",
    "Write a function to detect a cycle in a linked list.",
    "Implement a basic rate limiter in Python.",
]

SEED_PROMPTS_ANALYSIS = [
    "Analyze the time complexity of quicksort in best, average, and worst cases.",
    "Compare REST and GraphQL APIs. When would you use each?",
    "What are the trade-offs between microservices and monolithic architecture?",
    "Analyze the CAP theorem and its implications for distributed systems.",
    "Compare different approaches to handling authentication in web applications.",
    "What are the pros and cons of using an ORM vs raw SQL?",
    "Analyze the security implications of storing passwords. What's best practice?",
    "Compare synchronous and asynchronous programming models.",
    "What are the trade-offs in choosing between strong and eventual consistency?",
    "Analyze different caching strategies and when to use each.",
]


def get_builtin_prompts(category: str = "all") -> List[Dict[str, Any]]:
    """Get built-in seed prompts."""
    prompts = []
    
    if category in ["all", "reasoning"]:
        prompts.extend([
            {"prompt": p, "category": "reasoning"}
            for p in SEED_PROMPTS_REASONING
        ])
    
    if category in ["all", "code"]:
        prompts.extend([
            {"prompt": p, "category": "code"}
            for p in SEED_PROMPTS_CODE
        ])
    
    if category in ["all", "analysis"]:
        prompts.extend([
            {"prompt": p, "category": "analysis"}
            for p in SEED_PROMPTS_ANALYSIS
        ])
    
    return prompts


# =============================================================================
# Data Generator
# =============================================================================

@dataclass
class GenerationConfig:
    """Configuration for data generation."""
    
    # Input
    prompts_path: Optional[str] = None
    use_builtin: bool = False
    builtin_category: str = "all"
    
    # Output
    output_path: str = "data/distillation/train.jsonl"
    
    # Generation settings
    samples_per_prompt: int = 1
    max_concurrent: int = 5
    
    # Teachers
    teachers: List[str] = None
    teacher_strategy: str = "all"
    
    # OpenCode server
    opencode_host: str = "127.0.0.1"
    opencode_port: int = 4096
    
    # Filtering
    min_response_length: int = 50
    max_response_length: int = 10000
    
    def __post_init__(self):
        if self.teachers is None:
            self.teachers = [
                TeacherModel.AZURE_CLAUDE.value,
                TeacherModel.GLM_47.value,
                TeacherModel.MINIMAX_M21.value,
            ]


class DataGenerator:
    """
    Generates training data from teacher models.
    
    Usage:
        config = GenerationConfig(prompts_path="prompts.txt")
        generator = DataGenerator(config)
        await generator.generate()
    """
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        
        # Load prompts
        if config.prompts_path:
            self.prompts = load_prompts(config.prompts_path)
        elif config.use_builtin:
            self.prompts = get_builtin_prompts(config.builtin_category)
        else:
            raise ValueError("Must provide prompts_path or use_builtin=True")
        
        logger.info(f"Loaded {len(self.prompts)} prompts")
        
        # Statistics
        self.stats = {
            "prompts_processed": 0,
            "samples_generated": 0,
            "samples_filtered": 0,
            "errors": 0,
        }
    
    def _filter_response(self, response: EnsembleResponse) -> bool:
        """Check if response passes quality filters."""
        for model, resp in response.responses.items():
            content_len = len(resp.content)
            
            if content_len < self.config.min_response_length:
                logger.debug(f"Filtered {model}: too short ({content_len})")
                return False
            
            if content_len > self.config.max_response_length:
                logger.debug(f"Filtered {model}: too long ({content_len})")
                # Don't filter, just truncate later
        
        return len(response.responses) > 0
    
    def _format_sample(
        self,
        prompt_data: Dict[str, Any],
        response: EnsembleResponse,
    ) -> Dict[str, Any]:
        """Format sample for training."""
        return {
            "prompt": response.prompt,
            "responses": {
                model: resp.content[:self.config.max_response_length]
                for model, resp in response.responses.items()
            },
            "metadata": {
                "category": prompt_data.get("category", "general"),
                "teachers_used": list(response.responses.keys()),
                "total_tokens": response.total_tokens,
                "latency_ms": response.total_latency_ms,
                "generated_at": datetime.utcnow().isoformat(),
                "errors": response.errors,
            },
        }
    
    async def generate(self):
        """Run data generation."""
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting generation...")
        logger.info(f"  Prompts: {len(self.prompts)}")
        logger.info(f"  Samples per prompt: {self.config.samples_per_prompt}")
        logger.info(f"  Teachers: {self.config.teachers}")
        logger.info(f"  Output: {output_path}")
        
        # Create ensemble config
        ensemble_config = EnsembleConfig(
            teachers=[
                TeacherConfig(model_id=t)
                for t in self.config.teachers
            ],
            strategy=self.config.teacher_strategy,
        )
        
        async with TeacherEnsembleContext(
            config=ensemble_config,
            opencode_host=self.config.opencode_host,
            opencode_port=self.config.opencode_port,
        ) as ensemble:
            
            with open(output_path, 'w') as f:
                pbar = tqdm(self.prompts, desc="Generating")
                
                for prompt_data in pbar:
                    prompt = prompt_data["prompt"]
                    
                    for sample_idx in range(self.config.samples_per_prompt):
                        try:
                            # Query teachers
                            response = await ensemble.query(prompt)
                            
                            # Filter
                            if not self._filter_response(response):
                                self.stats["samples_filtered"] += 1
                                continue
                            
                            # Format and write
                            sample = self._format_sample(prompt_data, response)
                            f.write(json.dumps(sample) + "\n")
                            f.flush()
                            
                            self.stats["samples_generated"] += 1
                            
                        except Exception as e:
                            logger.error(f"Error generating sample: {e}")
                            self.stats["errors"] += 1
                    
                    self.stats["prompts_processed"] += 1
                    pbar.set_postfix({
                        "samples": self.stats["samples_generated"],
                        "errors": self.stats["errors"],
                    })
        
        logger.info(f"\nGeneration complete!")
        logger.info(f"  Prompts processed: {self.stats['prompts_processed']}")
        logger.info(f"  Samples generated: {self.stats['samples_generated']}")
        logger.info(f"  Samples filtered: {self.stats['samples_filtered']}")
        logger.info(f"  Errors: {self.stats['errors']}")
        logger.info(f"  Output: {output_path}")


# =============================================================================
# Quick Generation Function
# =============================================================================

async def generate_dataset(
    output_path: str = "data/distillation/train.jsonl",
    prompts_path: Optional[str] = None,
    use_builtin: bool = True,
    num_samples: int = 100,
    teachers: Optional[List[str]] = None,
    opencode_port: int = 4096,
):
    """
    Quick function to generate a dataset.
    
    Args:
        output_path: Where to save the dataset
        prompts_path: Path to prompts file (optional)
        use_builtin: Use built-in prompts
        num_samples: Number of samples to generate
        teachers: List of teacher models to use
        opencode_port: OpenCode server port
    """
    config = GenerationConfig(
        output_path=output_path,
        prompts_path=prompts_path,
        use_builtin=use_builtin if not prompts_path else False,
        teachers=teachers,
        opencode_port=opencode_port,
    )
    
    generator = DataGenerator(config)
    await generator.generate()


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate distillation training data from teacher models"
    )
    
    # Input
    parser.add_argument(
        "--prompts", "-p",
        type=str,
        default=None,
        help="Path to prompts file (JSONL or TXT)",
    )
    parser.add_argument(
        "--builtin",
        action="store_true",
        help="Use built-in seed prompts",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        choices=["all", "reasoning", "code", "analysis"],
        help="Category of built-in prompts",
    )
    
    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/distillation/train.jsonl",
        help="Output JSONL path",
    )
    
    # Generation
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=1,
        help="Samples to generate per prompt",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent requests",
    )
    
    # Teachers
    parser.add_argument(
        "--teachers",
        type=str,
        nargs="+",
        default=None,
        help="Teacher models to use",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        choices=["all", "random", "round_robin"],
        help="Teacher selection strategy",
    )
    
    # Server
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="OpenCode server host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4096,
        help="OpenCode server port",
    )
    
    args = parser.parse_args()
    
    # Validate
    if not args.prompts and not args.builtin:
        parser.error("Must provide --prompts or --builtin")
    
    # Create config
    config = GenerationConfig(
        prompts_path=args.prompts,
        use_builtin=args.builtin,
        builtin_category=args.category,
        output_path=args.output,
        samples_per_prompt=args.samples_per_prompt,
        max_concurrent=args.max_concurrent,
        teachers=args.teachers,
        teacher_strategy=args.strategy,
        opencode_host=args.host,
        opencode_port=args.port,
    )
    
    # Run
    generator = DataGenerator(config)
    asyncio.run(generator.generate())


if __name__ == "__main__":
    main()
