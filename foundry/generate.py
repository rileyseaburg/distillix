"""
Dataset Generation Pipeline for Knowledge Distillation.

Generates training data by querying teacher models through OpenCode server.
Includes data sanitization to filter API refusals and low-quality responses.

Features:
  - Multi-teacher ensemble querying
  - Data sanitizer: Filters refusals, markdown artifacts, broken JSON
  - "Phi Protocol": Textbook-quality generation prompts
  - Fill-In-Middle (FIM) data augmentation

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
# Data Sanitizer - Filter API Refusals and Low-Quality Responses
# =============================================================================

# Refusal patterns that indicate the model declined to answer
REFUSAL_PATTERNS = [
    # Direct refusals
    "i cannot",
    "i can't",
    "i'm unable to",
    "i am unable to",
    "i'm not able to",
    "i am not able to",
    "i won't",
    "i will not",
    "i don't think i can",
    "i do not think i can",
    
    # Apology refusals
    "i'm sorry, but i",
    "i am sorry, but i",
    "sorry, but i can't",
    "sorry, i cannot",
    "i apologize, but",
    "my apologies, but",
    
    # Policy refusals
    "as an ai",
    "as a language model",
    "as an assistant",
    "i'm designed to",
    "i am designed to",
    "it would not be appropriate",
    "it wouldn't be appropriate",
    "i'm not comfortable",
    "i am not comfortable",
    
    # Capability refusals
    "i don't have the ability",
    "i do not have the ability",
    "i don't have access",
    "i do not have access",
    "beyond my capabilities",
    "outside my capabilities",
    
    # Safety refusals
    "potentially harmful",
    "could be harmful",
    "might be harmful",
    "unsafe content",
    "inappropriate content",
    "violates",
    "against my guidelines",
    "against my programming",
]

# Patterns indicating incomplete or broken responses
BROKEN_PATTERNS = [
    # Truncation indicators
    "...[truncated]",
    "[continued]",
    "[rest of response]",
    "```\n\n```",  # Empty code blocks
    
    # Error messages
    "error:",
    "exception:",
    "traceback:",
    "failed to",
    
    # Placeholder text
    "[insert",
    "[your ",
    "[add ",
    "todo:",
    "fixme:",
    "xxx:",
]

# Markdown artifacts that should be cleaned
MARKDOWN_ARTIFACTS = [
    ("```python\n```", ""),
    ("```\n```", ""),
    ("```javascript\n```", ""),
    ("```typescript\n```", ""),
    ("```json\n```", ""),
    ("\n\n\n\n", "\n\n"),
    ("\n\n\n", "\n\n"),
]


@dataclass
class SanitizationResult:
    """Result of sanitizing a response."""
    is_valid: bool
    content: str
    rejection_reason: Optional[str] = None
    quality_score: float = 1.0


class DataSanitizer:
    """
    Sanitizes teacher responses to filter out:
      - API refusals ("I'm sorry, as an AI...")
      - Broken/truncated responses
      - Low-quality or off-topic content
      - Markdown artifacts
    
    This is CRITICAL for small model training.
    "Garbage In" kills 125M models 10x faster than large models.
    """
    
    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 10000,
        min_quality_score: float = 0.3,
        check_code_validity: bool = True,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.min_quality_score = min_quality_score
        self.check_code_validity = check_code_validity
        
        # Statistics
        self.stats = {
            "total_checked": 0,
            "passed": 0,
            "rejected_refusal": 0,
            "rejected_broken": 0,
            "rejected_too_short": 0,
            "rejected_too_long": 0,
            "rejected_low_quality": 0,
        }
    
    def _check_refusal(self, content: str) -> Optional[str]:
        """Check if response contains a refusal pattern."""
        content_lower = content.lower()
        
        # Check first 500 chars for refusal (usually at start)
        check_region = content_lower[:500]
        
        for pattern in REFUSAL_PATTERNS:
            if pattern in check_region:
                return f"refusal_pattern:{pattern}"
        
        return None
    
    def _check_broken(self, content: str) -> Optional[str]:
        """Check if response is broken or incomplete."""
        content_lower = content.lower()
        
        for pattern in BROKEN_PATTERNS:
            if pattern.lower() in content_lower:
                return f"broken_pattern:{pattern}"
        
        # Check for unbalanced code blocks
        code_blocks = content.count("```")
        if code_blocks % 2 != 0:
            return "unbalanced_code_blocks"
        
        # Check for unbalanced brackets in code
        if self.check_code_validity:
            open_parens = content.count("(") - content.count(")")
            open_brackets = content.count("[") - content.count("]")
            open_braces = content.count("{") - content.count("}")
            
            if abs(open_parens) > 5 or abs(open_brackets) > 5 or abs(open_braces) > 5:
                return "severely_unbalanced_brackets"
        
        return None
    
    def _clean_markdown(self, content: str) -> str:
        """Clean up markdown artifacts."""
        for pattern, replacement in MARKDOWN_ARTIFACTS:
            content = content.replace(pattern, replacement)
        
        return content.strip()
    
    def _calculate_quality_score(self, content: str, prompt: str) -> float:
        """
        Calculate a quality score for the response.
        
        Factors:
          - Length relative to prompt complexity
          - Code-to-text ratio for coding prompts
          - Structural markers (headers, lists, code blocks)
          - Repetition detection
        """
        score = 1.0
        
        # Length scoring
        length = len(content)
        if length < 100:
            score *= 0.5
        elif length < 200:
            score *= 0.8
        
        # Check for excessive repetition
        words = content.lower().split()
        if len(words) > 20:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                score *= 0.3  # Heavy penalty for repetition
            elif unique_ratio < 0.5:
                score *= 0.7
        
        # Bonus for structural elements (indicates well-organized response)
        if "```" in content:
            score *= 1.1  # Has code blocks
        if any(marker in content for marker in ["1.", "2.", "- ", "* "]):
            score *= 1.05  # Has lists
        if "#" in content[:200]:
            score *= 1.05  # Has headers
        
        # Check if response addresses the prompt
        prompt_words = set(prompt.lower().split()[:10])
        content_words = set(content.lower().split()[:100])
        overlap = len(prompt_words & content_words)
        if overlap < 2:
            score *= 0.5  # Response seems off-topic
        
        return min(score, 1.0)
    
    def sanitize(self, content: str, prompt: str = "") -> SanitizationResult:
        """
        Sanitize a single response.
        
        Args:
            content: The response content to sanitize
            prompt: The original prompt (for quality scoring)
        
        Returns:
            SanitizationResult with validity and cleaned content
        """
        self.stats["total_checked"] += 1
        
        # Check length first (fast)
        if len(content) < self.min_length:
            self.stats["rejected_too_short"] += 1
            return SanitizationResult(
                is_valid=False,
                content=content,
                rejection_reason=f"too_short:{len(content)}<{self.min_length}",
            )
        
        # Truncate if too long (don't reject, just truncate)
        if len(content) > self.max_length:
            content = content[:self.max_length]
            # Try to truncate at a sentence boundary
            last_period = content.rfind(". ")
            if last_period > self.max_length * 0.8:
                content = content[:last_period + 1]
        
        # Check for refusals
        refusal = self._check_refusal(content)
        if refusal:
            self.stats["rejected_refusal"] += 1
            return SanitizationResult(
                is_valid=False,
                content=content,
                rejection_reason=refusal,
            )
        
        # Check for broken responses
        broken = self._check_broken(content)
        if broken:
            self.stats["rejected_broken"] += 1
            return SanitizationResult(
                is_valid=False,
                content=content,
                rejection_reason=broken,
            )
        
        # Clean markdown artifacts
        content = self._clean_markdown(content)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(content, prompt)
        
        if quality_score < self.min_quality_score:
            self.stats["rejected_low_quality"] += 1
            return SanitizationResult(
                is_valid=False,
                content=content,
                rejection_reason=f"low_quality:{quality_score:.2f}<{self.min_quality_score}",
                quality_score=quality_score,
            )
        
        # Passed all checks
        self.stats["passed"] += 1
        return SanitizationResult(
            is_valid=True,
            content=content,
            quality_score=quality_score,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sanitization statistics."""
        total = max(self.stats["total_checked"], 1)
        return {
            **self.stats,
            "pass_rate": self.stats["passed"] / total,
            "refusal_rate": self.stats["rejected_refusal"] / total,
        }


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

# =============================================================================
# "Phi Protocol" Textbook-Quality Prompts
# =============================================================================
# Microsoft's Phi-1 (1.3B) beat 100B models on coding by training on
# "textbook quality" synthetic data instead of raw GitHub dumps.
#
# Key insight: 1 token of "textbook" ≈ 100 tokens of "raw code"
#
# These prompts ask teachers to generate educational, structured content
# with clear explanations and examples.
# =============================================================================

# Template for textbook-style prompts
PHI_TEXTBOOK_TEMPLATE = """Write a textbook chapter section on the following topic.

TOPIC: {topic}

FORMAT YOUR RESPONSE AS:
1. **Concept Overview**: A clear 2-3 sentence explanation of what this is and why it matters.
2. **How It Works**: Step-by-step explanation of the underlying mechanism.
3. **Code Example**: A well-commented, production-quality code example.
4. **Common Pitfalls**: 2-3 mistakes beginners make and how to avoid them.
5. **When to Use**: Practical guidance on when this approach is appropriate.

Write for an intermediate programmer who understands basic programming but is learning this specific concept."""

SEED_PROMPTS_REASONING = [
    PHI_TEXTBOOK_TEMPLATE.format(topic="Recursion in Programming - how functions call themselves to solve problems"),
    PHI_TEXTBOOK_TEMPLATE.format(topic="Binary Search - the divide and conquer algorithm for sorted arrays"),
    PHI_TEXTBOOK_TEMPLATE.format(topic="Hash Tables - O(1) lookup data structures and collision handling"),
    PHI_TEXTBOOK_TEMPLATE.format(topic="Big O Notation - analyzing algorithm time and space complexity"),
    PHI_TEXTBOOK_TEMPLATE.format(topic="Backpropagation - how neural networks learn through gradient descent"),
    PHI_TEXTBOOK_TEMPLATE.format(topic="Garbage Collection - automatic memory management in programming languages"),
    PHI_TEXTBOOK_TEMPLATE.format(topic="Concurrency vs Parallelism - the difference and when to use each"),
    PHI_TEXTBOOK_TEMPLATE.format(topic="The SOLID Principles - writing maintainable object-oriented code"),
    PHI_TEXTBOOK_TEMPLATE.format(topic="SQL vs NoSQL - choosing the right database for your use case"),
    PHI_TEXTBOOK_TEMPLATE.format(topic="REST vs GraphQL - API design patterns and trade-offs"),
]

PHI_CODE_TEMPLATE = """Write a comprehensive implementation guide for the following programming task.

TASK: {task}

FORMAT YOUR RESPONSE AS:
1. **Problem Statement**: Clearly define what we're building and why.
2. **Approach**: Explain the algorithm/design pattern we'll use.
3. **Implementation**: Write clean, well-documented Python code.
4. **Complexity Analysis**: Time and space complexity of the solution.
5. **Test Cases**: Show example inputs and expected outputs.
6. **Edge Cases**: Handle boundary conditions and error cases.

Write production-quality code with proper error handling and type hints."""

SEED_PROMPTS_CODE = [
    PHI_CODE_TEMPLATE.format(task="Implement a Binary Search Tree with insert, search, and delete operations"),
    PHI_CODE_TEMPLATE.format(task="Build an LRU Cache using a hash map and doubly linked list"),
    PHI_CODE_TEMPLATE.format(task="Write a function to detect cycles in a linked list using Floyd's algorithm"),
    PHI_CODE_TEMPLATE.format(task="Implement Depth-First Search and Breadth-First Search for graphs"),
    PHI_CODE_TEMPLATE.format(task="Build a thread-safe rate limiter using the token bucket algorithm"),
    PHI_CODE_TEMPLATE.format(task="Implement a trie (prefix tree) for autocomplete functionality"),
    PHI_CODE_TEMPLATE.format(task="Write a merge sort implementation with detailed step-by-step tracing"),
    PHI_CODE_TEMPLATE.format(task="Build a simple expression parser and evaluator for arithmetic"),
    PHI_CODE_TEMPLATE.format(task="Implement a producer-consumer queue with proper synchronization"),
    PHI_CODE_TEMPLATE.format(task="Write a decorator that implements memoization with cache invalidation"),
]

PHI_ANALYSIS_TEMPLATE = """Provide a technical analysis of the following software engineering topic.

TOPIC: {topic}

FORMAT YOUR RESPONSE AS:
1. **Executive Summary**: Key takeaway in 2-3 sentences.
2. **Technical Deep Dive**: Explain how this works under the hood.
3. **Comparison Table**: If applicable, compare alternatives.
4. **Real-World Example**: A practical scenario demonstrating the concept.
5. **Best Practices**: Industry-standard recommendations.
6. **Code Demonstration**: Illustrative code if applicable.

Write for a senior developer evaluating technology choices."""

SEED_PROMPTS_ANALYSIS = [
    PHI_ANALYSIS_TEMPLATE.format(topic="CAP Theorem - consistency, availability, partition tolerance trade-offs"),
    PHI_ANALYSIS_TEMPLATE.format(topic="Microservices vs Monoliths - architectural trade-offs at scale"),
    PHI_ANALYSIS_TEMPLATE.format(topic="Password Storage Security - hashing, salting, and modern best practices"),
    PHI_ANALYSIS_TEMPLATE.format(topic="Caching Strategies - write-through, write-back, and cache invalidation"),
    PHI_ANALYSIS_TEMPLATE.format(topic="Event Sourcing vs CRUD - when to use each pattern"),
    PHI_ANALYSIS_TEMPLATE.format(topic="Synchronous vs Asynchronous Programming - I/O patterns and scalability"),
    PHI_ANALYSIS_TEMPLATE.format(topic="ORM vs Raw SQL - performance, maintainability, and type safety"),
    PHI_ANALYSIS_TEMPLATE.format(topic="JWT vs Session Authentication - security and scalability considerations"),
    PHI_ANALYSIS_TEMPLATE.format(topic="Strong vs Eventual Consistency - distributed systems trade-offs"),
    PHI_ANALYSIS_TEMPLATE.format(topic="Horizontal vs Vertical Scaling - when to scale out vs scale up"),
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
    
    Includes data sanitization to filter:
      - API refusals ("I'm sorry, as an AI...")
      - Broken/truncated responses  
      - Low-quality or off-topic content
    
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
        
        # Data sanitizer for filtering low-quality responses
        self.sanitizer = DataSanitizer(
            min_length=config.min_response_length,
            max_length=config.max_response_length,
            min_quality_score=0.3,
        )
        
        # Statistics
        self.stats = {
            "prompts_processed": 0,
            "samples_generated": 0,
            "samples_filtered": 0,
            "errors": 0,
        }
    
    def _filter_response(self, response: EnsembleResponse, prompt: str) -> Dict[str, str]:
        """
        Filter and sanitize responses using DataSanitizer.
        
        Returns:
            Dict of model -> sanitized content (only valid responses)
        """
        valid_responses = {}
        
        for model, resp in response.responses.items():
            result = self.sanitizer.sanitize(resp.content, prompt)
            
            if result.is_valid:
                valid_responses[model] = result.content
            else:
                logger.debug(f"Filtered {model}: {result.rejection_reason}")
        
        return valid_responses
    
    def _format_sample(
        self,
        prompt_data: Dict[str, Any],
        response: EnsembleResponse,
        sanitized_responses: Dict[str, str],
    ) -> Dict[str, Any]:
        """Format sample for training with sanitized responses."""
        return {
            "prompt": response.prompt,
            "responses": sanitized_responses,  # Use sanitized content
            "metadata": {
                "category": prompt_data.get("category", "general"),
                "teachers_used": list(sanitized_responses.keys()),
                "teachers_filtered": [
                    m for m in response.responses.keys() 
                    if m not in sanitized_responses
                ],
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
                            
                            # Sanitize and filter responses
                            sanitized = self._filter_response(response, prompt)
                            
                            if not sanitized:
                                # All responses were filtered out
                                self.stats["samples_filtered"] += 1
                                continue
                            
                            # Format and write
                            sample = self._format_sample(prompt_data, response, sanitized)
                            f.write(json.dumps(sample) + "\n")
                            f.flush()
                            
                            self.stats["samples_generated"] += 1
                            
                        except Exception as e:
                            logger.error(f"Error generating sample: {e}")
                            self.stats["errors"] += 1
                    
                    self.stats["prompts_processed"] += 1
                    pbar.set_postfix({
                        "samples": self.stats["samples_generated"],
                        "filtered": self.stats["samples_filtered"],
                        "errors": self.stats["errors"],
                    })
        
        # Log sanitizer stats
        sanitizer_stats = self.sanitizer.get_stats()
        logger.info(f"\nGeneration complete!")
        logger.info(f"  Prompts processed: {self.stats['prompts_processed']}")
        logger.info(f"  Samples generated: {self.stats['samples_generated']}")
        logger.info(f"  Samples filtered: {self.stats['samples_filtered']}")
        logger.info(f"  Errors: {self.stats['errors']}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"\nSanitizer Statistics:")
        logger.info(f"  Pass rate: {sanitizer_stats['pass_rate']:.1%}")
        logger.info(f"  Refusals filtered: {sanitizer_stats['rejected_refusal']}")
        logger.info(f"  Broken responses: {sanitizer_stats['rejected_broken']}")
        logger.info(f"  Low quality: {sanitizer_stats['rejected_low_quality']}")


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
