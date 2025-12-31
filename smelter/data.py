"""
Dataset and DataLoader for BitNet Distillation Training.

Handles:
  - Loading JSONL datasets with teacher responses
  - Tokenization with padding/truncation
  - Collation for batch processing
  - Multi-teacher response handling

Copyright (c) 2025 Distillix. All Rights Reserved.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch import Tensor
from typing import Optional, Dict, List, Any, Iterator, Union
from pathlib import Path
import random
from dataclasses import dataclass


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DistillationSample:
    """Single sample for distillation training."""
    prompt: str
    response: str  # Selected teacher response
    teacher_name: str
    all_responses: Dict[str, str]  # All teacher responses
    metadata: Dict[str, Any]


@dataclass
class BatchEncoding:
    """Tokenized batch ready for model."""
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor
    teacher_names: List[str]


# =============================================================================
# Tokenizer Wrapper
# =============================================================================

class TokenizerWrapper:
    """
    Wrapper for HuggingFace tokenizers with distillation-specific methods.
    """
    
    def __init__(
        self,
        tokenizer_name_or_path: str = "meta-llama/Llama-2-7b-hf",
        max_length: int = 1024,
        padding_side: str = "right",
        truncation_side: str = "right",
    ):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            trust_remote_code=True,
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.tokenizer.padding_side = padding_side
        self.tokenizer.truncation_side = truncation_side
        self.max_length = max_length
    
    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)
    
    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id
    
    def encode_for_distillation(
        self,
        prompt: str,
        response: str,
        add_eos: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Encode prompt + response for distillation training.
        
        Format: <prompt><response>[EOS]
        Labels: -100 for prompt tokens (don't compute loss), response tokens otherwise
        
        Args:
            prompt: Input prompt text
            response: Teacher's response text
            add_eos: Whether to add EOS token at end
        
        Returns:
            Dict with 'input_ids', 'attention_mask', 'labels'
        """
        # Tokenize prompt and response separately to know boundary
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        
        if add_eos:
            response_ids = response_ids + [self.eos_token_id]
        
        # Combine
        input_ids = prompt_ids + response_ids
        
        # Truncate if needed
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            # Ensure we still have some response tokens
            if len(input_ids) <= len(prompt_ids):
                # Truncate prompt to leave room for response
                prompt_len = self.max_length // 2
                input_ids = prompt_ids[:prompt_len] + response_ids[:self.max_length - prompt_len]
        
        # Create labels: -100 for prompt, actual ids for response
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        
        # Pad to max_length
        pad_len = self.max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [self.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    
    def decode(self, token_ids: Union[List[int], Tensor]) -> str:
        """Decode token IDs to text."""
        if isinstance(token_ids, Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


# =============================================================================
# JSONL Dataset
# =============================================================================

class DistillationDataset(Dataset):
    """
    Dataset for distillation training from JSONL files.
    
    Expected JSONL format:
    {
        "prompt": "User's input...",
        "responses": {
            "azure-claude": "Teacher 1 response...",
            "glm-4.7": "Teacher 2 response...",
            "minimax-m2.1": "Teacher 3 response..."
        },
        "metadata": {"timestamp": ..., "tokens": ...}
    }
    
    Or simple format:
    {
        "prompt": "...",
        "response": "...",
        "teacher": "teacher-name"
    }
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: TokenizerWrapper,
        teacher_strategy: str = "random",  # "random", "round_robin", "specific"
        specific_teacher: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.teacher_strategy = teacher_strategy
        self.specific_teacher = specific_teacher
        
        # Load data
        self.samples = self._load_data(max_samples)
        
        # Round-robin state
        self._rr_index = 0
    
    def _load_data(self, max_samples: Optional[int]) -> List[Dict]:
        """Load JSONL data."""
        samples = []
        
        with open(self.data_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue
        
        return samples
    
    def _select_teacher_response(self, sample: Dict) -> tuple:
        """Select which teacher's response to use for this sample."""
        # Handle simple format
        if "response" in sample:
            return sample.get("teacher", "unknown"), sample["response"]
        
        # Handle multi-teacher format
        responses = sample.get("responses", {})
        if not responses:
            raise ValueError(f"Sample has no responses: {sample}")
        
        teachers = list(responses.keys())
        
        if self.teacher_strategy == "specific" and self.specific_teacher:
            if self.specific_teacher in responses:
                return self.specific_teacher, responses[self.specific_teacher]
            # Fallback to first available
            return teachers[0], responses[teachers[0]]
        
        elif self.teacher_strategy == "round_robin":
            teacher = teachers[self._rr_index % len(teachers)]
            self._rr_index += 1
            return teacher, responses[teacher]
        
        else:  # random
            teacher = random.choice(teachers)
            return teacher, responses[teacher]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        prompt = sample["prompt"]
        teacher_name, response = self._select_teacher_response(sample)
        
        # Tokenize
        encoded = self.tokenizer.encode_for_distillation(prompt, response)
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": encoded["labels"],
            "teacher_name": teacher_name,
        }


# =============================================================================
# Streaming Dataset (for large files)
# =============================================================================

class StreamingDistillationDataset(IterableDataset):
    """
    Streaming dataset for large JSONL files that don't fit in memory.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: TokenizerWrapper,
        teacher_strategy: str = "random",
        buffer_size: int = 10000,
        shuffle: bool = True,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.teacher_strategy = teacher_strategy
        self.buffer_size = buffer_size
        self.shuffle = shuffle
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        buffer = []
        
        with open(self.data_path, 'r') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    buffer.append(sample)
                    
                    if len(buffer) >= self.buffer_size:
                        if self.shuffle:
                            random.shuffle(buffer)
                        
                        for s in buffer:
                            yield self._process_sample(s)
                        
                        buffer = []
                
                except json.JSONDecodeError:
                    continue
        
        # Process remaining
        if buffer:
            if self.shuffle:
                random.shuffle(buffer)
            for s in buffer:
                yield self._process_sample(s)
    
    def _process_sample(self, sample: Dict) -> Dict[str, Any]:
        """Process a single sample."""
        prompt = sample["prompt"]
        
        # Select response
        if "response" in sample:
            teacher_name = sample.get("teacher", "unknown")
            response = sample["response"]
        else:
            responses = sample.get("responses", {})
            teachers = list(responses.keys())
            teacher_name = random.choice(teachers)
            response = responses[teacher_name]
        
        # Tokenize
        encoded = self.tokenizer.encode_for_distillation(prompt, response)
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": encoded["labels"],
            "teacher_name": teacher_name,
        }


# =============================================================================
# Data Collator
# =============================================================================

class DistillationCollator:
    """
    Collate function for distillation batches.
    """
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, Any]]) -> BatchEncoding:
        """Collate batch of samples."""
        input_ids = torch.stack([s["input_ids"] for s in batch])
        attention_mask = torch.stack([s["attention_mask"] for s in batch])
        labels = torch.stack([s["labels"] for s in batch])
        teacher_names = [s["teacher_name"] for s in batch]
        
        return BatchEncoding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            teacher_names=teacher_names,
        )


# =============================================================================
# DataLoader Factory
# =============================================================================

def create_dataloader(
    data_path: str,
    tokenizer: TokenizerWrapper,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    teacher_strategy: str = "random",
    streaming: bool = False,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """
    Create a DataLoader for distillation training.
    
    Args:
        data_path: Path to JSONL data file
        tokenizer: TokenizerWrapper instance
        batch_size: Batch size
        shuffle: Whether to shuffle (ignored for streaming)
        num_workers: Number of data loading workers
        teacher_strategy: How to select teacher responses
        streaming: Use streaming dataset (for large files)
        max_samples: Maximum samples to load (non-streaming only)
    
    Returns:
        DataLoader instance
    """
    collator = DistillationCollator(pad_token_id=tokenizer.pad_token_id)
    
    if streaming:
        dataset = StreamingDistillationDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            teacher_strategy=teacher_strategy,
            shuffle=shuffle,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collator,
            num_workers=num_workers,
        )
    else:
        dataset = DistillationDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            teacher_strategy=teacher_strategy,
            max_samples=max_samples,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True,
        )


# =============================================================================
# Utility Functions
# =============================================================================

def create_sample_data(
    output_path: str,
    num_samples: int = 100,
    teachers: List[str] = None,
) -> None:
    """
    Create sample JSONL data for testing.
    
    Args:
        output_path: Path to write JSONL file
        num_samples: Number of samples to generate
        teachers: List of teacher names
    """
    teachers = teachers or ["azure-claude", "glm-4.7", "minimax-m2.1"]
    
    prompts = [
        "Explain the concept of machine learning in simple terms.",
        "Write a Python function to calculate fibonacci numbers.",
        "What is the capital of France and what is it known for?",
        "How does photosynthesis work?",
        "Explain the difference between a list and a tuple in Python.",
        "What are the benefits of regular exercise?",
        "Describe the water cycle in detail.",
        "How do neural networks learn?",
        "What is the theory of relativity?",
        "Explain object-oriented programming concepts.",
    ]
    
    with open(output_path, 'w') as f:
        for i in range(num_samples):
            prompt = prompts[i % len(prompts)]
            
            # Generate fake responses for each teacher
            responses = {}
            for teacher in teachers:
                responses[teacher] = f"This is a sample response from {teacher} for prompt {i}. " * 3
            
            sample = {
                "prompt": prompt,
                "responses": responses,
                "metadata": {
                    "sample_id": i,
                    "timestamp": "2025-01-01T00:00:00Z",
                }
            }
            
            f.write(json.dumps(sample) + "\n")
    
    print(f"Created {num_samples} samples at {output_path}")


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    import tempfile
    import os
    
    print("Testing data pipeline...")
    
    # Create temporary sample data
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "test_data.jsonl")
        create_sample_data(data_path, num_samples=20)
        
        print(f"\nCreated test data at {data_path}")
        
        # Test tokenizer (would need transformers installed)
        try:
            tokenizer = TokenizerWrapper(
                tokenizer_name_or_path="gpt2",  # Smaller for testing
                max_length=256,
            )
            print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
            
            # Test dataset
            dataset = DistillationDataset(
                data_path=data_path,
                tokenizer=tokenizer,
                teacher_strategy="random",
            )
            print(f"Dataset size: {len(dataset)}")
            
            # Test single sample
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Input IDs shape: {sample['input_ids'].shape}")
            print(f"Teacher: {sample['teacher_name']}")
            
            # Test dataloader
            dataloader = create_dataloader(
                data_path=data_path,
                tokenizer=tokenizer,
                batch_size=4,
                num_workers=0,
            )
            
            batch = next(iter(dataloader))
            print(f"\nBatch:")
            print(f"  Input IDs: {batch.input_ids.shape}")
            print(f"  Attention mask: {batch.attention_mask.shape}")
            print(f"  Labels: {batch.labels.shape}")
            print(f"  Teachers: {batch.teacher_names}")
            
            print("\nData pipeline test passed!")
            
        except ImportError as e:
            print(f"\nSkipping tokenizer test (transformers not installed): {e}")
            print("Data structure tests passed!")
