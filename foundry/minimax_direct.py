"""
Direct MiniMax API Client for High-Throughput Data Generation.

Uses Anthropic SDK with MiniMax's compatible endpoint for maximum velocity.

Throughput: ~500 RPM, 100 TPS (lightning model)

Usage:
    python -m foundry.minimax_direct --count 1000 --output data/train.jsonl

Copyright (c) 2025 Distillix. All Rights Reserved.
"""

import anthropic
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import sys

# MiniMax API Configuration
MINIMAX_BASE_URL = "https://api.minimax.io/anthropic"
MINIMAX_API_KEY = "sk-api-vli90ZHzgEZxnG_JVzeDZD_1-nOQiI2lhb1PKghH1yotqk6RE05maodQD4ogYcmb19aV-DDI-LkhuTJHRPqfcFXTk_L5f4MOq3OsLDys0HNZpktQjmMB8go"

# Models
MODEL_STANDARD = "MiniMax-M2.1"  # 60 TPS, $1.2/M output
MODEL_LIGHTNING = "MiniMax-M2.1-lightning"  # 100 TPS, $2.4/M output

# Rate limiting
MAX_CONCURRENT = 50  # Stay under 500 RPM with safety margin
RPM_LIMIT = 450  # Requests per minute limit


@dataclass
class GenerationResult:
    """Result of a single generation."""
    prompt: str
    response: str
    thinking: Optional[str]
    tokens_in: int
    tokens_out: int
    latency_ms: float
    success: bool
    error: Optional[str] = None


# Phi Protocol Templates
PHI_CODE_TEMPLATE = """Write a comprehensive guide for: {topic}

Include:
1. Concept explanation (2-3 sentences)
2. How it works step-by-step
3. Python code example with comments
4. Common mistakes and how to avoid them
5. When to use this approach

Write for an intermediate programmer."""

PHI_TOPICS = [
    # ===========================================
    # RUST
    # ===========================================
    "Rust ownership and borrowing",
    "Rust lifetimes and references",
    "Rust Result and Option types for error handling",
    "Rust traits and generics",
    "Rust async/await with Tokio",
    "Rust pattern matching and enums",
    "Rust smart pointers (Box, Rc, Arc)",
    "Rust closures and iterators",
    "Rust modules and crates organization",
    "Rust unsafe code and FFI",
    "Rust memory safety without garbage collection",
    "Rust concurrency with channels (mpsc)",
    "Rust Mutex and RwLock for shared state",
    "Rust macros (declarative and procedural)",
    "Rust error handling with thiserror and anyhow",
    "Rust serde for serialization/deserialization",
    "Rust web frameworks (Actix, Axum)",
    "Rust CLI tools with clap",
    "Rust testing and benchmarking",
    "Rust WebAssembly compilation",
    
    # ===========================================
    # MACHINE LEARNING
    # ===========================================
    "Neural network backpropagation from scratch",
    "PyTorch tensor operations and autograd",
    "Training loops and optimization in PyTorch",
    "Transformer architecture explained",
    "Attention mechanism implementation",
    "BERT and GPT architecture differences",
    "Fine-tuning pretrained models",
    "LoRA and parameter-efficient fine-tuning",
    "Quantization techniques (INT8, INT4)",
    "Knowledge distillation for model compression",
    "Gradient checkpointing for memory efficiency",
    "Mixed precision training (FP16, BF16)",
    "Distributed training with PyTorch DDP",
    "Hyperparameter tuning strategies",
    "Learning rate schedulers",
    "Batch normalization vs Layer normalization",
    "Dropout and regularization techniques",
    "Cross-entropy loss and its variants",
    "Embedding layers and word vectors",
    "Tokenization (BPE, WordPiece, SentencePiece)",
    
    # ===========================================
    # KUBERNETES
    # ===========================================
    "Kubernetes Pods and containers",
    "Kubernetes Deployments and ReplicaSets",
    "Kubernetes Services (ClusterIP, NodePort, LoadBalancer)",
    "Kubernetes Ingress and traffic routing",
    "Kubernetes ConfigMaps and Secrets",
    "Kubernetes Persistent Volumes and Claims",
    "Kubernetes StatefulSets for stateful apps",
    "Kubernetes DaemonSets and Jobs",
    "Kubernetes RBAC and security",
    "Kubernetes Helm charts",
    "Kubernetes operators pattern",
    "Kubernetes horizontal pod autoscaling",
    "Kubernetes resource limits and requests",
    "Kubernetes networking (CNI, Calico, Cilium)",
    "Kubernetes service mesh (Istio basics)",
    "Kubernetes logging and monitoring (Prometheus, Grafana)",
    "Kubernetes debugging and troubleshooting",
    "Kubernetes multi-cluster management",
    "Kubernetes GitOps with ArgoCD",
    "Kubernetes cost optimization",
    
    # ===========================================
    # PROXMOX
    # ===========================================
    "Proxmox VE installation and setup",
    "Proxmox VM creation and management",
    "Proxmox LXC containers vs VMs",
    "Proxmox storage configuration (ZFS, LVM, Ceph)",
    "Proxmox networking (bridges, VLANs, bonds)",
    "Proxmox clustering and high availability",
    "Proxmox backup and restore strategies",
    "Proxmox templates and cloning",
    "Proxmox GPU passthrough",
    "Proxmox API and automation",
    "Proxmox firewall configuration",
    "Proxmox resource pools and permissions",
    "Proxmox live migration",
    "Proxmox Ceph integration for distributed storage",
    "Proxmox cloud-init for VM automation",
    "Proxmox monitoring and alerts",
    "Proxmox performance tuning",
    "Proxmox disaster recovery planning",
    "Proxmox vs ESXi comparison",
    "Proxmox homelab best practices",
    
    # ===========================================
    # UI / FRONTEND
    # ===========================================
    "React component lifecycle and hooks",
    "React state management (useState, useReducer)",
    "React context API for global state",
    "React performance optimization (memo, useMemo, useCallback)",
    "React Server Components and Next.js App Router",
    "TypeScript with React best practices",
    "CSS-in-JS (styled-components, Emotion)",
    "Tailwind CSS utility-first approach",
    "Responsive design patterns",
    "Accessibility (a11y) in web applications",
    "Form handling and validation",
    "Data fetching patterns (SWR, React Query)",
    "Authentication flows in SPAs",
    "WebSocket real-time updates",
    "Progressive Web Apps (PWA)",
    "Browser DevTools debugging",
    "Frontend testing (Jest, React Testing Library)",
    "E2E testing with Playwright",
    "Web performance metrics (Core Web Vitals)",
    "Design systems and component libraries",
]


class MiniMaxGenerator:
    """High-throughput generator using MiniMax API directly."""
    
    def __init__(
        self,
        model: str = MODEL_LIGHTNING,
        max_concurrent: int = MAX_CONCURRENT,
    ):
        self.client = anthropic.Anthropic(
            base_url=MINIMAX_BASE_URL,
            api_key=MINIMAX_API_KEY,
        )
        self.model = model
        self.semaphore = Semaphore(max_concurrent)
        
        # Stats
        self.total_requests = 0
        self.successful = 0
        self.failed = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
    
    def generate_one(self, prompt: str) -> GenerationResult:
        """Generate a single response."""
        start_time = time.time()
        
        with self.semaphore:
            self.total_requests += 1
            
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Extract response
                response_text = ""
                thinking_text = None
                
                for block in message.content:
                    if hasattr(block, 'text'):
                        response_text += block.text
                    elif hasattr(block, 'thinking'):
                        thinking_text = block.thinking
                
                self.successful += 1
                self.total_tokens_in += message.usage.input_tokens
                self.total_tokens_out += message.usage.output_tokens
                
                return GenerationResult(
                    prompt=prompt,
                    response=response_text,
                    thinking=thinking_text,
                    tokens_in=message.usage.input_tokens,
                    tokens_out=message.usage.output_tokens,
                    latency_ms=latency_ms,
                    success=True,
                )
                
            except Exception as e:
                self.failed += 1
                return GenerationResult(
                    prompt=prompt,
                    response="",
                    thinking=None,
                    tokens_in=0,
                    tokens_out=0,
                    latency_ms=(time.time() - start_time) * 1000,
                    success=False,
                    error=str(e),
                )
    
    def generate_batch(
        self,
        prompts: List[str],
        output_path: str,
        max_workers: int = 50,
    ) -> List[GenerationResult]:
        """Generate responses for multiple prompts in parallel."""
        
        print(f"Starting generation...")
        print(f"  Model: {self.model}")
        print(f"  Prompts: {len(prompts)}")
        print(f"  Workers: {max_workers}")
        print(f"  Output: {output_path}")
        print()
        
        results = []
        start_time = time.time()
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_prompt = {
                    executor.submit(self.generate_one, prompt): prompt
                    for prompt in prompts
                }
                
                # Process results as they complete
                for i, future in enumerate(as_completed(future_to_prompt)):
                    result = future.result()
                    results.append(result)
                    
                    if result.success:
                        # Write to file immediately
                        sample = {
                            "prompt": result.prompt,
                            "response": result.response,
                            "metadata": {
                                "tokens_in": result.tokens_in,
                                "tokens_out": result.tokens_out,
                                "latency_ms": result.latency_ms,
                                "model": self.model,
                            }
                        }
                        if result.thinking:
                            sample["thinking"] = result.thinking
                        
                        f.write(json.dumps(sample) + "\n")
                        f.flush()
                    
                    # Progress update
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed * 60  # per minute
                    print(f"\r[{i+1}/{len(prompts)}] "
                          f"OK:{self.successful} FAIL:{self.failed} "
                          f"Rate:{rate:.1f}/min "
                          f"Tokens:{self.total_tokens_out:,}", end="")
        
        print()
        elapsed = time.time() - start_time
        
        print()
        print("=" * 60)
        print("GENERATION COMPLETE")
        print("=" * 60)
        print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Successful: {self.successful}")
        print(f"  Failed: {self.failed}")
        print(f"  Tokens in: {self.total_tokens_in:,}")
        print(f"  Tokens out: {self.total_tokens_out:,}")
        print(f"  Est. cost: ${self.total_tokens_out / 1_000_000 * 2.4:.2f} (lightning)")
        print(f"  Output: {output_path}")
        print("=" * 60)
        
        return results


def generate_prompts(count: int) -> List[str]:
    """Generate prompts using Phi Protocol template."""
    prompts = []
    
    # Cycle through topics
    for i in range(count):
        topic = PHI_TOPICS[i % len(PHI_TOPICS)]
        prompt = PHI_CODE_TEMPLATE.format(topic=topic)
        prompts.append(prompt)
    
    return prompts


def main():
    parser = argparse.ArgumentParser(description="High-throughput MiniMax data generation")
    parser.add_argument("--count", "-n", type=int, default=100, help="Number of samples")
    parser.add_argument("--output", "-o", type=str, default="data/distillation/train.jsonl")
    parser.add_argument("--model", "-m", type=str, default="lightning", 
                        choices=["standard", "lightning"])
    parser.add_argument("--workers", "-w", type=int, default=50)
    args = parser.parse_args()
    
    model = MODEL_LIGHTNING if args.model == "lightning" else MODEL_STANDARD
    
    print("=" * 60)
    print("DISTILLIX HIGH-THROUGHPUT DATA GENERATION")
    print("=" * 60)
    
    # Generate prompts
    prompts = generate_prompts(args.count)
    
    # Create generator
    generator = MiniMaxGenerator(model=model, max_concurrent=args.workers)
    
    # Generate
    generator.generate_batch(prompts, args.output, max_workers=args.workers)


if __name__ == "__main__":
    main()
