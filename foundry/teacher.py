"""
Multi-Teacher Ensemble for Knowledge Distillation.

Manages querying multiple teacher models through OpenCode server
and aggregating their responses for training data generation.

Teachers:
  1. Azure AI Foundry (Anthropic Claude)
  2. ZAI Coding Plan (GLM 4.7)
  3. MiniMax (M2.1)

Copyright (c) 2025 Distillix. All Rights Reserved.
"""

import asyncio
import logging
import random
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .opencode_client import OpenCodeClient, TeacherResponse, Session


logger = logging.getLogger("distillix.foundry")


# =============================================================================
# Teacher Configuration
# =============================================================================

class TeacherModel(Enum):
    """Available teacher models."""
    
    # Azure AI Foundry - Anthropic Claude
    AZURE_CLAUDE = "azure/claude-sonnet-4-5"
    
    # ZAI Coding Plan - GLM 4.7
    GLM_47 = "zai-coding-plan/glm-4.7"
    
    # MiniMax M2.1
    MINIMAX_M21 = "minimax/MiniMax-M2.1"
    
    @classmethod
    def all(cls) -> List[str]:
        return [m.value for m in cls]


# Default teacher ensemble
DEFAULT_TEACHERS = [
    TeacherModel.AZURE_CLAUDE.value,
    TeacherModel.GLM_47.value,
    TeacherModel.MINIMAX_M21.value,
]


@dataclass
class TeacherConfig:
    """Configuration for a single teacher."""
    model_id: str
    weight: float = 1.0  # Sampling weight
    max_retries: int = 3
    timeout: float = 120.0
    
    # Rate limiting
    requests_per_minute: Optional[int] = None
    
    # Quality threshold (skip responses below this score)
    min_quality_score: float = 0.0


@dataclass
class EnsembleConfig:
    """Configuration for teacher ensemble."""
    teachers: List[TeacherConfig] = field(default_factory=list)
    
    # Sampling strategy
    strategy: str = "all"  # "all", "random", "weighted", "round_robin"
    
    # Aggregation
    aggregate_method: str = "keep_all"  # "keep_all", "vote", "best"
    
    # System prompt for all teachers
    system_prompt: Optional[str] = None
    
    @classmethod
    def default(cls) -> "EnsembleConfig":
        """Create default configuration with all teachers."""
        return cls(
            teachers=[
                TeacherConfig(TeacherModel.AZURE_CLAUDE.value, weight=1.0),
                TeacherConfig(TeacherModel.GLM_47.value, weight=1.0),
                TeacherConfig(TeacherModel.MINIMAX_M21.value, weight=1.0),
            ],
            strategy="all",
            system_prompt=(
                "You are a helpful AI assistant. Provide clear, accurate, "
                "and well-reasoned responses. Think step by step when solving problems."
            ),
        )


# =============================================================================
# Ensemble Response
# =============================================================================

@dataclass
class EnsembleResponse:
    """Response from teacher ensemble."""
    prompt: str
    responses: Dict[str, TeacherResponse]
    
    # Aggregated response (if applicable)
    selected_response: Optional[TeacherResponse] = None
    
    # Metadata
    total_tokens: int = 0
    total_latency_ms: float = 0
    errors: Dict[str, str] = field(default_factory=dict)
    
    def to_training_format(self) -> Dict[str, Any]:
        """Convert to training data format."""
        return {
            "prompt": self.prompt,
            "responses": {
                name: resp.content
                for name, resp in self.responses.items()
            },
            "metadata": {
                "total_tokens": self.total_tokens,
                "total_latency_ms": self.total_latency_ms,
                "teachers_used": list(self.responses.keys()),
                "errors": self.errors,
            },
        }


# =============================================================================
# Teacher Ensemble
# =============================================================================

class TeacherEnsemble:
    """
    Multi-teacher ensemble for knowledge distillation.
    
    Queries multiple teacher models and aggregates their responses.
    
    Usage:
        ensemble = TeacherEnsemble()
        await ensemble.initialize()
        
        response = await ensemble.query("Explain machine learning")
        print(response.responses["azure/claude-sonnet-4-5"].content)
    """
    
    def __init__(
        self,
        config: Optional[EnsembleConfig] = None,
        opencode_host: str = "127.0.0.1",
        opencode_port: int = 4096,
    ):
        self.config = config or EnsembleConfig.default()
        
        # OpenCode client
        self.client = OpenCodeClient(
            host=opencode_host,
            port=opencode_port,
        )
        
        # Session pool (one per teacher to avoid context mixing)
        self.sessions: Dict[str, Session] = {}
        
        # Round-robin state
        self._rr_index = 0
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_tokens": 0,
        }
    
    async def initialize(self):
        """Initialize sessions for all teachers."""
        logger.info("Initializing teacher ensemble...")
        
        # Check server health
        healthy = await self.client.health_check()
        if not healthy:
            raise ConnectionError(
                "OpenCode server not available. "
                "Start with: opencode serve --port 4096"
            )
        
        # Create sessions for each teacher
        for teacher_config in self.config.teachers:
            model_id = teacher_config.model_id
            try:
                session = await self.client.create_session(
                    title=f"distillix-{model_id.replace('/', '-')}"
                )
                self.sessions[model_id] = session
                logger.info(f"Created session for {model_id}: {session.id}")
            except Exception as e:
                logger.error(f"Failed to create session for {model_id}: {e}")
        
        logger.info(f"Initialized {len(self.sessions)} teacher sessions")
    
    async def cleanup(self):
        """Clean up sessions."""
        for model_id, session in self.sessions.items():
            try:
                await self.client.delete_session(session.id)
                logger.debug(f"Deleted session for {model_id}")
            except Exception as e:
                logger.warning(f"Failed to delete session for {model_id}: {e}")
        
        self.sessions.clear()
    
    def _select_teachers(self) -> List[TeacherConfig]:
        """Select teachers based on strategy."""
        teachers = self.config.teachers
        
        if self.config.strategy == "all":
            return teachers
        
        elif self.config.strategy == "random":
            # Random single teacher
            weights = [t.weight for t in teachers]
            return [random.choices(teachers, weights=weights, k=1)[0]]
        
        elif self.config.strategy == "weighted":
            # Weighted random selection
            weights = [t.weight for t in teachers]
            # Select multiple based on weights
            num_select = min(2, len(teachers))
            selected = random.choices(teachers, weights=weights, k=num_select)
            return list(set(selected))  # Remove duplicates
        
        elif self.config.strategy == "round_robin":
            # Round robin single teacher
            teacher = teachers[self._rr_index % len(teachers)]
            self._rr_index += 1
            return [teacher]
        
        else:
            return teachers
    
    async def _query_single_teacher(
        self,
        teacher_config: TeacherConfig,
        prompt: str,
    ) -> Optional[TeacherResponse]:
        """Query a single teacher with retries."""
        model_id = teacher_config.model_id
        
        session = self.sessions.get(model_id)
        if session is None:
            logger.warning(f"No session for {model_id}")
            return None
        
        for attempt in range(teacher_config.max_retries):
            try:
                response = await self.client.send_message(
                    session_id=session.id,
                    prompt=prompt,
                    model=model_id,
                    system_prompt=self.config.system_prompt,
                )
                
                self.stats["successful_queries"] += 1
                self.stats["total_tokens"] += response.tokens_used
                
                return response
                
            except Exception as e:
                logger.warning(
                    f"Query failed for {model_id} (attempt {attempt + 1}): {e}"
                )
                
                if attempt < teacher_config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        self.stats["failed_queries"] += 1
        return None
    
    async def query(
        self,
        prompt: str,
        teachers: Optional[List[str]] = None,
    ) -> EnsembleResponse:
        """
        Query teacher ensemble.
        
        Args:
            prompt: User prompt
            teachers: Optional specific teachers to query
        
        Returns:
            EnsembleResponse with all teacher outputs
        """
        self.stats["total_queries"] += 1
        
        # Select teachers
        if teachers:
            selected = [
                tc for tc in self.config.teachers
                if tc.model_id in teachers
            ]
        else:
            selected = self._select_teachers()
        
        # Query all selected teachers concurrently
        tasks = [
            self._query_single_teacher(tc, prompt)
            for tc in selected
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect responses and errors
        responses = {}
        errors = {}
        total_tokens = 0
        total_latency = 0
        
        for teacher_config, result in zip(selected, results):
            model_id = teacher_config.model_id
            
            if isinstance(result, Exception):
                errors[model_id] = str(result)
            elif result is None:
                errors[model_id] = "No response"
            else:
                responses[model_id] = result
                total_tokens += result.tokens_used
                total_latency += result.latency_ms
        
        # Create ensemble response
        ensemble_response = EnsembleResponse(
            prompt=prompt,
            responses=responses,
            total_tokens=total_tokens,
            total_latency_ms=total_latency,
            errors=errors,
        )
        
        # Aggregate if needed
        if self.config.aggregate_method == "best" and responses:
            # Select response with most tokens (proxy for detail)
            best = max(responses.values(), key=lambda r: len(r.content))
            ensemble_response.selected_response = best
        
        return ensemble_response
    
    async def query_batch(
        self,
        prompts: List[str],
        max_concurrent: int = 5,
    ) -> List[EnsembleResponse]:
        """
        Query ensemble with multiple prompts.
        
        Args:
            prompts: List of prompts
            max_concurrent: Maximum concurrent requests
        
        Returns:
            List of EnsembleResponses
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def query_with_limit(prompt: str) -> EnsembleResponse:
            async with semaphore:
                return await self.query(prompt)
        
        tasks = [query_with_limit(p) for p in prompts]
        return await asyncio.gather(*tasks)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful_queries"] / 
                max(self.stats["total_queries"], 1)
            ),
            "avg_tokens_per_query": (
                self.stats["total_tokens"] /
                max(self.stats["successful_queries"], 1)
            ),
        }


# =============================================================================
# Context Manager
# =============================================================================

class TeacherEnsembleContext:
    """
    Context manager for teacher ensemble.
    
    Usage:
        async with TeacherEnsembleContext() as ensemble:
            response = await ensemble.query("Explain ML")
    """
    
    def __init__(self, **kwargs):
        self.ensemble = TeacherEnsemble(**kwargs)
    
    async def __aenter__(self) -> TeacherEnsemble:
        await self.ensemble.initialize()
        return self.ensemble
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.ensemble.cleanup()


# =============================================================================
# Testing
# =============================================================================

async def test_ensemble():
    """Test teacher ensemble."""
    print("Testing teacher ensemble...")
    
    async with TeacherEnsembleContext(opencode_port=4096) as ensemble:
        # Single query
        print("\nQuerying all teachers...")
        response = await ensemble.query("What is 2 + 2? Answer with just the number.")
        
        print(f"Prompt: {response.prompt}")
        print(f"Responses received: {len(response.responses)}")
        
        for model, resp in response.responses.items():
            print(f"\n{model}:")
            print(f"  Content: {resp.content[:100]}...")
            print(f"  Tokens: {resp.tokens_used}")
            print(f"  Latency: {resp.latency_ms:.0f}ms")
        
        if response.errors:
            print(f"\nErrors: {response.errors}")
        
        # Stats
        print(f"\nEnsemble stats: {ensemble.get_stats()}")
        
        # Training format
        print(f"\nTraining format:")
        print(response.to_training_format())
    
    print("\nEnsemble test passed!")


if __name__ == "__main__":
    asyncio.run(test_ensemble())
