"""
OpenCode Server HTTP Client.

Client for interacting with OpenCode server to query teacher models
for knowledge distillation data generation.

API Reference: https://opencode.ai/docs

Copyright (c) 2025 Distillix. All Rights Reserved.
"""

import asyncio
import json
import time
from typing import Optional, Dict, List, Any, AsyncIterator
from dataclasses import dataclass, field
import logging

try:
    import httpx
except ImportError:
    httpx = None

try:
    import aiohttp
except ImportError:
    aiohttp = None


logger = logging.getLogger("distillix.foundry")


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Message:
    """Message in a session."""
    role: str  # "user" or "assistant"
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Session:
    """OpenCode session."""
    id: str
    title: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class TeacherResponse:
    """Response from a teacher model."""
    model: str
    content: str
    tokens_used: int = 0
    latency_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# OpenCode HTTP Client
# =============================================================================

class OpenCodeClient:
    """
    HTTP client for OpenCode server API.
    
    Supports both sync (httpx) and async (aiohttp) operations.
    
    Usage:
        client = OpenCodeClient(host="127.0.0.1", port=4096)
        
        # Check health
        if await client.health_check():
            # Create session and query model
            session = await client.create_session()
            response = await client.send_message(
                session_id=session.id,
                prompt="Explain machine learning",
                model="azure/claude-sonnet-4-5"
            )
            print(response.content)
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4096,
        timeout: float = 120.0,
    ):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        
        # Validate we have a client library
        if httpx is None and aiohttp is None:
            raise ImportError(
                "Either httpx or aiohttp is required. "
                "Install with: pip install httpx aiohttp"
            )
    
    # =========================================================================
    # Health & Status
    # =========================================================================
    
    async def health_check(self) -> bool:
        """
        Check if OpenCode server is healthy.
        
        GET /global/health
        
        Returns:
            True if server is healthy
        """
        try:
            response = await self._get("/global/health")
            return response.get("healthy", False)
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    async def get_providers(self) -> Dict[str, Any]:
        """
        Get available providers and models.
        
        GET /provider
        
        Returns:
            Provider information
        """
        return await self._get("/provider")
    
    async def list_models(self) -> List[str]:
        """
        List all available models.
        
        Returns:
            List of model IDs (e.g., ["azure/claude-sonnet-4-5", ...])
        """
        providers = await self.get_providers()
        models = []
        
        for provider in providers.get("all", []):
            for model in provider.get("models", []):
                model_id = f"{provider['id']}/{model['id']}"
                models.append(model_id)
        
        return models
    
    # =========================================================================
    # Session Management
    # =========================================================================
    
    async def create_session(self, title: Optional[str] = None) -> Session:
        """
        Create a new session.
        
        POST /session
        
        Args:
            title: Optional session title
        
        Returns:
            Session object
        """
        body = {}
        if title:
            body["title"] = title
        
        response = await self._post("/session", body)
        
        return Session(
            id=response["id"],
            title=response.get("title"),
            created_at=response.get("createdAt"),
        )
    
    async def get_session(self, session_id: str) -> Session:
        """
        Get session details.
        
        GET /session/:id
        """
        response = await self._get(f"/session/{session_id}")
        
        return Session(
            id=response["id"],
            title=response.get("title"),
            created_at=response.get("createdAt"),
        )
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        DELETE /session/:id
        """
        await self._delete(f"/session/{session_id}")
        return True
    
    async def list_sessions(self) -> List[Session]:
        """
        List all sessions.
        
        GET /session
        """
        response = await self._get("/session")
        
        return [
            Session(
                id=s["id"],
                title=s.get("title"),
                created_at=s.get("createdAt"),
            )
            for s in response
        ]
    
    # =========================================================================
    # Message API
    # =========================================================================
    
    async def send_message(
        self,
        session_id: str,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> TeacherResponse:
        """
        Send a message and wait for response.
        
        POST /session/:id/message
        
        Args:
            session_id: Session ID
            prompt: User prompt
            model: Model to use (e.g., "azure/claude-sonnet-4-5")
            system_prompt: Optional system prompt
        
        Returns:
            TeacherResponse with model output
        """
        start_time = time.time()
        
        body = {
            "parts": [
                {"type": "text", "text": prompt}
            ]
        }
        
        if model:
            body["model"] = model
        
        if system_prompt:
            body["system"] = system_prompt
        
        response = await self._post(
            f"/session/{session_id}/message",
            body,
            timeout=self.timeout,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract text from response parts
        content = ""
        tokens_used = 0
        
        parts = response.get("parts", [])
        for part in parts:
            if part.get("type") == "text":
                content += part.get("text", "")
        
        # Try to get token usage from metadata
        info = response.get("info", {})
        tokens_used = info.get("tokens", 0)
        
        return TeacherResponse(
            model=model or "default",
            content=content,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            metadata={
                "message_id": info.get("id"),
                "session_id": session_id,
            },
        )
    
    async def send_message_async(
        self,
        session_id: str,
        prompt: str,
        model: Optional[str] = None,
    ) -> None:
        """
        Send a message without waiting for response.
        
        POST /session/:id/prompt_async
        
        Useful for fire-and-forget requests.
        """
        body = {
            "parts": [
                {"type": "text", "text": prompt}
            ]
        }
        
        if model:
            body["model"] = model
        
        await self._post(f"/session/{session_id}/prompt_async", body)
    
    # =========================================================================
    # Events (Server-Sent Events)
    # =========================================================================
    
    async def subscribe_events(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Subscribe to server events via SSE.
        
        GET /event
        
        Yields:
            Event dictionaries
        """
        if aiohttp is None:
            raise ImportError("aiohttp required for SSE: pip install aiohttp")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/event",
                headers={"Accept": "text/event-stream"},
            ) as response:
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            continue
    
    # =========================================================================
    # HTTP Helpers
    # =========================================================================
    
    async def _get(self, path: str) -> Any:
        """Make GET request."""
        if httpx is not None:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}{path}")
                response.raise_for_status()
                return response.json()
        else:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}{path}") as response:
                    response.raise_for_status()
                    return await response.json()
    
    async def _post(
        self,
        path: str,
        body: Dict[str, Any] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Make POST request."""
        timeout = timeout or self.timeout
        
        if httpx is not None:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}{path}",
                    json=body or {},
                )
                response.raise_for_status()
                
                # Handle empty response
                if response.status_code == 204:
                    return {}
                
                return response.json()
        else:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}{path}",
                    json=body or {},
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    response.raise_for_status()
                    
                    if response.status == 204:
                        return {}
                    
                    return await response.json()
    
    async def _delete(self, path: str) -> Any:
        """Make DELETE request."""
        if httpx is not None:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.delete(f"{self.base_url}{path}")
                response.raise_for_status()
                
                if response.status_code == 204:
                    return {}
                
                return response.json()
        else:
            async with aiohttp.ClientSession() as session:
                async with session.delete(f"{self.base_url}{path}") as response:
                    response.raise_for_status()
                    
                    if response.status == 204:
                        return {}
                    
                    return await response.json()


# =============================================================================
# Synchronous Wrapper
# =============================================================================

class OpenCodeClientSync:
    """
    Synchronous wrapper for OpenCodeClient.
    
    Usage:
        client = OpenCodeClientSync(port=4096)
        response = client.query("Explain ML", model="azure/claude-sonnet-4-5")
    """
    
    def __init__(self, **kwargs):
        self._async_client = OpenCodeClient(**kwargs)
        self._loop = None
    
    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
            return self._loop
    
    def _run(self, coro):
        """Run coroutine synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(coro)
    
    def health_check(self) -> bool:
        return self._run(self._async_client.health_check())
    
    def create_session(self, title: Optional[str] = None) -> Session:
        return self._run(self._async_client.create_session(title))
    
    def send_message(
        self,
        session_id: str,
        prompt: str,
        model: Optional[str] = None,
    ) -> TeacherResponse:
        return self._run(
            self._async_client.send_message(session_id, prompt, model)
        )
    
    def query(
        self,
        prompt: str,
        model: Optional[str] = None,
        session_title: str = "distillation",
    ) -> TeacherResponse:
        """
        One-shot query: create session, send message, return response.
        
        Args:
            prompt: User prompt
            model: Model to query
            session_title: Session title
        
        Returns:
            TeacherResponse
        """
        session = self.create_session(session_title)
        return self.send_message(session.id, prompt, model)


# =============================================================================
# Testing
# =============================================================================

async def test_client():
    """Test OpenCode client."""
    client = OpenCodeClient(port=4096)
    
    print("Testing OpenCode client...")
    
    # Health check
    healthy = await client.health_check()
    print(f"Server healthy: {healthy}")
    
    if not healthy:
        print("Server not running. Start with: opencode serve --port 4096")
        return
    
    # List models
    try:
        models = await client.list_models()
        print(f"Available models: {len(models)}")
        for m in models[:5]:
            print(f"  - {m}")
    except Exception as e:
        print(f"Failed to list models: {e}")
    
    # Create session
    session = await client.create_session("test-session")
    print(f"Created session: {session.id}")
    
    # Send test message
    print("Sending test message...")
    response = await client.send_message(
        session_id=session.id,
        prompt="Say 'Hello from Distillix!' and nothing else.",
        model="opencode/glm-4.7-free",  # Using free tier for test
    )
    
    print(f"Response: {response.content[:100]}...")
    print(f"Tokens: {response.tokens_used}")
    print(f"Latency: {response.latency_ms:.0f}ms")
    
    # Cleanup
    await client.delete_session(session.id)
    print("Session deleted")
    
    print("\nClient test passed!")


if __name__ == "__main__":
    asyncio.run(test_client())
