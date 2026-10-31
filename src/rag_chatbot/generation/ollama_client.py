"""Async client wrapping the Ollama HTTP API."""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from rag_chatbot.errors import OllamaError

logger = logging.getLogger(__name__)

Message = dict[str, str]


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        keep_alive: str = "30m",
        num_ctx: int = 4096,
        max_tokens: int = 400,
        temperature: float = 0.1,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
        )
        self.model = model
        self._keep_alive = keep_alive
        self._num_ctx = num_ctx
        self._max_tokens = max_tokens
        self._temperature = temperature

    def _payload(
        self,
        messages: list[Message],
        *,
        stream: bool,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        return {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": self._temperature if temperature is None else temperature,
                "num_ctx": self._num_ctx,
                "num_predict": max_tokens or self._max_tokens,
            },
        }

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload = self._payload(messages, stream=False, model=model, temperature=temperature, max_tokens=max_tokens)
        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise OllamaError(_describe(exc.response)) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama unreachable: {exc!r}") from exc
        return response.json().get("message", {}).get("content", "")

    async def chat_stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        payload = self._payload(messages, stream=True, model=model, temperature=temperature, max_tokens=max_tokens)
        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                if response.is_error:
                    await response.aread()
                    raise OllamaError(_describe(response))
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if "error" in data:
                        raise OllamaError(str(data["error"]))
                    if piece := data.get("message", {}).get("content"):
                        yield piece
                    if data.get("done"):
                        return
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama unreachable: {exc!r}") from exc

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"Cannot list Ollama models: {exc!r}") from exc
        return response.json().get("models", [])

    async def is_healthy(self) -> bool:
        try:
            response = await self._client.get("/api/tags", timeout=3.0)
            return response.is_success
        except httpx.HTTPError:
            return False

    async def warm_up(self) -> None:
        """Load the model into memory so the first user query does not pay the load cost."""
        try:
            await self._client.post(
                "/api/generate", json={"model": self.model, "prompt": "", "keep_alive": self._keep_alive}
            )
        except httpx.HTTPError:
            logger.warning("Ollama warm-up failed; the first request may be slow", exc_info=True)

    async def aclose(self) -> None:
        await self._client.aclose()


def _describe(response: httpx.Response) -> str:
    return f"Ollama returned {response.status_code}: {response.text[:300]}"
