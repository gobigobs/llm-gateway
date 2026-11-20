"""Ollama backend client for llm-gateway.

Talks to the local Ollama server's ``POST /api/chat`` endpoint and
translates between Ollama's chat format and the OpenAI chat-completions
shape the gateway exposes. Raises :class:`OllamaError` on any failure
(HTTP error, missing model, timeout, connection error) so the caller
can decide whether to fall back.
"""
from __future__ import annotations

import httpx

from .tokens import estimate_messages_tokens, estimate_tokens


class OllamaError(Exception):
    """Any failure talking to Ollama (HTTP error, timeout, bad model)."""


class OllamaClient:
    def __init__(self, base_url: str, timeout_seconds: float = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds
        # trust_env=False: this VM's proxy env vars contain IPv6 bracket
        # entries (no_proxy="[::1]" etc.) that httpx's URL parser
        # rejects ("Invalid port: ':1]'"). The Ollama backend is
        # localhost anyway, so no proxy is ever wanted here.
        self._client = httpx.AsyncClient(timeout=timeout_seconds, trust_env=False)

    async def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Call Ollama /api/chat. Returns the raw Ollama JSON body.

        Raises OllamaError with a human-readable message on failure.
        """
        payload: dict = {"model": model, "messages": messages, "stream": False}
        options: dict = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if options:
            payload["options"] = options
        try:
            resp = await self._client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise OllamaError(f"timed out after {self.timeout:g}s: {exc}") from exc
        except httpx.RequestError as exc:
            raise OllamaError(f"cannot reach Ollama at {self.base_url}: {exc}") from exc
        if resp.status_code != 200:
            detail = ""
            try:
                detail = resp.json().get("error", "")
            except Exception:
                detail = resp.text[:200]
            raise OllamaError(f"Ollama HTTP {resp.status_code}: {detail or resp.text[:200]}")
        try:
            return resp.json()
        except ValueError as exc:
            raise OllamaError(f"Ollama returned non-JSON: {resp.text[:200]}") from exc

    async def list_models(self) -> list[str]:
        """Names of models Ollama currently has downloaded."""
        try:
            resp = await self._client.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            return [m.get("name", "") for m in resp.json().get("models", [])]
        except Exception:
            return []

    async def close(self) -> None:
        await self._client.aclose()


def to_openai_response(
    ollama_body: dict,
    gateway_model: str,
    model_used: str,
    prompt_tokens: int,
    cached: bool = False,
    fallback_used: bool = False,
) -> dict:
    """Convert an Ollama chat body into an OpenAI chat-completion body.

    Token counts prefer Ollama's real counts (``prompt_eval_count`` /
    ``eval_count``) and fall back to the chars/4 estimate. The extra
    ``gateway`` object is a non-standard extension — OpenAI clients
    ignore unknown top-level fields — and is how callers can see
    cache/fallback behaviour without parsing headers.
    """
    message = ollama_body.get("message", {}) or {}
    content = message.get("content", "")
    real_prompt = ollama_body.get("prompt_eval_count")
    real_completion = ollama_body.get("eval_count")
    completion_tokens = (
        int(real_completion) if real_completion is not None else estimate_tokens(content)
    )
    if real_prompt is None:
        # prompt_tokens was estimated from the request before the call.
        pass
    else:
        prompt_tokens = int(real_prompt)
    return {
        "id": f"chatcmpl-{ollama_body.get('created_at', 'local')}",
        "object": "chat.completion",
        "created": 0,
        "model": gateway_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": ollama_body.get("done_reason", "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "gateway": {
            "cached": cached,
            "fallback_used": fallback_used,
            "model_used": model_used,
            "token_counts_estimated": real_prompt is None or real_completion is None,
        },
    }
