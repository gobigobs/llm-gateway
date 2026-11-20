"""llm-gateway — one OpenAI-compatible API in front of local Ollama models.

Request pipeline for POST /v1/chat/completions (mirrors the LiteLLM /
AI-gateway pattern: auth -> rate limit -> cache -> route -> fallback
-> log):

1. **Auth** — ``Authorization: Bearer <key>``; keys live in config.yaml.
2. **Rate limit** — per-key, per-minute sliding window; 429 + Retry-After.
3. **Cache** — exact match on (model + messages + temperature +
   max_tokens) in SQLite; hits never touch Ollama.
4. **Route** — gateway model name -> Ollama model from config.yaml.
5. **Fallback** — primary failure (bad model, 5xx, timeout, Ollama
   down) retries the configured fallback model and flags it.
6. **Log** — every request (hit, miss, fallback, 429, error) is written
   to SQLite with token counts and estimated cost; GET /stats reads it.

Only local, free backends: no paid API keys anywhere in this codebase.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import GatewayConfig, load_config
from .ollama_client import OllamaClient, OllamaError, to_openai_response
from .ratelimit import RateLimiter
from .store import Store
from .tokens import estimate_messages_tokens, estimate_tokens

STATIC_DIR = Path(__file__).resolve().parent / "static"


# ---------------------------------------------------------------- models
class ChatMessage(BaseModel):
    role: str
    content: str | list  # string, or OpenAI content-parts list


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False


# ------------------------------------------------------------------ app
def create_app(config: GatewayConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    store = Store(cfg.db_path)
    limiter = RateLimiter()
    ollama = OllamaClient(cfg.ollama_base_url, cfg.ollama_timeout)

    app = FastAPI(
        title="llm-gateway",
        description=(
            "OpenAI-compatible gateway in front of local Ollama models: "
            "routing, exact-match caching, automatic fallback, per-key rate "
            "limits, cost tracking and a live dashboard. No paid APIs."
        ),
        version="1.0.0",
    )
    app.state.cfg = cfg
    app.state.store = store
    app.state.limiter = limiter
    app.state.ollama = ollama

    # ------------------------------------------------------------ helpers
    def openai_error(status: int, message: str, code: str):
        return JSONResponse(
            status_code=status,
            content={
                "error": {"message": message, "type": "gateway_error", "code": code}
            },
        )

    def require_key(authorization: str | None = Header(default=None)):
        """Resolve a Bearer key to its config entry (401 when missing/bad)."""
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Missing Authorization: Bearer <api-key> header.",
            )
        key = authorization[len("Bearer ") :].strip()
        entry = cfg.keys.get(key)
        if entry is None:
            raise HTTPException(status_code=401, detail="Invalid API key.")
        return {"key": key, **entry}

    def estimated_cost(gateway_model: str, prompt_tokens: int, completion_tokens: int) -> float:
        price = cfg.price_for(gateway_model)
        return (
            prompt_tokens / 1000.0 * float(price.get("prompt", 0.0))
            + completion_tokens / 1000.0 * float(price.get("completion", 0.0))
        )

    # ------------------------------------------------------------- routes
    @app.get("/health")
    async def health():
        """No-auth liveness check; also reports which models Ollama has."""
        models = await ollama.list_models()
        return {
            "status": "ok",
            "ollama_url": cfg.ollama_base_url,
            "ollama_reachable": bool(models),
            "ollama_models": models,
            "gateway_models": [m["id"] for m in cfg.gateway_model_names()],
            "cache_enabled": cfg.cache_enabled,
        }

    @app.get("/")
    async def dashboard():
        """The live dashboard (single self-contained HTML page)."""
        return FileResponse(STATIC_DIR / "dashboard.html", media_type="text/html")

    @app.get("/v1/models")
    async def list_models(key=Depends(require_key)):
        return {"object": "list", "data": cfg.gateway_model_names()}

    @app.get("/stats")
    async def stats(key=Depends(require_key)):
        """Live totals from the SQLite request log (see README for fields)."""
        data = store.stats()
        data["pricing_note"] = (
            "Local Ollama models: ₹0 per 1k tokens. Estimated cost uses the "
            "price table in config.yaml (all ₹0 for local models)."
        )
        data["token_note"] = (
            "Token counts use Ollama's real prompt_eval_count/eval_count when "
            "reported; otherwise a chars/4 estimate (see app/tokens.py)."
        )
        return data

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        started = time.monotonic()
        # --- auth (manual here so we can log 401s the same way) --------
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return openai_error(401, "Missing Authorization: Bearer <api-key> header.", "invalid_api_key")
        key_value = auth_header[len("Bearer ") :].strip()
        key_entry = cfg.keys.get(key_value)
        if key_entry is None:
            return openai_error(401, "Invalid API key.", "invalid_api_key")
        key_name = key_entry["name"]
        rpm = key_entry["requests_per_minute"]

        def log(**kw):
            store.log_request(
                api_key_name=key_name,
                latency_ms=(time.monotonic() - started) * 1000.0,
                **kw,
            )

        # --- request validation ----------------------------------------
        if body.stream:
            return openai_error(
                400,
                "Streaming (stream=true) is not supported by this gateway yet; "
                "send stream=false (the default).",
                "streaming_not_supported",
            )
        if not body.messages:
            return openai_error(400, "messages must be a non-empty list.", "invalid_request")
        messages = [m.model_dump() for m in body.messages]

        resolved = cfg.resolve_model(body.model)
        if resolved is None:
            available = sorted({m["id"] for m in cfg.gateway_model_names()})
            return openai_error(
                404,
                f"Unknown model '{body.model}'. Available gateway models: "
                + ", ".join(available),
                "model_not_found",
            )
        _, primary_ollama_model = resolved

        # --- rate limit (checked before cache: a flood is a flood) -----
        allowed, retry_after = limiter.check(key_value, rpm)
        if not allowed:
            log(
                gateway_model=body.model,
                model_used="",
                rate_limited=True,
                prompt_tokens=estimate_messages_tokens(messages),
            )
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry_after)},
                content={
                    "error": {
                        "message": (
                            f"Rate limit exceeded for key '{key_name}': "
                            f"{rpm} requests/minute. Retry after {retry_after}s."
                        ),
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                    }
                },
            )

        prompt_tokens = estimate_messages_tokens(messages)

        # --- cache ------------------------------------------------------
        cache_key = None
        if cfg.cache_enabled:
            cache_key = store.cache_key(
                body.model, messages, body.temperature, body.max_tokens
            )
            hit = store.cache_get(cache_key, cfg.cache_ttl)
            if hit is not None:
                response = dict(hit["response"])
                response["gateway"] = {
                    **response.get("gateway", {}),
                    "cached": True,
                }
                log(
                    gateway_model=body.model,
                    model_used=response.get("gateway", {}).get("model_used", primary_ollama_model),
                    cached=True,
                    fallback_used=response.get("gateway", {}).get("fallback_used", False),
                    prompt_tokens=hit["prompt_tokens"],
                    completion_tokens=hit["completion_tokens"],
                    estimated_cost_inr=estimated_cost(
                        body.model, hit["prompt_tokens"], hit["completion_tokens"]
                    ),
                )
                return JSONResponse(
                    content=response, headers={"X-Gateway-Cache": "HIT"}
                )

        # --- route with fallback ---------------------------------------
        ollama_body = None
        model_used = primary_ollama_model
        fallback_used = False
        primary_error = ""
        try:
            ollama_body = await ollama.chat(
                primary_ollama_model, messages, body.temperature, body.max_tokens
            )
        except OllamaError as exc:
            primary_error = str(exc)
            fallback_model = cfg.fallback_model()
            if fallback_model != primary_ollama_model or True:
                # Always attempt the configured fallback entry, even when
                # it names the same Ollama model: the entry may differ
                # after a config change, and the attempt itself is what
                # proves the fallback path works.
                try:
                    ollama_body = await ollama.chat(
                        fallback_model, messages, body.temperature, body.max_tokens
                    )
                    model_used = fallback_model
                    fallback_used = True
                except OllamaError as exc2:
                    log(
                        gateway_model=body.model,
                        model_used=primary_ollama_model,
                        error=True,
                        prompt_tokens=prompt_tokens,
                    )
                    return openai_error(
                        502,
                        "Both primary and fallback models failed. "
                        f"Primary: {primary_error} Fallback: {exc2}",
                        "model_unavailable",
                    )
        if ollama_body is None:  # pragma: no cover - defensive
            return openai_error(502, "No response from any model.", "model_unavailable")

        response = to_openai_response(
            ollama_body,
            gateway_model=body.model,
            model_used=model_used,
            prompt_tokens=prompt_tokens,
            cached=False,
            fallback_used=fallback_used,
        )
        usage = response["usage"]
        # The OpenAI "created" field should be a real Unix timestamp.
        response["created"] = int(time.time())
        response["id"] = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        if cfg.cache_enabled and cache_key:
            store.cache_put(
                cache_key, body.model, response, usage["prompt_tokens"], usage["completion_tokens"]
            )
        log(
            gateway_model=body.model,
            model_used=model_used,
            fallback_used=fallback_used,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            estimated_cost_inr=estimated_cost(
                body.model, usage["prompt_tokens"], usage["completion_tokens"]
            ),
        )
        headers = {
            "X-Gateway-Cache": "MISS",
            "X-Gateway-Model-Used": model_used,
        }
        if fallback_used:
            headers["X-Gateway-Fallback"] = f"primary failed ({primary_error})"
        return JSONResponse(content=response, headers=headers)

    @app.on_event("shutdown")
    async def shutdown():
        await ollama.close()

    return app


app = create_app()
