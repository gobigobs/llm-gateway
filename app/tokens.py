"""Token counting for llm-gateway.

Ollama's /api/chat returns real counts (``prompt_eval_count`` /
``eval_count``) when the model reports them, and the gateway uses those
whenever they are present. When they are absent (or for the prompt before
the call is made) we fall back to the standard rough heuristic used
across the industry for English text: ~4 characters per token
(OpenAI's own rule of thumb is ~3/4 of a word per token, i.e. ~4
chars/token). Estimates are labelled as estimates in /stats.
"""
from __future__ import annotations

CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    """Rough token estimate for a piece of text (chars / 4, min 1)."""
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate prompt tokens for a chat-completions message list."""
    total = 0
    for m in messages:
        # Small per-message overhead, matching the ~3 tokens/message
        # overhead OpenAI documents for chat formatting.
        total += 3
        content = m.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):  # content parts
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += estimate_tokens(part["text"])
        total += estimate_tokens(str(m.get("role", "")))
    return total
