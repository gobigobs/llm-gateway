"""Configuration loading for llm-gateway.

One YAML file (default ``config.yaml`` next to the project root) drives
everything. ``LLMGW_CONFIG`` may point at a different file — the
verification script uses this to test fallback and rate limiting with
throwaway configs without touching the real one.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


class GatewayConfig:
    def __init__(self, data: dict, source: Path):
        self.source = source
        self.raw = data
        self.host: str = data.get("server", {}).get("host", "127.0.0.1")
        self.port: int = int(data.get("server", {}).get("port", 8000))
        self.ollama_base_url: str = data.get("ollama", {}).get(
            "base_url", "http://localhost:11434"
        )
        self.ollama_timeout: float = float(
            data.get("ollama", {}).get("timeout_seconds", 300)
        )
        self.models: dict = data.get("models", {})
        self.cache_enabled: bool = bool(data.get("cache", {}).get("enabled", True))
        self.cache_ttl: int = int(data.get("cache", {}).get("ttl_seconds", 3600))
        self.default_rpm: int = int(
            data.get("rate_limit", {}).get("default_requests_per_minute", 30)
        )
        self.pricing: dict = data.get("pricing_inr_per_1k_tokens", {}) or {}
        self.db_path: Path = self._resolve_db_path(
            data.get("storage", {}).get("db_path", "data/gateway.db")
        )
        # API keys: name -> {key, requests_per_minute, note}
        self.keys: dict[str, dict] = {}
        for entry in data.get("auth", {}).get("keys", []):
            self.keys[entry["key"]] = {
                "name": entry.get("name", "unnamed"),
                "note": entry.get("note", ""),
                "requests_per_minute": int(
                    entry.get("requests_per_minute", self.default_rpm)
                ),
            }

    @staticmethod
    def _resolve_db_path(raw: str) -> Path:
        p = Path(raw)
        return p if p.is_absolute() else PROJECT_ROOT / p

    # -- model routing -------------------------------------------------
    def resolve_model(self, gateway_name: str) -> tuple[str, str] | None:
        """Return (role, ollama_model) for a gateway name or alias.

        Role is ``"primary"`` or ``"fallback"``. Returns None when the
        name is unknown.
        """
        for role in ("primary", "fallback"):
            entry = self.models.get(role)
            if not entry:
                continue
            names = [entry.get("gateway_name")] + list(entry.get("aliases", []))
            if gateway_name in names:
                return role, entry["ollama_model"]
        return None

    def primary_model(self) -> str:
        return self.models["primary"]["ollama_model"]

    def fallback_model(self) -> str:
        return self.models["fallback"]["ollama_model"]

    def gateway_model_names(self) -> list[dict]:
        """Models as advertised by GET /v1/models."""
        out = []
        for role in ("primary", "fallback"):
            entry = self.models.get(role)
            if not entry:
                continue
            out.append(
                {
                    "id": entry["gateway_name"],
                    "object": "model",
                    "owned_by": "llm-gateway/local-ollama",
                    "permission": [],
                }
            )
            for alias in entry.get("aliases", []):
                out.append(
                    {
                        "id": alias,
                        "object": "model",
                        "owned_by": "llm-gateway/local-ollama",
                        "permission": [],
                    }
                )
        return out

    def price_for(self, gateway_name: str) -> dict:
        return self.pricing.get(
            gateway_name, self.pricing.get("default", {"prompt": 0.0, "completion": 0.0})
        )


def load_config(path: str | Path | None = None) -> GatewayConfig:
    """Load config from ``path``, ``LLMGW_CONFIG`` or the default file."""
    chosen = Path(path) if path else Path(os.environ.get("LLMGW_CONFIG", DEFAULT_CONFIG))
    with open(chosen, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return GatewayConfig(data, chosen)
