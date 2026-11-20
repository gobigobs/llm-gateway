"""SQLite storage for llm-gateway: response cache + request log.

Two tables, one file, WAL mode so the dashboard can read while the API
writes:

* ``cache`` — exact-match response cache. Key = SHA-256 of the canonical
  request (gateway model + messages + temperature + max_tokens).
  A hit returns the stored completion without ever calling Ollama.
* ``requests`` — one row per API call (including cache hits, fallbacks
  and rate-limit rejections) so /stats and the dashboard are computed
  from real data, never from in-memory counters that vanish on restart.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path


class Store:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    gateway_model TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    api_key_name TEXT NOT NULL,
                    gateway_model TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    cached INTEGER NOT NULL DEFAULT 0,
                    fallback_used INTEGER NOT NULL DEFAULT 0,
                    rate_limited INTEGER NOT NULL DEFAULT 0,
                    error INTEGER NOT NULL DEFAULT 0,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    latency_ms REAL NOT NULL DEFAULT 0,
                    estimated_cost_inr REAL NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
                """
            )
            self._conn.commit()

    # -- cache ----------------------------------------------------------
    @staticmethod
    def cache_key(
        gateway_model: str, messages: list, temperature: float | None, max_tokens: int | None
    ) -> str:
        canonical = json.dumps(
            {
                "model": gateway_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def cache_get(self, key: str, ttl_seconds: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response_json, prompt_tokens, completion_tokens, created_at"
                " FROM cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        if ttl_seconds > 0 and (time.time() - row["created_at"]) > ttl_seconds:
            with self._lock:
                self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                self._conn.commit()
            return None
        return {
            "response": json.loads(row["response_json"]),
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
        }

    def cache_put(
        self, key: str, gateway_model: str, response: dict, prompt_tokens: int, completion_tokens: int
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache"
                " (key, gateway_model, response_json, prompt_tokens, completion_tokens, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    key,
                    gateway_model,
                    json.dumps(response, ensure_ascii=False),
                    prompt_tokens,
                    completion_tokens,
                    time.time(),
                ),
            )
            self._conn.commit()

    # -- request log ----------------------------------------------------
    def log_request(self, **fields) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO requests
                   (ts, api_key_name, gateway_model, model_used, cached,
                    fallback_used, rate_limited, error,
                    prompt_tokens, completion_tokens, latency_ms, estimated_cost_inr)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fields.get("ts", time.time()),
                    fields.get("api_key_name", "unknown"),
                    fields.get("gateway_model", ""),
                    fields.get("model_used", ""),
                    int(fields.get("cached", False)),
                    int(fields.get("fallback_used", False)),
                    int(fields.get("rate_limited", False)),
                    int(fields.get("error", False)),
                    int(fields.get("prompt_tokens", 0)),
                    int(fields.get("completion_tokens", 0)),
                    float(fields.get("latency_ms", 0)),
                    float(fields.get("estimated_cost_inr", 0.0)),
                ),
            )
            self._conn.commit()

    # -- stats ----------------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            row = self._conn.execute(
                """SELECT
                       COUNT(*) AS total,
                       COALESCE(SUM(cached), 0) AS cache_hits,
                       COALESCE(SUM(fallback_used), 0) AS fallbacks,
                       COALESCE(SUM(rate_limited), 0) AS rate_limited,
                       COALESCE(SUM(error), 0) AS errors,
                       COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                       COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                       COALESCE(SUM(estimated_cost_inr), 0.0) AS estimated_cost_inr,
                       COALESCE(AVG(CASE WHEN cached = 0 AND rate_limited = 0
                                         THEN latency_ms END), 0) AS avg_latency_ms
                   FROM requests"""
            ).fetchone()
            per_model = self._conn.execute(
                """SELECT gateway_model, COUNT(*) AS n,
                          COALESCE(SUM(prompt_tokens), 0) AS pt,
                          COALESCE(SUM(completion_tokens), 0) AS ct
                   FROM requests GROUP BY gateway_model ORDER BY n DESC"""
            ).fetchall()
            per_key = self._conn.execute(
                """SELECT api_key_name, COUNT(*) AS n
                   FROM requests GROUP BY api_key_name ORDER BY n DESC"""
            ).fetchall()
            cache_rows = self._conn.execute("SELECT COUNT(*) AS n FROM cache").fetchone()
            recent = self._conn.execute(
                """SELECT ts, api_key_name, gateway_model, model_used, cached,
                          fallback_used, rate_limited, error,
                          prompt_tokens, completion_tokens,
                          ROUND(latency_ms) AS latency_ms
                   FROM requests ORDER BY id DESC LIMIT 20"""
            ).fetchall()
        total = row["total"]
        cache_hits = row["cache_hits"]
        return {
            "total_requests": total,
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hits / total, 4) if total else 0.0,
            "cache_entries": cache_rows["n"],
            "fallback_events": row["fallbacks"],
            "rate_limit_hits": row["rate_limited"],
            "errors": row["errors"],
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
            "total_tokens": row["prompt_tokens"] + row["completion_tokens"],
            "estimated_cost_inr": round(row["estimated_cost_inr"], 4),
            "avg_latency_ms": round(row["avg_latency_ms"], 1),
            "per_model": [
                {
                    "model": r["gateway_model"],
                    "requests": r["n"],
                    "prompt_tokens": r["pt"],
                    "completion_tokens": r["ct"],
                }
                for r in per_model
            ],
            "per_key": [
                {"key_name": r["api_key_name"], "requests": r["n"]} for r in per_key
            ],
            "recent_requests": [dict(r) for r in recent],
        }
