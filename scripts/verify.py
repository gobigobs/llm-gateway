#!/usr/bin/env python3
"""End-to-end verification for llm-gateway.

Starts the real servers (main config + two throwaway test configs),
exercises every feature against the live local Ollama model, and prints
a PASS/FAIL report. No mocks: a cached response must actually be fast
and flagged, the fallback must actually answer after the primary fails,
and the 4th rapid call must actually get a 429.

Usage:  .venv/bin/python scripts/verify.py
Requires: Ollama running on localhost:11434 with qwen2.5:3b-instruct.

Why the tests are shaped this way (learned 2026-11-17, when a first
version of this script reported 3 false failures):

* The SQLite cache and request log persist between runs, so the
  cache/fallback prompts carry a random nonce per run — otherwise the
  "first call" would be a hit left over from a previous run.
* An uncached call to the local 3B model takes ~60s on this CPU-only
  VM — about as long as the 60s rate-limit window. Four sequential
  uncached probes therefore arrive minutes apart and never trip the
  limiter, even though the limiter is correct. So the rate-limit test
  pre-warms the cache with one slow call, restarts the server (fresh
  in-memory limiter, same SQLite cache), then fires four rapid
  identical calls that are instant cache hits — the 4th must 429.
* Ports 8010-8012 must be free when the script starts: a stale server
  still bound to one of them would answer /health and silently
  invalidate the results, so the script refuses to run until it is
  stopped.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / "bin" / "python"
DEMO = "demo-key-12345"
CHAT_TIMEOUT = 320  # local 3B on CPU is slow, especially on first load

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))


def http(method: str, url: str, key: str | None = None, body: dict | None = None,
         timeout: int = 30) -> tuple[int, dict, dict]:
    req = urllib.request.Request(url, method=method)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data=data, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}"), dict(resp.headers)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            parsed = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            parsed = {"raw": raw[:300].decode(errors="replace")}
        return e.code, parsed, dict(e.headers)
    except Exception as e:  # connection errors etc.
        return 0, {"error": str(e)}, {}


def http_text(method: str, url: str, timeout: int = 30) -> tuple[int, str]:
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except Exception as e:
        return 0, str(e)


class Server:
    def __init__(self, config: str, port: int):
        self.port = port
        env = dict(os.environ, LLMGW_CONFIG=str(ROOT / config))
        self.proc = subprocess.Popen(
            [str(VENV_PY), "-m", "uvicorn", "app.main:app",
             "--host", "127.0.0.1", "--port", str(port)],
            cwd=ROOT, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def wait_ready(self, timeout: int = 30) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            status, _ = http_text("GET", f"http://127.0.0.1:{self.port}/health", timeout=5)
            if status == 200:
                return True
            time.sleep(1)
        return False

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def chat(port: int, prompt: str, model: str = "qwen2.5-3b", key: str = DEMO):
    t0 = time.monotonic()
    status, body, headers = http(
        "POST", f"http://127.0.0.1:{port}/v1/chat/completions", key=key,
        body={"model": model,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=CHAT_TIMEOUT,
    )
    return status, body, headers, time.monotonic() - t0


def main() -> int:
    print("== llm-gateway verification ==")
    print(f"   Ollama: http://localhost:11434  (expect qwen2.5:3b-instruct)")

    # ---- pre-flight: test ports must be free (see module docstring) ---
    for p in (8010, 8011, 8012):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) == 0:
                print(f"ERROR: port {p} is already in use — a stale gateway "
                      f"server is running there. Stop it and re-run.")
                return 2
    nonce = uuid.uuid4().hex[:8]  # keeps this run's prompts out of old caches

    # ---- main server ------------------------------------------------
    main = Server("config.yaml", 8010)
    print("\n[1] Starting main server (config.yaml, port 8010)…")
    check("server starts", main.wait_ready(), "GET /health -> 200")
    try:
        status, body, _ = http("GET", "http://127.0.0.1:8010/health")
        check("health reports Ollama model",
              "qwen2.5:3b-instruct" in body.get("ollama_models", []),
              str(body.get("ollama_models")))

        status, _ = http_text("GET", "http://127.0.0.1:8010/")
        check("dashboard loads", status == 200, f"GET / -> {status}")

        status, body, _ = http("GET", "http://127.0.0.1:8010/v1/models", key=DEMO)
        ids = [m["id"] for m in body.get("data", [])]
        check("GET /v1/models lists gateway models",
              status == 200 and "qwen2.5-3b" in ids, str(ids))

        status, body, _ = http("GET", "http://127.0.0.1:8010/v1/models")
        check("missing API key -> 401", status == 401, f"-> {status}")

        status, body, _ = http(
            "POST", "http://127.0.0.1:8010/v1/chat/completions", key="wrong-key",
            body={"model": "qwen2.5-3b", "messages": [{"role": "user", "content": "hi"}]})
        check("bad API key -> 401", status == 401, f"-> {status}")

        status, body, _ = http(
            "POST", "http://127.0.0.1:8010/v1/chat/completions", key=DEMO,
            body={"model": "no-such-model", "messages": [{"role": "user", "content": "hi"}]})
        check("unknown model -> 404", status == 404, f"-> {status}")

        # ---- live chat + cache --------------------------------------
        # Nonce in the prompt: the SQLite cache persists between runs, so
        # a fixed prompt would already be cached and the first call here
        # would (correctly) be a HIT, failing the MISS check below.
        prompt = (f"What is 2+2? Reply with just the number and one short "
                  f"sentence. (verify run {nonce})")
        print("\n[2] Live chat completion (first call loads the model — may take a while)…")
        status, body, headers, dt1 = chat(8010, prompt)
        content = ""
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            pass
        check("chat completion succeeds",
              status == 200 and bool(content.strip()),
              f"-> {status} in {dt1:.1f}s, answer: {content.strip()[:80]!r}")
        check("first call is NOT cached",
              body.get("gateway", {}).get("cached") is False,
              f"cached={body.get('gateway', {}).get('cached')}")
        check("usage tokens reported",
              body.get("usage", {}).get("total_tokens", 0) > 0,
              str(body.get("usage")))

        print("\n[3] Same prompt again — must hit the cache…")
        status, body2, headers2, dt2 = chat(8010, prompt)
        same = False
        try:
            same = body2["choices"][0]["message"]["content"] == content
        except (KeyError, IndexError, TypeError):
            pass
        check("cached response flagged",
              status == 200 and body2.get("gateway", {}).get("cached") is True,
              f"cached={body2.get('gateway', {}).get('cached')}, "
              f"X-Gateway-Cache={headers2.get('x-gateway-cache') or headers2.get('X-Gateway-Cache')}")
        check("cached response is identical", same)
        check("cached call is fast",
              dt2 < 5.0, f"{dt2:.2f}s vs first call {dt1:.1f}s")

        status, body, _ = http("GET", "http://127.0.0.1:8010/stats", key=DEMO)
        check("GET /stats shows live numbers",
              status == 200 and body.get("total_requests", 0) >= 2
              and body.get("cache_hit_rate", 0) > 0,
              f"total={body.get('total_requests')}, "
              f"hit_rate={body.get('cache_hit_rate')}, "
              f"cost=₹{body.get('estimated_cost_inr')}")
    finally:
        main.stop()

    # ---- fallback server --------------------------------------------
    fb = Server("config.test-fallback.yaml", 8011)
    print("\n[4] Fallback test — primary is a model that does not exist…")
    check("fallback server starts", fb.wait_ready())
    try:
        # Nonce so this genuinely exercises the fallback path instead of
        # returning a cached answer stored by an earlier run.
        status, body, headers, dt = chat(
            8011, f"Say the word FALLBACK-OK and nothing else. (verify run {nonce})",
            key=DEMO)
        gw = body.get("gateway", {})
        content = ""
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            pass
        check("fallback answers after primary fails",
              status == 200 and bool(content.strip()),
              f"-> {status} in {dt:.1f}s, answer: {content.strip()[:60]!r}")
        check("fallback flagged in response",
              gw.get("fallback_used") is True,
              f"fallback_used={gw.get('fallback_used')}, model_used={gw.get('model_used')}")
    finally:
        fb.stop()

    # ---- rate-limit server ------------------------------------------
    rl = Server("config.test-ratelimit.yaml", 8012)
    print("\n[5] Rate-limit test — demo key allows 3 req/min on this config…")
    check("rate-limit server starts", rl.wait_ready())
    try:
        probe = "Rate limit probe: reply with the single word OK."
        # Pre-warm: one slow uncached call populates the SQLite cache, so
        # the timed probes below are instant cache hits. Without this,
        # four sequential uncached calls take ~60s each on this CPU-only
        # VM — longer than the 60s rate-limit window — and the 4th call
        # would (correctly) not be limited. Restarting gives a fresh
        # in-memory limiter while the SQLite cache survives the restart.
        print("    (pre-warming the cache with one slow model call…)")
        s, _, _, _ = chat(8012, probe, key=DEMO)
        check("cache pre-warm call succeeds", s == 200, f"-> {s}")
        rl.stop()
        time.sleep(2)
        rl = Server("config.test-ratelimit.yaml", 8012)
        check("rate-limit server restarts (fresh limiter)", rl.wait_ready())
        statuses = []
        for i in range(4):
            s, b, h, _ = chat(8012, probe, key=DEMO)
            statuses.append(s)
        check("first 3 rapid calls allowed", statuses[:3] == [200, 200, 200],
              str(statuses))
        check("4th rapid call -> 429", statuses[3] == 429, str(statuses))
        status, body, _ = http("GET", "http://127.0.0.1:8012/stats", key=DEMO)
        check("rate-limit hit recorded in /stats",
              body.get("rate_limit_hits", 0) >= 1,
              f"rate_limit_hits={body.get('rate_limit_hits')}")
    finally:
        rl.stop()

    # ---- summary -----------------------------------------------------
    print("\n== result ==")
    failed = [r for r in results if not r[1]]
    for name, ok, detail in results:
        pass
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED:")
        for name, _, detail in failed:
            print(f"  - {name} {detail}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
