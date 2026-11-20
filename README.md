# llm-gateway

![Verified 23/23](https://img.shields.io/badge/verified-23%2F23-brightgreen)
![Local only](https://img.shields.io/badge/local--only-no%20paid%20APIs-blue)
![Cost ₹0](https://img.shields.io/badge/cost-%E2%82%B90%20per%20token-gold)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-%2B%20SQLite%20%2B%20Ollama-009688)

**One OpenAI-compatible API in front of your local Ollama models — so repeat questions answer themselves.** Send a normal `POST /v1/chat/completions`: the gateway authenticates your key, enforces your per-minute limit, serves repeats instantly from cache, routes, falls back automatically, logs everything. Runs entirely on your own machine — no paid keys, no accounts, no telemetry.

![Dashboard](docs/screenshots/dashboard.png)

![Playground — a real cached answer in 6 ms](docs/screenshots/dashboard-playground.png)

*Real captures, 18 Nov 2026: playground shows a genuine cached answer — `CACHE HIT ⚡`, 6 ms, 83 tokens; dashboard a live 57.1% hit rate across 14 requests.*

## Quick start — 30 seconds

```bash
# 1. Ollama running with a model downloaded (`ollama list` to check).
ollama serve

# 2. Start the gateway (default http://127.0.0.1:8000).
./run.sh

# 3. Ask it something (demo key: demo-key-12345).
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer demo-key-12345" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-3b",
       "messages": [{"role": "user", "content": "What is an LLM gateway, in one sentence?"}]}'
```

Open **http://127.0.0.1:8000/** and try the playground — ask twice; the second answer comes back a `CACHE HIT` in milliseconds. Port busy? `LLMGW_PORT=8005 ./run.sh`. Other config? `LLMGW_CONFIG=/path/to/other.yaml ./run.sh`.

## What it does

| Feature | How it works | Proof (measured live) |
|---|---|---|
| **OpenAI-compatible API** | `POST /v1/chat/completions`, `GET /v1/models` — any OpenAI client works unchanged | Verified, no mocks |
| **Exact-match caching** | SHA-256 of the request, in SQLite (TTL 3600 s); hits skip Ollama | Repeat: **0.00 s** vs **4.6 s** fresh, byte-identical |
| **Automatic fallback** | Primary failure → retry fallback entry, flagged | Answered in **2.3 s**, `fallback_used: true` |
| **Per-key rate limits** | Sliding-window, per-minute, in memory; `429` + `Retry-After` | 4 rapid calls at 3 req/min → **[200, 200, 200, 429]** |
| **Cost tracking** | Tokens + estimated cost in `/stats` and the dashboard | Local models: **₹0** |
| **Live dashboard** | Totals, hit rate, fallbacks, 429s, tokens, cost, per-model/key tables, last 20 requests, playground; refreshes every 5 s | See screenshots above |
| **Request logging** | Every request — hit, miss, fallback, 429, error — to SQLite (WAL mode) | `/stats` survives restarts |

## Demo script (2 minutes)

1. `./run.sh`, open http://127.0.0.1:8000/ — .
2. Ask in the playground — first answer takes a few seconds; **Total requests** hits 1.
3. Ask the **same question again** — instant, `CACHE HIT ⚡`, 0 ms row next to the earlier `MISS` row.
4. Prove the guardrails:
   ```bash
   curl -s http://127.0.0.1:8000/v1/models   # no key → 401
   curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
     -H "Authorization: Bearer demo-key-12345" -H "Content-Type: application/json" \
     -d '{"model": "no-such-model", "messages": [{"role": "user", "content": "hi"}]}'  # → 404
   ```

## Endpoints

All endpoints except `/` and `/health` need `Authorization: Bearer <api-key>`; keys in `config.yaml` (`demo-key-12345`: 60/min, `dev-key-67890`: 10/min).

| Endpoint | Method | What it does |
|---|---|---|
| `/` | GET | Live dashboard. No auth. |
| `/health` | GET | Liveness + Ollama models (`status`, `ollama_reachable`, `ollama_models`, `gateway_models`, `cache_enabled`). No auth. |
| `/v1/models` | GET | OpenAI-style list of gateway names and aliases (`qwen2.5-3b`, `chat`, `default`, `qwen2.5-3b-fallback`, `fallback`). |
| `/v1/chat/completions` | POST | OpenAI-compatible completion (`model`, `messages`, optional `temperature` 0–2, `max_tokens`; `stream` must be false). Adds a `gateway` object (`cached`, `fallback_used`, `model_used`, `token_counts_estimated`) + headers `X-Gateway-Cache: HIT/MISS`, `X-Gateway-Model-Used`, `X-Gateway-Fallback`. |
| `/stats` | GET | Live totals: requests, hits, hit rate, entries, fallbacks, 429s, errors, tokens, cost, latency, `per_model`, `per_key`, last 20 requests — the same numbers the dashboard shows. |

### Error behaviour

| Situation | Response |
|---|---|
| Missing or wrong API key | `401` |
| Unknown `model` | `404` `model_not_found`, listing the available models |
| Empty `messages`, or `stream: true` | `400` |
| Key over its per-minute limit | `429` + `Retry-After` header |
| Primary fails **and** fallback fails | `502`, naming both errors |
| Primary fails, fallback answers | `200`, `gateway.fallback_used: true` + `X-Gateway-Fallback` header |

## How a request flows

```
client → auth → rate limit → cache (SHA-256, SQLite) → route → fallback → log
```

## Honest notes — read before showing this to anyone

- **Only one model is downloaded here: `qwen2.5:3b-instruct`.** So `primary` and `fallback` both point at it. The fallback *mechanism* is real and verified (`scripts/verify.py`); for a genuinely different fallback, `ollama pull llama3.2:3b`, change `fallback.ollama_model` — no code change needed.
- **First (uncached) answers are slow on CPU** — 5-30 s; the first call after Ollama starts can take ~3 min while the model loads (one warm-up call on 18 Nov took 267 s). Repeats are instant (0–10 ms).
- **Streaming is not supported.** `stream: true` returns HTTP 400; send `stream: false` (the default).
- **Rate limits live in memory and reset on restart.** Multi-worker deployments need a shared store (e.g. Redis).
- **Token counts are partly estimated.** Ollama's real `prompt_eval_count` / `eval_count` are used when reported; otherwise the ~4 chars/token estimate, and `/stats` says so.
- **Estimated cost is ₹0 because the models are local.** The price table exists so the same code reports real costs if a paid backend is ever added.
- **The demo keys in `config.yaml` are public.** Rotate them before exposing the gateway beyond `localhost`. **Local only** — nothing leaves the machine.

## Verification — 23/23, no mocks

`scripts/verify.py` exercises every feature against the live local model, no mocks (non-zero exit on failure):

```bash
.venv/bin/python scripts/verify.py
```

Last run (18 Nov 2026, `qwen2.5:3b-instruct`): **23/23 passed**:

- First call 16.7 s (`cached: false`); same prompt again: byte-identical, **0.00 s**, `cached: true`, `X-Gateway-Cache: HIT`.
- Fallback (`config.test-fallback.yaml`, primary = nonexistent model): answered in 2.9 s, `fallback_used: true`.
- Rate limit (`config.test-ratelimit.yaml`, 3 req/min): four rapid calls → `[200, 200, 200, 429]`; `/stats` recorded the 429.
- Missing key → 401, bad key → 401, unknown model → 404; `/health` lists `qwen2.5:3b-instruct`.

Why the tests are shaped this way (top of `scripts/verify.py`): the SQLite cache persists, so prompts carry a per-run nonce; and an uncached call takes ~as long as the 60 s window, so the test pre-warms the cache, restarts, then fires four rapid cache hits. The limiter alone: `[True, True, True, False, False]` at limit 3.

Re-take screenshots after any dashboard change (gateway running, e.g. `LLMGW_PORT=8005 ./run.sh`):

```bash
node ~/workspace/.scratch/shot/capture-llm-gateway.mjs http://127.0.0.1:8005/
# Writes docs/screenshots/*.png - open each and confirm it is not blank.
```

Bundled Chromium won't start headless here, so screenshots use a Playwright-managed Chromium.

## Project layout

```
app/
  main.py            routes, auth, rate-limit, cache, fallback, logging
  config.py          YAML config + model resolution
  ollama_client.py   httpx Ollama client + OpenAI response shaping
  ratelimit.py       per-key sliding-window limiter (in-memory)
  store.py           SQLite cache + request log + /stats (WAL mode)
  tokens.py          token estimation (~4 chars/token)
  static/dashboard.html  self-contained dashboard + playground
config.yaml                one file: host/port, Ollama, routing, keys, cache TTL, pricing, SQLite path (see "Honest notes")
config.test-fallback.yaml  test-only: primary = nonexistent model
config.test-ratelimit.yaml test-only: demo key at 3 req/min
scripts/verify.py          end-to-end verification (23 checks, no mocks)
scripts/screenshot.py      re-take screenshots (Python)
docs/screenshots/         real dashboard screenshots
run.sh               start the gateway
requirements.txt     fastapi, uvicorn, httpx, pyyaml
data/                SQLite databases (git-ignored)
```

## How this could lead to paid work

Small SaaS/product teams spending real money on paid LLM APIs feel it: repeat questions billed twice, one outage taking their feature down, no per-customer limits, no idea what each customer costs. This is a working demo of exactly that fix on free local models. Paid work: freelance/consulting to build the same gateway on a team's paid provider, or a hosted multi-tenant version. No income promised — a portfolio piece, not a product yet.

## LinkedIn-ready paragraph

> I built llm-gateway: one OpenAI-compatible API in front of local LLM models, with the guardrails production gateways have — exact-match response caching, automatic model fallback, per-API-key rate limits, and cost tracking with a live dashboard. It runs entirely on my own machine on a free local model (Ollama + qwen2.5:3b), so every request costs ₹0. Verified end to end, no mocks: 23/23 checks pass, including a repeat question answered from cache in 0.00 s vs 4.6 s fresh, a fallback answering after the primary model failed, and a 4th rapid request correctly getting a 429. FastAPI + SQLite + Ollama. The interesting part wasn't the proxying — it was the verification: the first version of my test script reported failures that turned out to be stale cache rows and a rate-limit window shorter than one slow model call, not bugs in the gateway.

## Knowledge used

- **Tech:** FastAPI + Uvicorn, httpx, PyYAML, SQLite (WAL), vanilla JS/HTML/CSS, Playwright.
- **Patterns:** the LiteLLM / AI-gateway pattern — auth → rate limit → cache → route → fallback → log; SHA-256 exact-match caching; per-key sliding-window limits; SQLite as the durable log behind `/stats`; numbers survive restarts.
- **Sources:** OpenAI Chat Completions reference; Ollama API docs; FastAPI docs; LiteLLM proxy docs; OpenAI's ~4 chars/token rule. All verified live — no paid API used.
