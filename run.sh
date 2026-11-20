#!/usr/bin/env bash
# Start llm-gateway (local only).
# Prerequisites: Ollama running with qwen2.5:3b-instruct downloaded:
#   ollama serve          # in another terminal, if not already running
#   ollama list           # should show qwen2.5:3b-instruct
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi

# Optional overrides: LLMGW_CONFIG=/path/to/other.yaml  LLMGW_PORT=8000
export LLMGW_CONFIG="${LLMGW_CONFIG:-$PWD/config.yaml}"
PORT="${LLMGW_PORT:-8000}"

echo "llm-gateway starting on http://127.0.0.1:$PORT"
echo "  dashboard : http://127.0.0.1:$PORT/"
echo "  health    : http://127.0.0.1:$PORT/health"
echo "  demo key  : demo-key-12345"
echo "  config    : $LLMGW_CONFIG"
exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "$PORT"
