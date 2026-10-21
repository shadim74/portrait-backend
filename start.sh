#!/usr/bin/env bash
# One-command local start for the talking-portrait backend (macOS, Apple Silicon).
# No Homebrew or admin password needed. Safe to run again: skips what's done.
set -euo pipefail
cd "$(dirname "$0")"

say() { printf "\n\033[1;33m▶ %s\033[0m\n" "$1"; }

# --- 1. uv: installs Python for us, into your user folder ------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv (Python manager, no password needed)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# --- 2. Python 3.12 environment + packages ---------------------------------
if [ ! -d .venv ]; then
  say "Creating Python 3.12 environment"
  uv venv --python 3.12 .venv
fi
source .venv/bin/activate
if [ ! -f .venv/.installed ] || [ requirements.txt -nt .venv/.installed ]; then
  say "Installing Python packages (first time takes a few minutes)"
  uv pip install -r requirements.txt
  touch .venv/.installed
fi

# --- 3. Settings tuned for a MacBook Air M2 16 GB --------------------------
if [ ! -f .env ]; then
  say "Writing .env with Mac settings"
  cat > .env <<'EOF'
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3:4b-instruct
LLM_MAX_TOKENS=200
DEVICE=cpu
COMPUTE_TYPE=int8
WHISPER_MODEL=base
LIPSYNC_ENGINE=none
PERSONAS_FILE=personas.json
MAX_HISTORY=20
EOF
fi
MODEL="$(grep '^LLM_MODEL=' .env | cut -d= -f2)"

# --- 4. Ollama -------------------------------------------------------------
OLLAMA="$(command -v ollama || true)"
[ -z "$OLLAMA" ] && [ -x /Applications/Ollama.app/Contents/Resources/ollama ] \
  && OLLAMA=/Applications/Ollama.app/Contents/Resources/ollama
if [ -z "$OLLAMA" ]; then
  echo "Ollama is not installed. Download it from https://ollama.com, open it once, then run this again."
  exit 1
fi
if ! curl -s http://localhost:11434/api/tags >/dev/null; then
  say "Starting Ollama"
  open -a Ollama 2>/dev/null || ("$OLLAMA" serve >/dev/null 2>&1 &)
  for _ in $(seq 1 30); do curl -s http://localhost:11434/api/tags >/dev/null && break; sleep 1; done
fi
if ! "$OLLAMA" list | awk '{print $1}' | grep -qx "$MODEL"; then
  say "Downloading $MODEL (one time, ~2.5 GB)"
  "$OLLAMA" pull "$MODEL"
fi

# --- 5. Run ----------------------------------------------------------------
say "Starting server: open http://localhost:8000 (Ctrl+C to stop)"
(sleep 8 && open http://localhost:8000) &
exec uvicorn app.main:app --host 127.0.0.1 --port 8000
