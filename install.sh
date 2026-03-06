#!/bin/bash
set -e

# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Make uv available in this session
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Install code-server for the embedded VS Code editor
if ! command -v code-server &>/dev/null; then
  curl -fsSL https://code-server.dev/install.sh | sh
fi

git clone https://github.com/ksenxx/kiss_ai.git
cd kiss_ai

uv venv --python 3.13
source .venv/bin/activate
uv sync

if [[ -n "${ANTHROPIC_API_KEY}" ]]; then
  echo "KISS Sorcar requires ANTHROPIC_API_KEY in the environment" 
  exit 1
fi
if [[ -n "${GEMINI_API_KEY}" ]]; then
  echo "KISS Sorcar requires GEMINI_API_KEY in the environment"
  exit 1
fi

if [[ -n "${ANTHROPIC_API_KEY}" ]]; then
  uv run sorcar --model_name "claude-opus-4-6"
elif [[ -n "${OPENAI_API_KEY}" ]]; then
  uv run sorcar --model_name "gpt-5.2"
elif [[ -n "${GEMINI_API_KEY}" ]]; then
  uv run sorcar --model_name "gemini-3.1-pro-preview"
elif [[ -n "${TOGETHER_API_KEY}" ]]; then
  uv run sorcar --model_name "moonshotai/Kimi-K2.5"
elif [[ -n "${OPENROUTER_API_KEY}" ]]; then
  uv run sorcar --model_name "openrouter/anthropic/claude-opus-4-6"
elif [[ -n "${MINIMAX_API_KEY}" ]]; then
  uv run sorcar --model_name "minimax-m2.5"
else
  echo "❌ Unexpected error: no API key detected even after check."
  exit 1
fi

