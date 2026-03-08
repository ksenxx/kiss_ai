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
fi
if [[ -n "${GEMINI_API_KEY}" ]]; then
  echo "KISS Sorcar requires GEMINI_API_KEY in the environment"
fi
