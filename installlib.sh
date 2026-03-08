#!/bin/bash
set -e

# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh


mkdir -p myproject
cd myproject
uv init --python 3.13
uv add kiss-agent-framework
echo "Installed kiss-agent-framework in a fresh environment in the folder myproject."

if [[ -z "${ANTHROPIC_API_KEY}" ]]; then
  echo "KISS Sorcar requires ANTHROPIC_API_KEY in the environment" 
  exit 1
fi
if [[ -z "${GEMINI_API_KEY}" ]]; then
  echo "KISS Sorcar requires GEMINI_API_KEY in the environment"
  exit 1
fi
