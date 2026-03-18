#!/bin/bash
set -e

# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Make uv available in this session
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

uv sync
