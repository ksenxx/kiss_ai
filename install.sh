curl -LsSf https://astral.sh/uv/install.sh | sh

# Install code-server for the embedded VS Code editor
if ! command -v code-server &>/dev/null; then
  curl -fsSL https://code-server.dev/install.sh | sh
fi

git clone https://github.com/ksenxx/kiss_ai.git
cd kiss_ai

uv init --python 3.13
uv sync
if [[ -z "${GEMINI_API_KEY}" && -z "${OPENAI_API_KEY}" && -z "${ANTHROPIC_API_KEY}" && -z "${TOGETHER_API_KEY}" && -z "${OPENROUTER_API_KEY}" && -z "${MINIMAX_API_KEY}" ]]; then
  echo "❌ At least one API key must be set in the environment. Set GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, TOGETHER_API_KEY, OPENROUTER_API_KEY, or MINIMAX_API_KEY."
  exit 1
fi

if [[ -n "${ANTHROPIC_API_KEY}" ]]; then
  uv run python -m kiss.agents.assistant.assistant --model_name "claude-opus-4-6"
elif [[ -n "${OPENAI_API_KEY}" ]]; then
  uv run python -m kiss.agents.assistant.assistant --model_name "gpt-5.2"
elif [[ -n "${GEMINI_API_KEY}" ]]; then
  uv run python -m kiss.agents.assistant.assistant --model_name "gemini-3.1-pro-preview"
elif [[ -n "${TOGETHER_API_KEY}" ]]; then
  uv run python -m kiss.agents.assistant.assistant --model_name "moonshotai/Kimi-K2.5"
elif [[ -n "${OPENROUTER_API_KEY}" ]]; then
  uv run python -m kiss.agents.assistant.assistant --model_name "openrouter/anthropic/claude-opus-4-6"
elif [[ -n "${MINIMAX_API_KEY}" ]]; then
  uv run python -m kiss.agents.assistant.assistant --model_name "minimax-m2.5"
else
  echo "❌ Unexpected error: no API key detected even after check."
  exit 1
fi

