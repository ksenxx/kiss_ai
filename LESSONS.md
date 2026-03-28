# Lessons

- When running long bash commands, use `timeout_seconds=600` parameter directly instead of background processes with `nohup` and temp file polling. The `nohup` + background approach causes timeouts on its own due to shell session handling.
- `uv run check --full` runs: clean artifacts, uv sync, generate-api-docs, compileall, ruff, mypy, pyright, and mdformat --check on markdown files. Defined in `src/kiss/scripts/check.py`.
- To fix markdown formatting issues flagged by `mdformat --check`, simply run `uv run mdformat <file>`.
