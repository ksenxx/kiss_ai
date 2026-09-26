---
title: 'pytest configuration: testpaths, default deselection, markers (slow, process_killer,
  live_api, live_cli)'
uuid: 4b6d137e-c5fc-4122-aa62-a12ef8cc2138
summary: pytest settings in pyproject.toml - testpaths src/kiss/tests, addopts deselecting
  slow and process_killer, marker meanings, no timeout, opt-in coverage, --model option,
  how to run marked sets.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# pytest configuration and markers

All of this lives in `[tool.pytest.ini_options]` in `pyproject.toml`.

- `testpaths = ["src/kiss/tests"]`. Files are `test_*.py`, classes `Test*`, functions `test_*`.
- `addopts = "-m 'not slow and not process_killer'"`, so a plain `uv run pytest` skips both
  marked sets. To run them:
  - `uv run pytest -m slow` (slow tests, including the Docker ones)
  - `uv run pytest -m process_killer`. Run this set on its own, after everything else.
  A command-line `-m` replaces the `-m` from `addopts` (the option is a plain store, last one
  wins), so clearing `addopts` is not needed; `-o addopts=""` is harmless.
- `timeout = 0`: pytest-timeout is installed but has no default limit. Add
  `--timeout N --timeout-method=thread` yourself when a split must not hang.
- Coverage is not in `addopts`. Measured branch tracing slowed the suite 5x to 20x. Opt in with
  `uv run pytest --cov=src/kiss --cov-branch`. The root `conftest.py` registers inert
  `--cov`/`--cov-branch` options when pytest-cov is disabled (`-p no:cov`), so those flags still
  parse.
- `filterwarnings` silences only google-genai's `_UnionGenericAlias` DeprecationWarning.

## Markers
| Marker | Meaning |
|---|---|
| `slow` | Long tests, including Docker. Deselected by default. |
| `process_killer` | Sends signals (`os.killpg` bursts, SIGKILL cleanup) to whole process groups, which can kill sibling pytest processes when many splits run at once. Deselected by default. Used by the install.sh signal-immunity tests, e.g. `tests/agents/vscode/test_install_script_new_session_immunity.py`. |
| `live_api` | Makes real HTTP calls to a provider API. Skipped when the required key is unset. |
| `live_cli` | Spawns the real claude/codex CLI, so it opts out of the CLI-locator stub (`src/kiss/tests/cli_locator_stub.py`). |
| `redundancy_check` | Declared in the config, but no test under `src/kiss` uses it. (`src/kiss/scripts/redundancy_analyzer.py` finds redundant tests from coverage dynamic contexts, not from this marker.) |

## Options added by `src/kiss/tests/conftest.py`
- `--model` (default `DEFAULT_MODEL = "claude-opus-4-6"`) selects the model for model tests.
- `collect_ignore = ["run_all_models_test.py"]`.
- API-key helpers for gating real-model tests: `has_openai_api_key`,
  `has_anthropic_api_key`, `has_gemini_api_key`, `has_openrouter_api_key`, ...,
  `get_required_api_key_for_model`, `skip_if_no_api_key_for_model`.
- Platform gates: `posix_only(reason)` (a `skipif` on Windows naming the mechanism),
  `requires_unix_sockets`, `is_root()`. Use `is_root()` instead of `os.geteuid() == 0`.

## Sources
- `pyproject.toml` (`[tool.pytest.ini_options]`)
- `conftest.py` (`pytest_addoption`, module docstring)
- `src/kiss/tests/conftest.py` (`pytest_addoption`, `posix_only`, `is_root`, `skip_if_no_api_key_for_model`)
- `src/kiss/tests/cli_locator_stub.py`
