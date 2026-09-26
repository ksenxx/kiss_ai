---
title: 'kiss.core.config: Config model, KISS_* env toggles, artifact dir, DEFAULT_MAX_BUDGET'
uuid: 55621718-a4f3-47a6-8324-09054be49c30
summary: 'core/config.py Config model: API keys from env, KISS_* cost-lever toggles
  (read dedupe, compaction, context fraction), kiss_home, .kiss.artifacts job dir,
  DEFAULT_MAX_BUDGET'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# kiss.core.config: Config model, env toggles, artifact dir

`src/kiss/core/config.py` is the process-level (not user-settings) configuration. User settings
persisted in `~/.kiss/config.json` live in `vscode_config.py` instead (see
`config-vscode-config-json`).

## kiss_home()
Returns `Path($KISS_HOME)` when set, else `~/.kiss`. Resolved on **every call**, so a `KISS_HOME`
set after import (the test suite's conftest does this) is honoured. Code must call `kiss_home()`
at use time instead of caching a module-level path. Layout: `config-kiss-home-layout`.

## Config (pydantic BaseModel), DEFAULT_CONFIG = Config()
- API keys read from env via `default_factory`: `GEMINI_API_KEY`, `OPENAI_API_KEY`,
  `ANTHROPIC_API_KEY`, `ANTHROPIC_WORKSPACE_ID` (sent as `anthropic-workspace-id`, required for
  identity-linked Anthropic keys), `TOGETHER_API_KEY`, `OPENROUTER_API_KEY`, `ZAI_API_KEY`,
  `MOONSHOT_API_KEY`.
- Token-cost levers, each a plain env toggle so a regression is a flag flip:

| Field | Env var | Default |
|---|---|---|
| `read_dedupe` | `KISS_READ_DEDUPE` | True |
| `read_outline_lines` | `KISS_READ_OUTLINE_LINES` | 2000 (0 disables) |
| `tool_output_compaction` | `KISS_TOOL_OUTPUT_COMPACTION` | True |
| `compaction_start_tokens` | `KISS_COMPACTION_START_TOKENS` | 100_000 |
| `compaction_step_tokens` | `KISS_COMPACTION_STEP_TOKENS` | 100_000 |
| `tool_output_max_chars` | `KISS_TOOL_OUTPUT_MAX_CHARS` | 50_000 |
| `context_limit_fraction` | `KISS_CONTEXT_LIMIT_FRACTION` | 0.7 |
| `tool_profiles` | `KISS_TOOL_PROFILES` | True |
| `chat_history_digest` | `KISS_CHAT_HISTORY_DIGEST` | True |
| `dispatch_path_rewrite` | `KISS_DISPATCH_PATH_REWRITE` | True |

- Helpers: `_env_flag` (`0/false/no/off` any case = False, other non-empty = True, empty = default),
  `_env_int`, `_env_float` (junk or non-finite falls back to the default).
- Values are read when `Config()` is constructed, i.e. at import for `DEFAULT_CONFIG`.

## Budgets
`DEFAULT_MAX_BUDGET = 100.0` USD is the single source for `Config.max_budget` and
`vscode_config.DEFAULTS["max_budget"]` (they used to disagree: 200.0 vs 100). `Config.max_budget` is
only the default for channel-agent CLI runs. `KISSAgent` (10.0) and `RelentlessAgent`
(`relentless_agent.DEFAULT_MAX_BUDGET`, 200.0) do not consult it.

## Artifact directory
`get_artifact_dir()` lazily creates `<project>/.kiss.artifacts/jobs/job_<timestamp>_<rand>` once per
process, guarded by a lock, and never changes it: trajectories are resolved at save time, so a
mid-run swap used to send a running agent's trajectory to a different root. `artifact_dir` is an
`os.PathLike` proxy calling `get_artifact_dir()`. `get_jobs_root(base_dir)` returns the `jobs`
parent that the trajectory visualizer uses.

## Sources
- `src/kiss/core/config.py` (`kiss_home`, `Config`, `DEFAULT_CONFIG`, `DEFAULT_MAX_BUDGET`, `_env_flag`, `get_artifact_dir`, `get_jobs_root`, `_ArtifactDirProxy`)
