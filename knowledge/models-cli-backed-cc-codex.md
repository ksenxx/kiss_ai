---
title: CLI-backed models cc/* (Claude Code) and codex/* (Codex CLI) and run-to-completion
uuid: 1ddd6e12-eca6-48de-926c-c4fee2990497
summary: 'cc/* and codex/* CLI models: KISSAgent run-to-completion path (one generate(),
  3600 s timeout, finish wrapping, mid-run set_model), CLI flags, stream parsing,
  timeouts, usage, zero prices.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# CLI-backed models: `cc/*` and `codex/*`

These models call a locally installed, subscription-logged-in CLI instead of an HTTP API. Both
subclass `CLITextModel` (`src/kiss/core/models/model.py`). The `model()` factory in
`model_info.py` picks `CodexModel` for `codex/` names and `ClaudeCodeModel` for `cc/` names;
`model_runs_task_to_completion(name)` is True exactly for names starting with `cc/` or `codex/`.

## Run-to-completion in KISSAgent
- `CLITextModel.runs_task_to_completion = True` (base `Model` has `False`). The CLIs are full
  coding agents with native tools (Bash, Edit, ...) that run directly on the host, so these
  models cannot honour `docker_image` isolation.
- `_run_agentic_loop` checks `self.model.runs_task_to_completion` at the start of **every**
  iteration, not only the first: Sorcar's `set_model` tool can switch the live model to a CLI
  agent mid-run, and the remaining task then goes to that agent in one shot.
- `_run_task_to_completion()`:
  - sets `timeout: CLI_TASK_TIMEOUT_SECONDS` (3600 s of output silence) when `model_config` has
    no `timeout`, on a **copy** of the config (callers' dicts are never mutated).
  - `_generate_once()` makes one `generate()` call with accounting; on failure, partial usage
    from `take_partial_usage_response()` is still billed.
  - `_wrap_in_finish_contract(text)` wraps the CLI's final message via the registered `finish`
    (`success=True, is_continue=False, summary_in_html=text`) when that `finish` has a
    `summary_in_html` parameter, so YAML-parsing callers (RelentlessAgent, ChatSorcarAgent)
    work; otherwise the text is returned unchanged.
- KISS tools stay registered but are never shown to the CLI.

## Prompt construction (`CLITextModel._build_prompt`)
- One-message conversation -> the task text itself; multi-turn (e.g. after a mid-run switch) ->
  flattened `[User]: / [Assistant]: / [Tool Result]:` transcript (`_conversation_as_dialogue`);
  the CLIs are stateless across invocations.
- `model_config["system_instruction"]` is appended after `CLI_SYSTEM_PROMPT_HEADER`
  (`"\n\n# You new system prompt follows:\n"`), so the CLI keeps its native system prompt.
- Attachments are not supported (ignored with a warning); tool-result attachments produce a note
  in the prompt saying they could not be shown.

## Command lines
- Claude Code (`ClaudeCodeModel._build_cli_args`): `claude --print --disable-slash-commands
  --dangerously-skip-permissions --no-session-persistence --model <name after cc/>
  --output-format stream-json --verbose --include-partial-messages`. `cc/opus` -> `--model opus`.
  `claude` must be on `PATH` (`_find_claude_cli`).
- Codex (`CodexModel._build_cli_args`): `codex exec --json --skip-git-repo-check
  --dangerously-bypass-approvals-and-sandbox [-m <name after codex/>]`; `codex/default` omits
  `-m`. Binary via `find_codex_executable()`: `PATH` first, then bundled desktop-app paths
  (`_UI_CANDIDATE_PATHS`, e.g. `/Applications/Codex.app/Contents/Resources/codex`).
Permissions/sandbox are bypassed on purpose: KISS is the already-authorized outer agent.

## Subprocess supervision (`CLITextModel._cli_turn`, `_CLIProcess`)
- `model_config["timeout"]` = allowed **output silence** (default 300 s per turn); exceeding it
  raises the retryable `TimeoutError` from `_cli_stall_error` (never `KISSError`, which would
  abort the task).
- `model_config["work_dir"]` is the child's cwd (framework-only key) so native tools act on the
  task's work tree, not the daemon's.
- stderr is drained continuously (a full 64 KiB pipe would block the CLI); user Stop becomes
  `KeyboardInterrupt`; an open thinking block is closed on every exit.

## Stream parsing
- Claude Code: stream-json events (`stream_event` wrappers unwrapped by
  `_iter_stream_json_events`); assistant text from every message is accumulated, native
  `tool_use` blocks are shown as thinking lines (`$ command` for Bash, `Name({...})` otherwise,
  deduplicated by id).
- Codex: JSONL events (`thread.started`, `item.started`/`item.completed` for
  `command_execution` and `agent_message`, `turn.completed` with usage, `turn.failed`/`error` ->
  `KISSError` with stderr).

## KISS-level tool calls (direct use only)
Outside run-to-completion, `generate_and_process_with_tools` injects the text-based tool prompt
into `system_instruction` (`_install_tools_prompt_in_system_instruction`) and parses
`tool_calls` JSON from the output (`_parse_text_based_tool_calls`). Claude Code kills the CLI as
soon as the first complete run of `tool_calls` blocks appears (`_ToolCallFilteredStream`,
`_tool_bearing_turn`).

## Usage and cost
- Claude Code: `usage.input_tokens`, `output_tokens`, `cache_read_input_tokens`, cache writes
  via the shared `cache_creation_tokens`. If `generate()` raises mid-stream, usage accumulated
  from `message_delta` events is returned once by `take_partial_usage_response()`.
- Codex: `input_tokens` minus `cached_input_tokens` and `cache_write_input_tokens`; output
  already includes reasoning tokens.
- Catalog prices for `cc/*` and `codex/*` are 0.0 (subscription), so `calculate_cost` returns 0.
- No embeddings: `CLITextModel.get_embedding` raises `KISSError`.

## Sources
- `src/kiss/core/models/model.py` (`CLITextModel`, `CLI_SYSTEM_PROMPT_HEADER`, `_build_prompt`, `_cli_turn`, `_CLIProcess`, `_cli_stall_error`, `_ToolCallFilteredStream`)
- `src/kiss/core/models/claude_code_model.py` (`ClaudeCodeModel`, `_build_cli_args`, `_find_claude_cli`, `_iter_stream_json_events`, `take_partial_usage_response`, `extract_input_output_token_counts_from_response`)
- `src/kiss/core/models/codex_model.py` (`CodexModel`, `find_codex_executable`, `_UI_CANDIDATE_PATHS`, `_build_cli_args`, `_parse_stream_events`)
- `src/kiss/core/models/model_info.py` (`model_runs_task_to_completion`, `model`)
- `src/kiss/core/kiss_agent.py` (`CLI_TASK_TIMEOUT_SECONDS`, `_run_agentic_loop`, `_run_task_to_completion`, `_generate_once`, `_wrap_in_finish_contract`, `_registered_finish_and_params`)
