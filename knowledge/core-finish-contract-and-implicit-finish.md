---
title: The finish tool contract and implicit finish on text-only turns
uuid: 50d8a660-984b-41e2-b7c1-1c3048a9dfdb
summary: Built-in finish(result) vs structured utils.finish YAML (success, is_continue,
  summary_in_html); text-only turns get a nudge, then an implicit finish gated by
  hook and guard.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# The `finish` contract and implicit finish

## Two contracts
- **Built-in** `KISSAgent.finish(result: str) -> str` returns the plain text. `_setup_tools` adds it
  only when no registered tool is named `finish`.
- **Structured** `kiss.core.utils.finish(success, is_continue=False, summary_in_html="", suggested_next_task="")`
  returns YAML with `success`, `is_continue`, `summary` (always HTML via `ensure_html`, which converts
  Markdown or plain text) and an optional `suggested_next_task`. Booleans are coerced with `_coerce_bool`,
  so `"false"` strings work. RelentlessAgent and Sorcar register this one and parse the result as YAML.
- The agent detects which contract is active by checking whether the registered `finish` has a
  `summary_in_html` parameter (`_registered_finish_and_params`).
- `finish` is never registered for per-tool-call interrupt. Its result is the task's result.

## Text-only turns
- The first turn with no tool call appends the nudge "**Your response MUST have at least one function call... call the `finish` tool now...**".
- The second consecutive one (`MAX_CONSECUTIVE_NO_TOOL_CALLS = 2`):
  - with empty text, raises `_EmptyModelResponseError`, which leads to a fallback model or failure;
  - with text, gives an **implicit finish** if `_implicit_finish_allowed()`; otherwise the nudge is sent again.
- `_implicit_finish_allowed` applies the vetoes a real `finish` would face: `tool_call_hook("finish", {})`
  must return `"OK"`, and `tool_call_guard("finish", {})` must return `None`. Sorcar blocks finish while a
  queued user message is pending, so the follow-up is not dropped.
- `_implicit_finish_result` keeps the contract. Structured: `success=True, is_continue=False`, and the summary
  is the explanation plus "Last status from the model: <text>". This is terminal on purpose, so
  RelentlessAgent does not resume a model that only talks. Built-in: the last text.

## CLI run-to-completion
`_wrap_in_finish_contract(text)` wraps a CLI model's final output as a successful structured finish when
the structured contract is active. See `models-cli-backed-cc-codex`.

## History
The stagnant-tool-call implicit finish was removed (452e5d8a9). Only the text-only net remains.

## Sources
- `src/kiss/core/kiss_agent.py` (`finish`, `_execute_step`, `_implicit_finish_allowed`, `_implicit_finish_result`, `_wrap_in_finish_contract`)
- `src/kiss/core/utils.py` (`finish`, `ensure_html`)
