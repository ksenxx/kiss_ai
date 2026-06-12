# Task: Diagnose and fix `claude-fable-5` silent task death ("(no result)")

## Symptom

Four `task_history` rows in `~/.kiss/sorcar.db` (ids 3706, 3707, 3708,
3710), all running the same RECEIPT.pdf review prompt on model
`claude-fable-5`, terminated after 5-6 steps with persisted summary
`"No summary available"`. The same prompt on `claude-opus-4-7` (task
3709\) completed normally in 29 steps. Event log for task 3710 shows
4 successful tool_call+tool_result pairs followed by:

```
seq 16: text_end
seq 17: {"type":"result","text":"(no result)","step_count":5}
seq 18: {"type":"task_done"}
```

## Root cause

The `claude-fable-5` provider adapter occasionally returns an
assistant turn with empty `content` and no `tool_calls` after a
tool_result (likely a streaming/reasoning-block parsing issue).

In `src/kiss/core/kiss_agent.py::_execute_step`, when `function_calls`
is empty, the agent increments `_consecutive_no_tool_calls`. Once
that counter reaches `MAX_CONSECUTIVE_NO_TOOL_CALLS` (=2), the loop
treats the last text-only response as an implicit finish and returns
`str(response_text)`. For `claude-fable-5` empty turns,
`response_text` is `""`, so an empty string propagates outward:

- `JsonPrinter._broadcast_result` substitutes the literal
  `"(no result)"` for empty body (json_printer.py:547).
- `RelentlessAgent.perform_task` parses the empty YAML
  (`yaml.safe_load("") -> None -> {}`), returns `""`.
- `task_runner._run_task_inner` falls back to the string
  `"No summary available"` and persists that.

The user sees nothing actionable.

## Fix

In `_execute_step` at the `_consecutive_no_tool_calls >= MAX`
branch, when `response_text` is empty or whitespace-only, raise a
`KISSError` with a clear diagnostic instead of returning `""`.
`RelentlessAgent` already routes `KISSError` into a `success=False`
result event with the exception message as summary, so the user
sees the actual cause and can act on it.

Non-empty text-only responses still trigger the implicit-finish
behavior (covered by the existing
`test_no_tool_call_loop.py::test_agent_returns_after_consecutive_no_tool_call_responses`).

## Integration test

New file
`src/kiss/tests/core/test_empty_response_silent_death.py`:

- `test_always_empty_response_raises_kiss_error` — `HTTPServer`
  always returns `{"content": ""}` with no `tool_calls`. Asserts
  `KISSError` raised with "empty" in the message within ≤2 steps.
- `test_empty_after_tool_call_raises_kiss_error` — exact production
  shape: step 1 model issues a `Bash` tool_call, all subsequent
  turns are empty. Asserts `KISSError` raised within ≤3 steps.

Both tests use the real `KISSAgent.run` end-to-end against a real
`http.server.HTTPServer` returning OpenAI-compatible JSON — no
mocks, per testing rules. Both tests **fail** before the fix
(reproducing the bug) and **pass** after the fix.

## Verification

- `uv run pytest -v src/kiss/tests/core/test_empty_response_silent_death.py src/kiss/tests/core/test_no_tool_call_loop.py` → 4 passed.
- `uv run check --full` → all checks pass after `mdformat` re-formats `PROGRESS.md`.

## Files modified

- `src/kiss/core/kiss_agent.py` — added empty-response guard at the
  `MAX_CONSECUTIVE_NO_TOOL_CALLS` branch.
- `src/kiss/tests/core/test_empty_response_silent_death.py` — new
  integration test reproducing the claude-fable-5 silent death.
