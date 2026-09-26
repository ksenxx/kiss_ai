---
title: 'Relentless continuation: finish(is_continue=True), sub-sessions and the summarizer'
uuid: 928a0cf3-0c09-4c9a-8c73-da16524688d9
summary: How RelentlessAgent.perform_task chains sub-sessions via finish(is_continue=True),
  CONTINUATION_PROMPT, the failed-session summarizer, MAX_ZERO_PROGRESS_SESSIONS stall
  guard and merged results.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Relentless continuation

`RelentlessAgent.perform_task` runs a task as a sequence of **sub-sessions**. Each sub-session is a
new `KISSAgent` with an empty conversation, so a long task survives context exhaustion.

## Loop (per `session in range(max_sub_sessions)`)
1. Compute `remaining_budget = max_budget - banked spend`. If nothing is left, it raises `BudgetExceededError`, or returns a
   partial result when running as a sub-agent (see `sorcar-budget-and-step-limits`).
2. Create the executor and copy the hooks onto it: `pre_step_hook`, `tool_call_guard`, `context_reset_hook`,
   `llm_call_hook` and `tool_call_hook`. Set `budget_check_hook = self._check_total_budget`. When `session > 0`,
   call `context_reset_hook()` (the `UsefulTools.forget_reads` Read-dedupe reset).
3. Run with `TASK_PROMPT` = task description + `previous_progress`. Attachments are sent only in session 0.
4. Parse the YAML result from `finish`:
   - `success` true, or `is_continue` false: return it. If earlier sessions exist, the summary is rewritten to
     `<h3>Previous Session N</h3>` sections, a separator, then `<h3>Final Session</h3>`, and a merged
     result event is emitted (`_emit_merged_result_event`).
   - otherwise append the summary and build `CONTINUATION_PROMPT` ("# Task Progress (Continuation N)",
     "DON'T redo completed work", "rethink the strategy") from `_capped_progress_text`, which keeps the
     newest summaries within `MAX_PROGRESS_CHARS = 60_000` and prepends "(k earlier attempt summaries omitted.)".

The model is told to do this by `IMPORTANT_INSTRUCTIONS` ("If the task is not complete and you are at
risk of running out of context length, you MUST call finish(success=False, is_continue=True, ...)").

## Exceptions inside a session
- `BudgetExceededError`: terminal (see the budget page).
- Any other `Exception`: the error becomes terminal, returned as `finish(False, False, "Type: msg")`,
  when **either** it is not a context overflow and (it has a `__cause__` or is not a `KISSError`),
  **or** the executor made at most 1 step. Otherwise it is converted into a continuation: the
  step-limit `KISSError` ("exceeded N steps") and `ContextWindowExceededError` both land here.
  `_summarize_failed_session` runs a summarizer agent with `SUMMARIZER_PROMPT`. That agent reads the
  saved trajectory file (the first ~50 and last ~200 lines) and returns an HTML chronology. With
  `append_basic_tools=False` the summarizer is skipped and the summary is just "Agent failed: ...".
- `BaseException` (stop / KeyboardInterrupt): usage is banked, then re-raised.

## Stall guard
A continuation counts as *stalled* when `executor.tool_calls_made == 0` (no tool call other than
`finish`) or its summary equals the previous one. After `MAX_ZERO_PROGRESS_SESSIONS = 2` consecutive
stalls, `_fail_after_sessions` raises `KISSError`, with the prior summaries composed by
`_build_exhaustion_summary`. Running out of `max_sub_sessions` raises "Task failed after N sub-sessions".
The guard was added in commit 2cd5c1858 after tasks were seen spinning through empty continuations.

## System prompt per session
It is built once per task: `self.system_prompt` + `IMPORTANT_INSTRUCTIONS` (work-dir line, PID
warning) + the "# Task Settings" section + `~/.kiss/SORCAR.md`. See `sorcar-system-prompt-assembly`.

## Sources
- `src/kiss/agents/sorcar/relentless_agent.py` (`perform_task`, `CONTINUATION_PROMPT`, `SUMMARIZER_PROMPT`, `MAX_ZERO_PROGRESS_SESSIONS`, `_capped_progress_text`, `_summarize_failed_session`, `_fail_after_sessions`, `_build_exhaustion_summary`)
- `src/kiss/core/utils.py` (`finish`)
- `src/kiss/core/kiss_agent.py` (`tool_calls_made`, `_run_agentic_loop`)
