---
title: Context compaction of old tool outputs (cache-aware gate)
uuid: 70b86fb8-1956-4ad8-947f-be9e1e9589a7
summary: context_compaction.py stubs old large tool outputs in place from 100k tokens
  (keep newest 6, skip Edit/Write/finish); applied only if it drops 25% and is not
  near hand-off.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Context compaction of old tool outputs

Every step re-sends the whole conversation, so old large tool outputs are paid for again and again.
`src/kiss/core/context_compaction.py` replaces their text with a stub. KISSAgent calls it from
`_maybe_compact_conversation()` at the start of each step, before the model call.

## Trigger (in `KISSAgent._maybe_compact_conversation`)
1. The step is skipped unless `DEFAULT_CONFIG.tool_output_compaction` is on (env `KISS_TOOL_OUTPUT_COMPACTION`, default true).
2. It is also skipped until `context_tokens_used` (the size of the last request) reaches `_next_compaction_at`.
   That starts at `compaction_start_tokens` (100k) and is reset per run.
3. `plan_compaction(model.conversation)`, then `dropped_tokens = dropped_chars(plan) // 4`.
4. `should_compact(context, dropped, handoff_tokens)` must be true (see below). If it is false, nothing
   changes and the plan is **re-computed on every later step**, because more results age out.
5. On success, `_next_compaction_at = context + compaction_step_tokens` (100k), `apply_compaction(plan)`
   runs, and then `context_reset_hook()` (Sorcar: `useful_tools.forget_reads`, so the Read tool's
   "unchanged since your earlier Read" dedupe does not point at stubbed text).

## What gets stubbed (`plan_compaction`)
- Only tool results **before the last assistant turn**. Results the model has not yet seen are never touched.
- The newest `KEEP_RECENT_TOOL_RESULTS = 6` results are kept.
- Results of `MIN_COMPACT_CHARS = 500` characters or fewer are skipped, and so are results already starting with `[compacted tool output:`.
- `PROTECTED_TOOLS = {"Edit", "Write", "finish"}` are never stubbed. Tool names come from call ids.
- Three conversation shapes are handled: Chat-Completions `role: tool`, Anthropic `tool_result` blocks
  (a string or a list of text parts), and OpenAI Responses/Gemini `function_call_output`.
- The stub (`make_stub`) reads `[compacted tool output: N chars; the first 200 follow. To see it again, Read
  the file or re-run a read-only command; do not repeat a command that has side effects.]` followed by the first 200 chars.
- Edits happen **in place and never change the list length**, so `_llm_hook_conversation_index` stays valid.
  `KISSAgent.messages` (the trajectory) and persisted events keep the full text.

## Cache-economics gate (`should_compact`)
Any edit is a full prompt-cache miss from the first edited message (for example, cache write 1.25x vs
read 0.1x of the input price). An audit found many compactions cost more than they saved, which is why the gate exists:
- `dropped_tokens >= MIN_DROP_FRACTION (0.25) * context_tokens`, and
- `context_tokens < HANDOFF_PROXIMITY_FRACTION (0.8) * handoff_tokens`. Close to hand-off there are too
  few steps left to pay back the miss. `handoff_tokens = CONTEXT_LIMIT_FRACTION * max_context`, and the
  second check is skipped when the window is unknown.

`compact_tool_results()` is the version without the gate (plan and apply). The agent does not use it.

## Gotcha for models reading stubs
The model has to re-run a read-only call to get a stubbed output back. This is why the stub text warns
against repeating side-effecting commands.

## Sources
- `src/kiss/core/context_compaction.py` (`plan_compaction`, `should_compact`, `apply_compaction`, `make_stub`, constants)
- `src/kiss/core/kiss_agent.py` (`KISSAgent._maybe_compact_conversation`, `_handoff_context_tokens`)
- `src/kiss/core/config.py` (`tool_output_compaction`, `compaction_start_tokens`, `compaction_step_tokens`)
- `src/kiss/tests/core/test_context_compaction.py`
