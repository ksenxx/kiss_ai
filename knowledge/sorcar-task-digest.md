---
title: 'task_digest: compact digests of a persisted task transcript'
uuid: 1466cb68-2987-4d26-a89f-7c3e11fdb224
summary: 'task_digest: turns a task''s sorcar.db events into numbered entries and
  renders transcript_page, overview and entry_detail for the /ask and task-update
  agents.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# task_digest

The persisted event stream of a task (`~/.kiss/sorcar.db`) contains thousands of streamed `text_delta` /
`thinking_delta` chunks and large tool arguments and outputs, far too much to give an LLM raw. `task_digest.py`
compresses it.

## Entries
`digest_events` produces numbered `Entry(kind, text, name)` records: one per tool call, tool result, coalesced
streamed run (`_StreamBuffer`: `thinking_delta`→THOUGHT, `text_delta`→ASSISTANT, `system_output`→OUTPUT),
progress summary, user message, `/ask` answer and final result. Display-only events (`system_prompt`,
`task_settings`, `new_tab`, `tasks_updated`, `subagentDone`, ...) are skipped (`_SKIPPED_EVENT_TYPES`).
A `summary` tool call becomes a **SUMMARY** entry holding its `description`. Tool-call arguments are
reassembled from both places they can be stored (`_tool_call_args`).

Per-kind clip limits (`_CLIP`): THOUGHT 400, ASSISTANT 800, OUTPUT 500, TOOL CALL 400, RESULT 500, and
SUMMARY/FINISH/TASK RESULT 4000 characters.

## Renderers
- `transcript_page(...)`: a clipped, optionally filtered page of entries (at most `MAX_PAGE = 400`). It is the
  `task_transcript` tool of the task-update SEA (`seas/task_update_sea.py`) and the `/ask` agent (`third_party_agents/ask_sea.py`).
- `overview(task_id)`: one-call orientation. It shows the header (`header_lines`: metadata, timing, spend), the sub-agent
  tasks (`child_tasks`), later user messages, the last 3 `/ask` answers, the task's own progress summaries (first 2 and
  last 10, each clipped to 1500 characters) and the latest 30 entries.
- `entry_detail(task_id, n)`: one entry in full, up to `MAX_DETAIL_CHARS = 100_000`, e.g. a complete command
  output or diff.

All three read the live database in-process and flush queued events first, so the newest steps of a running task
are visible.

## Sources
- `src/kiss/agents/sorcar/task_digest.py` (`digest_events`, `Entry`, `_StreamBuffer`, `transcript_page`, `overview`, `entry_detail`, `load_task`, `child_tasks`, `header_lines`)
- `src/kiss/agents/seas/task_update_sea.py`, `src/kiss/agents/third_party_agents/ask_sea.py` (consumers)
