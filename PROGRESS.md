# PROGRESS

## Task

> Check why the last task failed. Fix the issue. Use internet search
> extensively.

## Status: DONE

## Root cause

The previous task failed with:

```
Error code: 400 - {'error': {'message': 'Function tools with
reasoning_effort are not supported for gpt-5.5 in
/v1/chat/completions. Please use /v1/responses instead.', ...}}
```

`src/kiss/core/models/openai_compatible_model.py::__init__` defaults
`reasoning_effort="xhigh"` for the gpt-5.5 family (via
`MODEL_INFO[...].thinking`). Sorcar's agentic loop always sends a tools
list, and OpenAI's `/v1/chat/completions` rejects `tools` +
`reasoning_effort` for GPT-5.x / o-series reasoning models — it requires
the new `/v1/responses` API instead.

## Fix

Minimal, non-invasive: in
`generate_and_process_with_tools()`, drop `reasoning_effort` from
`kwargs` when `tools` are attached. The no-tools `generate()` path keeps
the xhigh default. Migrating the whole transport to `/v1/responses`
would be a major rewrite (different request/streaming/tool schemas)
and is out of scope.

## Verification

- `uv run pytest src/kiss/tests/core/models/test_openai_xhigh_default.py -v` → 15 passed (12 existing + 3 new).
- `uv run check --full` → all checks pass.

## New tests added

`src/kiss/tests/core/models/test_openai_xhigh_default.py`:

- `test_reasoning_effort_dropped_when_tools_attached`
- `test_reasoning_effort_kept_when_no_tools`
- `test_openrouter_gpt_5_5_tools_strips_reasoning_effort`
