# PROGRESS

## Task

> Remove tests `TestCachePricing::test_openai_gpt5_cache_read_is_tenth` and
> `TestCachePricing::test_openai_pro_cache_read_uses_full_input_price`.
> Fix tests `TestLiveTaskIdFallback::test_overlay_live_metrics_prefers_agent_last_task_id`
> and `TestLiveTaskIdFallback::test_overlay_live_metrics_falls_back_to_task_history_id`.

## Status: DONE

## Work completed

1. Read `SORCAR.md` — empty, so no project-specific overrides apply.

1. Located the four affected tests:

   - `src/kiss/tests/core/models/test_model_implementations.py` —
     `TestCachePricing` at line 192, the two doomed tests at lines 213 and 334.
   - `src/kiss/tests/agents/vscode/test_simplification_lockdown_server.py` —
     `TestLiveTaskIdFallback` at line 248 with the two overlay tests at
     lines 285 and 305.

1. **Removed** `test_openai_gpt5_cache_read_is_tenth` (referenced missing
   `MODEL_INFO["gpt-5-codex"]`) and `test_openai_pro_cache_read_uses_full_input_price`
   (referenced missing `MODEL_INFO["gpt-5.5-pro"]`).

1. **Diagnosed the live-metrics failures.** `VSCodeServer._overlay_live_metrics`
   now writes the live `model`, `is_worktree`, `is_parallel`, and
   `auto_commit_mode` fields on top of the existing `tokens` / `cost` / `steps`
   overlay (see `src/kiss/agents/vscode/server.py:563-625`). The two tests
   used a strict `dict ==` comparison against a metrics-only expected value,
   so they broke when the overlay widened.

1. **Fixed both tests** by setting deterministic values for the new
   attributes (`agent.model_name`, `tab.use_worktree`, `tab.use_parallel`,
   `tab.auto_commit_mode`) and asserting the full expected dict, preserving
   the lockdown semantics:

   ```python
   matched: dict = {"tokens": 0, "cost": 0.0, "steps": 0}
   server._overlay_live_metrics(matched, 7)
   assert matched == {
       "tokens": 555, "cost": 1.25, "steps": 9,
       "model": "test-model",
       "is_worktree": True, "is_parallel": False, "auto_commit_mode": False,
   }
   ```

1. Ran the impacted suites:

   - `pytest src/kiss/tests/agents/vscode/test_simplification_lockdown_server.py::TestLiveTaskIdFallback`
     → 4 passed.
   - `pytest src/kiss/tests/core/models/test_model_implementations.py::TestCachePricing`
     → 19 passed (down from 21 after removing the two deletions).

1. Ran `uv run check --full`: ruff, mypy, and pyright all pass. mdformat
   flagged a pre-existing formatting drift in `PROGRESS.md` which is rewritten
   here.
