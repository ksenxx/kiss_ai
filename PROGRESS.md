# Task: Any ChatSorcarAgent must be registered with the server

## Status: DONE

## Invariant

Every live `ChatSorcarAgent` instance must be discoverable through some
entry of `_RunningAgentState.running_agent_states` such that
`state.agent is self`.

Consumers of this invariant:

- `VSCodeServer._reattach_running_chat` — subscribes a freshly opened
  viewer tab to an in-flight chat.
- `VSCodeServer._get_running_task_ids` — renders the running indicator
  next to in-flight tasks in the History sidebar.
- `ChatSorcarAgent._run_tasks_parallel` — scans the registry for
  `state.agent is self` to resolve the parent's `parent_tab_id` so
  sub-agent `new_tab` broadcasts carry a routing hint that keeps
  phantom sub-agent tabs out of unrelated webviews.

## Where it was broken

- UI launches — OK (server's `_run_task_inner` registers).
- Worktree CLI launches — OK
  (`WorktreeSorcarAgent.run` called `_register_running_state`).
- Parallel sub-agents — OK (`_run_tasks_parallel` registers).
- **Plain `ChatSorcarAgent` runs (CLI / third-party / remote
  webapp) — BROKEN.** Nothing inserted a `_RunningAgentState` for
  the running agent, so the scan above returned no match.

## Fix

1. Moved `_register_running_state` / `_unregister_running_state` from
   `WorktreeSorcarAgent` into `ChatSorcarAgent`. They are idempotent —
   skip insertion when an entry whose `chat_id == self._chat_id` is
   already present.
1. Called `registered_here = self._register_running_state()` in
   `ChatSorcarAgent.run()` immediately after the `chat_id` is minted,
   and called `self._unregister_running_state()` in the same method's
   `finally` block when `registered_here` is `True`.
1. Removed the now-redundant duplicate method definitions from
   `WorktreeSorcarAgent` (still called from `WorktreeSorcarAgent.run`
   via inheritance). Removed the orphaned `_RunningAgentState` import
   from that module.

## Integration tests added

`src/kiss/tests/agents/sorcar/test_chat_agent_state_registration.py`
— 4 tests, all pass:

- `test_chat_agent_self_registers_during_run` — confirms the entry is
  present mid-`run()` with `state.agent is agent`,
  `state.chat_id == agent.chat_id`, `state.is_task_active is True`.
- `test_registry_cleaned_up_after_run` — confirms `finally` removes
  the entry.
- `test_registry_cleaned_up_when_run_raises` — confirms `finally`
  cleans up on the error path.
- `test_parent_tab_id_lookup_succeeds_for_cli_parent` — exact
  reproduction of the `_run_tasks_parallel` parent-tab-id scan,
  asserts it returns a non-empty id for a CLI parent.

## Verification

- New tests: 4 pass.
- Related sweep (existing): 24 pass
  (`test_running_agents_registry`, `test_running_agent_state_on_run`,
  `test_running_agent_states_dict_race`,
  `test_cli_running_task_history_dot`,
  `test_cli_history_click_resumes_live_stream`,
  `test_cli_daemon_live_stream`,
  `test_chat_parallel_new_tab_stale_task_id`,
  `test_autocommit_off_on_failure`, `test_ask_user_immediate_response`).
- `uv run check --full` — all green.
