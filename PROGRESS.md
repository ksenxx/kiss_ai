# Task: Any ChatSorcarAgent must be registered with the server (invariant)

## Status: PLANNING (no code edits yet)

## What "registered with the server" means

There are TWO process-global registries in this codebase:

1. **`ChatSorcarAgent.running_agents: dict[int, ChatSorcarAgent]`** (file:
   `src/kiss/agents/sorcar/chat_sorcar_agent.py`, declared L54).
   Keyed by `task_history.id`. Populated at L385 of
   `ChatSorcarAgent.run()`:

   ```python
   ChatSorcarAgent.running_agents[task_id] = self
   ```

   Removed at L535 in the `finally`:

   ```python
   ChatSorcarAgent.running_agents.pop(task_id, None)
   ```

   Existing regression test: `test_running_agents_registry.py`.

1. **`_RunningAgentState.running_agent_states: dict[str, _RunningAgentState]`**
   (file: `src/kiss/agents/sorcar/running_agent_state.py` L72).
   Keyed by frontend tab id. Holds a `_RunningAgentState` whose
   `state.agent` may be a (`WorktreeSorcarAgent`/`ChatSorcarAgent`)
   instance. Used by:

   - `VSCodeServer._reattach_running_chat` to subscribe a fresh
     viewer tab to an in-flight UI-launched task.
   - `VSCodeServer._get_running_task_ids()` (running indicator in
     History sidebar) — this iterates `state.agent._last_task_id` etc.
   - Parallel sub-agent flow inside
     `ChatSorcarAgent._run_tasks_parallel` (L266-317) — creates a
     fresh `_RunningAgentState` per sub-agent with `state.agent = sub_agent`
     and registers/unregisters around the run.

## Where the invariant breaks

The user's task is "Any ChatSorcarAgent must be registered with the
server". The natural production-correct reading:

For every `ChatSorcarAgent` instance whose `run()` is currently
active, there must exist a `_RunningAgentState` entry such that
`state.agent is the_agent` (i.e. the server can find the running
agent via the per-tab state registry).

This invariant IS held for:

- UI-launched top-level tasks — `VSCodeServer._TaskRunnerMixin._run_task_inner`
  installs the `_RunningAgentState` with `state.agent = agent` BEFORE
  calling `agent.run(...)` and clears it in `finally`.
- Parallel sub-agents — `_run_tasks_parallel` explicitly does
  `_RunningAgentState.register(sub_tab_id, sub_state)` (L276) where
  `sub_state.agent = agent` (L271), and unregisters in `finally` (L317).

This invariant is BROKEN for:

- **CLI-launched top-level tasks** — `worktree_sorcar_agent.main()` →
  `run_with_steering(agent, run_kwargs)` → `agent.run(...)` without
  ever creating a `_RunningAgentState` for the agent. CLI runs only
  populate `ChatSorcarAgent.running_agents`, not the per-tab state
  registry. Previous task fixes added the `_cli_running_tasks: set[int]`
  side-channel on `RemoteAccessServer` (populated via UDS
  `cliTaskStart`/`cliTaskEnd`) but the CORE in-process invariant
  itself — that a running agent appears in the state registry — is
  still broken.
- Possibly **third-party channel agents** and **remote webapp**
  invocations that drive `ChatSorcarAgent` directly. Need to verify.

Note that the CLI runs in a SEPARATE process from the daemon's
VS Code server, so the "register with the server" must be read as
"with the per-process `_RunningAgentState` registry" — because that
is what `ChatSorcarAgent._run_tasks_parallel` (which runs IN the
CLI process) consults at L240-244 to discover the parent's
`parent_tab_id` for new_tab fan-out.

Concretely: in the CLI process, when an agent runs, NO entry is in
`_RunningAgentState.running_agent_states`, so the loop at
`_run_tasks_parallel` L241-244 finds nothing and `parent_tab_id`
stays "" — meaning sub-agent `new_tab` broadcasts lose the
`parent_tab_id` routing hint and webviews bound to other chats
materialise phantom tabs. This is a concrete observable bug
caused by the broken invariant.

## Plan

1. Read `worktree_sorcar_agent.py` (small file) to see exactly how
   `agent.run(...)` is invoked from `main()` and where the missing
   `_RunningAgentState` registration ought to happen.
1. Check the third-party channel agents and webapp invocations for
   the same gap.
1. Write integration tests that fail before the fix:
   - **Test A**: instantiate a `ChatSorcarAgent` (CLI-style), patch
     its `super().run` to assert that during the run the agent IS
     in `_RunningAgentState.running_agent_states` keyed by SOME tab
     id with `state.agent is agent`. Before fix this assertion
     fails. Place: `src/kiss/tests/agents/sorcar/test_chat_agent_state_registration.py`.
   - **Test B**: CLI parallel sub-agent scenario — parent CLI agent
     calls `_run_tasks_parallel` with one task; assert the spawned
     sub-agent observes a non-empty `parent_tab_id` in its
     `_subagent_info`. Before fix this is "". (Will need to read
     the parent-tab-id lookup carefully — at L240-244 it scans
     `running_agent_states` for `state.agent is self`. So the test
     is: spawn `_run_tasks_parallel` from a parent
     `ChatSorcarAgent` and assert the sub-agent's `_subagent_info["parent_tab_id"]`
     is non-empty.)
1. Implement the fix in `ChatSorcarAgent.run()`:
   - In `run()`, when there is no pre-existing `_RunningAgentState`
     for this agent (not a UI launch, not a sub-agent that was
     pre-registered), register one ourselves at the start of the
     run and unregister in the `finally`. Tab id: a stable
     synthetic id like `f"cli-{task_id}"` derived from the freshly
     allocated `task_history` id. State has `agent = self`,
     `chat_id`, `last_user_prompt`, `is_task_active = True`.
   - Detect "already registered" by scanning the registry under
     `_registry_lock` (mirroring L240-244) — if any state has
     `state.agent is self`, skip self-registration.
1. Re-run integration tests; they should pass.
1. Run `uv run check --full`.
1. Clean up `tmp/`.

## Files to read in next session (small, targeted)

- `src/kiss/agents/sorcar/worktree_sorcar_agent.py` (CLI main + run
  override).
- `src/kiss/agents/vscode/server.py` `_TaskRunnerMixin._run_task_inner`
  to confirm exact existing registration pattern, so the CLI-side
  self-registration mirrors it.
- `src/kiss/tests/agents/sorcar/test_running_agents_registry.py`
  as the template for the new test harness.

## Snippets needed when implementing

Existing parallel-subagent self-registration (good model — but it
runs in the SUB-thread, not at `run()` entry):

```python
sub_state = _RunningAgentState(sub_tab_id, model or "")
sub_state.chat_id = chat_id
sub_state.is_task_active = True
sub_state.agent = agent
_RunningAgentState.register(sub_tab_id, sub_state)
try:
    ...
finally:
    _RunningAgentState.unregister(sub_tab_id)
```

`run()` registration block to add (sketch, runs INSIDE `run()`
after `task_id = _add_task(...)`):

```python
# Self-register so the per-tab state registry invariant holds
# even for CLI / third-party invocations that never went through
# VSCodeServer._run_task_inner.
self_state_tab_id: str | None = None
with _RunningAgentState._registry_lock:
    already = any(
        st.agent is self
        for st in _RunningAgentState.running_agent_states.values()
    )
if not already:
    self_state_tab_id = f"cli-{task_id}"
    st = _RunningAgentState(self_state_tab_id, self.model_name or "")
    st.chat_id = self._chat_id
    st.last_user_prompt = self._last_user_prompt
    st.is_task_active = True
    st.agent = self  # type: ignore[assignment]
    _RunningAgentState.register(self_state_tab_id, st)
```

And in `finally`:

```python
if self_state_tab_id is not None:
    _RunningAgentState.unregister(self_state_tab_id)
```

## Why this is safe

- `_RunningAgentState.agent` is typed `WorktreeSorcarAgent | None`
  but is documented as transient and overwritten in multiple paths
  by `ChatSorcarAgent` directly (see `_run_tasks_parallel` L271:
  `sub_state.agent = agent  # type: ignore[assignment]`). So
  assigning `ChatSorcarAgent` here is consistent with existing
  practice — we'll use the same `# type: ignore[assignment]`.
- UI launches already register via `_run_task_inner` BEFORE calling
  `agent.run()`. The `already` check prevents double registration.
- Sub-agents already register before calling `.run()`. Same — the
  `already` check skips self-registration.

## Context note

This session burnt most context reading large files. Resume from
this PROGRESS.md without re-reading `chat_sorcar_agent.py` from
scratch — use the snippets above.
