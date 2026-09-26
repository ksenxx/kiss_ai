---
title: 'kiss.server.sorcar.run: synchronous Python client for the daemon (daemon_client)'
uuid: 4ebebb55-2668-4153-bc25-7b7d11ba020b
summary: 'sorcar.run / daemon_client.run: submit a task to running kiss-web over sorcar.sock
  (KISS_SORCAR_SOCK), api-<hex> tab, taskId token, TaskResult, stop_on_timeout, closeTab.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# kiss.server.sorcar.run: synchronous Python client for the daemon (daemon_client)

`kiss.agents.sorcar.daemon_client.run` is the implementation, re-exported as `kiss.server.sorcar.run`. It lives in the sorcar layer so that sorcar-layer code can submit tasks without importing the server: the `run_agent` dispatch tool (`agent_dispatch.py`) and the cron scheduler. It needs an ALREADY RUNNING `kiss-web`. It never starts one.

```python
from kiss.server import sorcar
r = sorcar.run("Summarize README.md", work_dir="/repo")
r.text, r.success, r.cost, r.tokens, r.steps, r.chat_id, r.task_id
sorcar.run("Now fix the typos", chat_id=r.chat_id)   # continue the chat
```

## Parameters (keyword-only after `prompt`)
`work_dir`, `scope_work_dir`, `parent_task_id`, `parent_tab_id`, `parent_reviewer`, `side_channel`, `model`, `chat_id`, `system_prompt`, `tools` (path of a tools file), `extension_agent_path` (SEA script), `use_worktree=True`, `auto_commit=True`, `max_budget`, `model_config`, `use_web_tools`, `classify_tasks`, `use_memory` (None = daemon default), `is_parallel=True`, `append_basic_tools=True`, `append_to_system_prompt`, `append_to_prompt`, `tool_profile`, `docker_image`, `timeout=3600.0`, `stop_on_timeout=False`, `sock_path`.

An empty prompt raises `ValueError`. The tools file and agent path are validated and resolved on the client (`resolve_tools_file`, `resolve_agent_path`), and the daemon loads them (see `seas-agent-script-contract`).

## Socket resolution (`_resolve_sock_path`)
The explicit `sock_path` wins, then env `KISS_SORCAR_SOCK`, then `$KISS_HOME/sorcar.sock` (`_default_kiss_dir()`). With no daemon listening, `ConnectionError` is raised.

## Protocol
1. A new tab id `api-<uuid hex>` and a run token (uuid hex) are minted.
2. It sends one `run` line with wire fields (`tabId`, `taskId`=run token, `prompt`, `workDir`, `toolsFile`, `agentPath`, `useWorktree`, `autoCommit`, ...).
3. It reads newline-delimited events, keeping only those whose `tabId` equals its tab. `clear` gives `chat_id`. Any non-status event with `taskId` gives the persisted task id. `result` is remembered.
4. It returns when `status running=false` arrives after a `status running=true`. It does NOT return on `result`, because persistence, auto-commit and worktree cleanup run after it.
5. `finally`: if the call aborted with anything except `TimeoutError` (including the caller's `KeyboardInterrupt`), it sends `stop` with the run token. It always sends `closeTab` for its tab.

`_MAX_LINE_BYTES` (64 MiB) must match the daemon's frame limit. A daemon that closes the socket mid-task (for example the UDS drain timeout during a loop stall, see `server-broadcast-routing`) surfaces as `ConnectionError("... closed the connection before the task finished")`.

## Timeouts
- The default (`stop_on_timeout=False`): on expiry the caller gets `TimeoutError` and the task keeps running on the daemon.
- `stop_on_timeout=True` (used by `run_agent`, whose workspace reservation is released on return): it sends `stop`, then keeps reading up to `_STOP_CONFIRM_GRACE_SECONDS = 20` for the terminal status. If a SUCCESSFUL result plus the terminal status raced the stop, the finished `TaskResult` is returned. With no confirmation it raises `StopUnconfirmedTimeoutError` (a `TimeoutError` subclass).

## Why the run token
A synthetic tab can be reused after a run ends. The daemon applies a `stop` carrying `taskId` only to the state whose `client_run_token` matches, so a late cascade stop cannot kill a newer run.

## Sources
- `src/kiss/agents/sorcar/daemon_client.py` (`run`, `TaskResult`, `_resolve_sock_path`, `_MAX_LINE_BYTES`, `_STOP_CONFIRM_GRACE_SECONDS`, `StopUnconfirmedTimeoutError`, `_send_stop`)
- `src/kiss/server/sorcar.py` (module docstring, re-exports)
- `src/kiss/agents/sorcar/agent_dispatch.py` (caller)
