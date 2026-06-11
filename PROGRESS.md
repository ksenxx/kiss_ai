# Task: Diagnose why the previous bug-hunt task was interrupted; reproduce + fix

User report: the long-running bug-hunt orchestrator task ended with
"Task terminated unexpectedly (process killed)". Analyze logs/DB events,
reproduce with an integration test, fix.

## Forensics (logs + sorcar.db)

Timeline reconstructed from `~/.kiss/kiss-web-stderr.log` and
`~/.kiss/sorcar.db`:

1. The orchestrator task (task_history row 3556, chat 8e0f9308) was running
   bug-hunt iteration 7 with 7 parallel sub-agents (rows 3618-3624) under the
   kiss-web daemon pid 2884 (started 2026-06-10 20:00:42).
1. At **2026-06-11 00:37:45.657 pid 2884 received SIGTERM** with all 8 tasks
   in flight (`Signal SIGTERM received: ... active_tasks=[0e452e0c(task=None), task-3556__sub_0(task=3623), ...]`). A replacement daemon (pid 88382)
   started 6 seconds later — a daemon **restart**, not a crash.
1. **Who sent the SIGTERM**: sub-agent group E (row 3624) ran
   `uv run pytest src/kiss/tests/agents/vscode/test_bughunt7_config_junk_types.py`
   at **00:37:39 ("verify tests fail pre-fix") and again at 00:37:45** — the
   exact second of the SIGTERM. That pre-fix test drives `_cmd_save_config`
   with a junk `remote_password` (123456) against an in-process server;
   pre-fix the handler judged it a genuine password change and called
   `commands._restart_kiss_web_daemon()`, which runs
   `launchctl kickstart -k gui/<uid>/com.kiss.web-server` — **restarting the
   developer's real kiss-web daemon and killing the entire task tree that was
   running the test** (self-inflicted kill). The earlier 00:37:39 run is when
   the first SIGTERM-causing invocation happened; the daemon logged the
   signal at 00:37:45 on the second run.
1. Damage trail: the shutdown's pre-emptive persist
   (`_shutdown_persist_in_flight_results`) covered no rows (parent row 3556
   had already been rewritten by another daemon's startup orphan sweep —
   the documented cross-process behavior — so it no longer carried the
   sentinel; sub-agent rows were excluded because their registry tabs have
   no worker thread). On the next startups the orphan sweep rewrote the six
   surviving sentinel rows (3618-3623) and row 3556 to "Task terminated
   unexpectedly (process killed)".
1. **Row 3624 data loss**: that sub-agent's `ChatSorcarAgent.run` `finally`
   DID run during the shutdown's KeyboardInterrupt and persisted
   `result=""` — because only `except Exception` (not `KeyboardInterrupt`,
   a `BaseException`) rewrites `result_summary`. The empty string replaced
   the "Agent Failed Abruptly" sentinel, so the orphan sweep (which matches
   the sentinel exactly) can never repair the row: task 3624 shows an empty
   result forever.

## Bugs fixed (commit 8f44a9b1)

1. **Self-inflicted daemon kill** — `commands.py:_restart_kiss_web_daemon`
   now returns `bool` and refuses to kick the system LaunchAgent when
   `KISS_HOME` points at a non-default location (new helper
   `_kiss_home_is_default()`); every pytest process (conftest sets
   `KISS_HOME` to a temp dir) and any sandboxed run is therefore incapable
   of restarting the user's live daemon.
1. **Junk-typed config values** — implemented
   `vscode_config.sanitize_config()` (the fix the killed session had
   specified in its pre-fix test but never wrote): coerces every
   DEFAULTS-keyed value to its expected type; applied in `load_config`,
   `save_config`, and `_cmd_save_config` (before the password-change
   comparison, so a junk `remote_password` can no longer count as a genuine
   change and trigger a restart).
1. **Interrupted-run empty-result data loss** —
   `chat_sorcar_agent.py:ChatSorcarAgent.run` gained an
   `except BaseException` branch persisting the explicit
   `"Task interrupted"` marker instead of `""` on
   KeyboardInterrupt/SystemExit, keeping interrupted rows meaningful (the
   task-3624 incident).

## Tests (failing-first, all verified)

- `src/kiss/tests/agents/vscode/test_restart_guard_nondefault_home.py`
  (4 tests): genuine password change under a non-default `KISS_HOME` must
  not restart the daemon (reproduces the incident safely via stub
  `launchctl`/`systemctl` on a private PATH); helper returns False under
  custom home, True under default/unset home (controls). All 4 failed
  pre-fix.
- `src/kiss/tests/agents/sorcar/test_interrupted_result_not_empty.py`
  (3 tests): KeyboardInterrupt and SystemExit must persist
  "Task interrupted" (2 failed pre-fix); RuntimeError keeps "Task failed"
  (regression guard).
- Updated `test_save_config_daemon_restart.py` to simulate the default home
  (env flip; config writes stay isolated, restarts only reach PATH stubs).
- The killed session's leftover `test_bughunt7_config_junk_types.py`
  (6 tests) now passes with the implemented `sanitize_config`.

## Verification

- 18/18 tests across the four directly-affected files pass.
- Impacted sweep: 1214 tests across 100 test files (everything importing
  vscode_config/commands/ChatSorcarAgent) in 8 parallel shards — single
  failure `test_subagent_history_click.py::test_replay_subagent_does_not_invoke_reattach_running_chat`,
  reproduced identically with all changes stashed (pre-existing at the
  baseline commit from the killed session's dirty state; unrelated).
- `uv run check --full` clean.
