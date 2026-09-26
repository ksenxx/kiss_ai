---
title: Cross-platform process helpers (pid_alive, popen_process_group, kill_process_group,
  process_identity)
uuid: c1540d8e-ce87-4ec6-843d-da6974324565
summary: 'kiss.core.processes: use pid_alive, popen_process_group, kill_process_group,
  process_identity instead of os.kill(pid, 0) or os.killpg; Windows Job Objects; pid
  reuse safety'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Cross-platform process helpers

`src/kiss/core/processes.py` is the single place for liveness probes and "stop this child and all its
descendants". The POSIX idioms are **harmful** on Windows, not just missing: `os.kill(pid, 0)` there
means `CTRL_C_EVENT` and sends Ctrl+C to every process on the console, including the caller, and
`os.killpg`/`start_new_session` do not exist. Rule: never call those directly; use this module.

## API
- `IS_WINDOWS`, `SIGKILL` (falls back to `SIGTERM` where `SIGKILL` is missing).
- `pid_alive(pid)`: `pid <= 0` is always False (0 and negatives address groups, never signal them).
  POSIX: `os.kill(pid, 0)`; `PermissionError` means alive (another user's process). Windows:
  `OpenProcess` + `GetExitCodeProcess == STILL_ACTIVE`; access denied counts as alive.
- `new_process_group_kwargs()`: `{"start_new_session": True}` on POSIX,
  `CREATE_NEW_PROCESS_GROUP` on Windows.
- `popen_process_group(*args, **kwargs)`: `subprocess.Popen` with those kwargs; on Windows also
  assigns the child to a new Job Object (descendants inherit it), tracked in `_WINDOWS_JOBS`, with
  stale handles closed on later enrolments. Prefer this over the bare kwargs.
- `kill_process_group(pid, sig=SIGTERM)`: POSIX `os.killpg(pid, sig)`. Windows ignores `sig`:
  `TerminateJobObject` for children started by `popen_process_group`, else
  `taskkill /F /T /PID` (15 s timeout). Raises `ProcessLookupError` when the group is gone and
  `OSError` when Windows could not terminate a still-alive tree, so callers fall back to
  `proc.kill()` as after a failed `killpg`.
- `process_identity(pid)`: a fingerprint a recycled pid cannot share. POSIX:
  `ps -ww -p PID -o lstart= -o command=` (5 s timeout); Windows: creation time + full image path.
  Record it when capturing a pid and compare before every signal so a stranger that inherited the pid
  is never killed. Returns `None` when the process is gone.

## Users
`vscode_config` (sourcing shell RCs for key migration), `core/models/model.py`,
`agents/sorcar/useful_tools.py`, `_concurrency.py`, `git_worktree.py`, `cron_agent.py`,
`web_use_tool.py`, `server/web_server.py`, and the Signal/WhatsApp/muse_auth third-party agents.

## Sources
- `src/kiss/core/processes.py` (`pid_alive`, `new_process_group_kwargs`, `popen_process_group`, `kill_process_group`, `process_identity`, `IS_WINDOWS`, `SIGKILL`)
