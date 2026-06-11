# Task: investigate why task 3515 ("uv run check --full and fix") failed, reproduce with a test, fix

## Investigation

1. `~/.kiss/kiss-web-stderr.log`: task 3515 started 17:30:09; at 17:30:51 the daemon
   (pid 75085) logged `Signal SIGTERM received ... active_tasks=[c93785cb...]` and shut
   down, persisting "Task interrupted by server restart/shutdown". launchd (`KeepAlive`)
   respawned a fresh daemon (pid 75594) at 17:30:55.
1. `sqlite3 ~/.kiss/sorcar.db` events for task 3515: the agent ran
   `uv run check --full` in its worktree; the check completed (mdformat failed — what the
   agent was about to fix) and the interrupt arrived right after, before step 3's LLM call
   returned.
1. LaunchAgent plist `com.kiss.web-server` has NO WatchPaths → launchd did not initiate.
1. macOS unified log at 17:30:51.548: `launchctl[75589] launchctl kickstart: <private>`
   then `launchd: [gui/501/com.kiss.web-server [75085]:] signaled service: Terminated: 15`.
   So someone ran `launchctl kickstart -k gui/501/com.kiss.web-server`.
1. The only `launchctl kickstart` callers in the codebase:
   `kiss/agents/vscode/commands.py:_restart_kiss_web_daemon()` (called from
   `_cmd_save_config` whenever `cfg["remote_password"]` is non-empty) and
   `DependencyInstaller.ts` (writes "Dependency check started" to `~/.kiss/install.log` —
   no such entry at 17:30, ruled out).
1. Frontend `main.js` auto-posts `saveConfig` passively: `closeSettingsPanel()` and
   blur/change/Enter on the remote-password inputs call `saveSettingsIfPopulated()`,
   echoing back the saved non-empty `remote_password`.
1. Root cause: `_cmd_save_config` ends with
   `if cfg.get("remote_password", ""): _restart_kiss_web_daemon()` — it restarts the
   daemon even when the password is UNCHANGED, killing every in-flight task.

## Fix plan

- `commands.py::_cmd_save_config`: capture `prev_cfg = load_config()` before saving and
  only restart when the password is non-empty AND differs from the previous one.
- Integration test `src/kiss/tests/agents/vscode/test_save_config_daemon_restart.py`:
  FakeServer(\_CommandsMixin) pattern + a stub `launchctl`/`systemctl` on a private PATH
  that records invocations to a marker file (never touches the real daemon).
  Reproduced the bug first (unchanged-password test fails pre-fix), then fixed.

## Steps done

- Wrote test, confirmed `test_unchanged_password_does_not_restart_daemon` FAILS pre-fix.
- Applied fix to commands.py; all 4 tests pass.
- Ran impacted vscode tests + `uv run check --full`.
