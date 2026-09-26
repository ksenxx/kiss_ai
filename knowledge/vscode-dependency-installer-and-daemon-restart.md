---
title: DependencyInstaller - first-run setup and kiss-web daemon restart
uuid: 2362a03f-7ccc-49ed-bee1-75768a2d9c7c
summary: DependencyInstaller.ensureDependencies - uv, .venv, Chromium, API keys; kiss-web
  daemon restart via fingerprint, restart lock, restart-pending file, systemd/launchd.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# DependencyInstaller - first-run setup and kiss-web daemon restart

`src/DependencyInstaller.ts` (~3k lines) runs on every activation via `ensureDependencies()` (single-flight: concurrent callers share `pendingDeps`). Everything is logged to `<KISS_HOME>/install.log` (`~/.kiss/install.log`).

## Locating the Python project
`findKissProject()` (`src/kissPaths.ts`) returns the first directory whose `pyproject.toml` contains `name = "kiss`: `$KISS_PROJECT_PATH`, then the `kissSorcar.kissProjectPath` setting (both only in a trusted workspace), then the bundled `<extension>/kiss_project` (created by `copy-kiss.sh`). `findUvPath()` checks `~/.local/bin/uv`, `~/.cargo/bin/uv`, `/usr/local/bin/uv`, `/opt/homebrew/bin/uv`, then `which uv` (`execFileSync` with a 5 s timeout and `SIGKILL`).

## `ensureDependenciesImpl` flow
1. **Nothing-to-do fast path**: uv found, `.venv` exists, Chromium installed, daemon running, and neither `~/.kiss/.extension-updated` nor `~/.kiss/.kiss-web.restart-pending` exists -> only load API keys from the shell rc.
2. If Python in `.venv` is older than 3.13 (`MIN_PYTHON_MAJOR/MINOR`), delete `.venv` to recreate it.
3. `.extension-updated` present -> delete it and plan an "Installation complete" notification.
4. **uv + venv present**: install Playwright Chromium in the background (`install-deps` too on Linux), install git / the `code` CLI if missing, then `runFinalization`.
5. **Otherwise** a progress notification: install uv (pinned `UV_VERSION`, sha256-verified download), git, `code` CLI, `uv sync`, check Python, Playwright Chromium, then `runFinalization`.

`runFinalization` installs the CLI scripts (`installCliScript`), copies `MODEL_INFO.json`, installs cloudflared if needed, calls `restartKissWebDaemon`, then `ensureApiKeys()` (prompts if none of Claude Code, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` is available; writes the shell rc atomically) and `ensureRemotePassword`. Cross-window prompts use lock files in `~/.kiss` (`.api-keys.lock`, `.remote-password.lock`).

## Daemon restart (`restartKissWebDaemon`)
- Skipped on Windows and when `.venv/bin/kiss-web` is missing.
- `acquireDaemonRestartLock()` (`~/.kiss/.kiss-web.restart.lock`; a dead owner is reclaimed at once, an unreadable owner after 120 s, a live owner only after 600 s) makes only one window restart at a time; others skip.
- **Fingerprint** (`computeKissWebFingerprint`): sha256 of the `kiss-web` binary + the work dir + the newest mtime of any `.py` under `src/kiss` (excluding `__pycache__` and `tests`), stored in `~/.kiss/.kiss-web.fingerprint`.
- It probes TCP `127.0.0.1:8787` (`probeDaemonHealth`: alive/dead/unknown) and asks the Unix socket for active tasks (`daemonHasActiveTasks`), then `decideRestart` (`src/daemonHealth.js`):
  - `force` -> restart (`forced-by-user`);
  - active tasks > 0 -> skip (`active-tasks`);
  - alive but socket file missing -> restart;
  - alive but task probe failed -> skip (`alive-uncertain`);
  - fingerprint matches and not dead -> skip (`healthy-unchanged`);
  - else restart.
- A skipped restart with a fingerprint mismatch writes `~/.kiss/.kiss-web.restart-pending` (`markRestartPending`) and arms a retry timer (`RESTART_RETRY_MS`, 60 s, env `KISS_RESTART_RETRY_MS`). With active tasks it also offers a "restart now" notification (`offerForcedRestart`). Without the pending file, a deferred restart used to be lost forever because the update marker had already been deleted.
- Restart itself: kill whatever listens on port 8787, then per platform: macOS writes `~/Library/LaunchAgents/com.kiss.web-server.plist` and restarts it (`restartLaunchAgent`, `src/macLaunchd.js`); Linux writes `~/.config/systemd/user/kiss-web.service` (`Restart=always`) and restarts it with `systemctl --user`, falling back to spawning `kiss-web` directly (`spawnKissWebDirect`). `KISS_HOME` is propagated into the unit/plist; `KISS_SORCAR_SOCK` is not (client-side only).
- `verifyDaemonStartup` (`src/daemonRestartVerify.js`) polls up to 180 s, re-issuing the restart every 15 s; on success the new fingerprint is written and the pending file cleared.

## Operational gotchas
- A Sorcar task runs inside the kiss-web daemon; it must never restart or kill the daemon itself.
- Until a deferred restart happens, the daemon runs old Python modules while serving new `media/` files from disk, which produces mismatches between the two (see `vscode-remote-webapp-relation`).

## Sources
- `src/kiss/agents/vscode/src/DependencyInstaller.ts` (`ensureDependencies`, `ensureDependenciesImpl`, `runFinalization`, `restartKissWebDaemon`, `restartKissWebDaemonLocked`, `computeKissWebFingerprint`, `acquireDaemonRestartLock`, `markRestartPending`, `armRestartRetry`, `offerForcedRestart`, `ensureApiKeys`)
- `src/kiss/agents/vscode/src/daemonHealth.js` (`decideRestart`, `probeDaemonHealth`, `daemonHasActiveTasks`)
- `src/kiss/agents/vscode/src/daemonRestartVerify.js` (`verifyDaemonStartup`), `src/macLaunchd.js` (`restartLaunchAgent`)
- `src/kiss/agents/vscode/src/kissPaths.ts` (`findKissProject`, `findUvPath`)
- `src/kiss/agents/vscode/test/daemonRestartPendingRetry.test.js`
