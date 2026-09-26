---
title: Extension update check, Update button and automatic window reload
uuid: 096f76c4-13d2-49db-84af-07f75995ae1c
summary: UpdateChecker.js PyPI check (6h cooldown, 24h snooze), runUpdate running
  install.sh in a terminal, $KISS_HOME/.extension-updated marker, reloadGuard window
  reload.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Extension update check, Update button and automatic window reload

## Release check (`src/UpdateChecker.js`)
`checkForExtensionUpdate(opts)` is called once at activation:
- Current version: highest `ksenxx.kiss-sorcar-<version>` directory in the extensions root (`scanInstalledExtensionVersions`), else `src/kiss/core/_version.py` of the Python project.
- Latest version: `https://pypi.org/pypi/kiss-agent-framework/json` (15 s timeout).
- Cache: `<KISS_HOME>/.update-check.json`; no network call within 6 h (`DEFAULT_COOLDOWN_MS`). "Remind me later" calls `snoozeUpdateNotification` (24 h, `DEFAULT_SNOOZE_MS`); a release newer than the snoozed one still notifies (`isSnoozeActive`).
- The notification offers "Update now" (`sidebarView.runUpdate()`), "Update when idle" (forwards `updateWhenIdle` to the daemon) and "Remind me later".
All timing and fetch functions are injectable through `opts` for tests.

## Update button (`SorcarSidebarView.runUpdate`)
Reached from the notification or the settings panel (`runUpdate` message). It opens a VS Code terminal (`_openUpdateTerminal`) and runs, in order of preference:
1. `~/.kiss/kiss_ai/scripts/install.sh` (the locked bootstrap) with `KISS_NONINTERACTIVE=1`;
2. `~/.kiss/kiss_ai/install.sh` for an older clone without the bootstrap;
3. if there is no clone (`findInstallScript()` null, e.g. installed from a .vsix), the public curl bootstrap `bootstrapInstallUrl()` (override `KISS_UPDATE_BOOTSTRAP_URL`), which clones `~/.kiss/kiss_ai` first.
`KISS_HOME` is pinned to the extension host's value in the command line, because `install.sh` writes the reload marker into `$KISS_HOME` and the watcher looks at the host's `$KISS_HOME`. There is no per-window "already running" guard: `install.sh` holds a cross-process lock and the loser prints "another KISS update is already running".

## Reload marker (`extension.ts`)
- `install.sh` writes a UTC timestamp to `$KISS_HOME/.extension-updated` after installing the new VSIX.
- `extension.ts` watches it with `fs.watchFile(markerPath, {interval: 2000})`. A non-empty file with a new mtime starts `triggerReload()`: every 500 ms it calls `isReloadReady(out/extension.js, sorcar.sock, prevSize)` (`src/reloadGuard.js`). It reloads (`workbench.action.reloadWindow`) when the bundle size is stable and either the daemon socket exists or the code has been stable for 3 s, or after 15 s regardless.
- At the next activation the marker also forces the first-launch chat auto-open, `resetTipsOnExtensionUpdate()`, and in `ensureDependencies` the "Installation complete" notification and a daemon restart decision; `ensureDependencies` deletes the marker.
- The reload works only while a VS Code client is connected to the extension host.

## Related
- Deferred daemon restarts after an update: `vscode-dependency-installer-and-daemon-restart`.
- `installerPath.js` (`kissAiRoot`, `findInstallScript`, `bootstrapInstallUrl`) is the twin of the daemon's `web_server._bootstrap_install_url()`.

## Sources
- `src/kiss/agents/vscode/src/UpdateChecker.js` (`checkForExtensionUpdate`, `resolveCurrentVersion`, `snoozeUpdateNotification`)
- `src/kiss/agents/vscode/src/SorcarSidebarView.ts` (`runUpdate`, `updateWhenIdle`)
- `src/kiss/agents/vscode/src/installerPath.js`
- `src/kiss/agents/vscode/src/extension.ts` (`triggerReload`, `doReload`, marker watch)
- `src/kiss/agents/vscode/src/reloadGuard.js` (`isReloadReady`)
- `install.sh` (writes `.extension-updated`)
