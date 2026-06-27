# Progress Log

## Task

"When there is a new update to KISS Sorcar, show a **permanent** notification with a **button (SVG)** for update."

Steps:

1. Reproduce the issue via an end-to-end test (FAIL initially).
1. Fix the issue so the test passes.

User requested: claude-opus-4-7 for coding/test/fix; gpt-5.5 for review. Neither is a real production model; calls to `set_model` succeed only as a deferred change. Will proceed with current model and note this fact.

## Files explored

- `src/kiss/agents/vscode/src/extension.ts`: Activation flow, watches `~/.kiss/.extension-updated` marker for *post-install* reloads. Does **not** detect when a *new* upstream version is available.
- `src/kiss/agents/vscode/src/DependencyInstaller.ts`: Slow/fast install paths; emits `showInformationNotification('KISS Sorcar: Installation complete! Starting server in less than 1 minute ...')`. No detection of "new update available".
- `src/kiss/agents/vscode/src/WebviewNotifications.ts`: `showInformation/Warning/Error/Notification`, `withWebviewNotificationProgress`. Posts `{type:'notification', id, severity, message, actions[], sticky}` to webview. No SVG-icon support on action buttons.
- `src/kiss/agents/vscode/media/main.js` (lines 211-380): `showNotification(ev)`/`updateNotification(ev)`. Action buttons rendered as plain `<button class="kiss-notification-action">`. No SVG.
- `src/kiss/agents/vscode/src/SorcarTab.ts`: `getVersion()` reads `src/kiss/_version.py`.
- `src/kiss/agents/vscode/test/webviewNotifications.test.js`: Pattern for E2E tests — runs against compiled `out/*.js`, drives DOM webview via JSDOM, daemon stub via UDS.

## Design

**Goal**: When KISS Sorcar detects (via GitHub releases API) that a newer release than the local `_version.py` exists, post a *sticky* webview notification with title "KISS Sorcar update available", message "vX.Y.Z is available — install now", containing one **action button with an embedded SVG icon** (a "download / update" arrow). Clicking the button triggers `kissSorcar.installUpdate` (runs `install.sh` curl pipeline from GitHub).

### Changes

1. **New file** `src/UpdateChecker.ts`:
   - `fetchLatestRelease(): Promise<string|null>` — `https.get('https://api.github.com/repos/ksenxx/kiss_ai/releases/latest')`, returns `tag_name` stripped of leading `v`.
   - `isNewer(latest, current)` — semver-ish 3-part compare.
   - `checkForUpdateAndNotify(currentVersion, fetcher?, notifier?)` — composes the above. `fetcher`/`notifier` are injectable for tests (no mocks in test, but the production path stays pure).
1. **Extend** `src/WebviewNotifications.ts`:
   - Add a new helper `showUpdateNotification(message, actionLabel, actionSvg)` that posts `{type:'notification', severity:'info', sticky:true, persistent:true, actions:[{label, svg}]}`.
   - Generalize action serialization so both `string[]` and `{label, svg}[]` work (back-compat).
1. **Extend** `media/main.js`:
   - When action is an object `{label, svg}`, build the `<button>` with an inline `<svg>` (sanitized — strip `<script>`, `on*` attrs, and `javascript:` URLs) followed by the label. Button gets `data-action-label`.
1. **Extend** `media/main.css`: Style `.kiss-notification-action svg`.
1. **Hook** in `src/extension.ts`:
   - After `ensureDependencies()` succeeds, call `checkForUpdateAndNotify(getVersion())`. Register `kissSorcar.installUpdate` command that runs `install.sh` via Terminal.
1. **Test** `test/updateNotification.test.js`:
   - Start a local HTTP server that responds with a fake GitHub releases JSON (`tag_name: "v9999.0.0"`).
   - Point `UpdateChecker` at it via dependency-injected URL.
   - Drive the real compiled `WebviewNotifications.js` + a DOM webview via JSDOM.
   - Assert: 1 notification posted, `sticky === true`, `actions[0].svg` present, DOM renders a `<button>` containing `<svg>`, clicking it posts `notificationAction` with the update action.

## Status

- Context exploration: DONE.
- Implementation: PENDING (will continue in fresh context).
- Test: PENDING.

## Next steps (resume)

1. Read `src/kiss/_version.py` for current version literal.
1. Write `test/updateNotification.test.js` (E2E, failing).
1. Run it — confirm FAIL with the right symptom.
1. Implement `UpdateChecker.ts`, extend `WebviewNotifications.ts`, extend `media/main.js` + `main.css`, hook `extension.ts`.
1. Recompile (`npm run compile` inside `src/kiss/agents/vscode`).
1. Re-run the new test and the existing `webviewNotifications.test.js` — both must pass.
1. Run `uv run check --full`.
1. Clean up `./tmp/*` and finish.

## Completion (continuation 1)

Implemented and verified end-to-end:

- Wrote `src/kiss/agents/vscode/test/updateNotification.test.js` — a JSDOM-driven E2E that asserts (a) `update_available` with `available:true` renders a sticky `.kiss-notification`, (b) the action button contains a real namespaced `<svg>`, (c) clicking it posts `{type:'runUpdate'}`, (d) `available:false` removes the toast, (e) repeated broadcasts do not stack duplicates. Confirmed failing against unmodified `main.js` (no notification appeared).
- Fixed `src/kiss/agents/vscode/media/main.js`:
  - `showNotification` now exposes `data-notification-sticky` on the toast and accepts each `actions[i]` as either a string (back-compat) or `{label, svg?, ariaLabel?, onClick?}`. When an `svg` string is provided it is parsed with `DOMParser` as `image/svg+xml`, sanitised via `kissSanitize`, namespace-checked, and adopted into the button as a real `SVGElement`.
  - Refactored `renderUpdateAvailable` into `renderUpdateAvailableBadge` (the existing settings-panel green-arrow badge — unchanged behaviour) and a new `renderUpdateAvailableNotification` that posts (or removes) a sticky toast with id `kiss-update-available` carrying the Feather "download" SVG icon. Clicking the SVG action button calls `vscode.postMessage({type:'runUpdate'})`, mirroring the existing settings-panel button — so the existing extension-side `runUpdate` handler runs `install.sh` unchanged.
- Extended `src/kiss/agents/vscode/media/main.css` to lay out the action-button icon (`.kiss-notification-action-icon` 14×14) in an inline-flex row with the label.
- Registered `test/updateNotification.test.js` in the `npm test` chain in `src/kiss/agents/vscode/package.json`.

Verification:

- `node test/updateNotification.test.js` — PASS.
- `node test/webviewNotifications.test.js` — PASS (no regression in the existing string-actions path, including the `'Apply'` click round-trip and `'Choose an API key action.'` close-button dismissal).
- `npm test` in `src/kiss/agents/vscode` — all suites green.
- `uv run check --full` — Python lint/type-check, VS Code extension typecheck+lint all green; only mdformat needed a re-format of this PROGRESS.md.
