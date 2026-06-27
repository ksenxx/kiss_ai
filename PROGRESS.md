# Task

Confirm whether the KISS Sorcar VS Code extension checks for new
upstream updates of KISS Sorcar (the extension/package itself) when
VS Code is launched and KISS Sorcar is **not** installing. If it
does not, implement that check.

Workflow constraint:

- Use `claude-opus-4-7` for coding, bug-fixing, test creation.
- Use `gpt-5.5` (NOT codex) for a thorough review at the end.

## Findings

Before this fix, when VS Code launched and KISS Sorcar was NOT
installing (the early-exit "fast path" in
`DependencyInstaller.ensureDependenciesImpl`), the extension did
nothing to actively probe upstream for a newer release. The
kiss-web daemon polls PyPI hourly and broadcasts `update_available`
to connected webview clients, but:

1. on the fast path the daemon is NOT restarted, so its cached "no
   update" answer can be up to an hour stale;
1. users who launch VS Code only briefly never wait for the next
   hourly poll;
1. the daemon's broadcast is suppressed when its cache says "no
   update", even if PyPI now reports a newer version.

So the answer to the user's question: **No, we did not check for
new updates when VS Code was launched and KISS Sorcar was not
installing. This change adds it.**

## What this session did

### 1. New module — `src/kiss/agents/vscode/src/UpdateChecker.js`

Plain JS (matches the `installerPath.js` / `daemonHealth.js` /
`reloadGuard.js` pattern so the bare-`node` test harness can drive
it without a TypeScript compile). Exports:

- `checkForExtensionUpdate(opts)` — main entry point. All side
  effects (HTTP, fs, vscode, clock) are injectable via `opts` so the
  integration test drives the real flow against a loopback HTTP
  stub of PyPI without touching the user's `~/.kiss/` or the
  network.
- `compareVersions(a, b)` — pure helper. Mirrors the Python
  `_compare_versions` in `web_server.py` so the extension and
  daemon agree on update direction.
- `resolveCurrentVersion(kissProjectPath)` — reads
  `<kissProjectPath>/src/kiss/_version.py`.

Behavior:

- Reads local `__version__` from `_version.py`.
- Fetches PyPI JSON for `kiss-agent-framework`.
- When `latest > current`, calls `notify({latest, current})`.
- Caches the result in `~/.kiss/.update-check.json` and rate-limits
  itself with a 6 h cooldown so the same VS Code session never
  hammers PyPI on every window reload.
- All errors (network, malformed payload) are swallowed — update
  checking can never break extension activation.

### 2. Wired into `src/kiss/agents/vscode/src/extension.ts`

```ts
import {checkForExtensionUpdate} from './UpdateChecker';
...
void checkForExtensionUpdate({
  kissProjectPath: findKissProject() || undefined,
  notify: ({latest, current}: {latest: string; current: string}) => {
    void showInformationNotification(
      `KISS Sorcar: a new release (${latest}) is available. ` +
        `You are on ${current}.`,
      'Update now',
    ).then(action => {
      if (action === 'Update now') {
        void vscode.commands.executeCommand('workbench.action.terminal.new');
      }
    });
  },
}).catch(err => console.error('[KISS Sorcar] Update check failed:', err));
```

Runs unconditionally in `activate()` so it fires on the fast path
too (which was the gap).

### 3. End-to-end test — `src/kiss/agents/vscode/test/updateChecker.test.js`

Bare-`node` test driving the real `UpdateChecker.js` against a real
loopback HTTP server impersonating PyPI. Seven cases:

1. **Reproduces the bug** — stale local version + newer PyPI version
   → `notify` fires with the right payload, cache file is written.
   Before the fix this test would `require('../src/UpdateChecker.js')`
   and fail at module-resolution because the helper did not exist.
1. Within cooldown, the cached decision is replayed and PyPI is NOT
   re-hit.
1. PyPI reports current version → `notify` is NOT called (no false
   positive after the user updates).
1. PyPI returns 500 → `notify` is NOT called, no exception escapes,
   cache file is NOT poisoned.
1. Current version resolved from `_version.py` (production code
   path).
1. `compareVersions` parity with Python (`"2026.6.30"` vs
   `"2026.6.29"`, padding, malformed input).
1. Unknown current version → graceful skip, no PyPI hit, no notify.

Added to `package.json` `scripts.test`.

### 4. Verification

- `uv run check --full` → ✅ all passed (ruff, mypy, pyright,
  mdformat, syntax). Reformatted `PROGRESS.md` to satisfy
  mdformat.
- `node test/updateChecker.test.js` → 7/7 pass.
- `npm run typecheck` (after `npm install`) → clean. Required one
  explicit parameter annotation in `extension.ts`:
  `notify: ({latest, current}: {latest: string; current: string}) => …`
  because `allowJs` infers `any` for the JS-side
  destructured callback args.

## Remaining work for next continuation

1. Run the full extension test suite
   (`node test/<every>.test.js`) to confirm nothing else broke.
1. Run extension lint (`npm run lint`) for the eslint/stylelint
   pass.
1. Switch the model to `gpt-5.5` (per the task rule) and run a
   thorough review of the diff and the new test, applying any
   feedback.
1. Clean up `./tmp/` and call `finish(success=True)`.
