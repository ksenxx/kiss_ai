# PROGRESS — Hide KISS Sorcar tabs until kiss-web is started

## Task

Until the kiss-web daemon has connected, the VS Code KISS Sorcar
secondary-sidebar webview must NOT show its tab bar or welcome page.
Instead it must show "KISS Sorcar Server is starting ..." with a
spinner. Once the daemon socket is connected the regular UI must
appear. A daemon disconnect (server reset) must re-show the overlay
until auto-reconnect succeeds.

## Root cause

`media/chat.html` rendered `#app` (tabs, welcome, input area, …)
immediately on first paint, and `SorcarSidebarView.ts` never told the
webview about the daemon's UDS connection state. Result: while
`AgentClient` was still trying to reach `~/.kiss/sorcar.sock` on
launch the user saw a fully-rendered (but non-functional) sidebar.

## Fix

1. **`src/kiss/agents/vscode/src/types.ts`** — added a new
   `ToWebviewMessageBody` variant
   `{type: 'daemonStatus'; connected: boolean}` so the typecheck
   covers the new control message.

1. **`src/kiss/agents/vscode/src/AgentClient.ts`** — emit a new
   `'disconnect'` event on socket close so the sidebar can re-show
   the loading overlay during reconnects:

   ```ts
   sock.on('close', () => {
     this._connecting = false;
     this._socket = null;
     this.emit('disconnect');
     if (this._disposed) return;
     this._scheduleReconnect();
   });
   ```

1. **`src/kiss/agents/vscode/src/SorcarSidebarView.ts`** —
   added `private _daemonConnected = false` and three hooks:

   ```ts
   client.on('connect', () => {
     // (existing setWorkDir send …)
     this._daemonConnected = true;
     this._sendToWebview({type: 'daemonStatus', connected: true});
   });
   client.on('disconnect', () => {
     this._daemonConnected = false;
     this._sendToWebview({type: 'daemonStatus', connected: false});
   });
   // in case 'ready' (webview just attached its message listener):
   this._sendToWebview({
     type: 'daemonStatus',
     connected: this._daemonConnected,
   });
   ```

   The `ready` push covers webview reloads (e.g. after a VS Code
   reload window) where the webview's DOM was reset but the daemon is
   already connected — the overlay would otherwise stay up forever.

1. **`src/kiss/agents/vscode/media/chat.html`** — added a fixed
   `#kiss-server-loading` overlay element above `#app` and started
   `#app` with `style="display:none;"`:

   ```html
   <div id="kiss-server-loading" role="status" aria-live="polite">
     <div class="kiss-server-loading-inner">
       <div class="kiss-server-loading-spinner" aria-hidden="true"></div>
       <div class="kiss-server-loading-msg">KISS Sorcar Server is starting ...</div>
     </div>
   </div>
   <div id="app" style="display:none;">…</div>
   ```

1. **`src/kiss/agents/vscode/media/main.css`** — themed full-viewport
   overlay with a CSS-keyframe spinner, using the VS Code theme
   variables already exposed at `:root` (`--bg`, `--fg`, `--accent`,
   `--border`).

1. **`src/kiss/agents/vscode/media/main.js`** — added
   `setServerLoading(loading)` and a `case 'daemonStatus'` early in
   `handleEvent` that toggles `#kiss-server-loading` /
   `#app` `display` based on `ev.connected`.

## Reproduction / regression test

`src/kiss/agents/vscode/test/serverLoadingOverlay.test.js` (registered
in `package.json` `test` script) is a real end-to-end test against
the compiled `out/SorcarSidebarView.js` and `out/AgentClient.js`:

1. Spawns a real UDS server at `~/.kiss/sorcar.sock` (with `$HOME`
   redirected to a tempdir).
1. Stubs only the `vscode` module via the same `_vscode-stub.js`
   pattern used by `syncWorkDir.test.js`, and provides a fake
   `WebviewView` that captures `webview.postMessage(...)` and lets
   the test fire `onDidReceiveMessage` callbacks.
1. Asserts five things, each one matching a real bug surface:
   - the initial HTML from `buildChatHtml(…)` contains the
     `id="kiss-server-loading"` overlay AND `<div id="app" style="display:none">`,
   - a `ready` message posted to the view while the daemon socket is
     down triggers a `daemonStatus connected:false` reply (so a
     reloaded webview re-locks the overlay),
   - starting the UDS server causes the auto-reconnect to fire
     `connect`, which posts `daemonStatus connected:true`,
   - destroying the accepted server socket triggers
     `daemonStatus connected:false` (re-showing the overlay),
   - the next auto-reconnect again posts `connected:true`.

Reproduction verified by manually stripping the new `daemonStatus`
posts from the compiled JS — the test failed on tests 3, 4, 5 with
`no daemonStatus(connected:…)` waitFor timeouts. Restoring the fix
turned all five back green.

## Verification

- `npm run compile` — clean.
- `node test/serverLoadingOverlay.test.js` — 5/5 OK.
- `npm run check` — typecheck + lint + all 28 webview tests green
  (including the new one).
- `uv run check --full` — ruff, mypy, pyright, vscode-check,
  mdformat all green.

## Incidental clean-ups

- `media/main.css` had four pre-existing `stylelint`
  `rule-empty-line-before` / `at-rule-empty-line-before` violations
  in the new overlay block; auto-fixed via
  `npx stylelint media/**/*.css --fix`.
