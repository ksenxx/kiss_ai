---
title: kiss-web remote webapp vs VS Code webview (shared media/)
uuid: 540a5090-c818-4fc3-8d10-0dbc51b6f4c2
summary: kiss-web serves the extension's media/chat.html, main.js, api.js to browsers
  - _build_html, WebSocket acquireVsCodeApi shim, body.remote-chat, no CSP, sw.js
  service worker.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# kiss-web remote webapp vs VS Code webview (shared media/)

The browser client ("remote webapp", served by the kiss-web daemon) is not a separate frontend. It runs the extension's own `src/kiss/agents/vscode/media/` files unmodified.

## How the page is built
`_build_html()` in `src/kiss/server/web_server.py` loads the same `media/chat.html` that `SorcarTab.buildChatHtml` uses and substitutes remote values:
- `BODY_CLASS_ATTR` = `' class="remote-chat"'`; `CSP_META` = `""` (no CSP); `NONCE_ATTR` = `""`.
- Asset URLs are `/media/<name>?v=<hash>` (`_media_url`), served from the vscode `media/` directory.
- `MODEL_NAME` = `loading...`; `AUTH_MODAL` holds the password modal; `ENTERKEYHINT` = `' enterkeyhint="send"'`.
- `SHIM_SCRIPT` injects `window.__HLJS_THEME_CSS__` (dark/light highlight themes) and `_WS_SHIM_JS`.

Sharing the template is deliberate: an earlier separate page drifted in script order and DOM ids and broke the tab bar, the `+` button and task submission.

## The WebSocket shim
`_WS_SHIM_JS` defines `window.acquireVsCodeApi()` returning an object whose `postMessage` sends frames over a WebSocket to the daemon, and whose `getState`/`setState` emulate webview state. Every frame is a command from the daemon's API catalog (`kiss/server/sorcar.py`), dispatched by `ServerApi.dispatch`, the same dispatcher the Unix-socket path uses. So `api.js`'s `SORCAR_API_COMMANDS` must only contain catalog commands; host-only messages sent via `postToHost` have no meaning to the daemon.

## Code paths that differ by surface
`main.js` checks `document.body.classList.contains('remote-chat')` for browser-only features: the activity bar with Explorer (`listDir`/`dirListing`) and Source Control (`gitStatus`/`gitLog`) views, the light/dark theme toggle, work-dir pinning in `sessionStorage`, the Working-directory panel. VS Code-only behavior is keyed on `EDITOR_TAB_MODE`, `SIDEBAR_CHAT_MODE`, `POST_ACTIVE_TASK`, etc. (see `vscode-chat-html-build`). `test/activityBarViews.test.js` asserts the VS Code webview never posts the Explorer/SCM commands.

`media/remote-codex.css` and `media/brand.css` carry browser styling; `media/sw.js` is served at `/sw.js`.

## Service worker (`media/sw.js`)
The daemon replaces `__KISS_SW_SHELL__` with `{"version": <hash>, "urls": [...]}` (the page plus every cache-busted media URL), so any asset change produces a new worker and cache name (`kiss-shell-<version>`); old caches are dropped on activate. Strategies: `/media/*` cache-first; navigations to `/` network-first with a 4 s timeout (`NAV_TIMEOUT_MS`) falling back to the cached page tagged `<meta name="kiss-offline-shell">`; `/ws`, `/api/*`, trajectories and the voice model are not intercepted.

## Consequences for changes
- A change to `chat.html` placeholders must be made in BOTH `buildChatHtml` and `_build_html`, or one surface renders a raw `{{KEY}}`. After an extension update, a daemon still running old Python code serves new media files; a raw `{{PRODUCT_NAME}}` title was seen this way until kiss-web restarted.
- Theming: VS Code supplies `--vscode-*` CSS variables; the remote page provides its own palette in `web_server.py` (`body.remote-chat.light-theme` block).
- Test in both modes: jsdom tests can add `remote-chat` to `<body>` before evaluating `main.js`.

## Sources
- `src/kiss/server/web_server.py` (`_build_html`, `_WS_SHIM_JS`, `_media_url`)
- `src/kiss/agents/vscode/media/sw.js`
- `src/kiss/agents/vscode/media/main.js` (`remote-chat` checks)
- `src/kiss/agents/vscode/media/api.js`
- `src/kiss/agents/vscode/test/activityBarViews.test.js`
