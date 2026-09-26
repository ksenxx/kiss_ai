---
title: Chat webview HTML build, CSP and surface modes
uuid: 9a2fc1c5-4727-4cad-9e77-557857bc9bbc
summary: SorcarTab.buildChatHtml fills media/chat.html {{PLACEHOLDERS}} - nonce CSP,
  sha256 ?v= asset URLs, body classes selecting sidebar, editor-tab, history, meta
  or remote-chat surface.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Chat webview HTML build, CSP and surface modes

## One template, several surfaces
`media/chat.html` is a template with `{{KEY}}` placeholders. The extension fills it in `buildChatHtml(webview, extensionUri, selectedModel, bodyAttrs)` (`src/SorcarTab.ts`); the daemon fills the same file in `_build_html()` (`kiss/server/web_server.py`) for the remote webapp. `substituteTemplate(tpl, subs)` replaces known keys and leaves unknown ones untouched. `BODY_CLASS_ATTR`, `ENTERKEYHINT` and `NONCE_ATTR` are attribute strings with their own leading space; the template writes a space before them so htmlhint can parse the tags, and `substituteTemplate` drops that space for these keys.

Placeholders filled by the extension: `VIEWPORT`, `CSP_META`, `STYLE_HREF`, `BRAND_STYLE_HREF`, `HLJS_CSS_HREF`, `HEAD_STYLE`, `BODY_CLASS_ATTR`, `PRODUCT_NAME`, `TAGLINE`, `BRAND_JSON`, `INPUT_PLACEHOLDER`, `ENTERKEYHINT`, `MODEL_NAME` (HTML-escaped), `VERSION_SUFFIX`, `AUTH_MODAL`, `NONCE_ATTR`, script URLs (`HLJS_SRC`, `MARKED_SRC`, `API_SRC`, `PANEL_COPY_SRC`, `CTX_MENU_SRC`, `TREE_MENU_SRC`, `MAIN_SRC`, `TIPS_SRC`, `VOICE_SRC`), `SHIM_SCRIPT` (empty in VS Code), `TRICKS_JSON`, `TIPS_JSON`, `VOICE_CONFIG`. JSON blobs escape `</` as `<\/` so they cannot close the script tag.

## CSP
`default-src 'none'`; styles from `webview.cspSource` plus `'unsafe-inline'`; `script-src 'nonce-<nonce>' 'wasm-unsafe-eval'`; `worker-src blob:`; `connect-src` only the origin of `VOICE_MODEL_URL`; images from cspSource, `data:`, `https:`; `form-action`, `frame-src`, `object-src`, `base-uri` all `'none'`. The WASM/blob/connect entries exist for the in-page voice fallback (`vosk.js` runs a recognizer in a blob Worker). Any new script must be loaded with the nonce attribute; inline event handlers are blocked.

## Asset cache-busting
`u(name)` = `webview.asWebviewUri(media/name)` + `?v=<first 16 hex of sha256(file)>` (`mediaAssetVersion`), so a changed media file always gets a new URL. The remote webapp uses `/media/<name>?v=<hash>` from `web_server._media_url`.

## Surface selection by `<body>` class
| Surface | Body attrs | Set by |
|---|---|---|
| Sidebar chat view (mode off) | none; `main.js` adds `sidebar-chat-mode` | `SIDEBAR_CHAT_MODE` in `main.js` |
| Chat editor tab | `class="editor-tab-mode"` + `data-kiss-tab-id`, `data-kiss-tab-title`, `data-kiss-resume-chat-id`, `data-kiss-resume-task-id`, `data-kiss-pending-text`, `data-kiss-in-registry` | `editorTabBodyAttrs(init)` |
| History panel | `class="editor-tab-mode history-panel-mode"`, tab id `history-panel` | `historyPanelBodyAttrs()` |
| Task Info panel | `class="editor-tab-mode meta-panel-mode"`, tab id `meta-panel` | `metaPanelBodyAttrs()` |
| Remote webapp | `class="remote-chat"` | `web_server._build_html` |

`main.js` derives flags from these classes at startup: `EDITOR_TAB_MODE`, `HISTORY_PANEL_MODE`, `META_PANEL_MODE`, `SIDEBAR_CHAT_MODE`, `POST_META_UPDATES` (only real chat panels report task info), `POST_ACTIVE_TASK` (VS Code chat surfaces only). Features that exist only in the browser (activity bar Explorer/Source Control views, theme toggle) check `body.remote-chat`. The fixed tab ids `history-panel` and `meta-panel` never run tasks and are never announced to the daemon registry.

## Other inputs
- `getTricks()` (promptlet `## Trick` sections from `~/.kiss/MY_INJECTION.md`, seeded by `ensureUserAssetFromDefault`, plus the bundled injections file or `$KISS_INJECTIONS_PATH`), `getTips()` + `consumeTipsFirstRun()` (tips dialog shown once), `getVersion()`.
- `readSampleTasks(extensionRoot)` (`## Task` sections of `~/.kiss/MY_TASK_TEMPLATES.md`, among others) supplies the welcome suggestions the host sends after `ready`.
- `resetTipsOnExtensionUpdate()` is called at activation.

## Sources
- `src/kiss/agents/vscode/src/SorcarTab.ts` (`buildChatHtml`, `substituteTemplate`, `ATTR_STRING_KEYS`, `mediaAssetVersion`, `editorTabBodyAttrs`, `historyPanelBodyAttrs`, `metaPanelBodyAttrs`, `VOICE_MODEL_URL`)
- `src/kiss/agents/vscode/media/chat.html`
- `src/kiss/agents/vscode/media/main.js` (mode flags at the top of the IIFE)
- `src/kiss/server/web_server.py` (`_build_html`)
