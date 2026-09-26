---
title: VS Code extension and web UI - area overview
uuid: cc1c3bd3-f81e-403c-a902-62045596a59b
summary: Map of src/kiss/agents/vscode (KISS Sorcar VS Code extension, chat webview
  media shared with the kiss-web remote webapp), its files, and the vscode-* knowledge
  pages.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# VS Code extension and web UI - area overview

The extension (`package.json` name `kiss-sorcar`, publisher `ksenxx`, engine `vscode ^1.98.0`, entry `./out/extension.js`) is a thin host. All agent work happens in the **kiss-web daemon** (Python, `kiss.server`), which the extension reaches over a Unix socket. The chat UI is one large webview app (`media/main.js` + `media/chat.html` + `media/main.css`) that is also served, unmodified, by the daemon as the **remote webapp** in a plain browser.

## File map (`src/kiss/agents/vscode/`)
| Path | Role |
|---|---|
| `src/extension.ts` | `activate`/`deactivate`: registers the 3 webview views, commands, config listener, SCM commit-message hook, reload-marker watch, dependency setup, update check |
| `src/SorcarSidebarView.ts` | One controller per chat surface: builds the webview HTML, handles webview messages, owns an `AgentClient` to the daemon |
| `src/SorcarPanelManager.ts` | Editor-tabs mode: each chat is a `WebviewPanel` editor tab |
| `src/AgentClient.ts`, `src/SorcarApi.ts` | JSON-lines socket client and typed command wrapper |
| `src/types.ts` | `FromWebviewMessage`, `ToWebviewMessage`, `AgentCommand` unions |
| `src/SorcarTab.ts` | `buildChatHtml` (CSP, template fill), tips/tricks/version helpers, editor-tab body attrs |
| `src/WebviewNotifications.ts` | Toasts rendered inside the webview instead of native notifications |
| `src/DependencyInstaller.ts` | First-run setup (uv, venv, Chromium, git, API keys) and kiss-web daemon restart |
| `src/UpdateChecker.js`, `src/installerPath.js` | PyPI release check, update script location |
| `src/reloadGuard.js`, `src/daemonHealth.js`, `src/daemonRestartVerify.js`, `src/macLaunchd.js` | Window-reload and daemon-restart helpers (plain JS, compiled via `allowJs`) |
| `src/editorActionsLocation.ts` | Moves editor-title buttons into the title bar in editor-tabs mode |
| `src/voiceWake.ts`, `src/voiceAckPlayer.ts` | Host-side wake-word listener process and ack sound |
| `src/kissPaths.ts`, `src/userAssets.ts`, `src/brand.ts`, `src/gitApi.ts` | Path resolution, `~/.kiss` assets, branding, VS Code git API access |
| `media/` | Webview app: `main.js` (~21k lines), `chat.html`, `main.css`, `api.js`, `voice.js`, `vosk.js`, `tips.js`, `share.js`, context menus, `sw.js` (remote service worker), `brand.json`, icons |
| `test/` | ~360 node/jsdom suites run by `test/run-all.js` |
| `scripts/` | `apply-brand.js`, `hash-icons.js`, `package-vsix.js` (VSIX build) |
| `copy-kiss.sh` | Bundles the Python project into `kiss_project/` for the VSIX |
| `__init__.py` | Marks the directory as a Python package (the Python jsdom tests live in `src/kiss/tests/agents/vscode/`) |

## Pages in this area
- `vscode-activation-and-views` - activation sequence, views, commands, keybindings, settings
- `vscode-daemon-transport` - `AgentClient` socket protocol, reconnect, queued commands
- `vscode-webview-message-flow` - webview <-> host <-> daemon message routing
- `vscode-adding-a-webview-command` - checklist for a new command/event
- `vscode-editor-tabs-mode` - `SorcarPanelManager`, registry adoption, one-chat invariant
- `vscode-chat-html-build` - `buildChatHtml`, CSP, template placeholders, surface modes
- `vscode-remote-webapp-relation` - how kiss-web reuses `media/`
- `vscode-settings-panel` - settings drawer and config round-trip
- `vscode-dependency-installer-and-daemon-restart` - setup and restart decisions
- `vscode-update-and-window-reload` - update check, `.extension-updated` marker
- `vscode-build-and-lint` - `npm run compile`, `out/`, lint scopes
- `vscode-packaging-and-branding` - `copy-kiss.sh`, VSIX, hashed icons, brand.json
- `vscode-js-tests` - test runner and conventions
- `vscode-scm-commit-message` - SCM sparkle commit message generation
- `server-voice-wake-and-talk` - wake-word listener protocol and browser fallback
- `vscode-webview-notifications` - in-webview toasts and progress
- `vscode-ui-bug-classes` - recurring bug patterns and how they were fixed

## Sources
- `src/kiss/agents/vscode/package.json`
- `src/kiss/agents/vscode/src/extension.ts` (`activate`)
- `src/kiss/agents/vscode/tsconfig.json`
