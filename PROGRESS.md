# Task: Port Sorcar CLI Interactive to be a Client of the Web Server

## User Decisions (clarified 2025-XX-XX)

1. **Communication model: 1.b** — The CLI client must connect to an
   **already-running** `sorcar web` server over WebSocket (similar to
   how the remote web app at `https://<tunnel-url>` connects). No
   embedded in-process server.
1. **Feature placement** — Use server-provided features as much as
   possible. **Any feature that does not exist on the server must be
   moved to the server side**. CLI client = thin terminal UI on top of
   the WebSocket protocol.
1. **No backwards compatibility** — Standalone CLI-interactive mode is
   replaced; `sorcar` (with no `-t` / `-f`) now defaults to client
   mode.
1. **Multi-session execution OK** — proceed across `is_continue=True`
   sessions.

## Model assignments

- **Coding / bug-fixing / test creation**: `claude-opus-4-7`
- **Code review**: `gpt-5.5` (e.g. `codex/gpt-5.5`)

## Architecture

The existing `sorcar web` server (`kiss/agents/vscode/web_server.py`,
class `RemoteAccessServer`) already speaks the full webview protocol
over WebSocket:

- Receives commands: `ready`, `submit`, `cancel`, `userAnswer`,
  `clearChat`, `resumeSession` (→ chatId), `userActionDone` (→
  userAnswer "done"), `closeTab`, `mergeDecision`, `requestModelList`,
  `setModel`, … (see `_translate_webview_command` and
  `_run_cmd` in `web_server.py`).
- Sends events to webviews: status, message, tool, prompt, requestUser,
  cost, model list, merge events, etc.

The VS Code webview (`kiss/agents/vscode/src/*.ts`) and the remote web
app (`_build_html()` in web_server.py) are the two existing clients.
The new CLI client becomes the third.

## Plan

### Phase 1: Discovery (next session)

Files to read fully and summarize into
`tmp/file-information-cli-port.md`:

- `src/kiss/agents/vscode/web_server.py` — focus on:
  - `_translate_webview_command` (commands list)
  - `RemoteAccessServer._ws_handler`, `_authenticate_ws`,
    `_handle_command`, `_dispatch_client_command`, `broadcast`
  - `WebPrinter` (event emission)
  - HTML/JS in `_build_html()` to learn what the remote web app sends
- `src/kiss/agents/sorcar/cli_repl.py` — full file
- `src/kiss/agents/sorcar/cli_helpers.py`,
  `cli_steering.py`, `cli_prompt.py`, `cli_printer.py`, `cli_panel.py`
- `src/kiss/agents/sorcar/worktree_sorcar_agent.py` — `main()` flow
- `src/kiss/agents/sorcar/custom_commands.py`,
  `mcp_servers.py`, `mcp_cli.py`, `skills.py`
- `src/kiss/agents/vscode/helpers.py`,
  `autocomplete.py`, `commands.py`, `server.py` — to see what
  client-side state must exist
- `pyproject.toml` — find current `sorcar` entry point

Decision matrix to produce in
`tmp/file-information-cli-port.md`:

| CLI REPL feature | Server already supports? | New server cmd? | Client only? |
|-------------------------------|--------------------------|----------------------|--------------|
| `/help` | (text static) | yes — `helpInfo` | |
| `/clear`, `/new` | yes — `clearChat` | | |
| `/resume <id>` | yes — `resumeSession` | | |
| `/model <name>` | yes — `setModel` | | |
| `/model list` | yes — `requestModelList` | | |
| `/cost`, `/usage`, `/context` | partial — cost events | maybe `requestCost` | |
| `/commands` | not yet | yes — `listCommands` | |
| `/skills`, `/skills <name>` | not yet | yes — `listSkills` | |
| `/mcp` | not yet | yes — `listMcp` | |
| `/autocommit` | not yet | yes — `autoCommit` | |
| `/exit`, `/quit` | | | YES |
| @-mentions file picker | yes — autocomplete reqs | | hybrid |
| predictive ghost | yes — autocomplete reqs | | hybrid |
| prompt_toolkit | | | YES |
| readline fallback | | | YES |
| custom commands expansion | partial | yes | |
| worktree merge prompt | yes — mergeDecision | | |

(Update during Phase 1.)

### Phase 2: Server endpoints (only what is missing)

Add new commands to `RemoteAccessServer._handle_command` /
`_translate_webview_command` and corresponding events to `WebPrinter`
for any feature missing from the table above. **Reuse the existing
helpers from `cli_repl.py` etc. by moving them into a shared module if
needed — do not duplicate.**

### Phase 3: New CLI client module

Create `src/kiss/agents/sorcar/cli_client.py`:

- Connects to `wss://localhost:<port>` (or `--server` URL) using
  `websockets` (already a dep — see top of `web_server.py`).
- Re-uses existing `PtkLineReader` (`cli_prompt.py`), the input panel
  rendering (`cli_panel.py`), and the slash-command dispatch table
  (`SLASH_COMMANDS` in `cli_repl.py`) — but each slash command now
  sends a WebSocket message rather than calling local helpers.
- Re-uses `cli_printer.py` to render incoming server events to the
  terminal.
- Discovery: if no `--server` URL is given, read
  `~/.kiss/web_server.json` (written by `sorcar web`) for URL +
  password.
- Auth flow mirrors the browser: first message after connect is
  `{"type": "auth", "password": "..."}` per
  `_authenticate_ws` in `web_server.py`.
- Replaces `run_repl(...)` import in
  `worktree_sorcar_agent.main()` with `run_client(...)`.

### Phase 4: Wire `sorcar` interactive to the client

In `worktree_sorcar_agent.main()`:

- Replace `from kiss.agents.sorcar.cli_repl import run_repl` with
  `from kiss.agents.sorcar.cli_client import run_client`.
- Auto-discover or auto-launch behavior: if no server is reachable,
  print an instructive error pointing the user to
  `sorcar web` first. (Confirmed: no backwards-compat — do not start
  an in-process server.)
- Keep `sorcar mcp ...` and `--list-chat-id`, `--cleanup` as before
  (these are CLI management commands, not the interactive REPL).

### Phase 5: End-to-end tests

Create `src/kiss/tests/agents/sorcar/test_cli_client.py`:

- Spin up a real `RemoteAccessServer` on a random local port (re-use
  helpers from existing
  `src/kiss/tests/agents/vscode/test_web_server*.py` if any).
- Drive the new CLI client via subprocess with a piped stdin
  containing slash commands and a real task prompt.
- Assert: connection succeeds, `/help` prints the help, `/model list`
  prints models from the server, a submitted prompt triggers a
  `run` on the server, the streamed events end up in the client's
  stdout, `/exit` cleanly disconnects.
- No mocks, no test doubles. Each test independent. End-to-end only.

### Phase 6: Lint, typecheck, run impacted tests

`uv run check --full` → fix all errors.
Run only impacted tests (re-using core-count parallel split if >100).

### Phase 7: Review

Switch to `codex/gpt-5.5` (`set_model`) and ask it to review the diff.
Apply fixes with claude-opus-4-7.

## Session log

### Session 1

- Skimmed repo layout, sorcar dir, vscode dir.
- Read `worktree_sorcar_agent.main()` interactive dispatch.
- Read `cli_repl.py` header + outline of functions.
- Found `_translate_webview_command` in `web_server.py` listing the
  webview command set.
- Switched model to claude-opus-4-7.
- **Clarified scope with user.**
- Wrote this PROGRESS.md.

### Session 2

- Confirmed server handler table in
  `src/kiss/agents/vscode/commands.py` at line ~947 (`_HANDLERS`).
- Wrote `tmp/file-information-cli-port.md` with decision matrix:
  - `/clear`, `/new` → `newChat`
  - `/resume` → `resumeSession`
  - `/model <n>` → `selectModel`
  - `/model list` → `getModels`
  - `/autocommit` → `autocommitAction`
  - `@`-mentions → `getFiles` + `recordFileUsage`
  - predictive ghost → `complete`
  - worktree m/d → `worktreeAction` / `mergeAction`
  - **Missing server-side**: `/cost`, `/context`, `/commands`,
    `/skills`, `/mcp`, `/help`, custom-command expansion. Need
    new commands: `getCostSnapshot`, `listCustomCommands`,
    `expandCommand`, `listSkills`, `getSkill`, `listMcp`, `getHelp`.
- Identified auth/connect protocol from `_authenticate_ws`:
  WSS → `auth_required` → client sends `{type:"auth", password}` →
  `auth_ok` → webview-style commands.

### Next-session start prompt

1. `Read("src/kiss/agents/vscode/commands.py")` from line 150 to 950
   in chunks to study each `_cmd_*` handler.
1. `Read("src/kiss/agents/sorcar/cli_repl.py")` in chunks (200-line
   windows) — focus on `_handle_slash`, `_run_one`, `run_repl`,
   `_handle_autocommit`, `_handle_resume`, `_handle_model`.
1. `Read("src/kiss/agents/vscode/web_server.py")` chunks at lines
   2900-3500 — `_authenticate_ws`, `_ws_handler`,
   `_dispatch_client_command`, `_run_cmd`, URL/port discovery.
1. `Read("src/kiss/agents/sorcar/cli_printer.py")` and
   `cli_panel.py` fully.
1. Enumerate WebPrinter event types by grepping `_emit` /
   `broadcast` calls.
1. **Proceed to Phase 2 (add missing server commands)** in
   `src/kiss/agents/vscode/commands.py`, reusing helpers from
   `cli_repl.py`, `custom_commands.py`, `skills.py`, `mcp_servers.py`.
1. **Proceed to Phase 3 (build CLI client)** at
   `src/kiss/agents/sorcar/cli_client.py`.
1. Update `worktree_sorcar_agent.main()` to call `run_client`.
1. Write end-to-end tests in
   `src/kiss/tests/agents/sorcar/test_cli_client.py`.
1. Run `uv run check --full` and impacted tests; fix.
1. Switch to `codex/gpt-5.5` for review; switch back to
   `claude-opus-4-7` for fixes.

### Next-session start prompt

Continue from Phase 1 (Discovery). First action:

1. `Read("src/kiss/agents/sorcar/cli_repl.py")` in chunks to fully
   understand REPL features.
1. `Read("src/kiss/agents/vscode/web_server.py")` in chunks focusing
   on the handler functions listed above.
1. Populate the decision matrix in
   `tmp/file-information-cli-port.md`.
1. Proceed to Phase 2.
