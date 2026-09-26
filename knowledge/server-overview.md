---
title: Sorcar daemon (kiss-web) area overview
uuid: 4e6e5682-7ffd-4c8e-ad51-27792c6ad165
summary: 'Map of the Sorcar daemon area: src/kiss/server (web_server, server, task_runner,
  commands, sorcar API, printer, TLS, voice), daemon_client.py, rsorcar, sorcar launcher.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Sorcar daemon (kiss-web) area overview

A single long-lived process, `kiss-web`, serves every Sorcar user interface. The VS Code extension connects over a Unix-domain socket (UDS, `src/AgentClient.ts`); its chat webview talks to the extension via `postMessage`, and the extension relays over the UDS. Only the remote browser webapp connects over HTTPS/WSS on one TCP port. Python callers (`run_agent`, cron, scripts) connect through `kiss.server.sorcar.run`. One `VSCodeServer` backend owns tabs, agent states and task threads for all of them.

## File map (src/kiss/server/)
| File | Role |
|---|---|
| `web_server.py` | `RemoteAccessServer` (transport: UDS + HTTPS/WSS, auth, tunnel, TLS, watchdog, shutdown), `WebPrinter` (broadcast fan-out), `main()` CLI |
| `server.py` | `VSCodeServer` backend (composed of mixins), tab registry wiring, session replay |
| `commands.py` | `_CommandsMixin`: one `_cmd_*` handler per command (`run`, `stop`, `appendUserMessage`, `openTab`, ...) |
| `task_runner.py` | `_TaskRunnerMixin`: worker thread per task, stop/force-stop, follow-ups, ask-user |
| `agent_state.py` | `AgentState` + `agent_states` registry + `STATE_LOCK` |
| `tab_registry.py` | `TabRegistry`: server-canonical tab list persisted to `tabs.json` |
| `sorcar.py` | Wire API catalog `API`, `ServerApi.dispatch`, `authenticate`, re-export of `run` |
| `json_printer.py` | `JsonPrinter`: task-centric event recording/persistence/subscribers |
| `stall_watchdog.py` | GIL-independent stack dumper |
| `agent_file.py`, `tools_file.py` | Load SEA agent scripts and tools files in the daemon |
| `tls_certs.py`, `tls_trust.py` | Local CA + server certificate, trust-store installer |
| `voice_wake.py`, `voice_wake_control.py`, `talk_player.py` | Wake word listener child and native talk playback |
| `task_update.py` | Periodic task progress reports (`getTaskUpdate`) |
| others | `autocomplete.py`, `explorer.py`, `fs_actions.py`, `sidebar_panels.py`, `tips.py`, `tricks.py`, `user_assets.py`, `helpers.py`: command helpers for UI panels |

Outside the package: `src/kiss/agents/sorcar/daemon_client.py` (the synchronous client), `rsorcar` (deploy to a remote Linux box), `sorcar` (a local launcher script).

## Detail pages
- `server-daemon-startup-and-uds`: entry point, port, socket binding, single-daemon guard
- `server-wire-protocol-and-dispatch`: command catalog, framing, connId/tabId/workDir stamping
- `server-broadcast-routing`: how events reach clients (WebPrinter/JsonPrinter)
- `server-task-lifecycle`: `run` command to terminal status
- `server-steering-and-followups`: mid-run messages, `<task>` follow-ups, ask-user answers
- `server-stop-shutdown-and-reset`: stop, force-interrupt, SIGTERM, server reset, self-update barrier
- `server-agent-state`: per-task state object and invariants
- `server-tab-registry`: shared tabs, `ready` replay
- `server-remote-access-auth`: password, localhost lockdown, rate limiting
- `server-cloudflare-tunnel`: quick/named tunnels, survival across restarts, ntfy URL
- `server-tls-local-ca`: certificates and `--trust-ca`
- `server-sorcar-run-api`: `kiss.server.sorcar.run` client
- `seas-agent-script-contract`: SEA scripts and tools files
- `server-voice-wake-and-talk`: wake word and talk audio
- `server-stall-watchdog`: GIL stall diagnostics
- `server-concurrency-invariants`: lock order and bounded waits
- `server-rsorcar-deploy`: remote deploy script and launchers

## Sources
- `src/kiss/server/web_server.py` (`RemoteAccessServer`, `WebPrinter`, `main`)
- `src/kiss/server/sorcar.py` (module docstring, `ServerApi`)
- `src/kiss/server/server.py` (`VSCodeServer`)
- `src/kiss/agents/sorcar/daemon_client.py`
