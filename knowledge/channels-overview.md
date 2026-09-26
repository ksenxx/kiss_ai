---
title: Third-party channel agents, cron and credentials — area map
uuid: 42beedc6-ff59-4d58-a1c0-0e33c6bf8138
summary: Area map of third_party_agents channel agents (*_sea.py), _channel_agent_utils,
  muse_auth vault, sorcar cron_agent, agent_dispatch run_agent, channel_workspace.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Third-party channel agents, cron and credentials — area map

Sorcar talks to external services (Slack, Telegram, Gmail, GitHub, ...) through **channel
agents**: one `<name>_sea.py` module per service in `src/kiss/agents/third_party_agents/`.
A channel agent is not an executable agent; it is a *carrier* of channel identity whose
module-level `tools()` hands the kiss-web daemon a list of authenticated tool callables.

## Files
| Path | Role | Page |
| --- | --- | --- |
| `third_party_agents/<name>_sea.py` (45 files: 42 channels plus the non-channel `ask_sea`, `a2a_sea`, `oai_sea`, excluded by `agent_dispatch._NON_CHANNEL_MODULES`) | one module per channel: backend + agent class + `tools()` + `main()` | `channels-agent-module-structure`, `channels-catalog` |
| `third_party_agents/_channel_agent_utils.py` | `ToolMethodBackend`, `ChannelConfig`, `BaseChannelAgent`, `ChannelRunner`, `channel_main`, state-file helpers | `channels-agent-module-structure`, `channels-gateway-runner`, `channels-gateway-state-file` |
| `third_party_agents/_channel_cli.py` | shared argparse for `kiss-<channel>` CLIs | `channels-cli-flags` |
| `third_party_agents/_kiss_web_launcher.py` | `run_agent_via_kiss_web`, `KissWebChatAgent`: submit a task to the daemon | `channels-agent-module-structure` |
| `third_party_agents/_device_auth.py`, `_browser_handoff.py`, `_google_workspace_utils.py` | sign-in flows (RFC 8628 device grant, Nextcloud Login Flow v2, Google OAuth consent, default-browser hand-off) | `channels-authentication-flows` |
| `third_party_agents/_backend_utils.py` | embedded HTTP server / queue helpers for callback-style backends | `channels-gateway-runner` |
| `third_party_agents/auth_status.py` | subprocess that reports connected/not-connected for the Apps panel | `channels-auth-status` |
| `third_party_agents/muse_auth/` | credential vault daemon, surrogate tokens, Sentinel egress policy | `muse-auth-architecture`, `muse-auth-services-and-cli` |
| `sorcar/agent_dispatch.py` | the `run_agent` tool: channel / cron / agent-script dispatch | `channels-run-agent-dispatch` |
| `sorcar/channel_workspace.py` | ref-counted `KISS_CHANNEL_WORKSPACE` for multi-account channels | `channels-workspaces-multi-account` |
| `sorcar/cron_agent.py` | scheduled jobs (`jobs.json`), scheduler thread, delivery, `gateway_command` | `cron-job-store-and-schedules`, `cron-execution-and-delivery`, `cron-dispatch-and-gateways` |

Other pages: `channels-config-storage` (where credentials live), `channels-adding-a-new-channel`.

## Key invariants
- The sorcar layer never imports `kiss.agents.third_party_agents` statically; channel modules
  are soft plugins found by directory scan (`available_channels`) and imported by name
  (enforced by `kiss.tests.agents.sorcar.test_layering_invariants`). That is why
  `channel_workspace.py` lives in `sorcar/` and `_agent_class` matches `BaseChannelAgent` by
  class name rather than `issubclass`.
- Every task runs on the kiss-web daemon via `kiss.server.sorcar.run`; channel tools reach it
  through the `tools=` *file path* contract (the channel module's own file).
- State root is `$KISS_HOME` (default `~/.kiss`): configs under
  `third_party_agents/<service>/`, cron under `cron/`, vault under `muse_auth/`. Exception: Slack
  workspace tokens use a hard-coded `Path.home()/.kiss/third_party_agents/slack/`.
- Channel and cron dispatches never get a git worktree or auto-commit (`git_lifecycle=False`).

## Sources
- `src/kiss/agents/third_party_agents/README.md`
- `src/kiss/agents/sorcar/agent_dispatch.py` (module docstring, `available_channels`, `_agent_class`)
- `src/kiss/agents/sorcar/channel_workspace.py` (module docstring)
