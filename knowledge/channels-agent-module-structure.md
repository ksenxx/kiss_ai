---
title: Anatomy of a channel agent module (*_sea.py)
uuid: 482d6736-0ec8-40e2-b2da-a1c6a0fc6296
summary: 'Anatomy of a *_sea.py channel module: ToolMethodBackend public methods become
  tools, BaseChannelAgent carrier, auth tools, tools(), _make_backend, main() via
  channel_main.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Anatomy of a channel agent module (`*_sea.py`)

A typical service module (e.g. `telegram_sea.py`) follows the same five-part convention:

1. **`_config = ChannelConfig(<dir>, required_keys)`** — module-level config handle
   (see `channels-config-storage`). Optional: Slack and the Google modules have none, and
   `derive_state_path`/`channel_override_config` handle its absence. Its presence also decides where gateway state files go
   (`derive_state_path`) and enables per-channel model/budget overrides.
2. **`<Name>ChannelBackend(ToolMethodBackend)`** — the API client. `get_tool_methods()`
   returns every *public* callable, sorted by name, minus `_NON_TOOL_METHODS` (the channel
   protocol: `connect`, `find_channel`, `find_user`, `join_channel`, `poll_messages`,
   `send_message`, `send_typing`, `is_from_bot`, `strip_bot_mention`, `disconnect`,
   `get_tool_methods`, `poll_thread_messages`, `ack_message`, `bind_channel_state`). So
   **adding a public method adds an LLM tool**; its docstring is the tool description.
   Helpers must be `_`-prefixed. `ToolMethodBackend` supplies defaults: `find_channel` /
   `find_user` echo the input, `is_from_bot` etc. are trivial.
3. **`<Name>Agent(BaseChannelAgent)`** — exactly one subclass defined in the module (dispatch
   finds it with `agent_dispatch._agent_class`). Sets `self._backend`, overrides
   `_is_authenticated()` and `_get_auth_tools()` (closures such as `check_telegram_auth`,
   `authenticate_telegram`, `clear_telegram_auth`), and may set `channel_system_prompt`
   (appended to dispatched prompts). `_get_tools()` = auth tools always + backend tools only
   when authenticated.
4. **`tools()`** — `return <Name>Agent()._get_tools()`. This is the
   `kiss.server.sorcar.run` tools-file contract: the daemon imports the module file and calls
   it. `agent_tools_file(cls)` returns `""` for modules without `tools()`.
5. **`main()`** — `channel_main(<Name>Agent, "kiss-<name>", channel_name=..., make_backend=_make_backend)`,
   registered in `pyproject.toml` `[project.scripts]`. Optional **`_make_backend()`** returns a
   configured backend (or `sys.exit(1)` when unconfigured); it enables gateway poll mode and
   cron delivery. API-only services omit it.

## How a run executes
`BaseChannelAgent.run()` → `_kiss_web_launcher.run_agent_via_kiss_web(agent, prompt, **filter_launch_kwargs(kwargs))`.
The launcher starts (or reuses) an in-process daemon (`_ensure_api_server`), publishes the
workspace via `enter_workspace` (`KISS_CHANNEL_WORKSPACE`), submits `run` with
`tools=agent.tools_file`, appends `channel_system_prompt`, blocks, and writes
`last_run_result`, `budget_used`, `total_tokens_used`, `total_steps` back onto the agent.
`filter_launch_kwargs` silently drops anything outside `LAUNCH_KWARG_NAMES` (legacy
`SorcarAgent.run` kwargs such as `system_prompt`). `KissWebChatAgent` is a tool-less carrier
used by the gateway runner to resume a daemon chat (`resume_chat_by_id`).

## Gotchas
- Backend tools are snapshotted when a session starts: the session that stores a new token
  cannot call the new backend tools; the real work must be the next prompt (README
  "Authenticate by chatting").
- Muse-auth covered agents wire a surrogate in `__init__` (e.g. `TelegramAgent` calls
  `_backend._wire_muse()`); on `MuseAuthError` the agent stays constructible and tokenless so
  its auth tools still work (fail closed).
- History: agents were once `SorcarAgent` subclasses; commit `feb43d002` made them
  daemon-launched carriers, and `2a4b056a9` routed launches through `kiss.server.sorcar.run`.

## Sources
- `src/kiss/agents/third_party_agents/_channel_agent_utils.py` (`ToolMethodBackend`, `_NON_TOOL_METHODS`, `BaseChannelAgent`, `agent_tools_file`, `filter_launch_kwargs`, `LAUNCH_KWARG_NAMES`)
- `src/kiss/agents/third_party_agents/_kiss_web_launcher.py` (`run_agent_via_kiss_web`, `KissWebChatAgent`)
- `src/kiss/agents/third_party_agents/telegram_sea.py` (`TelegramAgent`, `tools`, `main`, `_make_backend`)
