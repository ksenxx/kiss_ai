---
title: How to add a new channel agent
uuid: 5a28b5a1-784a-4d02-91c4-c715102f3737
summary: 'Checklist to add a channel agent: <name>_sea.py, ChannelConfig _config,
  backend, agent class, tools(), _make_backend, pyproject script, Muse SERVICE_HOSTS,
  tests.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# How to add a new channel agent

Nothing is registered centrally: `available_channels()` globs `*_sea.py`, cron delivery and
`gateway_command` import `kiss.agents.third_party_agents.<name>_sea` by name, slash commands
register every `xxx_sea.py` as `/xxx`, and `auth_status` probes whatever the glob finds. So the
module file *is* the registration. Copy the shape of an existing module of the same kind
(`telegram_sea.py` for a polling bot, `ntfy_sea.py` for a small one, `github_sea.py` for an
API-only service, `dingtalk_sea.py`/`line_sea.py` for an embedded-callback backend).

## Checklist
1. **File name** `src/kiss/agents/third_party_agents/<name>_sea.py`, single lowercase word
   (commit `1f0f67bf3` renamed all modules to this form). The stem minus `_sea` is the channel
   name users say; `_squash` makes it case/space/hyphen-insensitive.
2. **Config**: `_config = ChannelConfig(Path.home()/".kiss"/"third_party_agents"/"<name>", ("token",))`.
   Keeping the name `_config` matters: `derive_state_path` and `channel_override_config` look
   up the module attribute `_config` to place gateway state and read
   `channel_model_name`/`channel_max_budget`.
3. **Backend**: `class <Name>ChannelBackend(ToolMethodBackend)`. Every public method becomes an
   LLM tool (docstring = description, typed args); keep helpers `_private`. For gateway support
   implement `connect`, `poll_messages(channel_id, oldest, limit) -> (messages, cursor)`,
   `send_message(channel_id, text, thread_ts="")`, `is_from_bot`, optionally
   `find_channel`/`find_user` (name → id), `poll_thread_messages` (thread continuity),
   `ack_message` (non-cursor polls), `send_typing`, `disconnect`. Messages are dicts with at
   least `ts`, `user`, `text`, optional `thread_ts`, `reply_count`.
4. **Agent**: exactly one `class <Name>Agent(BaseChannelAgent)` defined in the module; set
   `self._backend`; override `_is_authenticated` and `_get_auth_tools` (`check_<name>_auth`,
   `authenticate_<name>`, `clear_<name>_auth`); set a `channel_system_prompt` string (the
   dispatch test asserts it is a `str`). Accept `workspace=` only if the service is
   multi-account.
5. **`tools()`**: `return <Name>Agent()._get_tools()` (tools-file contract).
6. **`_make_backend()`** (optional): return a configured backend or `sys.exit(1)` with a
   message when unconfigured. Enables `--channel` poll mode and cron `deliver: <name>:<chat>`.
   Omit it for outbound-only APIs.
7. **`main()`**: `channel_main(<Name>Agent, "kiss-<name>", channel_name="<Label>", make_backend=_make_backend)`
   and add `kiss-<name> = "kiss.agents.third_party_agents.<name>_sea:main"` to
   `pyproject.toml` `[project.scripts]`.
8. **Muse-auth** (if the service uses an outbound bearer/header token): add the service and
   its allowed hosts to `muse_auth/_common.SERVICE_HOSTS` (empty tuple = bind to the enrolled
   origin for self-hosted servers), route requests through a surrogate
   (`client.bearer_surrogate`, `MuseBoundarySession`), and scrub the migrated secret with
   `_config.scrub_secrets(...)`. If the vault service name differs from the channel name, set
   module `_SERVICE` so `auth_status` finds it.
9. **Docs**: add a row to the catalog table in `third_party_agents/README.md` and update the
   counts there.
10. **Tests** in `src/kiss/tests/agents/third_party_agents/`: end-to-end tests against a local
    HTTP server (see `recording_http.py`, `muse_test_utils.py`), no mocks. The existing
    `test_agent_dispatch.py::test_every_channel_module_is_dispatchable` will import the new
    module automatically and fail if the contract is broken.

## Sources
- `src/kiss/agents/sorcar/agent_dispatch.py` (`available_channels`, `_agent_class`)
- `src/kiss/agents/third_party_agents/_channel_agent_utils.py` (`ToolMethodBackend`, `BaseChannelAgent`, `ChannelConfig`, `derive_state_path`, `channel_override_config`, `channel_main`)
- `src/kiss/agents/sorcar/cron_agent.py` (`_deliver_to_channel`, `gateway_command`)
- `src/kiss/tests/agents/third_party_agents/test_agent_dispatch.py` (`test_every_channel_module_is_dispatchable`)
