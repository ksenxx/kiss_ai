---
title: How run_agent dispatches by channel name, cron, or agent-script path
uuid: d4151093-423a-43ab-a80f-8140fb1a206a
summary: 'How run_agent dispatches: path mode (.py SEA, dummy_sea default), cron,
  channel lookup via _squash/available_channels, preamble, channel_work dir, timeouts,
  errors.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# How `run_agent` dispatches

`make_run_agent_tool(work_dir, parent_agent)` builds the per-task `run_agent` tool; its body
is `_run_agent`. Every dispatch ends in `_dispatch`, a call of
`kiss.agents.sorcar.daemon_client.run` (public `kiss.server.sorcar.run`) with the target file as
`extension_agent_path`. Inside the daemon the call goes back through the daemon's own socket
(recorded by `cron_agent.start_scheduler_thread`, read by `_daemon_sock_path`); standalone it
uses `KISS_SORCAR_SOCK` then `$KISS_HOME/sorcar.sock`.

## Argument validation
`task` must be non-empty; `max_budget` a positive finite number; `timeout` positive finite
seconds, default `DEFAULT_DISPATCH_TIMEOUT_SECONDS` = 300. Other string options
(`chat_id`, `system_prompt`, `tools`, `use_worktree`, `tool_profile`, ...) are parsed by
`_parse_run_options` into `RunOptions`. An agent script's own getters win over them.

## Three modes (decided on `agent.strip() or DEFAULT_AGENT_PATH`)
1. **Path mode** — the value ends in `.py` or contains `/` or `\`. Relative paths resolve
   against the *calling task's* work dir; the sub-task runs there with the normal worktree /
   auto-commit lifecycle. Empty `agent` means `src/kiss/agents/seas/dummy_sea.py`, a plain
   Sorcar session. With `DEFAULT_CONFIG.dispatch_path_rewrite`, parent-repo paths in the task
   are rewritten (`rewrite_parent_repo_paths`).
2. **Cron** — `_squash(agent) == "cron"`: dispatches `cron_agent.py` with
   `CRON_DISPATCH_PREAMBLE + task`, work dir `~/.kiss/cron/work`, `git_lifecycle=False`,
   `classify=False` (cron is the one mode that defaults task classification off).
3. **Channel** — `_squash` lowercases and strips spaces/hyphens/underscores, then matches
   against `available_channels()` (a directory glob of `*_sea.py`, no imports, excluding
   `_`-prefixed files and `_NON_CHANNEL_MODULES = {a2a_sea, ask_sea, oai_sea}`). The module is
   imported, `_agent_class` finds its `BaseChannelAgent` subclass, and the prompt becomes
   `preamble + task + channel_system_prompt`. The preamble tells the sub-session to use the
   channel tools immediately, never call `run_agent`, never edit source or run tests.
   Work dir is `$KISS_HOME/channel_work` (shared with earlier channel sessions),
   `git_lifecycle=False`; classification stays at the daemon default. The workspace is
   entered with `enter_workspace(workspace, timeout=min(timeout, WORKSPACE_WAIT_TIMEOUT_SECONDS=900))`
   and released in `finally` (see `channels-workspaces-multi-account`).

## Errors
- Unknown name → `_unknown_agent_error`: generic labels in `_GENERIC_AGENT_NAMES`
  ("general", "reviewer", "subagent", ...) get "use run_parallel" advice; otherwise a
  `difflib` closest-match hint plus the channel list.
- On timeout the sub-task is stopped; if the daemon never confirms the stop the reply is
  `stop_unconfirmed_error(name, timeout)` ("MAY STILL BE RUNNING"). Cron compares against this
  exact string.
- A call that waits for a conflicting workspace and then for the task can take up to
  `timeout + min(timeout, 900)` plus the 20 s stop-confirm grace.

## Related entry points
- Slash commands: every `xxx_sea.py` is registered as `/xxx`; the daemon turns the prompt into
  a direct `run_agent` call on that file (README "How a prompt reaches a channel").
- Sub-task spend is attributed to the parent via `_attribute_dispatch_usage`.
- Commits `e0045736b` (no worktree for channel dispatch) and `11e038f7a` (classification on
  for channel dispatch, off for cron) set the current lifecycle rules.

## Sources
- `src/kiss/agents/sorcar/agent_dispatch.py` (`_run_agent`, `_dispatch`, `available_channels`, `_squash`, `_agent_class`, `_NON_CHANNEL_MODULES`, `_unknown_agent_error`, `make_run_agent_tool`, `stop_unconfirmed_error`, `DEFAULT_AGENT_PATH`)
