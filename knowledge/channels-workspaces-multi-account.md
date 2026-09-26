---
title: 'Multi-account channels: KISS_CHANNEL_WORKSPACE and enter_workspace'
uuid: 43fce487-4f9a-49bd-818f-ce7a2dfd0ad9
summary: 'Multi-account channels: ref-counted KISS_CHANNEL_WORKSPACE env var (enter_workspace/exit_workspace),
  blocking on conflicting workspaces, Slack per-workspace tokens.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Multi-account channels: `KISS_CHANNEL_WORKSPACE`

A channel module's `tools()` runs inside the kiss-web daemon and receives no arguments, so
the account ("workspace") to authenticate as travels through the **process-global env var
`KISS_CHANNEL_WORKSPACE`** (`channel_workspace.WORKSPACE_ENV_VAR`). Only Slack reads it today
(`slack_sea.tools()` → `SlackAgent(workspace=os.environ.get("KISS_CHANNEL_WORKSPACE", "default") or "default")`).

## The registry
`sorcar/channel_workspace.py` keeps `_ACTIVE_WORKSPACES: dict[str, int]` guarded by a
`threading.Condition`:
- `enter_workspace(ws, timeout)` waits while any *different* workspace is active, then
  increments `ws`'s count and exports the env var. Same-workspace launches overlap freely.
  Returns `False` on timeout; the caller must then not launch and must not call
  `exit_workspace`.
- `exit_workspace(ws)` decrements; when nothing is active it removes the env var and
  `notify_all()`s waiters.

Why reference counting instead of save/restore: overlapping launches that snapshot and
restore the env var restore each other's values out of order, leaving a stale workspace
exported, or hand a running session the wrong account's credentials.

## Callers
- `agent_dispatch._run_agent` (channel mode) with `timeout=min(call timeout, WORKSPACE_WAIT_TIMEOUT_SECONDS=900)`;
  on failure returns "workspace ... could not be activated ... retry when it finishes".
- `_kiss_web_launcher.run_agent_via_kiss_web` (the `kiss-<channel>` CLIs and gateway ticks).
The module lives in the sorcar layer so dispatch can use it without importing
`third_party_agents` (layering invariant); the launcher imports it back, sharing one registry.

## Slack specifics
- Tokens: `~/.kiss/third_party_agents/slack/<workspace>/token.json` (key `access_token`).
  This path is hard-coded under `Path.home()/.kiss`, not `$KISS_HOME` (`_SLACK_DIR`). A legacy
  top-level `slack/token.json` is migrated to the `default` workspace.
- Muse vault service: `slack` for `default`; otherwise `slack-<slug>-<16 hex sha256>`
  (`_muse_service`). The 64-bit digest avoids collisions between workspace names that
  sanitize to the same slug. `muse_auth._common.builtin_hosts` lets `slack-*` inherit Slack's
  host allowlist, but each workspace service has its own independent policy.
- CLI extras: `kiss-slack --list-workspaces`, `--delete-workspace WS`; every CLI accepts
  `--workspace WS` (default `default`).
- Gateway state files include the workspace in their name (`derive_state_path`).

## Sources
- `src/kiss/agents/sorcar/channel_workspace.py` (`enter_workspace`, `exit_workspace`, `WORKSPACE_ENV_VAR`)
- `src/kiss/agents/sorcar/agent_dispatch.py` (`_run_agent`, `WORKSPACE_WAIT_TIMEOUT_SECONDS`)
- `src/kiss/agents/third_party_agents/slack_sea.py` (`tools`, `_token_path`, `_muse_service`, `_list_workspaces`)
- `src/kiss/agents/third_party_agents/muse_auth/_common.py` (`service_root`, `policy_service`, `builtin_hosts`)
