---
title: MCP (Model Context Protocol) servers in Sorcar
uuid: f3660e76-d6fd-4900-bbf2-c82c0fdee85c
summary: 'MCP servers: mcp.json files and precedence, <server>_<tool> wrappers from
  inputSchema, mcp_permissions, OAuth token storage, MCPManager pool and timeouts.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# MCP servers

## Configuration
Servers use the Claude Code shape `{"mcpServers": {name: entry}}`. Files are read from low to high
precedence, and later files win on a name clash:
1. `~/.kiss/mcp.json` (respects `KISS_HOME`): user servers
2. `<work_dir>/.mcp.json`: Claude Code's project file, so repos configured for Claude Code work unchanged
3. `<work_dir>/.kiss/mcp.json`: native project servers

A local entry looks like `{"type": "stdio", "command", "args", "env"}` and a remote one like `{"type": "http"|"sse", "url", "headers"}`.
`type` defaults to `stdio` when `command` is present and `http` otherwise (`_parse_server_entry`). Writes go through
`save_mcp_server` / `remove_mcp_server` with an atomic replace (`_atomic_write_config`).

## Tools exposed to the agent
`make_mcp_tools(work_dir)` connects to each server, lists its tools, and wraps each one
(`make_mcp_tool_wrapper`) as a Python function named `<server>_<tool>`. The function's `__signature__`
and docstring are synthesized from the MCP `inputSchema` (`_json_schema_to_annotation`,
`_python_param_name`). The result is a simplified projection: it keeps primitive types, names, descriptions and
`required`, but drops enums, bounds and nested schemas.
- A server that fails to connect is logged and skipped. A malformed tool schema skips only that tool.
- Names cannot collide with built-in tools (`_RESERVED_TOOL_NAMES`: Bash, Read, finish, browser tools, skill,
  decide, run_parallel, summary, run_agent, cron_job, ...) or with each other. Sanitized names are not
  injective, so duplicates get the suffixes `_2`, `_3`, ...; a duplicate registration would abort the tool loop.
- `SorcarAgent._get_tools` loads MCP tools only for the `full` profile, inside a try/except.

## Permissions
The `mcp_permissions` key in `~/.kiss/config.json` holds wildcard rules matched against the full tool name, with the last
match winning (`{"*": "allow", "mymcp_*": "deny"}`). Denied tools are never registered. The same rule engine is used for
skills (`skills.load_permission_rules`).

## OAuth
Remote servers use the MCP SDK's OAuth 2.1 provider (`build_oauth_provider`). Tokens persist per server under
`~/.kiss/mcp_auth/` (`FileTokenStorage`: owner-only atomic writes, an inter-process lock, collision-free file
names that avoid Windows reserved basenames). Agent runs only reuse and refresh tokens. There is **no
interactive browser login**, so a server needing one fails with a hint to provision the tokens by hand.

## Connection manager
`MCPManager.instance()` is a process-wide singleton running a private asyncio loop on a daemon thread. Each server
connection is owned by one long-lived task, because anyio cancel scopes must enter and exit in the same task. Limits:
`CONNECT_TIMEOUT = 60`, `CALL_TIMEOUT = 300` s, `IDLE_TIMEOUT = 600` s (idle connections are reaped on the next
connect, which prevents leaking stdio children in the long-lived daemon), `MAX_CONNECTIONS = 8` (LRU eviction),
and `HEALTH_INTERVAL = 30` s pings, so a dead server is noticed instead of blocking a call until `CALL_TIMEOUT`.

## Sources
- `src/kiss/agents/sorcar/mcp_servers.py` (`load_mcp_servers`, `make_mcp_tools`, `make_mcp_tool_wrapper`, `MCPManager`, `FileTokenStorage`, `build_oauth_provider`, `mcp_tool_permission`, `_RESERVED_TOOL_NAMES`)
- `src/kiss/agents/sorcar/sorcar_agent.py` (`_get_tools`)
