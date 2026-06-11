# Task: MCP management in Sorcar CLI — COMPLETE

`sorcar mcp add/list/get/remove/auth/logout/debug` incl. OAuth servers; permission
wildcards covering MCP tools; agent + REPL integration; tested extensively incl.
a real-model run of the actual sorcar CLI.

## What was built

1. **`src/kiss/agents/sorcar/mcp_servers.py`** (new)

   - Claude-compatible `{"mcpServers": {...}}` config from (low→high precedence):
     `~/.kiss/mcp.json` (user, honours KISS_HOME), `<work_dir>/.mcp.json`
     (Claude Code project file), `<work_dir>/.kiss/mcp.json` (project).
     Lenient parsing (`type` inferred from `command`/`url`); save/remove helpers.
   - `MCPManager` singleton: asyncio loop on a daemon thread; each server owned by
     ONE long-lived task (`_maintain_connection` with AsyncExitStack + stop event —
     anyio cancel scopes must enter/exit in the same task); sync facade
     `connect`/`call_tool`; stale connections stopped on config change; atexit
     shutdown. CONNECT_TIMEOUT=60s, CALL_TIMEOUT=300s.
   - Tool wrappers: `<server>_<tool>` `**kwargs` functions with synthesized
     `__signature__` + docstring (`Args:` lines) from the MCP inputSchema so
     kiss's signature/docstring schema builder reproduces it; results flattened
     (`Error:` prefix on isError).
   - **Permission wildcards**: `mcp_permissions` in `~/.kiss/config.json`,
     fnmatch patterns vs full `<server>_<tool>` name, last-match-wins (reuses
     `skills.skill_permission`); denied tools never registered.
   - **OAuth**: `FileTokenStorage` (tokens + client_info JSON at
     `~/.kiss/mcp_auth/<name>.json`, chmod 600); `build_oauth_provider` wraps the
     SDK's `OAuthClientProvider` (dynamic registration + PKCE); agent runs get
     non-interactive handlers that raise "run `sorcar mcp auth <name>`".
   - `make_mcp_tools(work_dir)` (fast [] when unconfigured; broken servers logged
     - skipped) and `format_mcp_listing(work_dir, connect=)`.

1. **`src/kiss/agents/sorcar/mcp_cli.py`** (new) — `run_mcp_cli(argv, work_dir)`:
   add (options-before-name, `--transport stdio|http|sse`, `--env`, `--header`,
   `--scope user|project`, `--` separator), list (`--ping` live status), get,
   remove, auth (`_OAuthCallbackServer` on localhost:0 + webbrowser +
   `--no-browser`), logout, debug (server info, capabilities, tools with
   inputSchema + allow/deny mark, resources, prompts). One-shot connects via
   `_connect_once` (single-task AsyncExitStack).

1. Wiring: `worktree_sorcar_agent.main()` dispatches `sorcar mcp ...`;
   `SorcarAgent._get_tools()` appends `make_mcp_tools` (guarded);
   `cli_repl.py` adds `/mcp` (live listing) + docs; `pyproject.toml` adds
   `mcp>=1.20.0` (was already a transitive dep).

## Testing

- `src/kiss/tests/agents/sorcar/test_sorcar_mcp.py`: **31 integration tests**
  (no mocks): config round-trips/scopes/.mcp.json compat/lenient parsing;
  permission wildcard semantics + config loading (via vscode_config.CONFIG_PATH,
  bound at import to conftest's KISS_HOME); live FastMCP stdio server spawn +
  wrapper signature/docstring checks + real tool calls; broken-server skip;
  CallToolResult flattening; FileTokenStorage round-trip (0600) + clear;
  real-HTTP OAuth callback server; non-interactive OAuth refusal; full CLI
  lifecycle in-process; `python -m ... mcp list` subprocess; REPL `/mcp` and
  `/help` subprocesses; `SorcarAgent._get_tools` exposure. All pass.
- Manual smoke: `sorcar mcp add/list/list --ping/get/debug/remove/logout` for a
  stdio server; `add --transport http` + `debug` + `auth --no-browser` against a
  local FastMCP streamable-http server ("Server did not require OAuth");
  `mcp_permissions {"testsrv_secret*": "deny"}` → 1/2 tools allowed.
- **Real-model e2e**: `uv run python -m kiss...worktree_sorcar_agent --no-web -b 1.0 -w <proj> -t "call testsrv_secret_word/testsrv_add"` → agent called both
  MCP tools, reported "the word is XYLOPHONE-99" and 42; $0.0567, 2 steps.
  Real `./sorcar` REPL: `/mcp` printed "✓ connected, 2/2 tools allowed".
- `uv run check --full` fully green; impacted suites (test_cli_repl,
  test_sorcar_agent, test_sorcar_skills, test_custom_commands — 109 tests) pass.
