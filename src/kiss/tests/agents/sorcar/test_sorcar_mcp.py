# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for sorcar MCP management.

These exercise the real behaviour end to end: MCP config files are
written to real user (``KISS_HOME``) and project directories, a real
stdio MCP server (FastMCP) is spawned and spoken to over the real
protocol, tool wrappers are built and invoked for real, the OAuth
token storage and callback server use the real filesystem and real
HTTP, and the ``sorcar mcp`` CLI plus the REPL ``/mcp`` command are
driven through real subprocesses.  No model calls are made.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import pty
import subprocess
import sys
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar.mcp_servers import (
    FileTokenStorage,
    MCPManager,
    MCPServerConfig,
    _noninteractive_redirect,
    _result_text,
    load_mcp_permissions,
    load_mcp_servers,
    make_mcp_tool_wrapper,
    make_mcp_tools,
    mcp_tool_permission,
    remove_mcp_server,
    save_mcp_server,
)
from kiss.ui.cli.mcp_cli import _OAuthCallbackServer, run_mcp_cli

_SERVER_SCRIPT = '''
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("testsrv")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


@mcp.tool()
def secret_word(prefix: str = "") -> str:
    """Return the secret magic word, optionally prefixed."""
    return prefix + "XYLOPHONE-99"


if __name__ == "__main__":
    mcp.run()
'''


@pytest.fixture
def isolated_homes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every user-level MCP location into *tmp_path*."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / ".kisshome"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "project").mkdir()
    return tmp_path


def _write_server_script(tmp_path: Path) -> Path:
    """Write the FastMCP stdio test server script and return its path."""
    script = tmp_path / "testsrv.py"
    script.write_text(_SERVER_SCRIPT, encoding="utf-8")
    return script


def _stdio_config(tmp_path: Path, name: str = "testsrv") -> MCPServerConfig:
    """Return a stdio server config running the FastMCP test server."""
    return MCPServerConfig(
        name=name,
        transport="stdio",
        command=sys.executable,
        args=(str(_write_server_script(tmp_path)),),
    )


@pytest.fixture
def mcp_permission_rules() -> object:
    """Write ``mcp_permissions`` rules to the real config and restore.

    ``vscode_config.CONFIG_PATH`` is bound at import time to the
    session-level ``KISS_HOME`` set by conftest.py, so in-process
    permission tests must write there (and restore afterwards).
    """
    from kiss.core.vscode_config import CONFIG_PATH

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    original = CONFIG_PATH.read_text() if CONFIG_PATH.exists() else None

    def _write(rules: dict[str, str]) -> None:
        CONFIG_PATH.write_text(
            json.dumps({"mcp_permissions": rules}), encoding="utf-8",
        )

    yield _write
    if original is None:
        CONFIG_PATH.unlink(missing_ok=True)
    else:
        CONFIG_PATH.write_text(original)


@pytest.fixture
def real_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> Iterator[None]:
    """Give ``sys.stdin`` (and the MCP errlog) a real file descriptor.

    The MCP stdio client transport spawns the server subprocess through
    anyio's ``open_process``, passing ``stderr=errlog`` where ``errlog``
    defaults to ``sys.stderr``; ``subprocess`` then calls
    ``errlog.fileno()`` on it.  Under pytest the std streams are replaced
    by in-memory capture objects (and ``sys.stdin`` by the
    ``DontReadFromInput`` stub) whose ``.fileno()`` raises
    ``io.UnsupportedOperation: fileno``, so the transport fails to start
    and the server is reported unavailable.

    This fixture opens a pseudo-terminal with :func:`pty.openpty` and
    points ``sys.stdin`` at its slave end, giving stdin a real OS file
    descriptor.  The server's stderr (the ``errlog``) is sent to a plain
    file: a regular file always has a real ``fileno`` and, unlike a pipe
    or pty, never blocks the child no matter how much it logs.

    ``mcp.client.stdio.stdio_client`` binds ``errlog=sys.stderr`` as a
    *default argument* at import time, so monkeypatching ``sys.stderr``
    afterwards does not reach that already-captured object; the bound
    default is therefore repointed at the same file.  ``sys.stdout`` is
    left untouched so ``capsys`` still captures the command's output.
    Everything is restored and every descriptor closed afterwards.
    """
    master_fd, slave_fd = pty.openpty()
    stdin_stream = os.fdopen(slave_fd, "r", closefd=True)
    errlog = (tmp_path / "mcp_errlog.txt").open("w", encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stdin_stream)
    monkeypatch.setattr(sys, "stderr", errlog)

    from mcp.client.stdio import stdio_client

    wrapped = stdio_client.__wrapped__  # type: ignore[attr-defined]
    monkeypatch.setattr(wrapped, "__defaults__", (errlog,))
    try:
        yield
    finally:
        errlog.close()
        stdin_stream.close()
        os.close(master_fd)


# ---------------------------------------------------------------------------
# Configuration files


def test_save_and_load_user_scope(isolated_homes: Path) -> None:
    """A server saved in the user scope is loaded back with source=user."""
    project = isolated_homes / "project"
    cfg = MCPServerConfig(name="s1", command="echo", args=("hi",))
    path = save_mcp_server(cfg, "user", str(project))
    assert path == isolated_homes / ".kisshome" / "mcp.json"
    servers = load_mcp_servers(str(project))
    assert servers["s1"].command == "echo"
    assert servers["s1"].args == ("hi",)
    assert servers["s1"].source == "user"


def test_project_overrides_user(isolated_homes: Path) -> None:
    """A project-scope server wins over a same-named user server."""
    project = isolated_homes / "project"
    save_mcp_server(
        MCPServerConfig(name="s", command="user-cmd"), "user", str(project),
    )
    save_mcp_server(
        MCPServerConfig(name="s", command="proj-cmd"), "project", str(project),
    )
    servers = load_mcp_servers(str(project))
    assert servers["s"].command == "proj-cmd"
    assert servers["s"].source == "project"


def test_claude_mcp_json_compat(isolated_homes: Path) -> None:
    """Claude Code's ``.mcp.json`` is read; ``type`` is inferred."""
    project = isolated_homes / "project"
    (project / ".mcp.json").write_text(json.dumps({
        "mcpServers": {
            "local": {"command": "echo", "args": ["x"], "env": {"K": "V"}},
            "remote": {"url": "https://example.com/mcp"},
            "ssesrv": {"type": "sse", "url": "https://example.com/sse"},
        }
    }), encoding="utf-8")
    servers = load_mcp_servers(str(project))
    assert servers["local"].transport == "stdio"
    assert servers["local"].env == (("K", "V"),)
    assert servers["local"].source == "claude-project"
    assert servers["remote"].transport == "http"
    assert servers["remote"].url == "https://example.com/mcp"
    assert servers["ssesrv"].transport == "sse"


def test_lenient_parsing_skips_bad_entries(isolated_homes: Path) -> None:
    """Unusable entries (not dicts, missing command/url) are skipped."""
    project = isolated_homes / "project"
    (project / ".mcp.json").write_text(json.dumps({
        "mcpServers": {
            "notadict": "echo hi",
            "no-command": {"type": "stdio"},
            "no-url": {"type": "http"},
            "good": {"command": "echo"},
        }
    }), encoding="utf-8")
    servers = load_mcp_servers(str(project))
    assert set(servers) == {"good"}


def test_remove_from_all_scopes(isolated_homes: Path) -> None:
    """``remove_mcp_server`` deletes the entry from every config file."""
    project = isolated_homes / "project"
    save_mcp_server(MCPServerConfig(name="s", command="a"), "user", str(project))
    save_mcp_server(MCPServerConfig(name="s", command="b"), "project", str(project))
    removed = remove_mcp_server("s", str(project))
    assert len(removed) == 2
    assert load_mcp_servers(str(project)) == {}
    assert remove_mcp_server("s", str(project)) == []


def test_to_json_roundtrip_remote(isolated_homes: Path) -> None:
    """Remote server configs round-trip through the JSON files."""
    project = isolated_homes / "project"
    cfg = MCPServerConfig(
        name="r", transport="http", url="https://x.example/mcp",
        headers=(("Authorization", "Bearer t"),),
    )
    save_mcp_server(cfg, "project", str(project))
    loaded = load_mcp_servers(str(project))["r"]
    assert loaded.url == cfg.url
    assert loaded.headers == cfg.headers
    assert loaded.transport == "http"


# ---------------------------------------------------------------------------
# Permission wildcards


def test_permission_wildcards_last_rule_wins() -> None:
    """Wildcard rules cover MCP tools with last-match-wins semantics."""
    rules = {"*": "allow", "mymcp_*": "deny"}
    assert mcp_tool_permission("mymcp_search", rules) == "deny"
    assert mcp_tool_permission("other_search", rules) == "allow"
    # Later allow rule re-enables one tool of a denied server.
    rules = {"mymcp_*": "deny", "mymcp_safe": "allow"}
    assert mcp_tool_permission("mymcp_safe", rules) == "allow"
    assert mcp_tool_permission("mymcp_rm", rules) == "deny"
    # Default is allow when no rule matches.
    assert mcp_tool_permission("anything", {}) == "allow"


def test_load_mcp_permissions_from_config(mcp_permission_rules) -> None:
    """``mcp_permissions`` is read from the kiss ``config.json``."""
    mcp_permission_rules({"*": "allow", "internal_*": "DENY "})
    rules = load_mcp_permissions()
    assert rules == {"*": "allow", "internal_*": "deny"}
    assert mcp_tool_permission("internal_x", rules) == "deny"


# ---------------------------------------------------------------------------
# Live stdio server: manager, wrappers, tool calls


def test_make_mcp_tools_live_call(
    isolated_homes: Path, tmp_path: Path,
) -> None:
    """Tools of a real stdio server are wrapped and callable."""
    project = isolated_homes / "project"
    save_mcp_server(_stdio_config(tmp_path), "user", str(project))
    tools = make_mcp_tools(str(project))
    names = {t.__name__ for t in tools}
    assert names == {"testsrv_add", "testsrv_secret_word"}

    add = next(t for t in tools if t.__name__ == "testsrv_add")
    sig = inspect.signature(add)
    assert sig.parameters["a"].annotation is int
    assert sig.parameters["a"].default is inspect.Parameter.empty
    assert "Add two integers" in (add.__doc__ or "")
    assert add(a=2, b=3) == "5"

    secret = next(t for t in tools if t.__name__ == "testsrv_secret_word")
    assert inspect.signature(secret).parameters["prefix"].default is None
    assert secret() == "XYLOPHONE-99"
    assert secret(prefix="say-") == "say-XYLOPHONE-99"


def test_make_mcp_tools_permission_filter(
    isolated_homes: Path, tmp_path: Path, mcp_permission_rules,
) -> None:
    """Denied MCP tools are never registered."""
    mcp_permission_rules({"*": "allow", "testsrv_secret*": "deny"})
    project = isolated_homes / "project"
    save_mcp_server(_stdio_config(tmp_path), "user", str(project))
    names = {t.__name__ for t in make_mcp_tools(str(project))}
    assert names == {"testsrv_add"}


def test_make_mcp_tools_no_servers_is_empty(isolated_homes: Path) -> None:
    """No configured servers → no tools, instantly."""
    assert make_mcp_tools(str(isolated_homes / "project")) == []


def test_make_mcp_tools_broken_server_skipped(isolated_homes: Path) -> None:
    """A server whose command cannot start is skipped, not fatal."""
    project = isolated_homes / "project"
    save_mcp_server(
        MCPServerConfig(name="broken", command="/nonexistent-cmd-xyz"),
        "user", str(project),
    )
    assert make_mcp_tools(str(project)) == []


def test_call_tool_unknown_server_errors() -> None:
    """Calling a tool on a never-connected server returns an error string."""
    out = MCPManager.instance().call_tool("no-such-server", "t", {})
    assert out.startswith("Error: MCP server 'no-such-server' is not connected")


def test_result_text_flattening() -> None:
    """``CallToolResult`` content is flattened; errors are prefixed."""
    from mcp.types import CallToolResult, TextContent

    ok = CallToolResult(
        content=[
            TextContent(type="text", text="one"),
            TextContent(type="text", text="two"),
        ],
    )
    assert _result_text(ok) == "one\ntwo"
    err = CallToolResult(
        content=[TextContent(type="text", text="boom")], isError=True,
    )
    assert _result_text(err) == "Error: boom"
    empty = CallToolResult(content=[])
    assert _result_text(empty) == "(empty result)"


def test_wrapper_docstring_carries_schema_descriptions() -> None:
    """Wrapper Args lines carry the inputSchema property descriptions."""
    from mcp.types import Tool

    tool = Tool(
        name="search",
        description="Search the index.\nSecond line.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The query text"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    )
    wrapper = make_mcp_tool_wrapper(MCPManager.instance(), "my srv", tool)
    assert wrapper.__name__ == "my_srv_search"
    doc = wrapper.__doc__ or ""
    assert doc.startswith("Search the index. Second line.")
    assert "query: The query text" in doc
    assert "limit: See tool description. (optional)" in doc
    params = inspect.signature(wrapper).parameters
    assert params["limit"].annotation is int
    assert params["limit"].default is None


# ---------------------------------------------------------------------------
# OAuth: token storage, callback server, non-interactive guard


def test_file_token_storage_roundtrip(isolated_homes: Path) -> None:
    """Tokens and client info persist to a 0600 file and clear()."""
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

    storage = FileTokenStorage("my/server")
    assert asyncio.run(storage.get_tokens()) is None
    tokens = OAuthToken(
        access_token="at-1", token_type="Bearer", refresh_token="rt-1",
    )
    asyncio.run(storage.set_tokens(tokens))
    info = OAuthClientInformationFull.model_validate(
        {"client_id": "cid",
         "redirect_uris": ["http://localhost:1234/callback"]},
    )
    asyncio.run(storage.set_client_info(info))

    assert storage.path.name == "my_server.json"
    assert (storage.path.stat().st_mode & 0o777) == 0o600
    back = asyncio.run(storage.get_tokens())
    assert back is not None and back.access_token == "at-1"
    back_info = asyncio.run(storage.get_client_info())
    assert back_info is not None and back_info.client_id == "cid"

    assert storage.clear() is True
    assert storage.clear() is False
    assert asyncio.run(storage.get_tokens()) is None


def test_oauth_callback_server_receives_redirect() -> None:
    """The local callback server captures code and state from a real GET."""
    server = _OAuthCallbackServer()
    try:
        url = f"http://localhost:{server.port}/callback?code=abc123&state=st9"
        with urllib.request.urlopen(url, timeout=10) as resp:
            assert resp.status == 200
            assert b"close this tab" in resp.read()
        code, state = server.wait(timeout=10)
        assert code == "abc123"
        assert state == "st9"
    finally:
        server.close()


def test_noninteractive_oauth_refuses_browser_flow() -> None:
    """Agent runs never block on OAuth; they direct to ``sorcar mcp auth``."""
    with pytest.raises(RuntimeError, match="sorcar mcp auth"):
        asyncio.run(_noninteractive_redirect("https://auth.example/authorize"))


# ---------------------------------------------------------------------------
# ``sorcar mcp`` CLI (in-process)


def test_cli_add_list_get_remove(
    isolated_homes: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The add → list → get → remove lifecycle works end to end."""
    project = str(isolated_homes / "project")
    assert run_mcp_cli(
        ["add", "--env", "K=V", "echo-srv", "--", "echo", "hello"], project,
    ) == 0
    assert run_mcp_cli(["list"], project) == 0
    assert run_mcp_cli(["get", "echo-srv"], project) == 0
    assert run_mcp_cli(["remove", "echo-srv"], project) == 0
    out = capsys.readouterr().out
    assert "Added stdio MCP server 'echo-srv'" in out
    assert "echo hello" in out
    assert '"command": "echo"' in out
    assert "Removed 'echo-srv'" in out
    assert run_mcp_cli(["get", "echo-srv"], project) == 1


def test_cli_add_remote_with_headers(
    isolated_homes: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Remote servers are added with transport and headers."""
    project = str(isolated_homes / "project")
    assert run_mcp_cli(
        [
            "add", "--transport", "http", "--scope", "project",
            "--header", "X-Key: abc", "rsrv", "https://x.example/mcp",
        ],
        project,
    ) == 0
    cfg = load_mcp_servers(project)["rsrv"]
    assert cfg.transport == "http"
    assert cfg.headers == (("X-Key", "abc"),)
    assert cfg.source == "project"
    assert "Added http MCP server 'rsrv'" in capsys.readouterr().out


def test_cli_add_missing_target_fails(
    isolated_homes: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """``add`` without a command/URL fails with a message."""
    assert run_mcp_cli(["add", "x"], str(isolated_homes / "project")) == 1
    assert "missing command" in capsys.readouterr().out


def test_cli_add_bad_env_exits(isolated_homes: Path) -> None:
    """``--env`` entries must look like KEY=VALUE."""
    with pytest.raises(SystemExit):
        run_mcp_cli(
            ["add", "--env", "NOEQUALS", "x", "echo"],
            str(isolated_homes / "project"),
        )


def test_cli_unknown_server_paths(
    isolated_homes: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """get/remove/auth/debug on unknown servers exit 1."""
    project = str(isolated_homes / "project")
    for argv in (
        ["get", "nope"], ["remove", "nope"], ["auth", "nope"], ["debug", "nope"],
    ):
        assert run_mcp_cli(argv, project) == 1
    assert capsys.readouterr().out.count("Unknown MCP server: nope") == 4


def test_cli_logout(
    isolated_homes: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """``logout`` deletes the token file and reports when absent."""
    from mcp.shared.auth import OAuthToken

    project = str(isolated_homes / "project")
    storage = FileTokenStorage("srv")
    asyncio.run(storage.set_tokens(OAuthToken(access_token="a", token_type="Bearer")))
    assert run_mcp_cli(["logout", "srv"], project) == 0
    assert run_mcp_cli(["logout", "srv"], project) == 1
    out = capsys.readouterr().out
    assert "Deleted stored OAuth tokens" in out
    assert "No stored OAuth tokens" in out


def test_cli_auth_stdio_is_noop(
    isolated_homes: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """``auth`` on a stdio server explains no login is needed."""
    project = str(isolated_homes / "project")
    save_mcp_server(_stdio_config(tmp_path), "user", project)
    assert run_mcp_cli(["auth", "testsrv"], project) == 0
    assert "needs no OAuth login" in capsys.readouterr().out


def test_cli_debug_live_server(
    isolated_homes: Path, tmp_path: Path,
    capsys: pytest.CaptureFixture[str], mcp_permission_rules,
    real_stdin: None,
) -> None:
    """``debug`` connects for real and dumps tools with permissions."""
    mcp_permission_rules({"testsrv_secret*": "deny"})
    project = str(isolated_homes / "project")
    save_mcp_server(_stdio_config(tmp_path), "user", project)
    assert run_mcp_cli(["debug", "testsrv"], project) == 0
    out = capsys.readouterr().out
    assert "✓ Connected." in out
    assert "✓ testsrv_add [allow]" in out
    assert "✗ testsrv_secret_word [deny]" in out
    assert "inputSchema" in out


def test_cli_debug_connection_failure(
    isolated_homes: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """``debug`` reports a failure to start the server command."""
    project = str(isolated_homes / "project")
    save_mcp_server(
        MCPServerConfig(name="bad", command="/nonexistent-cmd-xyz"),
        "user", project,
    )
    assert run_mcp_cli(["debug", "bad"], project) == 1
    assert "✗ Connection failed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Subprocesses: the real ``sorcar mcp`` entry point and the REPL ``/mcp``


def _subprocess_env(homes: Path) -> dict[str, str]:
    """Environment for subprocesses with isolated user dirs."""
    return dict(
        os.environ,
        KISS_HOME=str(homes / ".kisshome"),
        HOME=str(homes / "home"),
    )


def test_sorcar_mcp_subcommand_subprocess(
    isolated_homes: Path, tmp_path: Path,
) -> None:
    """``python -m kiss.ui.cli.sorcar_cli mcp`` dispatches to the CLI."""
    project = isolated_homes / "project"
    save_mcp_server(_stdio_config(tmp_path), "user", str(project))
    proc = subprocess.run(
        [
            sys.executable, "-m",
            "kiss.ui.cli.sorcar_cli", "mcp", "list",
        ],
        capture_output=True, text=True, timeout=120,
        cwd=str(project), env=_subprocess_env(isolated_homes),
    )
    assert proc.returncode == 0
    assert "testsrv" in proc.stdout
    assert "(user, stdio)" in proc.stdout


def test_agent_get_tools_includes_mcp_tools(
    isolated_homes: Path, tmp_path: Path, real_stdin: None,
) -> None:
    """``SorcarAgent._get_tools`` exposes MCP tools to the agent."""
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent

    project = isolated_homes / "project"
    save_mcp_server(_stdio_config(tmp_path), "user", str(project))
    agent = SorcarAgent("mcp-test")
    agent.work_dir = str(project)
    agent._use_web_tools = False
    names = {t.__name__ for t in agent._get_tools()}
    assert "testsrv_add" in names
    assert "testsrv_secret_word" in names
