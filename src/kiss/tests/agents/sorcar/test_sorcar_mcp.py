# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for sorcar MCP management.

These exercise the real behaviour end to end: MCP config files are
written to real user (``KISS_HOME``) and project directories, a real
stdio MCP server (FastMCP) is spawned and spoken to over the real
protocol, tool wrappers are built and invoked for real, and the OAuth
token storage uses the real filesystem.  No model calls are made.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar.mcp_servers import (
    _MAX_DECLARED_NODES,
    FileTokenStorage,
    MCPManager,
    MCPServerConfig,
    _Connection,
    _refuse_login,
    _result_text,
    load_mcp_servers,
    make_mcp_tool_wrapper,
    make_mcp_tools,
    remove_mcp_server,
    save_mcp_server,
)
from kiss.agents.sorcar.skills import load_permission_rules, skill_permission
from kiss.tests.conftest import IS_WINDOWS

_SERVER_SCRIPT = '''
from mcp.server.fastmcp import FastMCP
from mcp.types import EmbeddedResource, ResourceLink, TextContent, TextResourceContents

mcp = FastMCP("testsrv")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


@mcp.tool()
def secret_word(prefix: str = "") -> str:
    """Return the secret magic word, optionally prefixed."""
    return prefix + "XYLOPHONE-99"


@mcp.tool()
def read_file(path: str):
    """Return a file as GitHub's server does: a note, the text as an embedded resource."""
    contents = TextResourceContents(uri="repo://" + path, text="BODY-OF-" + path)
    return [
        TextContent(type="text", text="downloaded " + path),
        EmbeddedResource(type="resource", resource=contents),
        ResourceLink(type="resource_link", name="history", uri="repo://" + path + "/history"),
    ]


if __name__ == "__main__":
    mcp.run()
'''


@pytest.fixture
def isolated_homes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Redirect every user-level MCP location into *tmp_path*.

    The servers the test connected through the process-wide manager
    (``make_mcp_tools``) are stopped afterwards: the manager keeps its
    children for the idle timeout, so they outlived the test session.
    """
    monkeypatch.setenv("KISS_HOME", str(tmp_path / ".kisshome"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "project").mkdir()
    yield tmp_path
    MCPManager.disconnect_existing()


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

    ``user_config.CONFIG_PATH`` is bound at import time to the
    session-level ``KISS_HOME`` set by conftest.py, so in-process
    permission tests must write there (and restore afterwards).
    """
    from kiss.core.user_config import CONFIG_PATH

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

    This fixture points ``sys.stdin`` at ``os.devnull``, giving stdin a
    real OS file descriptor on every platform.  The server's stderr (the
    ``errlog``) is sent to a plain file: a regular file always has a real
    ``fileno`` and, unlike a pipe, never blocks the child no matter how
    much it logs.

    ``mcp.client.stdio.stdio_client`` binds ``errlog=sys.stderr`` as a
    *default argument* at import time, so monkeypatching ``sys.stderr``
    afterwards does not reach that already-captured object; the bound
    default is therefore repointed at the same file.  ``sys.stdout`` is
    left untouched so ``capsys`` still captures the command's output.
    Everything is restored and every descriptor closed afterwards.
    """
    # ``os.devnull`` has a real descriptor on every platform (Windows has
    # no pty) and nothing reads stdin while the client talks to the child.
    stdin_stream = open(os.devnull, encoding="utf-8")
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



def test_save_and_load_user_scope(isolated_homes: Path) -> None:
    """A server saved in the user scope is loaded back."""
    project = isolated_homes / "project"
    cfg = MCPServerConfig(name="s1", command="echo", args=("hi",))
    path = save_mcp_server(cfg, "user", str(project))
    assert path == isolated_homes / ".kisshome" / "mcp.json"
    servers = load_mcp_servers(str(project))
    assert servers["s1"].command == "echo"
    assert servers["s1"].args == ("hi",)


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


def test_the_home_directory_is_no_project(
    isolated_homes: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A task working in the user's home reads and writes the live user config.

    ``~/.kiss`` is the stock brand's global home.  Under another brand
    its ``mcp.json`` is a leftover, which was loaded over the live
    ``$KISS_HOME/mcp.json`` as the home directory's project file (and
    written to by a project-scope save).
    """
    home = isolated_homes / "home"
    monkeypatch.setenv("USERPROFILE", str(home))  # Path.home() on Windows
    stale = home / ".kiss" / "mcp.json"
    stale.parent.mkdir(parents=True)
    leftover = json.dumps({"mcpServers": {"notes": {"url": "https://stale.example/mcp"}}})
    stale.write_text(leftover, encoding="utf-8")
    live = isolated_homes / ".kisshome" / "mcp.json"
    notes = MCPServerConfig(name="notes", transport="http", url="https://live.example/mcp")
    assert save_mcp_server(notes, "user", str(home)) == live
    assert load_mcp_servers(str(home))["notes"].url == "https://live.example/mcp"
    other = MCPServerConfig(name="other", transport="http", url="https://other.example/mcp")
    assert save_mcp_server(other, "project", str(home)) == live
    assert remove_mcp_server("notes", str(home)) == [live]
    assert set(load_mcp_servers(str(home))) == {"other"}
    assert stale.read_text(encoding="utf-8") == leftover
    # The user file is read once, as the user's: its stdio server is not a
    # project's (no working directory, the same connection as from any
    # other task), and the home's Claude Code file still wins over it.
    save_mcp_server(MCPServerConfig(name="local", command="serve"), "user", str(home))
    claude = json.dumps({"mcpServers": {"other": {"url": "https://claude.example/mcp"}}})
    (home / ".mcp.json").write_text(claude, encoding="utf-8")
    servers = load_mcp_servers(str(home))
    assert servers["local"] == load_mcp_servers(str(isolated_homes / "project"))["local"]
    assert servers["local"].cwd == ""
    assert servers["other"].url == "https://claude.example/mcp"


def test_failed_config_write_leaves_no_temp_file(isolated_homes: Path) -> None:
    """A config write that cannot complete removes its ``.mcp.json-<hex>`` staging file.

    The config path is a directory, so the atomic ``os.replace`` of the
    staging file onto it fails; the error propagates and the staging
    file does not linger next to it.
    """
    project = isolated_homes / "project"
    config_dir = isolated_homes / ".kisshome" / "mcp.json"
    config_dir.mkdir(parents=True)
    with pytest.raises(OSError):
        save_mcp_server(MCPServerConfig(name="s", command="a"), "user", str(project))
    assert config_dir.is_dir()
    assert [p.name for p in config_dir.parent.glob(".mcp.json*") if p.is_file()] == []


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



def test_permission_wildcards_last_rule_wins() -> None:
    """Wildcard rules cover MCP tools with last-match-wins semantics."""
    rules = {"*": "allow", "mymcp_*": "deny"}
    assert skill_permission("mymcp_search", rules) == "deny"
    assert skill_permission("other_search", rules) == "allow"
    rules = {"mymcp_*": "deny", "mymcp_safe": "allow"}
    assert skill_permission("mymcp_safe", rules) == "allow"
    assert skill_permission("mymcp_rm", rules) == "deny"
    assert skill_permission("anything", {}) == "allow"


def test_load_mcp_permissions_from_config(mcp_permission_rules) -> None:
    """``mcp_permissions`` is read from the kiss ``config.json``."""
    mcp_permission_rules({"*": "allow", "internal_*": "DENY "})
    rules = load_permission_rules("mcp_permissions")
    assert rules == {"*": "allow", "internal_*": "deny"}
    assert skill_permission("internal_x", rules) == "deny"



def test_make_mcp_tools_live_call(
    isolated_homes: Path, tmp_path: Path,
) -> None:
    """Tools of a real stdio server are wrapped and callable."""
    project = isolated_homes / "project"
    save_mcp_server(_stdio_config(tmp_path), "user", str(project))
    tools = make_mcp_tools(str(project))
    names = {t.__name__ for t in tools}
    assert names == {"testsrv_add", "testsrv_read_file", "testsrv_secret_word"}

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

    # A file handed over as an embedded resource is text the model must get
    # (it read "[resource content omitted]"); a link names what it points at.
    read_file = next(t for t in tools if t.__name__ == "testsrv_read_file")
    assert read_file(path="a.txt") == (
        "downloaded a.txt\nBODY-OF-a.txt\n[resource_link content omitted: repo://a.txt/history]"
    )


def test_make_mcp_tools_permission_filter(
    isolated_homes: Path, tmp_path: Path, mcp_permission_rules,
) -> None:
    """Denied MCP tools are never registered."""
    mcp_permission_rules({"*": "allow", "testsrv_secret*": "deny"})
    project = isolated_homes / "project"
    save_mcp_server(_stdio_config(tmp_path), "user", str(project))
    names = {t.__name__ for t in make_mcp_tools(str(project))}
    assert names == {"testsrv_add", "testsrv_read_file"}


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


_WHERE_SCRIPT = '''
import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("where")


@mcp.tool()
def cwd() -> str:
    """Return the server's working directory."""
    return os.getcwd()


if __name__ == "__main__":
    mcp.run()
'''


def test_a_project_stdio_server_runs_in_its_project(isolated_homes: Path) -> None:
    """A stdio server of a project's config file runs in that project.

    Two projects check in the same ``.mcp.json`` naming a script by a
    relative path, as Claude Code runs it.  The children inherited the
    daemon's working directory, where the script is not found, and the
    two identical entries shared one connection key, so with an
    absolute path the second project was handed the first one's
    server.  A user-level server stays shared by every project.
    """
    entry = {"command": sys.executable, "args": ["where.py"]}
    tools = {}
    for name in ("one", "two"):
        project = isolated_homes / name
        project.mkdir()
        (project / "where.py").write_text(_WHERE_SCRIPT, encoding="utf-8")
        (project / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"where": entry}}), encoding="utf-8",
        )
        (tools[name],) = make_mcp_tools(str(project))
        # What a save writes back is the entry as checked in, not this checkout's path.
        assert load_mcp_servers(str(project))["where"].to_json() == {"type": "stdio", **entry}
    for name, tool in tools.items():
        assert os.path.samefile(tool(), isolated_homes / name)

    shared = MCPServerConfig(name="shared", command=sys.executable, args=("where.py",))
    save_mcp_server(shared, "user", str(isolated_homes / "one"))
    assert {
        load_mcp_servers(str(isolated_homes / name))["shared"] for name in ("one", "two")
    } == {shared}


def test_a_sign_in_that_cannot_be_set_up_costs_no_other_server(
    isolated_homes: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remote server whose OAuth provider cannot be built fails alone.

    The SDK's provider refuses a ``$KISS_MCP_CLIENT_METADATA_URL`` that is
    not HTTPS as it is built; that ``ValueError`` escaped ``connect_all``
    and cost the run every MCP tool, the local servers' included.
    """
    monkeypatch.setenv("KISS_MCP_CLIENT_METADATA_URL", "http://plain.example/client.json")
    project = isolated_homes / "project"
    save_mcp_server(_stdio_config(tmp_path), "user", str(project))
    remote = MCPServerConfig(name="remote", transport="http", url="https://127.0.0.1:9/mcp")
    save_mcp_server(remote, "user", str(project))
    assert len(make_mcp_tools(str(project))) == 3
    assert "client_metadata_url" in MCPManager.instance().connect(remote).error


def test_call_tool_unknown_server_errors() -> None:
    """Calling a tool on a never-connected server returns an error string."""
    out = MCPManager.instance().call_tool("no-such-server", "t", {})
    assert out.startswith("Error: MCP server 'no-such-server' is not connected")


def test_result_text_flattening() -> None:
    """``CallToolResult`` content is flattened; errors are prefixed."""
    from mcp.types import CallToolResult, ImageContent, TextContent

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
    image = CallToolResult(content=[ImageContent(type="image", data="aGk=", mimeType="image/png")])
    assert _result_text(image) == "[image content omitted]"


def test_wrapper_docstring_carries_schema_descriptions() -> None:
    """The wrapper's docstring is the tool's description; the declared schema carries the rest."""
    from mcp.types import Tool

    from kiss.core.tool_schema import declared_schema

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
    wrapper = make_mcp_tool_wrapper(MCPManager.instance(), "my srv", tool, "my srv")
    assert wrapper.__name__ == "my_srv_search"
    assert wrapper.__doc__ == "Search the index. Second line."
    declared = declared_schema(wrapper)
    assert declared is not None
    assert declared["description"] == "Search the index. Second line."
    assert declared["parameters"]["properties"]["query"]["description"] == "The query text"
    assert "description" not in declared["parameters"]["properties"]["limit"]
    params = inspect.signature(wrapper).parameters
    assert params["limit"].annotation is int
    assert params["limit"].default is None


def test_wrapper_declares_the_structured_input_schema() -> None:
    """The model sees the tool's arrays, objects, enums and null unions, not strings.

    The wrapper reduced every property to a Python type and let the
    model converter regenerate its schema: arrays and objects became
    ``{"type": "string"}``, enums, nested properties and ``null``
    unions vanished, so models sent strings where the server required
    structured input.  The declared schema keeps that structure under
    the sanitized parameter names, and leaves out the keywords the
    providers' tool declarations reject (``$schema``,
    ``additionalProperties``, ``title``).  A ``oneOf`` union is
    declared as ``anyOf``: Gemini's declaration schema knows no
    ``oneOf`` and refused the whole tool before any request, where the
    string the property used to fall back to went through.
    """
    from google.genai import types
    from mcp.types import Tool

    from kiss.agents.sorcar.decide_tool import DEFAULT_DECISIONS_MODEL
    from kiss.core.models.model_info import model

    tool = Tool(
        name="file",
        description="File findings.",
        inputSchema={
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}},
                        "required": ["id"],
                        "additionalProperties": False,
                    },
                },
                "severity": {"type": "string", "enum": ["low", "high"], "title": "Severity"},
                "max-age": {"type": ["integer", "null"], "description": "Days"},
                "query": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
            },
            "required": ["findings", "query"],
        },
    )
    wrapper = make_mcp_tool_wrapper(MCPManager.instance(), "srv", tool, "srv")
    schema = model(DEFAULT_DECISIONS_MODEL)._function_to_openai_tool(wrapper)["function"]
    assert schema["description"] == "File findings."
    assert schema["parameters"] == {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                    "required": ["id"],
                },
            },
            "severity": {"type": "string", "enum": ["low", "high"]},
            "max_age": {"anyOf": [{"type": "integer"}, {"type": "null"}], "description": "Days"},
            "query": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
        },
        "required": ["findings", "query"],
    }
    assert inspect.signature(wrapper).parameters["max_age"].annotation is int
    # The declaration Gemini builds from it, as ``generate_and_process_with_tools`` does.
    declaration = types.FunctionDeclaration(
        name=schema["name"], description=schema["description"], parameters=schema["parameters"],
    )
    assert declaration.parameters is not None
    assert declaration.parameters.properties is not None
    query_members = declaration.parameters.properties["query"].any_of
    assert query_members is not None
    assert [m.type for m in query_members] == ["INTEGER", "STRING"]


def test_wrapper_resolves_local_schema_references() -> None:
    """A ``$ref`` property declares the referenced structure, not a string.

    Pydantic (so FastMCP) emits a nested model as ``{"$ref":
    "#/$defs/Address"}`` with the model under ``$defs``.  The reducer
    dropped the reference and, left with no type, declared a string:
    the model was told to send a string where the server required an
    object.  A self-referencing model stops at the reference's type.  A
    reference to a reference (an alias) is followed to the end, the
    keywords next to each hop winning over the next (audit 2026-11-08,
    S5 F4: the alias was dereferenced once, leaving its own reference
    unresolved, and the object was declared a string); a cycle of
    aliases ends at the type of the first alias seen again.
    """
    from mcp.types import Tool

    from kiss.core.tool_schema import declared_schema

    tool = Tool(
        name="ship",
        description="Ship an order.",
        inputSchema={
            "$defs": {
                "Address": {
                    "type": "object",
                    "title": "Address",
                    "properties": {
                        "street": {"type": "string"},
                        "zip_code": {"type": "string", "description": "Postal code"},
                    },
                    "required": ["street", "zip_code"],
                },
                "Node": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
                    },
                    "required": ["label"],
                },
                "ShippingAddress": {"$ref": "#/$defs/Address", "description": "Alias"},
                "Ping": {"$ref": "#/$defs/Pong", "type": "integer"},
                "Pong": {"$ref": "#/$defs/Ping"},
            },
            "type": "object",
            "properties": {
                "address": {"$ref": "#/$defs/Address", "description": "Where to ship"},
                "tree": {"$ref": "#/$defs/Node"},
                "lost": {"$ref": "#/$defs/Missing"},
                "billing": {"$ref": "#/$defs/ShippingAddress"},
                "origin": {"$ref": "#/$defs/ShippingAddress", "description": "Sender"},
                "loop": {"$ref": "#/$defs/Pong"},
            },
            "required": ["address"],
        },
    )
    wrapper = make_mcp_tool_wrapper(MCPManager.instance(), "srv", tool, "srv")
    declared = declared_schema(wrapper)
    assert declared is not None
    assert declared["parameters"]["properties"] == {
        "address": {
            "type": "object",
            "description": "Where to ship",
            "properties": {
                "street": {"type": "string"},
                "zip_code": {"type": "string", "description": "Postal code"},
            },
            "required": ["street", "zip_code"],
        },
        "tree": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "children": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["label"],
        },
        "lost": {"type": "string"},
        "billing": {
            "type": "object",
            "description": "Alias",
            "properties": {
                "street": {"type": "string"},
                "zip_code": {"type": "string", "description": "Postal code"},
            },
            "required": ["street", "zip_code"],
        },
        "origin": {
            "type": "object",
            "description": "Sender",
            "properties": {
                "street": {"type": "string"},
                "zip_code": {"type": "string", "description": "Postal code"},
            },
            "required": ["street", "zip_code"],
        },
        "loop": {"type": "integer"},
    }
    assert inspect.signature(wrapper).parameters["address"].annotation is dict
    assert inspect.signature(wrapper).parameters["billing"].annotation is dict


def test_wrapper_merges_allof_and_resolves_the_root_schema() -> None:
    """Arguments behind an ``allOf`` or behind a reference of the schema's root are declared.

    A node is everything its ``allOf`` members are (pydantic wraps a
    described model field in a one-member ``allOf``): the reducer
    dropped the keyword and declared the object a string.  The root was
    read for ``properties`` and ``required`` as written, so a tool whose
    arguments its root names by reference or composition was declared
    with none, and its required arguments could not be sent.
    """
    from mcp.types import Tool

    from kiss.core.tool_schema import declared_schema

    defs = {
        "Address": {
            "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"],
        },
        "Args": {
            "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
        },
    }
    extra = {
        "properties": {"weight": {"type": "number"}, "city": {"type": "integer"}},
        "required": ["weight", "city"],
    }
    ship = Tool(name="ship", inputSchema={
        "$defs": defs,
        "type": "object",
        "properties": {
            "address": {"allOf": [{"$ref": "#/$defs/Address"}], "description": "Where to"},
            "parcel": {"allOf": [{"$ref": "#/$defs/Address"}, extra, "not a schema"]},
        },
        "required": ["address"],
    })
    declared = declared_schema(make_mcp_tool_wrapper(MCPManager.instance(), "srv", ship, "srv"))
    assert declared is not None
    address = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    assert declared["parameters"]["properties"] == {
        "address": {**address, "description": "Where to"},
        # Joined, the first to name a property declaring it.
        "parcel": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "weight": {"type": "number"}},
            "required": ["city", "weight"],
        },
    }
    assert list(declared["parameters"]["properties"]["parcel"]["properties"]) == ["city", "weight"]
    limit = {"limit": {"type": "integer"}}
    for root, signature in (
        ({"$defs": defs, "type": "object", "$ref": "#/$defs/Args"}, "(query: str) -> str"),
        ({"$defs": defs, "allOf": [{"$ref": "#/$defs/Args"}]}, "(query: str) -> str"),
        (
            {"$defs": defs, "properties": limit, "allOf": [{"$ref": "#/$defs/Args"}]},
            "(query: str, limit: int = None) -> str",
        ),
    ):
        wrapper = make_mcp_tool_wrapper(
            MCPManager.instance(), "srv", Tool(name="find", inputSchema=root), "srv",
        )
        assert str(inspect.signature(wrapper)) == signature, root
        declared = declared_schema(wrapper)
        assert declared is not None
        assert declared["parameters"]["required"] == ["query"], root
        assert declared["parameters"]["properties"]["query"] == {"type": "string"}, root
    # A reference followed for one part is on the path of that part's
    # children only: the same model named again beside it (an argument's
    # second allOf member, a second argument of a composed root) is
    # declared in full, not as a model that refers to itself.
    billing = {"properties": {"billing": {"$ref": "#/$defs/Address"}}}
    for schema in (
        {"$defs": defs, "properties": {"order": {"allOf": [{"$ref": "#/$defs/Address"}, billing]}}},
        {"$defs": defs, "properties": {"order": billing}, "allOf": [{"$ref": "#/$defs/Address"}]},
    ):
        declared = declared_schema(make_mcp_tool_wrapper(
            MCPManager.instance(), "srv", Tool(name="bill", inputSchema=schema), "srv",
        ))
        assert declared is not None
        assert declared["parameters"]["properties"]["order"]["properties"]["billing"] == address
    # Keywords beside a reference apply with its target's, as the server's
    # validator has them (it rejects an order without its city): joined too.
    beside = {"$defs": defs, "properties": {"order": {"$ref": "#/$defs/Address", **billing}}}
    declared = declared_schema(make_mcp_tool_wrapper(
        MCPManager.instance(), "srv", Tool(name="bill", inputSchema=beside), "srv",
    ))
    assert declared is not None
    assert declared["parameters"]["properties"]["order"] == {
        "type": "object",
        "properties": {"billing": address, "city": {"type": "string"}},
        "required": ["city"],
    }


def test_a_large_declaration_stops_following_references() -> None:
    """Definitions shared down a chain of models do not multiply without bound.

    Each of 18 models names the next one twice: declared in full, the
    last one is there 2**18 times, 17 MB from a 2 KB schema (and a hang
    a few models further).  Once a declaration has
    ``_MAX_DECLARED_NODES`` nodes, a reference declares its target's
    type only.
    """
    from mcp.types import Tool

    from kiss.core.tool_schema import declared_schema

    depth = 18
    twice = [{"a": {"$ref": f"#/$defs/D{i + 1}"}, "b": {"$ref": f"#/$defs/D{i + 1}"}}
             for i in range(depth)]
    defs = {f"D{i}": {"type": "object", "properties": twice[i]} for i in range(depth)}
    tool = Tool(name="deep", inputSchema={
        "$defs": {**defs, f"D{depth}": {"type": "string"}},
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/D0"}},
    })
    declared = declared_schema(make_mcp_tool_wrapper(MCPManager.instance(), "srv", tool, "srv"))
    assert declared is not None
    assert json.dumps(declared).count('"type"') <= _MAX_DECLARED_NODES
    node = declared["parameters"]["properties"]["x"]
    for _ in range(depth):  # the first path is declared to its end
        node = node["properties"]["a"]
    assert node == {"type": "string"}
    # A chain of aliases is followed without the interpreter's stack: one
    # longer than that is deep still ends at its target.
    links = sys.getrecursionlimit() + 100
    aliases = {f"A{i}": {"$ref": f"#/$defs/A{i + 1}"} for i in range(links)}
    chained = Tool(name="chained", inputSchema={
        "$defs": {**aliases, f"A{links}": {"type": "integer"}},
        "properties": {"n": {"$ref": "#/$defs/A0"}},
    })
    declared = declared_schema(make_mcp_tool_wrapper(MCPManager.instance(), "srv", chained, "srv"))
    assert declared is not None
    assert declared["parameters"]["properties"] == {"n": {"type": "integer"}}



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

    # Sanitization is lossy for "my/server", so an injective digest
    # suffix keeps it from sharing a file with e.g. "my server".
    assert storage.path.name.startswith("my_server.")
    assert storage.path.name.endswith(".json")
    assert storage.path.name != "my_server.json"
    assert FileTokenStorage("my server").path != storage.path
    if not IS_WINDOWS:  # Windows has no POSIX mode bits to tighten
        assert (storage.path.stat().st_mode & 0o777) == 0o600
    back = asyncio.run(storage.get_tokens())
    assert back is not None and back.access_token == "at-1"
    back_info = asyncio.run(storage.get_client_info())
    assert back_info is not None and back_info.client_id == "cid"

    assert storage.clear() is True
    assert storage.clear() is False
    assert asyncio.run(storage.get_tokens()) is None


def test_noninteractive_oauth_refuses_browser_flow() -> None:
    """Agent runs never block on OAuth; they direct to the sign-in tool and end the connection."""
    remote = MCPServerConfig(name="remote", transport="http", url="https://mcp.example/mcp")
    conn = _Connection(config=remote)
    with pytest.raises(RuntimeError, match="requires an OAuth sign-in.*connect_mcp_server"):
        asyncio.run(_refuse_login(conn, "https://auth.example/authorize"))
    assert conn.stop.is_set() and "requires an OAuth sign-in" in conn.error



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
