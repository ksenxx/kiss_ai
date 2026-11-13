# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""MCP servers the user defers sit behind ``mcp_tool_search`` and ``mcp_tool_call``.

Every MCP tool definition an agent loads is part of every model
request.  On request (a budget in ``config.json``, or ``alwaysLoad`` on
a server) ``make_mcp_tools`` loads only some servers upfront and stands
the two tools above in for the rest: the model sees the deferred tools'
names, loads the definitions it needs and runs a loaded tool by name.

Driven end to end: real stdio MCP servers (a low-level server replaying
a catalog and echoing each call), the real ``make_mcp_tools`` wiring,
the schema builder the models use, and one whole ``SorcarAgent`` run
against a scripted OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.decide_tool import DEFAULT_DECISIONS_MODEL
from kiss.agents.sorcar.mcp_servers import (
    MCPManager,
    MCPServerConfig,
    make_mcp_tools,
    save_mcp_server,
)
from kiss.core.models.model_info import model

_REPLAY_SERVER = '''
import json
import sys

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

catalog_path, name, log_path = sys.argv[1:4]
with open(catalog_path, encoding="utf-8") as handle:
    CATALOG = json.load(handle)
server = Server(name)


def log(line):
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(line + "\\n")


@server.list_tools()
async def list_tools():
    return [types.Tool(**tool) for tool in CATALOG]


@server.call_tool()
async def call_tool(tool, arguments):
    log("call " + tool)
    echo = {"server": name, "tool": tool, "arguments": arguments}
    return [types.TextContent(type="text", text=json.dumps(echo, sort_keys=True))]


async def main():
    log("start")
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


anyio.run(main)
'''

_TEXT = {"type": "string"}
_NUMBER = {"type": "integer"}


def _tool(name: str, description: str, required: tuple[str, ...] = (), **properties: Any) -> dict:
    """One catalog entry as an MCP server lists it."""
    schema = {"type": "object", "properties": properties, "required": list(required)}
    return {"name": name, "description": description, "inputSchema": schema}


TRACKER = [
    _tool(
        "create_issue", "Open a new issue in a repository.", ("repo", "title"),
        repo={"type": "string", "description": "Repository as owner/name."}, title=_TEXT,
        labels={"type": "array", "items": _TEXT},
        assignee={"type": "object", "properties": {"login": _TEXT}},
    ),
    _tool("list_issues", "List the issues of a repository.", ("repo",), repo=_TEXT),
    _tool(
        "add_issue_comment", "Add a comment to an issue.", ("repo", "number", "body"),
        repo=_TEXT, number=_NUMBER, body=_TEXT,
    ),
    _tool(
        "mergePullRequest", "Merge a pull request into its base branch.", ("repo", "number"),
        repo=_TEXT, number=_NUMBER,
        strategy={"type": "string", "description": "merge, squash or rebase."},
    ),
    _tool("list-branches", "List the branches of a repository.", ("repo",), repo=_TEXT),
    _tool("whoami", "Return the signed-in user."),
]
TRACKER_NAMES = [
    "tracker_create_issue", "tracker_list_issues", "tracker_add_issue_comment",
    "tracker_mergePullRequest", "tracker_list-branches", "tracker_whoami",
]
CLOCK = [_tool("now", "Return the current time in a timezone.", ("timezone",), timezone=_TEXT)]
META = ["mcp_tool_search", "mcp_tool_call"]


def _bulk(count: int) -> list[dict]:
    """A catalog of *count* tools with 1,500-character descriptions (over 45,000 in all for 30)."""
    return [
        _tool(f"op_{i}", f"Operation {i}. " + "x" * 1500, ("target",), target=_TEXT)
        for i in range(count)
    ]


@pytest.fixture
def real_stdin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Give ``sys.stdin`` and the MCP errlog real file descriptors."""
    # Imported before sys.stderr is redirected: the module binds
    # ``errlog=sys.stderr`` as a default argument at import time.
    from mcp.client.stdio import stdio_client

    stdin_stream = open(os.devnull, encoding="utf-8")
    errlog = (tmp_path / "mcp_errlog.txt").open("w", encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stdin_stream)
    monkeypatch.setattr(sys, "stderr", errlog)
    wrapped = stdio_client.__wrapped__  # type: ignore[attr-defined]
    monkeypatch.setattr(wrapped, "__defaults__", (errlog,))
    try:
        yield
    finally:
        errlog.close()
        stdin_stream.close()


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, real_stdin: None) -> Iterator[Path]:
    """An empty project with every user-level location redirected into *tmp_path*."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / ".kisshome"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / ".kisshome").mkdir()
    (tmp_path / "project").mkdir()
    try:
        yield tmp_path / "project"
    finally:
        MCPManager.instance().shutdown()


def _add_server(project: Path, name: str, catalog: list[dict], **entry: Any) -> Path:
    """Add a stdio server replaying *catalog* to ``<project>/.mcp.json``; return its log."""
    home = project.parent
    script = home / "replay_server.py"
    script.write_text(_REPLAY_SERVER, encoding="utf-8")
    catalog_path = home / f"{name}.catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    log = home / f"{name}.log"
    config_path = project / ".mcp.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {"mcpServers": {}}
    config["mcpServers"][name] = {
        "command": sys.executable,
        "args": [str(script), str(catalog_path), name, str(log)],
        **entry,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return log


def _set_config(project: Path, **keys: Any) -> None:
    """Write ``$KISS_HOME/config.json``."""
    (project.parent / ".kisshome" / "config.json").write_text(json.dumps(keys), encoding="utf-8")


def _load(project: Path, taken: Iterable[str] = ()) -> dict[str, Any]:
    """Every tool ``make_mcp_tools`` gives the agent for *project*, by name and in order."""
    return {tool.__name__: tool for tool in make_mcp_tools(str(project), taken)}


def _sent(tools: dict[str, Any]) -> list[dict[str, Any]]:
    """The definitions a model request carries for *tools*, built as the models build them."""
    builder = model(DEFAULT_DECISIONS_MODEL)
    return [entry["function"] for entry in builder._build_openai_tools_schema(tools)]


def _listed(tools: dict[str, Any]) -> list[str]:
    """The names of the tools a model request lists for *tools*, in order."""
    return [sent["name"] for sent in _sent(tools)]


def _chars(tools: dict[str, Any]) -> int:
    """The documented size of definitions: name, description and argument schema as JSON."""
    return sum(
        len(sent["name"]) + len(sent["description"]) + len(json.dumps(sent["parameters"]))
        for sent in _sent(tools)
    )


def _calls(log: Path) -> list[str]:
    """The tool calls a replay server received."""
    return [line for line in log.read_text().splitlines() if line.startswith("call ")]


def _found(result: str) -> list[str]:
    """The names of the definitions a search result carries, in order."""
    return re.findall(r"^(\S+): .*\narguments: ", result, flags=re.MULTILINE)


def test_nothing_is_deferred_unless_asked_for(project: Path) -> None:
    """Without a budget every tool is a direct tool, whatever the size, and nothing is added."""
    _add_server(project, "bulk", _bulk(30))
    _add_server(project, "tracker", TRACKER)
    _add_server(project, "clock", CLOCK)
    everything = [*(f"bulk_op_{i}" for i in range(30)), *TRACKER_NAMES, "clock_now"]
    tools = _load(project)
    assert list(tools) == _listed(tools) == everything
    assert json.loads(tools["tracker_whoami"]())["tool"] == "whoami"
    # a value that is no non-negative integer is no budget
    for junk in ("40", -1, True, None, 2.5):
        _set_config(project, mcp_upfront_tool_chars=junk)
        assert _listed(_load(project)) == everything, junk


def test_a_large_server_is_deferred_and_reached_in_two_steps(project: Path) -> None:
    """Over the budget the user set, the large server's definitions leave the request.

    Its tools stay reachable: the names are listed, a search returns a
    definition, and the call tool runs the tool on its server with the
    arguments intact.  The toolset the model was given does not change.
    """
    log = _add_server(project, "bulk", _bulk(30))
    _add_server(project, "clock", CLOCK)
    bulk_names = [f"bulk_op_{i}" for i in range(30)]
    everything = _load(project)
    assert _listed(everything) == [*bulk_names, "clock_now"]

    _set_config(project, mcp_upfront_tool_chars=40_000)
    tools = _load(project)
    assert _listed(tools) == ["clock_now", *META]
    # the deferred tools are registered under their names, for a model that calls one so
    assert list(tools) == ["clock_now", *META, *bulk_names]
    before = json.dumps(_sent(tools))
    assert len(before) < 0.1 * len(json.dumps(_sent(everything)))
    assert "Operation" not in before
    search, call = _sent(tools)[1:]
    assert "bulk (30): bulk_op_0, bulk_op_1, " in search["description"]
    assert search["parameters"]["required"] == ["query"]
    assert call["parameters"]["properties"]["arguments"]["type"] == "object"

    found = tools["mcp_tool_search"](query="bulk_op_7")
    assert found.startswith("bulk_op_7: Operation 7. xxx")
    assert '"required": ["target"]' in found and "mcp_tool_call(name, arguments)" in found
    assert _found(found) == ["bulk_op_7"]
    ran = tools["mcp_tool_call"](name="bulk_op_7", arguments={"target": "a b"})
    assert json.loads(ran) == {"server": "bulk", "tool": "op_7", "arguments": {"target": "a b"}}
    assert _calls(log) == ["call op_7"]
    assert json.dumps(_sent(tools)) == before


def test_a_call_before_the_definition_returns_the_definition_instead(project: Path) -> None:
    """A tool the model has only seen the name of is not run on a guess."""
    log = _add_server(project, "tracker", TRACKER, alwaysLoad=False)
    tools = _load(project)
    call = tools["mcp_tool_call"]
    guess = {"repo": "octo/demo", "summary": "Flaky test"}
    refused = call(name="tracker_create_issue", arguments=guess)
    assert refused.startswith("Not run: the definition of tracker_create_issue was not loaded.")
    assert "tracker_create_issue: Open a new issue in a repository." in refused
    assert '"required": ["repo", "title"]' in refused
    assert _calls(log) == []

    arguments = {
        "repo": "octo/demo", "title": "Flaky test", "labels": ["bug", "flaky"],
        "assignee": {"login": "octocat"},
    }
    ran = json.loads(call(name="tracker_create_issue", arguments=arguments))
    assert ran == {"server": "tracker", "tool": "create_issue", "arguments": arguments}
    # the server's own validation still answers a call that leaves out a required argument
    incomplete = call(name="tracker_create_issue", arguments={"repo": "octo/demo"})
    assert incomplete.startswith("Error: ") and "title" in incomplete
    assert _calls(log) == ["call create_issue"]

    # a model that calls a deferred tool under its own name is answered the same way
    direct = tools["tracker_list_issues"]
    refused = direct(repo="o/d")
    assert refused.startswith("Not run: the definition of tracker_list_issues was not loaded.")
    assert json.loads(direct(repo="o/d"))["arguments"] == {"repo": "o/d"}
    assert json.loads(call(name="tracker_list_issues", arguments={"repo": "x/y"}))["tool"] == (
        "list_issues"
    )
    assert _calls(log) == ["call create_issue", "call list_issues", "call list_issues"]


def test_call_arguments_and_unknown_names(project: Path) -> None:
    """Arguments are an object or its JSON text; anything else, or a wrong name, is explained."""
    log = _add_server(project, "tracker", TRACKER, alwaysLoad=False)
    tools = _load(project)
    call = tools["mcp_tool_call"]
    tools["mcp_tool_search"](query="tracker_whoami tracker_list_issues")
    assert json.loads(call(name="tracker_whoami"))["arguments"] == {}
    listed = json.loads(call(name="tracker_list_issues", arguments='{"repo": "o/d"}'))
    assert listed["arguments"] == {"repo": "o/d"}
    not_json = call(name="tracker_list_issues", arguments="{repo")
    assert not_json.startswith("Error: arguments is not valid JSON (")
    not_object = "Error: arguments must be a JSON object, got list."
    assert call(name="tracker_list_issues", arguments=["o/d"]) == not_object
    assert call(name="tracker_list_issues", arguments="[1]") == not_object
    unknown = call(name="tracker_merge", arguments={})
    assert unknown.startswith(
        "Error: 'tracker_merge' is not a deferred MCP tool. Closest: tracker_mergePullRequest, "
    )
    assert "mcp_tool_search" in unknown
    # a tool of the run that is not deferred, such as the agent's finish: it is called directly
    assert call(name="finish") == (
        "Error: 'finish' is not a deferred MCP tool. A tool that is not deferred does not go "
        "through mcp_tool_call: call it directly, under its own name. The deferred tools are "
        "the ones listed in the description of mcp_tool_search."
    )
    assert _calls(log) == ["call whoami", "call list_issues"]


def test_keyword_search_ranks_names_over_descriptions_over_arguments(project: Path) -> None:
    """Keywords match names, descriptions and arguments; rarer words and names count more."""
    _add_server(project, "tracker", TRACKER, alwaysLoad=False)
    search = _load(project)["mcp_tool_search"]
    # camelCase and kebab-case names split into words; a plural finds the singular
    assert _found(search(query="merge pull requests"))[0] == "tracker_mergePullRequest"
    assert _found(search(query="branch")) == ["tracker_list-branches", "tracker_mergePullRequest"]
    # a word that occurs only in an argument's description, or only as an argument's name
    assert _found(search(query="squash")) == ["tracker_mergePullRequest"]
    assert _found(search(query="labels")) == ["tracker_create_issue"]
    # "comment" names one tool, "issue" three: the rarer word decides, the name breaks the tie
    assert _found(search(query="issue comment")) == [
        "tracker_add_issue_comment", "tracker_create_issue", "tracker_list_issues",
    ]

    capped = search(query="repository", limit=2)
    assert capped.startswith("Showing 2 of 3 matches")
    assert _found(capped) == ["tracker_create_issue", "tracker_list-branches"]
    assert _found(search(query="repository", limit="2")) == _found(capped)
    assert _found(search(query="repositories", limit=2)) == _found(capped)
    assert search(query="repository", limit=0).startswith("Showing 1 of 3 matches")
    assert search(query="repository", limit="many") == (
        "Error: limit must be a positive integer, got 'many'."
    )
    # nothing matches, as when a model looks up a tool it already has: it is told to call that
    assert search(query="finish") == (
        "No deferred MCP tool matches 'finish'. A tool that is not deferred needs no lookup: "
        "call it directly, under its own name. The deferred tools are the ones listed in the "
        "description of mcp_tool_search; to find one of those, try other keywords."
    )
    # exact names return exactly those tools, once each, whatever the limit
    named = search(query="tracker_whoami, tracker_list_issues tracker_whoami", limit=1)
    assert _found(named) == ["tracker_whoami", "tracker_list_issues"]
    assert "Showing" not in named


def test_the_budget_fills_smallest_server_first(project: Path) -> None:
    """Unmarked servers load upfront from the smallest up while they fit the configured budget."""
    _add_server(project, "bulk", _bulk(30))
    _add_server(project, "tracker", TRACKER)
    _add_server(project, "clock", CLOCK)
    everything = _load(project)
    small = _chars({name: tool for name, tool in everything.items() if name[:5] != "bulk_"})

    _set_config(project, mcp_upfront_tool_chars=small)
    assert _listed(_load(project)) == [*TRACKER_NAMES, "clock_now", *META]
    _set_config(project, mcp_upfront_tool_chars=small - 1)
    tools = _load(project)
    assert _listed(tools) == ["clock_now", *META]
    index = _sent(tools)[1]["description"]
    assert index.index("bulk (30): ") < index.index("tracker (6): tracker_create_issue, ")
    _set_config(project, mcp_upfront_tool_chars=0)
    assert _listed(_load(project)) == META

    # of two servers of one size the one configured first is the one that fits
    _add_server(project, "dusk", CLOCK)
    _add_server(project, "dawn", CLOCK)
    _set_config(project)
    everything = _load(project)
    one = _chars({"dusk_now": everything["dusk_now"]})
    assert one == _chars({"dawn_now": everything["dawn_now"]})
    _set_config(project, mcp_upfront_tool_chars=2 * one - 1)
    tools = _load(project)
    assert _listed(tools) == ["dusk_now", *META]
    assert _sent(tools)[1]["description"].endswith("\nclock (1): clock_now\ndawn (1): dawn_now")


def test_always_load_overrides_the_budget_either_way(project: Path) -> None:
    """``alwaysLoad`` false defers a server without any budget; true keeps one out of the budget."""
    bulk_names = [f"bulk_op_{i}" for i in range(30)]
    _add_server(project, "bulk", _bulk(30), alwaysLoad=True)
    _add_server(project, "tracker", TRACKER, alwaysLoad="yes")  # not a boolean: unmarked
    _add_server(project, "clock", CLOCK, alwaysLoad=False)
    _add_server(project, "empty", [], alwaysLoad=False)
    tools = _load(project)
    assert _listed(tools) == [*bulk_names, *TRACKER_NAMES, *META]
    assert _sent(tools)[-2]["description"].endswith("\nclock (1): clock_now")
    # a marked server is outside the budget: with none left, only the unmarked one moves
    _set_config(project, mcp_upfront_tool_chars=0)
    assert _listed(_load(project)) == [*bulk_names, *META]


def test_a_saved_always_load_survives_and_does_not_restart_the_server(project: Path) -> None:
    """The setting is written with the entry and is no part of the connection's identity."""
    log = _add_server(project, "clock", CLOCK)
    entry = json.loads((project / ".mcp.json").read_text())["mcpServers"]["clock"]
    (project / ".mcp.json").unlink()
    config = MCPServerConfig(name="clock", command=entry["command"], args=tuple(entry["args"]))
    saved = save_mcp_server(replace(config, always_load=False), "project", str(project))
    assert json.loads(saved.read_text())["mcpServers"]["clock"]["alwaysLoad"] is False
    assert _listed(_load(project)) == META
    save_mcp_server(replace(config, always_load=True), "project", str(project))
    assert _listed(_load(project)) == ["clock_now"]
    assert log.read_text().splitlines() == ["start"]


def test_denied_tools_are_not_listed_and_clashing_names_get_a_suffix(project: Path) -> None:
    """Permissions apply before the split; the two tools are named like any other MCP tool."""
    catalog = [_tool("tool_call", "Place a call."), _tool("secret", "Hidden.")]
    _add_server(project, "mcp", catalog, alwaysLoad=False)
    _set_config(project, mcp_permissions={"mcp_secret": "deny"})
    tools = _load(project, taken={"mcp_tool_search"})
    assert _listed(tools) == ["mcp_tool_search_2", "mcp_tool_call_2"]
    assert list(tools) == ["mcp_tool_search_2", "mcp_tool_call_2", "mcp_tool_call"]
    search, call = _sent(tools)
    assert search["description"].endswith("\nmcp (1): mcp_tool_call")
    assert "mcp_tool_call_2" in search["description"]
    assert "mcp_tool_search_2" in call["description"]
    assert "is not a deferred MCP tool" in tools["mcp_tool_call_2"](name="mcp_secret")
    assert "mcp_tool_call_2(name, arguments)" in tools["mcp_tool_search_2"](query="mcp_tool_call")
    ran = json.loads(tools["mcp_tool_call_2"](name="mcp_tool_call"))
    assert (ran["server"], ran["tool"]) == ("mcp", "tool_call")


def test_an_agent_run_loads_and_calls_a_deferred_tool_with_a_constant_toolset(
    project: Path,
) -> None:
    """A whole run: the definition arrives as a tool result and the request's tools never change."""
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent
    from kiss.tests.core.test_default_max_steps import _all_text, _ScriptedServer

    _add_server(project, "tracker", TRACKER, alwaysLoad=False)
    arguments = {"repo": "octo/demo", "title": "Flaky test"}
    server = _ScriptedServer([
        ("mcp_tool_search", {"query": "tracker_create_issue"}),
        ("mcp_tool_call", {"name": "tracker_create_issue", "arguments": arguments}),
        # a deferred tool called under its own name, as some models do
        ("tracker_whoami", {}),
        ("finish", {"success": True, "is_continue": False, "summary_in_html": "<p>done</p>"}),
    ])
    try:
        SorcarAgent("test-deferred-mcp").run(
            model_name="gpt-4o-mini",
            prompt_template="Open an issue titled Flaky test in octo/demo, then finish.",
            work_dir=str(project),
            model_config=server.model_config(),
            web_tools=False,
            verbose=False,
        )
    finally:
        server.stop()
    first, second, third, fourth = server.request_bodies[:4]
    names = [tool["function"]["name"] for tool in first["tools"]]
    assert set(META) <= set(names) and not set(TRACKER_NAMES) & set(names)
    assert first["tools"] == second["tools"] == third["tools"] == fourth["tools"]
    assert "Open a new issue in a repository." not in _all_text(first)
    definition = "tracker_create_issue: Open a new issue in a repository."
    assert definition in _all_text(second["messages"])
    assert '\\"tool\\": \\"create_issue\\"' in json.dumps(third["messages"])
    assert "Not run: the definition of tracker_whoami was not loaded." in _all_text(
        fourth["messages"]
    )


def _two_calls(turn: int, first: tuple[str, dict], second: tuple[str, dict]) -> bytes:
    """One scripted model response with two tool calls, whose ids no other turn uses."""
    from kiss.tests.agents.sorcar.local_model_server import tool_calls_body

    body = tool_calls_body([first, second], 500)
    return body.replace(b'"call_2"', f'"call_{turn}b"'.encode())


def _results(request: dict[str, Any]) -> list[str]:
    """The results of the latest response's tool calls, as *request* gives them to the model."""
    results: list[str] = []
    for message in reversed(request["messages"]):
        if message["role"] != "tool":
            break
        results.insert(0, message["content"])
    return results


def test_a_call_runs_only_on_a_definition_its_conversation_already_had(project: Path) -> None:
    """A call runs when the model wrote it with the definition in its conversation.

    The calls of one model response were all written before any of
    their results, so a definition that a search or a refused call of
    that response returns is no ground for running another call of it;
    from the next response on it is.  A continuation session starts a
    new conversation, which has none of the earlier definitions.
    """
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent
    from kiss.tests.agents.sorcar.local_model_server import (
        MODEL,
        finish_body,
        serve,
        tool_call_body,
    )

    log = _add_server(project, "tracker", TRACKER, alwaysLoad=False)
    issue = ("mcp_tool_call", {
        "name": "tracker_create_issue", "arguments": {"repo": "o/d", "title": "Flaky"},
    })
    listing = ("mcp_tool_call", {"name": "tracker_list_issues", "arguments": {"repo": "o/d"}})
    script = [
        _two_calls(1, ("mcp_tool_search", {"query": "tracker_create_issue"}), issue),
        _two_calls(2, listing, listing),
        _two_calls(3, issue, listing),
        tool_call_body(
            "finish", {"success": False, "is_continue": True, "summary_in_html": "<p>half</p>"},
            500,
        ),
        tool_call_body(*issue, 500),
        tool_call_body(*issue, 500),
        finish_body("<p>done</p>"),
    ]
    with serve(script) as (url, requests):
        SorcarAgent("test-deferred-one-response").run(
            model_name=MODEL,
            prompt_template="Open an issue in o/d and list its issues, then finish.",
            work_dir=str(project),
            model_config={"base_url": url, "api_key": "local"},
            web_tools=False,
            verbose=False,
        )
    searched, called = _results(requests[1])
    assert searched.startswith("tracker_create_issue: Open a new issue in a repository.")
    assert called.startswith(
        "Not run: the definition of tracker_create_issue is in another result of this response"
    )
    first, second = _results(requests[2])
    assert first.startswith("Not run: the definition of tracker_list_issues was not loaded.")
    assert "tracker_list_issues: List the issues of a repository." in first
    assert second.startswith(
        "Not run: the definition of tracker_list_issues is in another result of this response"
    )
    assert "List the issues of a repository." not in second
    ran = [json.JSONDecoder().raw_decode(result)[0] for result in _results(requests[3])]
    assert [echo["tool"] for echo in ran] == ["create_issue", "list_issues"]
    # the second session: its first request carries no tool result yet
    assert _results(requests[4]) == [] and "<p>half</p>" in json.dumps(requests[4]["messages"])
    (again,) = _results(requests[5])
    assert again.startswith("Not run: the definition of tracker_create_issue was not loaded.")
    assert '"required": ["repo", "title"]' in again
    (rerun,) = _results(requests[6])
    assert json.JSONDecoder().raw_decode(rerun)[0]["tool"] == "create_issue"
    assert _calls(log) == ["call create_issue", "call list_issues", "call create_issue"]


def test_a_deferred_tool_called_by_name_in_cli_text_keeps_its_argument_types(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CLI model's call arrives as text; the deferred tool's own schema types its arguments."""
    from kiss.core.models.claude_code_model import ClaudeCodeModel
    from kiss.core.models.model import _build_text_based_tools_prompt
    from kiss.tests.conftest import install_cli_script
    from kiss.tests.core.models.test_cc_xml_invoke_tool_calls import _FAKE_CLAUDE, _delta_lines

    _add_server(project, "tracker", TRACKER, alwaysLoad=False)
    tools = _load(project)
    text = (
        '<invoke name="tracker_add_issue_comment">\n<parameter name="repo">o/d</parameter>\n'
        '<parameter name="number">7</parameter>\n<parameter name="body">123</parameter>\n</invoke>'
    )
    events = project.parent / "events.jsonl"
    events.write_text("\n".join(_delta_lines(text, 5)) + "\n", encoding="utf-8")
    bin_dir = project.parent / "bin"
    bin_dir.mkdir()
    install_cli_script(bin_dir / "claude", _FAKE_CLAUDE.format(events=str(events)))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    cli = ClaudeCodeModel("cc/opus", model_config={"timeout": 20})
    cli.initialize("Comment on issue 7 of o/d.")
    (call,), _, _ = cli.generate_and_process_with_tools(tools)
    assert call["arguments"] == {"repo": "o/d", "number": 7, "body": "123"}
    direct = tools[call["name"]]
    assert direct(**call["arguments"]).startswith("Not run: the definition of tracker_add_issue")
    assert json.loads(direct(**call["arguments"]))["arguments"] == call["arguments"]
    # the definition is still in no model request and not in the text catalog of tools
    assert _listed(tools) == META
    assert "Add a comment to an issue." not in _build_text_based_tools_prompt(tools)
