# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end: a path-mode ``run_agent`` sub-task inherits the caller's extra tools.

A task whose agent script adds tools through ``add_to_tools()`` runs
on a real daemon; its scripted model dispatches three ``run_agent``
sub-tasks.  Each sub-task's model request is a real chat-completions
call, so its ``tools`` array is exactly the toolset the sub-agent got:

* the plain sub-agent (``dummy_sea.py``) has the parent's tool, and a
  parent tool named like one of the sub-agent's built-ins is skipped;
* a sub-task whose own script also uses ``add_to_tools()`` has both
  sets, and a tool both scripts define by the same name once;
* a sub-task whose script fixes the whole toolset (``add_to_tools()``
  with ``settings()['tool_profile'] == "none"``) keeps exactly that set.

The tool callables cannot travel the wire: ``run_agent`` sends the
``inheritTools`` flag and the daemon takes them off the running parent
(``task_runner._parent_extra_tools``) and hands them to
``SorcarAgent.run(inherited_tools=...)``.

The same run checks that the parent's ``append_to_prompt`` text ends
every sub-task's prompt too: the daemon records the suffix it appended
on the parent agent (``SorcarAgent.run(prompt_suffix=...)``) and the
dispatch sends it as each sub-task's ``append_to_prompt``.
"""

from __future__ import annotations

import asyncio
import textwrap
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from kiss.server import sorcar, task_runner
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    request_text,
    tool_call_response,
)

PARENT_SCRIPT = textwrap.dedent('''
    def parent_ledger(entry: str) -> str:
        """Record *entry* in the parent's ledger."""
        return "recorded " + entry


    def number_of_cores() -> int:
        """The deployment's CPU quota (a name the built-in toolset also uses)."""
        return 2


    def add_to_tools():
        return [parent_ledger, number_of_cores]


    def settings():
        # Without run_parallel the parent has no built-in
        # number_of_cores, so its own tool of that name registers; a
        # sub-task WITH run_parallel has the built-in and must skip the
        # inherited one instead of failing on the duplicate name.
        return {"allow_fan_out": False}


    def add_to_system_prompt():
        return "PARENT-PROTOCOL: record every decision with parent_ledger."
''')

CHILD_ADDING_SCRIPT = textwrap.dedent('''
    def child_probe(what: str) -> str:
        """Probe *what*."""
        return "probed " + what


    def parent_ledger(entry: str) -> str:
        """The child's own copy of the parent's tool (same name)."""
        return "child recorded " + entry


    def add_to_tools():
        return [child_probe, parent_ledger]
''')

CHILD_FIXED_SCRIPT = textwrap.dedent('''
    def only_tool(what: str) -> str:
        """The one tool of this agent."""
        return what


    def settings():
        # ``none``: no built-in toolset, so add_to_tools() is the whole set.
        return {"tool_profile": "none"}


    def add_to_tools():
        return [only_tool]
''')


def _last_text(request: dict[str, Any]) -> str:
    """Return the text of the last message of a chat-completions *request*."""
    messages = request.get("messages", [])
    if not messages:
        return ""
    content = messages[-1].get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    return ""


def _tool_names(request: dict[str, Any]) -> list[str]:
    """Return the function names of the ``tools`` array of *request*."""
    return [
        tool["function"]["name"]
        for tool in request.get("tools", [])
        if isinstance(tool, dict) and "function" in tool
    ]


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Iterator[IsolatedKissHome]:
    """An isolated KISS home whose config disables worktrees and auto-commit."""
    home = IsolatedKissHome(prefix="kiss-inherit-tools-")
    home.write_config(is_worktree=False, auto_commit_mode=False, classify_tasks=False)
    try:
        yield home
    finally:
        home.cleanup()


@pytest.fixture
def daemon(env: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A real local daemon; yields its endpoint file, which ``run_agent`` dispatch uses."""
    endpoint_file = str(env.tmpdir / "sorcar-local.json")
    # The parent runs INSIDE the daemon; its ``run_agent`` resolves the
    # daemon through this variable (there is no cron-recorded endpoint).
    monkeypatch.setenv("KISS_SORCAR_LOCAL", endpoint_file)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    server = RemoteAccessServer(local_endpoint_file=endpoint_file, work_dir=str(env.repo))
    asyncio.run_coroutine_threadsafe(server.start_private_async(), loop).result(timeout=30)
    try:
        yield endpoint_file
    finally:
        asyncio.run_coroutine_threadsafe(server.stop_async(), loop).result(timeout=15)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


def test_run_agent_sub_tasks_get_the_parents_add_to_tools(
    env: IsolatedKissHome, daemon: str,
) -> None:
    """Three sub-tasks of one ``add_to_tools()`` parent: plain, adding, fixed toolset."""
    repo = env.repo
    (repo / "parent_sea.py").write_text(PARENT_SCRIPT, encoding="utf-8")
    (repo / "child_adding_sea.py").write_text(CHILD_ADDING_SCRIPT, encoding="utf-8")
    (repo / "child_fixed_sea.py").write_text(CHILD_FIXED_SCRIPT, encoding="utf-8")
    child_tools: dict[str, list[str]] = {}
    child_prompts: dict[str, str] = {}
    child_last_messages: dict[str, str] = {}
    parent_suffix = "\n\nPARENT-SUFFIX: end the summary with the word suffix."

    def dispatch(agent: str, marker: str, options: str = "") -> dict[str, Any]:
        return tool_call_response(
            "run_agent",
            {"agent": agent, "task": f"{marker} say done", "timeout": "120",
             "options": options},
        )

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        last = _last_text(request)
        # The parent after a ``run_agent`` result (the child's summary).
        if "done-CHILD-C" in last:
            return finish_response("parent-done")
        if "done-CHILD-B" in last:
            return dispatch(str(repo / "child_fixed_sea.py"), "CHILD-C")
        if "done-CHILD-A" in last:
            return dispatch(str(repo / "child_adding_sea.py"), "CHILD-B")
        # A child's first (and only) step.
        for marker in ("CHILD-A", "CHILD-B", "CHILD-C"):
            if marker in last:
                child_tools[marker] = _tool_names(request)
                child_prompts[marker] = request_text(request)
                child_last_messages[marker] = last
                return finish_response(f"done-{marker}")
        # The sequential parent's children inherit ``allow_fan_out``;
        # CHILD-A asks for fan-out explicitly.
        return dispatch("", "CHILD-A", '{"allow_fan_out": true}')

    model = StandInModelServer(responder)
    try:
        result = sorcar.run(
            "PARENT-TASK dispatch the three children",
            work_dir=str(repo),
            extension_agent_path=str(repo / "parent_sea.py"),
            model=STANDIN_MODEL,
            model_config=model.model_config,
            use_worktree=False,
            auto_commit=False,
            append_to_prompt=parent_suffix,
            endpoint_file=daemon,
            timeout=300,
        )
    finally:
        model.stop()
    assert result.success is True, result
    assert "parent-done" in result.text

    # Every sub-task's prompt ends with the parent's own prompt suffix.
    for marker in ("CHILD-A", "CHILD-B", "CHILD-C"):
        assert child_last_messages[marker].rstrip().endswith(parent_suffix.strip()), (
            marker, child_last_messages[marker][-300:],
        )

    # The plain sub-agent (dummy_sea.py) runs the basic toolset plus
    # the parent's tool — and reads the parent's protocol it refers to.
    plain = child_tools["CHILD-A"]
    assert "parent_ledger" in plain
    assert "finish" in plain and "Bash" in plain
    assert "PARENT-PROTOCOL" in child_prompts["CHILD-A"]
    # The sub-agent asked for run_parallel, so it has the built-in
    # ``number_of_cores``; the parent's same-named tool is skipped.
    assert "run_parallel" in plain
    assert plain.count("number_of_cores") == 1

    # A sub-task adding its own tools has both sets; the tool both
    # scripts define by the same name is present once (the child's own).
    adding = child_tools["CHILD-B"]
    assert "child_probe" in adding
    assert adding.count("parent_ledger") == 1
    assert "Bash" in adding
    # Nothing asked for fan-out: the sequential parent's choice is
    # inherited, so the parent's ``number_of_cores`` tool is the only one.
    assert "run_parallel" not in adding
    assert adding.count("number_of_cores") == 1

    # A sub-task whose script fixes the toolset keeps exactly that set.
    fixed = child_tools["CHILD-C"]
    assert "only_tool" in fixed
    assert "parent_ledger" not in fixed
    assert "Bash" not in fixed
    assert "finish" in fixed


def test_unknown_or_missing_parent_yields_no_tools() -> None:
    """No registered parent (or an empty id): nothing to inherit."""
    assert task_runner._parent_extra_tools("no-such-task") == []
    assert task_runner._parent_extra_tools("") == []
