# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``run_parallel(agent=...)``: fan-out children run as an agent script.

A real daemon runs a parent whose scripted model calls ``run_parallel``
with ``agent`` naming a SEA file.  Each child's request to the stand-in
model shows the SEA's configuration: its ``system_prompt()`` as the
base prompt, its ``add_to_system_prompt()`` after the parent's own
suffix, its ``prompt(task)`` wrapping the child's task, its
``add_to_tools()`` tool, and its ``tool_profile`` setting.  A channel
agent or an unknown agent is refused with an error string the parent
sees.  The children inherit the parent's model and sequential/parallel
choice through the same table as ``run_agent``.
"""

from __future__ import annotations

import asyncio
import textwrap
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from kiss.server import sorcar
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    request_text,
    tool_call_response,
)

CHILD_SEA = textwrap.dedent('''
    def child_probe(what: str) -> str:
        """Probe *what*."""
        return what


    def settings():
        return {"tool_profile": "bash", "is_parallel": False}


    def prompt(task: str) -> str:
        return "[child-sea] " + task + "\\n\\nCHILD-ADD"


    def system_prompt() -> str:
        return "CHILD-SEA SYSTEM PROMPT"


    def add_to_system_prompt() -> str:
        return "CHILD-SEA PROTOCOL"


    def add_to_tools():
        return [child_probe]
''')


def _tool_names(request: dict[str, Any]) -> list[str]:
    return [
        tool["function"]["name"]
        for tool in request.get("tools", [])
        if isinstance(tool, dict) and "function" in tool
    ]


def _system_text(request: dict[str, Any]) -> str:
    for message in request.get("messages", []):
        if message.get("role") == "system":
            content = message.get("content")
            return content if isinstance(content, str) else str(content)
    return ""


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    home = IsolatedKissHome(prefix="kiss-run-parallel-agent-")
    home.write_config(is_worktree=False, auto_commit_mode=False, classify_tasks=False)
    try:
        yield home
    finally:
        home.cleanup()


@pytest.fixture
def daemon(env: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    endpoint_file = str(env.tmpdir / "sorcar-local.json")
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


def test_run_parallel_children_run_as_the_named_agent_script(
    env: IsolatedKissHome, daemon: str,
) -> None:
    repo = env.repo
    (repo / "child_sea.py").write_text(CHILD_SEA, encoding="utf-8")
    children: dict[str, dict[str, Any]] = {}
    parent_requests: list[dict[str, Any]] = []

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        text = request_text(request)
        # A child's prompt carries the SEA's ``prompt(task)`` wrapping;
        # the parent's never does (its tool results hold only summaries).
        for marker in ("KID-1", "KID-2"):
            if f"[child-sea] {marker} say done" in text:
                children[marker] = {
                    "system": _system_text(request),
                    "text": text,
                    "tools": _tool_names(request),
                    "model": request.get("model"),
                }
                return finish_response(f"done-{marker}")
        parent_requests.append(request)
        step = len(parent_requests)
        if step == 1:
            return tool_call_response(
                "run_parallel",
                {"tasks": '["KID-1 say done", "KID-2 say done"]', "agent": "child_sea.py"},
            )
        if step == 2:
            return tool_call_response(
                "run_parallel", {"tasks": '["KID-3 say done"]', "agent": "no-such-agent-xyz"},
            )
        if step == 3:
            return tool_call_response(
                "run_parallel", {"tasks": '["KID-3 say done"]', "agent": "ntfy"},
            )
        return finish_response("parent-done")

    model = StandInModelServer(responder)
    try:
        result = sorcar.run(
            "PARENT-TASK fan out",
            work_dir=str(repo),
            model=STANDIN_MODEL,
            model_config=model.model_config,
            use_worktree=False,
            auto_commit=False,
            append_to_system_prompt="PARENT-SUFFIX-TEXT",
            endpoint_file=daemon,
            timeout=300,
        )
    finally:
        model.stop()
    assert result.success is True, result
    assert "parent-done" in result.text
    assert set(children) == {"KID-1", "KID-2"}, children
    for marker, child in children.items():
        assert child["system"].startswith("CHILD-SEA SYSTEM PROMPT"), child["system"][:200]
        suffix = child["system"]
        assert suffix.index("PARENT-SUFFIX-TEXT") < suffix.index("CHILD-SEA PROTOCOL")
        assert f"[child-sea] {marker} say done" in child["text"], child["text"][-400:]
        assert child["text"].rstrip().endswith("CHILD-ADD"), child["text"][-200:]
        assert sorted(child["tools"]) == ["Bash", "child_probe", "finish"], child["tools"]
        assert child["model"] == STANDIN_MODEL
    # The refused fan-outs reached the parent as error strings.
    parent_texts = [request_text(r) for r in parent_requests]
    assert any(
        "no-such-agent-xyz" in t and "Error:" in t for t in parent_texts
    ), parent_texts[-1][-500:]
    assert any("ntfy is a channel agent, which run_parallel cannot run" in t for t in parent_texts)


PARENT_SEA = textwrap.dedent('''
    def parent_probe(what: str) -> str:
        """Parent probe *what*."""
        return what


    def add_to_tools():
        return [parent_probe]
''')

PICKY_SEA_TEMPLATE = textwrap.dedent('''
    def settings():
        return {{"model_config": {model_config!r}}}


    def prompt(task: str) -> str:
        if "KID-BAD" in task:
            raise KeyError("no prompt for " + task)
        return "[picky] " + task
''')


def test_run_parallel_children_inherit_the_parent_and_a_broken_child_fails_alone(
    env: IsolatedKissHome, daemon: str,
) -> None:
    """The fan-out inheritance matches ``run_agent``'s; a per-task getter failure is per child.

    The children end their prompt with the parent's own prompt suffix,
    carry the parent's extra tool (its script's ``add_to_tools()``),
    run on the model configuration the child SEA names (a second
    stand-in server), and a ``prompt(task)`` that raises for ONE task
    fails that child alone while its sibling's result is collected.
    """
    repo = env.repo
    (repo / "parent_sea.py").write_text(PARENT_SEA, encoding="utf-8")
    children: dict[str, dict[str, Any]] = {}
    parent_requests: list[dict[str, Any]] = []

    def child_responder(request: dict[str, Any]) -> dict[str, Any]:
        text = request_text(request)
        children["KID-A"] = {"text": text, "tools": _tool_names(request)}
        return finish_response("done-KID-A")

    def parent_responder(request: dict[str, Any]) -> dict[str, Any]:
        parent_requests.append(request)
        if len(parent_requests) == 1:
            return tool_call_response(
                "run_parallel",
                {"tasks": '["KID-A say done", "KID-BAD say done"]', "agent": "picky_sea.py"},
            )
        return finish_response("parent-done")

    child_model = StandInModelServer(child_responder)
    parent_model = StandInModelServer(parent_responder)
    (repo / "picky_sea.py").write_text(
        PICKY_SEA_TEMPLATE.format(model_config=child_model.model_config), encoding="utf-8",
    )
    try:
        result = sorcar.run(
            "PARENT-TASK fan out",
            work_dir=str(repo),
            model=STANDIN_MODEL,
            model_config=parent_model.model_config,
            extension_agent_path=str(repo / "parent_sea.py"),
            use_worktree=False,
            auto_commit=False,
            append_to_prompt="PARENT-PROMPT-SUFFIX",
            endpoint_file=daemon,
            timeout=300,
        )
    finally:
        child_model.stop()
        parent_model.stop()
    assert result.success is True, result
    assert set(children) == {"KID-A"}, children
    child = children["KID-A"]
    assert "[picky] KID-A say done" in child["text"], child["text"][-400:]
    assert child["text"].rstrip().endswith("PARENT-PROMPT-SUFFIX"), child["text"][-200:]
    assert "parent_probe" in child["tools"], child["tools"]
    # The parent saw one success and one per-child failure.
    fanout_result = request_text(parent_requests[1])
    assert "done-KID-A" in fanout_result, fanout_result[-800:]
    assert "no prompt for KID-BAD" in fanout_result, fanout_result[-800:]
