# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for sorcar_agent.py: prompt construction, arg parsing, task resolution,
CLI callbacks, callback wiring, bash streaming, and autocomplete clipping.

The last three tests drive a real ``SorcarAgent`` run against the scripted local
model server: a run that changed code finishes, and merges its worktree, on the
first call, with no finish gate (no verifier run, no review, no second reading
of a ledger), and a caller's own ``finish`` tool replaces the built-in one.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import yaml

from kiss.agents.seas.coding import coding_sea
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.tests.agents.sorcar.local_model_server import MODEL, finish_body, serve, tool_call_body
from kiss.tests.server.conftest import init_repo, kiss_wt_branches, run_git
from kiss.tests.server.parallel_agent_harness import OfflineFastModel

DONE = "<p>hello.py written.</p>"
#: A check script in the run's scratch dir: it fails, and leaves a file behind when it runs.
FAILING_CHECK = "echo ran >> verifier-ran.txt\nexit 1\n"


class TestSorcarAgentCallbackWiring:
    def test_ask_user_question_without_callback(self) -> None:
        agent = SorcarAgent("test")
        try:
            tools = agent._get_tools()
            ask_tool = next(t for t in tools if t.__name__ == "ask_user_question")
            result = ask_tool("hello?")
            assert "not available" in result
        finally:
            if agent.web_use_tool is not None:
                agent.web_use_tool.close()

    def test_run_sets_callbacks_temporarily(self) -> None:
        agent = SorcarAgent("test")
        parent_class = cast(Any, agent.__class__.__mro__[1])
        original_perform = parent_class.perform_task
        captured: dict[str, object] = {}

        def ask_callback(question: str) -> str:
            return f"UI: {question}"

        def fake_perform(
            self: object, tools: list, attachments: list | None = None,
        ) -> str:
            del self, attachments
            captured["ask"] = getattr(agent, "_ask_user_question_callback", None)
            callables = [t for t in tools if callable(t)]
            ask_tool = next(t for t in callables if t.__name__ == "ask_user_question")
            captured["answer"] = ask_tool("hello")
            return "success: true\nis_continue: false\nsummary: ok\n"

        parent_class.perform_task = fake_perform  # type: ignore[method-assign]
        try:
            result = agent.run(
                prompt_template="task",
                ask_user_question_callback=ask_callback,
            )
        finally:
            parent_class.perform_task = original_perform  # type: ignore[method-assign]

        assert "success: true" in result
        assert captured["ask"] is ask_callback
        assert captured["answer"] == "UI: hello"
        assert getattr(agent, "_ask_user_question_callback", None) is None
        assert agent.web_use_tool is None


def _run_scripted(work_dir: Path, script: list[bytes],
                  tools: list[Callable[..., Any]] | None = None) -> tuple[dict, list[dict]]:
    """Run a real ``SorcarAgent`` in *work_dir* on the scripted model answers *script*.

    Returns:
        The run's result and the requests the model server received.
    """
    with serve(script) as (url, requests):
        result = SorcarAgent("no-gate").run(
            prompt_template="Write hello.py that prints hello.", model_name=MODEL,
            work_dir=str(work_dir), max_steps=10, max_budget=1.0,
            model_config={"base_url": url, "api_key": "local"}, tools=tools,
            use_memory=False, verbose=False, web_tools=False,
        )
    return yaml.safe_load(result), requests


def _finish_arguments(request: dict) -> list[str]:
    """The argument names of the ``finish`` tool *request* offers the model."""
    tool = next(t["function"] for t in request["tools"] if t["function"]["name"] == "finish")
    return sorted(tool["parameters"]["properties"])


def _tool_result(request: dict) -> str:
    """The last tool result among the messages of *request*."""
    return [str(m.get("content", "")) for m in request["messages"] if m.get("role") == "tool"][-1]


def test_a_coding_run_finishes_on_its_first_finish(tmp_path: Path) -> None:
    """A run that changed code is not gated: its first ``finish`` ends it.

    A failing ``verify.sh`` in the scratch dir is not run, the ledger the agent
    writes gets no second reading, the model server sees the agent's three steps
    and nothing else (no review), and neither the ``finish`` tool nor the system
    prompt carries a rule that came with the gates (verifier, ledger, evidence,
    several candidates, no early finish).
    """
    work = tmp_path / "work"
    (work / "tmp").mkdir(parents=True)
    (work / "tmp" / "verify.sh").write_text(FAILING_CHECK)
    ledger = {"file_path": str(work / "tmp" / "ledger.md"), "content": "1. hello means hello\n"}
    script = [
        tool_call_body("Write", ledger, 1000),
        tool_call_body("Write", {"file_path": str(work / "hello.py"),
                                 "content": "print('bye')\n"}, 1000),
        finish_body(DONE, prompt_tokens=1000),
    ]
    result, requests = _run_scripted(work, script)
    assert result["success"] is True and result["summary"] == DONE
    assert len(requests) == 3
    assert not (work / "verifier-ran.txt").exists()
    assert "Independent reading" not in _tool_result(requests[1])
    assert _finish_arguments(requests[0]) == [
        "is_continue", "success", "suggested_next_task", "summary_in_html"]
    system = str(requests[0]["messages"][0]["content"])
    assert "Pre-Finish Verification" in system
    for gone in ("verify.sh", "ledger.md", "finish gate", "`evidence` argument",
                 "evidence in the summary", "you guessed", "plausible answer",
                 "more than one candidate"):
        assert gone not in system, gone


def test_a_callers_finish_tool_replaces_the_built_in_one(tmp_path: Path) -> None:
    """The coding SEA's ``finish`` (with ``evidence``) is the run's only ``finish`` tool."""
    call = {"success": True, "summary_in_html": DONE, "evidence": '["pytest -q"]'}
    result, requests = _run_scripted(
        tmp_path, [tool_call_body("finish", call, 1000)], tools=[coding_sea.finish])
    assert result["success"] is True and result["summary"] == DONE
    assert _finish_arguments(requests[0]) == [
        "evidence", "is_continue", "success", "suggested_next_task", "summary_in_html"]
    names = [t["function"]["name"] for t in requests[0]["tools"]]
    assert names[0] == "finish" and names.count("finish") == 1


def test_a_worktree_merges_on_its_first_commit_and_merge(tmp_path: Path) -> None:
    """``commit_and_merge`` of a run that changed code merges at once, and the run finishes.

    The failing ``verify.sh`` the agent left in the worktree's scratch dir is not
    run before the merge or at the finish, and the merge result carries no findings.
    """
    repo = tmp_path / "repo"
    init_repo(repo)
    (repo / ".gitignore").write_text("tmp/\nverifier-ran.txt\n")
    run_git(repo, "add", ".gitignore")
    run_git(repo, "commit", "-q", "-m", "ignore tmp")
    script = [
        tool_call_body("create_worktree", {}, 1000),
        tool_call_body("Write", {"file_path": "tmp/verify.sh", "content": FAILING_CHECK}, 1000),
        tool_call_body("Write", {"file_path": "hello.py", "content": "print('bye')\n"}, 1000),
        tool_call_body("commit_and_merge", {}, 1000),
        finish_body(DONE, prompt_tokens=1000),
    ]
    # Without a provider commit_and_merge commits under its fallback message.
    with OfflineFastModel():
        result, requests = _run_scripted(repo, script)
    assert result["success"] is True and result["summary"] == DONE
    assert len(requests) == 5
    merged = _tool_result(requests[4])
    assert merged.startswith("Successfully merged branch") and "findings" not in merged
    assert (repo / "hello.py").read_text() == "print('bye')\n"
    assert kiss_wt_branches(repo) == []
    assert not list(tmp_path.rglob("verifier-ran.txt"))
