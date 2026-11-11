# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A killed agent run continues from its checkpoint in a new process.

``RelentlessAgent`` with a ``resume_dir`` writes the live session's
conversation before every model call and every tool call
(``KISSAgent._write_checkpoint``).  The test runs such an agent in a
child process against a scripted model server, kills the child with
SIGKILL while a tool call blocks, and runs the agent again on the same
directory: the second process must continue the SAME conversation, hand
the model an "interrupted, outcome unknown" result for the call that was
running, a "not run" result for the call queued behind it, a note naming
the interrupted call with its arguments, and keep the spend the first
process had already made.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from kiss.core.kiss_agent import INTERRUPTED_CALL_RESULT, NOT_STARTED_CALL_RESULT
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    StandInModelServer,
    finish_response,
    tool_call_response,
)

_CHILD = r"""
import sys, time, yaml
from kiss.agents.sorcar.relentless_agent import RelentlessAgent

resume_dir, base_url, marker, work_dir = sys.argv[1:5]


def note(text: str) -> str:
    '''Record *text*.

    Args:
        text: What to note.
    '''
    return f"noted {text}"


def block(seconds: int) -> str:
    '''Wait *seconds* (the test kills this process meanwhile).

    Args:
        seconds: How long to wait.
    '''
    open(marker, "w").close()
    time.sleep(seconds)
    return "unblocked"


agent = RelentlessAgent("resume-under-test")
agent.resume_dir = resume_dir
result = agent.run(
    model_name=sys.argv[5],
    prompt_template="Call note, then block, then finish.",
    max_steps=10,
    max_budget=5.0,
    max_sub_sessions=3,
    work_dir=work_dir,
    verbose=False,
    tools=[note, block],
    model_config={"base_url": base_url, "api_key": "sk-test"},
)
print("RESULT " + yaml.safe_load(result)["summary"])
print(f"BUDGET {agent.budget_used:.8f}")
"""


def _two_tool_calls(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """One assistant turn calling *first* and *second* (both ``{"name", "arguments"}``)."""
    body = tool_call_response(first["name"], first["arguments"])
    second_body = tool_call_response(second["name"], second["arguments"])
    body["choices"][0]["message"]["tool_calls"].append(
        second_body["choices"][0]["message"]["tool_calls"][0]
    )
    return body


class _Script:
    """The scripted model: note, then block + a queued note, then finish."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.finished = threading.Event()

    def respond(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        turn = len(self.requests)
        if turn == 1:
            return tool_call_response("note", {"text": "before the kill"})
        if turn == 2:
            return _two_tool_calls(
                {"name": "block", "arguments": {"seconds": 300}},
                {"name": "note", "arguments": {"text": "queued behind block"}},
            )
        self.finished.set()
        return finish_response("resumed after the kill")


def _run_child(home: Path, args: list[str]) -> subprocess.Popen[str]:
    env = dict(os.environ, KISS_HOME=str(home), PYTHONPATH=os.pathsep.join(
        os.path.abspath(p) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p
    ))
    return subprocess.Popen(
        [sys.executable, "-c", _CHILD, *args],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def _wait_for(path: Path, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.05)


def test_killed_run_resumes_from_its_checkpoint() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="kiss-resume-kill-"))
    home = tmp / ".kiss"
    work = tmp / "work"
    work.mkdir()
    marker = tmp / "blocking"
    directory = str(home / "resume" / "task-under-test")
    script = _Script()
    server = StandInModelServer(script.respond)
    try:
        args = [directory, server.url, str(marker), str(work), STANDIN_MODEL]
        first = _run_child(home, args)
        _wait_for(marker)
        time.sleep(0.5)  # the checkpoint of the blocking call is written before the call
        os.kill(first.pid, signal.SIGKILL)
        first.wait(timeout=30)
        checkpoint = json.loads((Path(directory) / "checkpoint-0.json").read_text())
        assert [c["name"] for c in checkpoint["calls"]] == ["block", "note"]
        assert checkpoint["done"] == []
        assert checkpoint["model_name"] == STANDIN_MODEL
        spent_before_kill = checkpoint["usage"][0]
        assert spent_before_kill > 0
        session = json.loads((Path(directory) / "session.json").read_text())
        assert session["session"] == 0 and session["banked"] == [0, 0, 0]

        second = _run_child(home, args)
        out, err = second.communicate(timeout=120)
        assert second.returncode == 0, err
        assert "resumed after the kill" in out, out
        budget_line = next(line for line in out.splitlines() if line.startswith("BUDGET "))
        assert float(budget_line.split()[1]) > spent_before_kill
        assert script.finished.is_set()
        assert not Path(directory).exists(), "the resume state outlives the finished task"
    finally:
        server.stop()

    # The resumed process continued the conversation the first one had.
    resumed = script.requests[2]["messages"]
    # A tool result carries the usage footer after the tool's own text.
    tool_results = [m["content"].split("\n\n")[0] for m in resumed if m.get("role") == "tool"]
    assert tool_results == [
        "noted before the kill", INTERRUPTED_CALL_RESULT, NOT_STARTED_CALL_RESULT,
    ]
    note = resumed[-1]
    assert note["role"] == "user"
    assert 'block({"seconds": 300}) was running' in note["content"], note["content"]
    assert "Check its effect before running it again" in note["content"]
    assistant_calls = [
        [c["function"]["name"] for c in m["tool_calls"]]
        for m in resumed if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    assert assistant_calls == [["note"], ["block", "note"]]
    assert len(script.requests) == 3, "the resumed run re-asked the model for finished steps"


def test_a_run_nothing_restarts_writes_no_resume_state() -> None:
    """A headless run (the ``kiss`` CLI) checkpoints nothing.

    Only the daemon restarts a task, and it says so by writing the
    task's launch record before the run starts; without one the run
    must not leave ``<kiss home>/resume/<task_id>`` behind for the
    worktree reclaim to mistake for a task waiting to be resumed.
    """
    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
    from kiss.tests.server.parallel_agent_harness import IsolatedKissHome

    home = IsolatedKissHome("kiss-cli-no-resume-")
    resume_root = home.kiss_home / "resume"
    requests: list[dict[str, Any]] = []

    def respond(request: dict[str, Any]) -> dict[str, Any]:
        requests.append(request)
        if len(requests) == 1:
            return tool_call_response("Bash", {
                "command": f"ls -A {resume_root} 2>/dev/null | wc -l",
                "description": "count resume directories mid-run",
            })
        return finish_response("done")

    server = StandInModelServer(respond)
    agent = ChatSorcarAgent("cli-run")
    try:
        agent.run(
            prompt_template="look for resume state",
            model_name=STANDIN_MODEL,
            model_config=server.model_config,
            work_dir=str(home.repo),
            web_tools=False,
            use_memory=False,
            max_budget=1.0,
        )
        assert agent.last_task_id
        assert agent.resume_dir is None
        listing = requests[1]["messages"][-1]
        assert listing["role"] == "tool"
        assert listing["content"].split("\n")[0].strip() == "0", listing["content"]
        assert not resume_root.exists()
    finally:
        server.stop()
        home.cleanup()
