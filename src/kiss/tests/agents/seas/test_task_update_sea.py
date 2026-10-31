# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the bundled task-update agent
(:mod:`kiss.agents.seas.task_update.task_update_sea`).

The agent-level test runs a real :class:`ChatSorcarAgent` ReAct loop
against the scripted local chat-completions server configured from the
SEA's methods; its ``task_transcript`` call reads a task persisted in
the test session's real SQLite history (``KISS_HOME`` is a temporary
directory, see ``conftest.py``), so the digest really flows through the
tool result into the finish summary.  The digest format itself is
pinned by ``src/kiss/tests/agents/sorcar/test_task_digest.py``
(``task_transcript`` wraps ``task_digest.transcript_page``).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from kiss.agents.seas.task_update import task_update_sea as sea
from kiss.agents.seas.task_update.task_update_sea import TaskUpdateSea
from kiss.agents.sorcar import sea_commands
from kiss.tests.agents.seas.conftest import persist_task, run_sea
from kiss.tests.agents.seas.sea_contract import (
    assert_worker_contract,
    system_message,
    tool_names,
    tool_results,
)
from kiss.tests.agents.sorcar.local_model_server import (
    finish_body,
    tool_call_body,
)


def _persist_task(prompt: str, events: list[dict[str, Any]]) -> str:
    """Persist a still-running task row (started 90 s ago, no spend recorded yet) with
    *events* and return its id."""
    return persist_task(
        prompt, events, model="test-model", startTs=int(time.time() * 1000) - 90_000,
        endTs=0, cost=0, steps=0, tokens=0,
    )


def test_sea_methods_and_prompt_follow_the_contract() -> None:
    """The SEA pins its run: transcript tool, bash profile, no extras."""
    assert sea.build_prompt("abc123") == (
        "What have the task with abc123 done so far and what are the partial results?"
    )
    agent = TaskUpdateSea()
    # ``system_prompt`` replaces the assembled prompt with the fixed one.
    assert agent.system_prompt("ASSEMBLED") == sea.SYSTEM_PROMPT
    assert "task_transcript" in sea.SYSTEM_PROMPT
    assert agent.tools([]) == [sea.task_transcript]
    assert agent.tools([print]) == [print, sea.task_transcript]
    assert_worker_contract(sea, agent, {"tool_profile": "bash", "max_budget": 1.0})


def test_agent_reads_the_transcript_and_finishes_with_the_report(tmp_path: Path) -> None:
    """With the SEA's configuration the model gets Bash, finish and task_transcript.

    The scripted model calls ``task_transcript`` with the id from the
    prompt and then finishes; the test checks the tools offered, the
    system prompt, the real digest in the tool-result message, and
    that the finish summary is the run's result.
    """
    task_id = _persist_task(
        "Refactor the parser",
        [
            {"type": "prompt", "text": "Refactor the parser"},
            {"type": "tool_call", "name": "Read", "path": "parser.py", "lang": "python"},
            {"type": "tool_result", "content": "def parse(): ..."},
        ],
    )
    report = "<h4>Goal</h4><ul><li>Refactor the parser</li></ul>"
    script = [
        tool_call_body("task_transcript", {"task_id": task_id}, prompt_tokens=500),
        finish_body(report, prompt_tokens=700),
    ]
    run = sea_commands.evaluate_sea([TaskUpdateSea()], sea.build_prompt(task_id))
    assert run.prompt == sea.build_prompt(task_id)
    assert run.tools_hook is not None and run.system_prompt_hook is not None
    parsed, agentic = run_sea(run, script, tmp_path, max_steps=4)
    assert parsed["success"] is True
    assert parsed["summary"] == report

    assert len(agentic) == 2
    for request in agentic:
        names = tool_names(request)
        assert names == {"Bash", "finish", "task_transcript"}, names
        system = system_message(request)
        assert system.startswith(sea.SYSTEM_PROMPT)
    user = next(m for m in agentic[0]["messages"] if m["role"] == "user")
    assert sea.build_prompt(task_id) in str(user["content"])
    results = tool_results(agentic[1])
    assert len(results) == 1
    digest = results[0]
    assert f"Task id: {task_id}" in digest
    assert "Task prompt: Refactor the parser" in digest
    assert '[0] TOOL CALL Read({"path": "parser.py"})' in digest
    assert "[1] RESULT: def parse(): ..." in digest
