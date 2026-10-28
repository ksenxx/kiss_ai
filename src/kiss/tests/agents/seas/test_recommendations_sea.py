# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the bundled recommendations agent
(:mod:`kiss.agents.seas.recommendations.recommendations_sea`).

The ``recent_tasks`` tool is exercised against tasks persisted in the
test session's real SQLite history (``KISS_HOME`` is a temporary
directory, see ``conftest.py``), ``connected_apps`` against the real
auth-status probe subprocess (a Brave Search key file under the test
home makes that channel count as connected), the memory tools against
the test home's memory directory, ``save_recommendations`` against the
real ``<home>/recommendations.json``, and the agent-level test runs a
real :class:`ChatSorcarAgent` ReAct loop against the scripted local
chat-completions server configured from the SEA's methods, so the
history really flows through the tool result into the stored file the
sidebar reads.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.seas.recommendations import recommendations_sea as sea
from kiss.agents.seas.recommendations.recommendations_sea import RecommendationsSea
from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar import sea_commands
from kiss.server import sidebar_panels
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

TEN = [
    "Fix the failing parser test in parser.py",
    "Add a changelog entry for the new /recommendations command",
    "Run the full test suite in /home/me/proj",
    "Review the open pull request on the cron scheduler",
    "Document the sidebar panels in README.md",
    "Draft Gmail replies to the unread meeting requests and put the slots in Google Calendar",
    "Summarise this week's Google Calendar meetings into a Google Doc and mail it to me",
    "Export the Gmail receipts of October into a Google Sheets ledger",
    "Profile the slow history query in persistence.py",
    "Add a smoke test for the Bedrock settings subpanel",
]

TOOL_NAMES = {
    "recent_tasks",
    "connected_apps",
    "memory_search",
    "memory_pull",
    "memory_read",
    "save_recommendations",
}


def _persist(
    task: str,
    age_seconds: float,
    *,
    work_dir: str = "/home/me/proj",
    result: str | None = None,
    success: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Persist a task row that started *age_seconds* ago; return its id.

    *result* is ``None`` for a task still running (no outcome recorded);
    an empty string is a finished run with an empty summary.
    """
    return persist_task(
        task,
        result=result,
        success=success,
        timestamp=time.time() - age_seconds,
        model="test-model",
        work_dir=work_dir,
        **(extra or {}),
    )


def _tool_names(tools: list[Any]) -> list[str]:
    return [tool.__name__ for tool in tools]


def test_sea_methods_and_prompt_follow_the_contract() -> None:
    """The SEA pins its run: its own tools, no built-in tools, a $2 budget."""
    agent = RecommendationsSea()
    assert agent.system_prompt("ASSEMBLED") == sea.SYSTEM_PROMPT
    for name in ("recent_tasks", "connected_apps", "memory_pull", "save_recommendations"):
        assert name in sea.SYSTEM_PROMPT
    assert "Suggested next" in sea.SYSTEM_PROMPT and "at least 3 of the 10" in sea.SYSTEM_PROMPT
    assert "when only one is" in sea.SYSTEM_PROMPT  # one connected app has nothing to combine
    assert (
        agent.prompt("now")
        == agent.prompt("anything else")
        == ("Suggest 10 tasks from my recent work and my connected apps.")
    )
    assert _tool_names(agent.tools([])) == [
        "recent_tasks",
        "connected_apps",
        "memory_search",
        "memory_pull",
        "memory_read",
        "save_recommendations",
    ]
    assert _tool_names(agent.tools([print])) == ["print", *_tool_names(agent.tools([]))]
    assert_worker_contract(
        sea,
        agent,
        {"tool_profile": "none", "max_budget": 2.0, "timeout": 900, "locked": ["tool_profile"]},
    )
    assert "RECOMMENDATIONS" in agent.description() and "ten" in agent.description()


def test_recent_tasks_lists_the_last_day_and_twenty_completed_tasks(
    empty_history: None,
) -> None:
    """Top-level user tasks, newest first, with outcome and result.

    Every task of the last 24 h is listed; older tasks (within 30 days)
    follow until 20 completed tasks are shown.  Sub-agents, side
    channels, cron prompt runs and this agent's own runs are left out;
    HTML results are flattened and long texts clipped.
    """
    assert sea.recent_tasks() == "No tasks in the last 30 days."
    _persist("Older than a month", 31 * 86400, result="done", success=True)
    _persist("Older than a day", 25 * 3600, result="done", success=True)
    parent = _persist(
        "Fix the parser",
        3 * 3600,
        result="<p>Fixed <b>parser.py</b></p>\n<ul><li>2 tests pass</li></ul>",
        success=True,
    )
    _persist(
        "Sub-agent work",
        2 * 3600,
        result="x",
        success=True,
        extra={"parent_task_id": parent},
    )
    _persist(
        "Side question",
        2 * 3600,
        result="x",
        success=True,
        extra={"subagent": {"parent_task_id": parent, "side_channel": True}},
    )
    _persist("Cron run", 90 * 60, result="x", success=True, extra={"sea": "cron_prompt_sea"})
    _persist(
        "Earlier suggestions",
        60 * 60,
        result="x",
        success=True,
        extra={"sea": "recommendations_sea"},
    )
    # A cron prompt job under another SEA opens with the preamble; an
    # in_chat cron job (the morning greeting) runs in a cron scratch dir.
    _persist(
        sea.PROMPT_PREAMBLE + "/weekly_report now",
        50 * 60,
        result="x",
        success=True,
        extra={"sea": "weekly_report_sea"},
    )
    _persist(
        "/good_morning now",
        45 * 60,
        result="Good morning!",
        success=True,
        work_dir=str(sea.kiss_home() / "cron" / "runs" / "abc-xyz"),
        extra={"sea": "good_morning_sea"},
    )
    _persist("Deploy the site", 40 * 60, result="Build broke", success=False, work_dir="/srv/site")
    # A run that finished with an empty summary is still a finished, successful run.
    _persist("Nothing to report", 35 * 60, result="", success=True)
    _persist("Still running " + "x" * 400, 30)
    lines = sea.recent_tasks().splitlines()
    # Fewer than 20 completed tasks in the last day: the day-old task is
    # listed too, the month-old one never.
    assert len(lines) == 5, lines
    assert lines[0].startswith("- 1m ago | /home/me/proj | unfinished | Still running xxx")
    # Clipped to MAX_TASK_CHARS; the unfinished row's placeholder result is not shown.
    assert lines[0].endswith("…") and "Agent Failed Abruptly" not in lines[0]
    assert len(lines[0].split(" | ", 3)[3]) == sea.MAX_TASK_CHARS
    assert lines[1] == "- 35m ago | /home/me/proj | ok | Nothing to report"
    assert lines[2] == "- 40m ago | /srv/site | failed | Deploy the site -> Build broke"
    assert lines[3] == (
        "- 3h ago | /home/me/proj | ok | Fix the parser -> Fixed parser.py 2 tests pass"
    )
    assert lines[4] == "- 1d ago | /home/me/proj | ok | Older than a day -> done"
    # The query itself keeps the automations (the tool drops them by
    # ``sea``) but never a sub-agent, a side channel or an older row.
    rows = th.user_tasks_since(time.time() - 24 * 3600)
    assert [str(r["task"])[:13] for r in rows] == [
        "Still running",
        "Nothing to re",
        "Deploy the si",
        "/good_morning",
        "You are runni",
        "Earlier sugge",
        "Cron run",
        "Fix the parse",
    ]
    assert [r["sea"] for r in rows] == [
        "",
        "",
        "",
        "good_morning_sea",
        "weekly_report_sea",
        "recommendations_sea",
        "cron_prompt_sea",
        "",
    ]
    assert [r["success"] for r in rows] == [None, True, False, True, True, True, True, True]
    assert rows[2]["workDir"] == "/srv/site" and rows[2]["model"] == "test-model"
    assert [sea._is_automation(r) for r in rows] == [
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        False,
    ]
    assert sea._is_automation({"task": sea.PROMPT, "sea": "", "workDir": "/x"})
    # With 19 completed tasks in the last day (three so far) the day-old
    # one still shows; the 20th pushes it out, while the unfinished one stays.
    for i in range(sea.MIN_COMPLETED - 4):
        _persist(f"Done today {i}", 10 * 60, result="x", success=True)
    listed = sea.recent_tasks()
    assert len(listed.splitlines()) == sea.MIN_COMPLETED + 1 and "Older than a day" in listed
    _persist("Done today, the twentieth", 10 * 60, result="x", success=True)
    listed = sea.recent_tasks()
    assert len(listed.splitlines()) == sea.MIN_COMPLETED + 1
    assert "Older than a day" not in listed and "Still running" in listed
    # The cap applies after the automations are dropped: a burst of
    # cron rows never hides a user task, and only the newest MAX_TASKS
    # user tasks are listed.
    for i in range(sea.MAX_TASKS + 1):
        _persist(f"Cron burst {i}", 10, result="x", success=True, extra={"sea": "cron_prompt_sea"})
    assert len(sea.recent_tasks().splitlines()) == sea.MIN_COMPLETED + 1
    for i in range(sea.MAX_TASKS):
        _persist(f"User task {i}", 5, result="x", success=True)
    capped = sea.recent_tasks().splitlines()
    assert len(capped) == sea.MAX_TASKS and all("User task" in line for line in capped)


def test_age_text_and_result_clipping() -> None:
    """Ages read m / h / d; a long result is clipped with an ellipsis."""
    assert sea._age(0) == "1m"
    assert sea._age(59 * 60) == "59m"
    assert sea._age(3 * 3600 + 5) == "3h"
    assert sea._age(49 * 3600) == "2d"
    assert sea._plain("  a \n b  ", 10) == "a b"
    clipped = sea._plain("word " * 100, sea.MAX_RESULT_CHARS)
    assert len(clipped) == sea.MAX_RESULT_CHARS and clipped.endswith("…")


def test_connected_apps_names_the_signed_in_channels(no_connected_apps: Path) -> None:
    """The real probe subprocess decides: no credential, no app; a Brave key, Brave Search."""
    assert sea.connected_apps() == "No third-party app is connected."
    brave = no_connected_apps
    brave.parent.mkdir(parents=True, exist_ok=True)
    brave.write_text(json.dumps({"api_key": "test-key"}))
    assert sea.connected_apps() == "Connected apps: Brave Search."


def test_memory_read_tools_follow_the_memory_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Memory off: no memory tool; on: the three read-only tools over the test home's pages."""
    monkeypatch.setenv("KISS_USE_MEMORY", "0")
    assert sea.memory_read_tools() == []
    assert _tool_names(RecommendationsSea().tools([])) == [
        "recent_tasks",
        "connected_apps",
        "save_recommendations",
    ]
    monkeypatch.setenv("KISS_USE_MEMORY", "1")
    tools = sea.memory_read_tools()
    assert _tool_names(tools) == ["memory_search", "memory_pull", "memory_read"]
    assert tools[2]("no-such-page") == "Error: no memory page named 'no-such-page'."
    # Read-only: the write tools are not offered.
    assert "memory_write" not in _tool_names(RecommendationsSea().tools([]))


def test_save_and_load_recommendations(recommendations_path: Path) -> None:
    """Exactly ten non-empty sentences are stored atomically; anything else is refused."""
    path = recommendations_path
    assert sea.load_recommendations() == {"generatedAt": 0.0, "tasks": []}
    assert sidebar_panels.recommendations_report() == {"generatedAt": 0, "tasks": []}
    # The model may hand the tool anything: a bare string or non-string
    # items are refused before they could be coerced into "tasks".
    assert sea.save_recommendations("abcde") == "Error: pass a list of 10 strings."  # type: ignore[arg-type]
    assert sea.save_recommendations([None, False, 0, {}, []]) == (  # type: ignore[list-item]
        "Error: pass a list of 10 strings."
    )
    assert sea.save_recommendations(TEN[:5]) == ("Error: pass exactly 10 non-empty tasks (got 5).")
    assert sea.save_recommendations([*TEN[:9], "  "]) == (
        "Error: pass exactly 10 non-empty tasks (got 10)."
    )
    assert not path.exists()
    before = time.time()
    assert sea.save_recommendations([" " + TEN[0] + "\n", *TEN[1:]]) == (
        "Saved 10 recommendations."
    )
    # Staged in a per-writer temp file (atomic_write_text), none left behind.
    assert [p.name for p in path.parent.glob(".recommendations.json-*")] == []
    stored = json.loads(path.read_text())
    assert stored["tasks"] == TEN
    assert before <= stored["generatedAt"] <= time.time()
    assert sea.load_recommendations() == {
        "generatedAt": stored["generatedAt"],
        "tasks": TEN,
    }
    report = sidebar_panels.recommendations_report()
    assert report == {"generatedAt": int(stored["generatedAt"] * 1000), "tasks": TEN}
    json.dumps(report)
    # A corrupt, foreign or half-typed file reads as "nothing yet".
    path.write_text("{not json")
    assert sea.load_recommendations() == {"generatedAt": 0.0, "tasks": []}
    path.write_text(json.dumps([1, 2]))
    assert sea.load_recommendations() == {"generatedAt": 0.0, "tasks": []}
    path.write_text(json.dumps({"tasks": "no"}))
    assert sea.load_recommendations() == {"generatedAt": 0.0, "tasks": []}
    path.write_text(json.dumps({"generatedAt": "soon", "tasks": ["x", "", 3]}))
    assert sea.load_recommendations() == {"generatedAt": 0.0, "tasks": ["x", "3"]}


def test_agent_reads_the_history_and_apps_and_stores_ten_suggestions(
    tmp_path: Path,
    empty_history: None,
    recommendations_path: Path,
    no_connected_apps: Path,
) -> None:
    """With the SEA's configuration the model gets its six tools and finish.

    The scripted model reads the history and the connected apps, stores
    ten sentences and finishes; the test checks the tools offered, the
    system prompt, the real history and probe answer in the tool-result
    messages, and the file the sidebar reads.
    """
    _persist("Fix the parser", 3 * 3600, result="Fixed parser.py", success=True)
    summary = "<ul>" + "".join(f"<li>{t}</li>" for t in TEN) + "</ul>"
    script = [
        tool_call_body("recent_tasks", {}, prompt_tokens=500),
        tool_call_body("connected_apps", {}, prompt_tokens=550),
        tool_call_body("save_recommendations", {"tasks": TEN}, prompt_tokens=600),
        finish_body(summary, prompt_tokens=700),
    ]
    run = sea_commands.evaluate_sea([RecommendationsSea()], "now")
    assert run.prompt == "Suggest 10 tasks from my recent work and my connected apps."
    assert run.tools_hook is not None and run.system_prompt_hook is not None
    parsed, agentic = run_sea(run, script, tmp_path, max_steps=6)
    assert parsed["success"] is True
    assert parsed["summary"] == summary
    assert sea.load_recommendations()["tasks"] == TEN
    assert sidebar_panels.recommendations_report()["tasks"] == TEN

    assert len(agentic) == 4
    for request in agentic:
        assert tool_names(request) == {"finish", *TOOL_NAMES}
        assert system_message(request).startswith(sea.SYSTEM_PROMPT)
    (listed,) = tool_results(agentic[1])
    assert listed.startswith("- 3h ago | /home/me/proj | ok | Fix the parser -> Fixed parser.py")
    assert tool_results(agentic[2])[-1].startswith("No third-party app is connected.")
    assert tool_results(agentic[3])[-1].startswith("Saved 10 recommendations.")
