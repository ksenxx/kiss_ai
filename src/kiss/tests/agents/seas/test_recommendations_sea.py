# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the bundled recommendations agent
(:mod:`kiss.agents.seas.recommendations.recommendations_sea`).

The ``recent_tasks`` tool is exercised against tasks persisted in the
test session's real SQLite history (``KISS_HOME`` is a temporary
directory, see ``conftest.py``), ``save_recommendations`` against the
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

import yaml

from kiss.agents.seas.base.base_sea import WORKER_DEFAULTS, WorkerSea
from kiss.agents.seas.recommendations import recommendations_sea as sea
from kiss.agents.seas.recommendations.recommendations_sea import RecommendationsSea
from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sea_settings import resolve_settings
from kiss.server import sidebar_panels
from kiss.tests.agents.seas.sea_contract import assert_no_removed_getters
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)
from kiss.tests.agents.sorcar.test_history_date_range import _set_timestamp

_SEA_PATH = Path(sea.__file__).resolve()

FIVE = [
    "Fix the failing parser test in parser.py",
    "Add a changelog entry for the new /recommendations command",
    "Run the full test suite in /home/me/proj",
    "Review the open pull request on the cron scheduler",
    "Document the sidebar panels in README.md",
]


def _persist(
    task: str,
    age_seconds: float,
    *,
    work_dir: str = "/home/me/proj",
    result: str | None = None,
    success: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Persist a task row that started *age_seconds* ago; return its id."""
    payload: dict[str, object] = {"model": "test-model", "work_dir": work_dir}
    payload.update(extra or {})
    task_id, _chat = th._add_task(task, extra=payload)
    if result is not None:
        th._save_task_result(result, task_id, success=success)
    _set_timestamp(task_id, time.time() - age_seconds)
    return task_id


def test_sea_methods_and_prompt_follow_the_contract() -> None:
    """The SEA pins its run: the two tools, no built-in tools, a $1 budget."""
    agent = RecommendationsSea()
    assert agent.system_prompt("ASSEMBLED") == sea.SYSTEM_PROMPT
    assert "recent_tasks" in sea.SYSTEM_PROMPT and "save_recommendations" in sea.SYSTEM_PROMPT
    assert agent.prompt("now") == agent.prompt("anything else") == (
        "Suggest 5 tasks from the last 24 hours."
    )
    assert agent.tools([]) == [sea.recent_tasks, sea.save_recommendations]
    assert agent.tools([print]) == [print, sea.recent_tasks, sea.save_recommendations]
    assert isinstance(agent, WorkerSea)
    assert agent.settings({}) == {
        "tool_profile": "none",
        "max_budget": 1.0,
        "timeout": 600,
        "locked": ["tool_profile"],
    }
    # The WorkerSea layer (no classifier, browser or memory) is laid
    # under the SEA's own keys by the loader.
    assert sea_commands.base_settings([agent]) == resolve_settings(
        {**WORKER_DEFAULTS, **agent.settings({})}
    )
    assert sea_commands.base_settings([agent])["auto_classify"] is False
    assert "RECOMMENDATIONS" in agent.description()
    assert_no_removed_getters(sea)


def test_slash_recommendations_resolves_to_the_bundled_sea() -> None:
    """``/recommendations now`` resolves to this SEA, which the chat and cron run directly."""
    assert sea_commands.get_command("recommendations") == _SEA_PATH
    hit = sea_commands.slash_command_task("/recommendations now")
    assert hit is not None
    task_text, path = hit
    assert (task_text, path) == ("now", _SEA_PATH)
    assert sea_commands.sea_settings(path) == sea_commands.base_settings([RecommendationsSea()])
    assert sea_commands.help_text_if_command("/recommendations help") == (
        RecommendationsSea().description()
    )


def test_recent_tasks_lists_the_users_last_day_only(empty_history: None) -> None:
    """Top-level user tasks of the last 24 h, newest first, with outcome and result.

    Sub-agents, side channels, cron prompt runs, this agent's own runs
    and older tasks are left out; HTML results are flattened and long
    texts clipped.
    """
    assert sea.recent_tasks() == "No tasks in the last 24 hours."
    _persist("Older than a day", 25 * 3600, result="done", success=True)
    parent = _persist(
        "Fix the parser", 3 * 3600,
        result="<p>Fixed <b>parser.py</b></p>\n<ul><li>2 tests pass</li></ul>", success=True,
    )
    _persist(
        "Sub-agent work", 2 * 3600, result="x", success=True,
        extra={"parent_task_id": parent},
    )
    _persist(
        "Side question", 2 * 3600, result="x", success=True,
        extra={"subagent": {"parent_task_id": parent, "side_channel": True}},
    )
    _persist("Cron run", 90 * 60, result="x", success=True, extra={"sea": "cron_prompt_sea"})
    _persist("Earlier suggestions", 60 * 60, result="x", success=True,
             extra={"sea": "recommendations_sea"})
    # A cron prompt job under another SEA opens with the preamble; an
    # in_chat cron job (the morning greeting) runs in a cron scratch dir.
    _persist(sea.PROMPT_PREAMBLE + "/weekly_report now", 50 * 60, result="x", success=True,
             extra={"sea": "weekly_report_sea"})
    _persist("/good_morning now", 45 * 60, result="Good morning!", success=True,
             work_dir=str(sea.kiss_home() / "cron" / "runs" / "abc-xyz"),
             extra={"sea": "good_morning_sea"})
    _persist("Deploy the site", 40 * 60, result="Build broke", success=False,
             work_dir="/srv/site")
    _persist("Still running " + "x" * 400, 30)
    lines = sea.recent_tasks().splitlines()
    assert len(lines) == 3, lines
    assert lines[0].startswith("- 1m ago | /home/me/proj | unfinished | Still running xxx")
    # Clipped to MAX_TASK_CHARS; the unfinished row's placeholder result is not shown.
    assert lines[0].endswith("…") and "Agent Failed Abruptly" not in lines[0]
    assert len(lines[0].split(" | ", 3)[3]) == sea.MAX_TASK_CHARS
    assert lines[1] == "- 40m ago | /srv/site | failed | Deploy the site -> Build broke"
    assert lines[2] == (
        "- 3h ago | /home/me/proj | ok | Fix the parser -> Fixed parser.py 2 tests pass"
    )
    # The query itself keeps the automations (the tool drops them by
    # ``sea``) but never a sub-agent, a side channel or an older row.
    rows = th.user_tasks_since(time.time() - 24 * 3600)
    assert [str(r["task"])[:13] for r in rows] == [
        "Still running", "Deploy the si", "/good_morning", "You are runni", "Earlier sugge",
        "Cron run", "Fix the parse",
    ]
    assert [r["sea"] for r in rows] == [
        "", "", "good_morning_sea", "weekly_report_sea", "recommendations_sea",
        "cron_prompt_sea", "",
    ]
    assert [r["success"] for r in rows] == [None, False, True, True, True, True, True]
    assert rows[1]["workDir"] == "/srv/site" and rows[1]["model"] == "test-model"
    assert [sea._is_automation(r) for r in rows] == [
        False, False, True, True, True, True, False,
    ]
    assert sea._is_automation({"task": sea.PROMPT, "sea": "", "workDir": "/x"})
    # The cap applies after the automations are dropped: a burst of
    # cron rows never hides a user task, and only the newest MAX_TASKS
    # user tasks are listed.
    for i in range(sea.MAX_TASKS + 1):
        _persist(f"Cron burst {i}", 10, result="x", success=True, extra={"sea": "cron_prompt_sea"})
    assert len(sea.recent_tasks().splitlines()) == 3
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


def test_save_and_load_recommendations(tmp_path: Path) -> None:
    """Exactly five non-empty sentences are stored atomically; anything else is refused."""
    path = sea.recommendations_file()
    path.unlink(missing_ok=True)
    assert sea.load_recommendations() == {"generatedAt": 0.0, "tasks": []}
    assert sidebar_panels.recommendations_report() == {"generatedAt": 0, "tasks": []}
    # The model may hand the tool anything: a bare string or non-string
    # items are refused before they could be coerced into "tasks".
    assert sea.save_recommendations("abcde") == "Error: pass a list of 5 strings."  # type: ignore[arg-type]
    assert sea.save_recommendations([None, False, 0, {}, []]) == (  # type: ignore[list-item]
        "Error: pass a list of 5 strings."
    )
    assert sea.save_recommendations(["a", "b"]) == (
        "Error: pass exactly 5 non-empty tasks (got 2)."
    )
    assert sea.save_recommendations(["a", "b", "c", "  ", "e"]) == (
        "Error: pass exactly 5 non-empty tasks (got 5)."
    )
    assert not path.exists()
    before = time.time()
    assert sea.save_recommendations([" " + FIVE[0] + "\n", *FIVE[1:]]) == (
        "Saved 5 recommendations."
    )
    # Staged in a per-writer temp file (atomic_write_text), none left behind.
    assert [p.name for p in path.parent.glob(".recommendations.json-*")] == []
    stored = json.loads(path.read_text())
    assert stored["tasks"] == FIVE
    assert before <= stored["generatedAt"] <= time.time()
    assert sea.load_recommendations() == {
        "generatedAt": stored["generatedAt"], "tasks": FIVE,
    }
    report = sidebar_panels.recommendations_report()
    assert report == {"generatedAt": int(stored["generatedAt"] * 1000), "tasks": FIVE}
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
    path.unlink()


def test_agent_reads_the_history_and_stores_five_suggestions(
    tmp_path: Path,
    empty_history: None,
) -> None:
    """With the SEA's configuration the model gets recent_tasks, save_recommendations and finish.

    The scripted model reads the history, stores five sentences and
    finishes; the test checks the tools offered, the system prompt, the
    real history in the tool-result message, and the file the sidebar
    reads.
    """
    sea.recommendations_file().unlink(missing_ok=True)
    _persist("Fix the parser", 3 * 3600, result="Fixed parser.py", success=True)
    summary = "<ul>" + "".join(f"<li>{t}</li>" for t in FIVE) + "</ul>"
    script = [
        tool_call_body("recent_tasks", {}, prompt_tokens=500),
        tool_call_body("save_recommendations", {"tasks": FIVE}, prompt_tokens=600),
        finish_body(summary, prompt_tokens=700),
    ]
    run = sea_commands.evaluate_sea([RecommendationsSea()], "now")
    settings = run.settings
    assert run.prompt == "Suggest 5 tasks from the last 24 hours."
    assert run.tools_hook is not None and run.system_prompt_hook is not None
    with serve(script) as (url, requests):
        agent = ChatSorcarAgent("recommendations-sea-test")
        result = agent.run(
            prompt_template=run.prompt,
            model_name=MODEL,
            work_dir=str(tmp_path),
            max_steps=5,
            max_budget=settings["max_budget"],
            model_config={"base_url": url, "api_key": "local"},
            tools_hook=run.tools_hook,
            tool_profile=settings["tool_profile"],
            system_prompt_hook=run.system_prompt_hook,
            web_tools=settings["use_web_tools"],
            use_memory=settings["use_memory"],
            verbose=False,
        )
    parsed = yaml.safe_load(result)
    assert parsed["success"] is True
    assert parsed["summary"] == summary
    assert sea.load_recommendations()["tasks"] == FIVE
    assert sidebar_panels.recommendations_report()["tasks"] == FIVE

    agentic = [r for r in requests if r.get("tools")]
    assert len(agentic) == 3, [list(r) for r in requests]
    for request in agentic:
        names = {t["function"]["name"] for t in request["tools"]}
        assert names == {"finish", "recent_tasks", "save_recommendations"}, names
        system = next(m for m in request["messages"] if m["role"] == "system")
        assert str(system["content"]).startswith(sea.SYSTEM_PROMPT)
    tool_results = [m for m in agentic[1]["messages"] if m["role"] == "tool"]
    assert len(tool_results) == 1
    assert str(tool_results[0]["content"]).startswith(
        "- 3h ago | /home/me/proj | ok | Fix the parser -> Fixed parser.py"
    )
    saved = [m for m in agentic[2]["messages"] if m["role"] == "tool"]
    assert str(saved[-1]["content"]).startswith("Saved 5 recommendations.")
    sea.recommendations_file().unlink(missing_ok=True)
