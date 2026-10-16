# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the bundled morning-greeting agent
(:mod:`kiss.agents.seas.good_morning.good_morning_sea`).

``stored_recommendations`` is exercised against the real
``<home>/recommendations.json`` (``KISS_HOME`` is a temporary directory,
see ``conftest.py``) and the agent-level test runs a real
:class:`ChatSorcarAgent` ReAct loop against the scripted local
chat-completions server configured from the SEA's methods: the stored
suggestions flow through the tool result into the greeting, and nothing
is generated.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from kiss.agents.seas.base.base_sea import WORKER_DEFAULTS, WorkerSea
from kiss.agents.seas.good_morning import good_morning_sea as sea
from kiss.agents.seas.good_morning.good_morning_sea import GoodMorningSea
from kiss.agents.seas.recommendations import recommendations_sea
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.sea_settings import resolve_settings
from kiss.tests.agents.seas.conftest import run_sea
from kiss.tests.agents.seas.sea_contract import assert_no_removed_getters
from kiss.tests.agents.sorcar.local_model_server import (
    finish_body,
    tool_call_body,
)

_SEA_PATH = Path(sea.__file__).resolve()

FIVE = [
    "Fix the failing parser test in parser.py",
    "Add a changelog entry for /recommendations",
    "Run the full test suite in /home/me/proj",
    "Review the open pull request on the cron scheduler",
    "Document the sidebar panels in README.md",
]


def test_sea_methods_and_prompt_follow_the_contract() -> None:
    """The SEA pins its run: the reader tool, no built-in tools, 50 cents."""
    agent = GoodMorningSea()
    assert agent.system_prompt("ASSEMBLED") == sea.SYSTEM_PROMPT
    assert "stored_recommendations" in sea.SYSTEM_PROMPT
    assert "Never invent" in sea.SYSTEM_PROMPT
    assert agent.prompt("now") == agent.prompt("") == sea.PROMPT
    assert agent.tools([]) == [sea.stored_recommendations]
    assert agent.tools([print]) == [print, sea.stored_recommendations]
    assert isinstance(agent, WorkerSea)
    assert agent.settings({}) == {
        "tool_profile": "none",
        "max_budget": 0.5,
        "timeout": 300,
        "locked": ["tool_profile"],
    }
    assert sea_commands.base_settings([agent]) == resolve_settings(
        {**WORKER_DEFAULTS, **agent.settings({})}
    )
    assert "never generated on the fly" in agent.description()
    assert_no_removed_getters(sea)


def test_slash_good_morning_resolves_to_the_bundled_sea() -> None:
    """``/good_morning now`` (the morning cron job's prompt) resolves to this SEA."""
    assert sea_commands.get_command("good_morning") == _SEA_PATH
    hit = sea_commands.slash_command_task("/good_morning now")
    assert hit is not None and hit == ("now", _SEA_PATH)
    assert sea_commands.help_text_if_command("/good_morning help") == (
        GoodMorningSea().description()
    )


def test_stored_recommendations_reads_the_file_only() -> None:
    """The tool reports the stored suggestions and their age; never more."""
    path = recommendations_sea.recommendations_file()
    path.unlink(missing_ok=True)
    assert sea.stored_recommendations() == "No recommendations stored yet."
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"generatedAt": time.time() - 23 * 60, "tasks": FIVE}))
    lines = sea.stored_recommendations().splitlines()
    assert lines[0] == "Suggested 23 minutes ago:"
    assert lines[1:] == [f"{i}. {task}" for i, task in enumerate(FIVE, 1)]
    path.write_text(json.dumps({"generatedAt": time.time() - 5 * 3600, "tasks": FIVE[:2]}))
    assert sea.stored_recommendations().startswith("Suggested 5 hours ago:\n1. ")
    path.write_text(json.dumps({"generatedAt": time.time() - 61, "tasks": FIVE[:1]}))
    assert sea.stored_recommendations().startswith("Suggested 1 minute ago:\n1. ")
    path.unlink()


def test_agent_greets_with_the_stored_suggestions(tmp_path: Path) -> None:
    """With the SEA's configuration the model gets only stored_recommendations and finish.

    The scripted model reads the stored file and finishes with the
    greeting; the test checks the tools offered, the system prompt and
    that the stored sentences reached the model unchanged.
    """
    path = recommendations_sea.recommendations_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"generatedAt": time.time() - 3600, "tasks": FIVE}))
    greeting = (
        "<p>Good morning!</p><ol>" + "".join(f"<li>{t}</li>" for t in FIVE)
        + "</ol><p>Which of these, or what else, should I do today?</p>"
    )
    script = [
        tool_call_body("stored_recommendations", {}, prompt_tokens=300),
        finish_body(greeting, prompt_tokens=400),
    ]
    run = sea_commands.evaluate_sea([GoodMorningSea()], "now")
    assert run.prompt == sea.PROMPT
    assert run.tools_hook is not None and run.system_prompt_hook is not None
    try:
        parsed, agentic = run_sea(run, script, tmp_path, max_steps=4)
    finally:
        path.unlink(missing_ok=True)
    assert parsed["success"] is True
    assert parsed["summary"] == greeting
    assert len(agentic) == 2
    for request in agentic:
        names = {t["function"]["name"] for t in request["tools"]}
        assert names == {"finish", "stored_recommendations"}, names
        system = next(m for m in request["messages"] if m["role"] == "system")
        assert str(system["content"]).startswith(sea.SYSTEM_PROMPT)
    (tool_result,) = [m for m in agentic[1]["messages"] if m["role"] == "tool"]
    assert str(tool_result["content"]).startswith("Suggested 1 hour ago:\n1. " + FIVE[0])
