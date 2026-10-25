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
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.sea_settings import resolve_settings
from kiss.tests.agents.seas.conftest import run_sea
from kiss.tests.agents.seas.sea_contract import (
    assert_no_removed_getters,
    system_message,
    tool_names,
    tool_results,
)
from kiss.tests.agents.sorcar.local_model_server import (
    finish_body,
    tool_call_body,
)

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


def test_stored_recommendations_reads_the_file_only(recommendations_path: Path) -> None:
    """The tool reports the stored suggestions and their age; never more."""
    path = recommendations_path
    assert sea.stored_recommendations() == "No recommendations stored yet."
    path.write_text(json.dumps({"generatedAt": time.time() - 23 * 60, "tasks": FIVE}))
    lines = sea.stored_recommendations().splitlines()
    assert lines[0] == "Suggested 23 minutes ago:"
    assert lines[1:] == [f"{i}. {task}" for i, task in enumerate(FIVE, 1)]
    path.write_text(json.dumps({"generatedAt": time.time() - 5 * 3600, "tasks": FIVE[:2]}))
    assert sea.stored_recommendations().startswith("Suggested 5 hours ago:\n1. ")
    path.write_text(json.dumps({"generatedAt": time.time() - 61, "tasks": FIVE[:1]}))
    assert sea.stored_recommendations().startswith("Suggested 1 minute ago:\n1. ")


def test_agent_greets_with_the_stored_suggestions(
    tmp_path: Path, recommendations_path: Path,
) -> None:
    """With the SEA's configuration the model gets only stored_recommendations and finish.

    The scripted model reads the stored file and finishes with the
    greeting; the test checks the tools offered, the system prompt and
    that the stored sentences reached the model unchanged.
    """
    recommendations_path.write_text(
        json.dumps({"generatedAt": time.time() - 3600, "tasks": FIVE}))
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
    parsed, agentic = run_sea(run, script, tmp_path, max_steps=4)
    assert parsed["success"] is True
    assert parsed["summary"] == greeting
    assert len(agentic) == 2
    for request in agentic:
        assert tool_names(request) == {"finish", "stored_recommendations"}
        assert system_message(request).startswith(sea.SYSTEM_PROMPT)
    (tool_result,) = tool_results(agentic[1])
    assert tool_result.startswith("Suggested 1 hour ago:\n1. " + FIVE[0])
