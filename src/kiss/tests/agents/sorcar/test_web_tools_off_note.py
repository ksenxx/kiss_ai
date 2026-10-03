# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The settings panel's "Use web tools" switch reaches the model's instructions.

With web tools off the browser tools are not built; these tests pin that
the system prompt then also says so (``WEB_TOOLS_OFF_NOTE``), because the
static Web Research rules of ``SYSTEM.md`` would otherwise still order
``go_to_url()`` research the agent has no tool for.  Real ``SorcarAgent``
runs against the scripted local model server; no test doubles.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from kiss.agents.sorcar.sorcar_agent import WEB_TOOLS_OFF_NOTE, SorcarAgent
from kiss.tests.agents.sorcar.local_model_server import MODEL, finish_body, serve

_BROWSER_TOOLS = {"go_to_url", "click", "type_text", "screenshot", "get_page_content"}


def _run(tmp_path: Path, web_tools: bool, tool_profile: str = "") -> dict:
    """Run one scripted step and return the first request the model saw."""
    with serve([finish_body("<p>ok</p>", prompt_tokens=500)]) as (url, requests):
        agent = SorcarAgent("web-tools-note")
        result = agent.run(
            prompt_template="Say ok.",
            model_name=MODEL,
            work_dir=str(tmp_path),
            max_steps=2,
            max_budget=1.0,
            model_config={"base_url": url, "api_key": "local"},
            use_memory=False,
            verbose=False,
            web_tools=web_tools,
            tool_profile=tool_profile,
        )
    assert yaml.safe_load(result)["success"] is True
    return requests[0]


def _system(request: dict) -> str:
    return str(next(m for m in request["messages"] if m["role"] == "system")["content"])


def test_web_tools_off_withholds_browser_tools_and_says_so(tmp_path: Path) -> None:
    request = _run(tmp_path, web_tools=False)
    names = {t["function"]["name"] for t in request["tools"]}
    assert not names & _BROWSER_TOOLS
    assert WEB_TOOLS_OFF_NOTE in _system(request)


def test_web_tools_on_keeps_the_prompt_unchanged(tmp_path: Path) -> None:
    request = _run(tmp_path, web_tools=True)
    assert "# Web tools are off" not in _system(request)
    assert _BROWSER_TOOLS <= {t["function"]["name"] for t in request["tools"]}


def test_restricted_profile_note_covers_web_tools(tmp_path: Path) -> None:
    """A browser-less profile already disclaims browser research; no second note."""
    request = _run(tmp_path, web_tools=False, tool_profile="shell")
    system = _system(request)
    assert "# Restricted tool profile: shell" in system
    assert "# Web tools are off" not in system


def test_review_profile_offers_browser_and_talk(tmp_path: Path) -> None:
    """The ``review`` profile browses and talks; its note lists those tools."""
    request = _run(tmp_path, web_tools=True, tool_profile="review")
    names = {t["function"]["name"] for t in request["tools"]}
    assert _BROWSER_TOOLS | {"talk"} <= names
    assert not names & {"Edit", "Write", "run_agent", "run_parallel"}
    system = _system(request)
    note = system.split("# Restricted tool profile: review", 1)[1]
    assert "go_to_url" in note and "talk" in note
    assert "# Web tools are off" not in system


def test_review_profile_with_web_tools_off_withholds_browser_and_says_so(
    tmp_path: Path,
) -> None:
    """Web tools off beats the profile: no browser tools, none promised, and the
    off-note is added so the Web Research rules are disclaimed."""
    request = _run(tmp_path, web_tools=False, tool_profile="review")
    names = {t["function"]["name"] for t in request["tools"]}
    assert not names & _BROWSER_TOOLS
    assert "talk" in names
    system = _system(request)
    note = system.split("# Restricted tool profile: review", 1)[1]
    assert "go_to_url" not in note.split("# Web tools are off", 1)[0]
    assert WEB_TOOLS_OFF_NOTE in system
