# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the bundled ``/devtools`` agent (``seas/devtools/devtools_sea.py``).

The SEA replaced the bundled ``browser-devtools`` skill: it appends the
DevTools workflow to the system prompt and switches the browser tools on,
so the tests check that the workflow names the six DevTools tools and
their real arguments and actions, the slash command resolves to this
file, the settings turn the browser on and pin nothing else, the
daemon-side loader stages the append as a system-prompt hook, and a real
:class:`ChatSorcarAgent` run configured that way (against the scripted
local chat-completions server) sends the default prompt with the workflow
at the end and offers the DevTools tools even though the run's own
``web_tools`` default is off.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.seas.devtools import devtools_sea
from kiss.agents.seas.devtools.devtools_sea import DevtoolsSea
from kiss.agents.sorcar import sea_commands, web_use_tool
from kiss.agents.sorcar.agent_dispatch import resolve_agent
from kiss.agents.sorcar.sea_apply import apply_sea
from kiss.core.base import SYSTEM_PROMPT, SYSTEM_PROMPT_LITE
from kiss.core.tool_verdict import ALLOW
from kiss.tests.agents.seas.conftest import run_sea
from kiss.tests.agents.seas.sea_contract import (
    assert_no_removed_getters,
    system_message,
    tool_names,
)
from kiss.tests.agents.sorcar.local_model_server import finish_body

_SEA_PATH = Path(devtools_sea.__file__).resolve()
_HEADING = devtools_sea.SECTION_HEADING
_DEVTOOLS_TOOLS = {
    "evaluate_script",
    "list_console_messages",
    "list_network_requests",
    "wait_for",
    "attach_to_chrome",
    "use_my_chrome",
}


def test_the_workflow_names_the_real_tools() -> None:
    """The SEA's text names the tools, arguments and actions that exist."""
    text = devtools_sea.SYSTEM_PROMPT
    assert text.startswith(_HEADING + "\n\n")
    assert text.count("\n## ") == 0
    for name in _DEVTOOLS_TOOLS:
        assert name in text, name
    # Every argument the workflow shows is a real parameter of the tool.
    for name, shown in (
        ("wait_for", {"text", "timeout", "disappear"}),
        ("list_console_messages", {"clear"}),
        ("list_network_requests", {"url_filter", "clear"}),
        ("attach_to_chrome", {"cdp_url"}),
    ):
        params = set(inspect.signature(getattr(web_use_tool.WebUseTool, name)).parameters)
        assert shown <= params, (name, shown - params)
    assert 'wait_for("Saved", timeout=10)' in text
    assert 'wait_for("Loading", disappear=True)' in text
    assert 'list_network_requests(url_filter="/api", clear=True)' in text
    assert "list_console_messages(clear=True)" in text
    # The ``use_my_chrome`` actions listed are the ones the tool accepts: an
    # unknown action is refused, before any bridge starts, with the valid ones.
    tool = web_use_tool.WebUseTool(user_data_dir=None)
    refusal = tool.use_my_chrome("no-such-action")
    tool.close()
    assert refusal.startswith("Error: unknown action 'no-such-action'. Use one of: "), refusal
    accepted = set(refusal.split("Use one of: ", 1)[1].rstrip(".").split(", "))
    for action in (
        "list_tabs", "open_tab", "navigate", "snapshot", "click", "type_text",
        "evaluate", "screenshot", "activate_tab", "close_tab",
    ):
        assert action in text and action in accepted, (action, accepted)
    # The JS recipes rendered with single braces (the text is an f-string).
    assert "{{" not in text and "}}" not in text
    assert "() => { const b = document.querySelector('button.submit');" in text
    assert "Never exfiltrate secrets" in text
    assert "confirm before anything\nirreversible" in text
    assert "skill" not in text.lower()
    assert "\u2014" not in text


def test_system_prompt_appends_the_workflow_once_at_the_end() -> None:
    """The default prompt is kept whole, in order, and ends with the workflow."""
    out = DevtoolsSea().system_prompt(SYSTEM_PROMPT)
    assert out == SYSTEM_PROMPT.rstrip("\n") + "\n\n" + devtools_sea.SYSTEM_PROMPT
    assert out.count(_HEADING) == 1
    assert out.lstrip().startswith("<identity>")
    # Appended after the caller's additions too, and idempotent.
    out = DevtoolsSea().system_prompt(SYSTEM_PROMPT + "\n\nCALLER\n")
    assert out.endswith("CALLER\n\n" + devtools_sea.SYSTEM_PROMPT)
    assert DevtoolsSea().system_prompt(out) == out
    assert DevtoolsSea().system_prompt(SYSTEM_PROMPT_LITE).count(_HEADING) == 1
    assert DevtoolsSea().system_prompt("BASE") == "BASE\n\n" + devtools_sea.SYSTEM_PROMPT
    # Neither default prompt carries the workflow of its own.
    for prompt in (SYSTEM_PROMPT, SYSTEM_PROMPT_LITE):
        assert _HEADING not in prompt


@pytest.mark.usefixtures("reset_registry")
def test_description_is_one_sentence_naming_the_command() -> None:
    """``/devtools help`` returns the description without running the SEA."""
    text = DevtoolsSea().description()
    assert text.count(". ") == 0 and text.endswith(".")
    assert "/devtools" in text
    assert sea_commands.help_text_if_command("/devtools help") == text.strip()
    assert sea_commands.get_command("devtools") == _SEA_PATH
    assert sea_commands.slash_command("/devtools why is Save inert") == (
        "why is Save inert", _SEA_PATH,
    )
    assert resolve_agent("devtools", "") == (str(_SEA_PATH), "devtools")


def test_settings_switch_the_browser_on_and_pin_nothing_else() -> None:
    """The run keeps the tab's model and toolset; only ``use_web_tools`` is forced on."""
    assert DevtoolsSea().settings({}) == {"use_web_tools": True}
    assert DevtoolsSea().settings({"use_web_tools": False, "max_budget": 2.0}) == {
        "use_web_tools": True, "max_budget": 2.0,
    }
    assert sea_commands.sea_settings(_SEA_PATH) == {"use_web_tools": True}
    assert DevtoolsSea().register_as_model() is False
    assert DevtoolsSea().tools([print]) == [print]
    assert DevtoolsSea().prompt("the task") == "the task"
    assert_no_removed_getters(devtools_sea)


def test_loader_stages_the_append_as_a_system_prompt_hook() -> None:
    """The daemon-side loader turns the browser on and stages the prompt hook, nothing else."""
    cmd: dict[str, Any] = {"seaPath": str(_SEA_PATH), "appendToSystemPrompt": "CALLER"}
    assert apply_sea(cmd) == {"useWebTools"}
    assert cmd["useWebTools"] is True
    assert cmd["appendToSystemPrompt"] == "CALLER"
    assert cmd["systemPromptHook"]("BASE\n\nCALLER") == (
        "BASE\n\nCALLER\n\n" + devtools_sea.SYSTEM_PROMPT
    )
    assert "prompt" not in cmd and "model" not in cmd and "toolProfile" not in cmd
    assert cmd["toolsHook"]([print]) == [print]
    assert cmd["llmCallHook"]([{"role": "user", "content": "x"}]) == [
        {"role": "user", "content": "x"}
    ]
    assert cmd["toolCallHook"]("Bash", {"command": "ls"}) == ALLOW


def test_agent_run_ends_the_default_prompt_with_the_workflow_and_offers_devtools(
    tmp_path: Path,
) -> None:
    """A run configured as the daemon configures it sends the default prompt plus the workflow.

    The scripted model finishes at once; the test checks the real request:
    the default Sorcar prompt (``<identity>``) comes first, the workflow
    follows its last section (``## Sorcar repo specific``) and only the
    agent's run-time blocks come after it, and the six DevTools tools are
    offered although :func:`run_sea` leaves ``web_tools`` off unless the
    SEA's settings turn it on.
    """
    run = sea_commands.evaluate_sea([DevtoolsSea()], "The Save button does nothing; find out why.")
    assert run.settings == {"use_web_tools": True}
    parsed, agentic = run_sea(
        run,
        [finish_body("<p>POST /api/settings returned 403.</p>", prompt_tokens=500)],
        tmp_path,
        max_steps=3,
        is_parallel=False,
    )
    assert parsed["success"] is True
    assert parsed["summary"] == "<p>POST /api/settings returned 403.</p>"

    assert len(agentic) == 1, agentic
    system = system_message(agentic[0])
    assert system.lstrip().startswith("<identity>"), system[:200]
    assert system.count(_HEADING) == 1
    assert devtools_sea.SYSTEM_PROMPT.rstrip("\n") in system
    assert system.index("## Sorcar repo specific") < system.index(_HEADING)
    after = system[system.index(_HEADING) + len(devtools_sea.SYSTEM_PROMPT.rstrip("\n")) :]
    assert "## " not in after, after
    assert "# MOST IMPORTANT INSTRUCTIONS" in after and "# Task Settings" in after
    names = tool_names(agentic[0])
    assert _DEVTOOLS_TOOLS | {"go_to_url", "click", "Bash", "Read", "finish"} <= names, names
    assert "# Restricted tool profile" not in system
