# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The ``tool_profile`` run parameter on a plain :class:`BaseSea` run.

Real :class:`~kiss.agents.sorcar.chat_sorcar_agent.ChatSorcarAgent`
runs against the scripted local chat-completions server; the tests
check the toolset and system prompt the model is offered.  Which
profile builds which toolset is
:mod:`kiss.tests.agents.sorcar.test_tool_profiles`'s concern; the
rejection of an unknown profile name is pinned there and in
:mod:`kiss.tests.server.test_tool_profile_run_param`.
"""

from __future__ import annotations

from pathlib import Path

from kiss.agents.seas.base.base_sea import BaseSea
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.tests.agents.seas.conftest import run_sea
from kiss.tests.agents.seas.sea_contract import system_message, tool_names
from kiss.tests.agents.sorcar.local_model_server import finish_body

_FILE_TOOLS = {"Bash", "Read", "Edit", "Write", "finish"}


def _plain_run() -> sea_commands.SeaRun:
    """A fresh run with no SEA behaviour: identity hooks, no settings, the task as prompt."""
    return sea_commands.evaluate_sea([BaseSea()], "Say done.")


def test_explicit_full_profile_offers_the_whole_toolset(tmp_path: Path) -> None:
    """``tool_profile="full"`` is accepted and offers more than Bash+finish."""
    script = [finish_body("<p>done</p>", prompt_tokens=500)]
    parsed, agentic = run_sea(_plain_run(), script, tmp_path, max_steps=3, tool_profile="full")
    assert parsed["success"] is True
    assert _FILE_TOOLS <= tool_names(agentic[0])
    assert "# Restricted tool profile" not in system_message(agentic[0])


def test_profile_does_not_leak_into_the_next_run_of_the_same_agent(tmp_path: Path) -> None:
    """A tab's agent is reused across runs: a later run without a profile gets the full set."""
    script = [finish_body("<p>done</p>", prompt_tokens=500)]
    agent = ChatSorcarAgent("profile-reset-test")
    _parsed, first = run_sea(
        _plain_run(), script, tmp_path, agent=agent, max_steps=3, tool_profile="bash"
    )
    assert tool_names(first[0]) == {"Bash", "finish"}
    again = sea_commands.evaluate_sea([BaseSea()], "Say done again.")
    _parsed, second = run_sea(again, script, tmp_path, agent=agent, max_steps=3)
    assert _FILE_TOOLS <= tool_names(second[0])


def test_profile_without_basic_tools_offers_finish_only_and_no_note(tmp_path: Path) -> None:
    """``append_basic_tools=False`` builds no toolset to cut down: finish only, no profile note."""
    script = [finish_body("<p>done</p>", prompt_tokens=500)]
    parsed, agentic = run_sea(
        _plain_run(),
        script,
        tmp_path,
        max_steps=3,
        tool_profile="bash",
        append_basic_tools=False,
    )
    assert parsed["success"] is True
    assert tool_names(agentic[0]) == {"finish"}
    assert "# Restricted tool profile" not in system_message(agentic[0])
