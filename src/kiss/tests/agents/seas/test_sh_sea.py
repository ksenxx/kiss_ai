# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the bundled ``/sh`` agent (:mod:`kiss.agents.seas.sh.sh_sea`).

The agent-level tests run a real :class:`ChatSorcarAgent` ReAct loop
against the scripted local chat-completions server
(:mod:`kiss.tests.agents.sorcar.local_model_server`) configured
exactly as the daemon configures it from the SEA's methods: the
``bash`` tool profile and the SEA's system prompt.  The only replaced
boundary is the LLM endpoint; the Bash tool really runs the command
in the work directory and its output really flows through the tool
result into the finish summary.
"""

from __future__ import annotations

from pathlib import Path

from kiss.agents.seas.base.base_sea import WorkerSea
from kiss.agents.seas.sh import sh_sea
from kiss.agents.seas.sh.sh_sea import ShSea
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.sea_settings import resolve_settings
from kiss.agents.sorcar.sorcar_agent import TOOL_PROFILES
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

_SEA_PATH = Path(sh_sea.__file__).resolve()


def test_sea_methods_follow_the_user_contract() -> None:
    """The SEA's methods pin the run: Bash-only, no worktree, no extras."""
    # The prompt's wording is free to evolve; the contract is that the
    # agent runs the user's command through Bash and hands the raw
    # output to ``finish``.  ``system_prompt`` REPLACES the assembled
    # prompt (the default Sorcar prompt would drown the three rules).
    sea = ShSea()
    prompt = sea.system_prompt("ASSEMBLED")
    assert prompt == sh_sea.SYSTEM_PROMPT
    assert "ASSEMBLED" not in prompt
    assert "call the Bash tool with the user's command exactly as written" in prompt
    assert "call the `finish` tool" in prompt
    assert "must never be empty" in prompt
    assert isinstance(sea, WorkerSea)
    assert sea.settings({}) == {"tool_profile": "bash", "locked": ["tool_profile"]}
    assert sea.settings({"model": "m"}) == {
        "model": "m",
        "tool_profile": "bash",
        "locked": ["tool_profile"],
    }
    assert TOOL_PROFILES["bash"] == frozenset({"Bash"})
    # ``WorkerSea`` turns classifier, browser and memory off under the
    # class's own keys; ``system_prompt`` is a hook
    # the daemon applies, so the resolved settings do not carry its text.
    assert resolve_settings(sea.settings({})) == sea.settings({})
    assert sea_commands.base_settings([sea]) == {
        "tool_profile": "bash",
        "locked": ["tool_profile"],
        "auto_classify": False,
        "use_web_tools": False,
        "use_memory": False,
    }
    assert_no_removed_getters(sh_sea)


def test_slash_sh_resolves_to_the_bundled_sea() -> None:
    """``/sh <command>`` resolves to the command text and this file (run directly)."""
    assert sea_commands.get_command("sh") == _SEA_PATH
    assert sea_commands.slash_command_task("/sh git status --short") == (
        "git status --short", _SEA_PATH)
    assert sea_commands.sea_settings(_SEA_PATH) == sea_commands.base_settings([ShSea()])
    assert type(sea_commands.load_sea(_SEA_PATH)).__name__ == "ShSea"


def test_bash_profile_runs_the_command_and_returns_its_output(tmp_path: Path) -> None:
    """With the SEA's configuration the model sees Bash+finish only and gets the real output.

    The scripted model calls ``Bash`` with the prompt's command and then
    finishes with the tool result it was given.  The test checks the
    tool schemas offered on every step, the system prompt, the real
    command output in the tool-result message, and the final summary.
    """
    command = "printf 'sh-sea-output %s' 42"
    script = [
        tool_call_body(
            "Bash",
            {"command": command, "description": "run the user's command"},
            prompt_tokens=500,
        ),
        finish_body("<pre>sh-sea-output 42</pre>", prompt_tokens=600),
    ]
    run = sea_commands.evaluate_sea([ShSea()], command)
    assert run.prompt == command
    # ``tools`` is not overridden: the staged tools hook is the identity.
    assert run.tools_hook([print]) == [print]
    parsed, agentic = run_sea(run, script, tmp_path, max_steps=4)
    assert parsed["success"] is True
    assert "sh-sea-output 42" in parsed["summary"]

    assert len(agentic) == 2
    for request in agentic:
        assert tool_names(request) == {"Bash", "finish"}
        system = system_message(request)
        assert system.startswith(sh_sea.SYSTEM_PROMPT), system[:200]
        assert "# Restricted tool profile: bash" in system
        assert "Bash" in system.split("# Restricted tool profile: bash", 1)[1]
    # The second step carries the Bash tool result: the command really ran.
    results = tool_results(agentic[1])
    assert len(results) == 1, agentic[1]["messages"]
    assert "sh-sea-output 42" in results[0]

