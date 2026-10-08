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
from typing import Any

import yaml

from kiss.agents.seas.base.base_sea import WorkerSea
from kiss.agents.seas.sh import sh_sea
from kiss.agents.seas.sh.sh_sea import ShSea
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sea_settings import resolve_settings
from kiss.agents.sorcar.sorcar_agent import TOOL_PROFILES
from kiss.tests.agents.seas.sea_contract import assert_no_removed_getters, system_message
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)

_SEA_PATH = Path(sh_sea.__file__).resolve()


def _tool_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the agentic requests (those carrying a tool list)."""
    return [r for r in requests if r.get("tools")]


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
    # ``WorkerSea`` turns worktree, auto-commit, classifier, browser and
    # memory off under the class's own keys; ``system_prompt`` is a hook
    # the daemon applies, so the resolved settings do not carry its text.
    assert resolve_settings(sea.settings({})) == sea.settings({})
    assert sea_commands.base_settings([sea]) == {
        "tool_profile": "bash",
        "locked": ["tool_profile"],
        "use_worktree": False,
        "auto_commit": False,
        "auto_classify": False,
        "use_web_tools": False,
        "use_memory": False,
    }
    assert_no_removed_getters(sh_sea)


def test_slash_sh_resolves_to_the_bundled_sea() -> None:
    """``/sh <command>`` resolves to the command text and this file (run directly)."""
    assert sea_commands.get_command("sh") == _SEA_PATH
    hit = sea_commands.slash_command_task("/sh git status --short")
    assert hit is not None
    task_text, path = hit
    assert path == _SEA_PATH
    assert task_text == "git status --short"
    assert sea_commands.sea_settings(path) == sea_commands.base_settings([ShSea()])
    assert type(sea_commands.load_sea(path)).__name__ == "ShSea"


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
    settings = run.settings
    assert run.prompt == command
    # ``tools`` is not overridden: the staged tools hook is the identity.
    assert run.tools_hook([print]) == [print]
    with serve(script) as (url, requests):
        agent = ChatSorcarAgent("sh-sea-test")
        result = agent.run(
            prompt_template=run.prompt,
            model_name=MODEL,
            work_dir=str(tmp_path),
            max_steps=4,
            max_budget=1.0,
            model_config={"base_url": url, "api_key": "local"},
            system_prompt_hook=run.system_prompt_hook,
            tool_profile=settings["tool_profile"],
            web_tools=settings["use_web_tools"],
            use_memory=settings["use_memory"],
            verbose=False,
        )
    parsed = yaml.safe_load(result)
    assert parsed["success"] is True
    assert "sh-sea-output 42" in parsed["summary"]

    agentic = _tool_requests(requests)
    assert len(agentic) == 2, [list(r) for r in requests]
    for request in agentic:
        names = {t["function"]["name"] for t in request["tools"]}
        assert names == {"Bash", "finish"}, names
        system = system_message(request)
        assert system.startswith(sh_sea.SYSTEM_PROMPT), system[:200]
        assert "# Restricted tool profile: bash" in system
        assert "Bash" in system.split("# Restricted tool profile: bash", 1)[1]
    # The second step carries the Bash tool result: the command really ran.
    tool_results = [m for m in agentic[1]["messages"] if m["role"] == "tool"]
    assert len(tool_results) == 1, agentic[1]["messages"]
    assert "sh-sea-output 42" in str(tool_results[0]["content"])


def test_explicit_full_profile_offers_the_whole_toolset(tmp_path: Path) -> None:
    """``tool_profile="full"`` is accepted and offers more than Bash+finish."""
    script = [finish_body("<p>done</p>", prompt_tokens=500)]
    with serve(script) as (url, requests):
        agent = ChatSorcarAgent("full-profile-test")
        result = agent.run(
            prompt_template="Say done.",
            model_name=MODEL,
            work_dir=str(tmp_path),
            max_steps=3,
            max_budget=1.0,
            model_config={"base_url": url, "api_key": "local"},
            tool_profile="full",
            web_tools=False,
            use_memory=False,
            verbose=False,
        )
    assert yaml.safe_load(result)["success"] is True
    names = {t["function"]["name"] for t in _tool_requests(requests)[0]["tools"]}
    assert {"Bash", "Read", "Edit", "Write", "finish"} <= names
    assert "# Restricted tool profile" not in system_message(_tool_requests(requests)[0])


def test_unknown_profile_is_rejected_before_the_model_is_called(tmp_path: Path) -> None:
    """A ``tool_profile`` that is not a ``TOOL_PROFILES`` key fails loudly."""
    with serve([finish_body("<p>never</p>", prompt_tokens=500)]) as (url, requests):
        agent = ChatSorcarAgent("bad-profile-test")
        try:
            agent.run(
                prompt_template="Say done.",
                model_name=MODEL,
                work_dir=str(tmp_path),
                max_steps=3,
                max_budget=1.0,
                model_config={"base_url": url, "api_key": "local"},
                tool_profile="bogus",
                web_tools=False,
                use_memory=False,
                verbose=False,
            )
        except ValueError as exc:
            assert "tool_profile must be one of" in str(exc)
            assert "'bogus'" in str(exc)
        else:  # pragma: no cover — the assertion is the test
            raise AssertionError("an unknown tool_profile must raise ValueError")
    assert requests == []


def test_profile_does_not_leak_into_the_next_run_of_the_same_agent(tmp_path: Path) -> None:
    """A tab's agent is reused across runs: a later run without a profile gets the full set."""
    script = [finish_body("<p>done</p>", prompt_tokens=500)]
    agent = ChatSorcarAgent("profile-reset-test")
    common: dict[str, Any] = {
        "model_name": MODEL,
        "work_dir": str(tmp_path),
        "max_steps": 3,
        "max_budget": 1.0,
        "web_tools": False,
        "use_memory": False,
        "verbose": False,
    }
    with serve(script) as (url, requests):
        agent.run(
            prompt_template="Say done.",
            tool_profile="bash",
            model_config={"base_url": url, "api_key": "local"},
            **common,
        )
        first = {t["function"]["name"] for t in _tool_requests(requests)[0]["tools"]}
    assert first == {"Bash", "finish"}
    with serve(script) as (url, requests):
        agent.run(
            prompt_template="Say done again.",
            model_config={"base_url": url, "api_key": "local"},
            **common,
        )
        second = {t["function"]["name"] for t in _tool_requests(requests)[0]["tools"]}
    assert {"Bash", "Read", "Edit", "Write", "finish"} <= second


def test_profile_without_basic_tools_offers_finish_only_and_no_note(tmp_path: Path) -> None:
    """``append_basic_tools=False`` builds no toolset to cut down: finish only, no profile note."""
    script = [finish_body("<p>done</p>", prompt_tokens=500)]
    with serve(script) as (url, requests):
        agent = ChatSorcarAgent("no-basic-tools-profile-test")
        result = agent.run(
            prompt_template="Say done.",
            model_name=MODEL,
            work_dir=str(tmp_path),
            max_steps=3,
            max_budget=1.0,
            model_config={"base_url": url, "api_key": "local"},
            tool_profile="bash",
            append_basic_tools=False,
            web_tools=False,
            use_memory=False,
            verbose=False,
        )
    assert yaml.safe_load(result)["success"] is True
    request = _tool_requests(requests)[0]
    assert {t["function"]["name"] for t in request["tools"]} == {"finish"}
    assert "# Restricted tool profile" not in system_message(request)
