# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the bundled ``/sh`` agent (:mod:`kiss.agents.seas.sh_sea`).

The agent-level tests run a real :class:`ChatSorcarAgent` ReAct loop
against the scripted local chat-completions server
(:mod:`kiss.tests.agents.sorcar.local_model_server`) configured
exactly as the daemon configures it from the SEA's getters: the
``bash`` tool profile and the SEA's system prompt.  The only replaced
boundary is the LLM endpoint; the Bash tool really runs the command
in the work directory and its output really flows through the tool
result into the finish summary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from kiss.agents.seas import sh_sea
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import TOOL_PROFILES
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


def _system_message(request: dict[str, Any]) -> str:
    """Return the system message text of one chat-completions request."""
    return str(next(m for m in request["messages"] if m["role"] == "system")["content"])


def test_sea_getters_follow_the_user_contract() -> None:
    """The SEA's getters pin the run: Bash-only, no worktree, no extras."""
    # The prompt's wording is free to evolve; the contract is that the
    # agent runs the user's command through Bash and hands the raw
    # output to ``finish``.
    prompt = sh_sea.system_prompt()
    assert prompt == sh_sea.SYSTEM_PROMPT
    assert "call the Bash tool with the user's command exactly as written" in prompt
    assert "call the `finish` tool" in prompt
    assert "must never be empty" in prompt
    assert sh_sea.tool_profile() == "bash"
    assert TOOL_PROFILES["bash"] == frozenset({"Bash"})
    assert sh_sea.use_worktree() is False
    assert sh_sea.auto_commit() is False
    assert sh_sea.classify_tasks() is False
    assert sh_sea.is_parallel() is False
    assert sh_sea.use_web_tools() is False
    assert sh_sea.use_memory() is False


def test_slash_sh_resolves_to_the_bundled_sea() -> None:
    """``/sh <command>`` is rewritten into a ``run_agent`` directive on this file."""
    assert sea_commands.get_command("sh") == _SEA_PATH
    rewritten = sea_commands.rewrite_prompt_if_command("/sh git status --short")
    assert rewritten is not None
    prompt, path = rewritten
    assert path == _SEA_PATH
    assert f'agent = "{_SEA_PATH}"' in prompt
    assert prompt.endswith("TASK TEXT FOR run_agent:\ngit status --short")


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
            "Bash", {"command": command, "description": "run the user's command"},
            prompt_tokens=500,
        ),
        finish_body("<pre>sh-sea-output 42</pre>", prompt_tokens=600),
    ]
    with serve(script) as (url, requests):
        agent = ChatSorcarAgent("sh-sea-test")
        result = agent.run(
            prompt_template=command,
            model_name=MODEL,
            work_dir=str(tmp_path),
            max_steps=4,
            max_budget=1.0,
            model_config={"base_url": url, "api_key": "local"},
            base_system_prompt=sh_sea.system_prompt(),
            tool_profile=sh_sea.tool_profile(),
            web_tools=sh_sea.use_web_tools(),
            use_memory=sh_sea.use_memory(),
            is_parallel=sh_sea.is_parallel(),
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
        system = _system_message(request)
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
    assert "# Restricted tool profile" not in _system_message(_tool_requests(requests)[0])


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
        "model_name": MODEL, "work_dir": str(tmp_path), "max_steps": 3,
        "max_budget": 1.0, "web_tools": False, "use_memory": False, "verbose": False,
    }
    with serve(script) as (url, requests):
        agent.run(
            prompt_template="Say done.", tool_profile="bash",
            model_config={"base_url": url, "api_key": "local"}, **common,
        )
        first = {t["function"]["name"] for t in _tool_requests(requests)[0]["tools"]}
    assert first == {"Bash", "finish"}
    with serve(script) as (url, requests):
        agent.run(
            prompt_template="Say done again.",
            model_config={"base_url": url, "api_key": "local"}, **common,
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
    assert "# Restricted tool profile" not in _system_message(request)
