# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``plain_io`` sends a SEA's system prompt to the model as it is.

A SEA that must control the exact prompt (an RL environment, a headless
evaluation) sets ``model_config["plain_io"]``.  These tests pin that the
system message on the wire is then the SEA's ``system_prompt`` return
byte for byte, with none of the trailers ``SorcarAgent.run`` and
``RelentlessAgent.perform_task`` otherwise append after it — the
``# Restricted tool profile`` note, the ``# Web tools are off`` note, the
editor-file line, the ``# MOST IMPORTANT INSTRUCTIONS`` block (work dir,
PID), the ``# Task Settings`` section and the user's ``AGENTS.md`` — and
that the same run without ``plain_io`` reads every one of them.  Real
``ChatSorcarAgent`` runs against the scripted local model server, the SEA
evaluated as the daemon evaluates a SEA run; no test doubles.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.seas.base.base_sea import BaseSea
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sea_commands import evaluate_sea
from kiss.agents.sorcar.sorcar_agent import WEB_TOOLS_OFF_NOTE
from kiss.core import config as config_module
from kiss.tests.agents.seas.sea_contract import system_message, tool_results
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)

PROMPT = (
    "You are a guess-the-number environment.\n"
    "Reply with exactly one tool call per turn; the observation is the tool's result.\n"
)
AGENTS_MD = "Standing instruction from the user's AGENTS.md: answer in haiku."
EDITOR_FILE = "/somewhere/open.py"


class ExactPromptSea(BaseSea):
    """A SEA whose ``system_prompt`` replaces the composed prompt whole."""

    def system_prompt(self, system_prompt: str) -> str:
        return PROMPT


@pytest.fixture
def agents_md() -> Iterator[Path]:
    """The user's ``AGENTS.md`` under the test ``KISS_HOME``, present for the test only."""
    path = config_module.agents_md_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(AGENTS_MD + "\n", encoding="utf-8")
    yield path
    path.unlink(missing_ok=True)


def _run(tmp_path: Path, plain_io: bool, tool_profile: str) -> list[dict[str, Any]]:
    """Run the SEA for two scripted steps; return the requests that offered tools."""
    run = evaluate_sea([ExactPromptSea()], "Find the secret number.")
    script = [
        tool_call_body("Bash", {"command": "echo 42", "description": "probe"}, 1000),
        finish_body("<p>42</p>", prompt_tokens=1200),
    ]
    model_config: dict[str, Any] = {"api_key": "local", "plain_io": plain_io}
    with serve(script) as (url, requests):
        ChatSorcarAgent(f"exact-prompt-{tool_profile}-{plain_io}").run(
            prompt_template=run.prompt,
            model_name=MODEL,
            work_dir=str(tmp_path),
            max_steps=4,
            max_budget=1.0,
            model_config=model_config | {"base_url": url},
            system_prompt_hook=run.system_prompt_hook,
            tools_hook=run.tools_hook,
            tool_call_hook=run.tool_call_hook,
            llm_call_hook=run.llm_call_hook,
            llm_response_hook=run.llm_response_hook,
            tool_profile=tool_profile,
            web_tools=False,
            use_memory=False,
            current_editor_file=EDITOR_FILE,
            verbose=False,
        )
    agentic = [r for r in requests if r.get("tools")]
    assert len(agentic) == 2, "the Bash step, then the finish"
    return agentic


# ``shell+browser`` takes the restricted-profile branch (and, with web tools
# off and the browser in the profile, the web-tools note inside it); ``full``
# takes the full-toolset branch where only the web-tools note applies.
PROFILES = ("shell+browser", "full")


@pytest.mark.parametrize("tool_profile", PROFILES)
def test_plain_io_sends_the_sea_prompt_as_it_is(
    tmp_path: Path, agents_md: Path, tool_profile: str
) -> None:
    """Every framework trailer is gone: the system message is the SEA's prompt, exactly."""
    first, second = _run(tmp_path, plain_io=True, tool_profile=tool_profile)
    assert system_message(first) == PROMPT
    assert system_message(second) == PROMPT, "the prompt holds across the run's turns"
    (bash_result,) = tool_results(second)
    assert "Steps:" not in bash_result, "plain_io also keeps the usage line off the wire"
    assert "42" in bash_result


@pytest.mark.parametrize("tool_profile", PROFILES)
def test_without_plain_io_the_same_run_reads_every_trailer(
    tmp_path: Path, agents_md: Path, tool_profile: str
) -> None:
    """The twin run without ``plain_io``: the prompt, then each trailer in order."""
    first, _second = _run(tmp_path, plain_io=False, tool_profile=tool_profile)
    system = system_message(first)
    assert system.startswith(PROMPT)
    trailer = system[len(PROMPT):]
    expected = [
        WEB_TOOLS_OFF_NOTE.strip(),
        f"- The path of the file open in the editor is {EDITOR_FILE}",
        "# MOST IMPORTANT INSTRUCTIONS",
        f"- Work dir: {tmp_path.resolve()}",
        f"- Current process PID: {os.getpid()}",
        "# Task Settings",
        AGENTS_MD,
    ]
    if tool_profile != "full":
        expected.insert(0, f"# Restricted tool profile: {tool_profile}")
    positions = [trailer.find(text) for text in expected]
    assert all(p >= 0 for p in positions), dict(zip(expected, positions, strict=True))
    assert positions == sorted(positions), "the trailers keep their order"
