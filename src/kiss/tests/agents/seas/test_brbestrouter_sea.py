# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the bundled brbestrouter SEA: the bestrouter on Amazon Bedrock.

The agent run uses the scripted local chat-completions server configured
the way the daemon configures the SEA; the models are checked against the
real catalog.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kiss.agents.seas.bestrouter import bestrouter_sea
from kiss.agents.seas.brbestrouter import brbestrouter_sea
from kiss.agents.seas.brbestrouter.brbestrouter_sea import BrbestrouterSea
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.sea_apply import apply_sea
from kiss.core.models.model_info import MODEL_INFO
from kiss.tests.agents.seas.conftest import run_sea
from kiss.tests.agents.seas.sea_contract import (
    assert_no_removed_getters,
    system_message,
    tool_names,
)
from kiss.tests.agents.sorcar.local_model_server import finish_body

_SEA_PATH = Path(brbestrouter_sea.__file__).resolve()


def test_models_are_the_bedrock_catalog_entries() -> None:
    """Opus 5.5 (medium, max) and Fable 5.1 (max) on Bedrock, all in the catalog."""
    assert brbestrouter_sea.SIMPLE_MODEL == "bedrock/global.anthropic.claude-opus-5-5-medium"
    assert brbestrouter_sea.PRIMARY_MODEL == "bedrock/global.anthropic.claude-opus-5-5-max"
    assert brbestrouter_sea.REVIEW_MODEL == "bedrock/global.anthropic.claude-fable-5-1-max"
    for model in (
        brbestrouter_sea.SIMPLE_MODEL,
        brbestrouter_sea.PRIMARY_MODEL,
        brbestrouter_sea.REVIEW_MODEL,
    ):
        assert model in MODEL_INFO, model
        assert MODEL_INFO[model].is_function_calling_supported, model


def test_sea_methods_follow_the_contract_and_name_the_models_literally() -> None:
    """The bestrouter protocol, headed for ``brbestrouter``, with the Bedrock names in it."""
    sea = BrbestrouterSea()
    assert sea.register_as_model() is True
    assert sea.settings({"tool_profile": "shell"}) == {
        "tool_profile": "shell", "model": brbestrouter_sea.SIMPLE_MODEL,
    }
    protocol = brbestrouter_sea.SYSTEM_PROMPT
    assert sea.system_prompt("BASE") == "BASE\n\n" + protocol
    assert protocol.startswith("## Model routing protocol (brbestrouter)\n\n")
    flat = " ".join(protocol.split())
    for phrase in (
        "Use 'bedrock/global.anthropic.claude-opus-5-5-medium' for simple questions and "
        "queries.",
        "Use 'bedrock/global.anthropic.claude-opus-5-5-max' model for all other tasks, "
        "including software development and analysis and writing.",
        "Use bedrock/global.anthropic.claude-fable-5-1-max through "
        "`run_agent(model=\"bedrock/global.anthropic.claude-fable-5-1-max\", "
        "tool_profile=\"review\", wait=\"false\")` for a thorough read-only review and "
        "debugging of the other model's work,",
        "Use at most 75% of the task budget in bedrock/global.anthropic.claude-fable-5-1-max,",
        "Use the model names literally without hallucinating new model names.",
    ):
        assert phrase in flat, phrase
    # The reviewer is a Claude model: the "(not codex)" warning about Astra is gone.
    assert "not codex" not in protocol
    assert f"'{bestrouter_sea.SIMPLE_MODEL}'" not in protocol
    assert f"'{bestrouter_sea.PRIMARY_MODEL}'" not in protocol
    # The reviewer is unquoted and wraps onto a new line: only the flat text shows a leak.
    assert f" {bestrouter_sea.REVIEW_MODEL} " in " ".join(bestrouter_sea.SYSTEM_PROMPT.split())
    assert f" {bestrouter_sea.REVIEW_MODEL} " not in flat
    # Same working rules as the bestrouter, word for word.
    rules = protocol.split("Working rules:", 1)[1]
    assert rules == bestrouter_sea.SYSTEM_PROMPT.split("Working rules:", 1)[1]
    description = sea.description()
    assert brbestrouter_sea.SIMPLE_MODEL in description
    assert brbestrouter_sea.PRIMARY_MODEL in description
    assert brbestrouter_sea.REVIEW_MODEL in description
    assert "/brbestrouter <task>" in description
    assert_no_removed_getters(brbestrouter_sea)


def test_agent_file_loader_pins_the_bedrock_simple_model() -> None:
    """The only setting is the model; the daemon-side loader pins it and stages the protocol."""
    assert sea_commands.sea_settings(_SEA_PATH) == {"model": brbestrouter_sea.SIMPLE_MODEL}
    cmd: dict[str, Any] = {"seaPath": str(_SEA_PATH), "toolProfile": "full", "model": "x"}
    assert apply_sea(cmd) == {"model"}
    assert cmd["model"] == brbestrouter_sea.SIMPLE_MODEL
    assert cmd["systemPromptHook"]("BASE") == "BASE\n\n" + brbestrouter_sea.SYSTEM_PROMPT
    assert cmd["toolsHook"]([finish_body]) == [finish_body]


def test_agent_run_carries_the_protocol_in_its_system_prompt(tmp_path: Path) -> None:
    """A scripted run sees the protocol once, after the default prompt; run_agent is offered."""
    run = sea_commands.evaluate_sea([BrbestrouterSea()], "say hello")
    parsed, agentic = run_sea(run, [finish_body("<p>hello</p>")], tmp_path, max_steps=2)
    assert parsed["success"] is True and "hello" in parsed["summary"]
    (request,) = agentic
    names = tool_names(request)
    assert {"run_parallel", "run_agent", "Read", "Edit", "finish"} <= names, names
    system = system_message(request)
    assert system.count(brbestrouter_sea.SYSTEM_PROMPT) == 1
    assert system.index(brbestrouter_sea.SYSTEM_PROMPT) > system.index("</identity>")
    assert "## Model routing protocol (bestrouter)" not in system
