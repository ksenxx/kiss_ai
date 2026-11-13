# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the bundled ccxbestrouter SEA: the bestrouter on Claude Code and Codex.

The agent run uses the scripted local chat-completions server configured
the way the daemon configures the SEA; the models are checked against the
real catalog.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kiss.agents.seas.bestrouter import bestrouter_sea
from kiss.agents.seas.ccxbestrouter import ccxbestrouter_sea
from kiss.agents.seas.ccxbestrouter.ccxbestrouter_sea import CcxbestrouterSea
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.sea_apply import apply_sea
from kiss.core.models.model_info import MODEL_INFO, _strip_thinking_alias, is_known_model
from kiss.tests.agents.seas.conftest import run_sea
from kiss.tests.agents.seas.sea_contract import (
    assert_no_removed_getters,
    system_message,
    tool_names,
)
from kiss.tests.agents.sorcar.local_model_server import finish_body

_SEA_PATH = Path(ccxbestrouter_sea.__file__).resolve()


def test_models_are_the_claude_code_and_codex_catalog_entries() -> None:
    """Opus 5.5 and Fable 5.2 through Claude Code, Astra at xhigh effort through Codex."""
    assert ccxbestrouter_sea.SIMPLE_MODEL == "cc/claude-opus-5-5-medium"
    assert ccxbestrouter_sea.PRIMARY_MODEL == "cc/claude-fable-5-2-max"
    assert ccxbestrouter_sea.REVIEW_MODEL == "codex/gpt-6-astra-xhigh"
    # The reviewer is an effort alias the catalog does not list: it is known to the
    # catalog lookup as the entry it runs as, ``codex/gpt-6-astra``.
    assert is_known_model(ccxbestrouter_sea.REVIEW_MODEL)
    for model in (
        ccxbestrouter_sea.SIMPLE_MODEL,
        ccxbestrouter_sea.PRIMARY_MODEL,
        _strip_thinking_alias(ccxbestrouter_sea.REVIEW_MODEL),
    ):
        assert model in MODEL_INFO, model
        assert MODEL_INFO[model].is_function_calling_supported, model


def test_sea_methods_follow_the_contract_and_name_the_models_literally() -> None:
    """The bestrouter protocol, headed for ``ccxbestrouter``, with the cc/codex names in it."""
    sea = CcxbestrouterSea()
    assert sea.register_as_model() is True
    assert sea.settings({"tool_profile": "shell"}) == {
        "tool_profile": "shell", "model": ccxbestrouter_sea.SIMPLE_MODEL,
    }
    protocol = ccxbestrouter_sea.SYSTEM_PROMPT
    assert sea.system_prompt("BASE") == "BASE\n\n" + protocol
    assert protocol.startswith("## Model routing protocol (ccxbestrouter)\n\n")
    flat = " ".join(protocol.split())
    for phrase in (
        "Use 'cc/claude-opus-5-5-medium' for simple questions and queries.",
        "Use 'cc/claude-fable-5-2-max' model for all other tasks, including "
        "software development and analysis and writing.",
        "Use codex/gpt-6-astra-xhigh through "
        "`run_agent(model=\"codex/gpt-6-astra-xhigh\", tool_profile=\"review\", "
        "wait=\"false\")` for a thorough read-only review and debugging of the other model's "
        "work,",
        "Use at most 75% of the task budget in codex/gpt-6-astra-xhigh,",
        "Use the model names literally without hallucinating new model names.",
    ):
        assert phrase in flat, phrase
    # The reviewer is the Codex model on purpose: the "(not codex)" warning is gone.
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
    assert ccxbestrouter_sea.SIMPLE_MODEL in description
    assert ccxbestrouter_sea.PRIMARY_MODEL in description
    assert ccxbestrouter_sea.REVIEW_MODEL in description
    assert "/ccxbestrouter <task>" in description
    assert_no_removed_getters(ccxbestrouter_sea)


def test_agent_file_loader_pins_the_claude_code_simple_model() -> None:
    """The only setting is the model; the daemon-side loader pins it and stages the protocol."""
    assert sea_commands.sea_settings(_SEA_PATH) == {"model": ccxbestrouter_sea.SIMPLE_MODEL}
    cmd: dict[str, Any] = {"seaPath": str(_SEA_PATH), "toolProfile": "full", "model": "x"}
    assert apply_sea(cmd) == {"model"}
    assert cmd["model"] == ccxbestrouter_sea.SIMPLE_MODEL
    assert cmd["systemPromptHook"]("BASE") == "BASE\n\n" + ccxbestrouter_sea.SYSTEM_PROMPT
    assert cmd["toolsHook"]([finish_body]) == [finish_body]


def test_agent_run_carries_the_protocol_in_its_system_prompt(tmp_path: Path) -> None:
    """A scripted run sees the protocol once, after the default prompt; run_agent is offered."""
    run = sea_commands.evaluate_sea([CcxbestrouterSea()], "say hello")
    parsed, agentic = run_sea(run, [finish_body("<p>hello</p>")], tmp_path, max_steps=2)
    assert parsed["success"] is True and "hello" in parsed["summary"]
    (request,) = agentic
    names = tool_names(request)
    assert {"run_parallel", "run_agent", "Read", "Edit", "finish"} <= names, names
    system = system_message(request)
    assert system.count(ccxbestrouter_sea.SYSTEM_PROMPT) == 1
    assert system.index(ccxbestrouter_sea.SYSTEM_PROMPT) > system.index("</identity>")
    assert "## Model routing protocol (bestrouter)" not in system
