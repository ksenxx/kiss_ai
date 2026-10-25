# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the bundled orbestrouter SEA: the bestrouter on OpenRouter models only.

The agent run uses the scripted local chat-completions server configured
the way the daemon configures the SEA; the models are checked against the
real catalog.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kiss.agents.seas.bestrouter import bestrouter_sea
from kiss.agents.seas.orbestrouter import orbestrouter_sea
from kiss.agents.seas.orbestrouter.orbestrouter_sea import OrbestrouterSea
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

_SEA_PATH = Path(orbestrouter_sea.__file__).resolve()


def test_both_models_are_openrouter_twins_of_the_bestrouter_models() -> None:
    """Fable 5.1 works, Astra reviews, both as priced ``openrouter/`` catalog entries."""
    assert orbestrouter_sea.PRIMARY_MODEL == "openrouter/anthropic/claude-fable-5.1"
    assert orbestrouter_sea.REVIEW_MODEL == "openrouter/openai/gpt-6-astra"
    assert bestrouter_sea.PRIMARY_MODEL == "claude-fable-5-1"
    assert bestrouter_sea.REVIEW_MODEL == "gpt-6-astra"
    for model in (orbestrouter_sea.PRIMARY_MODEL, orbestrouter_sea.REVIEW_MODEL):
        assert model in MODEL_INFO, model
        assert MODEL_INFO[model].input_price_per_1M > 0, model


def test_sea_methods_follow_the_contract_and_name_the_models_literally() -> None:
    """The bestrouter protocol, headed for ``orbestrouter``, with the OpenRouter names in it."""
    sea = OrbestrouterSea()
    assert sea.register_as_model() is True
    assert sea.settings({"tool_profile": "shell"}) == {
        "tool_profile": "shell", "model": orbestrouter_sea.PRIMARY_MODEL,
    }
    protocol = orbestrouter_sea.SYSTEM_PROMPT
    assert sea.system_prompt("BASE") == "BASE\n\n" + protocol
    assert protocol.startswith("## Model routing protocol (orbestrouter)\n\n")
    assert protocol == bestrouter_sea.routing_prompt(
        "orbestrouter", orbestrouter_sea.PRIMARY_MODEL, orbestrouter_sea.REVIEW_MODEL
    )
    flat = " ".join(protocol.split())
    for phrase in (
        "Use 'openrouter/anthropic/claude-fable-5.1' model for all tasks, including software "
        "development.",
        "Use openrouter/openai/gpt-6-astra (not codex) through "
        "`run_agent(model=\"openrouter/openai/gpt-6-astra\", tool_profile=\"review\", "
        "wait=\"false\")` for a thorough read-only review and debugging of the other model's "
        "work,",
        "Use at most 75% of the task budget in openrouter/openai/gpt-6-astra,",
        "Use the model names literally without hallucinating new model names.",
    ):
        assert phrase in flat, phrase
    # The bestrouter's direct-vendor names never leak into the OpenRouter protocol.
    assert "'claude-fable-5-1'" not in protocol and " gpt-6-astra " not in protocol
    # Same working rules as the bestrouter, word for word.
    rules = protocol.split("Working rules:", 1)[1]
    assert rules == bestrouter_sea.SYSTEM_PROMPT.split("Working rules:", 1)[1]
    description = sea.description()
    assert orbestrouter_sea.PRIMARY_MODEL in description
    assert orbestrouter_sea.REVIEW_MODEL in description
    assert "/orbestrouter <task>" in description
    assert_no_removed_getters(orbestrouter_sea)


def test_orbestrouter_is_a_model_picker_sea() -> None:
    """Registered under its command name in the picker registry; its only setting is the model."""
    assert sea_commands.model_seas()["orbestrouter"] == _SEA_PATH
    assert sea_commands.model_sea("orbestrouter") == _SEA_PATH
    assert sea_commands.sea_settings(_SEA_PATH) == {"model": orbestrouter_sea.PRIMARY_MODEL}


def test_agent_file_loader_pins_the_openrouter_primary_model() -> None:
    """The daemon-side loader sets the run's model and stages the protocol; no tools hook."""
    cmd: dict[str, Any] = {"seaPath": str(_SEA_PATH), "toolProfile": "full", "model": "x"}
    assert apply_sea(cmd) == {"model"}
    assert cmd["model"] == orbestrouter_sea.PRIMARY_MODEL
    assert cmd["systemPromptHook"]("BASE") == "BASE\n\n" + orbestrouter_sea.SYSTEM_PROMPT
    assert cmd["toolsHook"]([finish_body]) == [finish_body]


def test_agent_run_carries_the_openrouter_protocol_in_its_system_prompt(tmp_path: Path) -> None:
    """A scripted run sees the protocol once, after the default prompt; run_parallel is offered."""
    run = sea_commands.evaluate_sea([OrbestrouterSea()], "say hello")
    parsed, agentic = run_sea(run, [finish_body("<p>hello</p>")], tmp_path, max_steps=2)
    assert parsed["success"] is True and "hello" in parsed["summary"]
    (request,) = agentic
    names = tool_names(request)
    assert {"run_parallel", "run_agent", "Read", "Edit", "finish"} <= names, names
    system = system_message(request)
    assert system.startswith("<identity>")
    assert system.count(orbestrouter_sea.SYSTEM_PROMPT) == 1
    # The protocol follows the default prompt; only the run's settings blocks come after it.
    assert system.index(orbestrouter_sea.SYSTEM_PROMPT) > system.index("</identity>")
    after = system.split(orbestrouter_sea.SYSTEM_PROMPT, 1)[1]
    assert "# Task Settings" in after and "<identity>" not in after
    assert "## Model routing protocol (bestrouter)" not in system


def test_rsi7d_tunes_the_bestrouter_constant_but_not_the_derived_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/rsi7d``'s prompt editor sees the bestrouter's f-string; the clone's prompt is derived."""
    from kiss.agents.seas.rsi7d import rsi7d_sea
    from kiss.tests.agents.seas.conftest import fake_checkout

    fake_checkout(tmp_path, {
        "bestrouter": Path(bestrouter_sea.__file__).read_text(encoding="utf-8"),
        "orbestrouter": _SEA_PATH.read_text(encoding="utf-8"),
    })
    monkeypatch.chdir(tmp_path)
    shown = rsi7d_sea.sea_prompt("bestrouter")
    assert "# system_prompt() returns SYSTEM_PROMPT, an f-string" in shown, shown[:300]
    assert "Use '{PRIMARY_MODEL}' model for all tasks" in shown
    derived = rsi7d_sea.sea_prompt("orbestrouter")
    assert derived.startswith("Error: ") and "module-level string constant" in derived
