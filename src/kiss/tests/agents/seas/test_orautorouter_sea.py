# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the bundled orautorouter SEA: the autorouter on OpenRouter models only.

The tier table, the menu and the pick are exercised against the real model
catalog and the real credential check; the agent run uses the scripted local
chat-completions server configured the way the daemon configures the SEA.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.seas.autorouter import autorouter_sea
from kiss.agents.seas.autorouter.autorouter_sea import AutorouterSea
from kiss.agents.seas.orautorouter import orautorouter_sea
from kiss.agents.seas.orautorouter.orautorouter_sea import OrautorouterSea
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sea_apply import apply_sea
from kiss.core.models.model_info import MODEL_INFO, get_available_models
from kiss.tests.agents.seas.conftest import registered, run_sea, runnable_candidates
from kiss.tests.agents.seas.sea_contract import (
    assert_no_removed_getters,
    system_message,
    tool_names,
    tool_result_text,
    tool_results,
)
from kiss.tests.agents.sorcar.local_model_server import finish_body, tool_call_body

_SEA_PATH = Path(orautorouter_sea.__file__).resolve()
_TOOL_NAMES = {
    "model_menu", "pick_model", "estimate_cost", "observed_call_costs", "log_decision",
}


def test_sea_methods_follow_the_contract_and_share_the_autorouter_protocol() -> None:
    """Same settings and protocol as the autorouter, headed and introduced for OpenRouter."""
    sea = OrautorouterSea()
    prompt = orautorouter_sea.SYSTEM_PROMPT
    assert sea.system_prompt("ASSEMBLED") == "ASSEMBLED\n\n" + prompt
    assert prompt.startswith("## Model routing protocol (orautorouter)\n\nYou are the orautorouter")
    assert "`openrouter/<vendor>/<model>`" in prompt
    assert prompt != autorouter_sea.SYSTEM_PROMPT
    shared = autorouter_sea.SYSTEM_PROMPT.removeprefix(autorouter_sea.INTRO)
    assert shared.startswith(" Finish the task") and prompt.endswith(shared)
    assert sea.register_as_model() is True
    assert sea.settings({"tool_profile": "shell"}) == AutorouterSea().settings(
        {"tool_profile": "shell"}
    )
    assert sea.settings({})["model"].startswith(orautorouter_sea.VENDOR_PREFIX)
    # The menu and the pick are this module's; the rest is the autorouter's very functions.
    tools = {t.__name__: t for t in sea.tools([])}
    assert set(tools) == _TOOL_NAMES
    assert tools["model_menu"] is orautorouter_sea.model_menu
    assert tools["pick_model"] is orautorouter_sea.pick_model
    assert tools["estimate_cost"] is autorouter_sea.estimate_cost
    assert tools["log_decision"] is autorouter_sea.log_decision
    assert "OpenRouter" in sea.description() and "/orautorouter <task>" in sea.description()
    assert_no_removed_getters(orautorouter_sea)


def test_orautorouter_is_a_model_picker_sea() -> None:
    """Registered under its command name in the picker registry."""
    assert sea_commands.model_seas()["orautorouter"] == _SEA_PATH
    assert sea_commands.model_sea("orautorouter") == _SEA_PATH


def test_agent_file_loader_stages_the_openrouter_router() -> None:
    """The daemon-side loader applies the settings and stages the prompt and tools hooks."""
    cmd: dict[str, Any] = {"seaPath": str(_SEA_PATH), "toolProfile": "full", "model": "x"}
    assert apply_sea(cmd) == {"model", "classifyTasks", "useWebTools", "useMemory"}
    assert cmd["model"] == autorouter_sea.ROUTER_MODEL
    assert cmd["model"].startswith(orautorouter_sea.VENDOR_PREFIX)
    assert cmd["systemPromptHook"]("BASE") == "BASE\n\n" + orautorouter_sea.SYSTEM_PROMPT
    assert cmd["classifyTasks"] is False and cmd["useWebTools"] is True
    assert cmd["useMemory"] is True
    assert {tool.__name__ for tool in cmd["toolsHook"]([])} == _TOOL_NAMES


def test_every_tier_candidate_is_a_distinct_openrouter_catalog_model() -> None:
    """Every candidate is an ``openrouter/`` catalog model in exactly one tier; no tier is empty."""
    assert tuple(orautorouter_sea.TIERS) == autorouter_sea.TIER_NAMES
    seen: set[str] = set()
    for tier, candidates in orautorouter_sea.TIERS.items():
        assert len(candidates) >= 2, tier
        for model, note in candidates:
            assert model.startswith(orautorouter_sea.VENDOR_PREFIX), model
            assert model in MODEL_INFO, model
            assert MODEL_INFO[model].input_price_per_1M > 0, model
            assert note and model not in seen, model
            seen.add(model)
    # Every OpenRouter candidate the autorouter already lists sits in the same tier here.
    for tier, candidates in autorouter_sea.TIERS.items():
        ours = {model for model, _ in orautorouter_sea.TIERS[tier]}
        for model, _ in candidates:
            if model.startswith(orautorouter_sea.VENDOR_PREFIX):
                assert model in ours, (tier, model)
    frontier = {model for model, _ in orautorouter_sea.TIERS["frontier"]}
    assert frontier == {"openrouter/openai/gpt-6-astra", "openrouter/anthropic/claude-fable-5.1"}


def test_menu_and_pick_only_ever_name_openrouter_models() -> None:
    """The menu lists this module's tiers; the pick is the first runnable one, minus excludes."""
    menu = json.loads(orautorouter_sea.model_menu(100_000, 10_000))
    assert tuple(menu) == autorouter_sea.TIER_NAMES
    runnable = set(get_available_models())
    for tier, rows in menu.items():
        assert [r["model"] for r in rows] == [m for m, _ in orautorouter_sea.TIERS[tier]]
        for row in rows:
            assert row["model"].startswith(orautorouter_sea.VENDOR_PREFIX)
            assert row["runnable"] is (row["model"] in runnable)
            info = MODEL_INFO[row["model"]]
            assert row["estimated_usd"] == round(
                (100_000 * info.input_price_per_1M + 10_000 * info.output_price_per_1M) / 1e6, 4
            )
    assert orautorouter_sea.pick_model("huge").startswith("Error: unknown tier 'huge'")
    for tier in autorouter_sea.TIER_NAMES:
        usable = runnable_candidates(orautorouter_sea.TIERS, tier)
        picked = orautorouter_sea.pick_model(tier)
        if not usable:
            assert picked.startswith(f"Error: no runnable model in tier {tier!r}")
            continue
        chosen = json.loads(picked)
        assert chosen["model"] == usable[0] and chosen["tier"] == tier
        assert "runnable" not in chosen
        rest = orautorouter_sea.pick_model(tier, exclude=usable[0])
        if len(usable) > 1:
            assert json.loads(rest)["model"] == usable[1]
        else:
            assert rest.startswith(f"Error: no runnable model in tier {tier!r}")
        # The autorouter's pick of the same tier may differ: it also lists direct vendor models.
        theirs = autorouter_sea.pick_model(tier)
        assert theirs.startswith("Error") or json.loads(theirs)["model"] in MODEL_INFO


def test_agent_run_offers_the_openrouter_tools_and_logs_to_the_shared_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scripted model picks a medium model, logs it and finishes; the pick is OpenRouter."""
    home = tmp_path / "home"
    monkeypatch.setenv("KISS_HOME", str(home))
    runnable = runnable_candidates(orautorouter_sea.TIERS, "medium")
    model = runnable[0] if runnable else orautorouter_sea.TIERS["medium"][0][0]
    run = sea_commands.evaluate_sea([OrautorouterSea()], "add a --json flag to the export command")
    script = [
        tool_call_body("pick_model", {"tier": "medium"}, prompt_tokens=500),
        tool_call_body(
            "log_decision",
            {"unit": "implement flag", "tier": "medium", "model": model, "reason": "tier=medium"},
            prompt_tokens=600,
        ),
        finish_body(f"<p>routed to {model}</p>", prompt_tokens=700),
    ]
    agent = ChatSorcarAgent("orautorouter-sea-test")
    with registered(agent, "orautorouter-sea-test"):
        parsed, agentic = run_sea(run, script, tmp_path, max_steps=5, agent=agent)
    assert parsed["success"] is True and model in parsed["summary"]
    assert len(agentic) == 3
    for request in agentic:
        names = tool_names(request)
        assert _TOOL_NAMES | {"run_agent", "run_parallel", "set_model", "Bash"} <= names, names
        system = system_message(request)
        assert orautorouter_sea.SYSTEM_PROMPT in system
        assert "## Model routing protocol (autorouter)" not in system
    pick_result = tool_results(agentic[1])
    pick_text = tool_result_text(pick_result[0])
    if runnable:
        assert json.loads(pick_text)["model"] == model
        assert model.startswith(orautorouter_sea.VENDOR_PREFIX)
    else:
        assert pick_text.startswith("Error: no runnable model in tier 'medium'")
    ledger = home / "MODEL_DECISIONS.md"
    task_id = agent.last_task_id
    assert task_id
    assert ledger.read_text(encoding="utf-8").splitlines()[-1].endswith(
        f"| {task_id} | implement flag | medium | {model} | tier=medium | pending |"
    )


def test_picking_orautorouter_ensures_the_autorouter_weekly_evidence_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both routers read the same evidence file, so the hook ensures the same weekly job once."""
    from kiss.agents.sorcar.cron_agent import load_jobs

    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home"))
    assert OrautorouterSea().on_picked_as_model(str(tmp_path)).startswith("created:")
    (job,) = load_jobs()
    assert job["name"] == autorouter_sea.RSI7D_JOB_NAME and job["enabled"] is True
    assert AutorouterSea().on_picked_as_model(str(tmp_path)).startswith("exists:")
    assert OrautorouterSea().on_picked_as_model(str(tmp_path)).startswith("exists:")
    assert len(load_jobs()) == 1


def test_routing_prompt_reheads_the_autorouter_constant_and_rereads_the_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The autorouter's own intro reproduces its prompt; a derived one swaps heading and intro.

    The evidence is read at call time: after ``AUTOROUTER.md`` changes, a
    derived prompt carries the new text while the imported autorouter
    module's load-time constant does not.
    """
    intro = autorouter_sea.INTRO.split("\n\n", 1)[1]
    assert intro == "You are the autorouter."
    assert autorouter_sea.routing_prompt("autorouter", intro) == autorouter_sea.SYSTEM_PROMPT
    derived = autorouter_sea.routing_prompt("x", "You are x.")
    assert derived.startswith("## Model routing protocol (x)\n\nYou are x. Finish the task")
    assert "(autorouter)" not in derived and "You are the autorouter." not in derived

    home = tmp_path / "home"
    home.mkdir()
    (home / autorouter_sea.EVIDENCE_NAME).write_text(
        "| model | tasks |\n|---|---|\n| model-x | 12 |\n", encoding="utf-8"
    )
    monkeypatch.setenv("KISS_HOME", str(home))
    fresh = autorouter_sea.observed_evidence()
    assert fresh != autorouter_sea.EVIDENCE and "| model-x | 12 |" in fresh
    derived = autorouter_sea.routing_prompt("orautorouter", "You are the orautorouter.")
    assert fresh in derived and autorouter_sea.EVIDENCE not in derived
    assert autorouter_sea.EVIDENCE in autorouter_sea.SYSTEM_PROMPT
    assert fresh not in autorouter_sea.SYSTEM_PROMPT
    # The clone file, re-executed by the loader, builds its prompt from the fresh evidence.
    loaded = sea_commands.load_sea(_SEA_PATH)
    assert fresh in loaded.system_prompt("")


def test_routing_prompt_refuses_an_autorouter_prompt_that_lost_its_intro(tmp_path: Path) -> None:
    """A copy of the SEA whose constant opens differently fails loudly, not with two headings."""
    from kiss.agents.sorcar.sea_settings import execute_python_file

    # The anchors come from the SEA's own constants, so a reworded prompt
    # that keeps them in step with ``routing_prompt`` does not break this test.
    source = Path(autorouter_sea.__file__).read_text(encoding="utf-8")
    opening = "\n" + autorouter_sea.INTRO.rsplit("\n", 1)[-1]  # "\nYou are the autorouter."
    copy = tmp_path / "autorouter_sea.py"
    copy.write_text(source.replace(opening, "\nYou route.", 1), encoding="utf-8")
    namespace = execute_python_file(str(copy))
    with pytest.raises(ValueError, match="no longer opens with"):
        namespace["routing_prompt"]("x", "You are x.")
    # Likewise when the heading after the evidence block is renamed.
    heading = f"\n{autorouter_sea.AFTER_EVIDENCE.strip()}\n"  # "\n## Hard rules\n"
    copy.write_text(source.replace(heading, "\n## Rules\n", 1), encoding="utf-8")
    namespace = execute_python_file(str(copy))
    with pytest.raises(ValueError, match="evidence no longer ends at"):
        namespace["routing_prompt"]("x", "You are x.")


def test_routing_prompt_swaps_only_the_evidence_block_even_when_a_rule_matches_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load-time evidence equal to a rule sentence: the refresh must not rewrite that rule.

    The autorouter is re-executed from a copy (the loader's way) with
    ``AUTOROUTER.md`` holding a sentence of the hard rules, then the file
    changes to a table and the copy's ``routing_prompt`` is called.
    """
    from kiss.agents.sorcar.sea_settings import execute_python_file

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("KISS_HOME", str(home))
    rule = "Never downgrade a model the user named explicitly."
    assert autorouter_sea.SYSTEM_PROMPT.count(rule) == 1
    (home / autorouter_sea.EVIDENCE_NAME).write_text(rule, encoding="utf-8")
    copy = tmp_path / "autorouter_sea.py"
    copy.write_text(Path(autorouter_sea.__file__).read_text(encoding="utf-8"), encoding="utf-8")
    namespace = execute_python_file(str(copy))
    assert namespace["EVIDENCE"] == rule and namespace["SYSTEM_PROMPT"].count(rule) == 2

    table = "| model | tasks |\n|---|---|\n| model-x | 12 |"
    (home / autorouter_sea.EVIDENCE_NAME).write_text(table, encoding="utf-8")
    derived = namespace["routing_prompt"]("orautorouter", "You are the orautorouter.")
    assert derived.count(table) == 1 and derived.count(rule) == 1
    assert derived.index("## Observed model evidence") < derived.index(table)
    assert derived.index(table) < derived.index("## Hard rules") < derived.index(rule)
