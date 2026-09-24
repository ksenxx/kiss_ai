# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the bundled autoroute SEA.

The routing tools are exercised against the real model catalog
(:mod:`kiss.core.models.model_info`) and the real credential check
(``get_available_models``); the agent run uses the scripted local
chat-completions server (:mod:`kiss.tests.agents.sorcar.local_model_server`)
configured exactly the way the daemon configures the SEA, so the tools
offered, the system prompt and the ledger written to the KISS home with the
task's persisted id are the real ones.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.seas import autoroute_sea
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.models.model_info import MODEL_INFO, get_available_models
from kiss.server import agent_state
from kiss.server.agent_file import apply_agent_overrides
from kiss.server.tools_file import load_tools_file
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)

_SEA_PATH = Path(autoroute_sea.__file__).resolve()
_TOOL_NAMES = {"model_menu", "pick_model", "estimate_cost", "log_decision"}


def _runnable_candidates(tier: str) -> list[str]:
    """Return the tier's candidates whose provider credential is configured, in order."""
    available = set(get_available_models())
    return [model for model, _ in autoroute_sea.TIERS[tier] if model in available]


def test_sea_getters_follow_the_contract() -> None:
    """The getters pin the run: protocol prompt, four tools, run_parallel off, no extras."""
    prompt = autoroute_sea.system_prompt()
    assert prompt == autoroute_sea.SYSTEM_PROMPT
    flat = " ".join(prompt.split())
    for phrase in (
        "cost per accepted task",
        "`decide`",
        "`pick_model(tier, tokens_in, tokens_out, exclude)`",
        "`run_agent(task=..., model_name=<picked>)`, one call per unit",
        "`set_model` only at a phase boundary",
        "`log_decision(unit, tier, model, reason, outcome)`",
        "Never downgrade a model the user named explicitly.",
    ):
        assert phrase in flat, phrase
    assert {tool.__name__ for tool in autoroute_sea.tools()} == _TOOL_NAMES
    # run_parallel workers inherit the parent's custom system prompt, which
    # would make every routed unit a router; dispatch goes through run_agent.
    assert autoroute_sea.is_parallel() is False
    assert autoroute_sea.classify_tasks() is False
    assert autoroute_sea.use_web_tools() is False
    assert autoroute_sea.use_memory() is False
    # The caller's model, budget, worktree and tool profile must win: the
    # router runs on whatever the user or the dispatching agent chose.
    for getter in ("model", "max_budget", "use_worktree", "auto_commit", "tool_profile"):
        assert not hasattr(autoroute_sea, getter), getter


def test_slash_autoroute_resolves_to_the_bundled_sea() -> None:
    """``/autoroute <task>`` is rewritten into a ``run_agent`` directive on this file."""
    assert sea_commands.get_command("autoroute") == _SEA_PATH
    rewritten = sea_commands.rewrite_prompt_if_command("/autoroute add a --json flag")
    assert rewritten is not None
    prompt, path = rewritten
    assert path == _SEA_PATH
    assert f'agent = "{_SEA_PATH}"' in prompt
    assert prompt.endswith("TASK TEXT FOR run_agent:\nadd a --json flag")


def test_agent_file_loader_stages_the_sea_as_its_own_tools_file() -> None:
    """The daemon-side loader applies the getters and loads the four tools from the file."""
    cmd: dict[str, Any] = {"agentPath": str(_SEA_PATH), "toolProfile": "full", "model": "x"}
    overridden = apply_agent_overrides(cmd)
    assert overridden == {
        "systemPrompt",
        "toolsFile",
        "useParallel",
        "classifyTasks",
        "webTools",
        "useMemory",
    }
    assert cmd["systemPrompt"] == autoroute_sea.SYSTEM_PROMPT
    assert cmd["toolsFile"] == str(_SEA_PATH)
    assert cmd["useParallel"] is False
    assert cmd["classifyTasks"] is False and cmd["webTools"] is False and cmd["useMemory"] is False
    assert cmd["toolProfile"] == "full" and cmd["model"] == "x"
    assert {tool.__name__ for tool in load_tools_file(cmd["toolsFile"])} == _TOOL_NAMES


def test_tier_candidates_are_distinct_catalog_models() -> None:
    """Every candidate is priced by the catalog and belongs to exactly one tier."""
    assert tuple(autoroute_sea.TIERS) == autoroute_sea.TIER_NAMES == ("small", "medium", "frontier")
    seen: set[str] = set()
    for tier, candidates in autoroute_sea.TIERS.items():
        assert candidates, tier
        for model, note in candidates:
            assert model in MODEL_INFO, f"{tier}: {model} is not in the model catalog"
            assert note
            assert model not in seen, f"{model} appears in two tiers"
            seen.add(model)


def test_model_menu_prices_every_candidate_from_the_catalog() -> None:
    """The menu lists each tier in order with catalog prices, cost estimate and availability."""
    tokens_in, tokens_out = 1_000_000, 100_000
    menu = json.loads(autoroute_sea.model_menu(tokens_in, tokens_out))
    available = set(get_available_models())
    assert list(menu) == list(autoroute_sea.TIER_NAMES)
    for tier, records in menu.items():
        assert [r["model"] for r in records] == [m for m, _ in autoroute_sea.TIERS[tier]]
        for record, (model, note) in zip(records, autoroute_sea.TIERS[tier], strict=True):
            info = MODEL_INFO[model]
            assert record["input_per_1M"] == info.input_price_per_1M
            assert record["output_per_1M"] == info.output_price_per_1M
            assert record["estimated_usd"] == round(
                info.input_price_per_1M + info.output_price_per_1M / 10, 4
            )
            assert record["runnable"] is (model in available)
            assert record["note"] == note
    # Defaults: 200k prompt + 20k completion tokens per unit.
    default = json.loads(autoroute_sea.model_menu())["small"][0]
    info = MODEL_INFO[default["model"]]
    assert default["estimated_usd"] == round(
        info.input_price_per_1M * 0.2 + info.output_price_per_1M * 0.02, 4
    )


def test_pick_model_returns_the_first_runnable_candidate_and_honours_exclude() -> None:
    """The pick is the first configured candidate; excluded models are skipped in order."""
    for tier in autoroute_sea.TIER_NAMES:
        runnable = _runnable_candidates(tier)
        if not runnable:
            pytest.skip(f"no provider credential configured for any {tier} candidate")
        pick = json.loads(autoroute_sea.pick_model(tier, 1_000_000, 100_000))
        assert pick["model"] == runnable[0]
        assert pick["tier"] == tier
        assert "runnable" not in pick
        info = MODEL_INFO[runnable[0]]
        assert pick["estimated_usd"] == round(
            info.input_price_per_1M + info.output_price_per_1M / 10, 4
        )
        if len(runnable) > 1:
            second = json.loads(
                autoroute_sea.pick_model(tier, exclude=f"{runnable[0]}, other-model")
            )
            assert second["model"] == runnable[1]
        everyone = " ".join(runnable)
        message = autoroute_sea.pick_model(tier, exclude=everyone)
        assert message.startswith(f"Error: no runnable model in tier {tier!r}")
        assert everyone in message
    assert autoroute_sea.pick_model("huge") == (
        "Error: unknown tier 'huge'; use one of small, medium, frontier."
    )


def test_estimate_cost_uses_catalog_prices_and_rejects_unknown_models() -> None:
    """The estimate is (input price * tokens_in + output price * tokens_out) / 1M."""
    model = autoroute_sea.TIERS["medium"][0][0]
    info = MODEL_INFO[model]
    estimate = json.loads(autoroute_sea.estimate_cost(model, 2_000_000, 500_000))
    assert estimate == {
        "model": model,
        "input_per_1M": info.input_price_per_1M,
        "output_per_1M": info.output_price_per_1M,
        "estimated_usd": round(info.input_price_per_1M * 2 + info.output_price_per_1M * 0.5, 4),
    }
    assert autoroute_sea.estimate_cost("no-such-model") == (
        "Error: unknown model 'no-such-model'; it is not in the local model catalog."
    )


def test_log_decision_writes_a_table_in_the_kiss_home_with_a_dash_task_id_outside_a_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ledger is ``$KISS_HOME/MODEL_DECISIONS.md``; rows append; no task means ``-``.

    The KISS home is pointed at a fresh directory that does not exist yet,
    so the test also covers the directory creation and the header write.
    ``chdir`` into another directory proves the path no longer depends on
    the current directory.
    """
    home = tmp_path / "home"
    monkeypatch.setenv("KISS_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    ledger = home / "MODEL_DECISIONS.md"
    assert autoroute_sea.ledger_path() == ledger
    assert (
        autoroute_sea.log_decision("read logs", "small", "gpt-6-luna", "tier=small 0.9")
        == f"logged to {ledger}"
    )
    assert (
        autoroute_sea.log_decision(
            "fix | bug\nin parser",
            "medium",
            "gemini-3.8-flash",
            "escalated from small",
            "tests pass",
        )
        == f"logged to {ledger}"
    )
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# Model routing decisions"
    assert lines[2] == "| time (UTC) | task_id | unit | tier | model | reason | outcome |"
    assert lines[3] == "|---|---|---|---|---|---|---|"
    assert lines[4].endswith("| - | read logs | small | gpt-6-luna | tier=small 0.9 | pending |")
    assert lines[5].endswith(
        "| - | fix / bug in parser | medium | gemini-3.8-flash "
        "| escalated from small | tests pass |"
    )
    assert len(lines) == 6
    assert not (tmp_path / "tmp").exists()
    assert autoroute_sea.log_decision("x", "huge", "m", "r") == (
        "Error: unknown tier 'huge'; use one of small, medium, frontier."
    )
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 6


def test_agent_run_offers_routing_and_dispatch_tools_and_logs_with_the_task_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the SEA's configuration the model gets the four tools plus dispatch tools.

    The scripted model picks a medium model, logs the decision and finishes
    with the pick.  The test checks the tool schemas offered on every step
    (routing tools, ``run_agent``/``set_model``/``Bash`` present, no
    ``run_parallel``, browser or memory tools), the system prompt, the real
    pick — or the real "no runnable model" error on an installation without
    a medium-tier credential — in the tool-result message, and that the
    ledger row landed in ``$KISS_HOME/MODEL_DECISIONS.md`` (not under the
    task's work directory) stamped with the task's persisted id, found
    through the agent registry because the task thread is registered.
    """
    home = tmp_path / "home"
    monkeypatch.setenv("KISS_HOME", str(home))
    runnable = _runnable_candidates("medium")
    model = runnable[0] if runnable else autoroute_sea.TIERS["medium"][0][0]
    script = [
        tool_call_body("pick_model", {"tier": "medium"}, prompt_tokens=500),
        tool_call_body(
            "log_decision",
            {
                "unit": "implement flag",
                "tier": "medium",
                "model": model,
                "reason": "tier=medium 0.8",
            },
            prompt_tokens=600,
        ),
        finish_body(f"<p>routed to {model}</p>", prompt_tokens=700),
    ]
    agent = WorktreeSorcarAgent("autoroute-sea-test")
    state = agent_state.AgentState(
        "autoroute-sea-test", agent=agent, task_thread=threading.current_thread()
    )
    agent_state.register(state)
    try:
        with serve(script) as (url, requests):
            result = agent.run(
                prompt_template="add a --json flag to the export command",
                model_name=MODEL,
                work_dir=str(tmp_path),
                use_worktree=False,
                max_steps=5,
                max_budget=1.0,
                model_config={"base_url": url, "api_key": "local"},
                base_system_prompt=autoroute_sea.system_prompt(),
                tools=autoroute_sea.tools(),
                web_tools=autoroute_sea.use_web_tools(),
                use_memory=autoroute_sea.use_memory(),
                is_parallel=autoroute_sea.is_parallel(),
                verbose=False,
            )
    finally:
        agent_state.unregister("autoroute-sea-test", state)
    parsed = yaml.safe_load(result)
    assert parsed["success"] is True
    assert model in parsed["summary"]

    agentic = [r for r in requests if r.get("tools")]
    assert len(agentic) == 3, [list(r) for r in requests]
    for request in agentic:
        names = {t["function"]["name"] for t in request["tools"]}
        assert _TOOL_NAMES <= names, names
        assert {"run_agent", "set_model", "Bash", "run_commands_parallel", "finish"} <= names, names
        assert not {"run_parallel", "go_to_url", "click", "screenshot", "memory_search"} & names, (
            names
        )
        system = str(next(m for m in request["messages"] if m["role"] == "system")["content"])
        assert system.startswith(autoroute_sea.SYSTEM_PROMPT), system[:200]
    # Step 2 carries the real pick; step 3 carries the ledger path.
    # Tool results carry a "Steps: ..." status footer after a blank line.
    pick_result = [m for m in agentic[1]["messages"] if m["role"] == "tool"]
    assert len(pick_result) == 1, agentic[1]["messages"]
    pick_text = str(pick_result[0]["content"]).split("\n\nSteps:")[0]
    if runnable:
        assert json.loads(pick_text)["model"] == model
    else:
        assert pick_text.startswith("Error: no runnable model in tier 'medium'")
    ledger = home / "MODEL_DECISIONS.md"
    log_result = [m for m in agentic[2]["messages"] if m["role"] == "tool"]
    assert str(log_result[-1]["content"]).split("\n\nSteps:")[0] == f"logged to {ledger}"
    task_id = agent.last_task_id
    assert task_id
    rows = ledger.read_text(encoding="utf-8").splitlines()
    assert rows[-1].endswith(
        f"| {task_id} | implement flag | medium | {model} | tier=medium 0.8 | pending |"
    )
    assert not (tmp_path / "tmp" / "MODEL_DECISIONS.md").exists()
