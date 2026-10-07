# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the bundled autorouter SEA.

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

from kiss.agents.seas.autorouter import autorouter_sea
from kiss.agents.seas.autorouter.autorouter_sea import AutorouterSea
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.sea_apply import apply_sea
from kiss.agents.sorcar.sea_settings import resolve_settings
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.models.model_info import MODEL_INFO, get_available_models
from kiss.server import agent_state
from kiss.tests.agents.seas.sea_contract import assert_no_removed_getters
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)

_SEA_PATH = Path(autorouter_sea.__file__).resolve()
_TOOL_NAMES = {
    "model_menu", "pick_model", "estimate_cost", "observed_call_costs", "log_decision",
}


def _runnable_candidates(tier: str) -> list[str]:
    """Return the tier's candidates whose provider credential is configured, in order."""
    available = set(get_available_models())
    return [model for model, _ in autorouter_sea.TIERS[tier] if model in available]


def test_sea_methods_follow_the_contract() -> None:
    """The methods pin the run: added protocol, picker entry, five tools, run_parallel off."""
    sea = AutorouterSea()
    assert sea.system_prompt("ASSEMBLED") == "ASSEMBLED\n\n" + autorouter_sea.SYSTEM_PROMPT
    prompt = autorouter_sea.SYSTEM_PROMPT
    assert prompt.startswith("## Model routing protocol (autorouter)")
    assert sea.register_as_model() is True
    flat = " ".join(prompt.split())
    for phrase in (
        "cost per accepted task",
        "`decide`",
        "`pick_model(tier, tokens_in, tokens_out, exclude)`",
        "`run_agent(task=..., model=<picked>)`, one call per unit",
        "`set_model` only at a phase boundary",
        "`log_decision(unit, tier, model, reason, outcome)`",
        "`observed_call_costs(days, model)`",
        "One short clause per cell (cut at 120 characters)",
        "Never downgrade a model the user named explicitly.",
    ):
        assert phrase in flat, phrase
    # The protocol goes into every request of every routed task: about 1,000
    # tokens of protocol text plus at most EVIDENCE_MAX_CHARS of evidence.
    protocol = prompt.replace(autorouter_sea.observed_evidence(), "")
    assert len(protocol) <= 5_000, len(protocol)
    assert len(prompt) <= 5_000 + autorouter_sea.EVIDENCE_MAX_CHARS + len(
        autorouter_sea.EVIDENCE_CUT
    )
    assert [tool.__name__ for tool in sea.tools([print])][0] == "print"
    assert {tool.__name__ for tool in sea.tools([])} == _TOOL_NAMES
    # run_parallel workers inherit the parent's custom system prompt, which
    # would make every routed unit a router; dispatch goes through run_agent.
    assert sea.settings({}) == {
        "model": autorouter_sea.orchestrator_model(),
        "allow_fan_out": False,
        "auto_classify": False,
        "use_web_tools": False,
        "use_memory": False,
    }
    # The SEA's keys win over what the caller declared below it.
    assert sea.settings({"model": "x", "max_budget": 3.0}) == {
        "max_budget": 3.0, **sea.settings({})
    }
    # No preset named: the resolved settings are the five keys above under
    # the default ``session`` preset (which adds no defaults), so the
    # caller's budget, worktree, auto-commit and tool profile win.  The
    # protocol is ADDED to the default prompt, never a replacement.
    assert sea_commands.base_settings([sea]) == resolve_settings(sea.settings({})) == {
        "kind": "session", **sea.settings({})
    }
    assert_no_removed_getters(autorouter_sea)


def test_autorouter_is_a_model_picker_sea() -> None:
    """``register_as_model()`` lists the SEA under its command name in the picker registry."""
    assert sea_commands.model_seas()["autorouter"] == _SEA_PATH
    assert sea_commands.model_sea("autorouter") == _SEA_PATH


def test_slash_autorouter_resolves_to_the_bundled_sea() -> None:
    """``/autorouter <task>`` resolves to the task text and this file (run directly)."""
    assert sea_commands.get_command("autorouter") == _SEA_PATH
    hit = sea_commands.slash_command_task("/autorouter add a --json flag")
    assert hit is not None
    task_text, path = hit
    assert path == _SEA_PATH
    assert task_text == "add a --json flag"
    assert sea_commands.sea_settings(path) == {"kind": "session", **AutorouterSea().settings({})}
    assert sea_commands.help_text_if_command("/autorouter help") == AutorouterSea().description()


def test_agent_file_loader_stages_the_sea_tools() -> None:
    """The daemon-side loader applies the settings and stages the prompt and tools hooks."""
    cmd: dict[str, Any] = {
        "seaPath": str(_SEA_PATH),
        "toolProfile": "full",
        "model": "x",
        "appendToSystemPrompt": "CALLER TEXT",
    }
    overridden = apply_sea(cmd)
    # The hooks are written on every run, so the set lists the settings only.
    assert overridden == {
        "model",
        "isParallel",
        "classifyTasks",
        "useWebTools",
        "useMemory",
    }
    # The caller's suffix stays on the wire; the hook appends the protocol
    # to whatever prompt the run assembles (default + that suffix).
    assert cmd["appendToSystemPrompt"] == "CALLER TEXT"
    assert cmd["systemPromptHook"]("BASE\n\nCALLER TEXT") == (
        "BASE\n\nCALLER TEXT\n\n" + autorouter_sea.SYSTEM_PROMPT
    )
    assert "systemPrompt" not in cmd
    assert cmd["model"] == autorouter_sea.orchestrator_model()
    assert "toolsFile" not in cmd and "tools" not in cmd
    # ``tools()``: the router's tools come on top of the basic toolset;
    # only a ``none`` tool profile removes it, and nothing stages
    # ``appendBasicTools`` any more.
    assert "appendBasicTools" not in cmd
    assert cmd["isParallel"] is False
    assert cmd["classifyTasks"] is False and cmd["useWebTools"] is False
    assert cmd["useMemory"] is False
    assert cmd["toolProfile"] == "full"
    assert {tool.__name__ for tool in cmd["toolsHook"]([])} == _TOOL_NAMES


def test_tier_candidates_are_distinct_catalog_models() -> None:
    """Every candidate is priced by the catalog and belongs to exactly one tier."""
    assert tuple(autorouter_sea.TIERS) == autorouter_sea.TIER_NAMES
    assert autorouter_sea.TIER_NAMES == ("small", "medium", "frontier")
    seen: set[str] = set()
    for tier, candidates in autorouter_sea.TIERS.items():
        assert candidates, tier
        for model, note in candidates:
            assert model in MODEL_INFO, f"{tier}: {model} is not in the model catalog"
            assert note
            assert model not in seen, f"{model} appears in two tiers"
            seen.add(model)


def test_model_menu_prices_every_candidate_from_the_catalog() -> None:
    """The menu lists each tier in order with catalog prices, cost estimate and availability."""
    tokens_in, tokens_out = 1_000_000, 100_000
    menu = json.loads(autorouter_sea.model_menu(tokens_in, tokens_out))
    available = set(get_available_models())
    assert list(menu) == list(autorouter_sea.TIER_NAMES)
    for tier, records in menu.items():
        assert [r["model"] for r in records] == [m for m, _ in autorouter_sea.TIERS[tier]]
        for record, (model, note) in zip(records, autorouter_sea.TIERS[tier], strict=True):
            info = MODEL_INFO[model]
            assert record["input_per_1M"] == info.input_price_per_1M
            assert record["output_per_1M"] == info.output_price_per_1M
            assert record["estimated_usd"] == round(
                info.input_price_per_1M + info.output_price_per_1M / 10, 4
            )
            assert record["runnable"] is (model in available)
            assert record["note"] == note
    # Defaults: 200k prompt + 20k completion tokens per unit.
    default = json.loads(autorouter_sea.model_menu())["small"][0]
    info = MODEL_INFO[default["model"]]
    assert default["estimated_usd"] == round(
        info.input_price_per_1M * 0.2 + info.output_price_per_1M * 0.02, 4
    )


def test_pick_model_returns_the_first_runnable_candidate_and_honours_exclude() -> None:
    """The pick is the first configured candidate; excluded models are skipped in order."""
    for tier in autorouter_sea.TIER_NAMES:
        runnable = _runnable_candidates(tier)
        if not runnable:
            pytest.skip(f"no provider credential configured for any {tier} candidate")
        pick = json.loads(autorouter_sea.pick_model(tier, 1_000_000, 100_000))
        assert pick["model"] == runnable[0]
        assert pick["tier"] == tier
        assert "runnable" not in pick
        info = MODEL_INFO[runnable[0]]
        assert pick["estimated_usd"] == round(
            info.input_price_per_1M + info.output_price_per_1M / 10, 4
        )
        if len(runnable) > 1:
            second = json.loads(
                autorouter_sea.pick_model(tier, exclude=f"{runnable[0]}, other-model")
            )
            assert second["model"] == runnable[1]
        everyone = " ".join(runnable)
        message = autorouter_sea.pick_model(tier, exclude=everyone)
        assert message.startswith(f"Error: no runnable model in tier {tier!r}")
        assert everyone in message
    assert autorouter_sea.pick_model("huge") == (
        "Error: unknown tier 'huge'; use one of small, medium, frontier."
    )


def test_estimate_cost_uses_catalog_prices_and_rejects_unknown_models() -> None:
    """The estimate is (input price * tokens_in + output price * tokens_out) / 1M."""
    model = autorouter_sea.TIERS["medium"][0][0]
    info = MODEL_INFO[model]
    estimate = json.loads(autorouter_sea.estimate_cost(model, 2_000_000, 500_000))
    assert estimate == {
        "model": model,
        "input_per_1M": info.input_price_per_1M,
        "output_per_1M": info.output_price_per_1M,
        "estimated_usd": round(info.input_price_per_1M * 2 + info.output_price_per_1M * 0.5, 4),
    }
    assert autorouter_sea.estimate_cost("no-such-model") == (
        "Error: unknown model 'no-such-model'; it is not in the local model catalog."
    )


def test_observed_evidence_cuts_an_over_long_file_at_a_line_break(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file over the cap is cut at the last line break within the cap, plus the marker.

    A line break right at the cap keeps the line it ends (the cap counts the
    text, not the break); a first line longer than the cap is cut mid-line,
    the only case with no break to cut at; a file at the cap is kept whole.
    """
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    cap, cut = autorouter_sea.EVIDENCE_MAX_CHARS, autorouter_sea.EVIDENCE_CUT
    evidence = tmp_path / "AUTOROUTER.md"
    lines = ["x" * 49] * 49 + ["Y" * 50]  # 49 * 50 + 50 = 2,500 chars, break at index 2,500
    evidence.write_text("\n".join(lines) + "\nTAIL\n", encoding="utf-8")
    assert autorouter_sea.observed_evidence() == "\n".join(lines) + f"\n\n{cut}"
    evidence.write_text("\n".join(lines) + "Z\nTAIL\n", encoding="utf-8")  # last line 51 wide
    assert autorouter_sea.observed_evidence() == "\n".join(lines[:-1]) + f"\n\n{cut}"
    evidence.write_text("X" * (cap + 1) + "\nTAIL\n", encoding="utf-8")
    assert autorouter_sea.observed_evidence() == "X" * cap + f"\n\n{cut}"
    evidence.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert autorouter_sea.observed_evidence() == "\n".join(lines)
    evidence.write_text(" \n", encoding="utf-8")
    assert autorouter_sea.observed_evidence() == autorouter_sea.NO_EVIDENCE
    evidence.unlink()
    assert autorouter_sea.observed_evidence() == autorouter_sea.NO_EVIDENCE


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
    assert autorouter_sea.ledger_path() == ledger
    assert (
        autorouter_sea.log_decision("read logs", "small", "gpt-6-luna", "tier=small 0.9")
        == f"logged to {ledger}"
    )
    assert (
        autorouter_sea.log_decision(
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
    assert autorouter_sea.log_decision("x", "huge", "m", "r") == (
        "Error: unknown tier 'huge'; use one of small, medium, frontier."
    )
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 6
    # A paragraph in a cell is cut at the cell cap so the row stays terse.
    words = " ".join(f"w{i}" for i in range(60))  # 229 chars
    assert autorouter_sea.log_decision(words, "frontier", "m", words, words).startswith("logged")
    row = ledger.read_text(encoding="utf-8").splitlines()[-1]
    cells = row.strip("| ").split(" | ")
    cut = words[:117].rstrip() + "..."
    assert cells[2:] == [cut, "frontier", "m", cut, cut] and len(cut) <= 120
    exact = "x" * autorouter_sea.CELL_MAX_CHARS
    autorouter_sea.log_decision(exact, "small", "m", "r")
    assert f"| {exact} | small |" in ledger.read_text(encoding="utf-8").splitlines()[-1]


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
    model = runnable[0] if runnable else autorouter_sea.TIERS["medium"][0][0]
    run = sea_commands.evaluate_sea(
        [AutorouterSea()], "add a --json flag to the export command"
    )
    settings = run.settings
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
    agent = WorktreeSorcarAgent("autorouter-sea-test")
    state = agent_state.AgentState(
        "autorouter-sea-test", agent=agent, task_thread=threading.current_thread()
    )
    agent_state.register(state)
    try:
        with serve(script) as (url, requests):
            result = agent.run(
                prompt_template=run.prompt,
                model_name=MODEL,
                work_dir=str(tmp_path),
                use_worktree=False,
                max_steps=5,
                max_budget=1.0,
                model_config={"base_url": url, "api_key": "local"},
                system_prompt_hook=run.system_prompt_hook,
                tools_hook=run.tools_hook,
                web_tools=settings["use_web_tools"],
                use_memory=settings["use_memory"],
                is_parallel=settings["allow_fan_out"],
                verbose=False,
            )
    finally:
        agent_state.unregister("autorouter-sea-test", state)
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
        assert autorouter_sea.SYSTEM_PROMPT in system, system[:200]
        assert system.startswith("<identity>"), system[:200]
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


_CHECKOUT = _SEA_PATH.parents[5]
"""The KISS checkout the SEA under test lives in (``src/kiss/agents/seas/autorouter``)."""

_OWNER = autorouter_sea.kiss_checkout(str(_CHECKOUT))
"""The durable checkout a weekly job scheduled from here points at: ``_CHECKOUT`` itself, or
the main checkout when the tests run inside a linked task worktree."""


def test_kiss_checkout_walks_up_to_the_checkout_and_rejects_other_dirs(tmp_path: Path) -> None:
    """Any directory inside the checkout maps to its root; foreign dirs and '' map to ''."""
    owner = Path(_OWNER)
    assert (owner / ".git").is_dir() and (owner / autorouter_sea.RSI7D_SEA_RELATIVE).is_file()
    assert _OWNER in (str(_CHECKOUT), str(autorouter_sea._owning_checkout(_CHECKOUT)))
    assert autorouter_sea.kiss_checkout(str(_SEA_PATH.parent)) == _OWNER
    assert autorouter_sea.kiss_checkout(str(tmp_path)) == ""
    assert autorouter_sea.kiss_checkout("") == ""
    # The SEA tree without a ``.git`` marker is not a checkout either.
    copy = tmp_path / "copy"
    (copy / autorouter_sea.RSI7D_SEA_RELATIVE).parent.mkdir(parents=True)
    (copy / autorouter_sea.RSI7D_SEA_RELATIVE).write_text("", encoding="utf-8")
    assert autorouter_sea.kiss_checkout(str(copy)) == ""
    (copy / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
    assert autorouter_sea.kiss_checkout(str(copy / "src")) == str(copy)


def test_picking_autorouter_schedules_one_enabled_weekly_rsi7d_job_and_resumes_a_paused_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hook creates the job once, keeps an enabled one, and re-enables a paused one.

    The cron store is the real one under a temporary ``$KISS_HOME``; the
    job is the relay pattern of the hand-scheduled weekly sweep: worktree
    plus auto-commit in the checkout, a ``run_agent`` directive to the
    checkout's rsi7d file whose task text starts with the ``autorouter``
    scope, and the child run's caps nested inside the job's.
    """
    from kiss.agents.seas.rsi7d.rsi7d_sea import parse_scope
    from kiss.agents.sorcar.cron_agent import cron_job, load_jobs

    home = tmp_path / "home"
    monkeypatch.setenv("KISS_HOME", str(home))
    assert load_jobs() == []

    created = AutorouterSea().on_picked_as_model(str(_SEA_PATH.parent))
    assert created.startswith("created:"), created
    (job,) = load_jobs()
    assert job["name"] == autorouter_sea.RSI7D_JOB_NAME
    assert job["enabled"] is True
    assert job["schedule"] == autorouter_sea.RSI7D_JOB_SCHEDULE == "0 1 * * 6"
    assert job["work_dir"] == _OWNER
    assert job["use_worktree"] is True and job["auto_commit"] is True
    assert job["max_budget"] == 30.0 and job["timeout"] == 7800.0
    assert job["model_name"] in (
        autorouter_sea.RSI7D_JOB_MODEL, autorouter_sea.orchestrator_model(),
    )
    assert job["model_name"] in get_available_models()
    prompt = job["prompt"]
    assert prompt.startswith("Call the run_agent tool IMMEDIATELY")
    assert f"agent        = {autorouter_sea.RSI7D_SEA_RELATIVE!r}" in prompt
    assert f"task         = {autorouter_sea.RSI7D_TASK!r}" in prompt
    assert "timeout      = '7200'" in prompt and "max_budget   = '25.0'" in prompt
    assert f"model        = {job['model_name']!r}" in prompt
    assert "options      = '{\"use_worktree\": false, \"auto_commit\": false}'" in prompt
    assert "model_name" not in prompt
    scope = parse_scope(autorouter_sea.RSI7D_TASK)
    assert scope.names == ("autorouter",) and scope.error == ""

    # A second pick (from another directory of the checkout) changes nothing.
    again = AutorouterSea().on_picked_as_model(str(_CHECKOUT / "src"))
    assert yaml.safe_load(again)["exists"]["id"] == job["id"], again
    assert [j["id"] for j in load_jobs()] == [job["id"]]

    # A paused job is resumed rather than duplicated.
    assert cron_job("pause", job_id=job["id"]).startswith("pause:")
    assert load_jobs()[0]["enabled"] is False
    resumed = AutorouterSea().on_picked_as_model(str(_CHECKOUT))
    assert yaml.safe_load(resumed)["resumed"]["id"] == job["id"], resumed
    (job_after,) = load_jobs()
    assert job_after["enabled"] is True and job_after["next_run_at"]


def test_picking_autorouter_outside_a_checkout_schedules_a_scratch_dir_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a checkout: no work dir, no worktree, and a relay to the installed rsi7d file."""
    from kiss.agents.sorcar.cron_agent import load_jobs

    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home"))
    assert AutorouterSea().on_picked_as_model(str(tmp_path)).startswith("created:")
    (job,) = load_jobs()
    assert job["work_dir"] == "" and job["use_worktree"] is False and job["auto_commit"] is False
    installed = _SEA_PATH.parents[1] / "rsi7d" / "rsi7d_sea.py"
    assert installed.is_file()
    assert f"agent        = {str(installed)!r}" in job["prompt"]


def test_kiss_checkout_resolves_a_linked_worktree_to_its_owning_checkout(tmp_path: Path) -> None:
    """A task worktree (``.git`` file) is discarded later, so the durable checkout is returned."""
    import subprocess

    repo = tmp_path / "repo"
    (repo / autorouter_sea.RSI7D_SEA_RELATIVE).parent.mkdir(parents=True)
    (repo / autorouter_sea.RSI7D_SEA_RELATIVE).write_text("", encoding="utf-8")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "PATH": __import__("os").environ["PATH"]}
    for args in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "seas"],
                 ["worktree", "add", "-q", str(tmp_path / "wt")]):
        subprocess.run(["git", *args], cwd=repo, env=env, check=True)
    worktree = tmp_path / "wt"
    assert (worktree / ".git").is_file()
    assert autorouter_sea.kiss_checkout(str(worktree / "src")) == str(repo.resolve())
    # A ``.git`` file that is not a linked worktree's stays where it is.
    other = tmp_path / "other"
    (other / autorouter_sea.RSI7D_SEA_RELATIVE).parent.mkdir(parents=True)
    (other / autorouter_sea.RSI7D_SEA_RELATIVE).write_text("", encoding="utf-8")
    (other / ".git").write_text("gitdir: /somewhere/else\n", encoding="utf-8")
    assert autorouter_sea.kiss_checkout(str(other)) == str(other.resolve())


def test_concurrent_first_picks_schedule_a_single_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Picks racing from a checkout and from a scratch dir create one job, not one each.

    Through the daemon's real hook runner (``run_picked_hook``), which loads
    the SEA file afresh per call and lets the hooks overlap: the cron
    store's ``ensure`` is what keeps the job single.
    """
    from kiss.agents.sorcar.cron_agent import load_jobs

    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home"))
    sea_commands._reset_for_tests()
    assert sea_commands.model_sea("autorouter") == _SEA_PATH
    barrier = threading.Barrier(8, timeout=30)

    def pick(work_dir: str) -> None:
        barrier.wait()
        sea_commands.run_picked_hook("autorouter", work_dir)

    threads = [
        threading.Thread(
            target=pick, args=(str(_CHECKOUT) if i % 2 else str(tmp_path),), daemon=True
        )
        for i in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not any(thread.is_alive() for thread in threads), "a picked-hook thread hung"
    assert len(load_jobs()) == 1, load_jobs()


def test_kiss_checkout_resolves_a_relative_gitdir_against_the_worktree(tmp_path: Path) -> None:
    """``gitdir: ../repo/.git/worktrees/wt`` names the checkout relative to the worktree."""
    repo = tmp_path / "repo"
    (repo / autorouter_sea.RSI7D_SEA_RELATIVE).parent.mkdir(parents=True)
    (repo / autorouter_sea.RSI7D_SEA_RELATIVE).write_text("", encoding="utf-8")
    (repo / ".git").mkdir()
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: ../repo/.git/worktrees/wt\n", encoding="utf-8")
    assert autorouter_sea.kiss_checkout(str(worktree)) == str(repo.resolve())


def test_hook_thread_that_cannot_start_is_logged_and_the_caller_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """With the process out of threads, ``run_picked_hook`` logs a warning instead of raising."""
    from kiss.agents.sorcar.cron_agent import load_jobs
    from kiss.tests.conftest import nproc_limit_lowered_to_one, thread_start_can_be_starved

    if not thread_start_can_be_starved():
        pytest.skip("RLIMIT_NPROC does not starve Thread.start here (Windows or root)")
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home"))
    sea_commands._reset_for_tests()
    with nproc_limit_lowered_to_one(), caplog.at_level("WARNING", logger="kiss.sea_commands"):
        sea_commands.run_picked_hook("autorouter", str(tmp_path))
    assert any("on_picked_as_model() not run" in r.getMessage() for r in caplog.records)
    assert load_jobs() == []
