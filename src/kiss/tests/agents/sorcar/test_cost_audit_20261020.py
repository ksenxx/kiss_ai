# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests for the October 20, 2026 cost audit.

- The ``Budget:`` usage line a relentless agent's model sees counts the
  task's whole spend (sub-agent spend folded mid-session included)
  against the task's cap, as the budget check does.
- A finished task's row keeps its exact cost, not a 6-decimal rounding.
- The commit-message model call is charged to the task it commits for.

The relentless run drives a real :class:`RelentlessAgent` through a
stand-in ``claude`` executable on PATH (an API-key-billed ``cc/`` turn).
The commit-message tests call the real fast model and skip without an
API key for it.  Not covered: the ``except`` around
``_add_late_task_usage`` in ``SorcarAgent._retire_after_task_saved``,
reached only when the SQLite write itself fails.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.commit_message import generate_commit_message_from_diff
from kiss.agents.sorcar.persistence import _add_task, _get_db
from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.agents.sorcar.sorcar_agent import _agent_usage, _attribute_sub_usage
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.models.model_info import get_fast_model
from kiss.tests.conftest import install_cli_script, posix_only, skip_if_no_api_key_for_model

FOLDED_CHILD_COST = 2.0
MAX_BUDGET = 10.0

_FAKE_CLAUDE = """#!/usr/bin/env python3
import json, pathlib, sys
record_dir = pathlib.Path({record_dir!r})
n = len(list(record_dir.glob("call-*.txt")))
(record_dir / f"call-{{n}}.txt").write_text(sys.stdin.read())
turns = {turns!r}
for event in turns[min(n, len(turns) - 1)]:
    print(json.dumps(event), flush=True)
"""


def _claude_turn(text: str) -> list[dict[str, Any]]:
    """One ``claude --print`` stream answering *text* with 1000/100 tokens."""
    usage = {"input_tokens": 1000, "output_tokens": 100}
    return [
        {"type": "message_start", "message": {"model": "claude-sonnet-4-5", "usage": usage}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": usage},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}},
        {"type": "result", "result": text, "usage": usage},
    ]


_AGENT: list[RelentlessAgent] = []


def fold_child_spend() -> str:
    """Fold a finished sub-agent's spend into the running task.

    Returns:
        A short confirmation.
    """
    _attribute_sub_usage(_AGENT[0], FOLDED_CHILD_COST, 100, 0)
    return "folded"


_TOOL_CALL = json.dumps({"tool_calls": [{"name": "fold_child_spend", "arguments": {}}]})
_FINISH_CALL = json.dumps({
    "tool_calls": [{
        "name": "finish",
        "arguments": {"success": True, "is_continue": False, "summary_in_html": "done"},
    }],
})


@posix_only("the fake claude is a POSIX script")
def test_budget_line_counts_sub_agent_spend_and_the_task_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a child's spend is folded mid-session, the next usage line
    shows the task's cumulative spend against the task cap."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-test")
    monkeypatch.setattr(DEFAULT_CONFIG, "CLAUDE_CODE_OAUTH_TOKEN", "")
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
    bin_dir, record_dir = tmp_path / "bin", tmp_path / "records"
    bin_dir.mkdir()
    record_dir.mkdir()
    turns = [_claude_turn(_TOOL_CALL), _claude_turn(_TOOL_CALL), _claude_turn(_FINISH_CALL)]
    install_cli_script(
        bin_dir / "claude", _FAKE_CLAUDE.format(record_dir=str(record_dir), turns=turns),
    )
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    agent = RelentlessAgent("budget line e2e")
    _AGENT[:] = [agent]
    agent.run(
        model_name="cc/sonnet",
        prompt_template="Fold the child's spend, then finish.",
        max_budget=MAX_BUDGET,
        work_dir=str(tmp_path),
        tools=[fold_child_spend],
        verbose=False,
        model_config={"timeout": 30},
    )
    # Each step's tool folds one child; the prompt right after it must
    # already count it.
    for call, folds in (("call-1.txt", 1), ("call-2.txt", 2)):
        prompt = (record_dir / call).read_text()
        spent, cap = map(float, re.findall(r"Budget: \$([\d.]+)/\$([\d.]+)", prompt)[-1])
        assert cap == MAX_BUDGET
        assert folds * FOLDED_CHILD_COST < spent < (folds + 1) * FOLDED_CHILD_COST
    assert _agent_usage(agent)[0] > 2 * FOLDED_CHILD_COST


def test_final_row_keeps_the_exact_cost() -> None:
    """A sub-micro-dollar spend is persisted as is, not rounded to zero."""
    extra: dict[str, Any] = {"cost": 0.0, "tokens": 0, "steps": 0}
    task_id, _ = _add_task("exact cost task", "", extra)
    agent = ChatSorcarAgent("exact-cost")
    exact = 1.234567891e-7
    _attribute_sub_usage(agent, exact, 3, 1)
    payload: dict[str, Any] = {"endTs": int(time.time() * 1000)}
    agent._save_final_usage(payload, task_id)
    assert payload["cost"] == exact
    row = _get_db().execute(
        "SELECT cost FROM task_history WHERE id = ?", (task_id,),
    ).fetchone()
    assert float(row[0]) == pytest.approx(exact, rel=1e-9)


def _skip_unless_fast_model_is_api_billed() -> None:
    """Skip when the fast model has no API key or is a subscription CLI model."""
    fast = get_fast_model()
    if fast.startswith(("cc/", "codex/")):
        pytest.skip(f"fast model {fast} may bill a subscription at $0")
    skip_if_no_api_key_for_model(fast)


def test_commit_message_call_is_charged_to_the_task() -> None:
    """The fast model's commit-message call adds its spend to *bill_to*."""
    _skip_unless_fast_model_is_api_billed()
    agent = RelentlessAgent("commit-billing")
    diff = "diff --git a/x.txt b/x.txt\n+hello\n"
    message = generate_commit_message_from_diff(diff, bill_to=agent)
    assert message
    budget, tokens, steps = _agent_usage(agent)
    assert budget > 0
    assert tokens > 0
    assert steps == 0


def _finished_row(cost: float) -> str:
    """Add a finished task row costing *cost*; return its id."""
    extra: dict[str, Any] = {
        "cost": cost, "tokens": 100, "steps": 2, "endTs": int(time.time() * 1000),
    }
    return _add_task("retired worktree task", "", extra)[0]


def _row_cost(task_id: str) -> float:
    """The persisted ``cost`` of *task_id*."""
    row = _get_db().execute("SELECT cost FROM task_history WHERE id = ?", (task_id,)).fetchone()
    return float(row[0])


@pytest.mark.parametrize("retire", ["new_chat", "retire_for_disposal"])
def test_retiring_a_pending_worktree_charges_the_finished_task(
    tmp_path: Path, retire: str,
) -> None:
    """A worktree left pending by a finished task is committed (with a
    model-written message) when the chat is cleared or the tab disposed;
    that call's spend is added to the finished task's row and transcript."""
    from kiss.agents.sorcar.persistence import _flush_chat_events, _load_chat_events_by_task_id
    from kiss.tests.agents.sorcar.test_autocommit_notifications import (
        _RecordingPrinter,
        _setup_worktree_agent,
    )

    _skip_unless_fast_model_is_api_billed()
    agent, _repo, wt_dir = _setup_worktree_agent(tmp_path, f"retire-{retire.replace('_', '-')}")
    printer = _RecordingPrinter(wt_dir)
    agent.printer = printer  # type: ignore[assignment]
    task_id = _finished_row(0.5)
    agent._last_task_id = task_id
    (wt_dir / "work.txt").write_text("unmerged work\n")
    before = _agent_usage(agent)[0]

    getattr(agent, retire)()

    spent = _agent_usage(agent)[0] - before
    assert spent > 0
    assert _row_cost(task_id) == pytest.approx(0.5 + spent)
    _flush_chat_events(task_id)
    session = _load_chat_events_by_task_id(task_id)
    assert session is not None
    usage = [e for e in session["events"] if e.get("type") == "usage_info"]  # type: ignore[attr-defined]
    assert usage[-1]["cost_usd"] == pytest.approx(0.5 + spent)
    assert {"type": "tasks_updated"} in printer.events


def test_retiring_without_a_task_row_changes_nothing(tmp_path: Path) -> None:
    """An agent that never saved a row keeps the spend on its counters only."""
    from kiss.tests.agents.sorcar.test_autocommit_notifications import _setup_worktree_agent

    _skip_unless_fast_model_is_api_billed()
    agent, _repo, wt_dir = _setup_worktree_agent(tmp_path, "retire-no-row")
    (wt_dir / "work.txt").write_text("unmerged work\n")
    agent.new_chat()
    assert _agent_usage(agent)[0] > 0


def test_retiring_for_a_running_task_leaves_its_row_to_the_final_save(tmp_path: Path) -> None:
    """A row without an end time is still running: its own final save
    reads the live counters, so the row is not charged here."""
    from kiss.tests.agents.sorcar.test_autocommit_notifications import _setup_worktree_agent

    _skip_unless_fast_model_is_api_billed()
    agent, _repo, wt_dir = _setup_worktree_agent(tmp_path, "retire-running")
    task_id = _add_task("running task", "", {"cost": 0.5, "tokens": 100, "steps": 2})[0]
    agent._last_task_id = task_id
    (wt_dir / "work.txt").write_text("unmerged work\n")
    agent.retire_for_disposal()
    assert _agent_usage(agent)[0] > 0
    assert _row_cost(task_id) == pytest.approx(0.5)


class _InterruptedRetireAgent(ChatSorcarAgent):
    """Agent whose retirement is billed for its commit message, then stopped."""

    def _retire_previous_worktree(self) -> bool:
        """Bank a commit-message call's spend, then raise as a Stop would."""
        _attribute_sub_usage(self, 0.25, 40, 0)
        raise KeyboardInterrupt("stopped mid-retirement")


def test_interrupted_retirement_still_charges_the_finished_task() -> None:
    """Spend billed before a Stop interrupts the retirement reaches the row."""
    agent = _InterruptedRetireAgent("interrupted-retire")
    task_id = _finished_row(0.5)
    agent._last_task_id = task_id
    with pytest.raises(KeyboardInterrupt):
        agent._retire_after_task_saved()
    assert _row_cost(task_id) == pytest.approx(0.75)
