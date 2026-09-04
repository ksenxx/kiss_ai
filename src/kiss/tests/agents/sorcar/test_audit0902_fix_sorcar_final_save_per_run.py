# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Review fixes #5 and #8: the final history save describes THIS run only.

``ChatSorcarAgent.run`` persists a run twice: an early row from the
resolved kwargs and a final ``UPDATE`` from its ``finally``.

* **#5** When setup fails BEFORE ``super().run`` (a broken printer hook)
  the agent's ``_launch_model_name`` / ``model_name``, ``_is_parallel``,
  ``total_tokens_used``, ``budget_used`` and ``total_steps`` still hold
  the PREVIOUS run's values on a reused agent, and the final save
  copied them into the new, never-run task's row: another task's
  model, parallel mode, tokens and cost were charged to it.
* **#8** The standalone final save persisted ``tokens`` and ``cost`` but
  not ``steps`` although the schema has the column and the server-owned
  completion path fills it.

Real SQLite history in an isolated ``KISS_HOME``; the printer is the
real ``CapturePrinter`` (``JsonPrinter``) whose ``start_recording``
raises; the run whose bookkeeping is checked is a real
``ChatSorcarAgent.run`` whose ``super().run`` resolves to a
``SorcarAgent`` subclass that reports usage without a live model (the
same MRO pattern ``test_replay_events_outside_webview.py`` uses).
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.tests.server.parallel_agent_harness import (
    CapturePrinter,
    IsolatedKissHome,
)

_COLUMNS = (
    "model", "work_dir", "is_parallel", "is_worktree", "tokens", "cost",
    "steps", "max_budget", "auto_commit_mode",
)


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB."""
    isolated = IsolatedKissHome("kiss-audit0902-fix-final-save-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


class _RecorderDown(CapturePrinter):
    """A real server printer whose recording start fails."""

    def start_recording(self) -> None:
        raise RuntimeError("recording unavailable")


class _UsageReportingAgent(SorcarAgent):
    """A ``SorcarAgent`` whose ``run`` resolves the run like ``_reset`` does
    and reports a fixed spend, without invoking a model."""

    usage: tuple[float, int, int] = (5e-06, 15, 3)

    def run(self, prompt_template: str = "", **kwargs: Any) -> str:  # type: ignore[override]
        self._launch_model_name = self._resolve_model_name(kwargs.get("model_name"))
        self.model_name = self._launch_model_name
        self._is_parallel = bool(kwargs.get("is_parallel", True))
        self.work_dir = str(Path(kwargs.get("work_dir") or ".").resolve())
        self.budget_used, self.total_tokens_used, self.total_steps = self.usage
        return str(yaml.dump({"success": True, "summary": "done"}, sort_keys=False))


class _OfflineChatAgent(ChatSorcarAgent, _UsageReportingAgent):
    """Real ``ChatSorcarAgent.run`` bookkeeping over the offline run above."""


def _row(task_id: str) -> dict[str, Any]:
    db = persistence._get_db()
    with persistence._rw_lock.read_lock():
        row = db.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM task_history WHERE id = ?",
            (task_id,),
        ).fetchone()
    assert row is not None
    return {col: row[col] for col in _COLUMNS}


def test_standalone_run_persists_steps_with_tokens_and_cost(env: IsolatedKissHome) -> None:
    """#8: a completed standalone run records its step count."""
    agent = _OfflineChatAgent("audit0902-fix-steps")
    agent.run(
        prompt_template="count my steps",
        model_name="claude-fable-5-1",
        work_dir=str(env.repo),
        is_parallel=False,
        max_budget=2.5,
    )
    row = _row(agent.last_task_id)
    assert row["steps"] == 3, row
    assert row["tokens"] == 15
    assert row["cost"] == pytest.approx(5e-06)
    assert row["model"] == "claude-fable-5-1"
    assert row["is_parallel"] == 0
    assert row["work_dir"] == str(env.repo.resolve())
    assert row["max_budget"] == pytest.approx(2.5)


def test_setup_failure_on_reused_agent_records_this_runs_settings_and_zero_usage(
    env: IsolatedKissHome,
) -> None:
    """#5: a reused agent whose second run dies before ``super().run``
    persists the second run's model/mode and no usage — not the first's."""
    first_dir = env.repo
    second_dir = Path(tempfile.mkdtemp(prefix="audit0902-second-", dir=env.tmpdir))
    agent = _OfflineChatAgent("audit0902-fix-reused")
    agent.usage = (12.34, 1234, 7)
    agent.run(
        prompt_template="first run",
        model_name="claude-fable-5-1",
        work_dir=str(first_dir),
        is_parallel=True,
        max_budget=20.0,
    )
    first_id = agent.last_task_id
    assert _row(first_id) == {
        "model": "claude-fable-5-1",
        "work_dir": str(first_dir.resolve()),
        "is_parallel": 1,
        "is_worktree": 0,
        "tokens": 1234,
        "cost": pytest.approx(12.34),
        "steps": 7,
        "max_budget": pytest.approx(20.0),
        "auto_commit_mode": 1,
    }

    with pytest.raises(RuntimeError, match="recording unavailable"):
        agent.run(
            prompt_template="never runs",
            model_name="gpt-5.6-sol",
            work_dir=str(second_dir),
            is_parallel=False,
            max_budget=1.5,
            printer=_RecorderDown(),
        )
    second_id = agent.last_task_id
    assert second_id and second_id != first_id
    assert _row(second_id) == {
        "model": "gpt-5.6-sol",
        "work_dir": str(second_dir.resolve()),
        "is_parallel": 0,
        "is_worktree": 0,
        "tokens": 0,
        "cost": 0.0,
        "steps": 0,
        "max_budget": pytest.approx(1.5),
        "auto_commit_mode": 1,
    }
    # The first run's row is untouched.
    assert _row(first_id)["tokens"] == 1234


def test_setup_failure_on_fresh_agent_records_resolved_settings(
    env: IsolatedKissHome,
) -> None:
    """A never-run agent has no usage fields at all; the row still gets
    the resolved model/mode and zero usage."""
    agent = ChatSorcarAgent("audit0902-fix-fresh")
    with pytest.raises(RuntimeError, match="recording unavailable"):
        agent.run(
            prompt_template="never runs",
            model_name="claude-fable-5-1",
            work_dir=str(env.repo),
            printer=_RecorderDown(),
        )
    row = _row(agent.last_task_id)
    assert row["model"] == "claude-fable-5-1"
    assert row["is_parallel"] == 1
    assert (row["tokens"], row["cost"], row["steps"]) == (0, 0.0, 0)
