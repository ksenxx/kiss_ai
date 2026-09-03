# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-02 (sorcar-agents): the final history save must not blank ``work_dir``.

``ChatSorcarAgent.run`` persists the run's settings twice: an early
row (from the resolved kwargs) and a final save in its ``finally``
(from the live agent state).  The final save already guards ``model``
and ``max_budget`` against the case where setup failed BEFORE
``super().run`` ran ``_reset`` (a broken printer hook), but it read
``work_dir`` from ``self.work_dir`` — which is ``""`` on a fresh agent
(and the PREVIOUS run's directory on a reused one).  The final
``UPDATE`` therefore overwrote the early row's correct ``work_dir``
with ``""`` (or a stale directory) for every run that died during
setup, and the history sidebar showed the task without a project.

Real SQLite history in an isolated ``KISS_HOME``; the printer is a
real ``CapturePrinter`` whose ``start_recording`` raises, which is the
kind of setup failure the guard exists for.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.tests.server.parallel_agent_harness import (
    CapturePrinter,
    IsolatedKissHome,
)


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB."""
    isolated = IsolatedKissHome("kiss-audit0902-workdir-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


class _RecorderDown(CapturePrinter):
    """A real server printer whose recording start fails."""

    def start_recording(self) -> None:
        raise RuntimeError("recording unavailable")


def _row_work_dir(task_id: str) -> str:
    db = persistence._get_db()
    with persistence._rw_lock.read_lock():
        row = db.execute(
            "SELECT work_dir FROM task_history WHERE id = ?", (task_id,),
        ).fetchone()
    assert row is not None
    return str(row["work_dir"])


def test_setup_failure_keeps_resolved_work_dir(env: IsolatedKissHome) -> None:
    """A run that dies before ``_reset`` keeps the early row's work_dir."""
    project = Path(tempfile.mkdtemp(prefix="audit0902-project-", dir=env.tmpdir))
    agent = ChatSorcarAgent("audit0902-workdir")
    with pytest.raises(RuntimeError, match="recording unavailable"):
        agent.run(
            prompt_template="never runs",
            work_dir=str(project),
            printer=_RecorderDown(),
            model_name="claude-fable-5-1",
        )
    task_id = agent.last_task_id
    assert task_id
    assert _row_work_dir(task_id) == str(project.resolve())


def test_setup_failure_on_reused_agent_does_not_record_previous_dir(
    env: IsolatedKissHome,
) -> None:
    """A reused agent must not stamp the previous run's directory."""
    first = Path(tempfile.mkdtemp(prefix="audit0902-first-", dir=env.tmpdir))
    second = Path(tempfile.mkdtemp(prefix="audit0902-second-", dir=env.tmpdir))
    agent = ChatSorcarAgent("audit0902-reused")
    # Emulate the state a completed run leaves behind: _reset resolved
    # the previous directory onto the agent.
    agent.work_dir = str(first.resolve())
    with pytest.raises(RuntimeError):
        agent.run(
            prompt_template="never runs either",
            work_dir=str(second),
            printer=_RecorderDown(),
            model_name="claude-fable-5-1",
        )
    assert _row_work_dir(agent.last_task_id) == str(second.resolve())
