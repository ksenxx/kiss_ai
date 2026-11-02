# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The finish gate's review is part of a headless run: its trajectory and its steps.

``run_sea`` enters a ``collecting_trajectories`` scope (a ``ContextVar``)
so the result lists every trajectory file the run wrote, or none when
``save_trajectories=False``.  The coding gate runs its reviewer on a
thread of its own, which started with an empty context: the reviewer's
agent saw no scope, wrote its trajectory even with saving off, and
with saving on the file was missing from ``trajectory_paths``, so a
benchmark loop could not delete it (audit 2026-11-01, S1 F3).  The gate
also charged the review's money and tokens to the run but dropped its
model calls, so ``total_steps`` undercounted every review (S1 F4).

The gate threads now run under a copy of the calling thread's context
and the review's steps are charged with its spend.  Real ``run_sea`` of a
SEA pinned to the scripted local model server, which answers both the
run and its reviewer; a real verifier; nothing is mocked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kiss.agents.sorcar.headless import run_sea
from kiss.core import config as config_module
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    text_body,
)

URL_ENV = "AUDIT1101_GATE_SEA_URL"

SEA_SOURCE = f'''\
import os
from typing import Any

from kiss.agents.seas.base.base_sea import WorkerSea


class GateSea(WorkerSea):
    """Fixture SEA: an editing profile (so the finish is reviewed) on the local server."""

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        return settings | {{
            "model": {MODEL!r},
            "tool_profile": "shell+edit",
            "auto_classify": False,
            "model_config": {{
                "base_url": os.environ[{URL_ENV!r}], "api_key": "local", "plain_io": True,
            }},
        }}
'''


@pytest.fixture
def gate_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    """An isolated home (the reviewer is the local model), artifact dir, SEA and work dir.

    Returns:
        ``(sea_path, work_dir, artifacts)``: the SEA file, the work dir
        with a passing verifier, and the directory trajectories go to.
    """
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.json").write_text(json.dumps({"coding_gate_reviewer": MODEL}))
    monkeypatch.setenv("KISS_HOME", str(home))
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(config_module, "_artifact_dir", str(artifacts))
    sea_path = tmp_path / "gate" / "gate_sea.py"
    sea_path.parent.mkdir()
    sea_path.write_text(SEA_SOURCE, encoding="utf-8")
    work = tmp_path / "work"
    (work / "tmp").mkdir(parents=True)
    (work / "tmp" / "verify.sh").write_text("exit 0\n")
    return sea_path, work, artifacts


def _script() -> list[bytes]:
    """The run finishes at once; the reviewer (the next request) approves."""
    return [finish_body("<p>Completed.</p>"), text_body("VERDICT: OK")]


def test_with_saving_off_the_reviewer_writes_no_trajectory(
    gate_run: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    sea_path, work, artifacts = gate_run
    with serve(_script()) as (url, requests):
        monkeypatch.setenv(URL_ENV, url)
        outcome = run_sea(
            sea_path, "Check the prepared verifier.", work_dir=str(work), max_budget=0.2,
            save_trajectories=False,
        )
    assert outcome.error == "" and outcome.success, outcome
    assert len(requests) == 2, "the finish, then the gate's review"
    assert outcome.trajectory_paths == ()
    assert sorted(artifacts.rglob("*.yaml")) == []


def test_the_reviewer_trajectory_and_steps_are_part_of_the_result(
    gate_run: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    sea_path, work, artifacts = gate_run
    with serve(_script()) as (url, requests):
        monkeypatch.setenv(URL_ENV, url)
        outcome = run_sea(
            sea_path, "Check the prepared verifier.", work_dir=str(work), max_budget=0.2,
        )
    assert outcome.error == "" and outcome.success, outcome
    assert len(requests) == 2, "the finish, then the gate's review"
    written = set(artifacts.rglob("*.yaml"))
    assert set(outcome.trajectory_paths) == written
    assert any("Finish_reviewer" in path.name for path in written), sorted(written)
    # F4: the review's model call is a step of the run, like its spend.
    assert outcome.total_steps == len(requests)
    assert outcome.budget_used > 0 and outcome.total_tokens_used > 0
