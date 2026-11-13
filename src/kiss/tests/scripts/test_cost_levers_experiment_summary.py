# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the pure aggregation of ``kiss.scripts.cost_levers_experiment``.

The experiments themselves call paid models and are run by hand; only
the averaging/percentage logic, the bound on E3's ``git show``, the
session's temporary home and the fate of a run that fails (an unknown
model: no call is made) are tested here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.core.config import DEFAULT_CONFIG
from kiss.scripts.cost_levers_experiment import (
    LEVERS_OFF,
    Run,
    e1_read_hygiene,
    e2_review_profile,
    main,
    summarize,
)
from kiss.tests.conftest import posix_only


def test_summarize_averages_repeats_and_computes_changes() -> None:
    runs = [
        Run("E1", "off", 0, 100_000, 1.00, 10, 60.0),
        Run("E1", "on", 0, 60_000, 0.50, 8, 40.0),
        Run("E1", "off", 1, 120_000, 1.20, 12, 70.0),
        Run("E1", "on", 1, 50_000, 0.60, 8, 30.0),
        Run("E3", "previous", 0, 10, 0.0, 2, 1.0),
        Run("E3", "current", 0, 5, 0.0, 1, 1.0),
    ]
    rows = summarize(runs)
    assert [r["experiment"] for r in rows] == ["E1", "E3"]
    e1 = rows[0]
    assert e1["baseline"] == "off" and e1["lever"] == "on"
    assert e1["baseline_avg"]["tokens"] == 110_000 and e1["lever_avg"]["tokens"] == 55_000
    assert e1["cost_change"] == pytest.approx(-0.5)
    assert e1["token_change"] == pytest.approx(-0.5)
    assert e1["lever_avg"]["steps"] == 8
    # Zero-cost baselines do not divide by zero.
    assert rows[1]["cost_change"] == 0.0 and rows[1]["token_change"] == -0.5


def _process_state() -> tuple[object, ...]:
    """What a session redirects, and the temporary homes sessions have left behind."""
    homes = sorted(p.name for p in Path(tempfile.gettempdir()).glob("kiss_cost_experiment_*"))
    return (os.environ.get("KISS_HOME"), os.environ.get("KISS_MUSE_AUTH"), th._KISS_DIR,
            th._DB_PATH, homes)


def test_a_session_removes_its_temporary_home_and_restores_the_process(tmp_path: Path) -> None:
    """``main`` leaves no ``kiss_cost_experiment_*`` directory and no redirection behind."""
    before = _process_state()
    out = tmp_path / "results.json"
    assert main(["--repeats", "0", "--out", str(out)]) == 0
    assert json.loads(out.read_text()) == {"model": "claude-haiku-4-5", "runs": [], "summary": []}
    assert _process_state() == before
    with pytest.raises(SystemExit):  # an experiment that does not exist, before anything runs
        main(["--repeats", "0", "--only", "E1,E9", "--out", str(out)])
    assert _process_state() == before


def test_a_run_that_fails_is_not_a_measurement() -> None:
    """A run that did not finish its task ends the experiment: its near-zero usage read as savings.

    The levers an arm switched are back where they were, for whatever runs next.
    """
    with pytest.raises(RuntimeError, match="Unknown model name: no-such-model"):
        e2_review_profile("no-such-model", 0)
    levers = {name: getattr(DEFAULT_CONFIG, name) for name in LEVERS_OFF}
    with pytest.raises(RuntimeError, match="Unknown model name: no-such-model"):
        e1_read_hygiene("no-such-model", 0)
    assert {name: getattr(DEFAULT_CONFIG, name) for name in LEVERS_OFF} == levers


@pytest.mark.slow
@posix_only("stalls git on a FIFO")
def test_a_stalled_git_show_fails_instead_of_hanging_the_experiment(tmp_path: Path) -> None:
    """E3's lookup of the previous prompt gives up on a ``git show`` that stalls.

    Real git blocks opening its ``GIT_TRACE`` file while the FIFO has no
    reader; the child waits out the script's own bound (30 s) and fails.
    """
    fifo = tmp_path / "trace"
    os.mkfifo(fifo)
    code = "from kiss.scripts import cost_levers_experiment as e; e._previous_system_prompt()"
    try:
        done = subprocess.run(
            [sys.executable, "-c", code], env={**os.environ, "GIT_TRACE": str(fifo)},
            capture_output=True, text=True, timeout=300, check=False,
        )
    finally:
        os.close(os.open(fifo, os.O_RDONLY | os.O_NONBLOCK))  # frees a git still blocked on it
    assert done.returncode == 1 and "subprocess.TimeoutExpired" in done.stderr, done.stderr
