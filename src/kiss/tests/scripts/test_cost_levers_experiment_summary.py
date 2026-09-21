# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the pure aggregation of ``kiss.scripts.cost_levers_experiment``.

The experiments themselves call paid models and are run by hand; only
the averaging/percentage logic is tested here.
"""

from __future__ import annotations

import pytest

from kiss.scripts.cost_levers_experiment import Run, summarize


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
