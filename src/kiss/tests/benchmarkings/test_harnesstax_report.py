# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests of the HarnessTax ``analyze`` and ``report`` tools.

They run on the synthetic results tree built by the ``tree`` fixture of
``test_harnesstax_tooling.py``, and on the recorded study when it is present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.tests.benchmarkings.test_harnesstax_tooling import (  # noqa: F401
    MODEL,
    PHASE,
    REPO_ROOT,
    run_tool,
    tree,
)


def test_analyze_and_report(tree: dict[str, Path]) -> None:  # noqa: F811 (pytest fixture)
    """The trajectory analysis and the HTML report run end to end on the tree."""
    run_tool(tree, "audit", "--phase", PHASE, "--quarantine")
    proc = run_tool(tree, "aggregate", "--phase", PHASE)
    assert proc.returncode == 0, proc.stderr
    proc = run_tool(
        tree,
        "analyze",
        "--phase",
        PHASE,
        "--failures-only",
        "--benchmark",
        "swebench-lite",
        "--rep",
        "2",
    )
    assert proc.returncode == 0, proc.stderr
    assert "gave up" in proc.stdout and "== per model ==" in proc.stdout
    proc = run_tool(tree, "analyze", "--phase", PHASE, "--model", MODEL)
    assert proc.returncode == 0, proc.stderr
    assert "infra/timeout" in proc.stdout and "solved" in proc.stdout
    out = tree["root"] / "report.html"
    proc = run_tool(tree, "report", "--phase", PHASE, "--out", str(out))
    assert (
        proc.returncode != 0
        and "incomplete results" in proc.stderr
        and "claude-fable-5" in proc.stderr
    )
    proc = run_tool(tree, "report", "--phase", PHASE, "--out", str(out), "--allow-incomplete")
    assert proc.returncode == 0, proc.stderr
    html = out.read_text()
    # per benchmark: scatter, success bars, cost bars, task grid, turn-cap and spend-cap
    # curves (6 x 2), plus the wall-clock cap curve that only Terminal-Bench gets
    assert html.count("<svg") == EXPECTED_SVG_COUNT and "GPT-5.6 Luna · KISS Sorcar" in html
    assert "Incomplete run" in html and "gpt-5.6-luna 87 slot(s) off" in html
    assert "n/a</text>" not in html  # the blog has per-task Pi data for luna on both benchmarks
    assert "1/2 · 0/3</text>" in html  # astropy: KISS solved one of two attempts, Pi none


RECORDED_SUMMARY = (
    REPO_ROOT / "benchmarkings" / "harnesstax" / "results" / "baseline" / "summary.json"
)
EXPECTED_SVG_COUNT = 6 * 2 + 1


@pytest.mark.skipif(not RECORDED_SUMMARY.is_file(), reason="recorded baseline results not present")
def test_report_prose_on_the_recorded_study() -> None:
    """The full discussion renders from the recorded study and every claim check holds."""
    from benchmarkings.harnesstax import report

    html = report.render("baseline")
    assert (
        "Every per-model success rate" in html
        or "fall inside the blog's confidence interval" in html
    )
    assert "Incomplete run" not in html
    assert html.count("<svg") == EXPECTED_SVG_COUNT
