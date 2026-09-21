# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``kiss.scripts.check.run_checks``.

A 24-hour audit of ``sorcar.db`` (2026-09-21) found ``uv run check
--full`` executed 65 times across 30 tasks: the script stopped at the
first failing stage, so an agent fixed ruff, re-ran, fixed mypy, re-ran,
fixed pyright, re-ran — one full model step and about a minute per
stage.  ``run_checks`` now runs every stage and returns all failures.

The stages here are real commands (``true``, ``false``, ``sh -c
'touch …'``, ``uv sync`` on a directory without a project), so the
observed behavior is the real subprocess path.  ``main()`` itself is
not executed: it runs the repository's whole ruff/mypy/pyright pass,
which takes minutes and depends on the state of the checkout.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from kiss.scripts.check import run_checks

pytestmark = pytest.mark.skipif(
    shutil.which("sh") is None, reason="needs a POSIX shell"
)


def _touch_stage(marker: Path, description: str) -> tuple[list[str], str]:
    return (["sh", "-c", f"touch '{marker}'"], description)


def test_all_stages_run_after_a_failure(tmp_path: Path) -> None:
    """A failing stage does not stop the later stages from running."""
    marker = tmp_path / "later-stage-ran"
    failed = run_checks([
        (["false"], "Lint code (ruff)"),
        _touch_stage(marker, "Type check (mypy)"),
        (["false"], "Type check (pyright)"),
    ])
    assert failed == ["Lint code (ruff)", "Type check (pyright)"]
    assert marker.exists(), "the stage after the first failure must still run"


def test_all_passing_returns_empty(tmp_path: Path) -> None:
    """No failures gives an empty list."""
    marker = tmp_path / "ran"
    assert run_checks([(["true"], "a"), _touch_stage(marker, "b")]) == []
    assert marker.exists()


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_failed_uv_sync_stops_the_run(tmp_path: Path) -> None:
    """``uv sync`` is the prerequisite: when it fails nothing else runs."""
    marker = tmp_path / "must-not-run"
    empty_project = tmp_path / "no-project"
    empty_project.mkdir()
    failed = run_checks([
        (["uv", "sync", "--project", str(empty_project)], "Install dependencies (uv sync)"),
        _touch_stage(marker, "Lint code (ruff)"),
    ])
    assert failed == ["Install dependencies (uv sync)"]
    assert not marker.exists()
