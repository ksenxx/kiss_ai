# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of where the dev scripts under ``scripts/`` write by default.

``ui_style_probe.py`` and ``transport_bench/run_bench.py`` were run from the
checkout and wrote into its ``tmp/``; their default output is now in the
agents' temp directory, ``<OS temp dir>/kiss-tmp``.  The real scripts run as
child processes that have a temp directory of their own and start in an
empty folder.

The benchmark itself is not run here: it drives a daemon for minutes and
reads ``/proc``.  Its ``--help`` names the default it computed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from kiss.tests.conftest import posix_only

SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"


def _run(script: str, tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run *script* of ``scripts/`` in the empty folder ``cwd`` of *tmp_path*.

    Its temp directory is the folder ``temp`` there.
    """
    temp, cwd = tmp_path / "temp", tmp_path / "cwd"
    temp.mkdir(exist_ok=True)
    cwd.mkdir(exist_ok=True)
    env = {**os.environ, "TMPDIR": str(temp), "TEMP": str(temp), "TMP": str(temp), "COLUMNS": "500"}
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args], cwd=cwd, env=env,
        capture_output=True, text=True, timeout=600, check=False,
    )


@posix_only("the benchmark imports resource")
def test_the_transport_benchmark_names_its_results_in_the_agents_temp_directory(
    tmp_path: Path,
) -> None:
    done = _run("transport_bench/run_bench.py", tmp_path, "--help")
    assert done.returncode == 0, done.stderr
    default = (tmp_path / "temp").resolve() / "kiss-tmp" / "transport_bench" / "results.json"
    assert f"raw results file (default: {default})" in done.stdout
    assert list((tmp_path / "cwd").iterdir()) == []


def test_the_style_probe_writes_its_report_to_the_agents_temp_directory(tmp_path: Path) -> None:
    """Without a path the report is ``ui-style-probe.txt`` in the agents' temp directory.

    A path on the command line still names the report.  Each run drives a
    real browser through every scene of the chat page; where Playwright has
    no browser installed there is nothing to run.
    """
    pytest.importorskip("playwright.sync_api")
    default = (tmp_path / "temp").resolve() / "kiss-tmp" / "ui-style-probe.txt"
    named = tmp_path / "named" / "probe.txt"
    for report, args in ((default, []), (named, [str(named)])):
        done = _run("ui_style_probe.py", tmp_path, *args)
        if "Executable doesn't exist" in done.stderr:
            pytest.skip("Playwright's Chromium is not installed")
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == f"wrote {report}"
        assert "===== remote-desktop-dark welcome" in report.read_text()
    assert list((tmp_path / "cwd").iterdir()) == []
