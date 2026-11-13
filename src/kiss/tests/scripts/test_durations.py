# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of ``kiss.scripts.durations``: the test-duration store behind ``--splits``."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from kiss.scripts import durations, impacted_tests


def test_record_durations_merges_under_the_lock_and_load_tolerates_junk(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "durations.json"
    assert durations.load_durations(path) == {}
    durations.record_durations(path, {})
    assert not path.exists()
    durations.record_durations(path, {"a.py": 1.23456, "b.py": 2.0})
    durations.record_durations(path, {"b.py": 3.0, "c.py": 0.5})
    assert durations.load_durations(path) == {"a.py": 1.235, "b.py": 3.0, "c.py": 0.5}
    path.write_text("[1, 2]")
    assert durations.load_durations(path) == {}
    path.write_text("{not json")
    assert durations.load_durations(path) == {}
    # Audit SC1: a value that is no duration was raised (TypeError, ValueError)
    # instead of dropped, out of every later record and every --splits.
    path.write_text(
        '{"a.py": 2, "b.py": null, "c.py": "slow", "d.py": {"seconds": 1}, "e.py": true,'
        ' "f.py": NaN, "g.py": -1, "h.py": 1e999, "i.py": 1' + "0" * 400 + "}"
    )
    assert durations.load_durations(path) == {"a.py": 2.0}
    durations.record_durations(path, {"z.py": 0.5})
    assert durations.load_durations(path) == {"a.py": 2.0, "z.py": 0.5}


def test_durations_path_prefers_the_env_then_the_main_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KISS_TEST_DURATIONS", str(tmp_path / "d.json"))
    assert durations.durations_path() == tmp_path / "d.json"
    monkeypatch.delenv("KISS_TEST_DURATIONS")
    # Outside git the file is under the root given; inside the real
    # repository (a worktree included) it is the main checkout's.
    assert durations.durations_path(tmp_path) == tmp_path / "tmp" / "test-durations.json"
    common = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"], cwd=durations.PROJECT_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    main = (durations.PROJECT_ROOT / common).resolve().parent
    assert durations.durations_path() == main / "tmp" / "test-durations.json"


def test_the_suite_records_its_file_durations_for_the_next_split(tmp_path: Path) -> None:
    """A pytest session of this repository writes each file's seconds to the durations file."""
    store = tmp_path / "durations.json"
    # What another writer left in the machine's store is no failure of this session.
    store.write_text('{"src/kiss/tests/test_stale.py": null}')
    me = Path(__file__).relative_to(durations.PROJECT_ROOT).as_posix()
    env = {**os.environ, "KISS_TEST_DURATIONS": str(store)}
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--timeout=120",
         f"{me}::test_record_durations_merges_under_the_lock_and_load_tolerates_junk"],
        cwd=durations.PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=300,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    # The run begins with the interpreter it ran on (the root conftest), though -q drops
    # pytest's own header; only the conftest's file-descriptor notice may precede it.
    interpreter = f"interpreter: {sys.executable} (Python {platform.python_version()})"
    assert interpreter in run.stdout.splitlines()[:2], run.stdout
    recorded = durations.load_durations(store)
    assert list(recorded) == [me] and recorded[me] > 0, recorded
    # The next --splits of that file is balanced by what was recorded.
    commands = impacted_tests.split_commands([Path(__file__)], 2, durations=recorded)
    assert commands == [f"uv run pytest -q -p no:cacheprovider --timeout=600 {me}"]
