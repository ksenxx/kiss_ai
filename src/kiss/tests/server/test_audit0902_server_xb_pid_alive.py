# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-02 (cross-boundary): ``web_server`` uses the shared ``pid_alive``.

``web_server._is_pid_alive`` was one of three divergent copies of the
"is this pid alive" probe; it now delegates to
:func:`kiss.agents.sorcar._concurrency.pid_alive`.  The behaviour the
server relies on is exercised end to end through the cloudflared
pidfile adoption path (:func:`_try_adopt_existing_cloudflared`),
which must decline a pidfile that names a dead or non-positive pid
without probing any metrics port.  A real ``KISS_HOME`` temp dir holds
the pidfile; ``KISS_HOME`` is restored afterwards.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar._concurrency import pid_alive
from kiss.server import web_server as ws


@pytest.fixture
def kiss_home(tmp_path: Path) -> Iterator[Path]:
    """Point ``KISS_HOME`` at a throwaway directory for the pidfile."""
    saved = os.environ.get("KISS_HOME")
    os.environ["KISS_HOME"] = str(tmp_path)
    try:
        yield tmp_path
    finally:
        if saved is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = saved


def _write_pidfile(home: Path, pid: int) -> None:
    (home / "cloudflared.pid").write_text(
        json.dumps({"pid": pid, "metrics_port": 1}) + "\n", encoding="utf-8",
    )


def test_is_pid_alive_matches_shared_helper_on_real_processes() -> None:
    """Live child, reaped child, and the non-positive guards agree."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert ws._is_pid_alive(proc.pid) is pid_alive(proc.pid) is True
    finally:
        proc.kill()
        proc.wait()
    assert ws._is_pid_alive(proc.pid) is pid_alive(proc.pid) is False
    assert ws._is_pid_alive(0) is False
    assert ws._is_pid_alive(-1) is False


def test_adoption_declines_pidfile_of_reaped_process(kiss_home: Path) -> None:
    """A stale pidfile naming a dead pid is not adopted."""
    proc = subprocess.Popen(["true"])
    proc.wait()
    _write_pidfile(kiss_home, proc.pid)
    assert ws._cloudflared_pidfile() == kiss_home / "cloudflared.pid"
    assert ws._try_adopt_existing_cloudflared() is None


def test_adoption_declines_pidfile_with_non_positive_pid(kiss_home: Path) -> None:
    """``pid: 0`` must never be treated as a live cloudflared."""
    _write_pidfile(kiss_home, 0)
    assert ws._try_adopt_existing_cloudflared() is None
