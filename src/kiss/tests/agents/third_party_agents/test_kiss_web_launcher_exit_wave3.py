# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the channel launcher's in-process daemon cleans up when its process exits.

``_kiss_web_launcher._ensure_api_server`` keeps its endpoint file and
private tab registry in a ``kiss-tp-api-*`` temp directory that only
``_stop_api_server`` removes.  A process that started the daemon and
exited without calling it left that directory behind, one per process,
and they piled up under /tmp.  The module now registers
``_stop_api_server`` with :mod:`atexit`.  Each test runs a real child
interpreter with its own ``TMPDIR`` and ``KISS_HOME`` and checks what
is left once it has exited.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_START_DAEMON = """
import os
from kiss.agents.third_party_agents import _kiss_web_launcher as launcher
endpoint = launcher._ensure_api_server()
print(endpoint, os.path.isfile(endpoint), flush=True)
"""


def _run_child(tmp_path: Path, script: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run *script* in a child interpreter whose temp dir and homes are under *tmp_path*.

    Returns:
        The finished process and the child's ``TMPDIR``.
    """
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    env = {
        **os.environ,
        "TMPDIR": str(tmpdir),
        "KISS_HOME": str(tmp_path / "home"),
        "HOME": str(tmp_path),
    }
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc, tmpdir


def _assert_clean_exit(proc: subprocess.CompletedProcess[str], tmpdir: Path) -> None:
    """The child started the daemon, exited 0 quietly, and left no private directory."""
    assert proc.returncode == 0, proc.stderr
    endpoint, existed = proc.stdout.split()
    assert existed == "True"
    assert Path(endpoint).parent.parent.resolve() == tmpdir.resolve()
    assert not Path(endpoint).parent.exists()
    assert list(tmpdir.glob("kiss-tp-api-*")) == []
    assert "Traceback" not in proc.stderr, proc.stderr
    assert "Exception ignored" not in proc.stderr, proc.stderr


def test_exit_without_stopping_the_daemon_removes_its_private_directory(
    tmp_path: Path,
) -> None:
    """The atexit hook stops the daemon a process never stopped itself."""
    proc, tmpdir = _run_child(tmp_path, _START_DAEMON)
    _assert_clean_exit(proc, tmpdir)


def test_exit_after_an_explicit_stop_runs_the_hook_as_a_no_op(tmp_path: Path) -> None:
    """An explicit stop, then the atexit hook finding no daemon, exits cleanly."""
    script = _START_DAEMON + "launcher._stop_api_server()\nassert launcher._API_SERVER is None\n"
    proc, tmpdir = _run_child(tmp_path, script)
    _assert_clean_exit(proc, tmpdir)
