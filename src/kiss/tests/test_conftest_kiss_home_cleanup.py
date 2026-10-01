# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The session ``KISS_HOME`` is removed when the pytest process exits.

``src/kiss/tests/conftest.py`` creates a ``kiss_test_*`` directory at
import and, until ``_remove_test_kiss_home`` was registered with
``atexit``, nothing ever deleted it: 1,700 such trees (6 GB) had
accumulated in the macOS temp folder.  These tests drive real child
interpreters with a private ``TMPDIR`` and inspect what they leave
behind.

Two tests here are *probes*: they only run inside the nested pytest
process another test starts (selected by ``KISS_HOME_CLEANUP_PROBE``)
because one of them ends its session with ``pytest.exit``.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from kiss.tests.conftest import posix_only

_PROBE_ENV = "KISS_HOME_CLEANUP_PROBE"

_REPORT_HOME = """
import os, sys
import kiss.tests.conftest as conftest
home = conftest._test_kiss_home
assert os.path.isdir(home), home
assert os.environ["KISS_HOME"] == home
print(home)
"""

_FORK_THEN_EXIT = """
import os, sys
import kiss.tests.conftest as conftest
home = conftest._test_kiss_home
pid = os.fork()
if pid == 0:
    sys.exit(0)  # runs the inherited atexit table in the child
_, status = os.waitpid(pid, 0)
assert status == 0, status
print("after child exit:", os.path.isdir(home))
print(home)
"""

# persistence's own atexit handler drains queued events into sorcar.db,
# recreating the home if it is already gone; a pending event at exit
# used to leave ``kiss_test_*/sorcar.db`` behind.  The event is queued
# from an atexit handler registered *after* the conftest's, which atexit
# therefore runs *before* it: the event is still pending (the writer
# batches for 20 ms) when the home is removed, whatever the load.
_QUEUE_EVENT_AT_EXIT = """
import atexit
import kiss.tests.conftest as conftest
from kiss.agents.sorcar import persistence
task_id, _ = persistence._add_task("cleanup probe")
def queue_late():
    persistence._queue_chat_event({"type": "text", "text": "queued at exit"}, task_id)
atexit.register(queue_late)
print(conftest._test_kiss_home)
"""

# A test's child that outlives an interrupted run: on SIGTERM it records
# whether the session home still existed, then exits.
_SLEEPER = """
import os, signal, time
from pathlib import Path
def on_term(*_):
    Path(os.environ["PROBE_OUT"]).write_text(str(os.path.isdir(os.environ["KISS_HOME"])))
    os._exit(0)
signal.signal(signal.SIGTERM, on_term)
Path(os.environ["PROBE_OUT"] + ".ready").touch()
while True:
    time.sleep(0.1)
"""


def _private_env(tmpdir: Path, **extra: str) -> dict[str, str]:
    """Return the environment for a child whose temp folder is *tmpdir*."""
    return dict(os.environ, TMPDIR=str(tmpdir), TMP=str(tmpdir), TEMP=str(tmpdir), **extra)


def _run_python(code: str, tmpdir: Path) -> list[str]:
    """Run *code* in a fresh interpreter whose temp folder is *tmpdir*."""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=_private_env(tmpdir),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.splitlines()


def _run_nested_pytest(
    test_name: str, tmpdir: Path, **extra: str
) -> subprocess.CompletedProcess[str]:
    """Run one test of this file in a pytest child whose temp folder is *tmpdir*."""
    args = ["-m", "pytest", "-q", "-s", "-p", "no:cacheprovider", f"{__file__}::{test_name}"]
    return subprocess.run(
        [sys.executable, *args],
        env=_private_env(tmpdir, **extra),
        capture_output=True,
        text=True,
        check=False,
    )


def _reported_home(stdout: str, tmpdir: Path) -> Path:
    """Return the single ``KISS_HOME=`` path a probe printed, checking it lives in *tmpdir*."""
    lines = stdout.splitlines()
    homes = [line.removeprefix("KISS_HOME=") for line in lines if line.startswith("KISS_HOME=")]
    assert len(homes) == 1, stdout
    home = Path(homes[0])
    assert home.parent.resolve() == tmpdir.resolve()
    return home


def test_session_home_is_a_kiss_test_dir() -> None:
    """The running session's ``KISS_HOME`` is the conftest's own temp dir."""
    home = Path(os.environ["KISS_HOME"])
    assert home.name.startswith("kiss_test_")
    assert home.is_dir()
    print(f"KISS_HOME={home}")


@pytest.mark.skipif(
    os.environ.get(_PROBE_ENV) != "interrupt", reason="probe for the nested run below"
)
def test_probe_interrupted_run_with_live_child() -> None:
    """Start a child that outlives the test, then cut the session short."""
    print(f"KISS_HOME={os.environ['KISS_HOME']}")
    subprocess.Popen([sys.executable, "-c", _SLEEPER])
    ready = Path(os.environ["PROBE_OUT"] + ".ready")
    deadline = time.monotonic() + 30
    while not ready.exists():
        assert time.monotonic() < deadline, "sleeper never installed its SIGTERM handler"
        time.sleep(0.05)
    pytest.exit("probe: interrupted run")


def test_import_then_exit_removes_home(tmp_path: Path) -> None:
    """Importing the conftest and exiting leaves no ``kiss_test_*`` behind."""
    (home,) = _run_python(_REPORT_HOME, tmp_path)
    assert Path(home).parent.resolve() == tmp_path.resolve()
    assert not Path(home).exists()
    assert list(tmp_path.glob("kiss_test_*")) == []


def test_queued_event_at_exit_does_not_recreate_home(tmp_path: Path) -> None:
    """An event still queued at exit is drained before the home is removed."""
    (home,) = _run_python(_QUEUE_EVENT_AT_EXIT, tmp_path)
    if sys.platform == "win32":
        # The drain reopens sorcar.db and the connection is deliberately
        # left open (see ``_remove_test_kiss_home``); Windows cannot
        # delete an open file, so only the database may survive.
        leftovers = [p.name for p in Path(home).rglob("*")] if Path(home).exists() else []
        assert all(name.startswith("sorcar.db") for name in leftovers), leftovers
        return
    assert not Path(home).exists()
    assert list(tmp_path.glob("kiss_test_*")) == []


def test_pytest_run_removes_home(tmp_path: Path) -> None:
    """A real pytest process removes its session home on exit."""
    proc = _run_nested_pytest("test_session_home_is_a_kiss_test_dir", tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    home = _reported_home(proc.stdout, tmp_path)
    assert not home.exists()
    assert list(tmp_path.glob("kiss_test_*")) == []


@posix_only("SIGTERM handler in the leftover child")
def test_interrupted_run_reaps_children_before_removing_home(tmp_path: Path) -> None:
    """``pytest.exit`` mid-test: the leftover child is stopped first, then the home goes."""
    probe_out = tmp_path / "probe.txt"
    proc = _run_nested_pytest(
        "test_probe_interrupted_run_with_live_child",
        tmp_path,
        **{_PROBE_ENV: "interrupt", "PROBE_OUT": str(probe_out)},
    )
    assert "probe: interrupted run" in proc.stdout + proc.stderr, proc.stdout + proc.stderr
    home = _reported_home(proc.stdout, tmp_path)
    assert probe_out.read_text() == "True", "child was terminated only after its home was gone"
    assert not home.exists()
    assert list(tmp_path.glob("kiss_test_*")) == []


@posix_only("os.fork")
def test_forked_child_exit_keeps_parent_home(tmp_path: Path) -> None:
    """A forked child exiting through ``sys.exit`` must not delete its parent's home."""
    status, home = _run_python(_FORK_THEN_EXIT, tmp_path)
    assert status == "after child exit: True"
    assert not Path(home).exists()
