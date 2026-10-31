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

On POSIX the home lives in the session temp root (``kiss-pt-*``),
where ``tempfile`` and every child that inherits ``TMPDIR`` write
too; leaked test temp dirs had used up the inodes of the ``/tmp``
tmpfs by 2026-10-30.  The root goes at exit as well, and a root whose
owner died without running atexit goes with the next session.

Two tests here are *probes*: they only run inside the nested pytest
process another test starts (selected by ``KISS_HOME_CLEANUP_PROBE``)
because one of them ends its session with ``pytest.exit``.
"""

import os
import shutil
import subprocess
import sys
import tempfile
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

# persistence's own atexit handler drains queued events into history.db,
# recreating the home if it is already gone; a pending event at exit
# used to leave ``kiss_test_*/history.db`` behind.  The event is queued
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


def _check_home_location(home: Path, tmpdir: Path) -> None:
    """Check that *home* sits in a session root in *tmpdir* (POSIX) or in *tmpdir*."""
    if os.name == "posix":
        assert home.parent.name.startswith("kiss-pt-"), home
        assert home.parent.parent.resolve() == tmpdir.resolve()
    else:
        assert home.parent.resolve() == tmpdir.resolve()


def _assert_nothing_left(tmpdir: Path) -> None:
    """Check that no session home or session temp root is left in *tmpdir*."""
    assert list(tmpdir.glob("kiss_test_*")) == []
    assert list(tmpdir.glob("kiss-pt-*")) == []


def _reported_home(stdout: str, tmpdir: Path) -> Path:
    """Return the single ``KISS_HOME=`` path a probe printed, checking it lives in *tmpdir*."""
    lines = stdout.splitlines()
    homes = [line.removeprefix("KISS_HOME=") for line in lines if line.startswith("KISS_HOME=")]
    assert len(homes) == 1, stdout
    home = Path(homes[0])
    _check_home_location(home, tmpdir)
    return home


def test_session_home_is_a_kiss_test_dir() -> None:
    """The running session's ``KISS_HOME`` is the conftest's own temp dir."""
    home = Path(os.environ["KISS_HOME"])
    assert home.name.startswith("kiss_test_")
    assert home.is_dir()
    print(f"KISS_HOME={home}")


def test_session_home_snoozes_the_update_toast_for_every_release() -> None:
    """The daemon's own snooze reader sees the session home as "Remind me
    later" for any release, so no "KISS Sorcar X is available" toast can
    cover a chat page while PyPI is ahead of the checkout (the 2026-10-02
    cause of three PDF-toolbar click timeouts in ``test_sidebar_menus``)."""
    from kiss.server import web_server

    home = Path(os.environ["KISS_HOME"])
    assert web_server._update_check_cache_path() == home / ".update-check.json"
    assert web_server._is_update_snoozed("9999.12.31")
    assert web_server._is_update_snoozed(web_server._read_version() or "0")
    assert web_server._is_update_snoozed("0.0.1")


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
    _check_home_location(Path(home), tmp_path)
    assert not Path(home).exists()
    _assert_nothing_left(tmp_path)


def test_queued_event_at_exit_does_not_recreate_home(tmp_path: Path) -> None:
    """An event still queued at exit is drained before the home is removed."""
    (home,) = _run_python(_QUEUE_EVENT_AT_EXIT, tmp_path)
    if sys.platform == "win32":
        # The drain reopens history.db and the connection is deliberately
        # left open (see ``_remove_test_kiss_home``); Windows cannot
        # delete an open file, so only the database may survive.
        leftovers = [p.name for p in Path(home).rglob("*")] if Path(home).exists() else []
        assert all(name.startswith("history.db") for name in leftovers), leftovers
        return
    assert not Path(home).exists()
    _assert_nothing_left(tmp_path)


def test_pytest_run_removes_home(tmp_path: Path) -> None:
    """A real pytest process removes its session home on exit."""
    proc = _run_nested_pytest("test_session_home_is_a_kiss_test_dir", tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    home = _reported_home(proc.stdout, tmp_path)
    assert not home.exists()
    _assert_nothing_left(tmp_path)


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
    _assert_nothing_left(tmp_path)


@posix_only("os.fork")
def test_forked_child_exit_keeps_parent_home(tmp_path: Path) -> None:
    """A forked child exiting through ``sys.exit`` must not delete its parent's home."""
    status, home = _run_python(_FORK_THEN_EXIT, tmp_path)
    assert status == "after child exit: True"
    assert not Path(home).exists()
    _assert_nothing_left(tmp_path)


# The session leaves temp dirs behind: its own and one of a child that
# inherited TMPDIR.
_LEAK_TEMP_DIRS = """
import subprocess, sys, tempfile
import kiss.tests.conftest as conftest
print(conftest._session_tmp_root)
print(tempfile.mkdtemp())
child = "import tempfile; print(tempfile.mkdtemp())"
out = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, check=True)
print(out.stdout.strip())
"""

# A session that dies without running atexit, as pytest-timeout's
# os._exit does.
_DIE_WITHOUT_ATEXIT = """
import os
import kiss.tests.conftest as conftest
print(conftest._session_tmp_root, flush=True)
os._exit(0)
"""

# A session that runs until its stdin is closed.
_RUN_UNTIL_STDIN_CLOSES = """
import sys
import kiss.tests.conftest as conftest
print(conftest._session_tmp_root, flush=True)
sys.stdin.read()
"""

# multiprocessing registers its exit finalizer before the conftest's
# handler, so the finalizer runs after the handler.
_MULTIPROCESSING_TEMP_DIR = """
import multiprocessing.util
import kiss.tests.conftest as conftest
print(multiprocessing.util.get_temp_dir())
print(conftest._session_tmp_root)
"""

_START_SESSION = "import kiss.tests.conftest"


@posix_only("session temp root")
def test_session_temp_dirs_go_with_the_root(tmp_path: Path) -> None:
    """Temp dirs of the session and of its children land in the root and go at exit."""
    root, own, child = _run_python(_LEAK_TEMP_DIRS, tmp_path)
    assert Path(root).name.startswith("kiss-pt-")
    assert Path(root).parent.resolve() == tmp_path.resolve()
    assert Path(own).parent == Path(root)
    assert Path(child).parent == Path(root)
    assert list(tmp_path.iterdir()) == []


@posix_only("flock on the root's lock")
def test_next_session_sweeps_the_root_of_a_dead_session(tmp_path: Path) -> None:
    """A root whose owner died without atexit goes when the next session starts."""
    (root,) = _run_python(_DIE_WITHOUT_ATEXIT, tmp_path)
    assert (Path(root) / ".lock").is_file()
    _run_python(_START_SESSION, tmp_path)
    assert list(tmp_path.iterdir()) == []


@posix_only("flock on the root's lock")
def test_sweep_keeps_the_root_of_a_live_session(tmp_path: Path) -> None:
    """A starting session leaves the root of a running session alone."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _RUN_UNTIL_STDIN_CLOSES],
        env=_private_env(tmp_path),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        root = Path(proc.stdout.readline().strip())
        assert root.name.startswith("kiss-pt-"), root
        _run_python(_START_SESSION, tmp_path)
        assert (root / ".lock").is_file()
    finally:
        proc.communicate(input="", timeout=120)
    assert proc.returncode == 0
    assert list(tmp_path.iterdir()) == []


@posix_only("session temp root")
def test_multiprocessing_temp_dir_is_left_to_its_finalizer() -> None:
    """multiprocessing deletes its own ``pymp-*`` dir at exit without a
    traceback, and the next session deletes the root left around it."""
    # multiprocessing puts its socket dir in /tmp instead when the temp
    # dir is too long for a socket path, as ``tmp_path`` is.
    short = Path(tempfile.mkdtemp(prefix="mp", dir="/tmp"))
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _MULTIPROCESSING_TEMP_DIR],
            env=_private_env(short),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "Traceback" not in proc.stderr, proc.stderr
        mp_dir, root = proc.stdout.splitlines()
        assert Path(mp_dir).parent == Path(root)
        assert [p.name for p in Path(root).iterdir()] == [".lock"]
        _run_python(_START_SESSION, short)
        assert list(short.iterdir()) == []
    finally:
        shutil.rmtree(short, ignore_errors=True)


def test_running_session_writes_temp_files_into_its_root() -> None:
    """This session's ``tempfile`` writes into its root (POSIX)."""
    from kiss.tests import conftest

    if os.name != "posix":
        assert conftest._session_tmp_root is None
        return
    root = conftest._session_tmp_root
    assert root is not None
    assert Path(root).name.startswith("kiss-pt-")
    made = Path(tempfile.mkdtemp())
    try:
        assert made.parent == Path(root)
    finally:
        made.rmdir()
