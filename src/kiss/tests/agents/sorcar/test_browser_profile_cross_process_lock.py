# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The browser profile guard must be machine-wide, not per-process.

``_launch_browser`` once protected the check-then-use sequence
(``_resolve_user_data_dir`` → ``_clean_singleton_locks`` → launch) with a
``threading.RLock`` only; today the ``<profile>.lock`` file lock
(``_profile_lock``) is the sole guard, excluding threads and processes
alike since every holder opens its own file description.  The guarded
resource — the Chromium profile directory — is shared by *every* kiss process
on the machine (the ``kiss-web`` daemon, a CLI run, channel agents), so a
process-local lock gives no mutual exclusion at all: two processes both see a
lock-free profile, and the second one deletes the first one's live
``SingletonLock`` before opening the same profile a second time.  Chromium
then has one profile open twice, and the persisted logins the class promises
to preserve are corrupted.

These tests use two real OS processes, a real file lock, and two real
headless Chromium launches against a temp profile — never the user's real
``~/.kiss/browser_profile``.
"""

import functools
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.web_use_tool import WebUseTool, _read_lock_pid
from kiss.core.file_lock import exclusive_file_lock
from kiss.tests.conftest import posix_only
from kiss.tests.server.pause import Pause

#: Driver run in two concurrent processes.  Both wait on a barrier file so
#: they enter the profile-resolution window together, then each reports the
#: profile directory its Chromium actually holds: the ``--user-data-dir`` of
#: the browser's own command line, read over CDP (``SystemInfo.getInfo``).
#: Chromium's ``SingletonLock`` would name it on POSIX only (Windows Chromium
#: holds a ``lockfile`` instead), and a second Chromium opening the same
#: profile replaces the first one's lock, so a doubly opened profile would
#: show as held by one process only.
_BROWSER_DRIVER = """
import re, sys, time
from pathlib import Path

from kiss.agents.sorcar.web_use_tool import WebUseTool, _cdp_call

profile, barrier, tag = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
Path(str(barrier) + "." + tag).write_text("ready")
while not barrier.exists():
    time.sleep(0.02)

tool = WebUseTool(user_data_dir=profile, headless=True)
try:
    tree = tool.go_to_url("data:text/html,<h1>profile lock</h1>")
    browser = tool._context.browser
    cdp = _cdp_call(browser, browser._impl_obj.new_browser_cdp_session())
    command_line = _cdp_call(browser, cdp.send("SystemInfo.getInfo"))["commandLine"]
    # Windows quotes an argument with spaces in it; the test's paths have none.
    held = re.search(r'--user-data-dir="?([^" ]+)', command_line)
    print("PROFILE=" + (held.group(1) if held else ""))
    print("OK=" + str("profile lock" in tree))
    time.sleep(3)
finally:
    tool.close()
"""

#: Driver that takes the shared inter-process file lock and records the
#: interval it held it, so overlapping intervals prove the lock is broken.
_LOCK_DRIVER = """
import sys, time
from pathlib import Path

from kiss.core.file_lock import exclusive_file_lock

lock_path, out_path, barrier = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
Path(str(barrier) + "." + out_path.name).write_text("ready")
while not barrier.exists():
    time.sleep(0.02)

with exclusive_file_lock(lock_path):
    start = time.time()
    time.sleep(1.0)
    out_path.write_text(f"{start} {time.time()}")
"""


def _spawn(script: Path, *args: str, env_home: Path) -> subprocess.Popen:
    """Start a real child interpreter running *script* with an isolated home."""
    import os

    env = os.environ.copy()
    env["KISS_HOME"] = str(env_home)
    return subprocess.Popen(
        [sys.executable, str(script), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


def _release_barrier(barrier: Path, tags: list[str]) -> None:
    """Wait until every child signalled readiness, then open the barrier."""
    deadline = time.time() + 60
    while time.time() < deadline:
        if all(Path(f"{barrier}.{tag}").exists() for tag in tags):
            barrier.write_text("go")
            return
        time.sleep(0.02)
    raise AssertionError("children never signalled readiness")


def test_file_lock_excludes_other_processes(tmp_path):
    """The shared helper really serialises two OS processes."""
    script = tmp_path / "lock_driver.py"
    script.write_text(_LOCK_DRIVER, encoding="utf-8")
    lock_path = tmp_path / "shared.lock"
    barrier = tmp_path / "barrier"
    outs = [tmp_path / "a.txt", tmp_path / "b.txt"]

    procs = [
        _spawn(script, str(lock_path), str(out), str(barrier), env_home=tmp_path) for out in outs
    ]
    _release_barrier(barrier, [out.name for out in outs])
    for proc in procs:
        stdout, stderr = proc.communicate(timeout=120)
        assert proc.returncode == 0, stderr or stdout

    spans = []
    for out in outs:
        start, end = (float(v) for v in out.read_text().split())
        spans.append((start, end))
    spans.sort()
    assert spans[0][1] <= spans[1][0], f"lock holders overlapped: {spans}"


def test_two_processes_never_share_one_browser_profile(tmp_path):
    """Concurrent kiss processes must not open the same profile directory."""
    script = tmp_path / "browser_driver.py"
    script.write_text(_BROWSER_DRIVER, encoding="utf-8")
    profile = tmp_path / "browser_profile"
    barrier = tmp_path / "barrier"

    procs = [
        _spawn(script, str(profile), str(barrier), tag, env_home=tmp_path) for tag in ("one", "two")
    ]
    _release_barrier(barrier, ["one", "two"])

    profiles = []
    for proc in procs:
        stdout, stderr = proc.communicate(timeout=300)
        assert proc.returncode == 0, stderr or stdout
        assert "OK=True" in stdout, stdout
        line = next(li for li in stdout.splitlines() if li.startswith("PROFILE="))
        profiles.append(line.split("=", 1)[1])

    assert len(set(profiles)) == 2, f"both processes opened the same Chromium profile: {profiles}"
    assert str(profile) in profiles, profiles


def _cleans(profile: str, entry: dict[str, Any]) -> bool:
    """A :class:`Pause` hook: the singleton files of *profile* are about to be removed."""
    return bool(entry["profile_dir"] == profile)


def _open_then_close(tool: WebUseTool, opened: threading.Event, close: threading.Event) -> None:
    """Thread body: launch *tool*'s browser, set *opened*, close the tool once *close* is set."""
    try:
        tool._ensure_browser()
        opened.set()
        close.wait(120)
    finally:
        tool.close()


@posix_only("a symlink to the profile; Chromium's SingletonLock names the profile's holder")
def test_a_profile_and_a_link_to_it_are_one_profile_to_the_guard(tmp_path):
    """A launch through a symlink to a profile waits for the launch that has the profile (S18/F1).

    The lock file was named for the path as given, so the directory and
    a link to it were guarded by two locks: the launch through the link
    opened the profile the first launch had chosen and was about to
    clean, which then removed that browser's live ``SingletonLock`` and
    opened the profile a second time.  The first launch is held where
    it removes the singleton files, the lock its own.
    """
    actual, alias = tmp_path / "actual", tmp_path / "alias"
    actual.mkdir()
    alias.symlink_to(actual, target_is_directory=True)
    tools = [WebUseTool(user_data_dir=str(path), headless=True) for path in (actual, alias)]
    opened, close = [threading.Event(), threading.Event()], threading.Event()
    launches = [
        threading.Thread(target=_open_then_close, args=(tool, event, close))
        for tool, event in zip(tools, opened)
    ]
    try:
        cleanup = WebUseTool._clean_singleton_locks.__code__
        with Pause(cleanup, functools.partial(_cleans, str(actual))) as held:
            launches[0].start()
            assert held.paused.wait(60), "the first launch never reached the profile's cleanup"
            launches[1].start()
            deadline = time.monotonic() + 60
            while tools[1]._playwright is None and time.monotonic() < deadline:  # its driver starts
                time.sleep(0.01)
            assert not opened[1].wait(2), "the launch through the link did not wait for the lock"
            held.release()
            assert opened[0].wait(120) and opened[1].wait(120)
        assert tools[0]._browser_pid and tools[1]._browser_pid
        assert _read_lock_pid(str(actual)) == tools[0]._browser_pid
        assert _read_lock_pid(f"{alias}_1") == tools[1]._browser_pid
    finally:
        close.set()
        for launch in launches:
            if launch.ident is not None:
                launch.join(60)


def _take_profile_lock(tool: WebUseTool, taken: threading.Event) -> None:
    """Thread body: take *tool*'s profile lock and set *taken* while it is held."""
    with tool._profile_lock():
        taken.set()


@posix_only("a symlink to the profile")
def test_a_linked_profile_is_also_guarded_by_the_lock_named_for_the_link(tmp_path):
    """A daemon from before S18/F1 and this one exclude each other on a profile behind a link.

    That daemon locks ``<profile path as given>.lock``.  With the lock
    named for the resolved directory alone, a launch through a link no
    longer waited for the older daemon's launch through the same link
    (an upgrade leaves both running for a while): one profile, two
    browsers.  The lock named for the path as given is held as well,
    taken first.  Where the two names are one file (a link above the
    profile) it is locked once: a second open would wait for the first.
    """
    actual, alias = tmp_path / "actual", tmp_path / "alias"
    actual.mkdir()
    alias.symlink_to(actual, target_is_directory=True)
    linked, below = (WebUseTool(user_data_dir=str(path)) for path in (alias, alias / "below"))
    taken = threading.Event()
    waiter = threading.Thread(target=_take_profile_lock, args=(linked, taken), daemon=True)
    try:
        with exclusive_file_lock(Path(f"{alias}.lock")):  # the older daemon, launching
            waiter.start()
            assert not taken.wait(1), "the lock named for the link was not waited for"
        assert taken.wait(30), "the lock the older daemon released was never taken"
        waiter.join(30)
        with linked._profile_lock():  # under both names: the older daemon waits in turn
            for name in (alias, actual):
                with exclusive_file_lock(Path(f"{name}.lock"), blocking=False) as free:
                    assert not free, f"{name}.lock is not held"
        with below._profile_lock():  # times out when it locks its one file under both names
            pass
    finally:
        linked.close()
        below.close()
