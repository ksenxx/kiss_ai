# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the 2026-11-01 audit of :mod:`kiss.server.signin` (V9 F2-F4).

1. A login hook that failed is reported to the window under the lock
   that :meth:`SigninLogin.start` and :meth:`SigninLogin.cancel` take:
   a retry that started between the relay's "still current" check and
   its ``signinError`` used to see the old failure delivered after its
   own ``signinUrl``, which the window takes for the retry's failure.
2. A login hook whose output is not UTF-8 ends the login with a
   ``signinError`` like one that prints no JSON; the strict decoding
   used to kill the relay thread, leaving the hook current and its
   pipes open, and the window waiting for an answer that never came.
3. A ``status`` hook that outlives :data:`STATUS_TIMEOUT` is ended
   with its process group: a child it started (a poller, a browser
   opener) used to live on after the hook was killed alone.

And of the 2026-11-05 audit (A13):

4. A cancel or retry that lands once the relay thread has reaped the
   hook (it exited on its own) signals nothing: the group's pid may
   already be a stranger's, and a helper the hook left behind (the
   user's browser) is left alone as after any natural exit.  The
   sign-in that hook had finished is still announced to every window,
   after the window it was taken from by another one has heard so
   (2026-11-10, A/V6-F1 and F2).
5. A login whose relay thread cannot start (thread exhaustion) is
   ended and reported as a ``signinError``; it used to live on with
   its pipes open and nobody reading them.
6. A cancel whose end thread cannot start still sees the hook out,
   bounded, and leaves nothing unstarted for ``shutdown`` to join; a
   SIGTERM-ignoring hook used to be forgotten alive and ``shutdown``
   raised on the never-started thread.

The hooks are real executables under a temporary ``KISS_HOME``.
"""

from __future__ import annotations

import contextlib
import gc
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.core.processes import exited_unreaped, terminate_process_group
from kiss.server import signin
from kiss.server.signin import SIGNIN_HOOK, SigninLogin, send_signin
from kiss.tests.conftest import nproc_limit_lowered_to_one, posix_only, thread_start_can_be_starved
from kiss.tests.server.test_signin_window import forget_status

pytestmark = posix_only("the stub hooks are scripts with a shebang")

Event = dict[str, Any]

# The first ``login`` prints its URL, then fails; the second (the retry)
# prints its URL and waits for a paste like a real hook.
_FAILING_THEN_WAITING_HOOK = """#!/bin/sh
case "$1" in
  login)
    if [ ! -e "$KISS_HOME/first-login" ]; then
      : > "$KISS_HOME/first-login"
      echo '{"url": "http://127.0.0.1:1/old"}'
      echo '{"error": "OLD FAILURE"}'
      exit 1
    fi
    echo '{"url": "http://127.0.0.1:1/new"}'
    read -r line
    exit 0 ;;
esac
echo '{"logged_in": false}'
"""

_BINARY_OUTPUT_HOOK = f"""#!{sys.executable}
import os, sys
if sys.argv[1:2] == ["login"]:
    os.write(1, b"\\xff\\n")
    sys.exit(1)
print('{{"logged_in": false}}')
"""

# ``status`` starts a child that sleeps, records its pid, and sleeps
# itself past the daemon's deadline.
_SLOW_STATUS_HOOK = f"""#!{sys.executable}
import os, subprocess, sys, time
if sys.argv[1:2] == ["status"]:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    with open(os.path.join(os.environ["KISS_HOME"], "child.pid"), "w") as fh:
        fh.write(str(child.pid))
    time.sleep(60)
print('{{"logged_in": false}}')
"""


# ``login`` prints its URL, starts a helper in its own group (the hook
# itself, re-run as ``helper``: it notes a SIGTERM in a file and leaves),
# records the helper's pid, and exits 0 with its success line once the
# test creates ``finish``.
_EXITING_WITH_HELPER_HOOK = f"""#!{sys.executable}
import os, signal, subprocess, sys, time
home = os.environ["KISS_HOME"]
if sys.argv[1:2] == ["helper"]:
    def noted(signum, frame):
        open(os.path.join(home, "helper-signalled"), "w").close()
        sys.exit(0)
    signal.signal(signal.SIGTERM, noted)
    while True:
        time.sleep(0.05)
if sys.argv[1:2] == ["login"]:
    print('{{"url": "http://127.0.0.1:1/login"}}', flush=True)
    helper = subprocess.Popen([sys.executable, sys.argv[0], "helper"])
    with open(os.path.join(home, "helper.pid"), "w") as fh:
        fh.write(str(helper.pid))
    while not os.path.exists(os.path.join(home, "finish")):
        time.sleep(0.02)
    print('{{"logged_in": true, "email": "me@example.com"}}', flush=True)
    sys.exit(0)
print('{{"logged_in": false}}')
"""

# ``login`` prints its URL and waits for a paste like a real hook.
_WAITING_HOOK = """#!/bin/sh
case "$1" in
  login) echo '{"url": "http://127.0.0.1:1/login"}'; read -r line; exit 0 ;;
esac
echo '{"logged_in": false}'
"""

# ``login`` prints its URL, then ignores SIGTERM for a minute.
_STUBBORN_HOOK = """#!/bin/sh
case "$1" in
  login) echo '{"url": "http://127.0.0.1:1/login"}'; trap "" TERM; exec sleep 60 ;;
esac
echo '{"logged_in": false}'
"""


@contextlib.contextmanager
def _thread_stacks_too_large_to_map() -> Iterator[None]:
    """Make ``Thread.start`` fail for real while ``subprocess.Popen`` keeps working.

    ``RLIMIT_NPROC`` cannot tell the two apart (a fork counts against it
    like a thread); a 256 TiB stack is beyond any address space, so
    ``pthread_create`` cannot map it and ``Thread.start`` raises
    ``RuntimeError`` exactly as under thread exhaustion.
    """
    previous = threading.stack_size()
    threading.stack_size(2**48)
    try:
        yield
    finally:
        threading.stack_size(previous)


def _huge_stack_starves_threads() -> bool:
    """Whether :func:`_thread_stacks_too_large_to_map` makes ``Thread.start`` fail here."""
    try:
        with _thread_stacks_too_large_to_map():
            probe = threading.Thread(target=lambda: None)
            try:
                probe.start()
            except RuntimeError:
                return True
            probe.join()
    except (ValueError, OverflowError):
        pass
    return False


def _install_hook(home: Path, body: str) -> Path:
    """Write the executable sign-in hook *body* under *home*."""
    hook = home / SIGNIN_HOOK
    hook.write_text(body)
    hook.chmod(0o755)
    return hook


def _alive(pid: int) -> bool:
    """Whether a process with *pid* exists (a zombie counts as alive)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_until(condition: Callable[[], bool], seconds: float = 10.0) -> bool:
    """Poll *condition* for up to *seconds*; True once it holds."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def _relay_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "signin-login"]


def _hook_processes(hook: Path) -> list[subprocess.Popen[str]]:
    """The ``Popen`` objects of *hook*'s logins (the suite's reaper keeps them referenced)."""
    return [
        o
        for o in gc.get_objects()
        if isinstance(o, subprocess.Popen) and isinstance(o.args, list) and o.args[0] == str(hook)
    ]


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A temporary ``KISS_HOME`` with no cached ``status`` answer before or after."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    forget_status()
    yield tmp_path
    forget_status()


def test_a_replaced_logins_failure_is_not_reported_after_the_retrys_url(home: Path) -> None:
    """The old hook's ``signinError`` reaches the window before the retry's ``signinUrl``."""
    _install_hook(home, _FAILING_THEN_WAITING_HOOK)
    events: list[Event] = []
    delivering, release = threading.Event(), threading.Event()

    def reply(event: Event, broadcast: bool) -> None:
        # The window's queue, held up once: the old hook's failure is at
        # the door while the user clicks again.
        if event.get("error") == "OLD FAILURE" and not delivering.is_set():
            delivering.set()
            release.wait(10)
        events.append(event)

    def new_url_delivered() -> bool:
        return any(str(e.get("url", "")).endswith("/new") for e in events)

    login = SigninLogin()
    try:
        login.start("test", reply)
        assert delivering.wait(10), events
        retry = threading.Thread(target=login.start, args=("test", reply))
        retry.start()
        # Before the fix the retry went through here and its URL was
        # delivered while the old failure was still held up.
        _wait_until(new_url_delivered, 1.0)
        release.set()
        retry.join(10)
        assert not retry.is_alive()
        assert _wait_until(new_url_delivered), events
    finally:
        release.set()
        login.shutdown()
    assert [(e["type"], e.get("url", e.get("error"))) for e in events] == [
        ("signinUrl", "http://127.0.0.1:1/old"),
        ("signinError", "OLD FAILURE"),
        ("signinUrl", "http://127.0.0.1:1/new"),
    ]


def test_hook_output_that_is_not_utf8_ends_the_login_with_an_error(home: Path) -> None:
    """Undecodable output is "no sign-in page": a ``signinError``, the hook forgotten."""
    hook = _install_hook(home, _BINARY_OUTPUT_HOOK)
    events: list[Event] = []
    login = SigninLogin()
    try:
        login.start("test", lambda event, broadcast: events.append(event))
        assert _wait_until(lambda: not _relay_threads()), "the relay thread did not finish"
        assert [e["type"] for e in events] == ["signinError"], events
        assert events[0]["error"] == "the sign-in hook printed no sign-in page"
        assert login._proc is None
        (proc,) = _hook_processes(hook)
        assert proc.returncode == 1
        assert proc.stdin is not None and proc.stdin.closed
        assert proc.stdout is not None and proc.stdout.closed
    finally:
        login.shutdown()


def test_an_end_after_the_hooks_exit_signals_its_group_nothing(home: Path) -> None:
    """A cancel landing once the relay reaped the hook leaves its helper alone (item 4).

    Another window's cancel: the login's window hears of it, and the
    sign-in the hook had finished by then is announced all the same,
    with the ``status`` answer cached before it dropped.
    """
    _install_hook(home, _EXITING_WITH_HELPER_HOOK)
    events: list[Event] = []
    url_seen = threading.Event()

    def reply(event: Event, broadcast: bool) -> None:
        events.append({**event, "everyone": broadcast})
        if event["type"] == "signinUrl":
            url_seen.set()

    login = SigninLogin()
    pid_file = home / "helper.pid"
    try:
        send_signin(events.append)  # a page's ``ready``: "not logged in" is cached (no event)
        assert signin._status_cache is not None
        login.start("test", reply, "window-a")
        assert url_seen.wait(10), events
        assert _wait_until(lambda: pid_file.exists() and pid_file.read_text() != "")
        helper_pid = int(pid_file.read_text())
        proc = login._proc
        assert proc is not None
        with login._lock:  # once it has seen the hook exit, the relay parks on this lock to reap it
            (home / "finish").touch()
            assert _wait_until(lambda: exited_unreaped(proc)), "the hook did not exit"
            assert login._proc is proc
            login._end_locked("window-b")  # what cancel() and a start() do under the lock
            assert login._ending == [], "an end thread was started for a reaped hook"
        assert _wait_until(lambda: not _relay_threads() and not login._ending)
        assert proc.returncode == 0
        assert login._proc is None
        assert not (home / "helper-signalled").exists(), "the reaped hook's group was signalled"
        assert _alive(helper_pid), "the hook's helper did not survive the end"
    finally:
        with contextlib.suppress(OSError, ValueError):
            os.kill(int(pid_file.read_text()), signal.SIGKILL)
        login.shutdown()
    assert [(e["type"], e["everyone"]) for e in events] == [
        ("signinUrl", False),
        ("signinError", False),
        ("signinDone", True),
    ], events
    assert events[1]["error"] == signin.ENDED_ELSEWHERE and events[1]["provider"] == "test"
    assert events[2]["email"] == "me@example.com"
    assert signin._status_cache is None, "the answer cached before the sign-in outlived it"
    assert proc.stdin is not None and proc.stdin.closed
    assert proc.stdout is not None and proc.stdout.closed


def test_a_cancelled_hook_is_reaped_by_the_thread_that_signals_its_group_alone(
    home: Path,
) -> None:
    """The end thread's group signal never names a pid the relay freed by reaping the hook.

    The hook dies on the SIGTERM of ``cancel``; the end thread is held
    (a profile hook on its first statement) until the relay thread is
    gone, which is when a relay that reaps the exited hook has done so,
    and then signals the group: the pid must still be the end thread's
    to reap, else a stranger could own it by then.
    """
    _install_hook(home, _WAITING_HOOK)
    events: list[Event] = []
    url_seen = threading.Event()
    reaped_before_signal: list[int | None] = []

    def reply(event: Event, broadcast: bool) -> None:
        events.append(event)
        if event["type"] == "signinUrl":
            url_seen.set()

    def hold_end_until_the_relay_is_gone(frame: Any, event: str, arg: Any) -> None:
        if (
            event == "call"
            and threading.current_thread().name == "signin-end"
            and frame.f_code is terminate_process_group.__code__
        ):
            for relay in _relay_threads():
                relay.join(10)
            reaped_before_signal.append(frame.f_locals["proc"].returncode)

    login = SigninLogin()
    threading.setprofile(hold_end_until_the_relay_is_gone)
    try:
        login.start("test", reply)
        assert url_seen.wait(10), events
        proc = login._proc
        assert proc is not None
        login.cancel()
        assert _wait_until(lambda: not login._ending and not _relay_threads())
    finally:
        threading.setprofile(None)
        login.shutdown()
    assert reaped_before_signal == [None], (
        "the relay reaped the hook before its group was signalled"
    )
    assert proc.returncode == -signal.SIGTERM
    assert [e["type"] for e in events] == ["signinUrl"], events
    assert proc.stdin is not None and proc.stdin.closed
    assert proc.stdout is not None and proc.stdout.closed


@pytest.mark.skipif(
    not _huge_stack_starves_threads(), reason="an unmappable stack does not starve threads here"
)
def test_a_login_whose_relay_cannot_start_is_ended_and_reported(home: Path) -> None:
    """No relay thread: the hook is ended, its pipes closed, the window told (item 5)."""
    hook = _install_hook(home, _WAITING_HOOK)
    events: list[Event] = []
    login = SigninLogin()
    try:
        with _thread_stacks_too_large_to_map():
            login.start("test", lambda event, broadcast: events.append(event))
        assert [(e["type"], e["error"]) for e in events] == [
            ("signinError", "can't start new thread")
        ]
        assert login._proc is None
        assert login._ending == []
        assert not _relay_threads()
        (proc,) = _hook_processes(hook)
        assert proc.returncode == -signal.SIGTERM
        assert proc.stdin is not None and proc.stdin.closed
        assert proc.stdout is not None and proc.stdout.closed
    finally:
        login.shutdown()


@pytest.mark.skipif(
    not thread_start_can_be_starved(), reason="RLIMIT_NPROC cannot starve Thread.start on this host"
)
def test_a_cancel_whose_end_thread_cannot_start_still_ends_the_hook(home: Path) -> None:
    """The SIGTERM-ignoring hook is killed, bounded; ``shutdown`` has nothing unstarted (item 6)."""
    _install_hook(home, _STUBBORN_HOOK)
    events: list[Event] = []
    url_seen = threading.Event()

    def reply(event: Event, broadcast: bool) -> None:
        events.append(event)
        if event["type"] == "signinUrl":
            url_seen.set()

    login = SigninLogin()
    login.start("test", reply)
    proc = login._proc
    assert proc is not None
    try:
        assert url_seen.wait(10), events
        started = time.monotonic()
        with nproc_limit_lowered_to_one():
            login.cancel()
        assert time.monotonic() - started < 2 * signin.HOOK_END_TIMEOUT + 5.0
        assert proc.returncode == -signal.SIGKILL
        assert login._ending == []
        assert login._proc is None
        login.shutdown()
        assert _wait_until(lambda: not _relay_threads())
    finally:
        if proc.poll() is None:  # forgotten alive (the bug): not left to run out its minute
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        login.shutdown()
    assert [e["type"] for e in events] == ["signinUrl"], events
    assert proc.stdin is not None and proc.stdin.closed
    assert proc.stdout is not None and proc.stdout.closed


def test_status_hook_timeout_ends_the_hooks_descendants(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``status`` hook killed at the deadline takes the child it started with it."""
    monkeypatch.setattr(signin, "STATUS_TIMEOUT", 1.0)
    _install_hook(home, _SLOW_STATUS_HOOK)
    events: list[Event] = []
    pid_file = home / "child.pid"
    try:
        started = time.monotonic()
        send_signin(events.append)
        assert 1.0 <= time.monotonic() - started < 10.0
        assert events == []
        pid = int(pid_file.read_text())
        assert _wait_until(lambda: not _alive(pid), 5.0), "the hook's child outlived the deadline"
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), 9)
            except (OSError, ValueError):
                pass
