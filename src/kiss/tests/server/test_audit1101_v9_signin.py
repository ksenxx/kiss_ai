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

The hooks are real executables under a temporary ``KISS_HOME``.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.server import signin
from kiss.server.signin import SIGNIN_HOOK, SigninLogin, send_signin
from kiss.tests.conftest import posix_only
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
