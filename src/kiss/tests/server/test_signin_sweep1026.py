"""``SigninLogin`` after the 2026-10-26 sweep.

1. ``_relay`` closes the hook's pipes when the hook finishes on its
   own.  They used to stay open until the ``Popen`` object was
   collected at the relay thread's exit, which CPython reports as
   ``ResourceWarning: unclosed file`` once per completed login.  Under
   pytest the suite's subprocess reaper keeps every ``Popen``
   referenced, so that finaliser never runs here; the test finds the
   hook's ``Popen`` and looks at its pipes directly.
2. ``paste`` never blocks: the hook's stdin pipe is non-blocking, so a
   paste that does not fit (the hook is not reading) is refused at
   once instead of parking the writer on the pipe.  A parked writer
   held the stream's buffer lock, and the relay's close of that stream
   after the hook's exit waited for it: a hook whose descendant
   inherited the pipe and lived on (a sleeping helper) kept
   ``signinDone`` from ever reaching the window.
"""

from __future__ import annotations

import gc
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from kiss.server.signin import PASTE_LIMIT, SIGNIN_HOOK, SigninLogin
from kiss.tests.conftest import posix_only
from kiss.tests.server.test_signin_window import forget_status

HOOK = """#!/bin/sh
case "$1" in
  login)
    echo '{"url": "http://127.0.0.1:1/login"}'
    echo '{"logged_in": true, "email": "me@example.com"}'
    exit 0 ;;
esac
echo '{"logged_in": false}'
"""


def _relay_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "signin-login"]


def _hook_processes(hook: Path) -> list[subprocess.Popen[str]]:
    return [
        o for o in gc.get_objects()
        if isinstance(o, subprocess.Popen) and isinstance(o.args, list) and o.args[0] == str(hook)
    ]


@posix_only("the stub sign-in hook is a shell script")
def test_finished_hook_leaves_no_open_pipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    hook = tmp_path / SIGNIN_HOOK
    hook.write_text(HOOK)
    hook.chmod(0o755)
    events: list[dict[str, object]] = []
    done = threading.Event()

    def reply(event: dict[str, object], broadcast: bool) -> None:
        events.append(event)
        if event["type"] == "signinDone":
            done.set()

    login = SigninLogin()
    forget_status()
    try:
        login.start("test", reply)
        assert done.wait(10), events
        deadline = time.monotonic() + 10
        while _relay_threads() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _relay_threads()
    finally:
        forget_status()
    assert [e["type"] for e in events] == ["signinUrl", "signinDone"]
    assert events[1]["email"] == "me@example.com"
    (proc,) = _hook_processes(hook)
    assert proc.returncode == 0
    assert proc.stdin is not None and proc.stdin.closed
    assert proc.stdout is not None and proc.stdout.closed


LINGERING_HOOK = f"""#!{sys.executable}
import os, subprocess, sys, time
home = os.environ["KISS_HOME"]
if sys.argv[1:2] == ["login"]:
    print('{{"url": "http://127.0.0.1:1/login"}}', flush=True)
    # Inherits the daemon's stdin pipe and never reads it; outlives the hook.
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    with open(os.path.join(home, "sleeper.pid"), "w") as fh:
        fh.write(str(child.pid))
    while not os.path.exists(os.path.join(home, "finish")):
        time.sleep(0.02)
    print('{{"logged_in": true, "email": "me@example.com"}}', flush=True)
    sys.exit(0)
print('{{"logged_in": false}}')
"""


@posix_only("the stub sign-in hook is a script with a shebang")
def test_paste_the_hook_never_reads_cannot_delay_signin_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pastes fill the pipe and a descendant keeps it after the hook: ``signinDone`` still comes."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    hook = tmp_path / SIGNIN_HOOK
    hook.write_text(LINGERING_HOOK)
    hook.chmod(0o755)
    events: list[dict[str, object]] = []
    url_seen = threading.Event()
    done = threading.Event()

    def reply(event: dict[str, object], broadcast: bool) -> None:
        events.append(event)
        if event["type"] == "signinUrl":
            url_seen.set()
        if event["type"] == "signinDone":
            done.set()

    pastes: list[bool] = []

    def paste_until_refused() -> None:
        # 64 pastes of 4 KiB are more than any pipe buffers; the hook
        # reads none of them, so one of them must be refused.
        for _ in range(64):
            ok = login.paste("u" * PASTE_LIMIT)
            pastes.append(ok)
            if not ok:
                return

    login = SigninLogin()
    forget_status()
    paster = threading.Thread(target=paste_until_refused, daemon=True)
    try:
        login.start("test", reply)
        assert url_seen.wait(10), events
        paster.start()
        paster.join(10)
        (tmp_path / "finish").touch()
        assert done.wait(10), (events, pastes)
        deadline = time.monotonic() + 10
        while _relay_threads() and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        forget_status()
        pid_file = tmp_path / "sleeper.pid"
        if pid_file.exists():
            os.kill(int(pid_file.read_text()), signal.SIGKILL)
    assert not paster.is_alive(), "a paste the hook never read blocked its writer"
    assert pastes[-1] is False and True in pastes, pastes
    assert [e["type"] for e in events] == ["signinUrl", "signinDone"]
    assert not _relay_threads()
    (proc,) = _hook_processes(hook)
    assert proc.returncode == 0
    assert proc.stdin is not None and proc.stdin.closed
    assert proc.stdout is not None and proc.stdout.closed
