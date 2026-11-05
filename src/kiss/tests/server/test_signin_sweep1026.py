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
3. ``_relay`` reads the hook's stdout only until the hook's exit, not
   until EOF: a helper the hook started with its stdout inherited (a
   browser opener) kept the pipe's write end open after the hook had
   printed its last line and exited 0, and ``signinDone`` waited for
   the helper's exit.  At the exit it reads what the pipe holds then,
   not until the pipe runs dry: a helper that keeps writing kept
   ``signinDone`` waiting just the same.
4. ``paste`` ends the login when its write is cut short: a write of
   more than ``PIPE_BUF`` bytes into a nearly full pipe is partial, and
   its prefix, with no newline, would be glued to the next paste the
   hook read.
"""

from __future__ import annotations

import gc
import os
import select
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
        # Pastes of PIPE_BUF bytes (the line with its newline), which a
        # pipe takes whole or not at all; 128 KiB of them are more than
        # any pipe buffers, and the hook reads none, so one is refused.
        for _ in range(2 * 65536 // select.PIPE_BUF):
            ok = login.paste("u" * (select.PIPE_BUF - 1))
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


# ``login`` prints its URL, starts a helper (the program in
# ``$SIGNIN_TEST_HELPER``) that inherits its stdout and outlives it
# (records its pid), prints its success line and exits 0.
STDOUT_HELPER_HOOK = f"""#!{sys.executable}
import os, subprocess, sys
home = os.environ["KISS_HOME"]
if sys.argv[1:2] == ["login"]:
    print('{{"url": "http://127.0.0.1:1/login"}}', flush=True)
    child = subprocess.Popen([sys.executable, "-c", os.environ["SIGNIN_TEST_HELPER"]])
    with open(os.path.join(home, "sleeper.pid"), "w") as fh:
        fh.write(str(child.pid))
    print('{{"logged_in": true, "email": "me@example.com"}}', flush=True)
    sys.exit(0)
print('{{"logged_in": false}}')
"""

SILENT_HELPER = "import time; time.sleep(60)"

# Writes blank lines to the inherited stdout for as long as the pipe has a
# reader, so the pipe never runs empty; sleeps on once the reader is gone.
NOISY_HELPER = """import os, time
while True:
    try:
        os.write(1, b"\\n" * 65536)
    except OSError:
        time.sleep(60)
"""


@posix_only("the stub sign-in hook is a script with a shebang")
@pytest.mark.parametrize("helper", [SILENT_HELPER, NOISY_HELPER], ids=["silent", "noisy"])
def test_a_helper_holding_the_finished_hooks_stdout_cannot_delay_signin_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, helper: str
) -> None:
    """The hook exited 0 after its last line; its helper keeps its stdout: ``signinDone`` comes.

    Whether the helper is silent or keeps the pipe readable: the relay
    reads what the pipe held at the hook's exit, not until it runs dry.
    """
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setenv("SIGNIN_TEST_HELPER", helper)
    hook = tmp_path / SIGNIN_HOOK
    hook.write_text(STDOUT_HELPER_HOOK)
    hook.chmod(0o755)
    events: list[dict[str, object]] = []
    done = threading.Event()

    def reply(event: dict[str, object], broadcast: bool) -> None:
        events.append(event)
        if event["type"] == "signinDone":
            done.set()

    login = SigninLogin()
    forget_status()
    pid_file = tmp_path / "sleeper.pid"
    try:
        login.start("test", reply)
        assert done.wait(10), "signinDone waited for the helper holding the hook's stdout"
        deadline = time.monotonic() + 10
        while _relay_threads() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _relay_threads()
        os.kill(int(pid_file.read_text()), 0)  # the helper (the user's browser, say) lives on
    finally:
        forget_status()
        if pid_file.exists():
            os.kill(int(pid_file.read_text()), signal.SIGKILL)
    assert [e["type"] for e in events] == ["signinUrl", "signinDone"]
    assert login._proc is None
    (proc,) = _hook_processes(hook)
    assert proc.returncode == 0
    assert proc.stdin is not None and proc.stdin.closed
    assert proc.stdout is not None and proc.stdout.closed


# ``login`` prints its URL and waits for the sign-in without ever reading
# its stdin, so the pastes pile up in the pipe.
DEAF_HOOK = f"""#!{sys.executable}
import sys, time
if sys.argv[1:2] == ["login"]:
    print('{{"url": "http://127.0.0.1:1/login"}}', flush=True)
    time.sleep(60)
print('{{"logged_in": false}}')
"""


@posix_only("the stub sign-in hook is a script with a shebang")
def test_a_paste_cut_short_by_a_nearly_full_pipe_ends_the_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial write leaves a line fragment in the hook's stdin: the login is ended, silently."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    hook = tmp_path / SIGNIN_HOOK
    hook.write_text(DEAF_HOOK)
    hook.chmod(0o755)
    events: list[dict[str, object]] = []
    url_seen = threading.Event()

    def reply(event: dict[str, object], broadcast: bool) -> None:
        events.append(event)
        if event["type"] == "signinUrl":
            url_seen.set()

    login = SigninLogin()
    forget_status()
    try:
        login.start("test", reply)
        assert url_seen.wait(10), events
        pastes: list[bool] = []
        # Lines of PASTE_LIMIT + 1 bytes, more than PIPE_BUF: the first
        # that does not fit whole is written in part, not refused whole.
        for _ in range(64):
            pastes.append(login.paste("u" * PASTE_LIMIT))
            if not pastes[-1]:
                break
        assert pastes[-1] is False and True in pastes, pastes
        assert login._proc is None, "a cut-short paste left the login running on a corrupt stream"
        assert login.paste("VALID-RETRY") is False
        login.shutdown()
        deadline = time.monotonic() + 10
        while _relay_threads() and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        forget_status()
        login.shutdown()
    assert not _relay_threads()
    assert [e["type"] for e in events] == ["signinUrl"], events
    (proc,) = _hook_processes(hook)
    assert proc.returncode == -signal.SIGTERM
    assert proc.stdin is not None and proc.stdin.closed
    assert proc.stdout is not None and proc.stdout.closed
