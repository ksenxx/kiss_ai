# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: a device login whose CLI the timer killed is never published.

``codex_device_login`` bounds the wait for the URL and code with a
``threading.Timer`` that kills the CLI.  The timer runs concurrently with
the stdout loop, so it could fire after the loop had collected the
prompt but before ``timer.cancel()``: the CLI was dead, yet the prompt
was stored as the login alive and returned, so the panel showed a code
nobody was polling for (audit 2026-10-18, B3 F5).

With the real CLI that interleaving is a sub-millisecond window at the
deadline.  The fake ``codex`` here makes it deterministic: the process
the timer kills only waits, while the banner is printed by a grandchild
in its own session (the npm ``codex`` is likewise a wrapper around a
native binary), so the prompt still arrives after the timer has fired.
Either way the caller must get an error, never a prompt whose CLI is
gone.
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.core.models import codex_model
from kiss.tests.conftest import posix_only

pytestmark = posix_only("the fake codex is a /bin/sh script whose banner forks a setsid grandchild")

_LATE_CODEX = """#!/bin/sh
if [ "$1" = "login" ] && [ "$2" = "--device-auth" ]; then
  exec "{python}" "{banner}"
fi
echo "Not logged in" >&2
exit 1
"""

# The Popen'd process (the wrapper) waits; the grandchild leaves its
# process group, prints the banner late and then exits, closing the pipes.
_LATE_BANNER = """
import os, sys, time
if os.fork() == 0:
    os.setsid()
    time.sleep(0.6)
    print("1. Open this link in your browser and sign in to your account")
    print("   https://auth.openai.com/codex/device")
    print("2. Enter this one-time code (expires in 15 minutes)")
    print("   CODE-LATE")
    sys.stdout.flush()
    time.sleep(0.3)
    os._exit(0)
os.wait()
"""


@pytest.fixture
def late_codex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    banner = tmp_path / "banner.py"
    banner.write_text(_LATE_BANNER)
    script = bin_dir / "codex"
    script.write_text(_LATE_CODEX.format(python=sys.executable, banner=banner))
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    yield script
    for reaper in threading.enumerate():
        if reaper.name == "codex-device-login":
            reaper.join(10)


def test_prompt_arriving_after_the_timer_killed_the_cli_is_an_error(late_codex: Path) -> None:
    result = codex_model.codex_device_login(timeout=0.2)
    assert "error" in result, result
    assert "printed no login code" in result["error"]
    # Nothing is left behind as the login alive: the next call starts over.
    assert codex_model._device_login is None


# The wrapper exits at once; its grandchild leaves the process group with
# stdout inherited and holds it, silent, for longer than any timeout here.
# Its stderr goes to /dev/null so only stdout is held.
_HOLDING_BANNER = """
import os, sys, time
pid = os.fork()
if pid == 0:
    os.setsid()
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
    time.sleep(20)
    os._exit(0)
with open(sys.argv[1], "w") as f:
    f.write(str(pid))
"""

_HOLDING_CODEX = """#!/bin/sh
if [ "$1" = "login" ] && [ "$2" = "--device-auth" ]; then
  exec "{python}" "{banner}" "{pid_file}"
fi
echo "Not logged in" >&2
exit 1
"""


@pytest.fixture
def holding_codex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    banner = tmp_path / "banner.py"
    banner.write_text(_HOLDING_BANNER)
    pid_file = tmp_path / "holder.pid"
    script = bin_dir / "codex"
    script.write_text(
        _HOLDING_CODEX.format(python=sys.executable, banner=banner, pid_file=pid_file)
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    yield pid_file
    with contextlib.suppress(OSError, ValueError):
        os.kill(int(pid_file.read_text()), signal.SIGKILL)


def test_timeout_holds_while_a_stray_descendant_keeps_stdout_open(holding_codex: Path) -> None:
    """The wait for the code ends at *timeout* even when the pipe never reaches EOF.

    A plain read of the CLI's stdout returns only when every holder of
    the pipe's write end is gone; a descendant in its own session
    survives the timeout's group kill, so the caller (the settings
    panel's reply thread, holding the login lock) sat in that read for
    as long as the descendant lived.
    """
    started = time.monotonic()
    result = codex_model.codex_device_login(timeout=0.2)
    elapsed = time.monotonic() - started
    assert "error" in result, result
    assert "printed no login code within 0.2s" in result["error"]
    assert elapsed < 5, f"the login wait outlived its timeout: {elapsed:.1f}s"
    assert codex_model._device_login is None


# A CLI that floods stdout before the banner: the lines queue up faster
# than the caller takes them, so at the deadline the queue is still full.
_NOISY_BANNER = """
import sys, time
sys.stdout.write("noise\\n" * 200000)
sys.stdout.write("   https://auth.openai.com/codex/device\\n   CODE-NOISY\\n")
sys.stdout.flush()
time.sleep(20)
"""

_NOISY_CODEX = """#!/bin/sh
if [ "$1" = "login" ] && [ "$2" = "--device-auth" ]; then
  exec "{python}" "{banner}"
fi
echo "Not logged in" >&2
exit 1
"""


@pytest.fixture
def noisy_codex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    banner = tmp_path / "banner.py"
    banner.write_text(_NOISY_BANNER)
    script = bin_dir / "codex"
    script.write_text(_NOISY_CODEX.format(python=sys.executable, banner=banner))
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    yield script
    for reaper in threading.enumerate():
        if reaper.name == "codex-device-login":
            reaper.join(10)


def test_the_deadline_holds_while_stdout_has_queued_lines(noisy_codex: Path) -> None:
    """A prompt buried under more output than the timeout allows reading is not
    published: the deadline is checked before every line, not only when the
    queue runs dry, so a chatty CLI cannot hold the login lock past the timeout."""
    started = time.monotonic()
    result = codex_model.codex_device_login(timeout=0.1)
    elapsed = time.monotonic() - started
    assert "error" in result, result
    assert "printed no login code within 0.1s" in result["error"]
    assert elapsed < 5, f"the login wait outlived its timeout: {elapsed:.1f}s"
    assert codex_model._device_login is None
