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

import os
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.core.models import codex_model

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
