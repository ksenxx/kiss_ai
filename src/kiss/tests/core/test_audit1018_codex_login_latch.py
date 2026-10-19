# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: one Codex device login at a time (audit 2026-10-18).

A fake ``codex`` on PATH that takes a second to print its device code
and then, like the real CLI, stays alive polling for the user to enter
it (until the test ends it), driven through the settings commands of
``_CommandsMixin``:

* two ``startCodexLogin`` requests while the first is still waiting for
  its code (another surface's click) start one CLI, and both senders
  get the code;
* a request while that CLI is still alive is answered with the same
  code, at once, and starts no CLI;
* a request carrying ``fresh: true`` ends that CLI and starts a new one;
* once the CLI has exited, a request starts a new one;
* ``getClaudeCodeStatus`` / ``getCodexStatus`` answer their sender.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.server.commands import _CommandsMixin
from kiss.tests.conftest import posix_only

pytestmark = posix_only("the fake codex is a POSIX script")

# Run ``n`` prints code ``CODE-RUNn`` after a second, then stays alive
# until the stop file exists (30 s at most).
_SLOW_CODEX = """#!/bin/sh
echo "$@" >> {runs}
if [ "$1" = "login" ] && [ "$2" = "--device-auth" ]; then
  sleep 1
  echo "1. Open this link in your browser and sign in to your account"
  echo "   https://auth.openai.com/codex/device"
  echo "2. Enter this one-time code (expires in 15 minutes)"
  echo "   CODE-RUN$(wc -l < {runs} | tr -d ' ')"
  n=0
  while [ ! -e {stop} ] && [ $n -lt 300 ]; do sleep 0.1; n=$((n+1)); done
  exit 0
fi
echo "Not logged in" >&2
exit 1
"""


class _Printer:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def broadcast(self, msg: dict[str, Any]) -> None:
        with self.lock:
            self.messages.append(msg)


class _Server(_CommandsMixin):
    def __init__(self) -> None:
        self.printer: Any = _Printer()
        self.work_dir = "/tmp"
        self._state_lock = threading.RLock()

    def replies(self, event_type: str, count: int, timeout: float = 20.0) -> list[dict[str, Any]]:
        """The first *count* events of *event_type*, waiting for them."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.printer.lock:
                found = [m for m in self.printer.messages if m.get("type") == event_type]
            if len(found) >= count:
                return found[:count]
            time.sleep(0.02)
        raise AssertionError(f"{count} {event_type} events did not arrive within {timeout}s")


class _SlowCodex:
    """The fake ``codex`` on PATH: its run log and the file that ends a login."""

    def __init__(self, tmp_path: Path) -> None:
        self.runs = tmp_path / "runs"
        self.stop = tmp_path / "stop"
        self.bin_dir = tmp_path / "bin"
        self.bin_dir.mkdir()
        script = self.bin_dir / "codex"
        script.write_text(_SLOW_CODEX.format(runs=self.runs, stop=self.stop))
        script.chmod(0o755)

    def run_count(self) -> int:
        return len(self.runs.read_text().splitlines()) if self.runs.exists() else 0

    def end_logins(self) -> None:
        """End every CLI alive and wait for its reaper, so no login is pending."""
        self.stop.touch()
        for reaper in threading.enumerate():
            if reaper.name == "codex-device-login":
                reaper.join(10)
                assert not reaper.is_alive()


@pytest.fixture
def slow_codex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_SlowCodex]:
    codex = _SlowCodex(tmp_path)
    monkeypatch.setenv("PATH", f"{codex.bin_dir}{os.pathsep}{os.environ['PATH']}")
    yield codex
    codex.end_logins()


def test_a_login_in_flight_answers_every_sender_with_one_cli(slow_codex: _SlowCodex) -> None:
    server = _Server()
    server._cmd_start_codex_login({"connId": "c1"})
    server._cmd_start_codex_login({"connId": "c2"})
    first, second = server.replies("codexLogin", 2)
    assert {first["connId"], second["connId"]} == {"c1", "c2"}
    assert first["code"] == second["code"] == "CODE-RUN1"
    assert first["url"] == "https://auth.openai.com/codex/device"
    assert slow_codex.run_count() == 1
    # The CLI is alive, waiting for the user to enter CODE-RUN1: a
    # later request (a retry, another surface) is answered with that
    # same code at once, no second CLI.
    started = time.monotonic()
    server._cmd_start_codex_login({"connId": "c3"})
    (third,) = server.replies("codexLogin", 3)[2:]
    assert time.monotonic() - started < 0.9
    assert third["connId"] == "c3"
    assert third["code"] == "CODE-RUN1"
    assert slow_codex.run_count() == 1


def test_a_fresh_request_ends_the_login_alive_and_starts_another(
    slow_codex: _SlowCodex,
) -> None:
    server = _Server()
    server._cmd_start_codex_login({"connId": "c1"})
    (first,) = server.replies("codexLogin", 1)
    assert first["code"] == "CODE-RUN1"
    server._cmd_start_codex_login({"connId": "c2", "fresh": True})
    (second,) = server.replies("codexLogin", 2)[1:]
    assert second["connId"] == "c2"
    assert second["code"] == "CODE-RUN2"
    assert slow_codex.run_count() == 2
    # The first CLI was ended (its reaper is done); the second is the
    # login alive, so a plain request gets its code.
    reapers = [t for t in threading.enumerate() if t.name == "codex-device-login"]
    assert len(reapers) == 1
    server._cmd_start_codex_login({"connId": "c3"})
    (third,) = server.replies("codexLogin", 3)[2:]
    assert third["code"] == "CODE-RUN2"
    assert slow_codex.run_count() == 2


def test_a_request_after_the_cli_exited_starts_a_new_login(slow_codex: _SlowCodex) -> None:
    server = _Server()
    server._cmd_start_codex_login({"connId": "c1"})
    (first,) = server.replies("codexLogin", 1)
    assert first["code"] == "CODE-RUN1"
    slow_codex.end_logins()
    slow_codex.stop.unlink()
    server._cmd_start_codex_login({"connId": "c2"})
    (second,) = server.replies("codexLogin", 2)[1:]
    assert second["code"] == "CODE-RUN2"
    assert slow_codex.run_count() == 2


def test_status_commands_answer_their_sender(slow_codex: _SlowCodex) -> None:
    server = _Server()
    server._cmd_get_codex_status({"connId": "c7"})
    (status,) = server.replies("codexStatus", 1)
    assert status["connId"] == "c7"
    assert status["loggedIn"] is False
    server._cmd_get_claude_code_status({"connId": "c8"})
    (claude,) = server.replies("claudeCodeStatus", 1)
    assert claude["connId"] == "c8"
    assert "loggedIn" in claude
