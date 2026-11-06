# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regressions for the exec bridge's Stop and shell guards.

* A task Stop during ``ExecBridge.Bash`` returns within
  :data:`kiss.agents.sorcar.docker_manager._KILL_WAIT_S` even when the
  bridge's ``/kill`` handler (which sweeps the remote sandbox before it
  replies) is slow: the kill is posted from a helper thread instead of
  blocking the task thread for its 120 s socket timeout.
* ``ExecBridge.run_commands_parallel`` applies the same shell guards as
  ``ExecBridge.Bash`` (the destructive-command refusal and the install
  timeout lift), so a command ``Bash`` refuses never reaches the bridge
  through the fan-out.
* ``ExecBridge.Bash`` defaults ``max_output_chars`` to the Docker
  manager's :data:`MAX_OUTPUT_CHARS`, not a private literal.
* ``bridge_exec`` has an overall deadline (the command's plus
  :data:`kiss.agents.sorcar.exec_bridge._REPLY_MARGIN_S`): a bridge that
  trickles a reply body that never completes defeats urllib's idle
  timeout, so the wait itself must end, kill the command and report the
  timeout (the test waits that margin out: about a minute).
* A transport failure after ``/exec`` was accepted (the reply connection
  dropped) kills the command on the environment before the error
  propagates: the command must not go on after its tool call failed.
"""

from __future__ import annotations

import inspect
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

import pytest

from kiss.agents.sorcar.docker_manager import _KILL_WAIT_S, MAX_OUTPUT_CHARS, STOPPED_MESSAGE
from kiss.agents.sorcar.exec_bridge import _REPLY_MARGIN_S, ExecBridge, bridge_exec
from kiss.agents.sorcar.shell_guards import DESTRUCTIVE_VERDICT
from kiss.core.kiss_error import KISSError


class _Bridge(ThreadingHTTPServer):
    """A loopback bridge that records ``/exec`` commands and can stall ``/kill``.

    *exec_fault* makes ``/exec`` misbehave: ``"trickle"`` sends a reply
    body that never completes (a byte every 50 ms until the command's
    ``/kill`` arrives or the bridge is torn down), ``"drop"`` closes the
    connection without a reply.
    """

    daemon_threads = True

    def __init__(self, exec_delay: float, kill_delay: float, exec_fault: str = "") -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.exec_delay = exec_delay
        self.kill_delay = kill_delay
        self.exec_fault = exec_fault
        self.commands: list[str] = []
        self.tokens: list[str] = []
        self.kills: list[str] = []
        self.torn_down = threading.Event()


class _Handler(BaseHTTPRequestHandler):
    @property
    def bridge(self) -> _Bridge:
        """The owning ``_Bridge`` server."""
        return cast(_Bridge, self.server)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence request logging."""

    def _reply(self, obj: dict[str, Any]) -> None:
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        self._reply({"ok": True})

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/exec":
            self.bridge.commands.append(body["command"])
            self.bridge.tokens.append(body["token"])
            if self.bridge.exec_fault == "drop":
                return  # HTTP/1.0: the connection closes without a reply
            if self.bridge.exec_fault == "trickle":
                self._trickle(body["token"])
                return
            time.sleep(self.bridge.exec_delay)
            self._reply({"exit_code": 0, "output": "RAN"})
        else:
            self.bridge.kills.append(body["token"])
            time.sleep(self.bridge.kill_delay)
            self._reply({"ok": True})

    def _trickle(self, token: str) -> None:
        """Send a reply body that never completes, a byte every 50 ms, until *token* is killed."""
        self.send_response(200)
        self.end_headers()
        while token not in self.bridge.kills and not self.bridge.torn_down.is_set():
            self.wfile.write(b" ")
            time.sleep(0.05)


@contextmanager
def _bridge(
    exec_delay: float = 0.0, kill_delay: float = 0.0, exec_fault: str = "",
) -> Iterator[tuple[_Bridge, ExecBridge]]:
    server = _Bridge(exec_delay, kill_delay, exec_fault)
    thread = threading.Thread(target=server.serve_forever, name="test-bridge", daemon=True)
    thread.start()
    try:
        yield server, ExecBridge(f"exec:http://127.0.0.1:{server.server_port}")
    finally:
        server.torn_down.set()
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_stop_returns_within_kill_wait_even_when_the_kill_handler_is_slow() -> None:
    with _bridge(exec_delay=30, kill_delay=6) as (server, bridge):
        bridge.stop_event = threading.Event()
        threading.Timer(0.3, bridge.stop_event.set).start()
        started = time.monotonic()
        out = bridge.Bash("sleep 30", "stopped command", timeout_seconds=30)
        elapsed = time.monotonic() - started
        assert out.startswith(STOPPED_MESSAGE), out
        assert elapsed < 0.3 + _KILL_WAIT_S + 1.0, f"Stop took {elapsed:.1f}s"
        # The kill was still posted (the server receives it while sleeping).
        deadline = time.monotonic() + 2
        while not server.kills and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(server.kills) == 1


def test_run_commands_parallel_applies_the_bash_shell_guards() -> None:
    with _bridge() as (server, bridge):
        refusal = bridge.Bash("rm -rf /", "refused")
        assert refusal == DESTRUCTIVE_VERDICT, refusal
        report = bridge.run_commands_parallel(json.dumps(["rm -rf /", "echo ok"]))
        assert server.commands == ["echo ok"], server.commands
    assert report.startswith("2 commands: 1 succeeded, 1 failed, 0 timed out.")
    assert DESTRUCTIVE_VERDICT in report


def test_run_commands_parallel_rejects_bad_arguments_like_the_other_tools() -> None:
    with _bridge() as (server, bridge):
        assert bridge.run_commands_parallel("not json").startswith("Error:")
        assert bridge.run_commands_parallel('["echo hi"]', max_workers=-1) == (
            "Error: max_workers must be 0 or a positive integer, got -1."
        )
        assert server.commands == []


def test_bash_default_output_cap_is_the_shared_constant() -> None:
    default = inspect.signature(ExecBridge.Bash).parameters["max_output_chars"].default
    assert default == MAX_OUTPUT_CHARS


def _exec_through(url: str, results: list[tuple[int | None, str]]) -> None:
    """Run ``bridge_exec`` with a deadline that passes at once and record its result."""
    results.append(bridge_exec(url, "sleep 30", 0.01))


def test_a_reply_that_never_completes_ends_at_the_overall_deadline() -> None:
    with _bridge(exec_fault="trickle") as (server, bridge):
        results: list[tuple[int | None, str]] = []
        runner = threading.Thread(target=_exec_through, args=(bridge.url, results), daemon=True)
        started = time.monotonic()
        runner.start()
        runner.join(0.01 + _REPLY_MARGIN_S + _KILL_WAIT_S + 2.0)
        elapsed = time.monotonic() - started
        assert not runner.is_alive(), f"bridge_exec still blocked after {elapsed:.1f}s"
        assert elapsed >= 0.01 + _REPLY_MARGIN_S, f"gave up before the margin: {elapsed:.1f}s"
        assert results == [(None, "")], results
        assert server.kills == server.tokens == [server.tokens[0]]


def test_a_reply_dropped_after_exec_was_accepted_kills_the_command() -> None:
    with _bridge(exec_fault="drop") as (server, bridge):
        started = time.monotonic()
        with pytest.raises(KISSError, match="unreachable"):
            bridge_exec(bridge.url, "sleep 30", 30)
        assert time.monotonic() - started < _KILL_WAIT_S + 1.0
        assert server.commands == ["sleep 30"]
        assert server.kills == server.tokens == [server.tokens[0]]
