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

from kiss.agents.sorcar.docker_manager import _KILL_WAIT_S, MAX_OUTPUT_CHARS, STOPPED_MESSAGE
from kiss.agents.sorcar.exec_bridge import ExecBridge


class _Bridge(ThreadingHTTPServer):
    """A loopback bridge that records ``/exec`` commands and can stall ``/kill``."""

    daemon_threads = True

    def __init__(self, exec_delay: float, kill_delay: float) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.exec_delay = exec_delay
        self.kill_delay = kill_delay
        self.commands: list[str] = []
        self.kills: list[str] = []


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
            time.sleep(self.bridge.exec_delay)
            self._reply({"exit_code": 0, "output": "RAN"})
        else:
            self.bridge.kills.append(body["token"])
            time.sleep(self.bridge.kill_delay)
            self._reply({"ok": True})


@contextmanager
def _bridge(
    exec_delay: float = 0.0, kill_delay: float = 0.0,
) -> Iterator[tuple[_Bridge, ExecBridge]]:
    server = _Bridge(exec_delay, kill_delay)
    thread = threading.Thread(target=server.serve_forever, name="test-bridge", daemon=True)
    thread.start()
    try:
        yield server, ExecBridge(f"exec:http://127.0.0.1:{server.server_port}")
    finally:
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
        assert refusal.startswith("Blocked:"), refusal
        report = bridge.run_commands_parallel(json.dumps(["rm -rf /", "echo ok"]))
        assert server.commands == ["echo ok"], server.commands
    assert report.startswith("2 commands: 1 succeeded, 1 failed, 0 timed out.")
    assert "Blocked:" in report


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
