# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``run_agent`` dispatch ``timeout``.

The ``run_agent`` tool has a ``timeout`` parameter (a number string;
empty applies ``agent_dispatch.DEFAULT_DISPATCH_TIMEOUT_SECONDS``,
300 s).  On timeout the tool returns an error string and STOPS the
sub-task (``stop_on_timeout=True``): a channel sub-task must not
outlive its process-global workspace reservation — released the
moment the dispatch returns — or it could bind another account's
credentials when its channel tools load (gpt-5.6-sol review finding).
``daemon_client.run`` itself keeps the opposite default: on a plain
timeout it sends only ``closeTab``, never ``stop`` (the caller chose
to stop waiting, not to cancel the work), and it still accepts
``timeout=None`` ("no deadline": the event read wakes every
``_NO_DEADLINE_WAKE_SECONDS`` and retries, so an injected abort — the
``KeyboardInterrupt`` of a parent Stop — still gets delivered).

These tests run real UNIX-domain-socket daemon stand-ins and drive the
real client code:

* The ``run_agent`` tool (path mode, standard socket resolution via
  ``KISS_SORCAR_SOCK``) returns the delayed sub-task's YAML result,
  both with the default timeout and with an explicit one.  The delay
  is seconds, so this cannot behaviorally pin the default at exactly
  300 s — only a five-minute test could; the shrunk-constant timeout
  test below is the practical guard for the default wiring.
* A too-small ``timeout`` (explicit, or the shrunk default) yields the
  "did not finish within" error, with a ``stop`` + ``closeTab``
  cascade; the raw client sends the ``stop`` only when
  ``stop_on_timeout`` is passed, and then blocks until the daemon's
  terminal status confirms the task is dead (bounded by
  ``_STOP_CONFIRM_GRACE_SECONDS`` against a wedged daemon) — the
  ``run_agent`` channel dispatch releases its workspace reservation
  the moment the call returns, so the child must be dead by then.
* Invalid ``timeout`` strings are rejected before any dispatch.

Not covered here, and why: the stop-SEND failure branch (a broken
daemon connection at the exact moment the stop is written raises
``ConnectionError`` instead of a ``TimeoutError`` that would falsely
claim "was stopped") cannot be staged end-to-end without test
doubles — closing or shutting down a UDS peer makes the client's
blocking ``recv`` return EOF first, taking the ordinary
connection-loss path before any stop is attempted.
* ``daemon_client.run(timeout=None)`` returns a result delivered only
  after a delay, without raising ``TimeoutError``.
* A parent stopped (injected ``KeyboardInterrupt``) while blocked with
  ``timeout=None`` on a SILENT daemon still aborts at the next wake
  and cascades a ``stop`` to the dispatched task.
"""

from __future__ import annotations

import json
import shutil
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar import agent_dispatch, cron_agent, daemon_client
from kiss.agents.sorcar.agent_dispatch import make_run_agent_tool
from kiss.server.task_runner import inject_keyboard_interrupt
from kiss.tests.agents.sorcar.test_dispatch_stop_cascade import (
    _RecordingDaemon,
)


@pytest.fixture(autouse=True)
def _standalone_daemon_socket(monkeypatch: pytest.MonkeyPatch):
    """Isolate each test from a recorded in-process daemon socket.

    ``_dispatch`` prefers the socket the cron scheduler recorded at
    daemon boot (``cron_agent._daemon_sock_path``) over the standard
    ``KISS_SORCAR_SOCK`` resolution; a value left behind by another
    test module would divert the dispatch away from this module's fake
    daemons.
    """
    monkeypatch.setattr(cron_agent, "_daemon_sock_path", None)
    yield


class _SlowFinishDaemon:
    """A real UDS daemon stand-in whose result arrives after a delay.

    Accepts one connection, reads the initial ``run`` command, sends
    ``{"type": "status", "running": true}``, sleeps *delay* seconds,
    and only then sends the terminal ``result`` + ``status
    running=false`` pair — the shape of a long-running task on a live
    daemon.
    """

    def __init__(self, delay: float) -> None:
        """Bind a UNIX-domain listener in a fresh temp dir."""
        self.delay = delay
        self.run_cmd: dict[str, Any] | None = None
        self._dir = Path(tempfile.mkdtemp(prefix="kiss_no_timeout_"))
        self.sock_path = self._dir / "daemon.sock"
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(str(self.sock_path))
        self._srv.listen(1)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            conn, _ = self._srv.accept()
        except OSError:
            return
        with conn:
            reader = conn.makefile("rb")
            try:
                run_cmd: dict[str, Any] = json.loads(
                    reader.readline().decode("utf-8"),
                )
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                return
            self.run_cmd = run_cmd
            tab_id = run_cmd.get("tabId", "")

            def send(event: dict[str, Any]) -> None:
                try:
                    conn.sendall(json.dumps(event).encode("utf-8") + b"\n")
                except OSError:
                    pass

            send({"type": "status", "running": True, "tabId": tab_id})
            time.sleep(self.delay)
            send({
                "type": "result",
                "tabId": tab_id,
                "taskId": "task-slow-1",
                "success": True,
                "text": "slow but done",
                "cost": "$0.0100",
                "total_tokens": 5,
                "step_count": 1,
            })
            send({"type": "status", "running": False, "tabId": tab_id})
            # Absorb the client's closeTab before dropping the socket.
            try:
                reader.readline()
            except OSError:
                pass

    def close(self) -> None:
        """Shut down the listener and remove the temp socket dir."""
        try:
            self._srv.close()
        except OSError:
            pass
        self._thread.join(timeout=10)
        shutil.rmtree(self._dir, ignore_errors=True)


class _RawBytesDaemon:
    """A UDS daemon stand-in that sends a scripted byte payload.

    Accepts one connection, sends *payload* verbatim (never reading
    the client's ``run`` command — the socket buffers it), and keeps
    the connection open until closed, exercising the client's frame
    assembly on arbitrary wire bytes.
    """

    def __init__(self, payload: bytes) -> None:
        """Bind a UNIX-domain listener in a fresh temp dir."""
        self._dir = Path(tempfile.mkdtemp(prefix="kiss_no_timeout_"))
        self.sock_path = self._dir / "daemon.sock"
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(str(self.sock_path))
        self._srv.listen(1)
        self._conn: socket.socket | None = None
        self._payload = payload
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            self._conn, _ = self._srv.accept()
            self._conn.sendall(self._payload)
        except OSError:
            return

    def close(self) -> None:
        """Shut down the listener and remove the temp socket dir."""
        for closable in (self._conn, self._srv):
            try:
                if closable is not None:
                    closable.close()
            except OSError:
                pass
        self._thread.join(timeout=10)
        shutil.rmtree(self._dir, ignore_errors=True)


class _StopConfirmingDaemon:
    """A UDS daemon stand-in that confirms a ``stop`` with terminal status.

    Accepts one connection, reads the ``run`` command, sends ``status
    running=true`` (unless ``initial_running`` is false, modelling a
    task stopped during setup before that broadcast), then never
    finishes the task: it only records the client's further commands
    and, on receiving ``stop``, replies — after ``confirm_delay``
    seconds — with ``status running=false``, the shape of a live
    daemon stopping a task.
    """

    def __init__(
        self, confirm_delay: float = 0.0, initial_running: bool = True,
    ) -> None:
        """Bind a UNIX-domain listener in a fresh temp dir."""
        self.confirm_delay = confirm_delay
        self.initial_running = initial_running
        self.commands: list[dict[str, Any]] = []
        self.run_cmd: dict[str, Any] | None = None
        self._dir = Path(tempfile.mkdtemp(prefix="kiss_dispatch_timeout_"))
        self.sock_path = self._dir / "daemon.sock"
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(str(self.sock_path))
        self._srv.listen(1)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            conn, _ = self._srv.accept()
        except OSError:
            return
        with conn:
            reader = conn.makefile("rb")
            try:
                run_cmd: dict[str, Any] = json.loads(
                    reader.readline().decode("utf-8"),
                )
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                return
            self.run_cmd = run_cmd
            tab_id = run_cmd.get("tabId", "")

            def send(event: dict[str, Any]) -> None:
                try:
                    conn.sendall(json.dumps(event).encode("utf-8") + b"\n")
                except OSError:
                    pass

            if self.initial_running:
                send({"type": "status", "running": True, "tabId": tab_id})
            while True:
                try:
                    line = reader.readline()
                except OSError:
                    return
                if not line:
                    return
                try:
                    cmd = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                self.commands.append(cmd)
                if cmd.get("type") == "stop":
                    time.sleep(self.confirm_delay)
                    send({
                        "type": "status", "running": False, "tabId": tab_id,
                    })

    def wait_for_command(self, cmd_type: str, timeout: float = 5.0) -> bool:
        """Poll until a *cmd_type* command was recorded (or timeout)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(c.get("type") == cmd_type for c in self.commands):
                return True
            time.sleep(0.02)
        return False

    def close(self) -> None:
        """Shut down the listener and remove the temp socket dir."""
        try:
            self._srv.close()
        except OSError:
            pass
        self._thread.join(timeout=10)
        shutil.rmtree(self._dir, ignore_errors=True)


@pytest.mark.parametrize("terminated", [False, True])
def test_oversize_frame_rejected(
    monkeypatch: pytest.MonkeyPatch, terminated: bool,
) -> None:
    """A frame over the client cap fails loudly on both detection paths.

    ``terminated=False``: the cap's worth of newline-free bytes
    accumulates with no newline in sight (the pre-``recv`` check).
    ``terminated=True``: the newline lands in the same chunk that
    pushes the frame over the cap, so only the extracted-line length
    check can catch it (gpt-5.6-sol re-review finding: this path
    initially bypassed the cap).
    """
    monkeypatch.setattr(daemon_client, "_MAX_LINE_BYTES", 1024)
    payload = b"a" * 1500 + (b"\n" if terminated else b"")
    daemon = _RawBytesDaemon(payload)
    try:
        with pytest.raises(ConnectionError, match="frame larger"):
            daemon_client.run(
                "oversize probe", sock_path=daemon.sock_path, timeout=30.0,
            )
    finally:
        daemon.close()


def test_run_with_timeout_none_waits_for_delayed_result() -> None:
    """``daemon_client.run(timeout=None)`` has no deadline.

    The daemon stand-in delivers the result only after 1.5 s of
    silence.  With any finite deadline shorter than the delay the read
    loop raises ``TimeoutError``; ``timeout=None`` must keep waiting
    (wake-and-retry) until the result arrives.
    """
    daemon = _SlowFinishDaemon(delay=1.5)
    try:
        result = daemon_client.run(
            "slow child task", sock_path=daemon.sock_path, timeout=None,
        )
        assert result.success
        assert result.text == "slow but done"
        assert result.task_id == "task-slow-1"
    finally:
        daemon.close()


@pytest.mark.parametrize("timeout_arg", ["", "30"])
def test_run_agent_tool_waits_past_delayed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timeout_arg: str,
) -> None:
    """The ``run_agent`` tool waits out a delay shorter than its timeout.

    End-to-end through the real tool (path mode) and the standard
    ``KISS_SORCAR_SOCK`` socket resolution: the sub-task's result
    arrives after a delay well under the timeout (the 300-s default,
    and an explicit ``"30"``), and the tool returns the YAML result —
    not a "did not finish within …s" timeout message.
    """
    daemon = _SlowFinishDaemon(delay=1.5)
    monkeypatch.setenv("KISS_SORCAR_SOCK", str(daemon.sock_path))
    script = tmp_path / "slow_helper.py"
    script.write_text("def get_model() -> str:\n    return 'm'\n")
    try:
        out = make_run_agent_tool(str(tmp_path))(
            str(script), "say hi slowly", timeout=timeout_arg,
        )
        parsed = yaml.safe_load(out)
        assert parsed == {"success": True, "summary": "slow but done"}
        assert "did not finish within" not in out
        assert daemon.run_cmd is not None
        assert daemon.run_cmd.get("agentPath") == str(script)
    finally:
        daemon.close()


def test_interrupt_wakes_no_deadline_wait_on_silent_daemon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A parent Stop still aborts a ``timeout=None`` wait promptly.

    An injected async ``KeyboardInterrupt`` is delivered between
    bytecode instructions only, never inside a blocking C-level
    ``recv``, so a plain ``settimeout(None)`` read on a SILENT daemon
    would starve the stop cascade forever (gpt-5.6-sol review
    finding).  The no-deadline wait must instead wake periodically
    (``_NO_DEADLINE_WAKE_SECONDS``, shortened here so the test runs in
    milliseconds), let the interrupt fire, and cascade a ``stop`` +
    ``closeTab`` to the dispatched task.
    """
    monkeypatch.setattr(daemon_client, "_NO_DEADLINE_WAKE_SECONDS", 0.05)
    sock_dir = Path(tempfile.mkdtemp(prefix="kiss_no_timeout_"))
    path = sock_dir / "daemon.sock"
    daemon = _RecordingDaemon(path, mode="silent")
    outcome: dict[str, Any] = {}

    def call() -> None:
        try:
            daemon_client.run("silent child task", sock_path=path, timeout=None)
            outcome["exc"] = None
        except BaseException as exc:  # noqa: BLE001 — capture for assert
            outcome["exc"] = exc

    worker = threading.Thread(target=call, daemon=True)
    worker.start()
    try:
        deadline = time.monotonic() + 5
        while daemon.run_cmd is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert daemon.run_cmd is not None, "client never sent run"
        time.sleep(0.2)  # ensure the client is blocked in its read wait
        tid = worker.ident
        assert tid is not None
        assert inject_keyboard_interrupt(tid) == 1
        worker.join(timeout=10)
        assert not worker.is_alive(), (
            "the no-deadline wait never woke to deliver the injected "
            "KeyboardInterrupt — the stop cascade starves"
        )
        assert isinstance(outcome["exc"], KeyboardInterrupt)
        assert daemon.wait_for_command("stop"), (
            "the interrupted no-deadline dispatch never cascaded a stop"
        )
        assert daemon.wait_for_command("closeTab")
    finally:
        daemon.close()
        shutil.rmtree(sock_dir, ignore_errors=True)


def test_run_agent_tool_times_out_and_stops_the_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A too-small explicit ``timeout`` yields the error and a stop.

    End-to-end through the real tool (path mode): the daemon stand-in
    never finishes the task, so the tool must give up after the 0.5-s
    timeout, stop the dispatched task (``stop`` then ``closeTab``),
    await the stop's terminal-status confirmation, and return the "did
    not finish within 0.5s" error — a timed-out sub-task must not keep
    running (and, for channels, must not outlive its workspace
    reservation).
    """
    daemon = _StopConfirmingDaemon()
    monkeypatch.setenv("KISS_SORCAR_SOCK", str(daemon.sock_path))
    script = tmp_path / "helper.py"
    script.write_text("def get_model() -> str:\n    return 'm'\n")
    try:
        out = make_run_agent_tool(str(tmp_path))(
            str(script), "never finishes", timeout="0.5",
        )
        assert "did not finish within 0.5s" in out
        assert "was stopped" in out
        assert daemon.wait_for_command("stop"), (
            "the timed-out dispatch never stopped its sub-task"
        )
        assert daemon.wait_for_command("closeTab")
    finally:
        daemon.close()


def test_run_agent_tool_reports_unconfirmed_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A never-confirmed stop yields the "may still be running" error.

    End-to-end through the real tool (path mode) against a silent
    daemon stand-in: the ``stop`` sent on timeout is never answered,
    so once the (shrunk) confirmation grace expires the tool must NOT
    claim the task "was stopped" — it must say the task may still be
    running so the caller does not assume the work was cancelled.
    """
    monkeypatch.setattr(daemon_client, "_STOP_CONFIRM_GRACE_SECONDS", 0.3)
    sock_dir = Path(tempfile.mkdtemp(prefix="kiss_dispatch_timeout_"))
    path = sock_dir / "daemon.sock"
    daemon = _RecordingDaemon(path, mode="silent")
    monkeypatch.setenv("KISS_SORCAR_SOCK", str(path))
    script = tmp_path / "helper.py"
    script.write_text("def get_model() -> str:\n    return 'm'\n")
    try:
        out = make_run_agent_tool(str(tmp_path))(
            str(script), "never finishes", timeout="0.5",
        )
        assert "did not finish within 0.5s" in out
        assert "MAY STILL BE RUNNING" in out
        assert "was stopped" not in out
        assert daemon.wait_for_command("stop")
        assert daemon.wait_for_command("closeTab")
    finally:
        daemon.close()
        shutil.rmtree(sock_dir, ignore_errors=True)


@pytest.mark.parametrize("stop_on_timeout", [False, True])
def test_client_timeout_stop_cascade_is_opt_in(
    stop_on_timeout: bool,
) -> None:
    """``daemon_client.run`` stops a timed-out task only when asked.

    The public client keeps the documented timeout contract — a plain
    timeout sends ``closeTab`` and leaves the task running — while
    ``stop_on_timeout=True`` (what ``run_agent`` passes) sends a
    ``stop`` and BLOCKS until the daemon's terminal status confirms
    the task is dead: the daemon stand-in delays that confirmation by
    0.5 s, so an elapsed time past the delay proves the client waited
    for it rather than raising right after sending the stop.  The
    ``closeTab`` is sent after any ``stop``, so once it is observed
    the recorded commands are final.
    """
    daemon = _StopConfirmingDaemon(confirm_delay=0.5)
    try:
        begin = time.monotonic()
        with pytest.raises(TimeoutError, match="did not finish") as excinfo:
            daemon_client.run(
                "never finishes", sock_path=daemon.sock_path, timeout=0.3,
                stop_on_timeout=stop_on_timeout,
            )
        assert not isinstance(
            excinfo.value, daemon_client.StopUnconfirmedTimeoutError,
        ), "a confirmed (or never-attempted) stop must raise the plain error"
        elapsed = time.monotonic() - begin
        assert daemon.wait_for_command("closeTab")
        stopped = any(c.get("type") == "stop" for c in daemon.commands)
        assert stopped == stop_on_timeout
        if stop_on_timeout:
            assert elapsed >= 0.8, (
                "the client raised before the stop confirmation arrived"
            )
    finally:
        daemon.close()


def test_stop_confirmation_wait_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A daemon that never confirms the stop cannot wedge the client.

    The confirmation wait is bounded by
    ``_STOP_CONFIRM_GRACE_SECONDS`` (20 s in production, shrunk here):
    on a silent daemon the ``stop`` is sent but never answered, and
    once the grace expires the client must still raise — with
    ``StopUnconfirmedTimeoutError``, not the plain ``TimeoutError`` of
    a confirmed stop, because the stop stayed best-effort and the task
    may still be running.
    """
    monkeypatch.setattr(daemon_client, "_STOP_CONFIRM_GRACE_SECONDS", 0.3)
    sock_dir = Path(tempfile.mkdtemp(prefix="kiss_dispatch_timeout_"))
    path = sock_dir / "daemon.sock"
    daemon = _RecordingDaemon(path, mode="silent")
    try:
        begin = time.monotonic()
        with pytest.raises(
            daemon_client.StopUnconfirmedTimeoutError, match="did not finish",
        ):
            daemon_client.run(
                "never finishes", sock_path=path, timeout=0.3,
                stop_on_timeout=True,
            )
        assert time.monotonic() - begin < 5
        assert daemon.wait_for_command("stop")
        assert daemon.wait_for_command("closeTab")
    finally:
        daemon.close()
        shutil.rmtree(sock_dir, ignore_errors=True)


def test_stop_confirmed_before_initial_running_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stop confirmed without a prior ``running=true`` is confirmed.

    The daemon's stop watchdog can interrupt a task during its setup,
    BEFORE the initial ``running=true`` broadcast, while the daemon's
    ``finally`` still broadcasts the terminal ``running=false``
    (``task_runner._run_task``).  The client must accept that terminal
    status as stop confirmation — raising the plain ``TimeoutError``
    promptly — instead of ignoring it, wedging until the grace
    expires, and misreporting the stop as unconfirmed.
    """
    monkeypatch.setattr(daemon_client, "_STOP_CONFIRM_GRACE_SECONDS", 5.0)
    daemon = _StopConfirmingDaemon(initial_running=False)
    try:
        begin = time.monotonic()
        with pytest.raises(TimeoutError, match="did not finish") as excinfo:
            daemon_client.run(
                "never finishes", sock_path=daemon.sock_path, timeout=0.3,
                stop_on_timeout=True,
            )
        assert not isinstance(
            excinfo.value, daemon_client.StopUnconfirmedTimeoutError,
        ), "a daemon-confirmed stop must not be reported as unconfirmed"
        assert time.monotonic() - begin < 3, (
            "the client ignored the confirmation and waited out the grace"
        )
        assert daemon.wait_for_command("stop")
        assert daemon.wait_for_command("closeTab")
    finally:
        daemon.close()


def test_empty_timeout_applies_the_default_constant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty ``timeout`` falls back to the 300-s default constant.

    Waiting out the real 300-s default would take five minutes, so
    ``DEFAULT_DISPATCH_TIMEOUT_SECONDS`` (asserted to be 300 in
    production) is shrunk to 0.3 s and the tool is called WITHOUT a
    timeout argument against a silent daemon: the timeout error naming
    0.3 s proves the empty-string path reads the constant.
    """
    assert agent_dispatch.DEFAULT_DISPATCH_TIMEOUT_SECONDS == 300.0
    monkeypatch.setattr(
        agent_dispatch, "DEFAULT_DISPATCH_TIMEOUT_SECONDS", 0.3,
    )
    daemon = _StopConfirmingDaemon()
    monkeypatch.setenv("KISS_SORCAR_SOCK", str(daemon.sock_path))
    script = tmp_path / "helper.py"
    script.write_text("def get_model() -> str:\n    return 'm'\n")
    try:
        out = make_run_agent_tool(str(tmp_path))(str(script), "never finishes")
        assert "did not finish within 0.3s" in out
        assert "was stopped" in out
    finally:
        daemon.close()


@pytest.mark.parametrize("bad", ["abc", "1.5s", "0", "-5", "inf", "nan"])
def test_invalid_timeout_rejected_before_dispatch(bad: str) -> None:
    """A malformed or non-positive ``timeout`` is rejected up front.

    No daemon is listening anywhere in this test: the error must come
    from the tool's argument validation, before any path resolution or
    dispatch (the agent path passed here does not even exist).
    """
    out = make_run_agent_tool("")("no_such_agent.py", "task", timeout=bad)
    assert out.startswith("Error: timeout must be")
