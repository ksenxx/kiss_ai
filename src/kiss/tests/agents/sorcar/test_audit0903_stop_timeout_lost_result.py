# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: ``stop_on_timeout`` must not discard a raced natural finish.

Race (audit 0903, ``daemon_client.run``): with ``stop_on_timeout=True``
the deadline check and the daemon's natural task completion are a
cross-process check-then-act.  When the task finishes on its own just
as the client's deadline expires, the successful terminal ``result``
(and ``status running=false``) are already on the wire while the client
sends its now-useless ``stop``.  The old code then read the successful
result, stored it, and still raised ``TimeoutError`` from the
``stopping`` confirmation branch — reporting to the caller (the
``run_agent`` dispatch) that the task "did not finish … and was
stopped" while its completed result was in hand: a lost update.

The fix returns the stored result from ONLY the terminal-status
confirmation exit, and only when the result reports ``success``: every
daemon-side stop / cancel / failure path broadcasts its terminal
``result`` with ``success: false``
(``task_runner._broadcast_failure_result`` and the setup-failure
broadcasts are the only result-emitting failure paths), so a
``success: true`` result can only come from a run that completed on
its own, never from the client's stop taking effect.

The grace-expiry exit, by contrast, NEVER returns a stored result —
not even a successful one (gpt-5.6-sol review, Finding 1).  The result
is emitted by the agent BEFORE the daemon's persistence, auto-commit,
worktree cleanup, and presentation stages run, and ``_stop_task`` can
still take effect during those stages; only the terminal ``status
running=false`` broadcast by the daemon's outermost ``finally`` proves
the task thread is dead.  Returning success without it would let
``run_agent`` release its process-global workspace reservation while
the dispatched task is still touching the workspace.

Branch coverage of the modified code (the terminal-status
``result_event`` check and the unconditional grace-expiry raise),
including the branches exercised by pre-existing tests:

* stopping + terminal status + successful result → result returned
  (``test_stop_on_timeout_returns_naturally_finished_result``, and via
  the ``run_agent`` tool in
  ``test_run_agent_tool_returns_result_when_finish_races_timeout``).
* stopping + terminal status + FAILURE result (the stop killed the
  task) → plain ``TimeoutError``
  (``test_stop_killed_task_still_raises_timeout``).
* stopping + terminal status + NO result → plain ``TimeoutError``
  (pre-existing ``test_dispatch_timeout.py::
  test_client_timeout_stop_cascade_is_opt_in``).
* grace expired + successful result (terminal status never sent) →
  ``StopUnconfirmedTimeoutError``
  (``test_unconfirmed_stop_with_successful_result_still_raises``, and
  via the ``run_agent`` tool in
  ``test_run_agent_tool_reports_unconfirmed_stop_despite_result``).
* grace expired + no/failure result → ``StopUnconfirmedTimeoutError``
  (pre-existing ``test_dispatch_timeout.py::
  test_stop_confirmation_wait_is_bounded``, and
  ``test_unconfirmed_stop_with_failure_result_still_raises`` here).

All daemons below are real UNIX-domain-socket stand-ins (no mocks of
the code under test), following ``test_dispatch_timeout.py``.
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

from kiss.agents.sorcar import cron_agent, daemon_client
from kiss.agents.sorcar.agent_dispatch import make_run_agent_tool


@pytest.fixture(autouse=True)
def _standalone_daemon_socket(monkeypatch: pytest.MonkeyPatch):
    """Keep dispatches off any in-process daemon socket recorded at boot."""
    monkeypatch.setattr(cron_agent, "_daemon_sock_path", None)
    yield


class _RacingFinishDaemon:
    """A UDS daemon stand-in whose task finishes naturally past the timeout.

    Accepts one connection, reads the ``run`` command, sends ``status
    running=true``, and — from a separate timer thread, exactly like a
    real task thread finishing while the command reader is busy —
    sends a SUCCESSFUL terminal ``result`` after ``finish_delay``
    seconds, followed by ``status running=false`` unless
    ``send_terminal_status`` is false (modelling a daemon that wedges
    between the two broadcasts).  Client commands (``stop``,
    ``closeTab``) are recorded and otherwise ignored: by the time the
    stop arrives the task is finishing on its own, so a real daemon's
    run-token-guarded ``_stop_task`` would be a no-op too.
    """

    def __init__(
        self, finish_delay: float, send_terminal_status: bool = True,
    ) -> None:
        """Bind a UNIX-domain listener in a fresh temp dir."""
        self.finish_delay = finish_delay
        self.send_terminal_status = send_terminal_status
        self.commands: list[dict[str, Any]] = []
        self._dir = Path(tempfile.mkdtemp(prefix="kiss_audit0903_"))
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
            tab_id = run_cmd.get("tabId", "")
            send_lock = threading.Lock()

            def send(event: dict[str, Any]) -> None:
                with send_lock:
                    try:
                        conn.sendall(
                            json.dumps(event).encode("utf-8") + b"\n",
                        )
                    except OSError:
                        pass

            def finish() -> None:
                send({
                    "type": "result",
                    "tabId": tab_id,
                    "taskId": "task-raced-1",
                    "success": True,
                    "text": "finished on my own",
                    "cost": "$0.0200",
                    "total_tokens": 7,
                    "step_count": 2,
                })
                if self.send_terminal_status:
                    send({
                        "type": "status", "running": False, "tabId": tab_id,
                    })

            send({"type": "status", "running": True, "tabId": tab_id})
            finisher = threading.Timer(self.finish_delay, finish)
            finisher.daemon = True
            finisher.start()
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


class _StopKillDaemon:
    """A UDS daemon stand-in whose task dies only when stopped.

    Sends ``status running=true`` and never finishes the task; on the
    client's ``stop`` it broadcasts the failure ``result`` a real
    daemon's cancel path emits (``success: false``, "Task stopped by
    user" — see ``task_runner._broadcast_failure_result``), followed by
    the terminal ``status running=false`` unless
    ``send_terminal_status`` is false.
    """

    def __init__(self, send_terminal_status: bool = True) -> None:
        """Bind a UNIX-domain listener in a fresh temp dir."""
        self.send_terminal_status = send_terminal_status
        self.commands: list[dict[str, Any]] = []
        self._dir = Path(tempfile.mkdtemp(prefix="kiss_audit0903_"))
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
            tab_id = run_cmd.get("tabId", "")

            def send(event: dict[str, Any]) -> None:
                try:
                    conn.sendall(json.dumps(event).encode("utf-8") + b"\n")
                except OSError:
                    pass

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
                    send({
                        "type": "result",
                        "tabId": tab_id,
                        "taskId": "task-killed-1",
                        "success": False,
                        "text": "Task stopped by user",
                        "cost": "$0.0100",
                        "total_tokens": 3,
                        "step_count": 1,
                    })
                    if self.send_terminal_status:
                        send({
                            "type": "status",
                            "running": False,
                            "tabId": tab_id,
                        })

    def close(self) -> None:
        """Shut down the listener and remove the temp socket dir."""
        try:
            self._srv.close()
        except OSError:
            pass
        self._thread.join(timeout=10)
        shutil.rmtree(self._dir, ignore_errors=True)


def test_stop_on_timeout_returns_naturally_finished_result() -> None:
    """A natural finish racing the timeout's stop must not be discarded.

    The daemon stand-in finishes the task successfully 0.6 s in — past
    the client's 0.3-s deadline, so the client has already sent its
    ``stop`` — and the completed result + terminal status cross the
    stop on the wire.  The client holds the successful result when the
    terminal status confirms the task is dead: it must return it, not
    raise a ``TimeoutError`` claiming the task "was stopped".
    """
    daemon = _RacingFinishDaemon(finish_delay=0.6)
    try:
        result = daemon_client.run(
            "finishes while stop is in flight",
            sock_path=daemon.sock_path,
            timeout=0.3,
            stop_on_timeout=True,
        )
        assert result.success
        assert result.text == "finished on my own"
        assert result.cost == pytest.approx(0.02)
        assert result.tokens == 7
        assert result.steps == 2
        assert result.task_id == "task-raced-1"
        # The stop was still sent (the deadline did expire first).
        assert daemon.wait_for_command("stop")
        assert daemon.wait_for_command("closeTab")
    finally:
        daemon.close()


def test_run_agent_tool_returns_result_when_finish_races_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``run_agent`` tool reports the raced natural finish as success.

    End-to-end through the real tool (path mode, ``KISS_SORCAR_SOCK``
    resolution): the caller must get the sub-task's YAML result — its
    work is done and its spend/summary known — instead of the "did not
    finish within …s and was stopped" error string.
    """
    daemon = _RacingFinishDaemon(finish_delay=0.6)
    monkeypatch.setenv("KISS_SORCAR_SOCK", str(daemon.sock_path))
    script = tmp_path / "helper.py"
    script.write_text("def get_model() -> str:\n    return 'm'\n")
    try:
        out = make_run_agent_tool(str(tmp_path))(
            str(script), "finishes while stop is in flight", timeout="0.3",
        )
        assert "did not finish within" not in out
        assert yaml.safe_load(out) == {
            "success": True, "summary": "finished on my own",
        }
    finally:
        daemon.close()


def test_stop_killed_task_still_raises_timeout() -> None:
    """A stop-killed task's failure result must not mask the timeout.

    The daemon's cancel path broadcasts a ``success: false`` result
    ("Task stopped by user") before the terminal status.  That result
    is a consequence of the client's own stop, not completed work, so
    the client must keep raising the plain ``TimeoutError`` — returning
    the failure result would hide the timeout (and its "retry with a
    larger timeout" guidance) from the dispatching caller.
    """
    daemon = _StopKillDaemon()
    try:
        with pytest.raises(TimeoutError, match="did not finish") as excinfo:
            daemon_client.run(
                "never finishes",
                sock_path=daemon.sock_path,
                timeout=0.3,
                stop_on_timeout=True,
            )
        assert not isinstance(
            excinfo.value, daemon_client.StopUnconfirmedTimeoutError,
        ), "a daemon-confirmed stop must raise the plain TimeoutError"
    finally:
        daemon.close()


def test_unconfirmed_stop_with_successful_result_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful result does NOT settle a never-confirmed stop wait.

    The daemon delivers the successful result but wedges before the
    terminal ``status running=false`` broadcast.  The result is NOT
    proof the task thread is dead: the daemon emits it before its
    persistence / auto-commit / worktree-cleanup stages, and the stop
    can still take effect during them — so the workspace may still be
    in use.  Once the (shrunk) confirmation grace expires the client
    must raise ``StopUnconfirmedTimeoutError``, never return the
    result as if the task were confirmed finished.
    """
    monkeypatch.setattr(daemon_client, "_STOP_CONFIRM_GRACE_SECONDS", 1.0)
    daemon = _RacingFinishDaemon(finish_delay=0.6, send_terminal_status=False)
    try:
        with pytest.raises(
            daemon_client.StopUnconfirmedTimeoutError, match="did not finish",
        ):
            daemon_client.run(
                "finishes but the terminal status never comes",
                sock_path=daemon.sock_path,
                timeout=0.3,
                stop_on_timeout=True,
            )
        # The stop was sent (deadline expired before the finish).
        assert daemon.wait_for_command("stop")
    finally:
        daemon.close()


def test_run_agent_tool_reports_unconfirmed_stop_despite_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_agent`` must warn "MAY STILL BE RUNNING" without terminal status.

    End-to-end through the real tool: the daemon wedges after the
    successful result, before the terminal status, so the dispatch
    must surface the never-confirmed-stop error string — releasing the
    workspace with a success report here could hand the workspace to a
    later dispatch while this task still runs finalization.
    """
    monkeypatch.setattr(daemon_client, "_STOP_CONFIRM_GRACE_SECONDS", 1.0)
    daemon = _RacingFinishDaemon(finish_delay=0.6, send_terminal_status=False)
    monkeypatch.setenv("KISS_SORCAR_SOCK", str(daemon.sock_path))
    script = tmp_path / "helper.py"
    script.write_text("def get_model() -> str:\n    return 'm'\n")
    try:
        out = make_run_agent_tool(str(tmp_path))(
            str(script), "finishes without terminal status", timeout="0.3",
        )
        assert "MAY STILL BE RUNNING" in out
        assert "never confirmed" in out
        assert "finished on my own" not in out
    finally:
        daemon.close()


def test_unconfirmed_stop_with_failure_result_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure result does NOT settle a never-confirmed stop wait.

    The stop-killed failure result arrives but the terminal status
    never does: the failure result cannot distinguish "the stop killed
    the task" from a task whose death was never confirmed, so the
    client must still raise ``StopUnconfirmedTimeoutError`` after the
    grace — never return the failure result as if it were the task's
    outcome.
    """
    monkeypatch.setattr(daemon_client, "_STOP_CONFIRM_GRACE_SECONDS", 0.5)
    daemon = _StopKillDaemon(send_terminal_status=False)
    try:
        begin = time.monotonic()
        with pytest.raises(
            daemon_client.StopUnconfirmedTimeoutError, match="did not finish",
        ):
            daemon_client.run(
                "never finishes",
                sock_path=daemon.sock_path,
                timeout=0.3,
                stop_on_timeout=True,
            )
        assert time.monotonic() - begin < 5
    finally:
        daemon.close()
