# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``run_agent`` dispatch ``timeout``.

The ``run_agent`` tool has a ``timeout`` parameter (a number string;
empty applies the SEA's ``timeout`` setting, else
``agent_dispatch.DEFAULT_DISPATCH_TIMEOUT_SECONDS``, 3600 s).  It
bounds the CALL, not the sub-task: every dispatch runs as an agent job
on its own thread, and when the bound expires the tool returns the
job's still-running notice while the sub-task keeps running (no
``stop`` is sent); ``agent_job(id, "kill")`` is what stops it, through
the job's cancel event, and the kill blocks until the daemon's terminal
status confirms the task is dead (bounded by
``_STOP_CONFIRM_GRACE_SECONDS`` against a wedged daemon, after which
the text says the task MAY STILL BE RUNNING instead of claiming a
stop).  ``daemon_client.run`` keeps its own contract: on a plain
timeout it sends only ``closeTab``, never ``stop`` (the caller chose
to stop waiting, not to cancel the work), ``stop_on_timeout`` opts in
to a stop, and ``timeout=None`` ("no deadline": the event read wakes
every ``_NO_DEADLINE_WAKE_SECONDS`` and retries, so an injected abort —
the ``KeyboardInterrupt`` of a parent Stop — still gets delivered).

These tests run real local-WSS daemon stand-ins (served by
:func:`kiss.tests.local_ws.fake_daemon`) and drive the real client code:

* The ``run_agent`` tool (path mode, standard endpoint resolution via
  ``KISS_SORCAR_LOCAL``) returns the delayed sub-task's YAML result,
  both with the default timeout and with an explicit one.  The delay
  is seconds, so this cannot behaviorally pin the default at exactly
  3600 s — only an hour-long test could; the shrunk-constant timeout
  test below is the practical guard for the default wiring, and a
  script-declared ``timeout`` setting is exercised at its real value.
* A too-small ``timeout`` (explicit, or the shrunk default) yields the
  still-running job notice naming the bound and no ``stop``; killing
  the job sends ``stop`` + ``closeTab`` and charges the stopped task's
  spend to the caller.
* Invalid ``timeout`` strings are rejected before any dispatch.

Not covered here, and why: the stop-SEND failure branch (a broken
daemon connection at the exact moment the stop is written raises
``ConnectionError`` instead of a ``TimeoutError`` that would falsely
claim "was stopped") cannot be staged end-to-end without test
doubles — closing a WebSocket peer makes the client's blocking
``recv`` raise ``ConnectionClosed`` first, taking the ordinary
connection-loss path before any stop is attempted.
* ``daemon_client.run(timeout=None)`` returns a result delivered only
  after a delay, without raising ``TimeoutError``.
* A parent stopped (injected ``KeyboardInterrupt``) while blocked with
  ``timeout=None`` on a SILENT daemon still aborts at the next wake
  and cascades a ``stop`` to the dispatched task.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from kiss.agents.sorcar import agent_dispatch, cron_agent, daemon_client
from kiss.agents.sorcar.agent_dispatch import make_run_agent_tool
from kiss.server.task_runner import inject_keyboard_interrupt
from kiss.tests.agents.sorcar.test_dispatch_stop_cascade import (
    _RecordingDaemon,
)
from kiss.tests.local_ws import fake_daemon

_HELPER_SEA = """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'model': 'm'}
"""


@pytest.fixture(autouse=True)
def _standalone_daemon_endpoint(monkeypatch: pytest.MonkeyPatch):
    """Isolate each test from a recorded in-process daemon endpoint.

    ``_dispatch`` prefers the endpoint file the cron scheduler recorded
    at daemon boot (``cron_agent._daemon_endpoint_file``) over the
    standard ``KISS_SORCAR_LOCAL`` resolution; a value left behind by
    another test module would divert the dispatch away from this
    module's fake daemons.
    """
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    yield
    # A detached job a test left running would outlive its daemon stand-in.
    agent_dispatch.kill_jobs_of(None)


def _detached_job(out: str) -> agent_dispatch.AgentJob:
    """Return the still-running job the tool text *out* names (standalone owner)."""
    job = agent_dispatch.agent_jobs_of(None).get(agent_dispatch.notice_job_id(out))
    assert job is not None, out
    return job


async def _send_event(ws: ServerConnection, event: dict[str, Any]) -> bool:
    """Send one JSON event frame; False once the client has gone away."""
    try:
        await ws.send(json.dumps(event))
        return True
    except ConnectionClosed:
        return False


class _LocalDaemon:
    """Base for local-WSS daemon stand-ins served on a loop of their own thread.

    :func:`fake_daemon` completes the ``auth`` handshake, then
    :meth:`_handle` runs with the authenticated connection.  The
    synchronous ``daemon_client.run`` under test blocks the test
    thread, so the daemon lives on its own event loop.
    """

    def __init__(self, prefix: str) -> None:
        """Start serving; returns once the endpoint file is written."""
        self._dir = Path(tempfile.mkdtemp(prefix=prefix))
        self.endpoint_file = self._dir / "sorcar-local.json"
        self.run_cmd: dict[str, Any] | None = None
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._closed = asyncio.Event()
        self._thread = threading.Thread(
            target=self._loop.run_until_complete, args=(self._serve(),), daemon=True,
        )
        self._thread.start()
        assert self._ready.wait(10), "the daemon stand-in never came up"

    async def _serve(self) -> None:
        async with fake_daemon(self._dir, self._handle, endpoint_file=self.endpoint_file):
            self._ready.set()
            await self._closed.wait()

    async def _handle(self, ws: ServerConnection) -> None:
        raise NotImplementedError

    async def _read_run(self, ws: ServerConnection) -> str | None:
        """Record the client's initial ``run`` command; return its tab id."""
        try:
            run_cmd: dict[str, Any] = json.loads(await ws.recv())
        except (ConnectionClosed, json.JSONDecodeError, UnicodeDecodeError):
            return None
        self.run_cmd = run_cmd
        return str(run_cmd.get("tabId", ""))

    def close(self) -> None:
        """Stop serving, close every connection and remove the temp dir."""
        self._loop.call_soon_threadsafe(self._closed.set)
        self._thread.join(timeout=10)
        self._loop.close()
        shutil.rmtree(self._dir, ignore_errors=True)


class _SlowFinishDaemon(_LocalDaemon):
    """A daemon stand-in whose result arrives after a delay.

    Reads the initial ``run`` command, sends ``{"type": "status",
    "running": true}``, sleeps *delay* seconds, and only then sends the
    terminal ``result`` + ``status running=false`` pair — the shape of
    a long-running task on a live daemon.
    """

    def __init__(self, delay: float) -> None:
        self.delay = delay
        super().__init__("kiss_no_timeout_")

    async def _handle(self, ws: ServerConnection) -> None:
        tab_id = await self._read_run(ws)
        if tab_id is None:
            return
        await _send_event(ws, {"type": "status", "running": True, "tabId": tab_id})
        await asyncio.sleep(self.delay)
        await _send_event(ws, {
            "type": "result",
            "tabId": tab_id,
            "taskId": "task-slow-1",
            "success": True,
            "text": "slow but done",
            "cost": "$0.0100",
            "total_tokens": 5,
            "step_count": 1,
        })
        await _send_event(ws, {"type": "status", "running": False, "tabId": tab_id})
        # Absorb the client's closeTab before dropping the connection.
        try:
            await ws.recv()
        except ConnectionClosed:
            pass


class _OversizeFrameDaemon(_LocalDaemon):
    """A daemon stand-in that answers the ``run`` with one oversized frame.

    Sends *payload* as a single text frame (never reading the client's
    ``run`` command) and keeps the connection open until closed,
    exercising the client's ``max_size`` cap on an arbitrary frame.
    """

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        super().__init__("kiss_no_timeout_")

    async def _handle(self, ws: ServerConnection) -> None:
        try:
            await ws.send(self._payload.decode("ascii"))
        except ConnectionClosed:
            return
        await self._closed.wait()


class _StopConfirmingDaemon(_LocalDaemon):
    """A daemon stand-in that confirms a ``stop`` with terminal status.

    Reads the ``run`` command, sends ``status running=true`` (unless
    ``initial_running`` is false, modelling a task stopped during setup
    before that broadcast), then never finishes the task: it only
    records the client's further commands and, on receiving ``stop``,
    replies — after ``confirm_delay`` seconds — with ``status
    running=false``, the shape of a live daemon stopping a task.
    """

    def __init__(
        self,
        confirm_delay: float = 0.0,
        initial_running: bool = True,
        stopped_result: dict[str, Any] | None = None,
    ) -> None:
        """Start serving.

        *stopped_result*, when given, is sent as the stopped task's
        ``result`` event just before the terminal status, like the
        daemon's failure result that carries the spend so far.
        """
        self.confirm_delay = confirm_delay
        self.initial_running = initial_running
        self.stopped_result = stopped_result
        self.commands: list[dict[str, Any]] = []
        super().__init__("kiss_dispatch_timeout_")

    async def _handle(self, ws: ServerConnection) -> None:
        tab_id = await self._read_run(ws)
        if tab_id is None:
            return
        if self.initial_running:
            await _send_event(ws, {"type": "status", "running": True, "tabId": tab_id})
        try:
            async for message in ws:
                try:
                    cmd = json.loads(message)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                self.commands.append(cmd)
                if cmd.get("type") == "stop":
                    await asyncio.sleep(self.confirm_delay)
                    if self.stopped_result is not None:
                        await _send_event(ws, {**self.stopped_result, "tabId": tab_id})
                    await _send_event(ws, {
                        "type": "status", "running": False, "tabId": tab_id,
                    })
        except ConnectionClosed:
            return

    def wait_for_command(self, cmd_type: str, timeout: float = 5.0) -> bool:
        """Poll until a *cmd_type* command was recorded (or timeout)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(c.get("type") == cmd_type for c in self.commands):
                return True
            time.sleep(0.02)
        return False


def test_oversize_frame_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A frame over the client cap fails loudly.

    The client opens its connection with ``max_size=_MAX_LINE_BYTES``;
    ``websockets`` then closes the connection with code 1009 (message
    too big) when the daemon sends a larger frame, and the client must
    surface that as the "frame larger" ``ConnectionError`` rather than
    the generic "closed the connection" one.
    """
    monkeypatch.setattr(daemon_client, "_MAX_LINE_BYTES", 1024)
    daemon = _OversizeFrameDaemon(b"a" * 1500)
    try:
        with pytest.raises(ConnectionError, match="frame larger"):
            daemon_client.run(
                "oversize probe", endpoint_file=daemon.endpoint_file, timeout=30.0,
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
            "slow child task", endpoint_file=daemon.endpoint_file, timeout=None,
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
    ``KISS_SORCAR_LOCAL`` endpoint resolution: the sub-task's result
    arrives after a delay well under the timeout (the 3600-s default,
    and an explicit ``"30"``), and the tool returns the YAML result —
    not a "did not finish within …s" timeout message.
    """
    daemon = _SlowFinishDaemon(delay=1.5)
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    script = tmp_path / "slow_helper.py"
    script.write_text(_HELPER_SEA)
    try:
        out = make_run_agent_tool(str(tmp_path))(
            "say hi slowly", str(script), timeout=timeout_arg,
        )
        parsed = yaml.safe_load(out)
        assert parsed.pop("ran").startswith("sub-agent model=default ")
        assert parsed == {"success": True, "summary": "slow but done"}
        assert "did not finish within" not in out
        assert daemon.run_cmd is not None
        assert daemon.run_cmd.get("seaPath") == str(script)
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
    daemon = _RecordingDaemon(mode="silent")
    outcome: dict[str, Any] = {}

    def call() -> None:
        try:
            daemon_client.run(
                "silent child task", endpoint_file=daemon.endpoint_file, timeout=None,
            )
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


def test_run_agent_tool_timeout_detaches_the_task_into_a_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A too-small explicit ``timeout`` hands back a running job, not a stop.

    End-to-end through the real tool (path mode): the daemon stand-in
    never finishes the task, so after the 0.5-s bound the tool returns
    the job notice naming the bound and the job id; the sub-task is
    still running and no ``stop`` was sent.  ``agent_job(id, "kill")``
    is what stops it: ``stop`` then ``closeTab``, the stop's
    terminal-status confirmation awaited, and the stop error returned.
    """
    daemon = _StopConfirmingDaemon()
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    script = tmp_path / "helper.py"
    script.write_text(_HELPER_SEA)
    try:
        out = make_run_agent_tool(str(tmp_path))(
            "never finishes", str(script), timeout="0.5",
        )
        assert "is still running after 0.5s as job agent-" in out, out
        assert "did not finish" not in out and "was stopped" not in out
        job = _detached_job(out)
        assert job.thread.is_alive()
        assert not any(c.get("type") == "stop" for c in daemon.commands), (
            "the expired bound stopped the sub-task"
        )
        killed = agent_dispatch.make_agent_job_tool()(job.job_id, "kill")
        assert "was stopped before it finished" in killed, killed
        assert not job.thread.is_alive()
        assert daemon.wait_for_command("stop")
        assert daemon.wait_for_command("closeTab")
        # The killed job stays readable until its owner's run ends.
        assert agent_dispatch.make_agent_job_tool()(job.job_id, "tail") == killed
    finally:
        daemon.close()


def test_killing_a_detached_job_charges_the_stopped_tasks_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A killed sub-task's spend is charged to the caller.

    The daemon's failure result for the stopped task carries what it
    spent before the stop; the dispatch must fold that into the calling
    agent (as it does for a finished sub-task) and say so, instead of
    dropping the spend with the stop.
    """
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent

    daemon = _StopConfirmingDaemon(stopped_result={
        "type": "result", "taskId": "task-stopped-1", "success": False,
        "text": "Task stopped", "cost": "$1.0842", "total_tokens": 4321,
        "step_count": 7,
    })
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    script = tmp_path / "helper.py"
    script.write_text(_HELPER_SEA)
    parent = SorcarAgent("dispatch-timeout-parent")
    try:
        out = make_run_agent_tool(str(tmp_path), parent_agent=parent)(
            "never finishes", str(script), timeout="0.5",
        )
        assert "is still running after 0.5s" in out, out
        assert parent.budget_used == 0.0
        job_id = agent_dispatch.notice_job_id(out)
        assert job_id in agent_dispatch.agent_jobs_of(parent)
        killed = agent_dispatch.make_agent_job_tool(parent)(job_id, "kill")
    finally:
        agent_dispatch.kill_jobs_of(parent)
        daemon.close()
    assert "was stopped before it finished" in killed
    assert "$1.0842 spend is counted" in killed
    assert parent.budget_used == pytest.approx(1.0842)
    assert parent.total_tokens_used == 4321
    assert parent.total_steps == 7


def test_run_stop_on_timeout_error_carries_the_stopped_result() -> None:
    """``StoppedOnTimeoutError.result`` parses the stopped task's result.

    Without a ``result`` event before the terminal status the carried
    result reports no spend.
    """
    with_result = _StopConfirmingDaemon(stopped_result={
        "type": "result", "taskId": "task-stopped-2", "success": False,
        "text": "Task stopped", "cost": "$0.2500", "total_tokens": 99,
        "step_count": 2,
    })
    without_result = _StopConfirmingDaemon()
    try:
        with pytest.raises(daemon_client.StoppedOnTimeoutError) as err:
            daemon_client.run(
                "never finishes", timeout=0.3, stop_on_timeout=True,
                endpoint_file=with_result.endpoint_file,
            )
        assert err.value.result.cost == pytest.approx(0.25)
        assert err.value.result.tokens == 99
        assert err.value.result.steps == 2
        assert err.value.result.task_id == "task-stopped-2"
        with pytest.raises(daemon_client.StoppedOnTimeoutError) as err2:
            daemon_client.run(
                "never finishes", timeout=0.3, stop_on_timeout=True,
                endpoint_file=without_result.endpoint_file,
            )
        assert err2.value.result.cost == 0.0
        assert err2.value.result.tokens == 0
    finally:
        with_result.close()
        without_result.close()


def test_run_agent_tool_reports_unconfirmed_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A never-confirmed kill yields the "may still be running" error.

    End-to-end through the real tool (path mode) against a silent
    daemon stand-in: the ``stop`` the kill sends is never answered, so
    once the (shrunk) confirmation grace expires the kill must NOT
    claim the task "was stopped" — it must say the task may still be
    running so the caller does not assume the work was cancelled.
    """
    monkeypatch.setattr(daemon_client, "_STOP_CONFIRM_GRACE_SECONDS", 0.3)
    daemon = _RecordingDaemon(mode="silent")
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    script = tmp_path / "helper.py"
    script.write_text(_HELPER_SEA)
    try:
        out = make_run_agent_tool(str(tmp_path))(
            "never finishes", str(script), timeout="0.5",
        )
        job = _detached_job(out)
        killed = agent_dispatch.make_agent_job_tool()(job.job_id, "kill")
        assert killed == agent_dispatch.unconfirmed_stop_error("helper"), killed
        assert "MAY STILL BE RUNNING" in killed
        assert "was stopped" not in killed
        assert daemon.wait_for_command("stop")
        assert daemon.wait_for_command("closeTab")
    finally:
        daemon.close()


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
                "never finishes", endpoint_file=daemon.endpoint_file, timeout=0.3,
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
    daemon = _RecordingDaemon(mode="silent")
    try:
        begin = time.monotonic()
        with pytest.raises(
            daemon_client.StopUnconfirmedTimeoutError, match="did not finish",
        ):
            daemon_client.run(
                "never finishes", endpoint_file=daemon.endpoint_file, timeout=0.3,
                stop_on_timeout=True,
            )
        assert time.monotonic() - begin < 5
        assert daemon.wait_for_command("stop")
        assert daemon.wait_for_command("closeTab")
    finally:
        daemon.close()


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
                "never finishes", endpoint_file=daemon.endpoint_file, timeout=0.3,
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
    """An empty ``timeout`` falls back to the 3600-s default constant.

    Waiting out the real one-hour default is out of the question, so
    ``DEFAULT_DISPATCH_TIMEOUT_SECONDS`` (asserted to be 3600 in
    production) is shrunk to 0.3 s and the tool is called WITHOUT a
    timeout argument against a silent daemon: the job notice naming
    0.3 s proves the empty-string path reads the constant.
    """
    assert agent_dispatch.DEFAULT_DISPATCH_TIMEOUT_SECONDS == 3600.0
    monkeypatch.setattr(
        agent_dispatch, "DEFAULT_DISPATCH_TIMEOUT_SECONDS", 0.3,
    )
    daemon = _StopConfirmingDaemon()
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    script = tmp_path / "helper.py"
    script.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'model': 'm'}
""")
    try:
        out = make_run_agent_tool(str(tmp_path))("never finishes", str(script))
        assert "is still running after 0.3s as job agent-" in out, out
        agent_dispatch.kill_agent_job(_detached_job(out))
    finally:
        daemon.close()


def test_empty_timeout_takes_the_script_timeout_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty ``timeout`` takes ``settings()["timeout"]`` before the default.

    The script declares a 0.3-s timeout while the default constant is
    left at its production value: the job notice naming 0.3 s proves
    the SEA's setting is read.  An explicit argument still wins over it.
    """
    daemon = _StopConfirmingDaemon()
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    script = tmp_path / "helper.py"
    script.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'timeout': 0.3}
""")
    try:
        out = make_run_agent_tool(str(tmp_path))("never finishes", str(script))
        assert "is still running after 0.3s as job agent-" in out, out
        agent_dispatch.kill_agent_job(_detached_job(out))
    finally:
        daemon.close()
    daemon = _StopConfirmingDaemon()
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    try:
        out = make_run_agent_tool(str(tmp_path))(
            "never finishes", str(script), timeout="0.5",
        )
        assert "is still running after 0.5s as job agent-" in out, out
        agent_dispatch.kill_agent_job(_detached_job(out))
    finally:
        daemon.close()


@pytest.mark.parametrize("bad", ["abc", "1.5s", "0", "-5", "inf", "nan"])
def test_invalid_timeout_rejected_before_dispatch(
    bad: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed or non-positive ``timeout`` is rejected before any dispatch.

    The timeout is validated after the agent is resolved (a script's
    own ``timeout`` setting is the fallback for an empty argument), so
    the agent here is a real script — one that even declares a valid
    ``timeout`` of its own — and a daemon stand-in is listening: the
    error must come from the argument validation, and the daemon must
    never see a ``run`` command.
    """
    script = tmp_path / "helper.py"
    script.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'timeout': 5}
""")
    daemon = _StopConfirmingDaemon()
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    try:
        out = make_run_agent_tool(str(tmp_path))("task", str(script), timeout=bad)
        assert out.startswith("Error: timeout must be")
        assert repr(bad) in out
        assert daemon.run_cmd is None
    finally:
        daemon.close()
