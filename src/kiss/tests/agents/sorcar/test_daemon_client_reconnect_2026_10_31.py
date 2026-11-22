# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: ``daemon_client.run`` reconnects to the daemon and resumes a running task.

Proposal 6 of ``reports/kiss-sorcar-general-improvements.html`` (last
sentence): a connection drop between the client and the daemon while
the task still runs made the client report the task finished; the
agent went on working and spending.  Now the client reconnects with
backoff (``reconnect_seconds``), re-attaches to its tab, has the
daemon replay the tab's events and goes on waiting.

Everything here is real: a daemon (:class:`LocalDaemonTestCase`) under
an isolated home, a task whose model is the scripted stand-in server
(its second model call blocks until the test releases it, so the task
runs for as long as each scenario needs), and a loopback TCP relay
between the client and the daemon that the test cuts (every live
connection closed) or pauses (new connections refused) mid-task.
"""

from __future__ import annotations

import json
import socket
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import CodeType
from typing import Any

import pytest

from kiss.agents.sorcar import daemon_client, local_endpoint, persistence, sea_commands
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.core.config import kiss_home
from kiss.core.locked import _race_delay
from kiss.server import agent_state, task_runner, task_update
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.agents.sorcar.conftest import reset_sea_registry
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    LocalDaemonTestCase,
    StandInModelServer,
    finish_response,
    history_rows,
    request_text,
    tool_call_response,
    wait_for,
)
from kiss.tests.server.pause import Pause, Pauses
from kiss.tests.server.test_autocommit_wire_toggle import _FakeCredentials

pytestmark = pytest.mark.timeout(300)

#: Every stand-in completion reports 10 prompt + 5 completion tokens.
TOKENS_PER_CALL = 15
PROBE_PROMPT = "reconnect probe"
PARENT_PROMPT = "parent seed task"
SUMMARY = "reconnect probe finished"
#: A SEA whose ``description()`` (its ``/slowhelp help`` answer) waits for a ``release`` file.
SLOW_HELP_SEA = textwrap.dedent(
    """
    import pathlib
    import time

    from kiss.agents.seas.base.base_sea import BaseSea

    _DIR = pathlib.Path(__file__).resolve().parent


    class Sea(BaseSea):
        def description(self):
            (_DIR / "entered").write_text("1", encoding="utf-8")
            deadline = time.time() + 60
            while time.time() < deadline and not (_DIR / "release").exists():
                time.sleep(0.02)
            return "slow help text"
    """
)

# Not reachable over a real daemon, so not covered here: a daemon that
# closes the fresh connection before the ``resumeSession`` is written
# (``_connect_and_resume``'s send failure), a non-JSON frame
# (``_Inbox.recv_event``), a ``replayDone`` of a replay this client did
# not ask for, and a stop whose write fails again on the connection a
# reconnect just opened.  A started task the daemon never named (no
# ``clear`` before its ``running=true``, so ``_resumable`` is false and
# the drop raises at once) is a stand-in daemon's doing: the drop tests
# of test_dispatch_stop_cascade.py and test_cron_connection_lost_keeps_
# scratch_wave2.py cover that branch.


class _TcpRelay:
    """A loopback TCP relay in front of the daemon that the test can cut or pause.

    ``cut`` closes every live connection (the client sees EOF, like a
    daemon that dropped it); ``pause`` closes the listener so new
    connections are refused (like a daemon that is down) until
    ``resume`` binds it again on the same port.
    """

    def __init__(self, daemon_port: int) -> None:
        # The port connections are relayed to; a test that restarts the
        # daemon sets it to the new daemon's.
        self.daemon_port = daemon_port
        self._lock = threading.Lock()
        self._live: list[socket.socket] = []
        self.accepted = 0
        # While set, bytes from the client are read and discarded: the
        # connection stays up but nothing reaches the daemon.
        self.drop_to_daemon = False
        # When positive, a connection's client-to-daemon chunks from
        # this ordinal on are discarded: 3 lets the websocket upgrade
        # request (1) and the ``auth`` frame (2) through and swallows
        # the ``resumeSession`` a reconnect sends next, so the daemon
        # never replays the run on that connection.
        self.swallow_from_chunk = 0
        self.dropped = 0
        # Client-to-daemon websocket DATA frames (``auth``, ``run``,
        # ``resumeSession``, ``stop``) relayed / swallowed: unlike the
        # byte counts, these ignore the HTTP upgrade and the control
        # frames (pongs to the daemon's keep-alive pings, the client's
        # own pings), so a test can wait for a specific command.
        self.forwarded_messages = 0
        self.dropped_messages = 0
        self._listener = socket.create_server(("127.0.0.1", 0))
        self.port = self._listener.getsockname()[1]
        self._serve(self._listener)

    def _serve(self, listener: socket.socket) -> None:
        threading.Thread(
            target=self._accept_loop, args=(listener,), name="relay-accept", daemon=True,
        ).start()

    def _accept_loop(self, listener: socket.socket) -> None:
        while True:
            try:
                client, _ = listener.accept()
            except OSError:
                return  # the listener was closed (pause / close)
            try:
                upstream = socket.create_connection(("127.0.0.1", self.daemon_port))
            except OSError:
                client.close()
                continue
            with self._lock:
                self.accepted += 1
                self._live.extend((client, upstream))
            for src, dst, to_daemon in ((client, upstream, True), (upstream, client, False)):
                threading.Thread(
                    target=self._forward, args=(src, dst, to_daemon), name="relay-forward",
                    daemon=True,
                ).start()

    def _forward(self, src: socket.socket, dst: socket.socket, to_daemon: bool) -> None:
        """Copy bytes from *src* to *dst* until either side ends, then end both."""
        chunks = 0
        frames_in = _DataFrameCounter()
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                chunks += 1
                swallow = self.drop_to_daemon or (
                    self.swallow_from_chunk and chunks >= self.swallow_from_chunk
                )
                # A frame counts with the chunk that completes it, so a
                # frame split by TCP is attributed to the chunk that
                # decided its fate (forwarded or swallowed) last.
                frames = frames_in.feed(data) if to_daemon else 0
                if to_daemon and swallow:
                    with self._lock:
                        self.dropped += len(data)
                        self.dropped_messages += frames
                    continue
                dst.sendall(data)
                if to_daemon:
                    with self._lock:
                        self.forwarded_messages += frames
        except OSError:
            pass
        _shut(src)
        _shut(dst)

    def cut(self) -> None:
        """Close every live connection; the client and the daemon both see EOF."""
        with self._lock:
            live, self._live = self._live, []
        for sock in live:
            _shut(sock)

    def pause(self) -> None:
        """Refuse new connections until :meth:`resume`."""
        # A close alone leaves the thread blocked in ``accept`` holding
        # the kernel socket open (it would keep accepting); the shutdown
        # wakes it so the port is really closed.
        _shut(self._listener)

    def resume(self) -> None:
        """Accept connections again, on the same port."""
        self._listener = socket.create_server(("127.0.0.1", self.port))
        self._serve(self._listener)

    def close(self) -> None:
        """Stop accepting and drop every connection."""
        self.pause()
        self.cut()


def _frame_length(buf: bytes) -> int | None:
    """Return the whole length of the websocket frame *buf* starts with, or ``None`` when
    its header or payload is not all there yet."""
    if len(buf) < 2:
        return None
    length = buf[1] & 0x7F
    header = 2 + (4 if buf[1] & 0x80 else 0)  # the mask key follows the length
    if length == 126:
        if len(buf) < 4:
            return None
        length = int.from_bytes(buf[2:4], "big")
        header += 2
    elif length == 127:
        if len(buf) < 10:
            return None
        length = int.from_bytes(buf[2:10], "big")
        header += 8
    total = header + length
    return total if len(buf) >= total else None


class _DataFrameCounter:
    """Count the websocket data frames (text or binary) of one client-to-daemon byte stream.

    Fed every chunk of the stream in order; a frame counts once its last
    byte has been fed, whatever the TCP segmentation (a header split
    across two reads, two frames in one read).  The stream starts with
    the HTTP upgrade request, which is no frame: it is skipped up to its
    blank line.  Control frames (ping, pong, close) and continuations
    are not counted.
    """

    def __init__(self) -> None:
        self._pending = b""
        self._in_upgrade = True

    def feed(self, data: bytes) -> int:
        """Add *data* to the stream and return the data frames it completed."""
        self._pending += data
        if self._in_upgrade:
            end = self._pending.find(b"\r\n\r\n")
            if end < 0:
                return 0
            self._pending = self._pending[end + 4 :]
            self._in_upgrade = False
        count = 0
        while (total := _frame_length(self._pending)) is not None:
            if self._pending[0] & 0x0F in (0x1, 0x2):
                count += 1
            self._pending = self._pending[total:]
        return count


def _shut(sock: socket.socket) -> None:
    """Shut down and close *sock*, whatever state it is in."""
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()


class ReconnectHarness(LocalDaemonTestCase):
    """A real daemon behind a TCP relay, running a task whose second model call blocks.

    The stand-in model answers the probe task's first call with a quick
    ``Bash`` tool call (so the task has reported spend before the test
    cuts the connection), blocks its second call until ``release`` is
    set, then finishes the task with :data:`SUMMARY`.  A parent seed
    task (:data:`PARENT_PROMPT`, the parent of a sub-agent dispatch)
    finishes at once.
    """

    home_prefix = "kiss-daemon-client-reconnect-"

    def setUp(self) -> None:
        self.credentials = _FakeCredentials()
        self.addCleanup(self.credentials.restore)
        self.release = threading.Event()
        self.blocked = threading.Event()
        # Set by the client at each ``status running=true`` of the run
        # it takes in (:meth:`_start_run`).
        self.seen_running = threading.Event()
        self._calls = 0
        self._calls_lock = threading.Lock()
        self.standin = StandInModelServer(self._respond)
        self.addCleanup(self.standin.stop)
        super().setUp()
        self.home.write_config(max_budget=5.0, use_web_browser=False)
        self.start_daemon()
        # Runs before ``join_tasks``: a task still blocked in its model
        # call is released so the daemon's thread can end.
        self.addCleanup(self.release.set)
        self.relay = _TcpRelay(self.server.port)
        self.addCleanup(self.relay.close)
        self.relay_endpoint = Path(self.tmpdir) / "relay-endpoint.json"
        self._write_relay_endpoint()

    def _write_relay_endpoint(self) -> None:
        """Write the endpoint file the client dials: the relay's port, the daemon's credentials."""
        daemon_endpoint = local_endpoint.read_endpoint(Path(self.endpoint_file))
        assert daemon_endpoint is not None
        local_endpoint.write_endpoint(
            self.relay_endpoint,
            local_endpoint.LocalEndpoint(
                url=f"ws://127.0.0.1:{self.relay.port}/ws",
                token=daemon_endpoint.token,
                ca=daemon_endpoint.ca,
                pid=daemon_endpoint.pid,
            ),
        )

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        if PARENT_PROMPT in request_text(request):
            return finish_response("parent done")
        with self._calls_lock:
            self._calls += 1
            call = self._calls
        if call == 1:
            return tool_call_response("Bash", {"command": "echo probe", "description": "probe"})
        self.blocked.set()
        self.release.wait(timeout=120)
        return finish_response(SUMMARY)

    def _start_run(
        self, prompt: str = PROBE_PROMPT, **kwargs: Any,
    ) -> tuple[threading.Thread, dict[str, Any]]:
        """Run *prompt* through the relay on a thread; its outcome lands in the dict.

        The run's *running* event (the caller's, else one made here) is
        kept as :attr:`seen_running`.
        """
        out: dict[str, Any] = {}
        self.seen_running = kwargs.setdefault("running", threading.Event())

        def target() -> None:
            try:
                out["result"] = daemon_client.run(
                    prompt,
                    work_dir=self.repo,
                    endpoint_file=self.relay_endpoint,
                    model=STANDIN_MODEL,
                    model_config=self.standin.model_config,
                    max_budget=5.0,
                    use_web_tools=False,
                    auto_classify=False,
                    use_memory=False,
                    is_parallel=False,
                    timeout=240.0,
                    **kwargs,
                )
            except BaseException as exc:  # reported by the test
                out["error"] = exc

        worker = threading.Thread(target=target, name="daemon-client-run", daemon=True)
        worker.start()
        return worker, out

    def _await_blocked_task(self) -> None:
        """Wait until the task sits in its blocked second model call and the client knows it."""
        self.assertTrue(self.blocked.wait(timeout=120), "the task never reached its slow step")
        self._await_client()

    def _await_client(self) -> None:
        """Wait until the client has every event the daemon sent its run so far.

        The daemon sent the run's events (its chat and row, the first
        model call's spend) before the run got where the test waited
        for it, but they may still be on their way to the client: on a
        loaded machine a cut then left the client with no spend to
        report, or with no run to resume.  The status repeated here
        marks the stream: once the client has it, it has every event
        before it.  The run's own status (its start) is awaited first,
        so that it is not taken for the mark.
        """
        self.assertTrue(self.seen_running.wait(timeout=60), "the client never saw the run start")
        self.seen_running.clear()
        self.server._printer.broadcast(
            {"type": "status", "running": True, "tabId": self._probe_state().tab_id},
        )
        self.assertTrue(
            self.seen_running.wait(timeout=60), "the client never got the run's first events",
        )

    def _collect(self, worker: threading.Thread, out: dict[str, Any]) -> Any:
        worker.join(timeout=180)
        self.assertFalse(worker.is_alive(), "daemon_client.run did not return")
        if "error" in out:
            return out["error"]
        return out["result"]

    @staticmethod
    def _history_row(task_id: str) -> dict[str, Any]:
        """The task's ``task_history`` row: ``tokens``, ``cost``, ``steps`` and ``result``."""
        row = persistence._load_chat_events_by_task_id(task_id)
        assert row is not None, f"no task_history row {task_id}"
        extra: dict[str, Any] = json.loads(str(row.get("extra") or "{}"))
        events = row.get("events")
        results = [
            ev for ev in (events if isinstance(events, list) else [])
            if isinstance(ev, dict) and ev.get("type") == "result"
        ]
        extra["result"] = results[-1] if results else None
        return extra

    @staticmethod
    def _task_over() -> bool:
        """Whether no registered task of the daemon runs any more."""
        return all(not state.running() for state in agent_state.snapshot())

    def _probe_state(self) -> agent_state.AgentState:
        """The daemon's state of the running probe task (its ``api-…`` tab)."""
        states = [
            s for s in agent_state.snapshot() if s.tab_id.startswith("api-") and s.running()
        ]
        self.assertEqual(len(states), 1, states)
        return states[0]


class ReconnectTest(ReconnectHarness):
    """The client survives a dropped connection and still reports the task's own outcome."""

    def test_cut_mid_task_reconnects_and_returns_the_live_result(self) -> None:
        """A cut while the task runs: the client reconnects, the task finishes live."""
        with self.assertLogs(daemon_client.logger, level="INFO") as logs:
            worker, out = self._start_run(reconnect_seconds=60.0)
            self._await_blocked_task()
            self.relay.cut()
            self.assertTrue(
                wait_for(lambda: self.relay.accepted >= 2, timeout=60),
                "the client never reconnected through the relay",
            )
            self.release.set()
            result = self._collect(worker, out)
        self.assertIsInstance(result, daemon_client.TaskResult, result)
        self.assertTrue(result.success, result)
        self.assertIn(SUMMARY, result.text)
        self.assertEqual(result.reconnects, 1)
        # Two model calls, each reported once: the replayed transcript
        # repeats the first call's usage event without doubling it.
        self.assertEqual(result.tokens, 2 * TOKENS_PER_CALL)
        row = self._history_row(result.task_id)
        self.assertEqual(row["tokens"], result.tokens)
        self.assertEqual(row["steps"], result.steps)
        self.assertAlmostEqual(row["cost"], result.cost, places=9)
        self.assertGreater(result.cost, 0.0)
        self.assertIsNotNone(row["result"])
        self.assertTrue(row["result"].get("success"))
        joined = "\n".join(logs.output)
        self.assertIn("reconnecting (attempt 1)", joined)
        self.assertIn("reconnected to the sorcar daemon on attempt 1", joined)

    def test_result_published_while_away_is_delivered_by_the_replay(self) -> None:
        """The task finishes while the daemon is unreachable: the replay carries its result."""
        worker, out = self._start_run(reconnect_seconds=120.0)
        self._await_blocked_task()
        self.relay.pause()
        self.relay.cut()
        # The task ends with no client attached: its result and
        # terminal status go out while the client keeps retrying.
        self.release.set()
        self.assertTrue(wait_for(self._task_over, timeout=120), "the task did not end")
        self.relay.resume()
        result = self._collect(worker, out)
        self.assertIsInstance(result, daemon_client.TaskResult, result)
        self.assertTrue(result.success, result)
        self.assertIn(SUMMARY, result.text)
        self.assertGreaterEqual(result.reconnects, 1)
        self.assertEqual(result.tokens, 2 * TOKENS_PER_CALL)
        row = self._history_row(result.task_id)
        self.assertEqual(row["tokens"], result.tokens)
        self.assertEqual(row["steps"], result.steps)
        self.assertAlmostEqual(row["cost"], result.cost, places=9)

    def test_a_help_answered_while_away_is_delivered_by_the_replay(self) -> None:
        """``/slowhelp help`` is answered while the daemon is unreachable.

        No agent runs for it: the runner makes the run's row itself,
        under an id that did not name the run, so the client's recovery
        found no row of its run and reported the answered help lost.
        """
        sea = Path(self.tmpdir) / "seas" / "slowhelp" / "slowhelp_sea.py"
        sea.parent.mkdir(parents=True)
        sea.write_text(SLOW_HELP_SEA, encoding="utf-8")
        seas_md = kiss_home() / "SEAS.md"
        seas_md.write_text(f"{sea.parent.parent}\n", encoding="utf-8")
        self.addCleanup(reset_sea_registry)
        self.addCleanup(seas_md.unlink, missing_ok=True)
        sea_commands.refresh_registry()
        worker, out = self._start_run("/slowhelp help", reconnect_seconds=120.0)
        self.assertTrue(
            wait_for((sea.parent / "entered").exists, timeout=60), "description() never ran",
        )
        self._await_client()
        self.relay.pause()
        self.relay.cut()
        (sea.parent / "release").write_text("1", encoding="utf-8")
        self.assertTrue(wait_for(self._task_over, timeout=60), "the help task did not end")
        self.relay.resume()
        result = self._collect(worker, out)
        self.assertIsInstance(result, daemon_client.TaskResult, result)
        self.assertTrue(result.success, result)
        self.assertEqual(result.text, "slow help text")
        self.assertEqual([row["id"] for row in history_rows()], [result.task_id])

    def test_reconnect_seconds_zero_keeps_raising_at_the_first_cut(self) -> None:
        """``reconnect_seconds=0``: the cut raises ``ConnectionLostError`` at once, as before."""
        worker, out = self._start_run(reconnect_seconds=0)
        self._await_blocked_task()
        self.relay.cut()
        error = self._collect(worker, out)
        self.assertIsInstance(error, daemon_client.ConnectionLostError, error)
        self.assertIn("closed the connection before the task finished", str(error))
        self.assertNotIn("reconnect", str(error))
        # The spend seen so far (the first model call) rides on the error.
        spent = error.task_result
        self.assertEqual(spent.tokens, TOKENS_PER_CALL)
        self.assertEqual(spent.reconnects, 0)
        self.assertEqual(self.relay.accepted, 1)

    def test_daemon_down_longer_than_the_window_raises_with_the_spend_so_far(self) -> None:
        """The daemon stays unreachable past ``reconnect_seconds``: ``ConnectionLostError``."""
        worker, out = self._start_run(reconnect_seconds=2.0)
        self._await_blocked_task()
        self.relay.pause()
        started = time.monotonic()
        self.relay.cut()
        error = self._collect(worker, out)
        elapsed = time.monotonic() - started
        self.assertIsInstance(error, daemon_client.ConnectionLostError, error)
        self.assertIn("the run was not resumed within 2s", str(error))
        self.assertGreaterEqual(elapsed, 2.0)
        self.assertLess(elapsed, 30.0)
        spent = error.task_result
        self.assertEqual(spent.tokens, TOKENS_PER_CALL)
        self.assertEqual(spent.reconnects, 0)
        self.assertEqual(self.relay.accepted, 1)

    def test_a_daemon_that_lost_the_task_is_reported_as_a_lost_connection(self) -> None:
        """Back after the drop, the daemon knows neither a running task nor a result."""
        worker, out = self._start_run(reconnect_seconds=60.0)
        self._await_blocked_task()
        state = self._probe_state()
        thread = state.task_thread
        self.relay.pause()
        self.relay.cut()
        # What a daemon restart leaves behind: the task's history row
        # without a result and no live state for it.
        agent_state.unregister(state.task_id, state)
        self.relay.resume()
        error = self._collect(worker, out)
        self.assertIsInstance(error, daemon_client.ConnectionLostError, error)
        self.assertIn("neither running nor finished with a result", str(error))
        self.assertEqual(error.task_result.tokens, TOKENS_PER_CALL)
        self.assertEqual(error.task_result.reconnects, 1)
        # Let the (now unregistered) task end before the daemon is torn down.
        self.release.set()
        if thread is not None:
            thread.join(timeout=120)
            self.assertFalse(thread.is_alive(), "the task thread did not end")

    def test_a_second_drop_while_a_stop_awaits_confirmation_is_a_lost_connection(self) -> None:
        """After a reconnect, a stop was sent and the connection drops again: not resumed."""
        cancel = threading.Event()
        worker, out = self._start_run(reconnect_seconds=60.0, cancel=cancel)
        self._await_blocked_task()
        relayed = self.relay.forwarded_messages
        self.relay.cut()
        # The reconnect is through once its ``auth`` and ``resumeSession``
        # reached the daemon: ``accepted`` alone is the TCP connect, and
        # a swallow switched on before the handshake frames pass would
        # swallow them (and the reconnect) instead of the stop.
        self.assertTrue(
            wait_for(lambda: self.relay.forwarded_messages >= relayed + 2, timeout=60),
            "the client never reconnected through the relay",
        )
        # The stop is written to a connection that swallows it, so the
        # task cannot answer; then that connection drops too.  Only the
        # stop is a data frame here: a pong to a keep-alive ping is not.
        self.relay.drop_to_daemon = True
        cancel.set()
        self.assertTrue(
            wait_for(lambda: self.relay.dropped_messages > 0, timeout=30), "no stop was sent",
        )
        self.relay.cut()
        error = self._collect(worker, out)
        self.assertIsInstance(error, daemon_client.ConnectionLostError, error)
        self.assertIn("closed the connection before the task finished", str(error))
        self.assertIn("(after 1 reconnect(s))", str(error))
        self.assertEqual(error.task_result.tokens, TOKENS_PER_CALL)
        self.assertEqual(error.task_result.reconnects, 1)
        self.assertFalse(self._task_over(), "the swallowed stop must not have reached the task")

    def test_a_cancel_during_the_reconnect_backoff_sends_the_stop_at_once(self) -> None:
        """A kill while the client waits out a reconnect backoff does not wait it out.

        The daemon is down (the relay refuses connections), so the client
        sits in the pause before its next attempt.  The cancel cuts the
        pause short: the stop reaches the daemon, back by then, and is
        confirmed long before the pause would have ended.
        """
        cancel = threading.Event()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(daemon_client, "_RECONNECT_FIRST_BACKOFF_SECONDS", 40.0)
            patch.setattr(daemon_client, "_RECONNECT_MAX_BACKOFF_SECONDS", 40.0)
            with self.assertLogs(daemon_client.logger, level="INFO") as logs:
                worker, out = self._start_run(reconnect_seconds=120.0, cancel=cancel)
                self._await_blocked_task()
                self.relay.pause()
                self.relay.cut()
                self.assertTrue(
                    wait_for(lambda: any("next try in 40.0s" in line for line in logs.output),
                             timeout=30),
                    "the client never entered the reconnect backoff",
                )
                self.relay.resume()
                started = time.monotonic()
                cancel.set()
                error = self._collect(worker, out)
        elapsed = time.monotonic() - started
        self.assertIsInstance(error, daemon_client.CancelledError, error)
        self.assertTrue(error.confirmed, error)
        self.assertLess(elapsed, 30.0, "the cancel waited out the reconnect backoff")
        self.assertEqual(error.result.reconnects, 1)
        self.assertTrue(wait_for(self._task_over, timeout=60), "the task was not stopped")

    def test_a_sub_agent_finishing_while_away_is_resumed_from_its_transcript(self) -> None:
        """A sub-agent dispatch's replay carries no status: its transcript alone ends the wait."""
        parent = daemon_client.run(
            PARENT_PROMPT,
            work_dir=self.repo,
            endpoint_file=self.relay_endpoint,
            model=STANDIN_MODEL,
            model_config=self.standin.model_config,
            max_budget=5.0,
            use_web_tools=False,
            auto_classify=False,
            use_memory=False,
            is_parallel=False,
            timeout=120.0,
        )
        self.assertTrue(parent.success, parent)
        self.assertEqual(parent.reconnects, 0)
        worker, out = self._start_run(
            reconnect_seconds=120.0, parent_task_id=parent.task_id, parent_tab_id="webtab-parent",
        )
        self._await_blocked_task()
        self.relay.pause()
        self.relay.cut()
        self.release.set()
        self.assertTrue(wait_for(self._task_over, timeout=120), "the sub-agent did not end")
        self.relay.resume()
        result = self._collect(worker, out)
        self.assertIsInstance(result, daemon_client.TaskResult, result)
        self.assertTrue(result.success, result)
        self.assertIn(SUMMARY, result.text)
        self.assertGreaterEqual(result.reconnects, 1)
        self.assertEqual(result.tokens, 2 * TOKENS_PER_CALL)
        row = self._history_row(result.task_id)
        self.assertEqual(row["tokens"], result.tokens)
        self.assertEqual(row["subagent"]["parent_task_id"], parent.task_id)

    def test_a_transcript_broadcast_by_another_client_does_not_disturb_the_totals(self) -> None:
        """A history click on the task's tab replays its transcript to every client."""
        worker, out = self._start_run(reconnect_seconds=60.0)
        self._await_blocked_task()
        state = self._probe_state()
        viewer = local_endpoint.connect(Path(self.endpoint_file), max_size=1 << 26, parts=True)
        try:
            resume = {"type": "resumeSession", "chatId": state.chat_id, "tabId": state.tab_id}
            local_endpoint.send(viewer, json.dumps(resume))
            # The unscoped replay reaches the viewer too: once it has the
            # transcript, so has the client under test.
            joiner = local_endpoint.FrameJoiner()
            deadline = time.monotonic() + 60
            while True:
                self.assertLess(time.monotonic(), deadline, "no transcript was broadcast")
                whole = joiner.feed(json.loads(viewer.recv(timeout=60)))
                if whole is None:
                    continue
                frame = whole[0]
                if isinstance(frame, dict) and frame.get("type") == "task_events":
                    self.assertEqual(frame.get("tabId"), state.tab_id)
                    break
        finally:
            viewer.close()
        self.release.set()
        result = self._collect(worker, out)
        self.assertIsInstance(result, daemon_client.TaskResult, result)
        self.assertTrue(result.success, result)
        self.assertEqual(result.reconnects, 0)
        self.assertEqual(result.tokens, 2 * TOKENS_PER_CALL)
        self.assertEqual(self._history_row(result.task_id)["tokens"], result.tokens)

    def _dead_inbox(self) -> daemon_client._Inbox:
        """An inbox whose connection (through the relay) is already closed."""
        inbox = daemon_client._Inbox(
            local_endpoint.connect(self.relay_endpoint, max_size=1 << 20, parts=True),
            self.relay_endpoint,
            "api-stop-probe",
        )
        inbox.ws.close()
        return inbox

    def test_stop_path_reconnects_once_to_send_the_stop(self) -> None:
        """A stop that cannot be written on a dead connection goes out on a fresh one."""
        inbox = self._dead_inbox()
        seen = daemon_client._Observed(chat_id="chat-stop-probe")
        before = time.monotonic()
        deadline = daemon_client._stop_and_await_confirmation(
            inbox, "api-stop-probe", "token", "cancelled", 30.0, seen,
        )
        self.assertGreaterEqual(deadline, before + daemon_client._STOP_CONFIRM_GRACE_SECONDS)
        self.assertEqual(seen.reconnects, 1)
        self.assertEqual(self.relay.accepted, 2)
        # The new connection carries the stop; a second stop needs no reconnect.
        daemon_client._stop_and_await_confirmation(
            inbox, "api-stop-probe", "token", "timed-out", 30.0, seen,
        )
        self.assertEqual(seen.reconnects, 1)
        inbox.ws.close()
        # Without a reconnect window the failed write stays a lost connection.
        with self.assertRaises(daemon_client.ConnectionLostError) as raised:
            daemon_client._stop_and_await_confirmation(
                inbox, "api-stop-probe", "token", "cancelled", 0, seen,
            )
        self.assertIn("stopping the cancelled task", str(raised.exception))
        # So does a run the daemon never named: nothing to re-attach to.
        with self.assertRaises(daemon_client.ConnectionLostError) as raised:
            daemon_client._stop_and_await_confirmation(
                inbox, "api-stop-probe", "token", "cancelled", 30.0, daemon_client._Observed(),
            )
        self.assertIn("stopping the cancelled task", str(raised.exception))
        self.assertEqual(self.relay.accepted, 2)
        # A daemon that stays down makes the stop path give up too.
        down = self._dead_inbox()
        self.relay.pause()
        with self.assertRaises(daemon_client.ConnectionLostError) as raised:
            daemon_client._stop_and_await_confirmation(
                down, "api-stop-probe", "token", "timed-out", 1.0, seen,
            )
        self.assertIn("stopping the timed-out task", str(raised.exception))
        self.assertIn("the run was not resumed within 1s", str(raised.exception))
        self.assertEqual(seen.reconnects, 1)

    def test_a_reconnection_that_dies_before_the_replay_is_one_more_attempt(self) -> None:
        """The window opened by the first drop goes on; the next attempt waits its turn."""
        with self.assertLogs(daemon_client.logger, level="INFO") as logs:
            worker, out = self._start_run(reconnect_seconds=60.0)
            self._await_blocked_task()
            # The reconnect's ``resumeSession`` never reaches the daemon:
            # the new connection is up but the run is not replayed.
            self.relay.swallow_from_chunk = 3
            self.relay.cut()
            self.assertTrue(
                wait_for(lambda: self.relay.accepted >= 2 and self.relay.dropped > 0, timeout=60),
                "the client never reconnected through the relay",
            )
            self.relay.swallow_from_chunk = 0
            paused_at = time.monotonic()
            self.relay.cut()
            self.assertTrue(
                wait_for(lambda: self.relay.accepted >= 3, timeout=60),
                "the client did not reconnect a second time",
            )
            self.assertGreaterEqual(time.monotonic() - paused_at, 1.0)
            self.release.set()
            result = self._collect(worker, out)
        self.assertIsInstance(result, daemon_client.TaskResult, result)
        self.assertTrue(result.success, result)
        self.assertIn(SUMMARY, result.text)
        self.assertEqual(result.reconnects, 2)
        self.assertEqual(result.tokens, 2 * TOKENS_PER_CALL)
        self.assertEqual(self._history_row(result.task_id)["tokens"], result.tokens)
        joined = "\n".join(logs.output)
        self.assertIn(
            "reconnect attempt 1 failed (the re-established connection dropped before the "
            "daemon replayed the run); next try in 1.0s",
            joined,
        )
        self.assertIn("reconnected to the sorcar daemon on attempt 2", joined)

    def test_a_replay_the_daemon_fails_to_read_is_one_more_attempt(self) -> None:
        """The daemon cannot open its history when the client is back: the client asks again.

        A directory stands where ``history.db`` was when the reconnect's
        ``resumeSession`` arrives, so the replay's first read raises.
        The daemon sent no ``replayDone`` then, and the client waited
        out its whole window on a connection the task's events never
        reached.  The failed replay now ends with the failure, which
        the client takes for one more failed attempt of the window.
        """
        db_file = Path(persistence._DB_PATH)
        blocked = db_file.with_name("history.db.blocked")
        blocked.mkdir()
        with self.assertLogs(daemon_client.logger, level="INFO") as logs:
            worker, out = self._start_run(reconnect_seconds=60.0)
            self._await_blocked_task()
            persistence._close_thread_db()
            persistence._DB_PATH = blocked
            try:
                self.relay.cut()
                self.assertTrue(
                    wait_for(lambda: self.relay.accepted >= 3, timeout=30),
                    "the client did not ask for the replay a second time",
                )
            finally:
                persistence._close_thread_db()
                persistence._DB_PATH = db_file
                blocked.rmdir()
            self.release.set()
            result = self._collect(worker, out)
        self.assertIsInstance(result, daemon_client.TaskResult, result)
        self.assertTrue(result.success, result)
        self.assertIn(SUMMARY, result.text)
        self.assertGreaterEqual(result.reconnects, 2)
        self.assertEqual(result.tokens, 2 * TOKENS_PER_CALL)
        self.assertIn("the sorcar daemon could not replay the run", "\n".join(logs.output))

    def test_a_reconnection_the_daemon_never_replays_gives_up_at_the_window_end(
        self,
    ) -> None:
        """A re-established connection that stays silent does not hold the wait past the window.

        The reconnect succeeds but its ``resumeSession`` never reaches
        the daemon, so no replay comes and nothing else drops: the
        client itself must give up when the window opened by the first
        cut is spent (review of 2026-10-31: it waited for the next drop
        instead, 1.5 s and counting on a 0.3 s window).
        """
        worker, out = self._start_run(reconnect_seconds=2.0)
        self._await_blocked_task()
        self.relay.swallow_from_chunk = 3
        cut_at = time.monotonic()
        self.relay.cut()
        self.assertTrue(
            wait_for(lambda: self.relay.accepted >= 2 and self.relay.dropped > 0, timeout=60),
            "the client never reconnected through the relay",
        )
        error = self._collect(worker, out)
        elapsed = time.monotonic() - cut_at
        self.assertIsInstance(error, daemon_client.ConnectionLostError, error)
        self.assertIn("the run was not resumed within 2s (1 attempt(s))", str(error))
        self.assertIn("the re-established connection delivered no replay of the run", str(error))
        self.assertGreaterEqual(elapsed, 2.0)
        self.assertLess(elapsed, 30.0)
        self.assertEqual(error.task_result.tokens, TOKENS_PER_CALL)
        self.assertEqual(error.task_result.reconnects, 1)
        self.assertEqual(self.relay.accepted, 2)
        # The task itself runs on: the error is the client's, not the task's.
        self.assertFalse(self._task_over())

    def test_a_short_window_ends_the_wait_for_the_replay_on_time(self) -> None:
        """The reviewer's numbers: a 0.3 s window, the replay never comes, the wait ends at once.

        The receive is capped by the window's remaining time, so the
        error lands at the deadline and not at the next periodic wake.
        """
        worker, out = self._start_run(reconnect_seconds=0.3)
        self._await_blocked_task()
        self.relay.swallow_from_chunk = 3
        cut_at = time.monotonic()
        self.relay.cut()
        error = self._collect(worker, out)
        elapsed = time.monotonic() - cut_at
        self.assertIsInstance(error, daemon_client.ConnectionLostError, error)
        self.assertIn("the run was not resumed within 0.3s", str(error))
        self.assertGreaterEqual(elapsed, 0.3)
        self.assertLess(elapsed, 5.0)
        self.assertEqual(error.task_result.tokens, TOKENS_PER_CALL)


class SequentialBlocksReconnectTest(ReconnectHarness):
    """A run of two ``<task>`` blocks: the first finishes at once, the second (the probe) blocks."""

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        if PROBE_PROMPT not in request_text(request):
            return finish_response("first block done")
        self.blocked.set()
        self.release.wait(timeout=120)
        return finish_response(SUMMARY)

    def test_every_block_is_charged_once_across_a_late_charge_and_a_reconnect(self) -> None:
        """The run's spend is every block's own, whatever a late charge and a replay repeat.

        The blocks run one after the other, each with its own counters
        and row.  While the second runs, a side call of the first is
        paid: banked on the dispatching parent, which still runs, and
        announced to the waiting client with the first row's new totals.
        The connection then drops, and the replay holds the second
        block's transcript alone.  The client used to keep one task's
        totals and one sum of such charges: the announcement made the
        finished first block the run's task again (the one the reconnect
        resumed, so the wait ended with the first block's result while
        the second ran on), and its charge was subtracted from whichever
        block reported next.
        """
        parent_id, _chat = persistence._add_task("running parent")
        parent = ChatSorcarAgent("reconnect-running-parent")
        with parent._task_id_lock:
            parent._last_task_id = parent_id
        state = agent_state.AgentState(
            parent_id, agent=parent, tab_id="webtab-parent", server_owned=True,
        )
        agent_state.register(state)
        self.addCleanup(agent_state.unregister, parent_id, state)
        running = threading.Event()
        worker, out = self._start_run(
            f"<task>warm up</task><task>{PROBE_PROMPT}</task>",
            reconnect_seconds=60.0, parent_task_id=parent_id, parent_tab_id="webtab-parent",
            running=running,
        )
        self._await_blocked_task()
        probe = self._probe_state()
        first = next(row["id"] for row in history_rows() if row["task"].startswith("warm up"))
        # The status repeated after the charge marks the stream: once the
        # client has it, it has the charge's announcement.
        running.clear()
        task_update.charge_side_channel_usage(
            self.server._printer, None, first, 0.5, 500, 1, tab_id=probe.tab_id, key="side:a",
        )
        self.server._printer.broadcast({"type": "status", "running": True, "tabId": probe.tab_id})
        self.assertTrue(running.wait(timeout=60), "the client never got past the late charge")
        self.relay.cut()
        self.assertTrue(
            wait_for(lambda: self.relay.accepted >= 2, timeout=60),
            "the client never reconnected through the relay",
        )
        self.release.set()
        result = self._collect(worker, out)
        self.assertIsInstance(result, daemon_client.TaskResult, result)
        self.assertIn(SUMMARY, result.text)
        self.assertEqual(result.reconnects, 1)
        self.assertEqual(result.task_id, probe.task_id)
        first_row, second_row = self._history_row(first), self._history_row(result.task_id)
        # The first row holds the late charge, which the parent was
        # charged directly: the client reports the blocks' own spend.
        self.assertEqual(first_row["tokens"], TOKENS_PER_CALL + 500)
        self.assertEqual(parent.usage_snapshot(), (0.5, 500, 1))
        self.assertEqual((result.tokens, result.steps), (2 * TOKENS_PER_CALL, 2))
        self.assertAlmostEqual(
            result.cost, first_row["cost"] - 0.5 + second_row["cost"], places=9,
        )


class _HeldRecovery:
    """Holds the daemon's next recovery replay between its attach and its transcripts.

    ``_replay_session`` calls its ``_race_delay()`` test hook in that
    window; a ``sys.monitoring`` callback on the hook's code blocks the
    first such call until :attr:`go` is set.  Nothing of the daemon is
    replaced.
    """

    _TOOL = 4  # a ``sys.monitoring`` id no debugger, coverage or profiler takes

    def __init__(self) -> None:
        self.reached = threading.Event()
        self.go = threading.Event()
        sys.monitoring.use_tool_id(self._TOOL, "held-recovery")
        sys.monitoring.register_callback(self._TOOL, sys.monitoring.events.PY_START, self._hold)
        sys.monitoring.set_local_events(
            self._TOOL, _race_delay.__code__, sys.monitoring.events.PY_START,
        )

    def _hold(self, _code: CodeType, _offset: int) -> None:
        caller = sys._getframe(1).f_back
        if (
            caller is not None
            and caller.f_code.co_name == "_replay_session"
            and caller.f_locals["recovery"]
            and not self.reached.is_set()
        ):
            self.reached.set()
            self.go.wait(timeout=120)

    def close(self) -> None:
        """Let a held replay go and stop watching the hook."""
        self.go.set()
        sys.monitoring.set_local_events(self._TOOL, _race_delay.__code__, 0)
        sys.monitoring.register_callback(self._TOOL, sys.monitoring.events.PY_START, None)
        sys.monitoring.free_tool_id(self._TOOL)


class BlocksOverWhileAwayTest(ReconnectHarness):
    """A run of ``<task>`` blocks ``block-1`` to ``block-3``, each one model call the test holds.

    Audit 2026-11-21 (S4/F3): each block has a history row of its own,
    and a reconnecting client names the last row it saw.  The daemon
    answered with that row's transcript alone: a block that began and
    ended while the client was away was missing from the run's spend,
    and a run that ended meanwhile returned the result of the block the
    client had seen last.
    """

    def setUp(self) -> None:
        self.entered = {block: threading.Event() for block in (1, 2, 3)}
        self.gates = {block: threading.Event() for block in (1, 2, 3)}
        super().setUp()
        for gate in self.gates.values():
            self.addCleanup(gate.set)

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        text = request_text(request)
        # A later block's request can quote the earlier ones (its chat's context).
        block = next((n for n in (3, 2, 1) if f"block-{n}" in text), None)
        if block is None:
            return finish_response("parent done")
        self.entered[block].set()
        self.gates[block].wait(timeout=120)
        return finish_response(f"result of block {block}")

    def _seed(self) -> daemon_client.TaskResult:
        """Run the seed task (:data:`PARENT_PROMPT`) to its end; return its result."""
        seed = daemon_client.run(
            PARENT_PROMPT, work_dir=self.repo, endpoint_file=self.relay_endpoint,
            model=STANDIN_MODEL, model_config=self.standin.model_config, max_budget=5.0,
            use_web_tools=False, auto_classify=False, use_memory=False, is_parallel=False,
            timeout=120.0,
        )
        self.assertTrue(seed.success, seed)
        return seed

    def _parent(self) -> dict[str, Any]:
        """Run the seed parent task; return the arguments that dispatch a sub-agent of it."""
        return {"parent_task_id": self._seed().task_id, "parent_tab_id": "webtab-parent"}

    def _cut_in_block_one(
        self, blocks: int, **kwargs: Any,
    ) -> tuple[threading.Thread, dict[str, Any]]:
        """Start a run of *blocks* blocks; cut its connection once the client has block 1's row."""
        worker, out = self._start_run(
            "".join(f"<task>block-{n}</task>" for n in range(1, blocks + 1)),
            reconnect_seconds=60.0, **kwargs,
        )
        self.assertTrue(self.entered[1].wait(timeout=120), out)
        # Once the client has the mark, it has the events that named
        # block 1's row.  (The run's own status, come late, was taken
        # for the mark here: the cut then left the client with no row.)
        self._await_client()
        self.relay.pause()
        self.relay.cut()
        return worker, out

    def _reconnect(self) -> None:
        """Let the client back in and wait until it is."""
        accepted = self.relay.accepted
        self.relay.resume()
        self.assertTrue(
            wait_for(lambda: self.relay.accepted > accepted, timeout=60),
            "the client never reconnected through the relay",
        )

    def _check_outcome(
        self, worker: threading.Thread, out: dict[str, Any], blocks: list[int],
    ) -> None:
        """The run returned the last of *blocks*' result and row, with their summed spend."""
        result = self._collect(worker, out)
        self.assertIsInstance(result, daemon_client.TaskResult, result)
        rows = {row["task"]: row["id"] for row in history_rows()}
        spent = [self._history_row(rows[f"block-{n}"]) for n in blocks]
        self.assertTrue(result.success, result)
        self.assertIn(f"result of block {blocks[-1]}", result.text)
        self.assertEqual(result.task_id, rows[f"block-{blocks[-1]}"])
        self.assertEqual(result.reconnects, 1)
        self.assertEqual(
            (result.tokens, result.steps), (len(blocks) * TOKENS_PER_CALL, len(blocks)),
        )
        self.assertAlmostEqual(result.cost, sum(row["cost"] for row in spent), places=9)

    def _block_two_while_away(self, **kwargs: Any) -> None:
        worker, out = self._cut_in_block_one(3, **kwargs)
        self.gates[1].set()
        self.gates[2].set()
        self.assertTrue(self.entered[3].wait(timeout=120), "block 3 never began")
        self._reconnect()
        self.gates[3].set()
        self._check_outcome(worker, out, [1, 2, 3])

    def _over_while_away(self, deleted: bool = False, **kwargs: Any) -> None:
        worker, out = self._cut_in_block_one(2, **kwargs)
        self.gates[1].set()
        self.gates[2].set()
        self.assertTrue(wait_for(self._task_over, timeout=120), "the run did not end")
        if deleted:
            first = next(row["id"] for row in history_rows() if row["task"] == "block-1")
            persistence._delete_task(first)
        self._reconnect()
        self._check_outcome(worker, out, [2] if deleted else [1, 2])

    def _close_tab_and_finish(self) -> None:
        """Close the running run's tab as a user does, then let the run end and be forgotten.

        The ``closeTab`` comes from another connection and stops
        nothing.  Once the run is over the daemon disposes of the closed
        tab's state: only the history rows are left of the run.
        """
        state = self._probe_state()
        viewer = local_endpoint.connect(Path(self.endpoint_file), max_size=1 << 26)
        self.addCleanup(viewer.close)
        local_endpoint.send(viewer, json.dumps({"type": "closeTab", "tabId": state.tab_id}))
        self.assertTrue(wait_for(lambda: state.frontend_closed, timeout=30), "the tab never closed")
        self.assertTrue(state.running(), "closing the tab stopped its run")
        for gate in self.gates.values():
            gate.set()
        self.assertTrue(
            wait_for(lambda: agent_state.find_by_tab(state.tab_id) is None, timeout=120),
            "the closed tab's state outlived its run",
        )

    def _closed_and_over_while_away(self, **kwargs: Any) -> None:
        worker, out = self._cut_in_block_one(3, **kwargs)
        self._close_tab_and_finish()
        self._reconnect()
        self._check_outcome(worker, out, [1, 2, 3])

    def _cut_before_the_first_row(
        self, blocks: int, first_row: Pause, **kwargs: Any,
    ) -> tuple[threading.Thread, dict[str, Any]]:
        """Start a run of *blocks* blocks; cut its connection while its first row is held back.

        The client has the run's chat and no row of it: its reconnect
        names the run by its token.
        """
        running = threading.Event()
        worker, out = self._start_run(
            "".join(f"<task>block-{n}</task>" for n in range(1, blocks + 1)),
            reconnect_seconds=60.0, running=running, **kwargs,
        )
        self.assertTrue(first_row.paused.wait(timeout=60), "the run never allocated a row")
        self.assertTrue(running.wait(timeout=60), "the client never saw the run start")
        self.relay.pause()
        self.relay.cut()
        return worker, out

    def test_a_block_that_ran_while_away_is_charged(self) -> None:
        """Block 2 begins and ends while the client is away: the run's spend holds it."""
        self._block_two_while_away()

    def test_a_sub_agents_block_that_ran_while_away_is_charged(self) -> None:
        """As above for a sub-agent, whose blocks share their parent and no chat lineage."""
        self._block_two_while_away(**self._parent())

    def test_a_run_over_while_away_returns_its_last_block(self) -> None:
        """Both blocks end while the client is away: the result is block 2's, not the one seen."""
        self._over_while_away()

    def test_a_sub_agent_run_over_while_away_returns_its_last_block(self) -> None:
        """As above for a sub-agent, whose replay carries no status at all."""
        self._over_while_away(**self._parent())

    def test_a_block_whose_row_was_deleted_meanwhile_is_left_out(self) -> None:
        """A row of the run deleted from the history while away has nothing left to charge."""
        self._over_while_away(deleted=True)

    def test_a_block_that_begins_during_the_recovery_is_the_runs_last(self) -> None:
        """Block 3 runs between the recovery's attach and its transcripts.

        The daemon read the run's rows (blocks 1 and 2) before block 3
        had one, and block 3's events reach the client ahead of the older
        blocks' transcripts: those must not put block 2's result and row
        in place of block 3's.
        """
        worker, out = self._cut_in_block_one(3)
        self.gates[1].set()
        self.assertTrue(self.entered[2].wait(timeout=120), "block 2 never began")
        held = _HeldRecovery()
        self.addCleanup(held.close)
        self.relay.resume()
        self.assertTrue(held.reached.wait(timeout=60), "the daemon never replayed the run")
        self.gates[2].set()
        self.gates[3].set()
        self.assertTrue(wait_for(self._task_over, timeout=120), "the run did not end")
        held.go.set()
        self._check_outcome(worker, out, [1, 2, 3])

    def test_a_closed_tabs_run_over_while_away_returns_its_last_block(self) -> None:
        """The tab is closed, and blocks 2 and 3 run and end, while the client is away.

        The daemon knew a run's rows from a list it kept for the tab,
        which went with the closed tab's state when the run ended: the
        recovery got block 1's row alone, and the run returned block
        1's result and spend.  The rows are found in the history now.
        """
        self._closed_and_over_while_away()

    def test_a_closed_sub_agent_tabs_run_over_while_away_returns_its_last_block(self) -> None:
        """As above for a sub-agent's run."""
        self._closed_and_over_while_away(**self._parent())

    def test_a_closed_tabs_run_named_by_its_token_returns_its_last_block(self) -> None:
        """As above for a client cut before its run's first row: it names the run by its token."""
        with Pause(persistence._add_task.__code__, lambda _locals: True) as first_row:
            worker, out = self._cut_before_the_first_row(3, first_row)
        self._close_tab_and_finish()
        self._reconnect()
        self._check_outcome(worker, out, [1, 2, 3])

    def test_a_run_whose_rows_all_come_during_the_recovery_returns_its_last_block(self) -> None:
        """The recovery attaches before the run's first row; every block runs before its transcript.

        The daemon read no row of the run, and what it replays for a
        run without rows is no block's transcript: the run's rows are
        read again afterwards, and the run is replayed from them.
        """
        with Pause(persistence._add_task.__code__, lambda _locals: True) as first_row:
            worker, out = self._cut_before_the_first_row(3, first_row)
            held = _HeldRecovery()
            self.addCleanup(held.close)
            self.relay.resume()
            self.assertTrue(held.reached.wait(timeout=60), "the daemon never replayed the run")
        for gate in self.gates.values():
            gate.set()
        self.assertTrue(wait_for(self._task_over, timeout=120), "the run did not end")
        held.go.set()
        self._check_outcome(worker, out, [1, 2, 3])

    def test_a_run_over_before_the_daemon_restarted_returns_its_last_block(self) -> None:
        """The run ends and the daemon restarts while the client is away.

        The new daemon has the history and no memory of the run: the
        rows of the run are all it can answer from.
        """
        worker, out = self._cut_in_block_one(3)
        for gate in self.gates.values():
            gate.set()
        self.assertTrue(wait_for(self._task_over, timeout=120), "the run did not end")
        self.join_tasks()
        self._shutdown_server()
        self.server = RemoteAccessServer(
            local_endpoint_file=self.endpoint_file, work_dir=self.repo,
        )
        self.run_on_loop(self.server.start_private_async(), timeout=30)
        self.relay.daemon_port = self.server.port
        self._write_relay_endpoint()
        self._reconnect()
        self._check_outcome(worker, out, [1, 2, 3])

    def _assert_lost(self, error: Any, tokens: int) -> None:
        """The run ended as a lost connection that reports *tokens* spent."""
        self.assertIsInstance(error, daemon_client.ConnectionLostError, error)
        self.assertIn("neither running nor finished with a result", str(error))
        self.assertEqual(error.task_result.tokens, tokens)

    def _forget_run(self) -> None:
        """Drop the daemon's state of the running run: all a restarted daemon knows of it.

        A restart ends the run's thread with its process and leaves the
        rows the run had written.  A thread cannot be ended here: the
        held run stays as it is (its rows too) while the test looks,
        and is let go and joined before the daemon's home is removed.
        """
        state = self._probe_state()
        thread = state.task_thread
        assert thread is not None
        agent_state.unregister(state.task_id, state)
        self.addCleanup(self._join_forgotten, thread)

    def _join_forgotten(self, thread: threading.Thread) -> None:
        for gate in self.gates.values():
            gate.set()
        thread.join(timeout=120)
        self.assertFalse(thread.is_alive(), "the task thread did not end")

    def test_a_run_the_daemon_lost_in_a_later_block_is_a_lost_connection(self) -> None:
        """Block 1 is over and block 2 has no result when the daemon loses the run.

        The client used to keep the last result it was sent: block 1's,
        replayed with that block's spend, passed for the result of a
        run that never finished.
        """
        worker, out = self._cut_in_block_one(2)
        self.gates[1].set()
        self.assertTrue(self.entered[2].wait(timeout=120), "block 2 never began")
        self._forget_run()
        self._reconnect()
        self._assert_lost(self._collect(worker, out), TOKENS_PER_CALL)

    def _queue_block_two(self) -> None:
        """Queue ``<task>block-2</task>`` behind the running block, as a user's follow-up does."""
        state = self._probe_state()
        sender = local_endpoint.connect(Path(self.endpoint_file), max_size=1 << 26)
        self.addCleanup(sender.close)
        local_endpoint.send(sender, json.dumps(
            {"type": "appendUserMessage", "tabId": state.tab_id, "prompt": "<task>block-2</task>"},
        ))
        self.assertTrue(
            wait_for(lambda: bool(state.queued_followup_tasks), timeout=30),
            "the follow-up was never queued",
        )

    def _lost_after_block_one(self, code: CodeType, queued: bool = False, **kwargs: Any) -> None:
        """Block 1 has returned when the daemon loses the run, held at its next entry of *code*.

        Block 2 is part of the prompt, or (*queued*) a follow-up queued
        while block 1 ran.
        """
        with Pause(code, lambda entered: entered.get("task") != "block-1") as held:
            worker, out = self._cut_in_block_one(1 if queued else 2, **kwargs)
            if queued:
                self._queue_block_two()
            self.gates[1].set()
            self.assertTrue(held.paused.wait(timeout=60), "block 1 never returned")
            persistence._flush_chat_events()
            self._forget_run()
            self._reconnect()
            self._assert_lost(self._collect(worker, out), TOKENS_PER_CALL)

    def test_a_run_lost_between_two_blocks_is_a_lost_connection(self) -> None:
        """Block 1 is over and block 2 has no row yet when the daemon loses the run.

        Block 1's row is the newest the history has of the run, and it
        holds a result: the client returned it as the run's, a success
        of a run whose second block never ran.  The row says the run
        went on from it.
        """
        self._lost_after_block_one(persistence._add_task.__code__)

    def test_a_sub_agent_run_lost_between_two_blocks_is_a_lost_connection(self) -> None:
        """As above for a sub-agent's run."""
        self._lost_after_block_one(persistence._add_task.__code__, **self._parent())

    def test_a_run_lost_before_a_blocks_end_is_recorded_is_a_lost_connection(self) -> None:
        """Block 1 has returned and its end is not in its row yet when the daemon loses the run.

        A block whose row has no end is none the run is known to have
        ended with.
        """
        self._lost_after_block_one(task_runner._TaskRunnerMixin._persist_subtask_row.__code__)

    def test_a_run_lost_before_the_end_of_a_block_a_follow_up_was_queued_behind(self) -> None:
        """As above for a one-block run that a ``<task>`` follow-up was queued behind meanwhile.

        The row's id said whether a block was queued behind it when it
        was allocated: none was then, so block 1's result passed for
        the result of a run whose second block never ran.
        """
        self._lost_after_block_one(
            task_runner._TaskRunnerMixin._persist_subtask_row.__code__, queued=True,
        )

    def test_a_run_rejoined_before_its_last_blocks_end_is_recorded_returns_its_result(self) -> None:
        """The client is back while block 1, the run's only block, has returned and not ended.

        No row says yet that the run ended with block 1, and the run is
        on: its result is the one block 1 recorded, which the frame of
        block 1's unfinished row must not take back.
        """
        ending = Pause(task_runner._TaskRunnerMixin._persist_subtask_row.__code__, lambda _l: True)
        replayed = Pause(daemon_client._Inbox.replay_done.__code__, lambda _locals: True)
        with Pauses(ending, replayed):
            worker, out = self._cut_in_block_one(1)
            self.gates[1].set()
            self.assertTrue(ending.paused.wait(timeout=60), "block 1 never returned")
            persistence._flush_chat_events()
            self._reconnect()
            self.assertTrue(replayed.paused.wait(timeout=60), "the replay never ended")
        self._check_outcome(worker, out, [1])

    def test_a_run_whose_last_row_is_deleted_during_the_recovery_keeps_the_others_spend(
        self,
    ) -> None:
        """Block 3's row is deleted between the recovery's reading of the run's rows and its use.

        The daemon replayed the newest row it had read of, found none,
        and sent nothing of blocks 1 and 2 either: the client reported
        a lost run that had spent nothing.  The run's result went with
        the row; what the other blocks spent is still reported.
        """
        worker, out = self._cut_in_block_one(3)
        self._close_tab_and_finish()
        last = next(row["id"] for row in history_rows() if row["task"] == "block-3")
        held = _HeldRecovery()
        self.addCleanup(held.close)
        self.relay.resume()
        self.assertTrue(held.reached.wait(timeout=60), "the daemon never replayed the run")
        persistence._delete_task(last)
        held.go.set()
        self._assert_lost(self._collect(worker, out), 2 * TOKENS_PER_CALL)

    def test_a_charge_waits_for_the_frame_of_the_row_it_charges(self) -> None:
        """A side channel charges block 1 while the frame of what block 1 spent is on its way.

        The frame was read before the charge and sent after it: the
        client was told an amount the row no longer had, and a charge
        whose own event it got first was undone by the frame.  Reading
        and sending are one step now, which the charge waits for.
        """
        worker, out = self._cut_in_block_one(3)
        for gate in self.gates.values():
            gate.set()
        self.assertTrue(wait_for(self._task_over, timeout=120), "the run did not end")
        first = next(row["id"] for row in history_rows() if row["task"] == "block-1")
        printer = self.server._printer
        charge = threading.Thread(
            target=task_update.charge_side_channel_usage,
            args=(printer, None, first, 0.5, 500, 1),
            kwargs={"key": "late"},
            daemon=True,
        )
        frame = Pause(
            type(printer).broadcast.__code__,
            lambda entered: entered["event"].get("type") == "task_events"
            and entered["event"].get("task_id") == first,
        )
        with frame:
            self.relay.resume()
            self.assertTrue(frame.paused.wait(timeout=60), "the daemon never replayed block 1")
            charge.start()
            charge.join(timeout=1)
            self.assertEqual(
                self._history_row(first)["tokens"], TOKENS_PER_CALL,
                "the row was charged while its frame was on its way",
            )
        charge.join(timeout=60)
        self.assertEqual(self._history_row(first)["tokens"], TOKENS_PER_CALL + 500)
        result = self._collect(worker, out)
        self.assertIsInstance(result, daemon_client.TaskResult, result)
        self.assertIn("result of block 3", result.text)

    def test_a_charge_of_a_block_over_since_the_attach_is_not_undone_by_its_recording(self) -> None:
        """Block 1 ends and is charged between the recovery's attach and its transcripts.

        The recovery replayed what block 1 had recorded when it was
        attached to, after the frame of block 1's row: totals the row
        no longer had, which the charge was lost to for good (block 2
        has no row yet, so nothing is replayed again).  The row of a
        block that is over is sent after what the block recorded.
        """
        next_row = Pause(
            persistence._add_task.__code__, lambda entered: entered.get("task") == "block-2",
        )
        replayed = Pause(daemon_client._Inbox.replay_done.__code__, lambda _locals: True)
        with Pauses(next_row, replayed):
            worker, out = self._cut_in_block_one(2)
            first = self._probe_state().task_id
            held = _HeldRecovery()
            self.addCleanup(held.close)
            self.relay.resume()
            self.assertTrue(held.reached.wait(timeout=60), "the daemon never replayed the run")
            self.gates[1].set()
            self.assertTrue(next_row.paused.wait(timeout=60), "block 1 never ended")
            task_update.charge_side_channel_usage(
                self.server._printer, None, first, 0.5, 500, 1, key="late",
            )
            self.assertEqual(self._history_row(first)["tokens"], TOKENS_PER_CALL + 500)
            held.go.set()
            self.assertTrue(replayed.paused.wait(timeout=60), "the replay never ended")
        self.gates[2].set()
        result = self._collect(worker, out)
        self.assertIsInstance(result, daemon_client.TaskResult, result)
        self.assertIn("result of block 2", result.text)
        self.assertEqual((result.tokens, result.steps), (2 * TOKENS_PER_CALL + 500, 3))

    def test_a_run_lost_before_its_first_row_gets_no_other_tasks_result(self) -> None:
        """The daemon loses a run that has no row yet, in a chat whose last task has a result.

        The client names the run by its token.  The daemon answered a
        name it knew nothing of with the chat's latest row: the earlier
        task's transcript, whose result the client returned as its own.
        """
        chat_id = self._seed().chat_id
        with Pause(persistence._add_task.__code__, lambda _locals: True) as first_row:
            worker, out = self._cut_before_the_first_row(1, first_row, chat_id=chat_id)
            self._forget_run()
            self._reconnect()
            self._assert_lost(self._collect(worker, out), 0)
