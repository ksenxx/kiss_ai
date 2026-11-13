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
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import daemon_client, local_endpoint, persistence
from kiss.server import agent_state
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    LocalDaemonTestCase,
    StandInModelServer,
    finish_response,
    request_text,
    tool_call_response,
    wait_for,
)
from kiss.tests.server.test_autocommit_wire_toggle import _FakeCredentials

pytestmark = pytest.mark.timeout(300)

#: Every stand-in completion reports 10 prompt + 5 completion tokens.
TOKENS_PER_CALL = 15
PROBE_PROMPT = "reconnect probe"
PARENT_PROMPT = "parent seed task"
SUMMARY = "reconnect probe finished"

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
        self._daemon_port = daemon_port
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
                upstream = socket.create_connection(("127.0.0.1", self._daemon_port))
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
        daemon_endpoint = local_endpoint.read_endpoint(Path(self.endpoint_file))
        assert daemon_endpoint is not None
        self.relay_endpoint = Path(self.tmpdir) / "relay-endpoint.json"
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

    def _start_run(self, **kwargs: Any) -> tuple[threading.Thread, dict[str, Any]]:
        """Run the probe task through the relay on a thread; its outcome lands in the dict."""
        out: dict[str, Any] = {}

        def target() -> None:
            try:
                out["result"] = daemon_client.run(
                    PROBE_PROMPT,
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
        """Wait until the task sits in its blocked second model call (one call's spend reported)."""
        self.assertTrue(self.blocked.wait(timeout=120), "the task never reached its slow step")

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
