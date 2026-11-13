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
* The request thread of a reply that trickles past the deadline ends by
  itself (its read obeys the same deadline), so a bridge that never
  completes a body does not keep a reader thread and its socket alive
  after the call returned (audit 2026-11-07, A08 astra-8 / L4 F3).
  Its read deadline is a hair after the wait's, and the wait polls
  in 0.2 s slices: when the read gives up first, its failure is still
  the timeout (one kill, ``(None, "")``), not an "unreachable" error.
  The cancel event of the test holds the wait at its poll until just
  before the request's deadline, so the slice that follows is the one
  in which the read gives up: that order is the one tested.
* A Stop wins over a reply that completes as it lands, and a Stop set
  before the call never posts the command (A08 astra-9 / fable-4).
* A reply whose ``exit_code`` is no integer is an invalid reply that
  kills the command before the error propagates, and a ``/ping`` body
  that is no JSON object, or never completes, means the bridge is not
  alive (A08 astra-12 / fable-8, astra-7 / fable-3).
* ``run_commands_parallel`` reports a command the bridge could not run
  as ``exit -1`` with an ``Error:`` output, as the Docker tool does
  (A08 claude-6).
* The request's deadline bounds the whole request, not only the body's
  pieces: the status line and headers (read inside ``urlopen``) and a
  chunked body's size lines (read inside ``read1``) are as many socket
  reads as the peer makes of them, each within the idle timeout, so a
  peer trickling one byte at a time held the request, its thread and
  its socket for good (audit 2026-11-08, S4 F2).  The socket is shut
  down at the deadline instead, wherever the request is.
* The finish gate of a bridged task runs its commands through
  ``ExecBridge.run``, the raw primitive ``DockerManager.run`` is for a
  container: the bridge had none, so every gate command raised
  ``AttributeError`` (audit 2026-11-10, S4 F1).
* A bridge URL that names no endpoint (a port that is no number or out
  of range, an unclosed IPv6 bracket) and a reply nested too deep for
  the JSON decoder are an unreachable bridge like any other: the URL
  was parsed outside the request's error handling and ``RecursionError``
  was not among the reply errors, so ``bridge_alive`` raised a raw
  ``ValueError``/``RecursionError`` and ``bridge_exec``, whose request
  thread died of it, an ``IndexError`` (audit 2026-11-11, S9 F1).
* A bridge URL with an IPv6 host and no port (``http://[::1]``) dials
  that host on HTTP's default port: ``http.client`` was handed the
  bracket-less host with no port and split it again at its last colon,
  host ``:`` on port 1 (audit 2026-11-11, S9 re-fix).
* A Stop raised at the back edge of ``bridge_exec``'s wait loop kills
  the command like one raised anywhere else in the wait.  On CPython
  3.13 the back edge of its ``while worker.is_alive():`` loop was
  outside the exception table of the ``try`` around it, so the handler
  that posts the kill was skipped there.  The interpreter raises a
  pending Stop at that instruction only when the thread switch happens
  to fall on it, so the test raises there from an opcode trace event
  instead (audit 2026-11-11, S9 re-fix).
* Out of threads, a bridge call fails as an unreachable bridge does.
  The deadline timer's ``RuntimeError`` escaped ``_request``, so
  ``bridge_alive`` raised it and ``bridge_exec``, whose request thread
  died of it, an ``IndexError``; the kill it then could not post raised
  in its place, as did a request thread that could not start (audit
  2026-11-12, B4).
"""

from __future__ import annotations

import dis
import inspect
import json
import os
import select
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import FrameType
from typing import Any, cast

import pytest

from kiss.agents.sorcar.docker_manager import _KILL_WAIT_S, MAX_OUTPUT_CHARS, STOPPED_MESSAGE
from kiss.agents.sorcar.exec_bridge import (
    _REPLY_MARGIN_S,
    ExecBridge,
    _request,
    bridge_alive,
    bridge_exec,
)
from kiss.agents.sorcar.shell_guards import DESTRUCTIVE_VERDICT
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.benchmarks.terminal_bench.exec_bridge_server import ExecBridgeServer
from kiss.core.kiss_error import KISSError
from kiss.tests.agents.seas.test_coding_sea_evidence_gate import LocalEnvironment, _close, _Loop
from kiss.tests.conftest import nproc_limit_lowered_to_one, thread_start_can_be_starved
from kiss.tests.server.pause import Pause

#: A JSON reply nested deeper than the decoder recurses.
_DEEP_JSON = b"[" * 100_000 + b"]" * 100_000


class _Bridge(ThreadingHTTPServer):
    """A loopback bridge that records ``/exec`` commands and can stall ``/kill``.

    *exec_fault* makes ``/exec`` misbehave: ``"trickle"`` sends a reply
    body that never completes (a byte every 50 ms until the bridge is
    torn down, whatever ``/kill`` says), ``"drop"`` closes the
    connection without a reply, ``"bad-code"`` answers with an
    ``exit_code`` that is no integer, ``"deep"`` with :data:`_DEEP_JSON`.
    *ping_fault* makes ``/ping`` misbehave the same way: ``"trickle"``,
    ``"header-trickle"`` (the status line, then a header that never
    completes), ``"chunk-trickle"`` (complete headers, then a chunk size
    line that never completes), ``"list"`` (a JSON list) or ``"deep"``
    (:data:`_DEEP_JSON`).  *peer_gone* is set by a trickling handler
    whose client ended the connection.  *stop_on_exec* is set by the
    ``/exec`` handler just before it replies: the task's Stop landing as
    the reply completes.
    """

    daemon_threads = True

    def __init__(
        self,
        exec_delay: float,
        kill_delay: float,
        exec_fault: str = "",
        ping_fault: str = "",
        stop_on_exec: threading.Event | None = None,
    ) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.exec_delay = exec_delay
        self.kill_delay = kill_delay
        self.exec_fault = exec_fault
        self.ping_fault = ping_fault
        self.stop_on_exec = stop_on_exec
        self.commands: list[str] = []
        self.tokens: list[str] = []
        self.kills: list[str] = []
        #: ``time.monotonic()`` at which the ``/exec`` request arrived (0 until it does).
        self.exec_at = 0.0
        self.torn_down = threading.Event()
        self.peer_gone = threading.Event()


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
        if self.bridge.ping_fault == "trickle":
            self._trickle()
        elif self.bridge.ping_fault == "header-trickle":
            self._trickle(b"HTTP/1.1 200 OK\r\nX-Probe: ")
        elif self.bridge.ping_fault == "chunk-trickle":
            self._trickle(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n1;pad=")
        elif self.bridge.ping_fault in ("list", "deep"):
            data = b"[]" if self.bridge.ping_fault == "list" else _DEEP_JSON
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._reply({"ok": True})

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/exec":
            self.bridge.commands.append(body["command"])
            self.bridge.tokens.append(body["token"])
            self.bridge.exec_at = time.monotonic()
            if self.bridge.exec_fault == "drop":
                return  # HTTP/1.0: the connection closes without a reply
            if self.bridge.exec_fault == "trickle":
                self._trickle()
                return
            time.sleep(self.bridge.exec_delay)
            if self.bridge.stop_on_exec is not None:
                self.bridge.stop_on_exec.set()
            if self.bridge.exec_fault == "bad-code":
                self._reply({"exit_code": "bad", "output": "accepted"})
                return
            if self.bridge.exec_fault == "deep":
                self.send_response(200)
                self.send_header("Content-Length", str(len(_DEEP_JSON)))
                self.end_headers()
                self.wfile.write(_DEEP_JSON)
                return
            self._reply({"exit_code": 0, "output": "RAN"})
        else:
            self.bridge.kills.append(body["token"])
            time.sleep(self.bridge.kill_delay)
            self._reply({"ok": True})

    def _trickle(self, prefix: bytes = b"") -> None:
        """Send a reply that never completes: a byte every 50 ms until teardown.

        The bytes follow *prefix* (raw, unfinished HTTP) when given, else
        a complete status line and headers: the bytes are then a body.
        """
        if prefix:
            self.wfile.write(prefix)
        else:
            self.send_response(200)
            self.end_headers()
        while not self.bridge.torn_down.is_set():
            try:
                self.wfile.write(b" ")
                self.wfile.flush()
            except OSError:
                self.bridge.peer_gone.set()
                return  # the client closed its side
            time.sleep(0.05)


@contextmanager
def _bridge(
    exec_delay: float = 0.0,
    kill_delay: float = 0.0,
    exec_fault: str = "",
    ping_fault: str = "",
    stop_on_exec: threading.Event | None = None,
) -> Iterator[tuple[_Bridge, ExecBridge]]:
    server = _Bridge(exec_delay, kill_delay, exec_fault, ping_fault, stop_on_exec)
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


@pytest.mark.skipif(sys.platform != "linux", reason="the bridge's tagged-tree kill scans /proc")
def test_the_gate_shell_of_a_bridged_task_runs_its_command_raw(tmp_path: Path) -> None:
    """``SorcarAgent._gate_shell`` on a bridged task, over the real bridge server and a local
    environment: the output whole (three times what ``Bash`` keeps), the deadline kept, a task
    Stop ending the running command (its tree holds the pipe ``alive`` open) and, once set,
    posting nothing."""
    loop = _Loop()
    server = ExecBridgeServer(LocalEnvironment(str(tmp_path)), loop.loop)
    agent = SorcarAgent("bridged gate")
    agent.docker_manager = bridge = ExecBridge(f"exec:{server.url}")
    bridge.stop_event = stop = threading.Event()
    alive, late = tmp_path / "alive", tmp_path / "late"
    os.mkfifo(alive)
    reader = os.open(alive, os.O_RDONLY | os.O_NONBLOCK)

    def stop_once_started() -> None:
        if select.select([reader], [], [], 30)[0] and os.read(reader, 8) == b"started\n":
            stop.set()

    stopper = threading.Thread(target=stop_once_started)
    stopper.start()
    try:
        size = 3 * MAX_OUTPUT_CHARS
        assert agent._gate_shell(f"head -c {size} /dev/zero | tr '\\0' x", 30) == (0, "x" * size)
        assert agent._gate_shell("sleep 30", 1) == (None, "")
        running = f"exec 3>{alive}; echo started >&3; sleep 30"
        assert agent._gate_shell(running, 60) == (-1, STOPPED_MESSAGE)
        # End of file: the last process of the stopped command closed the pipe.
        assert select.select([reader], [], [], 10)[0] and os.read(reader, 8) == b""
        assert agent._gate_shell(f"touch {late}", 7) == (-1, STOPPED_MESSAGE)
    finally:
        _close(server, loop)
        stopper.join(35)
        os.close(reader)
    assert not late.exists(), "a stopped gate command was still posted"


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


def _exec_through(
    url: str, results: list[tuple[int | None, str]], cancel: threading.Event | None = None,
) -> None:
    """Run ``bridge_exec`` with a deadline that passes at once and record its result."""
    results.append(bridge_exec(url, "sleep 30", 0.01, cancel))


class _PollAfterRequest(threading.Event):
    """A cancel event whose poll, once ``/exec`` was posted, waits until just before its deadline.

    ``bridge_exec`` polls the event at the top of every wait slice
    (:func:`kiss.agents.sorcar.useful_tools._stop_requested`), checks
    its own deadline, then joins the request thread for 0.2 s.  Held
    here until 0.1 s before the request's deadline (its arrival plus
    the command's timeout and :data:`_REPLY_MARGIN_S`), the poll's
    slice is the one in which the request thread's read gives up: the
    wait ends through the thread's death, past the deadline, with the
    read's error as the outcome, which must still be the timeout.
    """

    def __init__(self, server: _Bridge, timeout: float) -> None:
        super().__init__()
        self.server = server
        self.timeout = timeout

    def is_set(self) -> bool:
        """Whether the event fired, answered just before the posted request's deadline."""
        if self.server.exec_at:
            deadline = self.server.exec_at + self.timeout + _REPLY_MARGIN_S
            self.wait(max(0.0, deadline - 0.1 - time.monotonic()))
        return super().is_set()


def _thread_alive(name: str) -> bool:
    """Whether a thread named *name* is alive."""
    return any(t.name == name for t in threading.enumerate())


def _ping_through(url: str, verdicts: list[bool]) -> None:
    """Run ``bridge_alive`` and record its verdict."""
    verdicts.append(bridge_alive(url))


def _request_through(url: str, errors: list[KISSError]) -> None:
    """Request ``/ping`` with a 0.5 s deadline and record the error it raises."""
    try:
        _request(url, "/ping", None, 0.5)
    except KISSError as error:
        errors.append(error)


@pytest.mark.parametrize("fault", ["header-trickle", "chunk-trickle"])
def test_a_header_or_chunk_size_that_never_completes_ends_at_the_deadline(fault: str) -> None:
    with _bridge(ping_fault=fault) as (server, bridge):
        errors: list[KISSError] = []
        requester = threading.Thread(
            target=_request_through, args=(bridge.url, errors), daemon=True,
        )
        started = time.monotonic()
        requester.start()
        requester.join(0.5 + 2.0)
        elapsed = time.monotonic() - started
        assert not requester.is_alive(), f"the request still blocked after {elapsed:.1f}s"
        assert elapsed >= 0.5, f"gave up before the deadline: {elapsed:.2f}s"
        assert len(errors) == 1 and "did not complete within the deadline" in str(errors[0]), errors
        # The request's socket went with it: the bridge's next write fails.
        assert server.peer_gone.wait(2), "the request's socket outlived the deadline"


def test_a_reply_that_never_completes_ends_at_the_overall_deadline() -> None:
    with _bridge(exec_fault="trickle") as (server, bridge):
        results: list[tuple[int | None, str]] = []
        runner = threading.Thread(
            target=_exec_through, args=(bridge.url, results, _PollAfterRequest(server, 0.01)),
            daemon=True,
        )
        started = time.monotonic()
        runner.start()
        runner.join(0.01 + _REPLY_MARGIN_S + _KILL_WAIT_S + 2.0)
        elapsed = time.monotonic() - started
        assert not runner.is_alive(), f"bridge_exec still blocked after {elapsed:.1f}s"
        assert elapsed >= 0.01 + _REPLY_MARGIN_S, f"gave up before the margin: {elapsed:.1f}s"
        assert results == [(None, "")], results
        assert server.kills == server.tokens == [server.tokens[0]]
        # The bridge keeps trickling (it ignores the kill): the request
        # thread still ends, with its socket, at its own deadline.
        worker = f"exec-bridge-{server.tokens[0][:8]}"
        deadline = time.monotonic() + 2.0
        while _thread_alive(worker) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _thread_alive(worker), "the request thread outlived the call"


def test_a_ping_that_never_completes_or_is_no_object_means_not_alive() -> None:
    with _bridge(ping_fault="list") as (_, bridge):
        assert bridge_alive(bridge.url) is False
    with _bridge(ping_fault="trickle") as (_, bridge):
        verdicts: list[bool] = []
        pinger = threading.Thread(target=_ping_through, args=(bridge.url, verdicts), daemon=True)
        started = time.monotonic()
        pinger.start()
        pinger.join(10 + 5.0)
        elapsed = time.monotonic() - started
        assert not pinger.is_alive(), f"bridge_alive still blocked after {elapsed:.1f}s"
        assert verdicts == [False], verdicts


def test_a_stop_set_before_the_call_never_posts_the_command() -> None:
    with _bridge() as (server, bridge):
        cancel = threading.Event()
        cancel.set()
        assert bridge_exec(bridge.url, "echo hi", 30, cancel) == (-1, STOPPED_MESSAGE)
        assert server.commands == [] and server.kills == []


def test_a_stop_landing_as_the_reply_completes_still_wins() -> None:
    cancel = threading.Event()
    with _bridge(stop_on_exec=cancel) as (server, bridge):
        assert bridge_exec(bridge.url, "echo hi", 30, cancel) == (-1, STOPPED_MESSAGE)
        assert server.commands == ["echo hi"]
        assert server.kills == server.tokens == [server.tokens[0]]


def test_an_exit_code_that_is_no_integer_is_an_invalid_reply_that_kills_the_command() -> None:
    with _bridge(exec_fault="bad-code") as (server, bridge):
        with pytest.raises(KISSError, match="invalid reply"):
            bridge_exec(bridge.url, "echo hi", 30)
        assert server.kills == server.tokens == [server.tokens[0]]


def test_run_commands_parallel_reports_an_unreachable_bridge_as_not_run() -> None:
    report = ExecBridge("exec:http://127.0.0.1:9").run_commands_parallel('["true"]')
    assert report.startswith("1 commands: 0 succeeded, 1 failed, 0 timed out."), report
    assert "[1/1] exit -1 in" in report and "Error: " in report, report


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:notaport", "http://127.0.0.1:99999", "http://[::1"],
)
def test_a_bridge_url_that_names_no_endpoint_is_an_unreachable_bridge(url: str) -> None:
    assert bridge_alive(url) is False
    with pytest.raises(KISSError, match="unreachable"):
        bridge_exec(url, "true", 5)
    with pytest.raises(KISSError, match="is not answering"):
        ExecBridge(f"exec:{url}").__enter__()
    report = ExecBridge(f"exec:{url}").run_commands_parallel('["true"]')
    assert report.startswith("1 commands: 0 succeeded, 1 failed, 0 timed out."), report
    assert "[1/1] exit -1 in" in report and "Error: " in report, report


@pytest.mark.parametrize(("url", "port"), [("http://[::1]", 80), ("http://[::1]:0", 0)])
def test_an_ipv6_bridge_url_dials_its_host_on_the_default_port_unless_it_names_one(
    url: str, port: int,
) -> None:
    """The address is read where the request makes its socket (a test cannot listen on port 80)."""
    dialled: list[tuple[str, int]] = []

    def record(frame: FrameType, event: str, arg: object) -> None:
        if event == "call" and frame.f_code is socket.create_connection.__code__:
            dialled.append(frame.f_locals["address"])

    profiler = sys.getprofile()
    sys.setprofile(record)
    try:
        bridge_alive(url)
    finally:
        sys.setprofile(profiler)
    assert dialled == [("::1", port)]


def test_a_stop_raised_at_the_back_edge_of_the_wait_loop_kills_the_command() -> None:
    code = bridge_exec.__code__
    raised: list[int] = []

    def at_back_edge(frame: FrameType, event: str, arg: object) -> Any:
        if frame.f_code is not code:
            return None
        frame.f_trace_opcodes = True
        back_edge = code.co_code[frame.f_lasti] == dis.opmap["JUMP_BACKWARD"]
        if event == "opcode" and back_edge and not raised:
            raised.append(frame.f_lineno)
            raise KeyboardInterrupt
        return at_back_edge

    with _bridge(exec_delay=30.0) as (server, bridge):
        tracer = sys.gettrace()
        sys.settrace(at_back_edge)
        try:
            with pytest.raises(KeyboardInterrupt):
                bridge_exec(bridge.url, "sleep 30", 30)
        finally:
            sys.settrace(tracer)
        assert raised, "the wait never went round its loop"
        assert server.kills == server.tokens == [server.tokens[0]]


def _is_the_deadline_timer(frame_locals: dict[str, Any]) -> bool:
    """Whether the thread being built is a request's deadline timer."""
    return isinstance(frame_locals["self"], threading.Timer)


def _is_the_request_thread(frame_locals: dict[str, Any]) -> bool:
    """Whether the thread being built is the one ``bridge_exec`` makes its request on."""
    return str(frame_locals.get("name") or "").startswith("exec-bridge-")


def _starve(
    first: Callable[[dict[str, Any]], bool], starved: ExitStack, frame_locals: dict[str, Any],
) -> bool:
    """``Pause`` matcher at ``Thread.__init__``: from the thread *first* names on, none starts."""
    if first(frame_locals):
        starved.enter_context(nproc_limit_lowered_to_one())
    return False


@pytest.mark.skipif(not thread_start_can_be_starved(), reason="Thread.start cannot be starved")
@pytest.mark.parametrize("first", [_is_the_deadline_timer, _is_the_request_thread])
def test_a_bridge_call_out_of_threads_is_an_unreachable_bridge(
    first: Callable[[dict[str, Any]], bool],
) -> None:
    with _bridge() as (server, bridge):
        with ExitStack() as starved, Pause(
            threading.Thread.__init__.__code__, partial(_starve, first, starved),
        ):
            with pytest.raises(KISSError, match="unreachable: can't start new thread"):
                bridge_exec(bridge.url, "true", 5)
            assert bridge_alive(bridge.url) is False
        assert server.commands == []
        assert bridge_alive(bridge.url) is True


def test_a_reply_nested_too_deep_to_decode_is_an_unreachable_bridge() -> None:
    with _bridge(ping_fault="deep", exec_fault="deep") as (server, bridge):
        assert bridge_alive(bridge.url) is False
        with pytest.raises(KISSError, match="unreachable"):
            bridge_exec(bridge.url, "echo hi", 30)
        # The command was accepted: it is killed before the error propagates.
        assert server.kills == server.tokens == [server.tokens[0]]


def test_a_reply_dropped_after_exec_was_accepted_kills_the_command() -> None:
    with _bridge(exec_fault="drop") as (server, bridge):
        started = time.monotonic()
        with pytest.raises(KISSError, match="unreachable"):
            bridge_exec(bridge.url, "sleep 30", 30)
        assert time.monotonic() - started < _KILL_WAIT_S + 1.0
        assert server.commands == ["sleep 30"]
        assert server.kills == server.tokens == [server.tokens[0]]
