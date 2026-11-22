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
* A Stop raised where the start of a helper thread (a request's
  deadline timer, the thread ``bridge_exec`` makes its request on, the
  one that posts its kill) has taken the lock of the thread's
  ``_started`` event strands no thread: each start is a step the Stop
  waits for (audit 2026-11-21, L3 O1).
"""

from __future__ import annotations

import dis
import inspect
import json
import queue
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import FrameType
from typing import Any, cast

import pytest

from kiss.agents.sorcar.docker_manager import _KILL_WAIT_S, MAX_OUTPUT_CHARS, STOPPED_MESSAGE
from kiss.agents.sorcar.exec_bridge import (
    _REPLY_MARGIN_S,
    ExecBridge,
    _kill_bounded,
    _request,
    bridge_alive,
    bridge_exec,
)
from kiss.agents.sorcar.shell_guards import DESTRUCTIVE_VERDICT
from kiss.core.kiss_error import KISSError
from kiss.core.tool_interrupt import inject_async_exception
from kiss.tests.conftest import nproc_limit_lowered_to_one, thread_start_can_be_starved
from kiss.tests.lock_injection import stop_as_it_takes_its_lock, stranded_threads
from kiss.tests.server.pause import Pause, Pauses

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
    "url", ["http://127.0.0.1:notaport", "http://127.0.0.1:99999", "http://[::1", "http://:1"],
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


def _is_a_name_lookup(frame_locals: dict[str, Any]) -> bool:
    """Whether the ``getaddrinfo`` being entered asks the resolver for ``localhost``."""
    return frame_locals["host"] == "localhost" and not frame_locals["flags"]


def test_a_name_lookup_that_hangs_ends_at_the_deadline_and_holds_one_thread() -> None:
    """Audit 2026-11-21 S5/F4: nothing bounds the resolver, so a request to a bridge named by
    a host name waited in ``getaddrinfo`` past its deadline (``bridge_alive`` past its ten
    seconds).  The lookup is a helper thread's: each request gives up at its own deadline, the
    requests for the name share the lookup in progress, and the bridge answers once it ends.

    Not landed here: a Ctrl-C (the main thread's own, which nothing holds back) that takes a
    request between the start of its lookup thread and the thread's entry in the table of
    lookups; ``_look_up`` then finds no entry of its own to remove, the one branch of it no
    test without a fault injector takes."""
    with _bridge() as (server, _):
        url = f"http://localhost:{server.server_port}"
        errors: list[KISSError] = []
        with Pause(socket.getaddrinfo.__code__, _is_a_name_lookup) as resolver:
            for _ in range(2):
                requester = threading.Thread(
                    target=_request_through, args=(url, errors), daemon=True)
                requester.start()
                requester.join(5)
                assert not requester.is_alive(), "the request still waits for the resolver"
            assert resolver.paused.is_set()
            assert len(errors) == 2 and all("unreachable" in str(error) for error in errors)
            assert sum(t.name == "exec-bridge-lookup" for t in threading.enumerate()) == 1
        ended = time.monotonic() + 5
        while _thread_alive("exec-bridge-lookup") and time.monotonic() < ended:
            time.sleep(0.01)
        assert not _thread_alive("exec-bridge-lookup")
        assert bridge_alive(url) is True


def _is_a_lookup_thread(frame_locals: dict[str, Any]) -> bool:
    """Whether the thread coming up (in ``Thread._bootstrap_inner``) is a name lookup's."""
    return bool(frame_locals["self"].name == "exec-bridge-lookup")


def _hold_a_name_lookup(
    held: list[threading.Thread], go: threading.Event, frame_locals: dict[str, Any],
) -> bool:
    """``Pause`` matcher at ``getaddrinfo``: every lookup of ``localhost`` waits there for *go*."""
    if _is_a_name_lookup(frame_locals):
        held.append(threading.current_thread())
        go.wait(60)
    return False


def _call_until_stopped(
    call: Callable[[], object], stopped: list[bool], profile: Any = None,
) -> None:
    """Make *call* (under the *profile* hook, if any); record whether a Stop ended it."""
    sys.setprofile(profile)
    try:
        call()
        stopped.append(False)
    except KeyboardInterrupt:
        stopped.append(True)
    finally:
        sys.setprofile(None)


def test_a_stop_as_the_lookup_thread_starts_leaves_the_lookup_to_the_next_request() -> None:
    """Audit 2026-11-21 S5/F4, review: the lookup thread was started before its lookup was
    entered in the table, and ``Thread.start`` waits for the thread to come up, where an
    injected Stop is raised.  A request stopped there left a resolver no later request found:
    every request for the name stopped that way, and the one after it, started one more
    thread.  The start and the entry are now one step, which a Stop waits for."""
    with _bridge() as (server, _):
        url = f"http://localhost:{server.server_port}"
        resolvers: list[threading.Thread] = []
        resolve = threading.Event()
        stopped: list[bool] = []
        errors: list[KISSError] = []
        bootstrap = threading.Thread._bootstrap_inner.__code__  # type: ignore[attr-defined]
        coming_up = Pause(bootstrap, _is_a_lookup_thread)
        hung = Pause(socket.getaddrinfo.__code__, partial(_hold_a_name_lookup, resolvers, resolve))
        stopped_one = threading.Thread(
            target=_call_until_stopped, args=(partial(bridge_alive, url), stopped), daemon=True)
        next_one = threading.Thread(target=_request_through, args=(url, errors), daemon=True)
        try:
            with Pauses(coming_up, hung):
                stopped_one.start()
                assert coming_up.paused.wait(5), "the request started no lookup thread"
                # The request is in ``Thread.start``, which the held thread has yet to answer.
                assert inject_async_exception(cast(int, stopped_one.ident), KeyboardInterrupt) == 1
                coming_up.release()
                stopped_one.join(5)
                assert stopped == [True], "the Stop did not end the request"
                held_by = time.monotonic() + 5
                while not resolvers and time.monotonic() < held_by:
                    time.sleep(0.01)
                next_one.start()
                next_one.join(5)
                assert not next_one.is_alive(), "the next request still waits for the resolver"
                assert len(errors) == 1 and "unreachable" in str(errors[0]), errors
                assert len(resolvers) == 1, "the next request started a lookup of its own"
        finally:
            resolve.set()
        ended = time.monotonic() + 5
        while _thread_alive("exec-bridge-lookup") and time.monotonic() < ended:
            time.sleep(0.01)
        assert not _thread_alive("exec-bridge-lookup")
        assert bridge_alive(url) is True


class _HeldAsItsWaitTakesTheLookup:
    """A profile hook for a requesting thread: hold it where its wait for the name's lookup
    (``Future.result``) has taken the lookup's condition, ahead of the ``with`` block that
    would release it (CPython gh-74174), until the test has injected the Stop."""

    def __init__(self) -> None:
        self.at_lock: queue.SimpleQueue[None] = queue.SimpleQueue()
        self.injected: queue.SimpleQueue[None] = queue.SimpleQueue()

    def __call__(self, frame: FrameType, event: str, arg: object) -> None:
        if (
            event == "return"
            and frame.f_code is threading.Condition.__enter__.__code__
            and cast(FrameType, frame.f_back).f_code is Future.result.__code__
        ):
            sys.setprofile(None)
            self.at_lock.put(None)
            self.injected.get()


def test_a_stop_as_a_request_begins_its_wait_for_the_lookup_leaves_the_lookup_usable() -> None:
    """The wait for the name's lookup takes the lookup's condition.  A Stop raised right after
    that left the condition held for good: the lookup's thread never published its answer, the
    lookup stayed in the table, and every later request for the name hung on it, whatever its
    deadline.  The wait is made in slices a Stop is held for."""
    with _bridge() as (server, _):
        url = f"http://localhost:{server.server_port}"
        hook = _HeldAsItsWaitTakesTheLookup()
        stopped: list[bool] = []
        requester = threading.Thread(
            target=_call_until_stopped,
            args=(partial(bridge_alive, url), stopped, hook), daemon=True)
        with Pause(socket.getaddrinfo.__code__, _is_a_name_lookup) as resolver:
            requester.start()
            hook.at_lock.get(timeout=5)
            assert inject_async_exception(cast(int, requester.ident), KeyboardInterrupt) == 1
            hook.injected.put(None)
            requester.join(5)
            assert stopped == [True], "the Stop did not end the request"
            assert resolver.paused.is_set()
        ended = time.monotonic() + 5
        while _thread_alive("exec-bridge-lookup") and time.monotonic() < ended:
            time.sleep(0.01)
        assert not _thread_alive("exec-bridge-lookup"), "the lookup's thread never published"
        assert bridge_alive(url) is True


@pytest.mark.parametrize(
    "starter", [_request, bridge_exec, _kill_bounded],
    ids=["deadline-timer", "request-thread", "killer"],
)
def test_a_stop_as_a_helper_thread_starts_strands_no_thread(
    starter: Callable[..., object],
) -> None:
    """``Thread.start`` waits for the new thread to come up, on the event's condition: a Stop
    raised as it had taken that lock left it held (gh-74174), and the thread blocked on it for
    good before it ran anything.  *starter* is the function whose helper thread the Stop lands
    in the start of; the kill's is started for the task's Stop, set here as the bridge
    replies.  The command is killed whichever start the Stop lands in."""
    cancel = threading.Event()
    with _bridge(stop_on_exec=cancel) as (server, bridge):
        call = (
            partial(bridge_alive, bridge.url) if starter is _request
            else partial(bridge_exec, bridge.url, "true", 30, cancel)
        )
        hook = stop_as_it_takes_its_lock(threading.Event.wait, threading.Thread.start, starter)
        stopped: list[bool] = []
        before = threading.enumerate()
        caller = threading.Thread(
            target=_call_until_stopped, args=(call, stopped, hook), daemon=True)
        caller.start()
        caller.join(20)
        assert stopped == [True], "the Stop did not end the call"
        assert stranded_threads(before) == []
        assert set(server.kills) == set(server.tokens)


@contextmanager
def _silent_listeners() -> Iterator[int]:
    """Listen on one port of ``127.0.0.1`` and ``::1`` with full backlogs; yield the port.

    A connect to either gets no answer.  The test is skipped where
    ``localhost`` does not name both addresses, or the kernel answers a
    connect to a full backlog.
    """
    with ExitStack() as opened:
        port = 0
        for family, address in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
            listener = opened.enter_context(socket.socket(family))
            try:
                listener.bind((address, port))
            except OSError:
                pytest.skip(f"cannot listen on {address}")
            listener.listen(0)
            port = listener.getsockname()[1]
            opened.enter_context(socket.create_connection((address, port), 5))  # fills it
            try:
                socket.create_connection((address, port), 0.3).close()
            except TimeoutError:
                continue
            except OSError:
                pass
            pytest.skip("a connect to a full backlog is answered here")
        named = {info[4][0] for info in socket.getaddrinfo(
            "localhost", port, type=socket.SOCK_STREAM)}
        if not {"127.0.0.1", "::1"} <= named:
            pytest.skip("localhost does not name both loopback addresses")
        yield port


def test_the_addresses_of_a_name_share_the_requests_deadline() -> None:
    """Audit 2026-11-21 S5/F4: each address of a name got the whole timeout in turn, so a
    bridge named ``localhost`` that answered on neither loopback address held a request for
    twice its deadline."""
    with _silent_listeners() as port:
        started = time.monotonic()
        with pytest.raises(KISSError, match="unreachable"):
            _request(f"http://localhost:{port}", "/ping", None, 1.5)
        assert time.monotonic() - started < 2.5


def _link_local_listener() -> socket.socket:
    """Return a socket listening on a link-local IPv6 address of this host (else skip)."""
    for interface, _ in socket.if_nameindex():
        try:
            with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as probe:
                probe.connect(("ff02::1", 9, 0, interface))  # sends nothing: names the source
                address = probe.getsockname()[0]
            if address.startswith("fe80:"):
                return socket.create_server((address, 0, 0, interface), family=socket.AF_INET6)
        except OSError:
            continue
    pytest.skip("no link-local IPv6 address to listen on")


def _answer_a_ping(listener: socket.socket) -> None:
    """Accept one connection on *listener* and answer its request as a bridge alive does."""
    connection, _ = listener.accept()
    with connection:
        connection.recv(65536)
        connection.sendall(b'HTTP/1.1 200 OK\r\nContent-Length: 12\r\n\r\n{"ok": true}')


def test_a_bridge_on_a_link_local_address_is_dialled_on_its_interface() -> None:
    """Audit 2026-11-21 S5/F4: the request looks its addresses up itself before it connects,
    and a link-local IPv6 one without its interface is no address (``EINVAL``).

    The URL names the interface as the host writes it: by its name (``%en0``) on POSIX, by
    its index (``%4``) on Windows, whose resolver takes no interface name there.
    """
    with _link_local_listener() as listener:
        address, port, _, interface = listener.getsockname()
        zone = interface if sys.platform == "win32" else socket.if_indextoname(interface)
        answering = threading.Thread(target=_answer_a_ping, args=(listener,), daemon=True)
        answering.start()
        assert bridge_alive(f"http://[{address}%{zone}]:{port}") is True
        answering.join(5)
