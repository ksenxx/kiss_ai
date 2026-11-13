# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a streaming request whose headers never arrive must fail.

Audit B1 (2026-10-01).  ``stop_aware_events`` arms its watchdog only once
``create(stream=True)`` has returned the response headers.  Before that,
the only clock was the SDK client's scalar 1800 s timeout, so a gateway
that accepted the TCP connection and then queued the request forever
parked the agent in ``recv()`` for 30 minutes per attempt, deaf to Stop.

The streaming creates now carry a per-request
``httpx.Timeout(stream_stall_timeout, connect=10)``, and a timeout raised
by either clock is reported as the same retryable stall error.  Each test
below talks to a real loopback HTTP server that reads the request and
writes nothing at all until the test releases it.  Non-streaming calls
keep the long client default: the control test proves a silent-but-slow
non-streaming reply still succeeds past the stall timeout.

Audit C4 (2026-11-08).  The per-request timeout bounds the silence
between two reads only: a gateway that trickled one header byte at a
time kept every read short of it, and until the header was complete
there was no response for the stream watchdog to cut, so the turn hung
for good and a Stop went unseen.  The stream is now opened under a
request watchdog (a Stop-aware absolute deadline), whose cut is
reported as the same stall or Stop; the client it cut is retired so the
next turn of the same model instance sends through a fresh one.  The
``_TricklingHeaderServer`` tests below talk to a raw loopback socket
that never finishes its header.

Audit B/C5 (2026-11-10).  Cutting the pool's sockets wakes a call
waiting on one, not the SDK asleep before its retry: a 429 whose
``Retry-After`` the SDK honours (up to a minute) held the turn past
the Stop and the opening deadline.  The opening call now runs on a
worker thread the turn leaves behind; a stream that still opens there
is closed by the worker.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Generator
from typing import Any

import pytest
from openai import BadRequestError, OpenAI

from kiss.core import stop_signal
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.core.tool_interrupt import inject_async_exception
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    assert_left_during_the_retry_sleep,
    chat_chunk,
    responses_event,
    retry_later,
)

_STALL_TIMEOUT = 1.0
# Generous bound: the SDK retries a pre-headers timeout once (``_MAX_RETRIES
# = 1``) with a sub-second backoff, so two silent attempts plus overhead
# must still finish well inside this; the pre-fix behaviour was 1800 s.
_DEADLINE = 12.0
_MODEL = "gpt-silent-gateway-under-test"


def _responses_stream_chunks() -> list[bytes]:
    """A complete Responses stream: one ``response.completed`` event."""
    return [
        responses_event(
            "response.completed",
            {
                "response": {
                    "id": "resp_late",
                    "status": "completed",
                    "output": [],
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 0,
                        "total_tokens": 1,
                    },
                }
            },
        )
    ]


def _chat_stream_chunks() -> list[bytes]:
    """A complete Chat Completions stream: one chunk carrying the text and the finish."""
    return [
        chat_chunk(
            {
                "id": "chatcmpl-late",
                "object": "chat.completion.chunk",
                "model": _MODEL,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "late"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
    ]


class _SilentGateway:
    """Reads every request and answers nothing until ``release`` is set.

    Args:
        release_after: When set, the gate opens this many seconds after a
            request arrives, so the silence is measured from the server's
            side rather than from before the client started.
    """

    def __init__(self, release_after: float | None = None) -> None:
        self.release = threading.Event()
        self.release_after = release_after

    def __call__(self, request: Request) -> Reply:
        """Hold the status line back; reply normally once released."""
        if self.release_after is not None:
            threading.Timer(self.release_after, self.release.set).start()
        if request.path == "/v1/responses":
            return Reply(headers_gate=self.release, sse_chunks=_responses_stream_chunks())
        if request.body.get("stream"):
            return Reply(headers_gate=self.release, sse_chunks=_chat_stream_chunks())
        return Reply(
            headers_gate=self.release,
            json_body={
                "id": "chatcmpl-late",
                "object": "chat.completion",
                "created": 0,
                "model": _MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "late but complete"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
            },
        )


class _TricklingHeaderServer:
    """A loopback endpoint that never finishes its response header.

    While ``trickle`` is set, a request gets the status line and the
    start of an ``X-Padding`` header at once, then one more header byte
    every 20 ms until ``release`` fires or the client hangs up: every
    read returns a byte, so the SDK's read timeout never fires, and until
    the header is complete there is no response for the stream watchdog
    to cut.  With ``trickle`` cleared a request gets a complete
    one-event stream, so a test can check that the same model instance
    recovers.  A connection that sends no request (the SDK's retry of a
    cut request, shut down before its request is written) is dropped
    unrecorded.
    """

    def __init__(self) -> None:
        self.trickle = True
        self.release = threading.Event()
        self.started = threading.Event()
        self.requests: list[str] = []
        self._listener = socket.socket()
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(8)
        self._listener.settimeout(0.05)  # ``accept`` notices the close within a slice
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    @property
    def base_url(self) -> str:
        """The ``/v1`` base URL to hand to the model under test."""
        return f"http://127.0.0.1:{self._listener.getsockname()[1]}/v1"

    def __enter__(self) -> _TricklingHeaderServer:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release.set()
        self._closed.set()
        self._thread.join(5.0)
        self._listener.close()

    def _serve(self) -> None:
        """Accept connections until the server exits; each is handled on its own thread."""
        while not self._closed.is_set():
            try:
                connection, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            connection.settimeout(None)
            threading.Thread(target=self._handle, args=(connection,), daemon=True).start()

    def _handle(self, connection: socket.socket) -> None:
        """Read one request's head, then trickle the header or send the complete stream."""
        with connection:
            connection.settimeout(_DEADLINE)
            head = b""
            try:
                while b"\r\n\r\n" not in head:
                    chunk = connection.recv(4096)
                    if not chunk:
                        return
                    head += chunk
                path = head.split(b" ", 2)[1].decode()
                self.requests.append(path)
                if self.trickle:
                    connection.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nX-Padding: "
                    )
                    self.started.set()
                    while not self.release.wait(0.02):
                        connection.sendall(b"x")
                    return
                chunks = (
                    _responses_stream_chunks()
                    if path.endswith("/responses")
                    else _chat_stream_chunks()
                )
                body = b"".join(chunks)
                connection.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    b"Content-Length: %d\r\nConnection: close\r\n\r\n%s" % (len(body), body)
                )
            except OSError:
                return


def _assert_trickling_header_is_cut_then_the_instance_recovers(model: Any, stop: bool) -> None:
    """Run one turn against a header that never completes, then a healthy turn.

    Args:
        model: The initialised model under test, pointed at a
            :class:`_TricklingHeaderServer` (``model.base_url``).
        stop: Whether the user presses Stop a quarter-stall into the
            turn (expected: the Stop surfaces) or not (expected: the
            stall surfaces at the stream-opening deadline).
    """
    with _TricklingHeaderServer() as server:
        model.base_url = server.base_url
        if stop:
            _assert_stopped(*_run_with_deadline(lambda: _generate_then_stop(model)))
        else:
            _assert_stalled(*_run_with_deadline(model.generate))
        assert server.started.is_set(), "the request never reached the server"
        server.trickle = False
        error, _elapsed = _run_with_deadline(model.generate)
        assert error is None, f"the next turn of the same instance failed: {error!r}"
        assert len(server.requests) == 2, server.requests


@pytest.fixture
def silent_server() -> Generator[tuple[ScriptedOpenAIServer, _SilentGateway]]:
    """An endpoint that accepts connections but sends no headers."""
    gateway = _SilentGateway()
    with ScriptedOpenAIServer(gateway) as server:
        yield server, gateway
        gateway.release.set()


def _run_with_deadline(call: Any) -> tuple[BaseException | None, float]:
    """Run *call* on a worker thread bounded by the test deadline.

    Args:
        call: A zero-argument callable performing the model turn.

    Returns:
        ``(exception_or_None, elapsed_seconds)``.  Fails the test when
        the call is still running at the deadline, which is the pre-fix
        behaviour (waiting for the client's 1800 s timeout).
    """
    outcome: dict[str, BaseException] = {}
    started = time.monotonic()

    def target() -> None:
        try:
            call()
        except BaseException as exc:  # noqa: BLE001 — reported to the test
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(_DEADLINE)
    if worker.is_alive():
        pytest.fail(
            f"still waiting for response headers {_DEADLINE}s later — "
            f"stream_stall_timeout={_STALL_TIMEOUT}s did not bound the request"
        )
    return outcome.get("error"), time.monotonic() - started


def _assert_stalled(error: BaseException | None, elapsed: float) -> None:
    """Assert the turn failed as a retryable stall, promptly.

    Args:
        error: The exception the turn raised, if any.
        elapsed: Seconds the turn took.
    """
    assert isinstance(error, TimeoutError), f"got {error!r}"
    assert "stream_stall_timeout" in str(error)
    assert elapsed >= _STALL_TIMEOUT, f"gave up after only {elapsed:.2f}s"
    assert elapsed < _DEADLINE / 2, f"stall took {elapsed:.1f}s to surface"


def _generate_then_stop(model: Any) -> None:
    """Run ``model.generate()`` on a thread whose Stop fires a quarter-stall in.

    Args:
        model: The initialised model under test.
    """
    stop = threading.Event()
    stop_signal.set_thread_stop_event(stop)
    try:
        threading.Timer(_STALL_TIMEOUT / 4, stop.set).start()
        model.generate()
    finally:
        stop_signal.set_thread_stop_event(None)


def _generate_under(model: Any, stop: threading.Event, errors: list[BaseException]) -> None:
    """Thread body: run ``model.generate()`` with *stop* bound as the thread's stop event.

    Args:
        model: The initialised model under test.
        stop: The stop event to bind.
        errors: Receives what the turn raised.
    """
    stop_signal.set_thread_stop_event(stop)
    try:
        model.generate()
    except BaseException as error:  # noqa: BLE001 - the error is the result
        errors.append(error)
    finally:
        stop_signal.set_thread_stop_event(None)


def _assert_stopped(error: BaseException | None, elapsed: float) -> None:
    """Assert the turn unwound as a user stop, not a retryable stall.

    Args:
        error: The exception the turn raised, if any.
        elapsed: Seconds the turn took.
    """
    assert isinstance(error, KeyboardInterrupt), f"got {error!r}"
    assert elapsed < _DEADLINE / 2, f"stop took {elapsed:.1f}s to surface"


def _tool_schema() -> list[dict[str, Any]]:
    """Return one trivial tool so the adaptive (tools) path is exercised."""
    return [
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Finish.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


class TestChatCompletionsHeadersNeverArrive:
    """v1 (Chat Completions) streaming must not wait 1800 s for headers."""

    def test_toolless_stream_times_out(
        self, silent_server: tuple[ScriptedOpenAIServer, _SilentGateway]
    ) -> None:
        """``generate()`` with a token callback raises the stall error."""
        server, _gateway = silent_server
        model = OpenAICompatibleModel(
            _MODEL,
            base_url=server.base_url,
            api_key="test-key",
            model_config={"stream_stall_timeout": _STALL_TIMEOUT},
            token_callback=lambda _t: None,
        )
        model.initialize("Say something.")
        error, elapsed = _run_with_deadline(model.generate)
        _assert_stalled(error, elapsed)
        assert server.requests, "the request never reached the server"

    def test_tool_calling_stream_times_out(
        self, silent_server: tuple[ScriptedOpenAIServer, _SilentGateway]
    ) -> None:
        """The agentic path (tools attached) is bounded the same way."""
        server, _gateway = silent_server
        model = OpenAICompatibleModel(
            _MODEL,
            base_url=server.base_url,
            api_key="test-key",
            model_config={"stream_stall_timeout": _STALL_TIMEOUT},
            token_callback=lambda _t: None,
        )
        model.initialize("Use the finish tool.")
        error, elapsed = _run_with_deadline(
            lambda: model.generate_and_process_with_tools({}, tools_schema=_tool_schema())
        )
        _assert_stalled(error, elapsed)

    def test_stop_before_headers_is_a_stop_not_a_stall(
        self, silent_server: tuple[ScriptedOpenAIServer, _SilentGateway]
    ) -> None:
        """A Stop pressed while headers are pending must not be retried.

        No watchdog exists before the headers arrive, so the per-request
        timeout is what ends the wait; it must surface as the user's
        ``KeyboardInterrupt`` rather than the retryable stall error.
        """
        server, _gateway = silent_server
        model = OpenAICompatibleModel(
            _MODEL,
            base_url=server.base_url,
            api_key="test-key",
            model_config={"stream_stall_timeout": _STALL_TIMEOUT},
            token_callback=lambda _t: None,
        )
        model.initialize("Say something.")
        _assert_stopped(*_run_with_deadline(lambda: _generate_then_stop(model)))
        assert server.requests, "the request never reached the server"

    @pytest.mark.parametrize("stop", [False, True], ids=["timeout", "stop"])
    def test_a_trickling_header_is_cut(self, stop: bool) -> None:
        """A header that never completes is cut at the deadline, or at once on Stop."""
        model = OpenAICompatibleModel(
            _MODEL,
            base_url="http://127.0.0.1:9/v1",
            api_key="test-key",
            model_config={"stream_stall_timeout": _STALL_TIMEOUT},
            token_callback=lambda _t: None,
        )
        model.initialize("Say something.")
        _assert_trickling_header_is_cut_then_the_instance_recovers(model, stop)

    def test_non_streaming_call_keeps_the_long_client_timeout(self) -> None:
        """Without a token callback the stall timeout must not apply.

        A long reasoning turn with no streaming legitimately sends nothing
        for minutes; the server here stays silent for 2.5x the stall
        timeout and the call must still complete with the full reply.
        """
        with ScriptedOpenAIServer(_SilentGateway(2.5 * _STALL_TIMEOUT)) as server:
            model = OpenAICompatibleModel(
                _MODEL,
                base_url=server.base_url,
                api_key="test-key",
                model_config={"stream_stall_timeout": _STALL_TIMEOUT},
            )
            model.initialize("Think hard.")
            error, elapsed = _run_with_deadline(model.generate)
            assert error is None, f"non-streaming call failed: {error!r}"
            assert elapsed >= 2.5 * _STALL_TIMEOUT
            assert model.conversation[-1]["content"] == "late but complete"
            assert len(server.requests) == 1


class TestResponsesHeadersNeverArrive:
    """v2 (Responses API) streaming must not wait 1800 s for headers."""

    def test_responses_stream_times_out(
        self, silent_server: tuple[ScriptedOpenAIServer, _SilentGateway]
    ) -> None:
        """A silent ``/v1/responses`` endpoint raises the stall error."""
        server, _gateway = silent_server
        model = OpenAICompatibleModel2(
            _MODEL,
            base_url=server.base_url,
            api_key="test-key",
            model_config={"stream_stall_timeout": _STALL_TIMEOUT},
            token_callback=lambda _t: None,
        )
        model.initialize("Say something.")
        error, elapsed = _run_with_deadline(model.generate)
        _assert_stalled(error, elapsed)
        assert server.requests, "the request never reached the server"

    def test_stop_before_headers_is_a_stop_not_a_stall(
        self, silent_server: tuple[ScriptedOpenAIServer, _SilentGateway]
    ) -> None:
        """v2 unwinds a pre-headers Stop as ``KeyboardInterrupt`` too."""
        server, _gateway = silent_server
        model = OpenAICompatibleModel2(
            _MODEL,
            base_url=server.base_url,
            api_key="test-key",
            model_config={"stream_stall_timeout": _STALL_TIMEOUT},
            token_callback=lambda _t: None,
        )
        model.initialize("Say something.")
        _assert_stopped(*_run_with_deadline(lambda: _generate_then_stop(model)))
        assert server.requests, "the request never reached the server"

    @pytest.mark.parametrize("stop", [False, True], ids=["timeout", "stop"])
    def test_a_trickling_header_is_cut(self, stop: bool) -> None:
        """v2 shares the stream-opening deadline and the Stop with v1."""
        model = OpenAICompatibleModel2(
            _MODEL,
            base_url="http://127.0.0.1:9/v1",
            api_key="test-key",
            model_config={"stream_stall_timeout": _STALL_TIMEOUT},
            token_callback=lambda _t: None,
        )
        model.initialize("Say something.")
        _assert_trickling_header_is_cut_then_the_instance_recovers(model, stop)


def _streaming_model(base_url: str, stall_timeout: float) -> OpenAICompatibleModel:
    """Return an initialised streaming chat model pointed at *base_url*."""
    model = OpenAICompatibleModel(
        _MODEL,
        base_url=base_url,
        api_key="test-key",
        model_config={"stream_stall_timeout": stall_timeout},
        token_callback=lambda _t: None,
    )
    model.initialize("Say something.")
    return model


@pytest.mark.parametrize("stop", [False, True], ids=["deadline", "stop"])
def test_the_sdks_retry_sleep_does_not_hold_the_stream_opening(stop: bool) -> None:
    """The Stop and the opening deadline end a turn whose SDK sleeps out a ``Retry-After``.

    The request given up is not sent again, and its client is retired.
    """
    with ScriptedOpenAIServer(retry_later) as server:
        model = _streaming_model(server.base_url, _STALL_TIMEOUT / 4)
        stop_event = threading.Event()
        errors: list[BaseException] = []
        turn = threading.Thread(
            target=_generate_under, args=(model, stop_event, errors), daemon=True
        )
        assert_left_during_the_retry_sleep(turn, stop_event if stop else None)
        assert isinstance(errors[0], KeyboardInterrupt if stop else TimeoutError), errors
        assert model.client is None, "the client of the request given up was kept"
        assert len(server.requests) == 1, "the request given up was sent again"


def test_a_forced_stop_while_the_stream_opens_leaves_no_request_behind(
    silent_server: tuple[ScriptedOpenAIServer, _SilentGateway],
) -> None:
    """The runner's injected ``KeyboardInterrupt`` ends the turn and its opening call's thread."""
    server, _gateway = silent_server
    model = _streaming_model(server.base_url, 10 * _DEADLINE)
    errors: list[BaseException] = []
    turn = threading.Thread(
        target=_generate_under, args=(model, threading.Event(), errors), daemon=True
    )
    turn.start()
    deadline = time.monotonic() + _DEADLINE
    while not server.requests and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.requests, "the request never reached the server"
    assert inject_async_exception(turn.ident or -1, KeyboardInterrupt) == 1
    turn.join(_DEADLINE)
    assert not turn.is_alive() and isinstance(errors[0], KeyboardInterrupt), errors
    while _opening_threads() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _opening_threads(), "the opening call runs on as an orphan thread"


def _opening_threads() -> list[threading.Thread]:
    """The live worker threads of stream-opening calls."""
    return [thread for thread in threading.enumerate() if thread.name == "model-stream-open"]


def test_a_stream_that_opens_after_the_turn_gave_it_up_is_closed() -> None:
    """The opening call's worker closes a stream nobody is left to read.

    The call here stands for one the abort could not reach: it answers
    only after the turn has left with the stall, with a real stream the
    server holds open, and the server then sees the client hang up.
    """
    left, hold = threading.Event(), threading.Event()

    def respond(_request: Request) -> Reply:
        return Reply(sse_chunks=_chat_stream_chunks(), hold=hold)

    with (
        ScriptedOpenAIServer(respond) as server,
        OpenAI(base_url=server.base_url, api_key="test-key") as late_client,
    ):
        model = _streaming_model(server.base_url, _STALL_TIMEOUT / 4)

        def open_late(**kwargs: Any) -> Any:
            assert left.wait(_DEADLINE), "the turn never gave the opening call up"
            return late_client.chat.completions.create(**kwargs)

        try:
            with pytest.raises(TimeoutError, match="stream_stall_timeout"):
                model._stream_request(open_late, dict(_REQUEST), list)
            left.set()
            assert server.client_disconnected.wait(_DEADLINE), "the late stream was left open"
        finally:
            left.set()
            hold.set()


_REQUEST: dict[str, Any] = {"model": _MODEL, "messages": [{"role": "user", "content": "hi"}]}


def _refusal(_request: Request) -> Reply:
    """Responder: refuse every request with a 400, which the SDK does not retry."""
    return Reply(status=400, json_body={"error": {"message": "refused"}})


def test_a_refusal_that_arrives_with_the_stop_is_raised_as_itself() -> None:
    """Only a connection error is read as the Stop's own cut; the client is retired either way.

    The Stop is set as the provider's 400 comes back, before the opening
    call's worker hands it over.
    """
    stop = threading.Event()
    with ScriptedOpenAIServer(_refusal) as server:
        model = _streaming_model(server.base_url, _STALL_TIMEOUT)
        model._ensure_client()
        create = model.client.chat.completions.create

        def refuse_as_the_user_stops(**kwargs: Any) -> Any:
            try:
                return create(**kwargs)
            finally:
                stop.set()

        stop_signal.set_thread_stop_event(stop)
        try:
            with pytest.raises(BadRequestError):
                model._stream_request(refuse_as_the_user_stops, dict(_REQUEST), list)
        finally:
            stop_signal.set_thread_stop_event(None)
        assert model.client is None, "the stopped turn's client was kept"
