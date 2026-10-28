# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop pressed during a non-streamed request ends the turn at once.

Without a token callback the OpenAI-compatible transports issue one
blocking ``create`` call.  No stream watchdog can cover it (there is no
response, hence no socket, until the provider answers) and the client's
30-minute timeout is kept on purpose: a non-streamed reasoning turn
legitimately sends nothing for minutes (see
``test_non_streaming_call_keeps_the_long_client_timeout``).  So a Stop
pressed while the provider was silent used to leave the agent parked in
``recv()`` until the reply or the timeout, and the reply then came back
as a finished turn.

``_blocking_request`` now runs the call on a worker thread, watches the
stop event meanwhile and, on a Stop, shuts the pending connection down
and raises the user's ``KeyboardInterrupt`` within a poll interval.  The
server here accepts the connection and never answers the first request,
which is exactly the case nothing else bounds.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

from kiss.core import stop_signal
from kiss.core.models.openai_compatible_model import (
    _ADAPTIVE_TOOL_EFFORT_LOCK,
    OpenAICompatibleModel,
)
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.openai_sse_harness import Reply, Request, ScriptedOpenAIServer

_MODEL = "gpt-blocking-under-test"
_STOP_AFTER = 0.3
# Generous next to the 0.1 s stop poll; far below the 1800 s client timeout.
_UNWIND_WITHIN = 3.0
_WORKER_THREAD = "model-blocking-request"
_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {"name": "finish", "description": "Finish.", "parameters": {}},
    }
]


def _json_reply(path: str) -> dict[str, Any]:
    """Return a complete (non-streamed) reply body for the Chat Completions or Responses path."""
    if path == "/v1/responses":
        return {
            "id": "resp_slow",
            "object": "response",
            "status": "completed",
            "model": _MODEL,
            "output": [
                {
                    "type": "message",
                    "id": "msg_slow",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "late but complete", "annotations": []}
                    ],
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        }
    return {
        "id": "chatcmpl-slow",
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
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


class _SilentThenPromptGateway:
    """Never answer the first request (not even a status line); answer every later one at once.

    ``release`` lets the fixture free the handler thread parked on the
    first request when the test is over.
    """

    def __init__(self) -> None:
        self.release = threading.Event()
        self._first = True

    def __call__(self, request: Request) -> Reply:
        """Return the held reply for the first request, an immediate one afterwards."""
        assert not request.body.get("stream"), "the callback-less path must not stream"
        body = _json_reply(request.path)
        if self._first:
            self._first = False
            return Reply(headers_gate=self.release, json_body=body)
        return Reply(json_body=body)


class _StopAsReplyLandsGateway:
    """Fire *stop* right before writing a complete reply.

    The reply then races the 0.1 s stop poll: the turn is either aborted
    just before the reply lands, or finishes the call with the Stop set
    and must bill the reply.  Both are a correct Stop.
    """

    def __init__(self, stop: threading.Event) -> None:
        self._stop = stop

    def __call__(self, request: Request) -> Reply:
        """Set the Stop and return the immediate reply."""
        self._stop.set()
        return Reply(json_body=_json_reply(request.path))


class _HeldEffortRejectionGateway:
    """Answer the first request with a 400 ``reasoning_effort`` rejection whose handling is held.

    The rejection is the one ``_create_chat_completion_adaptive`` retries
    without the effort.  Its bytes cannot be held back until after the
    Stop (the abort shuts the socket down, so a later reply never reaches
    the worker); instead the verdict-cache lock the worker takes right
    after receiving the 400 is held, which parks the worker between the
    rejection and its retry.  ``release`` lets the test choose when the
    retry may proceed.
    """

    def __init__(self) -> None:
        self.release = threading.Event()
        self._held = threading.Event()
        self._first = True

    def _hold_verdict_lock(self) -> None:
        """Own the verdict-cache lock until ``release`` fires."""
        with _ADAPTIVE_TOOL_EFFORT_LOCK:
            self._held.set()
            self.release.wait(_UNWIND_WITHIN * 3)

    def __call__(self, request: Request) -> Reply:
        """Return the 400 (once the lock is held) for the first request, a prompt reply later."""
        if self._first:
            self._first = False
            assert "reasoning_effort" in request.body, "the probe must carry the effort"
            threading.Thread(target=self._hold_verdict_lock, daemon=True).start()
            assert self._held.wait(_UNWIND_WITHIN)
            error = {
                "error": {
                    "message": "reasoning_effort is not supported with tools",
                    "type": "invalid_request_error",
                }
            }
            return Reply(status=400, json_body=error)
        return Reply(json_body=_json_reply(request.path))


def _run_with_stop(
    call: Callable[[], Any], stop: threading.Event, stop_after: float | None
) -> tuple[BaseException | None, float]:
    """Run *call* under *stop*, fired *stop_after* seconds in if given; return (error, elapsed)."""
    stop_signal.set_thread_stop_event(stop)
    started = time.monotonic()
    try:
        if stop_after is not None:
            threading.Timer(stop_after, stop.set).start()
        call()
    except BaseException as error:  # noqa: BLE001 - the error is the result
        return error, time.monotonic() - started
    finally:
        stop_signal.set_thread_stop_event(None)
    return None, time.monotonic() - started


def _worker_threads() -> list[threading.Thread]:
    """Return the live worker threads carrying a blocking model request."""
    return [t for t in threading.enumerate() if t.name == _WORKER_THREAD]


def _assert_unwound_promptly(server: Any, error: BaseException | None, elapsed: float) -> None:
    """The pending request was torn down as the user's Stop, within seconds, and not retried."""
    assert isinstance(error, KeyboardInterrupt), f"the stopped turn returned {error!r}"
    assert elapsed < _UNWIND_WITHIN, f"Stop took {elapsed:.2f}s to unwind the pending request"
    deadline = time.monotonic() + _UNWIND_WITHIN
    while _worker_threads() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _worker_threads(), "the aborted request is still blocked on its worker thread"
    assert len(server.requests) == 1, "the SDK retried the request the user stopped"


@pytest.fixture
def silent_server() -> Any:
    """A scripted server that never answers its first request and answers the rest at once."""
    gateway = _SilentThenPromptGateway()
    with ScriptedOpenAIServer(gateway) as scripted:
        yield scripted
    gateway.release.set()


def test_chat_completions_text_turn_unwinds_while_the_request_is_pending(
    silent_server: Any,
) -> None:
    """``_stream_text`` without a callback: the Stop interrupts the silent request."""
    model = OpenAICompatibleModel(_MODEL, base_url=silent_server.base_url, api_key="k")
    model.initialize("Say something.")
    error, elapsed = _run_with_stop(model.generate, threading.Event(), _STOP_AFTER)
    _assert_unwound_promptly(silent_server, error, elapsed)
    assert model.take_partial_usage_response() is None, "nothing arrived, nothing to bill"
    # The transport is still usable: the next turn gets a fresh connection.
    assert model.generate()[0] == "late but complete"
    assert len(silent_server.requests) == 2


def test_chat_completions_tools_turn_unwinds_while_the_request_is_pending(
    silent_server: Any,
) -> None:
    """The adaptive tools path without a callback is interrupted the same way."""
    model = OpenAICompatibleModel(_MODEL, base_url=silent_server.base_url, api_key="k")
    model.initialize("Use the finish tool.")
    error, elapsed = _run_with_stop(
        lambda: model.generate_and_process_with_tools({}, tools_schema=_TOOLS_SCHEMA),
        threading.Event(),
        _STOP_AFTER,
    )
    _assert_unwound_promptly(silent_server, error, elapsed)


def test_responses_turn_unwinds_while_the_request_is_pending(silent_server: Any) -> None:
    """v2's ``_request_response`` without a callback is interrupted the same way."""
    model = OpenAICompatibleModel2(_MODEL, base_url=silent_server.base_url, api_key="k")
    model.initialize("Say something.")
    error, elapsed = _run_with_stop(model.generate, threading.Event(), _STOP_AFTER)
    _assert_unwound_promptly(silent_server, error, elapsed)
    assert model.generate()[0] == "late but complete"
    assert len(silent_server.requests) == 2


def test_a_stop_landing_with_the_reply_is_still_a_stop() -> None:
    """A Stop set as the reply is written never comes back as a finished turn.

    When the reply wins the race with the stop poll it was billed, so it
    must be kept for the usage drain.
    """
    stop = threading.Event()
    with ScriptedOpenAIServer(_StopAsReplyLandsGateway(stop)) as server:
        model = OpenAICompatibleModel(_MODEL, base_url=server.base_url, api_key="k")
        model.initialize("Say something.")
        error, _elapsed = _run_with_stop(model.generate, stop, None)
        assert isinstance(error, KeyboardInterrupt), f"the stopped turn returned {error!r}"
        assert len(server.requests) == 1
        billed = model.take_partial_usage_response()
        if billed is not None:
            assert model.extract_input_output_token_counts_from_response(billed)[0] == 5


def test_a_reply_landing_as_the_stop_wakes_the_waiter_is_billed() -> None:
    """The reply written the instant the Stop fires is recorded for the usage drain.

    The server sets the Stop and writes the reply right after it, so the
    worker stores the reply while the waiting thread is waking up.  The
    turn is still a Stop, but the provider billed the reply, so it must
    come back from ``take_partial_usage_response`` instead of being
    dropped by a waiter that never looked at the finished worker.
    """
    stop = threading.Event()
    with ScriptedOpenAIServer(_StopAsReplyLandsGateway(stop)) as server:
        model = OpenAICompatibleModel(_MODEL, base_url=server.base_url, api_key="k")
        model.initialize("Say something.")
        error, _elapsed = _run_with_stop(model.generate, stop, None)
        assert isinstance(error, KeyboardInterrupt), f"the stopped turn returned {error!r}"
        assert len(server.requests) == 1
        assert not _worker_threads(), "the worker that stored the reply is still around"
        billed = model.take_partial_usage_response()
        assert billed is not None, "the reply that landed with the Stop was billed but lost"
        assert model.extract_input_output_token_counts_from_response(billed)[0] == 5


def test_an_abandoned_adaptive_probe_never_retries_through_the_next_turns_client() -> None:
    """A stopped tools turn whose 400 is handled later does not re-send itself.

    The adaptive tools path probes with ``reasoning_effort`` attached and
    retries once without it when the endpoint answers 400.  Here the
    worker is parked on the 400 until after the Stop unwound the turn and
    the next turn built a fresh client: the abandoned worker must not run
    its retry through that client, so the server sees the probe, the
    next turn and nothing else.
    """
    gateway = _HeldEffortRejectionGateway()
    with ScriptedOpenAIServer(gateway) as server:
        model = OpenAICompatibleModel(
            _MODEL,
            base_url=server.base_url,
            api_key="k",
            model_config={"reasoning_effort": "high"},
        )
        model.initialize("Use the finish tool.")
        error, elapsed = _run_with_stop(
            lambda: model.generate_and_process_with_tools({}, tools_schema=_TOOLS_SCHEMA),
            threading.Event(),
            _STOP_AFTER,
        )
        assert isinstance(error, KeyboardInterrupt), f"the stopped turn returned {error!r}"
        assert elapsed < _UNWIND_WITHIN
        assert len(server.requests) == 1
        # The next turn rebuilds the client and completes normally.
        assert model.generate()[0] == "late but complete"
        assert len(server.requests) == 2
        # Now the stopped turn's worker gets to act on its 400.
        gateway.release.set()
        deadline = time.monotonic() + _UNWIND_WITHIN
        while _worker_threads() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _worker_threads(), "the abandoned probe is still running"
        assert len(server.requests) == 2, "the stopped turn retried through the next turn's client"
        assert all("reasoning_effort" in r.body for r in server.requests if r.body.get("tools")), (
            "a retry without the effort was sent"
        )


def test_a_turn_nobody_can_stop_still_completes() -> None:
    """Control: outside a stoppable task (no stop event) the call runs inline and returns."""
    with ScriptedOpenAIServer(_StopAsReplyLandsGateway(threading.Event())) as server:
        model = OpenAICompatibleModel(_MODEL, base_url=server.base_url, api_key="k")
        model.initialize("Say something.")
        assert model.generate()[0] == "late but complete"
        assert model.take_partial_usage_response() is None
        assert not _worker_threads()


def test_a_turn_nobody_stopped_still_completes() -> None:
    """Control: with a stop event nobody fires, the worker-thread call returns the reply."""
    with ScriptedOpenAIServer(_StopAsReplyLandsGateway(threading.Event())) as server:
        model = OpenAICompatibleModel(_MODEL, base_url=server.base_url, api_key="k")
        model.initialize("Say something.")
        error, _elapsed = _run_with_stop(model.generate, threading.Event(), None)
        assert error is None, f"the unstopped turn failed: {error!r}"
        assert model.conversation[-1]["content"] == "late but complete"
        assert model.take_partial_usage_response() is None


def test_a_reply_landing_as_the_abort_runs_is_billed() -> None:
    """A reply that lands between the liveness check and the abort is kept for the bill.

    The Stop poll found the worker alive and tore the client down; the
    worker, which had the whole reply in hand, stores it a moment later.
    The provider billed that reply, so it must come back from
    ``take_partial_usage_response`` rather than be dropped because the
    waiter only looked at the outcome when the worker had already exited.
    """
    model = OpenAICompatibleModel(_MODEL, base_url="http://127.0.0.1:9", api_key="k")
    aborted = threading.Event()
    original_abort = model._abort_pending_request

    def abort_then_let_the_reply_land() -> None:
        original_abort()
        aborted.set()

    reply = object()

    def create(**_kwargs: Any) -> Any:
        assert aborted.wait(_UNWIND_WITHIN), "the Stop never aborted the pending request"
        return reply

    model._abort_pending_request = abort_then_let_the_reply_land  # type: ignore[method-assign]
    error, elapsed = _run_with_stop(
        lambda: model._blocking_request(create, {}), threading.Event(), _STOP_AFTER
    )
    assert isinstance(error, KeyboardInterrupt), f"the stopped turn returned {error!r}"
    assert elapsed < _UNWIND_WITHIN
    assert model.take_partial_usage_response() is reply, (
        "the reply that landed as the abort ran was billed but lost"
    )


def test_a_stopped_delegated_responses_turn_leaves_a_usable_client(silent_server: Any) -> None:
    """Stopping a tools turn delegated to the Responses API does not poison the next turn.

    The delegate sends through this model's own SDK client.  The abort
    closes that shared client and drops the delegate's reference; the
    parent must drop its reference too, or ``_reuse_client`` keeps the
    closed pool (its inputs did not change) and the next turn fails with
    ``APIConnectionError`` before any request reaches the server.
    """
    model = OpenAICompatibleModel(
        _MODEL,
        base_url=silent_server.base_url,
        api_key="k",
        model_config={"use_responses_api": True, "reasoning_effort": "high"},
    )
    model.initialize("Use the finish tool.")
    error, elapsed = _run_with_stop(
        lambda: model.generate_and_process_with_tools({}, tools_schema=_TOOLS_SCHEMA),
        threading.Event(),
        _STOP_AFTER,
    )
    _assert_unwound_promptly(silent_server, error, elapsed)
    assert silent_server.requests[0].path == "/v1/responses", "the turn was not delegated"
    assert model.client is None, "the parent kept the shared client the abort closed"
    # The next delegated turn rebuilds the shared client and completes.
    _calls, content, _response = model.generate_and_process_with_tools(
        {}, tools_schema=_TOOLS_SCHEMA
    )
    assert content == "late but complete"
    assert len(silent_server.requests) == 2


def test_a_reply_stored_before_a_forced_interrupt_is_billed() -> None:
    """A ``KeyboardInterrupt`` raised into the waiting thread keeps the stored reply.

    The runner's forced stop raises ``KeyboardInterrupt`` asynchronously
    in the task thread when the cooperative stop event is not honoured
    in time.  Here it lands in the waiter's poll right after the worker
    stored the reply: the turn unwinds as a Stop, and the reply, billed
    by the provider, is kept for the usage drain.
    """
    reply = object()
    waiting = threading.Event()

    class _InterruptedWait(threading.Event):
        def wait(self, timeout: float | None = None) -> bool:
            # Let the worker finish (it stores the reply and exits), then
            # interrupt the waiter in its poll.
            waiting.set()
            deadline = time.monotonic() + _UNWIND_WITHIN
            while _worker_threads() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not _worker_threads(), "the worker never stored its reply"
            raise KeyboardInterrupt

    def create(**_kwargs: Any) -> Any:
        assert waiting.wait(_UNWIND_WITHIN), "the waiter never polled the stop event"
        return reply

    model = OpenAICompatibleModel(_MODEL, base_url="http://127.0.0.1:9", api_key="k")
    stop_signal.set_thread_stop_event(_InterruptedWait())
    try:
        with pytest.raises(KeyboardInterrupt):
            model._blocking_request(create, {})
    finally:
        stop_signal.set_thread_stop_event(None)
    deadline = time.monotonic() + _UNWIND_WITHIN
    while _worker_threads() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert model.take_partial_usage_response() is reply, (
        "the reply stored before the forced interrupt was billed but lost"
    )
