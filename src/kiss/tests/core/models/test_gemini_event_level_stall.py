# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: Gemini must survive a stream that is quiet but not silent.

``GeminiModel`` bounds a stalled turn with a byte-level deadline: its
``_ResponseTrackingHttpxClient`` forces ``httpx.Timeout(stall_timeout)``
onto every request, so a socket carrying no bytes raises
``httpx.ReadTimeout`` and becomes the retryable ``TimeoutError`` the
agentic loop expects.

That deadline cannot see the other half of the problem.  ``google-genai``
parses the response with ``ApiClient._iter_response_stream``, which does::

    for line in self.response_stream.iter_lines():
      if not line:
        continue
      if line.startswith('data: '):
        yield line[len('data: '):]

Every blank line — the ordinary SSE keep-alive, and what a wedged
gateway or an over-eager proxy emits to hold a connection open — is
therefore real traffic that resets httpx's read clock and yields the
adapter nothing at all.  The agent is starved while the socket looks
perfectly healthy, which is exactly the failure
``stream_stall_timeout`` exists to bound: the same event-level blind
spot ``AnthropicModel`` closes with its watchdog for ``ping`` events.

Now that ``stop_aware_events`` raises a retryable ``TimeoutError`` on a
stall (rather than ending the iterator at EOF), the fix is to pass
``stall_timeout`` to it as well, so both levels are covered.

No mocks, patches, fakes or test doubles: a real ``ThreadingHTTPServer``
speaks the genuine Gemini SSE wire format to the real ``google-genai``
SDK over a real socket, and simply keeps writing blank lines.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Generator
from typing import Any

import httpx
import pytest
from google.genai import errors

from kiss.core import stop_signal
from kiss.core.kiss_error import ContentFilterError
from kiss.core.models.gemini_model import GeminiModel
from kiss.tests.core.models.gemini_sse_harness import (
    GeminiScript,
    chunk,
    serve,
    text_part,
)

_MODEL = "gemini-keepalive-under-test"
_USAGE = {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8}
# The count on a chunk sent while the answer is still being generated.
_RUNNING_USAGE = {"promptTokenCount": 5, "candidatesTokenCount": 1, "totalTokenCount": 6}
_STALL_TIMEOUT = 2.0
# Comfortably above the stall timeout, and far below the time an
# unbounded keep-alive stream would take (it never ends).
_DEADLINE = 20.0
_ANSWER = [chunk([text_part("Done.")], usage=_USAGE, finish_reason="STOP")]
# An error the endpoint reports inside a response it opened with a 200.
_OVERLOADED = [{"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}]
# Retry delays a test does not notice.
_QUICK = {"initial_delay": 0.01, "max_delay": 0.01, "jitter": 0.001}


@pytest.fixture
def gemini_endpoint() -> Generator[tuple[str, GeminiScript]]:
    """A real local Gemini endpoint for one test."""
    yield from serve()


def _make_model(
    monkeypatch: pytest.MonkeyPatch, base_url: str, **model_config: Any
) -> GeminiModel:
    """Build a real ``GeminiModel`` whose stream goes to the local endpoint."""
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
    model = GeminiModel(
        _MODEL,
        api_key="test-key",
        model_config={"stream_stall_timeout": _STALL_TIMEOUT, **model_config},
        token_callback=lambda _t: None,
    )
    model.initialize("Explain event-level stalls.")
    return model


def _run_bounded(model: GeminiModel) -> tuple[BaseException | None, float]:
    """Run ``generate()`` on a worker thread bounded by :data:`_DEADLINE`.

    Args:
        model: The adapter under test.

    Returns:
        ``(exception_or_None, elapsed_seconds)``.
    """
    outcome: dict[str, BaseException] = {}
    started = time.monotonic()

    def target() -> None:
        try:
            model.generate()
        except BaseException as exc:  # noqa: BLE001 — reported to the test
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(_DEADLINE)
    if worker.is_alive():
        pytest.fail(
            f"generate() was still reading a keep-alive-only stream "
            f"{_DEADLINE}s later — stream_stall_timeout={_STALL_TIMEOUT}s is "
            f"not enforced at the event level, only at the byte level"
        )
    return outcome.get("error"), time.monotonic() - started


def _generate_under(
    stop: threading.Event, model: GeminiModel, outcome: dict[str, BaseException]
) -> None:
    """Thread body: run ``generate()`` with *stop* bound as the thread's Stop; record its error."""
    stop_signal.set_thread_stop_event(stop)
    try:
        model.generate()
    except BaseException as exc:  # noqa: BLE001 — reported to the test
        outcome["error"] = exc


class TestKeepAliveOnlyStreamStallsOut:
    """A stream carrying only SDK-filtered bytes must still time out."""

    def test_blank_line_keepalives_do_not_hold_the_agent(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
    ) -> None:
        """Blank-line traffic must not reset the stall clock forever."""
        base_url, script = gemini_endpoint
        script.play([chunk([text_part("Working on it")])], after="keepalive")
        model = _make_model(monkeypatch, base_url)

        error, elapsed = _run_bounded(model)

        assert isinstance(error, TimeoutError), f"got {error!r}"
        assert "stall" in str(error).lower()
        assert elapsed < _DEADLINE / 2, f"the stall took {elapsed:.1f}s to surface"

    def test_keepalives_before_any_chunk_stall_out_too(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
    ) -> None:
        """The same holds when no chunk ever arrives at all."""
        base_url, script = gemini_endpoint
        script.play([], after="keepalive")
        model = _make_model(monkeypatch, base_url)

        error, elapsed = _run_bounded(model)

        assert isinstance(error, TimeoutError), f"got {error!r}"
        assert elapsed < _DEADLINE / 2, f"the stall took {elapsed:.1f}s to surface"

    def test_headers_that_never_complete_stall_out(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
    ) -> None:
        """A peer trickling header bytes is cut by a deadline the read clock cannot provide.

        No response exists before the headers are complete, so the event
        watchdog has nothing to half-close, and every byte resets the
        byte-level deadline.
        """
        base_url, script = gemini_endpoint
        script.play([], headers="drip")
        model = _make_model(monkeypatch, base_url)

        error, _elapsed = _run_bounded(model)

        assert isinstance(error, TimeoutError), f"got {error!r}"
        assert "stall" in str(error).lower()

    def test_a_stop_while_headers_trickle_in_ends_the_turn(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
    ) -> None:
        """A Stop pressed before the headers are complete is honoured within seconds."""
        base_url, script = gemini_endpoint
        script.play([], headers="drip")
        model = _make_model(monkeypatch, base_url)
        stop = threading.Event()
        outcome: dict[str, BaseException] = {}
        worker = threading.Thread(target=_generate_under, args=(stop, model, outcome), daemon=True)
        worker.start()
        assert script.serving.wait(10), "the endpoint never started its headers"
        stop.set()
        worker.join(5.0)
        assert not worker.is_alive(), "the Stop did not end the pre-header wait"
        assert isinstance(outcome.get("error"), KeyboardInterrupt), f"got {outcome!r}"

    def test_a_stop_during_the_tls_handshake_ends_the_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Stop pressed while the peer sits on the TLS ClientHello is honoured at once.

        The pool lists a connection only once it is set up, so the abort found
        nothing to cut and the turn waited out the 10 s connect timeout (audit
        A/C4-F1, B/C4-F1).
        """
        stop = threading.Event()
        outcome: dict[str, BaseException] = {}
        with socket.create_server(("127.0.0.1", 0)) as listener:
            listener.settimeout(10)
            model = _make_model(monkeypatch, f"https://127.0.0.1:{listener.getsockname()[1]}")
            worker = threading.Thread(
                target=_generate_under, args=(stop, model, outcome), daemon=True
            )
            worker.start()
            try:
                peer, _ = listener.accept()
                with peer:  # never answered: the client waits for a ServerHello
                    peer.settimeout(10)
                    assert peer.recv(1), "no ClientHello arrived"
                    stop.set()
                    worker.join(5.0)
                    stopped = not worker.is_alive()
            finally:
                stop.set()
                worker.join(_DEADLINE)  # the closed peer ends a handshake the Stop did not
        assert stopped, "the Stop did not end the TLS handshake"
        assert isinstance(outcome.get("error"), KeyboardInterrupt), f"got {outcome!r}"

    def test_a_stall_during_an_sdk_retry_is_still_cut(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
    ) -> None:
        """The watchdog must not settle on the error reply of an attempt that is retried.

        The first attempt is refused with a 503 the SDK finishes with, and
        is retried after a delay longer than the stall timeout; the retry
        then streams keep-alives only.  A watchdog that adopted the 503
        as "this stream's response" would abort a dead reply and leave
        the retry to run for good.
        """
        base_url, script = gemini_endpoint
        script.refusals = 1
        script.play([], after="keepalive")
        monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
        model = GeminiModel(
            _MODEL,
            api_key="test-key",
            model_config={
                "stream_stall_timeout": 0.5,
                "http_options": {
                    "retry_options": {
                        "attempts": 2,
                        "initial_delay": 1.0,
                        "max_delay": 1.0,
                        "jitter": 0.001,
                    }
                },
            },
            token_callback=lambda _t: None,
        )
        model.initialize("Explain event-level stalls.")

        error, elapsed = _run_bounded(model)

        assert len(script.requests) == 2, "the refused attempt was not retried"
        assert isinstance(error, TimeoutError), f"got {error!r}"
        assert "stall" in str(error).lower()
        assert elapsed < _DEADLINE / 2, f"the stall took {elapsed:.1f}s to surface"

    @pytest.mark.parametrize(
        ("refusals", "retry_options", "chunks", "requests", "failure"),
        [
            (2, {"attempts": 3, **_QUICK}, _ANSWER, 3, None),
            (2, {"attempts": 2, **_QUICK}, _ANSWER, 2, errors.ServerError),
            (1, {"http_status_codes": [429], **_QUICK}, _ANSWER, 1, errors.ServerError),
            (0, _QUICK, _OVERLOADED, 1, errors.ServerError),
        ],
        ids=["answered-in-time", "out-of-attempts", "another-status", "refused-in-an-open-stream"],
    )
    def test_a_refused_request_is_sent_again_as_the_retry_options_ask(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
        refusals: int,
        retry_options: dict[str, Any],
        chunks: list[dict[str, Any]],
        requests: int,
        failure: type[errors.APIError] | None,
    ) -> None:
        """The retry policy of ``http_options.retry_options`` is the SDK's, whoever applies it.

        A refusal with one of the status codes is retried while attempts
        are left; an error inside a response that opened is not.
        """
        base_url, script = gemini_endpoint
        script.refusals = refusals
        script.play(chunks)
        model = _make_model(monkeypatch, base_url, http_options={"retry_options": retry_options})

        error, _elapsed = _run_bounded(model)

        assert len(script.requests) == requests
        if failure is None:
            assert error is None and model.conversation[-1]["content"] == "Done.", f"got {error!r}"
        else:
            assert isinstance(error, failure) and error.code == 503, f"got {error!r}"

    def test_a_failed_connect_is_tried_again_as_a_refusal_is(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A connect error is retried like a refusal: three attempts wait two backoffs."""
        with socket.create_server(("127.0.0.1", 0)) as listener:
            down = f"http://127.0.0.1:{listener.getsockname()[1]}"  # refuses, once closed
        retry_options = {"attempts": 3, "initial_delay": 0.2, "max_delay": 0.2, "jitter": 0.001}
        model = _make_model(monkeypatch, down, http_options={"retry_options": retry_options})

        error, elapsed = _run_bounded(model)

        assert isinstance(error, httpx.ConnectError), f"got {error!r}"
        assert elapsed > 0.3, "the failed connect was not tried twice more"

    @pytest.mark.parametrize("cut", ["stop", "header-deadline"])
    def test_a_retry_backoff_ends_at_a_stop_and_at_the_header_deadline(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
        cut: str,
    ) -> None:
        """A turn waiting to retry leaves at the Stop, and when its headers are overdue.

        The SDK slept the backoff in ``time.sleep``, 30 s here, which
        neither of them ends (audit C5/F2).
        """
        base_url, script = gemini_endpoint
        script.refusals = 1
        script.play(_ANSWER)
        stall_timeout = _STALL_TIMEOUT if cut == "stop" else 0.2
        model = _make_model(
            monkeypatch,
            base_url,
            stream_stall_timeout=stall_timeout,
            http_options={
                "retry_options": {"attempts": 2, "initial_delay": 30.0, "max_delay": 30.0}
            },
        )
        stop = threading.Event()
        outcome: dict[str, BaseException] = {}
        worker = threading.Thread(target=_generate_under, args=(stop, model, outcome), daemon=True)
        worker.start()
        refused_by = time.monotonic() + 10
        while not script.requests and time.monotonic() < refused_by:
            time.sleep(0.01)
        time.sleep(0.3)  # the refusal has been read: the turn is in its backoff
        if cut == "stop":
            stop.set()
        worker.join(_DEADLINE)  # shorter than the backoff, longer than the header budget
        assert not worker.is_alive(), "the turn sleeps on in its backoff"
        expected = KeyboardInterrupt if cut == "stop" else TimeoutError
        assert isinstance(outcome.get("error"), expected), f"got {outcome!r}"
        assert len(script.requests) == 1, "the request was sent again after the cut"

    def test_healthy_stream_is_unaffected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
    ) -> None:
        """A stream that answers promptly must not be aborted."""
        base_url, script = gemini_endpoint
        script.play([chunk([text_part("All done.")], finish_reason="STOP")])
        model = _make_model(monkeypatch, base_url)

        content, _ = model.generate()

        assert content == "All done."


class TestBodyHeldOpenAfterTheTurnEnded:
    """A turn the provider ended is used at once, however long its body stays open.

    A gateway may hold a finished response open and feed it keep-alives.
    The adapter read on to the end of the body, so the stall watchdog cut
    a complete answer, which the agent then asked and paid for again, and
    a refusal was retried as a stall instead of failing over.
    """

    @pytest.mark.parametrize(
        "chunks",
        [
            [chunk([text_part("Done.")], usage=_USAGE, finish_reason="STOP")],
            [chunk([text_part("Done.")], finish_reason="STOP"), {"usageMetadata": _USAGE}],
            [
                chunk([text_part("Do")], usage=_RUNNING_USAGE),
                chunk([text_part("ne.")], finish_reason="STOP"),
                {"usageMetadata": _USAGE},
            ],
        ],
        ids=[
            "usage-with-the-finish-reason",
            "usage-after-the-finish-reason",
            "running-usage-then-usage-after-the-finish-reason",
        ],
    )
    def test_a_finished_answer_is_returned(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
        chunks: list[dict[str, object]],
    ) -> None:
        """The finish reason and the usage end the turn: one request, billed, the stream closed.

        A finish reason alone does not, whatever usage came before it: that
        was a running count, and the chunk that bills the turn may follow.
        """
        base_url, script = gemini_endpoint
        script.play(chunks, after="keepalive")
        model = _make_model(monkeypatch, base_url)

        content, response = model.generate()

        assert content == "Done."
        assert [m["role"] for m in model.conversation] == ["user", "assistant"]
        assert model.extract_input_output_token_counts_from_response(response) == (5, 3, 0, 0)
        assert model.take_partial_usage_response() is None
        assert len(script.requests) == 1
        assert script.hung_up.wait(10), "the held-open connection was not closed"

    @pytest.mark.parametrize(
        "terminal",
        [
            chunk([], usage=_USAGE, finish_reason="SAFETY"),
            {"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": _USAGE},
        ],
        ids=["safety-stop", "prompt-blocked"],
    )
    def test_a_refusal_is_a_content_filter_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
        terminal: dict[str, object],
    ) -> None:
        """A filter stop and a blocked prompt fail over; their usage stays billable."""
        base_url, script = gemini_endpoint
        script.play([terminal], after="keepalive")
        model = _make_model(monkeypatch, base_url)

        with pytest.raises(ContentFilterError) as refusal:
            model.generate()

        assert refusal.value.stop_reason == "SAFETY"
        assert [m["role"] for m in model.conversation] == ["user"]
        partial = model.take_partial_usage_response()
        assert model.extract_input_output_token_counts_from_response(partial) == (5, 3, 0, 0)
        assert len(script.requests) == 1
        assert script.hung_up.wait(10), "the held-open connection was not closed"
