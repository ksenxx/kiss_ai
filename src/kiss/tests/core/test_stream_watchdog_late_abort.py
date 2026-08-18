# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: ``watchdog.stop()`` must guarantee no later abort.

Audit finding 01/F9.  ``StreamAbortWatchdog.stop()`` only set an
``Event``; it never joined the thread, and ``_watch`` re-checked the stop
condition *after* returning from ``self._done.wait(poll)``.  A watchdog
descheduled just before that check therefore woke up after the caller had
already finished with the stream — by which point the SDK has released
the response and httpx has returned the keep-alive socket to its pool —
and called ``sock.shutdown(SHUT_RDWR)`` on it anyway, concurrently with
the caller's own teardown.  The next request that picked that socket out
of the pool failed with a transport error.

Reproducing that window with sleeps would be flaky, so it is forced
exactly: the stop event handed to the watchdog is a **real**
``threading.Event`` whose ``is_set()`` additionally rendezvouses with the
test, parking the watchdog thread precisely inside the check.  The
server, in turn, withholds the rest of the stream until the watchdog is
parked, so the ordering is deterministic on every run.  Everything else
is real — real SDK, real sockets, real SSE, real threads.
"""

from __future__ import annotations

import threading
from collections.abc import Generator

import pytest

from kiss.core import stop_signal
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.stream_abort import StreamAbortWatchdog
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    chat_chunk,
)

_MODEL = "gpt-late-abort-under-test"
_WATCHDOG_MARKER = "abort-watchdog"
_DEADLINE = 30.0


def _content_chunk(text: str, finish_reason: str | None = None) -> bytes:
    """Render one Chat Completions content chunk.

    Args:
        text: The delta text.
        finish_reason: The choice's finish reason, when terminal.

    Returns:
        The SSE bytes.
    """
    return chat_chunk(
        {
            "id": "chatcmpl-late",
            "object": "chat.completion.chunk",
            "model": _MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }
            ],
        }
    )


_USAGE_CHUNK = chat_chunk(
    {
        "id": "chatcmpl-late",
        "object": "chat.completion.chunk",
        "model": _MODEL,
        "choices": [],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
)

_DONE = b"data: [DONE]\n\n"


class RendezvousStopEvent(threading.Event):
    """A real stop event that parks the watchdog inside its stop check.

    ``is_set()`` is what ``StreamAbortWatchdog._watch`` calls right after
    waking from its poll — the exact point the audit describes the
    watchdog being descheduled at.  The first observation of a set event
    announces itself on :attr:`reached` and then blocks on
    :attr:`proceed`, which hands the test control of the interleaving
    without changing what the event *means* to any reader.
    """

    def __init__(self) -> None:
        """Create the event together with its two rendezvous flags."""
        super().__init__()
        self.reached = threading.Event()
        self.proceed = threading.Event()
        self._parked = False

    def is_set(self) -> bool:
        """Return the real flag, parking once on the first set observation.

        Returns:
            Whether a stop has been requested.
        """
        value = super().is_set()
        if value and not self._parked:
            self._parked = True
            self.reached.set()
            self.proceed.wait(timeout=_DEADLINE)
        return value


_SECOND_REPLY = {
    "id": "chatcmpl-second",
    "object": "chat.completion",
    "model": _MODEL,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
}


class _StopOnFirstToken:
    """Presses Stop the moment the first streamed token arrives.

    ``token_callback`` is the hook production code already invokes for
    every delta, so the stop lands at a real, deterministic point in the
    stream rather than after an arbitrary sleep.
    """

    def __init__(self, event: RendezvousStopEvent) -> None:
        """Remember the event to set.

        Args:
            event: The stop event bound to the calling thread.
        """
        self.event = event
        self.tokens: list[str] = []

    def __call__(self, token: str) -> None:
        """Record the token and request a stop on the first one."""
        self.tokens.append(token)
        self.event.set()


class _GatedStreamPolicy:
    """Streams a first token, waits for the watchdog, then finishes.

    Any request after the first is answered non-streamed, so a follow-up
    turn on the same model exercises the connection pool.
    """

    def __init__(self, gate: threading.Event) -> None:
        """Remember the gate that releases the rest of the first stream.

        Args:
            gate: Event the handler waits on before the second chunk.
        """
        self.gate = gate
        self.chunks = [
            _content_chunk("first"),
            _content_chunk(" rest", finish_reason="stop"),
            _USAGE_CHUNK,
            _DONE,
        ]
        self.served = 0
        self.lock = threading.Lock()

    def __call__(self, request: Request) -> Reply:
        """Return the gated stream first, then plain JSON replies."""
        with self.lock:
            self.served += 1
            first = self.served == 1
        if first:
            return Reply(sse_chunks=self.chunks, chunk_gate=self.gate)
        return Reply(json_body=_SECOND_REPLY)


def _watchdog_threads() -> set[str]:
    """Return the names of this suite's own live watchdog threads.

    Only the ``openai-*`` watchdogs started by the OpenAI-compatible
    adapter under test are counted.  Other tests in the same pytest
    process can leave daemon threads behind (real-LLM tasks, agent
    loops) that start and stop ``anthropic-…``/``gemini-…`` watchdogs
    concurrently with this test; counting those made the
    baseline-vs-after comparison race with unrelated churn.
    """
    return {
        f"{thread.name}#{thread.ident}"
        for thread in threading.enumerate()
        if thread.name.startswith("openai-") and _WATCHDOG_MARKER in thread.name
    }


@pytest.fixture
def rendezvous() -> Generator[RendezvousStopEvent]:
    """A stop event bound to this thread and cleared afterwards."""
    event = RendezvousStopEvent()
    stop_signal.set_thread_stop_event(event)
    yield event
    event.proceed.set()
    stop_signal.set_thread_stop_event(None)


class TestDisarmedWatchdogNeverClaimsAnAbort:
    """The guard itself: a wakeup that lost the race must do nothing."""

    def test_claim_after_stop_is_refused(self) -> None:
        """A watchdog polling a set stop event must not abort after stop().

        Exercises the ordering directly on a real SDK stream: the stop
        event is set, the watchdog is disarmed, and the very check
        ``_watch`` makes on its next wakeup must decline — otherwise the
        socket that was just returned to the pool gets shut down.
        """
        policy = _GatedStreamPolicy(threading.Event())
        policy.gate.set()
        stop_event = threading.Event()
        with ScriptedOpenAIServer(policy) as server:
            model = OpenAICompatibleModel(
                _MODEL,
                base_url=server.base_url,
                api_key="test-key",
                model_config={"stream_stall_timeout": 120.0},
            )
            model.initialize("Say something.")
            stream = model.client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            watchdog = StreamAbortWatchdog(
                stream, stop_event=stop_event, name="claim-test-watchdog"
            )
            try:
                stop_event.set()
                watchdog.stop()
                assert watchdog._claim_abort() is False
                assert watchdog.stopped is False
            finally:
                watchdog.stop()
                stream.close()

    def test_poll_interval_without_a_stall_timeout(self) -> None:
        """Without a stall timeout the watchdog still polls for stops."""
        policy = _GatedStreamPolicy(threading.Event())
        policy.gate.set()
        with ScriptedOpenAIServer(policy) as server:
            model = OpenAICompatibleModel(
                _MODEL, base_url=server.base_url, api_key="test-key"
            )
            model.initialize("Say something.")
            stream = model.client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            watchdog = StreamAbortWatchdog(
                stream, stall_timeout=None, name="poll-test-watchdog"
            )
            try:
                assert watchdog._poll_interval() == (
                    StreamAbortWatchdog._STOP_POLL_SECONDS
                )
                assert watchdog._claim_abort() is False
            finally:
                watchdog.stop()
                stream.close()


class TestWatchdogStopIsFinal:
    """``stop()`` must not leave an armed watchdog behind."""

    def test_no_watchdog_survives_the_call(
        self, rendezvous: RendezvousStopEvent
    ) -> None:
        """A stopped watchdog must be gone by the time the call unwinds.

        The releaser hands ``proceed`` back only after a delay, so a
        ``stop()`` that neither serialises with the abort decision nor
        joins the thread returns while the watchdog is still parked and
        still armed — which is the defect.
        """
        policy = _GatedStreamPolicy(rendezvous.reached)
        baseline = _watchdog_threads()

        def release_after_rendezvous() -> None:
            """Let the parked watchdog run on, once the caller is past it."""
            rendezvous.reached.wait(timeout=_DEADLINE)
            rendezvous.proceed.wait(timeout=0.3)
            rendezvous.proceed.set()

        releaser = threading.Thread(target=release_after_rendezvous, daemon=True)
        releaser.start()

        with ScriptedOpenAIServer(policy) as server:
            model = OpenAICompatibleModel(
                _MODEL,
                base_url=server.base_url,
                api_key="test-key",
                model_config={"stream_stall_timeout": 120.0},
                token_callback=_StopOnFirstToken(rendezvous),
            )
            model.initialize("Say something.")
            with pytest.raises(KeyboardInterrupt):
                model.generate()

            assert _watchdog_threads() == baseline, (
                "the watchdog outlived stop(): it is still parked in its "
                "stop check and can still shut down a pooled socket"
            )
        releaser.join(timeout=_DEADLINE)

    def test_pooled_connection_survives_a_stopped_request(
        self, rendezvous: RendezvousStopEvent
    ) -> None:
        """A follow-up request must not inherit a poisoned socket."""
        policy = _GatedStreamPolicy(rendezvous.reached)

        def release_after_rendezvous() -> None:
            """Let the parked watchdog run on, once the caller is past it."""
            rendezvous.reached.wait(timeout=_DEADLINE)
            rendezvous.proceed.wait(timeout=0.3)
            rendezvous.proceed.set()

        releaser = threading.Thread(target=release_after_rendezvous, daemon=True)
        releaser.start()

        with ScriptedOpenAIServer(policy) as server:
            model = OpenAICompatibleModel(
                _MODEL,
                base_url=server.base_url,
                api_key="test-key",
                model_config={"stream_stall_timeout": 120.0},
                token_callback=_StopOnFirstToken(rendezvous),
            )
            model.initialize("Say something.")
            with pytest.raises(KeyboardInterrupt):
                model.generate()

            rendezvous.clear()
            stop_signal.set_thread_stop_event(None)
            model.token_callback = None
            model.conversation = [{"role": "user", "content": "again"}]
            content, _response = model.generate()
            assert content == "ok"
        releaser.join(timeout=_DEADLINE)
