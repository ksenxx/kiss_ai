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
from kiss.tests.core.models.openai_sse_harness import (
    ScriptedOpenAIServer,
)
from kiss.tests.core.models.test_stream_watchdog_late_abort import (  # noqa: F401
    _DONE,
    _MODEL,
    _SECOND_REPLY,
    _USAGE_CHUNK,
    _content_chunk,
    _GatedStreamPolicy,
)

_WATCHDOG_MARKER = "abort-watchdog"


_DEADLINE = 30.0


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
