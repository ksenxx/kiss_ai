# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a failed Responses stream must not leak its watchdog.

Audit finding 02/C2.  ``stop_aware_events`` is a generator whose cleanup
— stopping the abort watchdog thread and releasing the HTTP response —
lives in a ``finally`` that runs only when the generator is exhausted,
closed, or garbage collected.  ``_consume_stream`` raised ``KISSError``
from *inside* its ``for event in stop_aware_events(...)`` loop on the two
most common provider failures (``response.failed`` / ``error`` and
``response.incomplete``), abandoning the generator.  The propagating
exception's traceback keeps the generator frame alive, so until the
traceback is released a daemon watchdog thread keeps polling every 0.1 s
— still armed, still holding the stream — and the connection is never
returned to httpx's pool.

No mocks: a real ``ThreadingHTTPServer`` streams genuine Responses-API
SSE events to the real OpenAI SDK, then holds the connection open with
keep-alive comments so the leak is observable both as a live thread and
as a connection the server never sees closed.
"""

from __future__ import annotations

import threading
from collections.abc import Generator

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    responses_event,
)

_MODEL = "gpt-responses-failure-under-test"
_WATCHDOG_MARKER = "abort-watchdog"

_PREFIX = [
    responses_event(
        "response.created",
        {"response": {"id": "resp_fail", "status": "in_progress", "output": []}},
    ),
    responses_event(
        "response.output_text.delta",
        {
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "partial",
        },
    ),
]

_FAILED = responses_event(
    "response.failed",
    {
        "response": {
            "id": "resp_fail",
            "status": "failed",
            "error": {"code": "server_error", "message": "upstream exploded"},
            "output": [],
        }
    },
)

_INCOMPLETE = responses_event(
    "response.incomplete",
    {
        "response": {
            "id": "resp_fail",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [],
        }
    },
)


class _WedgedFailurePolicy:
    """Streams a partial response, a terminal failure, then goes quiet."""

    def __init__(self, terminal: bytes) -> None:
        self.chunks = [*_PREFIX, terminal]
        self.release = threading.Event()

    def __call__(self, request: Request) -> Reply:
        """Return the scripted failing stream, holding the socket open."""
        return Reply(sse_chunks=self.chunks, hold=self.release)


def _watchdog_threads() -> set[str]:
    """Return the names of the live stream-abort watchdog threads."""
    return {
        f"{thread.name}#{thread.ident}"
        for thread in threading.enumerate()
        if _WATCHDOG_MARKER in thread.name
    }


def _make_model(base_url: str) -> OpenAICompatibleModel2:
    """Build a real streaming Responses model pointed at *base_url*.

    Args:
        base_url: The scripted server's ``/v1`` root.

    Returns:
        An initialized model whose ``token_callback`` forces streaming.
    """
    model = OpenAICompatibleModel2(
        _MODEL,
        base_url=base_url,
        api_key="test-key",
        # Far above the test's own bound: only the failure event may end
        # these calls, never the stall watchdog.
        model_config={"stream_stall_timeout": 120.0},
        token_callback=lambda _t: None,
    )
    model.initialize("Say something.")
    return model


@pytest.fixture
def failed_server() -> Generator[tuple[ScriptedOpenAIServer, _WedgedFailurePolicy]]:
    """A Responses endpoint whose stream ends with ``response.failed``."""
    policy = _WedgedFailurePolicy(_FAILED)
    with ScriptedOpenAIServer(policy) as server:
        yield server, policy
        policy.release.set()


@pytest.fixture
def incomplete_server() -> Generator[
    tuple[ScriptedOpenAIServer, _WedgedFailurePolicy]
]:
    """A Responses endpoint whose stream ends with ``response.incomplete``."""
    policy = _WedgedFailurePolicy(_INCOMPLETE)
    with ScriptedOpenAIServer(policy) as server:
        yield server, policy
        policy.release.set()


class TestFailedResponsesStreamCleanup:
    """Raising from inside the event loop must still run the cleanup."""

    def test_response_failed_leaves_no_watchdog_thread(
        self,
        failed_server: tuple[ScriptedOpenAIServer, _WedgedFailurePolicy],
    ) -> None:
        """``response.failed`` must not strand the watchdog or the socket.

        The assertions run while ``excinfo`` still holds the traceback —
        exactly the state the agent's retry logic is in when it catches
        the error and issues the next request.
        """
        server, _policy = failed_server
        baseline = _watchdog_threads()
        model = _make_model(server.base_url)

        with pytest.raises(KISSError) as excinfo:
            model.generate()

        assert "upstream exploded" in str(excinfo.value)
        assert _watchdog_threads() == baseline, (
            "a stream-abort watchdog outlived the failed stream while its "
            "traceback was still referenced"
        )
        assert server.client_disconnected.wait(timeout=5.0), (
            "the failed stream was never closed, so its connection was "
            "never returned to the pool"
        )

    def test_response_incomplete_leaves_no_watchdog_thread(
        self,
        incomplete_server: tuple[ScriptedOpenAIServer, _WedgedFailurePolicy],
    ) -> None:
        """The ``response.incomplete`` raise site needs the same cleanup."""
        server, _policy = incomplete_server
        baseline = _watchdog_threads()
        model = _make_model(server.base_url)

        with pytest.raises(KISSError) as excinfo:
            model.generate()

        assert "max_output_tokens" in str(excinfo.value)
        assert _watchdog_threads() == baseline
        assert server.client_disconnected.wait(timeout=5.0)
