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

import threading
import time
from collections.abc import Generator

import pytest

from kiss.core.models.gemini_model import GeminiModel
from kiss.tests.core.models.gemini_sse_harness import (
    GeminiScript,
    chunk,
    serve,
    text_part,
)

_MODEL = "gemini-keepalive-under-test"
_STALL_TIMEOUT = 2.0
# Comfortably above the stall timeout, and far below the time an
# unbounded keep-alive stream would take (it never ends).
_DEADLINE = 20.0


@pytest.fixture
def gemini_endpoint() -> Generator[tuple[str, GeminiScript]]:
    """A real local Gemini endpoint for one test."""
    yield from serve()


def _make_model(monkeypatch: pytest.MonkeyPatch, base_url: str) -> GeminiModel:
    """Build a real ``GeminiModel`` whose stream goes to the local endpoint."""
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
    model = GeminiModel(
        _MODEL,
        api_key="test-key",
        model_config={"stream_stall_timeout": _STALL_TIMEOUT},
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
