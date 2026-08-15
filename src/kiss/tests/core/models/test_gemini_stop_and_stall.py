# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a quiet Gemini stream must be stoppable and must time out.

Finding **I2** of ``tmp/audit/03-core-models-c.md``: Gemini was the only
network adapter that iterated its stream bare — no
:func:`kiss.core.models.stream_abort.stop_aware_events`, no stop event,
and a ``genai.Client`` built with no timeout at all, so
``model_config["stream_stall_timeout"]`` (which every other adapter
honours) was silently ignored.  A user pressing Stop while the stream
was quiet therefore waited for the SDK's own — effectively unbounded —
streaming timeout, exactly the dead-Stop-button regression
``reports/stop_button_delay_2026-08-05.html`` documents for Anthropic.

The tests use no mocks, patches or test doubles: a real
``ThreadingHTTPServer`` speaks the genuine Gemini SSE wire format to the
real ``google-genai`` SDK (see :mod:`gemini_sse_harness`), the stop event
is bound the way ``task_runner`` binds it (by assigning
``JsonPrinter._thread_local.stop_event`` on the calling thread), and the
server then simply stops writing.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Generator

import pytest

from kiss.core.models.gemini_model import GeminiModel
from kiss.server.json_printer import JsonPrinter
from kiss.tests.core.models.gemini_sse_harness import (
    GeminiScript,
    chunk,
    serve,
    text_part,
)

_MODEL = "gemini-stop-under-test"
_DEADLINE = 20.0
# Far above the stop test's deadline, so a passing stop test can only be
# explained by the stop and never by the stall watchdog.
_UNREACHABLE_STALL = 120.0
_SHORT_STALL = 2.0


@pytest.fixture
def gemini_endpoint() -> Generator[tuple[str, GeminiScript]]:
    """A real local Gemini endpoint for one test."""
    yield from serve()


def _make_model(
    monkeypatch: pytest.MonkeyPatch, base_url: str, stall_timeout: float,
) -> GeminiModel:
    """Build a real ``GeminiModel`` whose stream goes to the local endpoint."""
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
    tokens: list[str] = []
    model = GeminiModel(
        _MODEL,
        api_key="test-key",
        model_config={"stream_stall_timeout": stall_timeout},
        token_callback=tokens.append,
    )
    model.initialize("Explain the stop button.")
    return model


def _run_in_thread(model: GeminiModel) -> tuple[threading.Thread, dict[str, object]]:
    """Start ``generate()`` on a worker thread, recording how it unwound."""
    outcome: dict[str, object] = {}

    def target() -> None:
        try:
            outcome["result"] = model.generate()
        except BaseException as exc:  # noqa: BLE001 — reported to the test
            outcome["error"] = exc
        outcome["finished_at"] = time.monotonic()

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    return worker, outcome


class TestStopAbortsQuietGeminiStream:
    """Stop must unblock a Gemini stream that has gone silent."""

    def test_stop_raises_keyboard_interrupt_quickly(
        self, monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
    ) -> None:
        """One chunk arrives, then silence; Stop must land within seconds."""
        base_url, script = gemini_endpoint
        script.play([chunk([text_part("Working on it")])], after="silent")
        model = _make_model(monkeypatch, base_url, _UNREACHABLE_STALL)

        stop_event = threading.Event()
        printer = JsonPrinter()
        outcome: dict[str, object] = {}

        def target() -> None:
            printer._thread_local.stop_event = stop_event
            try:
                outcome["result"] = model.generate()
            except BaseException as exc:  # noqa: BLE001 — reported to the test
                outcome["error"] = exc
            finally:
                printer._thread_local.stop_event = None
            outcome["finished_at"] = time.monotonic()

        worker = threading.Thread(target=target, daemon=True)
        worker.start()
        assert script.serving.wait(timeout=_DEADLINE), "server never served"
        time.sleep(0.5)
        stopped_at = time.monotonic()
        stop_event.set()

        worker.join(_DEADLINE)
        assert not worker.is_alive(), (
            f"generate() still running {_DEADLINE}s after Stop — the stop "
            f"event never reaches the Gemini stream"
        )
        assert isinstance(outcome.get("error"), KeyboardInterrupt)
        elapsed = float(outcome["finished_at"]) - stopped_at  # type: ignore[arg-type]
        assert elapsed < 5.0, f"Stop took {elapsed:.1f}s to land"


class TestStallRaisesRetryableTimeout:
    """A stalled Gemini stream must raise the retryable ``TimeoutError``."""

    def test_stall_timeout_is_honoured(
        self, monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
    ) -> None:
        """``stream_stall_timeout`` must end a silent stream, with no Stop."""
        base_url, script = gemini_endpoint
        script.play([chunk([text_part("Working on it")])], after="silent")
        model = _make_model(monkeypatch, base_url, _SHORT_STALL)

        started = time.monotonic()
        worker, outcome = _run_in_thread(model)
        worker.join(_DEADLINE)

        assert not worker.is_alive(), (
            f"generate() ignored stream_stall_timeout={_SHORT_STALL}s"
        )
        error = outcome.get("error")
        assert isinstance(error, TimeoutError), f"got {error!r}"
        assert "stall" in str(error).lower()
        elapsed = float(outcome["finished_at"]) - started  # type: ignore[arg-type]
        assert elapsed < _SHORT_STALL + 6.0, f"stall took {elapsed:.1f}s"

    def test_healthy_stream_is_unaffected(
        self, monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
    ) -> None:
        """A stream that completes normally must not be aborted."""
        base_url, script = gemini_endpoint
        script.play([chunk([text_part("All done.")], finish_reason="STOP")])
        model = _make_model(monkeypatch, base_url, _SHORT_STALL)

        content, _ = model.generate()

        assert content == "All done."
