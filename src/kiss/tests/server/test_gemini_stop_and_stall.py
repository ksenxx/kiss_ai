# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `gemini_endpoint` module fixture is imported from
# the core/models twin of this file and injected via test parameters)
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

import pytest

from kiss.server.json_printer import JsonPrinter
from kiss.tests.core.models.gemini_sse_harness import (
    GeminiScript,
    chunk,
    text_part,
)
from kiss.tests.core.models.test_gemini_stop_and_stall import (  # noqa: F401
    _DEADLINE,
    _MODEL,
    _make_model,
    gemini_endpoint,
)

_UNREACHABLE_STALL = 120.0


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
