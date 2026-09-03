# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Gemini stall must be reported in the one wording every transport shares.

:mod:`kiss.core.models.stream_abort` documents ``stall_error`` as the
single source of the stall message: an adapter that reaches the same
condition by another route "still has to raise the same error for the
same condition, so the wording lives here once".  ``GeminiModel``
reaches it by two routes — the event-level watchdog inside
``stop_aware_events`` and httpx's byte-level ``ReadTimeout`` — and the
second one used to build its own message, so which text a user (or a
log grep) saw for an identical stall depended on which clock happened
to fire first.

The test drives a REAL ``GeminiModel`` against a real local HTTP
endpoint that goes silent after one chunk.  The stall timeout is chosen
so the byte-level clock wins: with 4.5 s the watchdog polls once a
second (``_poll_interval`` caps at 1.0 s) and can only notice the stall
at the 5 s poll, while httpx raises at 4.5 s.  Which path fired is
decided by a semantic marker, not by the clock: the byte-level branch
raises ``stall_error(...) from e`` with the ``httpx.TimeoutException``
as ``__cause__``, while the watchdog raises without a cause.  Only a
generous hang deadline remains as a timing bound.
"""

import threading
from collections.abc import Generator

import httpx
import pytest

from kiss.core.models.gemini_model import GeminiModel
from kiss.core.models.stream_abort import stall_error
from kiss.tests.core.models.gemini_sse_harness import (
    GeminiScript,
    chunk,
    serve,
    text_part,
)

_STALL = 4.5
_DEADLINE = 20.0


@pytest.fixture
def gemini_endpoint() -> Generator[tuple[str, GeminiScript]]:
    """A real local Gemini endpoint for one test."""
    yield from serve()


def test_byte_level_stall_uses_the_shared_stall_wording(
    monkeypatch: pytest.MonkeyPatch, gemini_endpoint: tuple[str, GeminiScript]
) -> None:
    base_url, script = gemini_endpoint
    script.play([chunk([text_part("Working on it"),])], after="silent")
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
    model = GeminiModel(
        "gemini-stall-wording-under-test",
        api_key="test-key",
        model_config={"stream_stall_timeout": _STALL},
        token_callback=lambda _token: None,
    )
    model.initialize("Explain the stall.")

    outcome: dict[str, object] = {}

    def target() -> None:
        try:
            outcome["result"] = model.generate()
        except BaseException as exc:  # noqa: BLE001 — reported to the test
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(_DEADLINE)

    assert not worker.is_alive(), f"generate() ignored stream_stall_timeout={_STALL}s"
    error = outcome.get("error")
    assert isinstance(error, TimeoutError), f"got {outcome!r}"
    assert isinstance(error.__cause__, httpx.TimeoutException), (
        f"cause {error.__cause__!r}: the watchdog beat httpx, so this run "
        "did not exercise the byte-level path"
    )
    assert str(stall_error(_STALL)) in str(error), str(error)
