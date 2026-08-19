# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Contract tests for ``stop_aware_events`` when the iterator ends at EOF.

Audit finding 01/F1.  ``stop_aware_events`` reports a stop *twice* — once
from the ``except`` arm, for the common case where the aborted socket
surfaces as a transport error, and once after the loop, because "an
aborted socket usually just ends the iterator" at EOF instead of raising.
Only the first arm knew about stalls, so a stall that ended the iterator
cleanly was returned to the caller as a successful (silently truncated)
completion.  Both arms now report both conditions, and this suite covers
the EOF half of that contract.

The stream here is a genuine Python iterator rather than a provider
socket: ``stop_aware_events`` is a public wrapper over *any* event
iterable, and iterating one that simply runs out is the only way to
reproduce EOF deterministically — a real HTTP stream aborted mid-body
always raises instead.  Nothing is mocked, patched or faked: the
watchdog, its thread, the stop event and the timings are all real, and
the abort genuinely runs against the object it was given.
"""

from __future__ import annotations

import threading
from collections.abc import Generator

import pytest

from kiss.core import stop_signal
from kiss.core.models.stream_abort import stop_aware_events
from kiss.tests.core.models.test_stop_aware_events_contract import (  # noqa: F401
    _EVENTS,
    _failing_events,
    _Recorder,
)


@pytest.fixture
def bound_stop_event() -> Generator[threading.Event]:
    """Bind a fresh stop event to this thread and clear it afterwards."""
    event = threading.Event()
    stop_signal.set_thread_stop_event(event)
    yield event
    stop_signal.set_thread_stop_event(None)


class TestStopReportedAfterAQuietEnd:
    """A stop must be reported even when the iterator merely runs out."""

    def test_stop_after_the_last_event(
        self, bound_stop_event: threading.Event
    ) -> None:
        """Setting the stop while iterating must not look like success."""
        on_abort = _Recorder()
        seen: list[str] = []
        with pytest.raises(KeyboardInterrupt):
            for event in stop_aware_events(iter(_EVENTS), on_abort=on_abort):
                seen.append(event)
                bound_stop_event.set()
        assert seen == _EVENTS
        assert on_abort.calls == 1

    def test_stop_without_an_abort_hook(
        self, bound_stop_event: threading.Event
    ) -> None:
        """The hook is optional; the stop must still be reported."""
        with pytest.raises(KeyboardInterrupt):
            for _event in stop_aware_events(iter(_EVENTS)):
                bound_stop_event.set()


class TestFailingStreams:
    """A stream that raises must be classified, not blindly re-raised."""

    def test_stop_wins_over_the_transport_error(
        self, bound_stop_event: threading.Event
    ) -> None:
        """A stop requested before the failure surfaces as the stop."""
        with pytest.raises(KeyboardInterrupt):
            for _event in stop_aware_events(_failing_events()):
                bound_stop_event.set()
