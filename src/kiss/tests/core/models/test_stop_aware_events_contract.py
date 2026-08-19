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

import time
from collections.abc import Iterator

import pytest

from kiss.core.models.stream_abort import stop_aware_events

_EVENTS = ["alpha", "beta", "gamma"]


class _Recorder:
    """Records that the abort hook ran."""

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.calls = 0

    def __call__(self) -> None:
        """Record one call."""
        self.calls += 1


def _failing_events() -> Iterator[str]:
    """Yield one event and then fail the way a dropped socket does.

    Yields:
        A single event before the failure.
    """
    yield _EVENTS[0]
    raise ConnectionResetError("peer went away")


class TestFailingStreams:
    """A stream that raises must be classified, not blindly re-raised."""

    def test_stall_wins_over_the_transport_error(self) -> None:
        """A stalled stream that then fails is still a retryable stall."""
        with pytest.raises(TimeoutError, match="stream_stall_timeout"):
            for _event in stop_aware_events(
                _failing_events(), stall_timeout=0.3
            ):
                time.sleep(1.0)

    def test_an_unrelated_failure_propagates(self) -> None:
        """Without a stop or a stall the original error must survive."""
        with pytest.raises(ConnectionResetError, match="peer went away"):
            for _event in stop_aware_events(_failing_events()):
                pass


class TestStallReportedAfterAQuietEnd:
    """A stall must not be reported as a truncated success."""

    def test_stall_detected_while_the_consumer_is_busy(self) -> None:
        """A consumer slower than the stall window loses the stream."""
        on_abort = _Recorder()
        seen: list[str] = []
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="stream_stall_timeout"):
            for event in stop_aware_events(
                iter(_EVENTS), stall_timeout=0.3, on_abort=on_abort
            ):
                seen.append(event)
                if len(seen) == 1:
                    time.sleep(1.0)
        assert time.monotonic() - started < 10.0
        assert on_abort.calls == 1

    def test_stall_without_an_abort_hook(self) -> None:
        """The hook is optional; the stall must still be reported."""
        seen: list[str] = []
        with pytest.raises(TimeoutError, match="stream_stall_timeout"):
            for event in stop_aware_events(iter(_EVENTS), stall_timeout=0.3):
                seen.append(event)
                if len(seen) == 1:
                    time.sleep(1.0)

    def test_a_prompt_consumer_is_left_alone(self) -> None:
        """Heartbeats from a keeping-up consumer prevent any stall."""
        seen = list(stop_aware_events(iter(_EVENTS), stall_timeout=5.0))
        assert seen == _EVENTS
