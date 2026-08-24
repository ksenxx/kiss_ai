# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for slack_agent's shared retry and cursor helpers (G-R2).

``poll_messages`` and ``poll_thread_messages`` used to carry verbatim
copies of the 3-attempt OSError backoff loop and the trailing
cursor-advance loop; both now live in module-local helpers
``_call_with_retry`` and ``_advance_cursor``.  These tests drive the
helpers with real callables — no mocks, patches, or test doubles.
"""

from __future__ import annotations

import time

import pytest

from kiss.agents.third_party_agents.slack_agent import _advance_cursor, _call_with_retry


class FlakyCall:
    """Real callable that raises OSError for its first *failures* calls."""

    def __init__(self, failures: int, result: str = "ok") -> None:
        self.failures = failures
        self.result = result
        self.calls = 0

    def __call__(self) -> str:
        """Raise OSError while failures remain, then return the result."""
        self.calls += 1
        if self.calls <= self.failures:
            raise OSError(f"transient error #{self.calls}")
        return self.result


class TestCallWithRetry:
    """3-attempt OSError retry loop shared by both slack poll methods."""

    def test_success_first_attempt(self) -> None:
        """A call that succeeds immediately is invoked exactly once."""
        fn = FlakyCall(failures=0, result="resp")
        assert _call_with_retry(fn, "messages") == "resp"
        assert fn.calls == 1

    def test_recovers_after_transient_failures(self) -> None:
        """Two OSErrors are retried with backoff; the third attempt wins."""
        fn = FlakyCall(failures=2, result="resp")
        start = time.monotonic()
        assert _call_with_retry(fn, "thread replies") == "resp"
        elapsed = time.monotonic() - start
        assert fn.calls == 3
        # Backoff slept 2**0 + 2**1 = 3 seconds between the attempts.
        assert elapsed >= 3.0

    def test_raises_after_three_failures(self) -> None:
        """The last OSError propagates when every attempt fails."""
        fn = FlakyCall(failures=5)
        with pytest.raises(OSError, match="transient error #3"):
            _call_with_retry(fn, "messages")
        assert fn.calls == 3

    def test_non_oserror_propagates_immediately(self) -> None:
        """Only OSError is retried; other exceptions escape on attempt one."""

        def boom() -> None:
            raise ValueError("not transient")

        with pytest.raises(ValueError, match="not transient"):
            _call_with_retry(boom, "messages")


class TestAdvanceCursor:
    """max-ts + 1 microsecond cursor advance shared by both poll methods."""

    def test_empty_messages_keep_cursor(self) -> None:
        """No messages leave the cursor unchanged."""
        assert _advance_cursor([], "1700000000.000000") == "1700000000.000000"

    def test_advances_past_newest_message(self) -> None:
        """The cursor lands one microsecond past the newest timestamp."""
        messages = [{"ts": "100.000000"}, {"ts": "200.000000"}, {"ts": "150.000000"}]
        assert _advance_cursor(messages, "0") == "200.000001"

    def test_messages_older_than_cursor_are_ignored(self) -> None:
        """Timestamps below the cursor never move it backwards."""
        messages = [{"ts": "100.000000"}]
        assert _advance_cursor(messages, "500.000000") == "500.000000"

    def test_message_equal_to_cursor_advances(self) -> None:
        """A ts exactly at the cursor still advances by one microsecond."""
        messages = [{"ts": "500.000000"}]
        assert _advance_cursor(messages, "500.000000") == "500.000001"

    def test_missing_ts_treated_as_zero(self) -> None:
        """Messages without a ts field count as timestamp 0."""
        assert _advance_cursor([{}], "0") == "0.000001"
