"""Tests for process shutdown when browser window closes.

Verifies that the sorcar process terminates after the browser disconnects,
via three mechanisms:
1. The /closing endpoint (called by beforeunload beacon)
2. The periodic no-client safety net (_watch_no_clients)
3. The SSE disconnect detection scheduling shutdown
"""




class TestShutdownTimerDuration:
    """The shutdown timer should use a short delay (not the old 120s)."""

    def test_timer_is_short(self) -> None:
        """Verify the source code uses a short timer, not 120 seconds."""
        import inspect

        from kiss.agents.sorcar import sorcar

        source = inspect.getsource(sorcar.run_chatbot)
        # The timer should be 10 seconds, not 120
        assert "call_later(10.0," in source
        assert "Timer(120.0," not in source
