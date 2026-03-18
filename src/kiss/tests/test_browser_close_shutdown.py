"""Tests for process shutdown when browser window closes.

Verifies that the sorcar process terminates after the browser disconnects,
via three mechanisms:
1. The /closing endpoint (called by beforeunload beacon)
2. The periodic no-client safety net (_watch_no_clients)
3. The SSE disconnect detection scheduling shutdown
"""




# ---------------------------------------------------------------------------
# kiss/agents/sorcar/sorcar.py — sorcar
# ---------------------------------------------------------------------------

class TestShutdownTimerDuration:
    """The shutdown timer should use a short delay for quick exit."""

    def test_timer_is_short(self) -> None:
        """Verify the source code uses a 1-second timer for fast shutdown."""
        import inspect

        from kiss.agents.sorcar import sorcar

        source = inspect.getsource(sorcar.run_chatbot)
        # The timer should be 1 second for fast shutdown on browser close
        assert "call_later(1.0," in source
        assert "call_later(10.0," not in source
        assert "Timer(120.0," not in source

    def test_no_client_safety_net_is_short(self) -> None:
        """Verify the no-client safety net uses a 2-second threshold."""
        import inspect

        from kiss.agents.sorcar import sorcar

        source = inspect.getsource(sorcar.run_chatbot)
        assert "no_client_since >= 2.0" in source
        assert "no_client_since >= 10.0" not in source
