# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the matrix_agent.py event-loop bug.

Covered bug (reproduced end-to-end, no mock/patch libraries):

The Matrix backend drove one shared nio ``AsyncClient`` through a fresh
``asyncio.run()`` per method, so every call after the first failed with
"Event loop is closed".  The fix routes all coroutines through one
persistent background event loop; these tests exercise that loop's full
lifecycle (two successive calls, disconnect, lazy restart) without nio.
"""

from __future__ import annotations

import asyncio

from kiss.agents.third_party_agents.matrix_agent import MatrixChannelBackend


class TestMatrixPersistentLoop:
    """Matrix backend must reuse one event loop across calls."""

    def test_two_successive_calls_share_one_loop(self) -> None:
        """Two successive coroutine runs succeed and reuse the same loop."""
        backend = MatrixChannelBackend()
        try:
            assert backend._run(asyncio.sleep(0, result="first")) == "first"
            loop = backend._loop
            assert loop is not None
            assert backend._run(asyncio.sleep(0, result="second")) == "second"
            assert backend._loop is loop
            assert loop.is_running()
        finally:
            backend.disconnect()

    def test_disconnect_stops_loop_and_allows_restart(self) -> None:
        """disconnect() stops the loop; a later call lazily restarts it."""
        backend = MatrixChannelBackend()
        assert backend._run(asyncio.sleep(0, result=1)) == 1
        loop = backend._loop
        thread = backend._loop_thread
        assert loop is not None and thread is not None
        backend.disconnect()
        assert backend._loop is None
        assert backend._loop_thread is None
        assert not thread.is_alive()
        assert not loop.is_running()
        try:
            assert backend._run(asyncio.sleep(0, result=7)) == 7
        finally:
            backend.disconnect()

    def test_coroutine_exceptions_propagate(self) -> None:
        """Exceptions raised inside coroutines surface to the caller."""
        backend = MatrixChannelBackend()

        async def _boom() -> None:
            raise ValueError("boom")

        try:
            try:
                backend._run(_boom())
            except ValueError as e:
                assert str(e) == "boom"
            else:  # pragma: no cover - failure path
                raise AssertionError("expected ValueError")
            assert backend._run(asyncio.sleep(0, result="ok")) == "ok"
        finally:
            backend.disconnect()
