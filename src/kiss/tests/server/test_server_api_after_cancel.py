# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A queued reply cancelled before it ran is closed, not leaked.

``ServerApi._spawn_after(pending, key, reply)`` runs *reply* behind the
task *pending* holds under *key*.  At shutdown ``cancel_background_replies``
cancels every such task; one cancelled before its first step, or while
still waiting on its predecessor, must close the inner *reply*
coroutine instead of dropping it unawaited (asyncio then prints
``RuntimeWarning: coroutine ... was never awaited`` at GC).  The tests
go through the real schedule and shutdown path on a ``ServerApi`` with
a bare backend (the constructor touches none of it).
"""

from __future__ import annotations

import asyncio
import gc
import inspect
import unittest
import warnings

from kiss.server.sorcar import ApiContext, ServerApi


class _Backend:
    """A backend ``ServerApi.__init__`` can bind to; no command reaches it here."""


async def _nothing() -> None:
    """A reply that does nothing."""


class TestAfterCancel(unittest.IsolatedAsyncioTestCase):
    """Cancelling a queued reply before it runs closes it; otherwise it runs in order."""

    def setUp(self) -> None:
        self.api = ServerApi(_Backend())  # type: ignore[arg-type]

    async def test_cancel_before_first_step_closes_reply(self) -> None:
        """Shutdown right after scheduling (the task never ran) closes the reply."""
        inner = _nothing()
        self.api._spawn_after(self.api._pending_history, "conn", inner)
        await self.api.cancel_background_replies()
        self.assertEqual(inspect.getcoroutinestate(inner), inspect.CORO_CLOSED)
        self.assertEqual(self.api._pending_history, {})

    async def test_cancel_while_waiting_closes_reply(self) -> None:
        """Shutdown while the reply waits on its predecessor closes it (never started)."""
        ran: list[str] = []

        async def reply() -> None:
            ran.append("reply")

        blocked = asyncio.get_running_loop().create_future()

        async def predecessor() -> None:
            await blocked

        inner = reply()
        self.api._spawn_after(self.api._pending_history, "conn", predecessor())
        self.api._spawn_after(self.api._pending_history, "conn", inner)
        await asyncio.sleep(0)  # both tasks take their first step
        self.assertEqual(inspect.getcoroutinestate(inner), inspect.CORO_CREATED)
        await self.api.cancel_background_replies()
        self.assertEqual(inspect.getcoroutinestate(inner), inspect.CORO_CLOSED)
        self.assertEqual(ran, [])
        self.assertEqual(self.api._pending_history, {})

    async def test_replies_run_in_order_after_predecessor(self) -> None:
        """Without a cancel the replies of one key run one after the other."""
        ran: list[str] = []
        release = asyncio.get_running_loop().create_future()

        async def first() -> None:
            await release
            ran.append("first")

        async def second() -> None:
            ran.append("second")

        self.api._spawn_after(self.api._pending_history, "conn", first())
        self.api._spawn_after(self.api._pending_history, "conn", second())
        await asyncio.sleep(0)
        self.assertEqual(ran, [])
        release.set_result(None)
        await asyncio.gather(*self.api._background_replies)
        self.assertEqual(ran, ["first", "second"])
        self.assertEqual(self.api._pending_history, {})

    async def test_cancelled_remote_history_write_leaves_nothing_unawaited(self) -> None:
        """A machine's history write cancelled before it ran is closed whole.

        ``dispatch`` queues a ``MACHINE_HISTORY_WRITES`` command stamped
        with a machine as a reply that forwards it and then nudges every
        page to refetch.  Cancelled before its first step, or while queued
        behind a predecessor, the reply is closed unrun; it must not have
        wrapped a second coroutine, created in advance, that the close
        cannot reach (the GC then warned ``_forwarded_reply ... was never
        awaited``).  No machine is contacted: nothing here runs.
        """
        cmd = {"type": "setFavorite", "taskId": "task-1", "isFavorite": True, "machine": "box"}
        blocked = asyncio.get_running_loop().create_future()

        async def predecessor() -> None:
            await blocked

        def context(conn_id: str) -> ApiContext:
            return ApiContext(endpoint=None, conn_state={"conn_id": conn_id}, is_local=True)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            self.api._spawn_after(self.api._pending_actions, ("queued", "box"), predecessor())
            await self.api.dispatch(dict(cmd), context("queued"))
            await asyncio.sleep(0)  # the write now waits on its predecessor
            await self.api.dispatch(dict(cmd), context("fresh"))  # never takes a step
            await self.api.cancel_background_replies()
            await asyncio.sleep(0)
            gc.collect()
        self.assertEqual(
            [str(w.message) for w in caught if "never awaited" in str(w.message)], [],
        )
        self.assertEqual(self.api._pending_actions, {})


if __name__ == "__main__":
    unittest.main()
