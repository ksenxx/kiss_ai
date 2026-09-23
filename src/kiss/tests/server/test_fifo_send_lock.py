# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end behaviour of :class:`kiss.server.web_server.FifoSendLock`.

The lock replaced ``asyncio.Lock`` as the per-endpoint send lock after a
benchmark daemon with 60 streaming tasks wedged in ``deque.remove`` inside
``asyncio.Lock.acquire`` (O(n) per woken waiter, O(n^2) per backlog).
"""

import asyncio
import time
import unittest

from kiss.server.web_server import FifoSendLock, WebPrinter


class FifoSendLockTests(unittest.TestCase):
    """FIFO order, cancellation safety and linear cost of the send lock."""

    def test_waiters_run_in_fifo_order(self) -> None:
        """Contenders acquire the lock in the order they started waiting."""

        async def scenario() -> list[int]:
            lock = FifoSendLock()
            order: list[int] = []

            async def contender(i: int) -> None:
                async with lock:
                    order.append(i)
                    await asyncio.sleep(0)

            await lock.acquire()
            self.assertTrue(lock.locked())
            tasks = [asyncio.create_task(contender(i)) for i in range(50)]
            await asyncio.sleep(0)  # every contender is now queued
            lock.release()
            await asyncio.gather(*tasks)
            self.assertFalse(lock.locked())
            return order

        self.assertEqual(asyncio.run(scenario()), list(range(50)))

    def test_cancelled_waiter_is_skipped_and_handed_over_lock_is_passed_on(self) -> None:
        """Cancelling a queued waiter, or one already handed the lock, never loses the lock."""

        async def scenario() -> list[str]:
            lock = FifoSendLock()
            log: list[str] = []

            async def contender(name: str) -> None:
                async with lock:
                    log.append(name)

            await lock.acquire()
            queued = asyncio.create_task(contender("queued-cancelled"))
            handed = asyncio.create_task(contender("handed-cancelled"))
            survivor = asyncio.create_task(contender("survivor"))
            await asyncio.sleep(0)
            queued.cancel()  # cancelled while still waiting in the deque
            await asyncio.sleep(0)
            lock.release()  # ownership goes to ``handed``, which has not resumed yet
            handed.cancel()  # cancelled after the hand-over, before it runs
            await asyncio.gather(queued, handed, return_exceptions=True)
            await survivor
            self.assertTrue(queued.cancelled())
            self.assertTrue(handed.cancelled())
            self.assertFalse(lock.locked())
            return log

        self.assertEqual(asyncio.run(scenario()), ["survivor"])

    def test_large_backlog_drains_in_linear_time(self) -> None:
        """A backlog of 20k queued sends drains in well under a second."""

        async def scenario() -> float:
            lock = FifoSendLock()
            done = 0

            async def contender() -> None:
                nonlocal done
                async with lock:
                    done += 1

            await lock.acquire()
            tasks = [asyncio.create_task(contender()) for _ in range(20_000)]
            await asyncio.sleep(0)
            started = time.monotonic()
            lock.release()
            await asyncio.gather(*tasks)
            self.assertEqual(done, 20_000)
            return time.monotonic() - started

        self.assertLess(asyncio.run(scenario()), 2.0)

    def test_printer_send_lock_is_fifo_lock_per_tracked_endpoint(self) -> None:
        """``WebPrinter.send_lock`` returns one stored FifoSendLock per pending-send endpoint."""

        async def scenario() -> None:
            printer = WebPrinter()
            endpoint = object()
            untracked = printer.send_lock(endpoint)
            self.assertIsInstance(untracked, FifoSendLock)
            self.assertNotIn(endpoint, printer._send_locks)
            printer._pending_sends[endpoint] = set()
            lock = printer.send_lock(endpoint)
            self.assertIs(lock, printer.send_lock(endpoint))
            async with lock:
                self.assertTrue(lock.locked())
            self.assertFalse(lock.locked())

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
