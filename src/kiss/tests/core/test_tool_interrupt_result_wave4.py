# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""``result_interruptible`` returns the outcome of a future that finishes as a wait slice expires.

Review round 2 (Q1_core).  ``future.result(timeout=slice)`` raises
``TimeoutError`` when the slice expires, and the future can finish before
the helper then looks at ``future.done()``.  The helper re-raised the
slice's ``TimeoutError`` for such a future, even with no deadline at all:
a memory search whose worker finished at a slice boundary failed with
``TimeoutError`` (``memory_search``, ``memory_pull``) or recalled nothing
(``recall``), and a worker's own exception or a cancellation came out as
a timeout too.

The interleaving is forced with a ``sys.settrace`` hook, not a mock: when
the slice's ``TimeoutError`` reaches the helper's frame, the hook lets a
real worker thread finish the real ``Future`` and waits until it has, so
the helper's done check runs right after.  Nothing under test is replaced.
"""

import sys
import threading
from collections.abc import Callable
from concurrent.futures import CancelledError, Future
from types import FrameType
from typing import Any

import pytest

from kiss.core.tool_interrupt import result_interruptible


def _succeed(future: Future[str]) -> None:
    """Finish *future* with a search hit."""
    future.set_result("successful search hit")


def _fail(future: Future[str]) -> None:
    """Finish *future* with the worker's own error."""
    future.set_exception(ValueError("the worker failed"))


def _cancel(future: Future[str]) -> None:
    """Cancel *future* before it ran."""
    future.cancel()


class _FinishAtSliceTimeout:
    """Trace hook that finishes a future the moment a wait slice on it times out.

    A worker thread waits for :attr:`release`; the hook sets it when a
    ``TimeoutError`` reaches a ``result_interruptible`` frame while the
    future is still pending, then waits until the worker has finished it.
    """

    def __init__(self, future: Future[str], finish: Callable[[Future[str]], None]) -> None:
        self.future = future
        self.finish = finish
        self.release = threading.Event()
        self.finished = threading.Event()
        self.slice_timeouts = 0
        self.worker = threading.Thread(target=self._work, name="finish-at-slice-timeout")

    def _work(self) -> None:
        """Finish the future once released."""
        if self.release.wait(10):
            self.finish(self.future)
            self.finished.set()

    def trace(self, frame: FrameType, event: str, arg: Any) -> Any:
        """Global tracer: trace only ``result_interruptible``'s frames."""
        return self.local if frame.f_code is result_interruptible.__code__ else None

    def local(self, frame: FrameType, event: str, arg: Any) -> Any:
        """Local tracer: at a slice's ``TimeoutError``, let the worker finish the future."""
        if event == "exception" and issubclass(arg[0], TimeoutError) and not self.future.done():
            self.slice_timeouts += 1
            self.release.set()
            if not self.finished.wait(10):
                raise RuntimeError("the worker did not finish the future")
        return self.local


def _wait_with_finish_at_slice_timeout(
    finish: Callable[[Future[str]], None], timeout: float | None
) -> tuple[Any, int]:
    """Run ``result_interruptible`` while the future finishes at its first slice timeout.

    Returns:
        ``("returned", value)`` or ``("raised", exception)``, and how many
        slice timeouts the hook saw on the pending future.
    """
    future: Future[str] = Future()
    hook = _FinishAtSliceTimeout(future, finish)
    hook.worker.start()
    previous = sys.gettrace()
    sys.settrace(hook.trace)
    try:
        outcome: Any = ("returned", result_interruptible(future, timeout))
    except Exception as error:
        outcome = ("raised", error)
    finally:
        sys.settrace(previous)
        hook.release.set()
        hook.worker.join(10)
    assert not hook.worker.is_alive()
    return outcome, hook.slice_timeouts


@pytest.mark.parametrize("timeout", [None, 30.0], ids=["no-deadline", "deadline"])
def test_a_result_landing_as_the_slice_expires_is_returned(timeout: float | None) -> None:
    """The successful result comes back, not the slice's TimeoutError."""
    outcome, slice_timeouts = _wait_with_finish_at_slice_timeout(_succeed, timeout)

    assert slice_timeouts == 1, "the future finished at the first slice timeout"
    assert outcome == ("returned", "successful search hit")


def test_a_worker_error_landing_as_the_slice_expires_is_raised() -> None:
    """The worker's own exception propagates, not a TimeoutError."""
    (kind, error), slice_timeouts = _wait_with_finish_at_slice_timeout(_fail, None)

    assert slice_timeouts == 1
    assert kind == "raised"
    assert type(error) is ValueError and str(error) == "the worker failed"


def test_a_cancellation_landing_as_the_slice_expires_is_raised() -> None:
    """A future cancelled at the slice boundary raises CancelledError, not TimeoutError."""
    (kind, error), slice_timeouts = _wait_with_finish_at_slice_timeout(_cancel, None)

    assert slice_timeouts == 1
    assert kind == "raised"
    assert isinstance(error, CancelledError)
