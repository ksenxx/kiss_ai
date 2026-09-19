# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""An accepted Stop must be enforced even when its watchdog thread cannot start.

``_TaskRunnerMixin._stop_task`` (task_runner.py) acknowledges the Stop
to the UI (``stop_ack accepted=True``), sets the cooperative stop
event, and then starts the watchdog thread that injects
``KeyboardInterrupt`` into a task thread which does not honour the
event.  That ``Thread.start()`` was unguarded: under thread exhaustion
it raised ``RuntimeError: can't start new thread`` out of the command
handler, and a task blocked in a non-cooperative call — which is
exactly the case the watchdog exists for — was never interrupted,
although the user had already been told the click landed.

Reproduced for real: a registered task whose worker thread ignores the
stop event, and ``RLIMIT_NPROC`` lowered to 1 around the ``_stop_task``
call so the watchdog's ``Thread.start`` genuinely fails (threads count
against the limit for a non-root uid; the technique of
``test_concaudit_w6_commit_msg_claim.py``).  With the fix the stop is
enforced inline: the worker receives the interrupt and exits.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import pytest

from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import VSCodeServer
from kiss.tests.conftest import is_root
from kiss.tests.server._memory_printer import MemoryPrinter

# The resource module (RLIMIT_*) only exists on POSIX; Windows skips.
resource = pytest.importorskip("resource")


def _thread_start_can_be_starved() -> bool:
    """True when lowering RLIMIT_NPROC actually makes Thread.start fail here."""
    if is_root():
        return False
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (1, hard))
    except (ValueError, OSError):
        return False
    try:
        probe = threading.Thread(target=lambda: None)
        try:
            probe.start()
        except RuntimeError:
            return True
        probe.join()
        return False
    finally:
        resource.setrlimit(resource.RLIMIT_NPROC, (soft, hard))


@pytest.fixture
def clean_registry() -> Iterator[None]:
    """Run with an empty agent-state registry and leave it empty."""
    agent_state.agent_states.clear()
    try:
        yield
    finally:
        agent_state.agent_states.clear()


class _StubbornWorker:
    """A real thread that ignores the cooperative stop event.

    It spins in Python (so an injected ``KeyboardInterrupt`` lands at
    the next bytecode) and records whether it was interrupted.
    """

    def __init__(self) -> None:
        self.interrupted = threading.Event()
        self.finished = threading.Event()
        self.thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self) -> None:
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                time.sleep(0.005)
        except KeyboardInterrupt:
            self.interrupted.set()
        finally:
            self.finished.set()


def test_stop_is_enforced_inline_when_watchdog_thread_cannot_start(
    clean_registry: None,
) -> None:
    """A watchdog spawn failure still interrupts the non-cooperative worker."""
    if not _thread_start_can_be_starved():
        pytest.skip("RLIMIT_NPROC cannot starve Thread.start on this host")
    printer = MemoryPrinter()
    server = VSCodeServer(printer=printer)
    tab_id = "TAB-F2-STOP"
    worker = _StubbornWorker()
    state = AgentState(
        "f2-stop-task",
        tab_id=tab_id,
        server_owned=True,
        stop_event=threading.Event(),
        task_thread=worker.thread,
    )
    agent_state.register(state)
    worker.thread.start()

    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    resource.setrlimit(resource.RLIMIT_NPROC, (1, hard))
    try:
        t0 = time.monotonic()
        server._stop_task(tab_id)
        elapsed = time.monotonic() - t0
    finally:
        resource.setrlimit(resource.RLIMIT_NPROC, (soft, hard))

    assert state.stop_event is not None and state.stop_event.is_set()
    assert worker.finished.wait(15), "the worker was never interrupted"
    assert worker.interrupted.is_set(), "the worker exited without an interrupt"
    # The inline watchdog waits 1 s for the cooperative stop, then
    # injects once; it must not sit through its 5 s retry either.
    assert elapsed < 10, elapsed
    acks = [e for e in printer.emitted if e.get("type") == "stop_ack"]
    assert acks and acks[-1].get("accepted") is True, printer.emitted


def test_stop_uses_the_watchdog_thread_when_it_can_start(
    clean_registry: None,
) -> None:
    """The ordinary path still enforces the stop asynchronously."""
    printer = MemoryPrinter()
    server = VSCodeServer(printer=printer)
    tab_id = "TAB-F2-STOP-OK"
    worker = _StubbornWorker()
    state = AgentState(
        "f2-stop-task-ok",
        tab_id=tab_id,
        server_owned=True,
        stop_event=threading.Event(),
        task_thread=worker.thread,
    )
    agent_state.register(state)
    worker.thread.start()
    t0 = time.monotonic()
    server._stop_task(tab_id)
    # The command handler returns at once; the watchdog does the waiting.
    assert time.monotonic() - t0 < 0.5
    assert worker.finished.wait(15), "the worker was never interrupted"
    assert worker.interrupted.is_set()
