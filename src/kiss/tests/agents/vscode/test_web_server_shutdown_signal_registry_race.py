# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: SIGHUP handler must not race the agent-state registry.

Background
----------
The remote-access web server's :meth:`RemoteAccessServer._handle_shutdown_signal`
iterates ``_RunningAgentState.running_agent_states.items()`` to log a
snapshot of in-flight agent tasks when a catchable termination signal
(SIGTERM / SIGHUP) is delivered.  Before the fix, that iteration ran
*without* holding ``_RunningAgentState._registry_lock``.

Signal handlers run synchronously on the main thread, interrupting
whatever bytecode happens to be executing.  When a worker thread is
mid-mutation of ``running_agent_states`` (registering a fresh tab via
:meth:`_RunningAgentState.register`, disposing a finished one via
:meth:`_RunningAgentState.dispose`) the signal handler's
``dict.items()`` iterator races the mutation and raises
``RuntimeError: dictionary changed size during iteration`` *from
inside the signal handler*.  That RuntimeError is not a
:class:`KeyboardInterrupt`, so it bypasses the
``except KeyboardInterrupt`` arm in :meth:`RemoteAccessServer.start`,
escapes ``asyncio.run`` uncaught, and crashes the daemon with an
unhandled traceback — visible to the user as a kiss-web flap.

This test reproduces the race by hammering the signal handler from
the main thread while a worker thread continuously adds and removes
entries from the registry.  We use SIGHUP rather than SIGTERM
because the SIGTERM path also raises :class:`KeyboardInterrupt` on
the first invocation (deliberately, to drive shutdown) and would
make it impossible to deliver the signal in a tight loop without
unwinding the test.  SIGHUP follows the *identical* iteration code
path inside the handler but does *not* raise, so we can deliver it
many times in a row to expose the race.
"""

from __future__ import annotations

import signal
import threading
import time
import unittest
from typing import cast

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.server.web_server import RemoteAccessServer


class TestShutdownSignalRegistryRace(unittest.TestCase):
    """``_handle_shutdown_signal`` survives concurrent registry mutation."""

    def setUp(self) -> None:
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
        )
        with _RunningAgentState._registry_lock:
            self._preserved = dict(_RunningAgentState.running_agent_states)
            _RunningAgentState.running_agent_states.clear()

    def tearDown(self) -> None:
        with _RunningAgentState._registry_lock:
            _RunningAgentState.running_agent_states.clear()
            _RunningAgentState.running_agent_states.update(self._preserved)

    def test_signal_handler_does_not_race_registry_mutation(self) -> None:
        """Hammer the SIGHUP path concurrently with registry churn.

        Pre-fix this test reliably surfaces
        ``RuntimeError: dictionary changed size during iteration``
        within a handful of iterations on a typical macOS / Linux
        host.  Post-fix the handler snapshots under
        ``_registry_lock`` and the race is impossible.
        """
        stop = threading.Event()
        errors: list[BaseException] = []

        class _GilYieldingTab:
            is_task_active = False
            task_history_id = None
            last_task_id = None

            def __getattribute__(self, name: str) -> object:
                time.sleep(0)
                return object.__getattribute__(self, name)

        template = cast(_RunningAgentState, _GilYieldingTab())
        with _RunningAgentState._registry_lock:
            for k in range(200):
                stable_id = f"stable-{k}"
                _RunningAgentState.running_agent_states[stable_id] = (
                    template
                )

        def churn() -> None:
            """Mutate the registry without holding ``_registry_lock``.

            Direct unlocked ``dict`` mutation is the worst-case
            equivalent of a worker thread that has *not* yet
            acquired the lock but whose ``__setitem__`` /
            ``pop`` still goes through.  In production, ``register``
            and ``unregister`` *do* hold the lock — but the fix
            under test makes the signal handler hold the SAME lock,
            so the contract is "either side holding the lock is
            enough".  This stress-test exercises the still-bad case
            where the handler is the *only* one not holding it.
            """
            i = 0
            try:
                while not stop.is_set():
                    for j in range(50):
                        key = f"churn-{i}-{j}"
                        _RunningAgentState.running_agent_states[key] = (
                            template
                        )
                    for j in range(50):
                        _RunningAgentState.running_agent_states.pop(
                            f"churn-{i}-{j}", None,
                        )
                    i += 1
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        churn_thread = threading.Thread(target=churn, daemon=True)
        churn_thread.start()

        deadline = time.monotonic() + 2.0
        handler_errors: list[BaseException] = []
        invocations = 0
        try:
            while time.monotonic() < deadline:
                try:
                    self.server._handle_shutdown_signal(signal.SIGHUP)
                except BaseException as exc:  # noqa: BLE001
                    handler_errors.append(exc)
                invocations += 1
        finally:
            stop.set()
            churn_thread.join(timeout=2.0)

        self.assertFalse(
            errors,
            f"Churn thread raised: {errors!r}",
        )
        self.assertGreater(
            invocations,
            3,
            "Test did not run the handler enough times",
        )
        self.assertFalse(
            handler_errors,
            "Signal handler raised under concurrent registry mutation: "
            f"{handler_errors!r}",
        )


if __name__ == "__main__":
    unittest.main()
