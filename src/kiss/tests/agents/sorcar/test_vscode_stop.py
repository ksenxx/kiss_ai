# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Test that the VS Code server stop button works.

The stop command must be processed while a task is running: ``_cmd_run``
starts ``_run_task`` in a background thread, so the command dispatcher
(``_handle_command``) stays free to process a subsequent ``stop``.  The
old per-tab stdin/stdout subprocess transport has been replaced by the
single ``kiss-web`` daemon, so this test drives :class:`VSCodeServer`
in-process through ``_handle_command`` — the same entry point the
daemon's transports call.

Also tests the force-stop mechanism that uses
``ctypes.pythonapi.PyThreadState_SetAsyncExc`` to interrupt a task
thread that is blocked in I/O and never reaches a cooperative
``_check_stop()`` call.
"""

import os
import threading
import time
import unittest

import pytest


class TestVSCodeServerStop(unittest.TestCase):
    """Integration test: stop command interrupts a running task."""

    @pytest.mark.slow
    def test_stop_command_interrupts_running_task(self) -> None:
        """Dispatch a run command, then a stop command, and verify the task stops."""
        from kiss.agents.sorcar.running_agent_state import _RunningAgentState
        from kiss.server.server import VSCodeServer

        kiss_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )))

        server = VSCodeServer()
        events: list[dict] = []
        lock = threading.Lock()

        def capture(e: dict) -> None:
            with lock:
                events.append(e)

        server.printer.broadcast = capture  # type: ignore[assignment]
        tab_id = "vscode-stop-e2e"

        def _saw(predicate, timeout: float) -> bool:
            deadline = time.time() + timeout
            while time.time() < deadline:
                with lock:
                    if any(predicate(e) for e in events):
                        return True
                time.sleep(0.1)
            return False

        try:
            # The dispatcher must return promptly (_cmd_run starts the
            # task in a background thread).  Dispatch on a helper
            # thread and require it to return within 5s so a
            # regression that makes ``run`` block the dispatcher loop
            # fails loudly here instead of hanging the test.
            dispatched = threading.Event()
            dispatch_errors: list[BaseException] = []

            def _dispatch_run() -> None:
                try:
                    server._handle_command({
                        "type": "run",
                        "tabId": tab_id,
                        "prompt": "Count to one trillion very slowly",
                        "model": "claude-opus-4-6",
                        "workDir": kiss_root,
                    })
                except BaseException as exc:  # pragma: no cover - fail
                    dispatch_errors.append(exc)
                finally:
                    dispatched.set()

            threading.Thread(target=_dispatch_run, daemon=True).start()
            assert dispatched.wait(5), (
                "run command blocked the dispatcher for >5s"
            )
            assert not dispatch_errors, (
                f"run dispatch raised: {dispatch_errors}"
            )

            got_running = _saw(
                lambda e: e.get("type") == "status" and e.get("running") is True,
                30,
            )
            with lock:
                snapshot = list(events)
            assert got_running, f"Never saw status running=true. Events: {snapshot}"

            # The stop command must be processed while the task is running.
            server._handle_command({"type": "stop", "tabId": tab_id})

            # Direct proof that the stop was processed: the tab's
            # cooperative stop event must be set (a naturally-finishing
            # task could otherwise fake the terminal event below).
            with _RunningAgentState._registry_lock:
                tab = _RunningAgentState.running_agent_states.get(tab_id)
                stop_event = tab.stop_event if tab is not None else None
            assert stop_event is not None and stop_event.is_set(), (
                "stop command did not set the tab's stop_event"
            )

            got_stopped = _saw(
                lambda e: (
                    e.get("type") in ("task_stopped", "task_done", "task_error")
                    or (e.get("type") == "status" and e.get("running") is False)
                ),
                60,
            )
            with lock:
                snapshot = list(events)
            assert got_stopped, (
                f"Stop command was not processed while task was running. Events: {snapshot}"
            )
        finally:
            server._stop_task(tab_id)
            with _RunningAgentState._registry_lock:
                tab = _RunningAgentState.running_agent_states.get(tab_id)
                task_thread = tab.task_thread if tab is not None else None
            if task_thread is not None:
                task_thread.join(timeout=60)
            _RunningAgentState.unregister(tab_id)


class TestForceStopMechanism(unittest.TestCase):
    """Test the force-stop watchdog that interrupts blocked task threads."""

    def test_status_running_false_after_force_stop(self) -> None:
        """After force-stop, the finally block still broadcasts status:running:false."""
        from kiss.server.server import VSCodeServer

        server = VSCodeServer()
        events: list[dict] = []
        lock = threading.Lock()
        tab_id = "1"

        def capture(e: dict) -> None:
            with lock:
                events.append(e)

        server.printer.broadcast = capture  # type: ignore[assignment]

        def blocking_task() -> None:
            server.printer.broadcast({"type": "status", "running": True})
            try:
                while True:
                    time.sleep(0.05)
            except KeyboardInterrupt:
                pass
            finally:
                server.printer.broadcast({"type": "status", "running": False})

        stop_event = threading.Event()
        tab = server._get_tab(tab_id)
        tab.stop_event = stop_event
        thread = threading.Thread(target=blocking_task, daemon=True)
        tab.task_thread = thread
        thread.start()
        time.sleep(0.1)

        server._stop_task(tab_id)
        thread.join(timeout=10)

        with lock:
            status_events = [e for e in events if e.get("type") == "status"]
        assert any(e.get("running") is False for e in status_events), (
            f"Should have status:running:false after stop. Events: {status_events}"
        )


if __name__ == "__main__":
    unittest.main()
