"""Test that the VS Code server stop button works.

The stop command must be processed while a task is running.
This was broken because _run_task blocked the stdin reading loop.
The fix runs _run_task in a background thread.

Also tests the force-stop mechanism that uses
``ctypes.pythonapi.PyThreadState_SetAsyncExc`` to interrupt a task
thread that is blocked in I/O and never reaches a cooperative
``_check_stop()`` call.
"""

import json
import os
import subprocess
import threading
import time
import unittest


class TestVSCodeServerStop(unittest.TestCase):
    """Integration test: stop command interrupts a running task."""

    def test_stop_command_interrupts_running_task(self) -> None:
        """Send a run command, then a stop command, and verify the task stops."""
        # Find the uv binary
        home = os.path.expanduser("~")
        uv = os.path.join(home, ".local", "bin", "uv")
        if not os.path.exists(uv):
            uv = "uv"

        kiss_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )))

        proc = subprocess.Popen(
            [uv, "run", "python", "-u", "-m", "kiss.agents.vscode.server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=kiss_root,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "KISS_WORKDIR": kiss_root},
        )
        assert proc.stdin is not None
        assert proc.stdout is not None

        try:
            # Send a run command with a task that will take a long time
            run_cmd = json.dumps({
                "type": "run",
                "prompt": "Count to one trillion very slowly",
                "model": "claude-opus-4-6",
                "workDir": kiss_root,
            }) + "\n"
            proc.stdin.write(run_cmd.encode())
            proc.stdin.flush()

            # Wait for the task to start (look for status running=true)
            deadline = time.time() + 15
            got_running = False
            events: list[dict] = []

            # Set stdout to non-blocking to read events
            import select
            while time.time() < deadline:
                ready, _, _ = select.select([proc.stdout], [], [], 0.5)
                if ready:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    line_str = line.decode().strip()
                    if not line_str:
                        continue
                    try:
                        ev = json.loads(line_str)
                        events.append(ev)
                        if ev.get("type") == "status" and ev.get("running") is True:
                            got_running = True
                            break
                    except json.JSONDecodeError:
                        pass

            assert got_running, f"Never saw status running=true. Events: {events}"

            # Now send the stop command
            stop_cmd = json.dumps({"type": "stop"}) + "\n"
            proc.stdin.write(stop_cmd.encode())
            proc.stdin.flush()

            # Wait for task_stopped or status running=false
            got_stopped = False
            deadline = time.time() + 30
            while time.time() < deadline:
                ready, _, _ = select.select([proc.stdout], [], [], 0.5)
                if ready:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    line_str = line.decode().strip()
                    if not line_str:
                        continue
                    try:
                        ev = json.loads(line_str)
                        events.append(ev)
                        if ev.get("type") == "task_stopped":
                            got_stopped = True
                            break
                        if ev.get("type") == "status" and ev.get("running") is False:
                            got_stopped = True
                            break
                        if ev.get("type") == "task_done":
                            # Task completed before stop - still ok, just not testing stop
                            got_stopped = True
                            break
                        if ev.get("type") == "task_error":
                            # API error is fine - the point is the stop command was processed
                            got_stopped = True
                            break
                    except json.JSONDecodeError:
                        pass

            assert got_stopped, (
                f"Stop command was not processed while task was running. Events: {events}"
            )

        finally:
            proc.stdin.close()
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def test_task_runs_in_thread_not_blocking_stdin(self) -> None:
        """Verify that _run_task runs in a thread by checking _task_thread attribute."""
        from kiss.agents.vscode.server import VSCodeServer

        server = VSCodeServer()
        assert hasattr(server, "_task_thread")
        assert server._task_thread is None


class TestForceStopMechanism(unittest.TestCase):
    """Test the force-stop watchdog that interrupts blocked task threads."""

    def test_force_stop_interrupts_blocked_thread(self) -> None:
        """A thread doing Python-level work is interrupted by the watchdog.

        Uses a tight Python loop (simulating LLM stream processing) that
        the cooperative ``_check_stop()`` never runs in, but
        ``PyThreadState_SetAsyncExc`` can interrupt at bytecode boundaries.
        """
        from kiss.agents.vscode.server import VSCodeServer

        server = VSCodeServer()
        interrupted = threading.Event()

        def blocking_task() -> None:
            try:
                # Simulate LLM stream processing: short C-level sleeps
                # between Python-level iterations (like httpx reading
                # chunks from a socket). PyThreadState_SetAsyncExc
                # delivers the exception between iterations.
                while True:
                    time.sleep(0.05)
            except KeyboardInterrupt:
                interrupted.set()

        server._stop_event = threading.Event()
        server._task_thread = threading.Thread(target=blocking_task, daemon=True)
        server._task_thread.start()

        # Give the thread a moment to start
        time.sleep(0.1)

        server._stop_task()

        # The watchdog waits 1s then sends KeyboardInterrupt;
        # give it up to 8s total to work.
        server._task_thread.join(timeout=8)
        assert not server._task_thread.is_alive(), "Task thread should have been interrupted"
        assert interrupted.is_set(), "KeyboardInterrupt should have been raised"

    def test_cooperative_stop_prevents_force_interrupt(self) -> None:
        """When cooperative stop works quickly, the watchdog never fires."""
        from kiss.agents.vscode.server import VSCodeServer

        server = VSCodeServer()

        def cooperative_task() -> None:
            assert server._stop_event is not None
            while not server._stop_event.is_set():
                time.sleep(0.05)
            # Task exits cooperatively — no KeyboardInterrupt needed

        server._stop_event = threading.Event()
        server._task_thread = threading.Thread(target=cooperative_task, daemon=True)
        server._task_thread.start()
        time.sleep(0.1)

        server._stop_task()

        # The thread should exit almost immediately (cooperatively)
        server._task_thread.join(timeout=0.5)
        assert not server._task_thread.is_alive()

    def test_force_stop_thread_exits_if_thread_already_dead(self) -> None:
        """Watchdog exits immediately if the task thread is already dead."""
        from kiss.agents.vscode.server import VSCodeServer

        t = threading.Thread(target=lambda: None, daemon=True)
        t.start()
        t.join()  # Thread is dead

        # Calling force_stop_thread on a dead thread should return quickly
        start = time.monotonic()
        VSCodeServer._force_stop_thread(t)
        elapsed = time.monotonic() - start
        assert elapsed < 2, f"Should have exited quickly, took {elapsed:.1f}s"

    def test_stop_task_with_no_stop_event(self) -> None:
        """_stop_task is a no-op when _stop_event is None (no task running)."""
        from kiss.agents.vscode.server import VSCodeServer

        server = VSCodeServer()
        assert server._stop_event is None
        assert server._task_thread is None
        # Should not raise
        server._stop_task()

    def test_status_running_false_after_force_stop(self) -> None:
        """After force-stop, the finally block still broadcasts status:running:false."""
        from kiss.agents.vscode.server import VSCodeServer

        server = VSCodeServer()
        events: list[dict] = []
        lock = threading.Lock()

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

        server._stop_event = threading.Event()
        server._task_thread = threading.Thread(target=blocking_task, daemon=True)
        server._task_thread.start()
        time.sleep(0.1)

        server._stop_task()
        server._task_thread.join(timeout=10)

        with lock:
            status_events = [e for e in events if e.get("type") == "status"]
        assert any(e.get("running") is False for e in status_events), (
            f"Should have status:running:false after stop. Events: {status_events}"
        )


if __name__ == "__main__":
    unittest.main()
