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
            run_cmd = json.dumps({
                "type": "run",
                "prompt": "Count to one trillion very slowly",
                "model": "claude-opus-4-6",
                "workDir": kiss_root,
            }) + "\n"
            proc.stdin.write(run_cmd.encode())
            proc.stdin.flush()

            deadline = time.time() + 15
            got_running = False
            events: list[dict] = []

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

            stop_cmd = json.dumps({"type": "stop"}) + "\n"
            proc.stdin.write(stop_cmd.encode())
            proc.stdin.flush()

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
                            got_stopped = True
                            break
                        if ev.get("type") == "task_error":
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


class TestForceStopMechanism(unittest.TestCase):
    """Test the force-stop watchdog that interrupts blocked task threads."""

    def test_status_running_false_after_force_stop(self) -> None:
        """After force-stop, the finally block still broadcasts status:running:false."""
        from kiss.agents.vscode.server import VSCodeServer

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
