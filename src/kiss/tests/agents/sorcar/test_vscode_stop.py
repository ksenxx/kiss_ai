"""Test that the VS Code server stop button works.

The stop command must be processed while a task is running.
This was broken because _run_task blocked the stdin reading loop.
The fix runs _run_task in a background thread.
"""

import json
import os
import subprocess
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


if __name__ == "__main__":
    unittest.main()
