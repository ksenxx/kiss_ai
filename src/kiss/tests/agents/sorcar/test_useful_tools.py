# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for useful_tools.py module."""

import json
import os
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
import tracemalloc
from contextlib import suppress
from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import (
    _MAX_BINARY_READ_BYTES,
    NOT_STARTED,
    UsefulTools,
    _truncate_output,
)
from kiss.core.processes import pid_alive


@pytest.fixture
def temp_test_dir():
    test_dir = Path(tempfile.mkdtemp()).resolve()
    original_dir = Path.cwd()
    os.chdir(test_dir)
    yield test_dir
    os.chdir(original_dir)
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture
def tools(temp_test_dir):
    return UsefulTools(), temp_test_dir


class TestUsefulTools:

    def test_write_to_directory_path(self, tools):
        ut, test_dir = tools
        subdir = test_dir / "subdir"
        subdir.mkdir()
        result = ut.Write(str(subdir), "content")
        assert "Error:" in result


@pytest.fixture
def streaming_tools(temp_test_dir):
    streamed: list[str] = []
    ut = UsefulTools(stream_callback=streamed.append)
    return ut, temp_test_dir, streamed


@pytest.fixture(params=[False, True], ids=["nonstreaming", "streaming"])
def any_tools(request, temp_test_dir):
    if request.param:
        return UsefulTools(stream_callback=lambda _: None), temp_test_dir
    return UsefulTools(), temp_test_dir
@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="Unix-only test (signals, bash scripts)")
class TestAdversarial:
    """Adversarial tests to try to break the Popen/killpg changes."""

    def test_interrupt_kills_child(self, any_tools):
        """KeyboardInterrupt must kill the child process group."""
        import _thread
        import threading
        import time

        ut, test_dir = any_tools
        pid_file = test_dir / "interrupt_child.pid"
        script = test_dir / "interrupt_target.sh"
        script.write_text(
            f"#!/bin/bash\necho $$ > {pid_file}\nsleep 100\n"
        )
        script.chmod(0o755)

        child_pid = None

        def send_interrupt():
            nonlocal child_pid
            for _ in range(20):
                time.sleep(0.1)
                if pid_file.exists():
                    # Poll for parseable content, not mere existence:
                    # the shell redirect creates the file empty before
                    # echo writes the PID.
                    pid_text = pid_file.read_text().strip()
                    if pid_text.isdigit():
                        child_pid = int(pid_text)
                        break
            if child_pid:
                _thread.interrupt_main()

        t = threading.Thread(target=send_interrupt, daemon=True)
        t.start()
        try:
            ut.Bash(str(script), "interruptible", timeout_seconds=30)
        except KeyboardInterrupt:
            pass
        t.join(timeout=5)

        if child_pid is None:
            pytest.skip("Script didn't start in time")

        time.sleep(0.3)
        alive = False
        try:
            os.kill(child_pid, 0)
            alive = True
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        assert not alive, f"Child {child_pid} survived KeyboardInterrupt!"

class TestBugs:
    """Tests that expose bugs in useful_tools.py."""

    def test_truncate_output_tiny_limit(self):
        big = "X" * 200
        result = _truncate_output(big, 5)
        assert len(result) <= 5

    def test_read_refuses_an_oversized_binary_without_loading_it(self, temp_test_dir):
        """Read checks the binary size cap before reading, so an oversized PNG costs no memory.

        A sparse file one byte over the cap with a PNG signature (invalid
        UTF-8) used to be read whole by the text attempt before the
        binary branch measured it (A10-7).
        """
        big = temp_test_dir / "big.png"
        with open(big, "wb") as handle:
            handle.write(b"\x89PNG\r\n\x1a\n")
            handle.truncate(_MAX_BINARY_READ_BYTES + 1)
        ut = UsefulTools(work_dir=str(temp_test_dir))
        tracemalloc.start()
        try:
            result = ut.Read(str(big), max_lines=1)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert result.startswith("Error: Binary file") and "too large" in result, result
        assert peak < _MAX_BINARY_READ_BYTES // 4, f"Read allocated {peak} bytes to refuse the file"


class TestConcurrentEdits:
    """Two agents (two tool instances, two threads) editing one file lose no change.

    ``Edit`` is a read-modify-write of the whole file; without a lock a
    sibling's write landing between another's read and write was undone
    by that write, both calls reporting success (A10-2).  Edit has no
    synchronisation point a test could hold without a test double, so
    the race is provoked by volume: forty edits per thread on a 4 MB
    file, every one of which must survive.
    """

    def test_concurrent_edits_of_one_file_keep_every_change(self, temp_test_dir):
        path = temp_test_dir / "shared.txt"
        edits_per_thread = 40
        lines = [f"{prefix}{i}=0\n" for prefix in "ab" for i in range(edits_per_thread)]
        path.write_text("".join(lines) + ("x" * 200 + "\n") * 20000)
        tools = [UsefulTools(work_dir=str(temp_test_dir)) for _ in "ab"]
        for tool in tools:
            assert "a0=0" in tool.Read(str(path), max_lines=2)
        results: dict[str, list[str]] = {}

        def edit(prefix: str, tool: UsefulTools) -> None:
            results[prefix] = [
                tool.Edit(str(path), f"{prefix}{i}=0\n", f"{prefix}{i}=1\n")
                for i in range(edits_per_thread)
            ]

        threads = [
            threading.Thread(target=edit, args=(prefix, tool))
            for prefix, tool in zip("ab", tools, strict=True)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert all(r.startswith("Successfully") for rs in results.values() for r in rs), results
        text = path.read_text()
        lost = [
            f"{prefix}{i}" for prefix in "ab" for i in range(edits_per_thread)
            if f"{prefix}{i}=1\n" not in text
        ]
        assert lost == [], f"edits undone by a concurrent Edit: {lost}"


@pytest.mark.skipif(sys.platform == "win32", reason="Unix-only test (signals, bash scripts)")
class TestStopEvent:
    """Tests that stop_event kills child processes promptly."""

    def test_stop_event_kills_streaming_child(self, temp_test_dir):
        """When stop_event is set, the streaming Bash child must be killed."""
        import threading
        import time

        stop_event = threading.Event()
        streamed: list[str] = []
        ut = UsefulTools(stream_callback=streamed.append, stop_event=stop_event)

        pid_file = temp_test_dir / "stop_child.pid"
        script = temp_test_dir / "stop_target.sh"
        script.write_text(
            f"#!/bin/bash\necho $$ > {pid_file}\nwhile true; do echo tick; sleep 0.2; done\n"
        )
        script.chmod(0o755)

        child_pid = None

        def set_stop_after_start():
            nonlocal child_pid
            for _ in range(50):
                time.sleep(0.1)
                if pid_file.exists():
                    # Poll for parseable content, not mere existence:
                    # the shell redirect creates the file empty before
                    # echo writes the PID.
                    pid_text = pid_file.read_text().strip()
                    if pid_text.isdigit():
                        child_pid = int(pid_text)
                        break
            if child_pid:
                stop_event.set()

        t = threading.Thread(target=set_stop_after_start, daemon=True)
        t.start()
        ut.Bash(str(script), "stoppable", timeout_seconds=30)
        t.join(timeout=5)

        if child_pid is None:
            pytest.skip("Script didn't start in time")

        time.sleep(0.5)
        alive = False
        try:
            os.kill(child_pid, 0)
            alive = True
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        assert not alive, f"Child {child_pid} survived stop_event!"

    def test_stop_event_kills_nonstreaming_child(self, temp_test_dir):
        """When stop_event is set, the non-streaming Bash child must be killed."""
        import threading
        import time

        stop_event = threading.Event()
        ut = UsefulTools(stop_event=stop_event)

        pid_file = temp_test_dir / "stop_child_ns.pid"
        script = temp_test_dir / "stop_target_ns.sh"
        script.write_text(
            f"#!/bin/bash\necho $$ > {pid_file}\nsleep 100\n"
        )
        script.chmod(0o755)

        child_pid = None

        def set_stop_after_start():
            nonlocal child_pid
            for _ in range(50):
                time.sleep(0.1)
                if pid_file.exists():
                    # Poll for parseable content, not mere existence:
                    # the shell redirect creates the file empty before
                    # echo writes the PID.
                    pid_text = pid_file.read_text().strip()
                    if pid_text.isdigit():
                        child_pid = int(pid_text)
                        break
            if child_pid:
                stop_event.set()

        t = threading.Thread(target=set_stop_after_start, daemon=True)
        t.start()
        ut.Bash(str(script), "stoppable", timeout_seconds=30)
        t.join(timeout=5)

        if child_pid is None:
            pytest.skip("Script didn't start in time")

        time.sleep(0.5)
        alive = False
        try:
            os.kill(child_pid, 0)
            alive = True
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        assert not alive, f"Child {child_pid} survived stop_event!"

    def test_stop_set_before_a_parallel_call_starts_no_command(self, temp_test_dir):
        """A pool entered with the task's stop event already set starts nothing (A10-5).

        The workers used to be submitted before the stop was looked at,
        so a quick command ran (and was then killed) after the Stop.
        """
        stop_event = threading.Event()
        stop_event.set()
        marker = temp_test_dir / "started.marker"
        ut = UsefulTools(stop_event=stop_event, work_dir=str(temp_test_dir))
        report = ut.run_commands_parallel(json.dumps([f"touch {marker}", f"touch {marker}"]))
        assert report.count(NOT_STARTED) == 2, report
        assert not marker.exists(), "a command ran after the task was stopped"

    def test_stop_kills_a_background_jobs_orphans_after_its_shell_exited(self, temp_test_dir):
        """A job's descendant that outlives the job's shell is killed by a later task Stop (A10-6).

        ``sleep 100 &`` stays in the job's process group after the shell
        exits; the watcher used to end with the shell, leaving nothing to
        kill the group when the Stop came.
        """
        stop_event = threading.Event()
        ut = UsefulTools(stop_event=stop_event, work_dir=str(temp_test_dir))
        pid_file = temp_test_dir / "orphan.pid"
        script = temp_test_dir / "orphan.sh"
        script.write_text(f"sleep 100 &\necho $! > {pid_file}\n")
        started = ut.Bash(f"bash {script}", "orphan", background=True)
        match = re.search(r"job (j[0-9a-f]+)", started)
        assert match is not None, started
        job_id = match.group(1)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if "exited with code 0" in ut.bash_job(job_id) and pid_file.is_file():
                break
            time.sleep(0.05)
        assert "exited with code 0" in ut.bash_job(job_id)
        orphan = int(pid_file.read_text().strip())
        assert pid_alive(orphan)
        try:
            stop_event.set()
            deadline = time.monotonic() + 10
            while pid_alive(orphan) and time.monotonic() < deadline:
                time.sleep(0.05)
            assert not pid_alive(orphan), f"the job's orphan {orphan} survived the task Stop"
        finally:
            with suppress(OSError):
                os.kill(orphan, signal.SIGKILL)


class TestStreamingTimeout:
    """Test that streaming Bash correctly reports timeout."""

    def test_streaming_bash_timeout(self, temp_test_dir):
        """Streaming Bash must return timeout error when command exceeds timeout."""
        streamed: list[str] = []
        ut = UsefulTools(stream_callback=streamed.append)
        result = ut.Bash("sleep 10", "should timeout", timeout_seconds=0.5)
        assert "timeout" in result.lower()

    def test_timer_firing_after_completion_keeps_output(self, temp_test_dir):
        """A timer that fires after the command already finished must not clobber output.

        The command completes almost instantly, but a slow stream callback
        delays the loop past the timeout, so the Timer deterministically
        fires in the gap between command completion and ``timer.cancel()``.
        The successful output must be preserved, not replaced with a
        timeout error.
        """
        def slow_stream(_line: str) -> None:
            time.sleep(1.0)

        ut = UsefulTools(stream_callback=slow_stream)
        result = ut.Bash("echo hi", "quick command", timeout_seconds=0.5)
        assert "timeout" not in result.lower()
        assert "hi" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
