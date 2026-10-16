# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Real-container regressions: the docker ``Bash`` cancel path is bounded.

Audit 2026-10-15 (docker-fix, second round).  A review of the unified
``DockerManager._exec`` against a real daemon showed four ways the task
thread could still block past its deadline or leak a reaper:

1. The cancel path ran the kill sweep (one ``exec_run``) synchronously
   on the task thread; a daemon slow to answer it held ``Bash`` for as
   long as the sweep took.  The sweep now runs on the reaper thread and
   the task thread waits for it a bounded, sliced time.
2. The reaper was installed only after that sweep, so a forced
   ``ToolCallInterrupted`` landing during the sweep left a start that
   the daemon had delayed past the deadline with nobody to kill it.
   The reaper is installed before any wait.
3. After the stream ended, the task thread polled ``exec_inspect`` for
   the exit code; one inspect blocks as long as the daemon likes and a
   non-null exit code was accepted without rechecking the deadline or
   the stop event.  The worker thread polls now; the task thread only
   waits on the queue in slices.
4. An ``exec_start`` the daemon refused (``409`` for a paused container)
   was treated as an in-flight start: the cancel path installed a reaper
   that polled an exec that never started until the container closed.
   A refusal is recorded and no reaper is kept for it.

Each test runs a real exec in a real container; the latency shims
replace a method on the real docker-py client or container object,
exactly as the other timeout tests in this directory do, and the
interrupt is the real ``tool_interrupt.inject_async_exception``.
"""

import threading
import time
import unittest
from typing import Any, cast

import docker.errors

from kiss.agents.sorcar.docker_manager import _KILL_WAIT_S, DockerManager
from kiss.core import tool_interrupt
from kiss.core.tool_interrupt import ToolCallInterrupted
from kiss.tests.agents.sorcar.docker_test_containers import is_docker_available
from kiss.tests.agents.sorcar.test_docker_nonstream_timeout_kill import (
    _count_processes_script,
)


def _run_bash_recording(
    env: DockerManager, command: str, timeout_seconds: int, outcome: dict[str, Any],
) -> None:
    """Thread body: run ``env.Bash`` and record its result or exception."""
    try:
        outcome["result"] = env.Bash(command, "interrupted", timeout_seconds=timeout_seconds)
    except BaseException as exc:
        outcome["error"] = exc


def _wait_for_thread_count(expected: int, seconds: float) -> int:
    """Return the thread count once it drops to *expected*, or the last seen count."""
    deadline = time.monotonic() + seconds
    count = threading.active_count()
    while count > expected and time.monotonic() < deadline:
        time.sleep(0.2)
        count = threading.active_count()
    return count


class _StopOnceOutputIsQueued(threading.Event):
    """A stop event that fires, during the task thread's check, once the output is queued.

    ``is_set()`` answers False until ``armed`` (set by the ``exec_start``
    shim), then waits for ``output_queued`` (set by the ``exec_inspect``
    shim, which the worker reaches only after it queued every piece of
    the command's output) and answers True.  The Stop therefore lands
    between two queue slices with the output still sitting in the
    queue, the interleaving a loaded host produced by chance.
    """

    def __init__(self) -> None:
        super().__init__()
        self.armed = threading.Event()
        self.output_queued = threading.Event()

    def is_set(self) -> bool:
        if not self.armed.is_set():
            return False
        self.output_queued.wait(10)
        return True


@unittest.skipUnless(is_docker_available(), "Docker daemon is not running")
class TestExecCancelBounded(unittest.TestCase):
    """Timeout, Stop and a refused start never block or leak past the deadline."""

    env: DockerManager

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = DockerManager("python:3.11-slim")
        cls.env.open()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def setUp(self) -> None:
        self.api = cast(Any, self.env.client.api)
        self.container = cast(Any, self.env.container)
        self.first_call = threading.Event()

    def tearDown(self) -> None:
        self.env.stop_event = None
        for owner, name in ((self.api, "exec_start"), (self.api, "exec_inspect"),
                            (self.container, "exec_run")):
            if name in vars(owner):
                delattr(owner, name)

    def _wait_until_no_process(self, needle_head: str, needle_tail: str) -> None:
        """Poll the container until no process matches the needle."""
        script = _count_processes_script(needle_head, needle_tail)
        deadline = time.monotonic() + 20
        count = ""
        while time.monotonic() < deadline:
            count = self.env.Bash(script, "count leftover processes")
            count = count.strip().splitlines()[-1]
            if count == "0":
                return
            time.sleep(0.5)
        self.fail(f"command still running in the container ({count} processes)")

    def _slow_exec_run(self, seconds: float) -> None:
        """Make the first kill sweep (``container.exec_run``) take *seconds*."""
        real_exec_run = self.container.exec_run
        first_call = self.first_call

        def delayed_exec_run(*args: Any, **kwargs: Any) -> Any:
            if not first_call.is_set():
                first_call.set()
                time.sleep(seconds)
            return real_exec_run(*args, **kwargs)

        self.container.exec_run = delayed_exec_run

    def _slow_exec_start(self, seconds: float, pause_container: bool = False) -> None:
        """Delay the first ``exec_start`` by *seconds*; optionally pause the container first.

        Once the delayed real start has returned, ``self.started`` is set
        and ``self.started_exec_id`` names the exec, so a test can wait
        for the late start to have happened before checking that the
        reaper killed it.
        """
        real_exec_start = self.api.exec_start
        first_call = threading.Event()
        container = self.container
        self.started = threading.Event()
        self.started_exec_id = ""

        def delayed_exec_start(exec_id: str, *args: Any, **kwargs: Any) -> Any:
            if first_call.is_set():
                return real_exec_start(exec_id, *args, **kwargs)
            first_call.set()
            time.sleep(seconds)
            if pause_container:
                container.pause()
            try:
                stream = real_exec_start(exec_id, *args, **kwargs)
            finally:
                if pause_container:
                    container.unpause()
            self.started_exec_id = exec_id
            self.started.set()
            return stream

        self.api.exec_start = delayed_exec_start

    def _slow_exec_inspect(self, seconds: float, stop: threading.Event | None = None) -> None:
        """Delay the first ``exec_inspect`` by *seconds*, setting *stop* first when given."""
        real_exec_inspect = self.api.exec_inspect
        first_call = threading.Event()

        def delayed_exec_inspect(*args: Any, **kwargs: Any) -> Any:
            if not first_call.is_set():
                first_call.set()
                if stop is not None:
                    stop.set()
                time.sleep(seconds)
            return real_exec_inspect(*args, **kwargs)

        self.api.exec_inspect = delayed_exec_inspect

    def _wait_until_exec_finished(self, exec_id: str) -> None:
        """Poll ``exec_inspect`` until *exec_id* has had a pid and is no longer running."""
        deadline = time.monotonic() + 20
        info: dict[str, Any] = {}
        while time.monotonic() < deadline:
            info = self.api.exec_inspect(exec_id)
            if info.get("Pid") and not info.get("Running"):
                return
            time.sleep(0.2)
        self.fail(f"the delayed exec did not start and die: {info}")

    def test_timeout_returns_before_a_slow_kill_sweep_finishes(self) -> None:
        """(1) A kill sweep the daemon withholds does not hold ``Bash`` past its deadline."""
        self._slow_exec_run(5)
        start = time.monotonic()
        result = self.env.Bash("sleep 60", "hang", timeout_seconds=1)
        elapsed = time.monotonic() - start
        self.assertIn("timed out", result)
        self.assertLess(elapsed, 4.5)  # returned while the 5 s sweep was still pending
        self._wait_until_no_process("sle", "ep 60")  # the sweep still happened

    def test_forced_interrupt_during_the_kill_still_reaps_a_delayed_start(self) -> None:
        """(2) A Stop forced during the kill leaves a reaper behind for a start delayed past it."""
        self._slow_exec_start(3)
        self._slow_exec_run(3)
        outcome: dict[str, Any] = {}
        thread = threading.Thread(
            target=_run_bash_recording, args=(self.env, "sleep 47", 1, outcome),
        )
        thread.start()
        assert thread.ident is not None
        # The deadline passes at 1 s and the sweep is in flight until 4 s:
        # the forced interrupt lands in between.
        time.sleep(2)
        tool_interrupt.inject_async_exception(thread.ident, ToolCallInterrupted)
        thread.join(30)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(outcome.get("error"), ToolCallInterrupted)
        # ``Bash`` returned at ~2 s while the start is still delayed until
        # 3 s; a process count taken now would be 0 without proving
        # anything.  Wait for the late start to have happened, then for
        # the daemon to report that exec as started and finished (it
        # would run 47 s unless the reaper killed it), then for its tree.
        self.assertTrue(self.started.wait(10), "the delayed exec_start never returned")
        self._wait_until_exec_finished(self.started_exec_id)
        self._wait_until_no_process("sle", "ep 47")

    def _stop_once_output_arrives(self, text: str) -> None:
        """Stream callback: fire the task Stop as soon as the command has printed."""
        del text
        self.stop_armed_at = time.monotonic()
        self.stop.set()

    def test_timeout_wins_over_a_slow_exit_code_inspect(self) -> None:
        """(3) A deadline that expires while the exit code is awaited is a timeout.

        ``echo done`` finishes within milliseconds of starting; the 3 s
        deadline leaves room for a busy daemon's startup latency and is
        still well inside the 6 s the first inspect takes.
        """
        self._slow_exec_inspect(6)
        start = time.monotonic()
        result = self.env.Bash("echo done", "finished before its exit code", timeout_seconds=3)
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 5.5)
        self.assertIn("timed out", result)
        self.assertIn("done", result)

    def test_stop_wins_over_a_slow_exit_code_inspect(self) -> None:
        """(3) A Stop that lands while the exit code is awaited ends the command."""
        self._slow_exec_inspect(3)
        self.stop = threading.Event()
        self.stop_armed_at = 0.0
        self.env.stop_event = self.stop
        self.env.stream_callback = self._stop_once_output_arrives
        try:
            result = self.env.Bash("echo done", "finished before its exit code", timeout_seconds=30)
        finally:
            self.env.stream_callback = None
        elapsed = time.monotonic() - self.stop_armed_at
        self.assertLess(elapsed, 2.5)
        self.assertIn("Killed: the task was stopped.", result)
        self.assertIn("done", result)

    def test_stop_during_the_final_inspect_wins_over_the_exit_code(self) -> None:
        """(3) A Stop set while the worker fetches the exit code beats the exit code.

        The Stop is set inside the inspect that fetches ``echo done``'s
        exit code, 0.1 s before it answers: the task thread is already
        waiting on the queue and the worker's final item lands in that
        same slice.  A verdict taken from "the worker finished" instead
        of from the stop event returned ``done`` as a success.
        """
        self.stop = threading.Event()
        self.env.stop_event = self.stop
        for _ in range(3):
            self.stop.clear()
            self._slow_exec_inspect(0.1, stop=self.stop)
            result = self.env.Bash("echo done", "stopped as it finishes", timeout_seconds=30)
            self.assertIn("Killed: the task was stopped.", result)
            self.assertIn("done", result)
            del self.api.exec_inspect  # the next round shims the real method again

    def test_stop_between_two_slices_keeps_the_output_already_queued(self) -> None:
        """(3) Output the worker queued before a Stop landed is part of the result.

        ``exec_start`` is held for 0.3 s so the task thread's first 0.2 s
        slice is empty; the stop event then answers its next check only
        once the worker has queued ``done`` and moved on to the exit
        code.  The wait used to end right there, leaving ``done`` in the
        queue: the result was a bare ``Killed: the task was stopped.``
        (the 2026-10-15 full run hit this interleaving under load).
        """
        stop = _StopOnceOutputIsQueued()
        self.env.stop_event = stop
        real_exec_start = self.api.exec_start
        real_exec_inspect = self.api.exec_inspect

        def held_exec_start(*args: Any, **kwargs: Any) -> Any:
            stop.armed.set()
            time.sleep(0.3)
            return real_exec_start(*args, **kwargs)

        def exec_inspect_after_output(*args: Any, **kwargs: Any) -> Any:
            stop.output_queued.set()
            return real_exec_inspect(*args, **kwargs)

        self.api.exec_start = held_exec_start
        self.api.exec_inspect = exec_inspect_after_output
        result = self.env.Bash("echo done", "stopped between slices", timeout_seconds=30)
        self.assertIn("Killed: the task was stopped.", result)
        self.assertIn("done", result)

    def test_a_command_that_outlives_its_deadline_is_always_a_timeout(self) -> None:
        """(3) ``sleep 1.1`` under a 1 s deadline reports the timeout, however the wait ends.

        The verdict must come from the clock, not from whether the
        worker's final item happened to be taken: a task thread held
        between its deadline check and the queue wait once accepted the
        exit code of a command that finished after the deadline.
        """
        for _ in range(5):
            start = time.monotonic()
            result = self.env.Bash("sleep 1.1", "just past the deadline", timeout_seconds=1)
            elapsed = time.monotonic() - start
            self.assertIn("timed out", result)
            self.assertLess(elapsed, 1 + _KILL_WAIT_S + 1)

    def test_refused_start_creates_no_reaper(self) -> None:
        """(4) An ``exec_start`` the daemon rejects is propagated and leaves no reaper thread."""
        threads_before = threading.active_count()
        for _ in range(3):
            self._slow_exec_start(0, pause_container=True)
            with self.assertRaises(docker.errors.APIError) as raised:
                self.env.Bash("echo never", "refused start", timeout_seconds=10)
            self.assertEqual(raised.exception.status_code, 409)
        threads_after = _wait_for_thread_count(threads_before, 5)
        self.assertLessEqual(
            threads_after, threads_before, [t.name for t in threading.enumerate()],
        )
        self.assertEqual(self.env.Bash("echo back", "container works").strip(), "back")

    def test_refusal_after_a_timeout_ends_the_reaper(self) -> None:
        """(4) A reaper installed before the refusal arrived exits once it does."""
        threads_before = threading.active_count()
        self._slow_exec_start(3, pause_container=True)
        result = self.env.Bash("sleep 48", "refused after the deadline", timeout_seconds=1)
        self.assertIn("timed out", result)
        threads_after = _wait_for_thread_count(threads_before, 15)
        self.assertLessEqual(
            threads_after, threads_before, [t.name for t in threading.enumerate()],
        )


if __name__ == "__main__":
    unittest.main()
