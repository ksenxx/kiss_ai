# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Server-only tests extracted from
``kiss.tests.agents.sorcar.test_bash_stream_thread_routing``.

Moved here because their full dependency closure touches only
kiss.agents.sorcar and kiss.server (task: relocate
core+sorcar+server-only test methods to tests/server).  The
calling-thread test, which does not exercise kiss.server, stays in
the original module.
"""

from __future__ import annotations

import functools
import itertools
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import task_digest
from kiss.agents.sorcar.docker_manager import _timeout_message, _with_exit_code
from kiss.agents.sorcar.useful_tools import UsefulTools, sleep_alternative
from kiss.server.json_printer import JsonPrinter


class _CapturingPrinter(JsonPrinter):
    """JsonPrinter that additionally captures every broadcast event.

    Per project test policy the transport is captured by subclassing
    ``broadcast()`` (see ``test_printer_equivalence.py``); the base
    class recording/persistence path still runs unchanged.
    """

    def __init__(self) -> None:
        super().__init__()
        self.captured: list[dict[str, Any]] = []
        self._captured_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record the (taskId-injected) event and capture a copy."""
        event = self._inject_task_id(event)
        with self._captured_lock:
            self.captured.append(dict(event))
        super().broadcast(event)


class BashStreamTaskRoutingTest(unittest.TestCase):
    """Bash streamed output must be attributed to the calling task."""

    TASK_ID = "task-42"

    def _run_bash_with_streaming(
        self,
    ) -> tuple[_CapturingPrinter, list[dict[str, Any]], str]:
        """Run a streaming Bash command through a real JsonPrinter.

        Mirrors the production wiring: the agent thread sets the
        thread-local ``task_id``, starts recording, and passes a
        ``stream_callback`` that calls ``printer.print(...,
        type="bash_stream")`` — exactly what
        ``SorcarAgent._get_tools._stream`` does.

        Returns:
            Tuple of (printer, recorded display events, Bash result).
        """
        printer = _CapturingPrinter()
        printer._thread_local.task_id = self.TASK_ID
        printer.start_recording()

        def _stream(text: str) -> None:
            printer.print(text, type="bash_stream")

        tools = UsefulTools(stream_callback=_stream)
        result = tools.Bash(
            "echo hello_stream; sleep 0.25; echo world_stream",
            description="stream two lines with a pause",
            timeout_seconds=30,
        )
        printer.print("Read", type="tool_call", tool_input={})
        events = printer.stop_recording()
        return printer, events, result

    def test_streamed_output_is_recorded_under_the_task(self) -> None:
        """The task's recording must contain the streamed bash output.

        Before the fix the reader thread recorded ``system_output``
        under the ``""`` fallback key, so the task's own recording —
        what the chat page replays into the tool panel's bash sub panel
        and what gets persisted to history — was missing the output.
        """
        _, events, result = self._run_bash_with_streaming()
        self.assertIn("hello_stream", result)
        self.assertIn("world_stream", result)
        sys_out = "".join(
            e.get("text", "") for e in events if e.get("type") == "system_output"
        )
        self.assertIn("hello_stream", sys_out)
        self.assertIn("world_stream", sys_out)

    def test_streamed_events_carry_the_task_id(self) -> None:
        """Every broadcast ``system_output`` event must carry ``taskId``.

        An event without ``taskId`` is treated by ``WebPrinter`` as a
        global system event and is sent verbatim to EVERY connected
        client — that is the "garbage in the chat page" symptom.
        """
        printer, _, _ = self._run_bash_with_streaming()
        sys_events = [
            e for e in printer.captured if e.get("type") == "system_output"
        ]
        self.assertTrue(sys_events, "no system_output events were broadcast")
        for ev in sys_events:
            self.assertEqual(
                ev.get("taskId"),
                self.TASK_ID,
                f"system_output event lost its task attribution: {ev!r}",
            )


class StreamedResultStatusTest(unittest.TestCase):
    """A call streamed whole records, as its result, what the tool wrote around the output.

    The stream carries the command's output only.  The result of such
    a call used to be recorded blank, so the transcript of a failed, a
    timed-out or a left-running command did not say so, and the
    progress-log digest replayed a failed here-document copy as made.
    """

    def setUp(self) -> None:
        self.printer = _CapturingPrinter()
        self.printer._thread_local.task_id = "task-status"
        self.printer.start_recording()
        self.tools = UsefulTools(
            stream_callback=functools.partial(self.printer.print, type="bash_stream"),
        )

    def _record(self, result: str) -> str:
        """Print *result* as the ``Bash`` result ending the call; return the content recorded."""
        self.printer.print(result, type="tool_result", tool_name="Bash", is_error=False)
        return str(self.printer.captured[-1]["content"])

    def _bash(self, **bash: Any) -> str:
        """Make a real ``Bash`` call as the agent does; return the content its result records."""
        self.printer.print("Bash", type="tool_call", tool_input=bash)
        self.returned = self.tools.Bash(description="status", **bash)
        return self._record(self.returned)

    def _waited(self) -> str:
        """Return the note of a long sleep spent waiting on a job, as the tools write it."""
        self.tools.Bash("sleep 0.2", "job", background=True)
        note = self.tools.Bash("sleep 40; true", "waits", timeout_seconds=60)
        # On a line of its own (audit 2026-11-21 G1): glued to the result, it joined the
        # exit code, an ``until`` verdict or the output's last line.
        shape = r"\A\nWaited on job \w+ for up to 40 s instead of sleeping \(.+\)\.\n\Z"
        self.assertRegex(note, shape)
        return note

    def test_the_host_tools_status_is_kept(self) -> None:
        self.assertEqual(self._bash(command="echo out; exit 3"), "Error (exit code 3):")
        self.assertEqual(
            self._bash(command="echo out; sleep 30", timeout_seconds=1),
            "Error: Command execution timeout after 1s. Output before the timeout:",
        )
        left = self._bash(command="echo out; sleep 30", timeout_seconds=1, on_timeout="background")
        self.tools.bash_job(self.returned.split()[9], action="kill")
        self.assertRegex(left, r"\AStill running after 1s; left running as background job \w+ .+\Z")
        # Output is the stream's, however it reads: recorded again it would show twice.
        self.assertEqual(self._bash(command="echo 'Error: disk full'; echo '[exit code: 1]'"), "")
        for line in ("REFUSED: the command did not run: no", "Error: the command did not run: no."):
            self.assertEqual(self._bash(command=f"echo '{line}'"), "")
        # A refusal follows no stream (the tools refuse before anything runs): recorded whole.
        self.assertEqual(self._bash(command="sleep 600", timeout_seconds=10), self.returned)
        self.assertTrue(self.returned.startswith("REFUSED: the Bash call did NOT run"))

    def test_the_container_tools_status_is_kept(self) -> None:
        timed_out = "Error: command timed out after 5s and was killed. Output before the timeout:"
        polled = _with_exit_code("out", 2) + "\n[until 'x': not matched after 2 runs]"
        # What the container ``Bash`` puts after the result of a command with a long sleep
        # (review L2/a: the status before such a note was dropped).
        slept, waited = sleep_alternative(40), self._waited()
        for result, status in (
            (_with_exit_code("out\n", 2), "[exit code: 2]"),
            (polled, "[exit code: 2]"),
            (_timeout_message(5, "out\n"), timed_out),
            (_with_exit_code("out\n", 2) + slept, "[exit code: 2]"),
            (_with_exit_code("out\n", 2) + waited, "[exit code: 2]"),
            (polled + slept, "[exit code: 2]"),
            (polled + waited, "[exit code: 2]"),
            ("out\n" + slept, ""),  # passed: a note is no status
            ("out\n" + waited, ""),
        ):
            self.printer.print("out\n", type="bash_stream")
            self.assertEqual(self._record(result), status)

    def test_a_failed_here_document_copy_is_not_replayed_as_the_progress_log(self) -> None:
        with tempfile.TemporaryDirectory() as created:
            folder = Path(created).as_posix()  # for bash: a backslash of Windows is its escape
            made, failed = f"{folder}/PROGRESS.md", f"{folder}/none/PROGRESS.md"
            for target, log in ((made, "plan\n"), (failed, "")):
                self._bash(command=f"cat > {target} <<'EOF'\nplan\nEOF\n")
                events = self.printer.stop_recording()
                self.assertEqual(task_digest.progress_log_tail({"events": events}), log, target)
                self.printer.start_recording()
            # As the container tool ends the same two commands after a long sleep (review L2/a),
            # with either note; without output nothing is streamed and the note is recorded too.
            notes = (sleep_alternative(40), self._waited())
            for (target, code, log), output, note in itertools.product(
                ((made, 0, "plan\n"), (failed, 1, "")), ("out\n", ""), notes,
            ):
                echo = "echo out; " if output else ""
                call = {"command": f"sleep 40; {echo}cat > {target} <<'EOF'\nplan\nEOF\n"}
                self.printer.print("Bash", type="tool_call", tool_input=call)
                if output:
                    self.printer.print(output, type="bash_stream")
                recorded = self._record(_with_exit_code(output, code) + note)
                self.assertEqual(note in recorded, not output, recorded)
                events = self.printer.stop_recording()
                self.assertEqual(
                    task_digest.progress_log_tail({"events": events}), log, (target, recorded),
                )
                self.printer.start_recording()
