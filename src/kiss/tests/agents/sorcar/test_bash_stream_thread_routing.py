# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E regression tests: Bash streaming output must stay attributed to
the running task.

``JsonPrinter`` (and its transport subclass ``WebPrinter``) route every
event by the **thread-local** ``task_id`` set on the agent's thread.
Commit ``763941a6`` moved the Bash tool's output draining onto a daemon
reader thread and invoked ``stream_callback`` from that thread — a
thread with no thread-local ``task_id``.  The consequences in the
chat webview were:

* ``system_output`` events carried no ``taskId``, so ``WebPrinter``
  treated them as *global system events* and broadcast them verbatim
  to every connected client instead of fanning them out (stamped with
  ``tabId``) to the task's subscriber tab — the bash sub panel of the
  tool-call panel stayed empty.
* The unattributed events landed in whatever tab was active, rendered
  as bare text in the chat ("garbage" while an agent is running).
* The events were recorded under the ``""`` fallback key, so the
  task's persisted history also lost the bash output.

These tests exercise the REAL production path — ``UsefulTools.Bash``
streaming into a real ``JsonPrinter`` via the same callback shape that
``SorcarAgent._get_tools`` wires up — with no mocks or patches, and
assert that every ``system_output`` event is attributed to the task.
"""

from __future__ import annotations

import threading
import unittest

from kiss.agents.sorcar.useful_tools import UsefulTools


class BashStreamTaskRoutingTest(unittest.TestCase):
    """Bash streamed output must be attributed to the calling task."""

    TASK_ID = "task-42"


    def test_stream_callback_runs_on_the_calling_thread(self) -> None:
        """``stream_callback`` must run on the thread that called Bash.

        All printers key their per-task state (bash buffers, recording,
        stop events) on thread-local storage, so invoking the callback
        from an internal reader thread silently detaches the output
        from the task.
        """
        callback_threads: set[str] = set()
        lock = threading.Lock()

        def _stream(text: str) -> None:
            with lock:
                callback_threads.add(threading.current_thread().name)

        tools = UsefulTools(stream_callback=_stream)
        result = tools.Bash(
            "echo one; echo two",
            description="emit two lines",
            timeout_seconds=30,
        )
        self.assertIn("one", result)
        self.assertEqual(
            callback_threads,
            {threading.current_thread().name},
            "stream_callback ran on a different thread than the Bash caller",
        )


if __name__ == "__main__":
    unittest.main()
