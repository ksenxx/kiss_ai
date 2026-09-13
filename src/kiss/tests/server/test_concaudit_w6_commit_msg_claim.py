# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""W6 concurrency audit: ``generateCommitMessage`` claim survives a failed spawn.

``_cmd_generate_commit_message`` publishes the tab's in-flight claim
(``_commit_msg_tabs``) under ``_state_lock`` and then starts the worker
thread whose ``finally`` releases it.  When ``Thread.start()`` itself
raises — ``RuntimeError: can't start new thread`` under thread/process
exhaustion — the worker never runs, so nothing ever released the claim
and every later request for that tab was dropped as a "duplicate".

The failure is reproduced for real by lowering ``RLIMIT_NPROC`` to 1
for the duration of the call (threads count against the limit for a
non-root uid), the same way the daemon would hit it under exhaustion.
No mocks or patches are involved.
"""

from __future__ import annotations

import os
import resource
import tempfile
import threading
import time
import unittest

import pytest

from kiss.server.server import VSCodeServer
from kiss.tests.server._memory_printer import MemoryPrinter


def _thread_start_can_be_starved() -> bool:
    """True when lowering RLIMIT_NPROC actually makes Thread.start fail here."""
    if os.getuid() == 0:
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


class TestCommitMessageClaimReleasedOnSpawnFailure(unittest.TestCase):
    """A failed worker spawn must not wedge the tab's commit-message button."""

    def test_claim_released_when_thread_start_raises(self) -> None:
        if not _thread_start_can_be_starved():
            pytest.skip("RLIMIT_NPROC cannot starve Thread.start on this host")
        printer = MemoryPrinter()
        server = VSCodeServer(printer=printer)
        tab_id = "TAB-W6-COMMIT"
        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        resource.setrlimit(resource.RLIMIT_NPROC, (1, hard))
        try:
            with self.assertRaises(RuntimeError):
                server._cmd_generate_commit_message({
                    "type": "generateCommitMessage", "tabId": tab_id,
                })
        finally:
            resource.setrlimit(resource.RLIMIT_NPROC, (soft, hard))

        with server._state_lock:
            self.assertNotIn(tab_id, server._commit_msg_tabs)

        # The tab is usable again: the retry dispatches a worker, which
        # reports on a non-git folder and re-arms the claim on exit.
        with tempfile.TemporaryDirectory() as non_git_dir:
            server._cmd_generate_commit_message({
                "type": "generateCommitMessage",
                "tabId": tab_id,
                "workDir": non_git_dir,
            })
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                with server._state_lock:
                    if tab_id not in server._commit_msg_tabs:
                        break
                time.sleep(0.01)
        replies = [
            e for e in printer.emitted
            if e.get("type") == "commitMessage" and e.get("tabId") == tab_id
        ]
        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0].get("error"), "Not a git repository.")
        with server._state_lock:
            self.assertNotIn(tab_id, server._commit_msg_tabs)


if __name__ == "__main__":
    unittest.main()
