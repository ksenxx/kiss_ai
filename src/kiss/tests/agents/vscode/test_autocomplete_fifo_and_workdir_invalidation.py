# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Autocomplete safety regressions for the on-disk fallback (server audit).

Commit ``5a63abb8`` made ``read_active_file_head`` reachable again for
connections that report an ``activeFile`` without any
``activeFileContent``.  Two defects around that fallback are pinned
here, both end-to-end through the real :class:`VSCodeServer` command
handlers and the shared autocomplete worker (no mocks):

* ``activeFile`` is a client-supplied path and every connection shares
  ONE autocomplete worker.  A plain ``open()`` of a writer-less FIFO
  blocks forever, wedging ghost-text completion for every other VS
  Code window.  The fallback must reject non-regular files and open
  with ``O_NONBLOCK`` so no pathological path can block the worker.

* ``setWorkDir`` cleared the stored active-file snapshot but left the
  connection's ``_complete_seq_latest`` entry untouched, so an
  in-flight completion computed against the OLD workspace passed the
  worker's post-computation freshness check and emitted stale
  old-workspace identifiers after the switch.
"""

from __future__ import annotations

import os
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from kiss.server import agent_state
from kiss.server.autocomplete import read_active_file_head
from kiss.server.server import VSCodeServer


class TestAutocompleteFifoAndWorkDirInvalidation(unittest.TestCase):
    """End-to-end ``complete``/``setWorkDir`` → worker → event checks."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []
        self.server.printer.broadcast = self.events.append  # type: ignore[assignment]

    def tearDown(self) -> None:
        agent_state.agent_states.clear()
        self._tmp.cleanup()

    def _completions_for(
        self, query: str, timeout: float = 5.0,
    ) -> list[dict[str, str]]:
        """Return the completions the worker broadcast for *query*."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for ev in self.events:
                if ev.get("type") == "completions" and ev.get("query") == query:
                    return list(ev.get("completions", []))
            time.sleep(0.01)
        self.fail(f"no completions event for query {query!r} within {timeout}s")

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires POSIX FIFOs")
    def test_fifo_active_file_does_not_block_other_connections(self) -> None:
        """A FIFO ``activeFile`` must not wedge the shared worker.

        Connection A supplies a writer-less FIFO as its active file (no
        buffer snapshot, so the on-disk fallback is taken); connection
        B's ordinary live-buffer completion is queued behind it on the
        single worker.  Before the fix the worker blocked forever in
        ``open(fifo)`` and B's reply never arrived.
        """
        fifo = self.root / "wedge.fifo"
        os.mkfifo(fifo)
        self.server._cmd_complete({
            "type": "complete",
            "query": "fifoQu",
            "activeFile": str(fifo),
            "connId": "win-a",
        })
        self.server._cmd_complete({
            "type": "complete",
            "query": "unbloc",
            "activeFile": str(self.root / "live.py"),
            "activeFileContent": "unblockedIdentifier = 1\n",
            "connId": "win-b",
        })
        texts = [c["text"] for c in self._completions_for("unbloc")]
        self.assertIn(
            "unblockedIdentifier", texts,
            "a FIFO activeFile on another connection blocked the shared "
            "autocomplete worker",
        )
        # The FIFO request itself was answered (worker not wedged) and
        # yielded no identifier candidates from the non-regular file.
        replies_a = [
            ev for ev in self.events
            if ev.get("type") == "completions" and ev.get("query") == "fifoQu"
        ]
        self.assertTrue(replies_a, "the FIFO request itself was never answered")
        self.assertEqual(
            [c for c in replies_a[0].get("completions", [])
             if c.get("type") == "identifier"],
            [],
            "identifiers were harvested from a FIFO",
        )

    def test_non_regular_files_yield_empty_content(self) -> None:
        """``read_active_file_head`` rejects device nodes outright.

        ``/dev/zero`` never blocks but is an infinite stream; before
        the regular-file check the fallback happily read 50 000 NUL
        characters from it.
        """
        if os.path.exists("/dev/zero"):
            self.assertEqual(read_active_file_head("/dev/zero"), "")
        if os.path.exists("/dev/null"):
            self.assertEqual(read_active_file_head("/dev/null"), "")
        # Regular-file behaviour is unchanged.
        path = self.root / "plain.py"
        path.write_text("plainIdentifier = 1\n", encoding="utf-8")
        self.assertEqual(
            read_active_file_head(str(path)), "plainIdentifier = 1\n",
        )

    def test_set_work_dir_suppresses_in_flight_old_workspace_result(
        self,
    ) -> None:
        """``setWorkDir`` invalidates the connection's in-flight request.

        The test thread holds the (reentrant) ``_state_lock`` across
        enqueueing an old-workspace fallback request and the
        ``setWorkDir`` call: the worker cannot pass ``_complete``'s
        lock-guarded freshness checks until the lock is released, so
        the request is deterministically still in flight when the
        workspace switches.  Before the fix the request's sequence
        entry survived the switch, both freshness checks passed, and
        stale old-workspace identifiers were emitted.
        """
        old_root = self.root / "old_ws"
        old_root.mkdir()
        new_root = self.root / "new_ws"
        new_root.mkdir()
        old_file = old_root / "legacy.py"
        old_file.write_text("oldWorkspaceIdentifier = 1\n", encoding="utf-8")

        with self.server._state_lock:
            self.server._cmd_complete({
                "type": "complete",
                "query": "oldWork",
                "activeFile": str(old_file),
                "connId": "win-old",
            })
            self.server._cmd_set_work_dir({
                "type": "setWorkDir",
                "workDir": str(new_root),
                "connId": "win-old",
            })
        # Sentinel on a DIFFERENT connection (a same-connection request
        # would itself mark the old one stale and mask the regression).
        # The single worker answers in FIFO order, so once the
        # sentinel's reply arrives the old-workspace request has been
        # fully processed.
        self.server._cmd_complete({
            "type": "complete",
            "query": "sentin",
            "activeFile": str(new_root / "sentinel.py"),
            "activeFileContent": "sentinelIdentifier = 1\n",
            "connId": "win-sentinel",
        })
        texts = [c["text"] for c in self._completions_for("sentin")]
        self.assertIn("sentinelIdentifier", texts)
        stale = [
            ev for ev in self.events
            if ev.get("type") in ("completions", "ghost")
            and ev.get("query") == "oldWork"
        ]
        self.assertEqual(
            stale, [],
            "an in-flight old-workspace completion was emitted after "
            "setWorkDir switched the workspace",
        )


    def test_set_work_dir_return_fences_in_flight_publication(self) -> None:
        """No old-workspace event may be broadcast after ``setWorkDir`` returns.

        Closes the residual window between the worker's final freshness
        check and the actual event publication: with a check-then-emit
        outside the lock, ``setWorkDir`` could complete in that gap and
        the already-checked stale result still escaped afterwards.  The
        check and the emit are now atomic under ``_state_lock``, so
        ``setWorkDir`` (which takes the same lock) cannot return until
        an in-flight publication has finished.

        The broadcast hook widens the historical microsecond gap into a
        deterministic 0.3 s one: it signals when the worker has passed
        the freshness check and entered publication, then sleeps before
        recording the event.  The hook never blocks on the main thread,
        so it is deadlock-free whichever side of the lock it runs on.
        """
        old_root = self.root / "old_ws2"
        old_root.mkdir()
        new_root = self.root / "new_ws2"
        new_root.mkdir()
        old_file = old_root / "legacy.py"
        old_file.write_text("staleFenceIdentifier = 1\n", encoding="utf-8")

        in_emit = threading.Event()

        def gating_broadcast(event: dict[str, Any]) -> None:
            if event.get("query") == "staleFen" and not in_emit.is_set():
                in_emit.set()
                time.sleep(0.3)
            self.events.append(event)

        self.server.printer.broadcast = gating_broadcast  # type: ignore[assignment]
        self.server._cmd_complete({
            "type": "complete",
            "query": "staleFen",
            "activeFile": str(old_file),
            "connId": "win-fence",
        })
        self.assertTrue(
            in_emit.wait(5.0), "worker never reached publication",
        )
        # The worker is now inside publication, past every freshness
        # check.  A returned setWorkDir must fence it out.
        self.server._cmd_set_work_dir({
            "type": "setWorkDir",
            "workDir": str(new_root),
            "connId": "win-fence",
        })
        events_at_return = len(self.events)
        # Drain the worker with a sentinel on another connection.
        self.server._cmd_complete({
            "type": "complete",
            "query": "fenceSe",
            "activeFile": str(new_root / "sentinel.py"),
            "activeFileContent": "fenceSentinel = 1\n",
            "connId": "win-fence-sentinel",
        })
        texts = [c["text"] for c in self._completions_for("fenceSe")]
        self.assertIn("fenceSentinel", texts)
        stale_after_return = [
            ev for ev in self.events[events_at_return:]
            if ev.get("query") == "staleFen"
        ]
        self.assertEqual(
            stale_after_return, [],
            "an old-workspace autocomplete event was broadcast after "
            "setWorkDir had already returned",
        )


if __name__ == "__main__":
    unittest.main()
