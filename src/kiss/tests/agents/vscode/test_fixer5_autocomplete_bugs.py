# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Fixer-5 autocomplete bugs (findings F5-01, F5-02, F5-03).

F5-01 — an explicitly EMPTY live editor snapshot (``""``) used to be
treated as "no snapshot supplied", falling back to the older on-disk
file and resurrecting identifiers the user had deleted from the
unsaved buffer.  The disk fallback must only trigger when the snapshot
is genuinely unavailable (``None``).

F5-02 — request-sequence freshness was checked only BEFORE the
(potentially slow) completion computation, so a request that was
superseded mid-computation still broadcast its stale result.

F5-03 — a slow ``@``-mention directory scan for one work_dir could
finish after a newer ``getFiles`` on the same connection (different
tab / work_dir) and overwrite the newer picker contents with files
from the wrong workspace.
"""

from __future__ import annotations

import os
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from kiss.server.server import VSCodeServer
from kiss.server.task_runner import _RunningAgentState


class _AutocompleteHarness(unittest.TestCase):
    """Shared VSCodeServer + broadcast-capture setup."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []
        self.server.printer.broadcast = self.events.append  # type: ignore[assignment]

    def tearDown(self) -> None:
        _RunningAgentState.running_agent_states.clear()
        self._tmp.cleanup()


class TestEmptySnapshotIsNotDiskFallback(_AutocompleteHarness):
    """F5-01: ``snapshot_content=""`` means an empty editor, not "unknown"."""

    def _write_stale_file(self) -> str:
        path = self.root / "stale.py"
        path.write_text("staleIdentifier = 1\nstaleIdentifier2 = 2\n")
        return str(path)

    def test_empty_editor_buffer_yields_no_disk_identifiers(self) -> None:
        """An empty live buffer must not harvest deleted identifiers."""
        path = self._write_stale_file()
        matches = self.server._active_file_identifier_matches(
            "stale", snapshot_file=path, snapshot_content="",
        )
        self.assertEqual(
            matches, [],
            "identifiers were harvested from the on-disk file although "
            "the live editor buffer is explicitly empty",
        )

    def test_missing_snapshot_still_falls_back_to_disk(self) -> None:
        """``None`` (snapshot unavailable) keeps the on-disk fallback."""
        path = self._write_stale_file()
        matches = self.server._active_file_identifier_matches(
            "stale", snapshot_file=path, snapshot_content=None,
        )
        self.assertIn("staleIdentifier", matches)

    def test_live_buffer_wins_over_disk(self) -> None:
        """A non-empty live buffer is used verbatim, never the disk copy."""
        path = self._write_stale_file()
        matches = self.server._active_file_identifier_matches(
            "fresh", snapshot_file=path,
            snapshot_content="freshIdentifier = 3\n",
        )
        self.assertIn("freshIdentifier", matches)
        self.assertNotIn("staleIdentifier", matches)


class TestStaleCompleteDropped(_AutocompleteHarness):
    """F5-02: a request superseded mid-computation must not emit."""

    def test_superseded_request_emits_nothing(self) -> None:
        """Freshness is re-checked after the slow computation.

        The active-file read is blocked deterministically with a FIFO:
        the request thread blocks opening it for read until this test
        (after superseding the request) attaches a writer.
        """
        fifo = str(self.root / "blocked.py")
        os.mkfifo(fifo)
        conn_id = "conn-A"
        with self.server._state_lock:
            self.server._complete_seq_latest[conn_id] = 1

        worker = threading.Thread(
            target=self.server._complete,
            args=("stale", 1, fifo, None, "", conn_id),
            daemon=True,
        )
        worker.start()
        # Let the request pass its entry freshness check and block on
        # the FIFO open inside the identifier harvest.
        time.sleep(0.3)
        self.assertTrue(worker.is_alive(), "request should be blocked on the FIFO")
        # A newer request arrives on the same connection.
        with self.server._state_lock:
            self.server._complete_seq_latest[conn_id] = 2
        # Unblock the old request's file read.
        fd = os.open(fifo, os.O_WRONLY)
        os.write(fd, b"staleIdentifier = 1\n")
        os.close(fd)
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive())
        emitted_types = [e.get("type") for e in self.events]
        self.assertNotIn(
            "ghost", emitted_types,
            f"stale request broadcast a ghost suggestion: {self.events!r}",
        )
        self.assertNotIn(
            "completions", emitted_types,
            f"stale request broadcast completions: {self.events!r}",
        )

    def test_fresh_request_still_emits(self) -> None:
        """The post-computation check lets an un-superseded request emit."""
        path = self.root / "live.py"
        path.write_text("liveIdentifier = 1\n")
        conn_id = "conn-B"
        with self.server._state_lock:
            self.server._complete_seq_latest[conn_id] = 7
        self.server._complete("live", 7, str(path), None, "", conn_id)
        emitted_types = [e.get("type") for e in self.events]
        self.assertIn("ghost", emitted_types)
        self.assertIn("completions", emitted_types)


class TestStaleFilePickerReplyDropped(_AutocompleteHarness):
    """F5-03: a superseded ``getFiles`` scan must not emit its reply."""

    def _wait(self, predicate: Any, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        raise AssertionError(f"timed out; events: {self.events[-5:]!r}")

    def test_slow_scan_does_not_overwrite_newer_workdir(self) -> None:
        """Same prefix, same connection, two work_dirs: old reply dropped."""
        slow_dir = self.root / "slow"
        slow_dir.mkdir()
        for i in range(4000):
            (slow_dir / f"filler_{i:05d}.py").write_text("x = 1\n")
        (slow_dir / "marker_slow.py").write_text("x = 1\n")
        fast_dir = self.root / "fast"
        fast_dir.mkdir()
        (fast_dir / "marker_fast.py").write_text("y = 2\n")

        conn_id = "conn-C"
        # No pause between the two requests: the second must supersede
        # the first while the first's directory scan is still running.
        self.server._get_files("marker", work_dir=str(slow_dir), conn_id=conn_id)
        self.server._get_files("marker", work_dir=str(fast_dir), conn_id=conn_id)

        def _both_scans_done() -> bool:
            with self.server._state_lock:
                return (
                    str(slow_dir) in self.server._file_cache
                    and str(fast_dir) in self.server._file_cache
                )

        self._wait(_both_scans_done)
        # Allow any (buggy) post-scan emission to land.
        time.sleep(0.3)
        populated = [
            e for e in self.events
            if e.get("type") == "files" and e.get("files")
        ]
        self.assertTrue(populated, f"no populated files reply: {self.events!r}")
        for ev in populated:
            texts = [f.get("text", "") for f in ev["files"]]
            self.assertFalse(
                any("marker_slow" in t for t in texts),
                "a stale scan for the superseded work_dir was emitted "
                f"and would overwrite the newer picker: {texts[:3]!r}",
            )
        final_texts = [f.get("text", "") for f in populated[-1]["files"]]
        self.assertTrue(
            any("marker_fast" in t for t in final_texts), final_texts,
        )


if __name__ == "__main__":
    unittest.main()
