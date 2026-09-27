# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Race condition tests for ``kiss.server``.

Each test first demonstrates a real data-race between two or more
threads in the current code path.  After the matching lock fix is
applied to production code the same test must pass consistently —
proving the race has been eliminated.

These tests use deterministic synchronisation harnesses (not mocks or
fakes of production behaviour) to force the exact interleaving that
exposes each race.  They avoid DB I/O and heavy agent machinery so
they can surface races reliably.

The former ``TestFileCacheOverwriteRace`` (a main-thread ``_scan_files``
overwriting a fresher background scan of ``_file_cache``) is gone with
its subject: ``FileIndexRegistry`` builds every index on one worker
thread, so two scans of a root can no longer run concurrently.
:class:`TestOverlappingIndexRequests` covers what remains of that
intent — overlapping requests for one root must leave exactly one
index that matches the disk.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

from kiss.server.file_index import FileIndexRegistry


class TestOverlappingIndexRequests(unittest.TestCase):
    """Overlapping ``ensure``/``refresh`` calls for one root converge on
    a single index reflecting the files on disk.

    The worker is held after the first build by a gate callback so a
    second ``ensure`` (from another thread) and a ``refresh`` (from the
    main thread) are queued while the first result is already
    published and the tree has changed underneath it.  Once the gate
    opens the queued jobs must rebuild — not drop the request or keep
    the stale result — and the registry must hold exactly one index
    for the root, listing the current files.
    """

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-index-race-"))
        self.root = self.tmpdir / "root"
        for rel in ("a.py", "sub/b.py"):
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.root / rel).write_text("")
        self.registry = FileIndexRegistry(
            home=str(self.tmpdir / "home"), cache_dir=self.tmpdir / "cache",
        )

    def tearDown(self) -> None:
        self.registry.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_overlapping_requests_leave_one_consistent_index(self) -> None:
        reg = self.registry
        root = str(self.root)
        first_built = threading.Event()
        gate = threading.Event()

        def hold_worker() -> None:
            first_built.set()
            gate.wait(timeout=10)

        reg.ensure(root, hold_worker)
        self.assertTrue(first_built.wait(10), "first build never finished")
        first = reg.view_for(root)
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(first.paths, ["a.py", "sub/", "sub/b.py"])

        # The worker is parked inside ``hold_worker``.  Change the tree
        # and queue two overlapping requests for the same root from two
        # threads while the first index is still the published one.
        time.sleep(0.02)  # coarse mtime clocks: the directory must look changed
        (self.root / "c.py").write_text("")
        (self.root / "sub" / "b.py").unlink()
        second_done = threading.Event()
        t = threading.Thread(target=reg.ensure, args=(root, second_done.set), daemon=True)
        t.start()
        reg.refresh(root)
        t.join(timeout=5)
        self.assertFalse(second_done.is_set(), "second build ran while the worker was gated")

        gate.set()
        self.assertTrue(second_done.wait(10), "queued ensure never completed")
        # Drain the refresh job too (FIFO: this callback runs after it).
        drained = threading.Event()
        reg.ensure(root, drained.set)
        self.assertTrue(drained.wait(10))

        self.assertEqual(list(reg._indexes), [root], "exactly one index per root")
        view = reg.view_for(root)
        assert view is not None
        self.assertEqual(view.paths, ["a.py", "c.py", "sub/"])
        on_disk = sorted(
            str(p.relative_to(self.root)) + ("/" if p.is_dir() else "")
            for p in self.root.rglob("*")
        )
        self.assertEqual(sorted(view.paths), on_disk)


if __name__ == "__main__":
    unittest.main()
