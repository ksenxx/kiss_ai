# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Race condition tests for ``kiss.agents.vscode``.

Each test first demonstrates a real data-race between two or more
threads in the current code path.  After the matching lock fix is
applied to production code the same test must pass consistently —
proving the race has been eliminated.

These tests use deterministic synchronisation harnesses (not mocks or
fakes of production behaviour) to force the exact interleaving that
exposes each race.  They avoid DB I/O and heavy agent machinery so
they can surface races reliably.
"""

from __future__ import annotations

import threading
import unittest

from kiss.agents.vscode.server import VSCodeServer

# The historical ``TestBroadcastOrderingRace`` exercised the
# ``_stdout_lock`` ordering guarantee inside the now-deleted
# ``VSCodePrinter``.  Under the single-daemon architecture the
# extension talks to ``kiss-web`` over a Unix-domain socket and there
# is no stdout transport — ``WebPrinter._send_to_ws_clients`` performs
# the socket-level writes through a dedicated asyncio loop, so the
# stdout-order invariant the old test pinned no longer exists.


class TestFileCacheOverwriteRace(unittest.TestCase):
    """``VSCodeServer._get_files`` must not overwrite a newer cache.

    ``_refresh_file_cache`` spawns a background thread that scans
    files and writes ``self._file_cache``.  If ``_get_files`` sees
    ``cache is None`` concurrently, it scans again and blindly writes
    its own (older) result under ``_state_lock`` without re-checking
    — a slower main-thread scan therefore replaces a fresher
    background result.

    The test forces this interleaving with two events: the main-thread
    scan starts and blocks until the background refresh publishes its
    fresher value, after which the stale scan returns.  A correct
    ``_get_files`` must NOT overwrite the already-published cache.
    """

    def test_background_refresh_is_not_overwritten(self) -> None:
        server = VSCodeServer()
        server._file_cache = {}
        server.printer.broadcast = lambda *_a, **_k: None  # type: ignore[method-assign]  # silence output

        wd = server.work_dir
        fresh = ["fresh/file.py"]
        scan_started = threading.Event()
        bg_done = threading.Event()

        def bg_refresh() -> None:
            scan_started.wait(timeout=5)
            with server._state_lock:
                server._file_cache[wd] = fresh
            bg_done.set()

        t = threading.Thread(target=bg_refresh, daemon=True)
        t.start()

        from kiss.agents.vscode import diff_merge

        original_scan = diff_merge._scan_files

        def sync_scan(_work_dir: str) -> list[str]:
            scan_started.set()
            bg_done.wait(timeout=5)
            return ["stale/file.py"]

        diff_merge._scan_files = sync_scan  # type: ignore[assignment]
        try:
            server._get_files("")
        finally:
            diff_merge._scan_files = original_scan  # type: ignore[assignment]
            t.join(timeout=2)

        self.assertEqual(
            server._file_cache.get(wd), fresh,
            "background refresh result must not be overwritten by a slower scan",
        )


if __name__ == "__main__":
    unittest.main()
