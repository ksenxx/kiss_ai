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

import io
import json
import sys
import threading
import time
import unittest

from kiss.agents.vscode.server import VSCodePrinter, VSCodeServer


class TestBroadcastOrderingRace(unittest.TestCase):
    """``VSCodePrinter.broadcast`` must keep recording order == stdout order.

    The original implementation acquired ``_lock`` (for recording) and
    ``_stdout_lock`` (for stdout) in two separate critical sections,
    so two concurrent broadcasts could interleave and produce a
    recording order that differed from the stdout order.

    This test forces the race deterministically by wrapping
    ``printer._stdout_lock`` so the first thread to reach the lock is
    suspended BEFORE it actually acquires it — letting the second
    thread complete its full broadcast first and thereby get its
    stdout write out before the first thread's.
    """

    def test_recording_matches_stdout_order_under_concurrency(self) -> None:
        printer = VSCodePrinter()
        printer.start_recording()

        inner_lock = printer._stdout_lock

        class _OrderingLock:
            """Context-manager wrapper that delays the first thread.

            The first thread to call ``__enter__`` is held before
            actually acquiring ``inner_lock`` until ``allow_first``
            is signalled — any later thread proceeds immediately.
            """

            def __init__(self) -> None:
                self.first_ident: int | None = None
                self.first_blocked = threading.Event()
                self.allow_first = threading.Event()

            def __enter__(self) -> None:
                tid = threading.get_ident()
                if self.first_ident is None:
                    self.first_ident = tid
                    self.first_blocked.set()
                    self.allow_first.wait(timeout=5)
                inner_lock.acquire()

            def __exit__(self, *_a: object) -> None:
                inner_lock.release()

        ordering = _OrderingLock()
        printer._stdout_lock = ordering  # type: ignore[assignment]

        captured: list[dict] = []
        capture_lock = threading.Lock()

        class _CapturingStdout(io.StringIO):
            def write(self, s: str) -> int:  # type: ignore[override]
                for line in s.splitlines():
                    if not line:
                        continue
                    with capture_lock:
                        captured.append(json.loads(line))
                return len(s)

            def flush(self) -> None:  # type: ignore[override]
                pass

        orig_stdout = sys.stdout
        sys.stdout = _CapturingStdout()
        try:
            def t1_run() -> None:
                printer.broadcast({"type": "system_prompt", "text": "first"})

            def t2_run() -> None:
                # Wait until T1 has finished its _lock critical section
                # and is suspended inside the _stdout_lock wrapper.
                ordering.first_blocked.wait(timeout=5)
                printer.broadcast({"type": "system_prompt", "text": "second"})

            t1 = threading.Thread(target=t1_run)
            t2 = threading.Thread(target=t2_run)
            t1.start()
            t2.start()
            # Give T1 a head start to reach and block inside the
            # wrapper, and T2 a chance to progress as far as it can
            # (all the way through broadcast pre-fix; blocked on
            # ``_lock`` post-fix).  Then release T1.
            ordering.first_blocked.wait(timeout=5)
            time.sleep(0.1)
            ordering.allow_first.set()
            t1.join(timeout=5)
            t2.join(timeout=5)
        finally:
            sys.stdout = orig_stdout
            printer._stdout_lock = inner_lock

        recorded = printer.stop_recording()
        # With split-lock broadcast: recorded = [first, second] but
        # stdout captured = [second, first] — race confirmed.
        # With a single-lock (fixed) broadcast: both orders match.
        self.assertEqual(len(recorded), 2)
        self.assertEqual(len(captured), 2)
        self.assertEqual(
            [e["text"] for e in recorded],
            [e["text"] for e in captured],
            "recording order must equal stdout write order",
        )


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
        server._file_cache = None
        server.printer.broadcast = lambda *_a, **_k: None  # type: ignore[method-assign]  # silence output

        fresh = ["fresh/file.py"]
        scan_started = threading.Event()
        bg_done = threading.Event()

        def bg_refresh() -> None:
            scan_started.wait(timeout=5)
            with server._state_lock:
                server._file_cache = fresh
            bg_done.set()

        t = threading.Thread(target=bg_refresh, daemon=True)
        t.start()

        # Install a module-level ``_scan_files`` whose only difference
        # from the real one is that it blocks on events so the
        # interleaving is deterministic.  It is a real function (not
        # a mock of scanning behaviour) that returns a deterministic
        # value after the synchronisation completes.
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
            server._file_cache, fresh,
            "background refresh result must not be overwritten by a slower scan",
        )


if __name__ == "__main__":
    unittest.main()
