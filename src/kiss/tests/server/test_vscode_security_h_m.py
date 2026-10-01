# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for HIGH (H1-H10) and MEDIUM (M1-M5) severity fixes
in src/kiss/agents/vscode/.  Each Python-side fix has a behavioural test
that fails when the fix is reverted.

TS-side fixes (DependencyInstaller, SorcarSidebarView, kissPaths,
SorcarTab) are spot-checked via source-grep tests because the test
harness has no TypeScript runtime.

(M5 covered ``_save_untracked_base``/``_diff_files`` of the interactive
diff/merge review workflow; that workflow was removed from the server,
so those tests are gone.)
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from kiss.tests.conftest import posix_only


class TestH9AutocompleteNonBlocking(unittest.TestCase):
    """``getFiles`` must return promptly even while the index worker is busy.

    The ``@``-mention index is built on ``FileIndexRegistry``'s single
    worker thread.  A ``getFiles`` for a root that is not indexed yet
    must therefore answer at once with a ``loading`` placeholder and
    deliver the populated list later — never run the scan on the
    message-handling thread or wait for the worker to become free.
    """

    def test_get_files_does_not_block_while_worker_is_busy(self) -> None:
        from kiss.server.file_index import FileIndexRegistry
        from kiss.server.server import VSCodeServer

        tmpdir = Path(tempfile.mkdtemp(prefix="kiss-h9-"))
        work_dir = tmpdir / "work"
        other_root = tmpdir / "other"
        for rel in ("work/a.py", "work/b/c.py", "other/z.py"):
            (tmpdir / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmpdir / rel).write_text("")

        broadcasts: list[dict] = []
        lock = threading.Lock()

        def capture(msg: dict[str, Any]) -> None:
            with lock:
                broadcasts.append(dict(msg))

        server = VSCodeServer()
        server.printer.broadcast = capture  # type: ignore[method-assign, assignment]
        server.work_dir = str(work_dir)
        server._file_index.stop()
        registry = FileIndexRegistry(home=str(tmpdir / "home"), cache_dir=tmpdir / "cache")
        server._file_index = registry
        try:
            # Park the worker: it builds ``other_root`` and then blocks in
            # the callback until the gate opens, so a build of ``work_dir``
            # cannot start before that.
            worker_parked = threading.Event()
            gate = threading.Event()

            def park_worker() -> None:
                worker_parked.set()
                gate.wait(timeout=10)

            registry.ensure(str(other_root), park_worker)
            self.assertTrue(worker_parked.wait(5.0), "worker never reached the gate")

            t0 = time.monotonic()
            server._handle_command(
                {"type": "getFiles", "prefix": "a", "workDir": str(work_dir)},
            )
            dt = time.monotonic() - t0
            self.assertLess(dt, 0.5,
                            f"getFiles blocked for {dt:.2f}s while the index worker was busy")
            with lock:
                seen = list(broadcasts)
            self.assertEqual(len(seen), 1, f"expected only the loading placeholder; got {seen}")
            self.assertEqual(seen[0]["type"], "files")
            self.assertTrue(seen[0].get("loading"))
            self.assertEqual(seen[0]["files"], [])
            self.assertEqual(seen[0]["prefix"], "a")

            gate.set()
            deadline = time.monotonic() + 5.0
            populated: list[dict] = []
            while time.monotonic() < deadline and not populated:
                with lock:
                    populated = [
                        m for m in broadcasts
                        if m["type"] == "files" and not m.get("loading")
                    ]
                time.sleep(0.01)
            self.assertEqual(
                len(populated), 1,
                f"no populated reply after the gate opened: {broadcasts}",
            )
            self.assertEqual(populated[0]["prefix"], "a")
            self.assertEqual([f["text"] for f in populated[0]["files"]], ["./a.py"])
        finally:
            registry.stop()
            shutil.rmtree(tmpdir, ignore_errors=True)



class TestM1GitHasTimeout(unittest.TestCase):
    """``_git`` must abort a hung git instead of blocking forever.

    Asserted through behaviour rather than through the shape of the
    subprocess call: the server modules call the single hardened
    runner ``git_worktree._git`` (``Popen`` + ``killpg``) directly, so
    a test that spied on ``subprocess.run``'s keyword arguments was
    pinning an implementation that no longer exists.
    """

    def _install_hanging_git(self, tmp_path: Path) -> Path:
        """Create a stub ``git`` that never returns, and return its dir."""
        bin_dir = tmp_path / "stub-bin"
        bin_dir.mkdir()
        stub = bin_dir / "git"
        stub.write_text("#!/bin/sh\nsleep 60\n", encoding="utf-8")
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return bin_dir

    @posix_only("the hanging git stub on PATH is a /bin/sh script")
    def test_hanging_git_is_abandoned_within_the_timeout(self) -> None:
        """A hung git yields returncode 124 well before it exits."""
        from kiss.agents.sorcar import git_worktree

        tmpdir = tempfile.mkdtemp(prefix="kiss-m1-timeout-")
        try:
            bin_dir = self._install_hanging_git(Path(tmpdir))
            saved_path = os.environ["PATH"]
            saved_timeout = git_worktree._GIT_TIMEOUT_SECONDS
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{saved_path}"
            git_worktree._GIT_TIMEOUT_SECONDS = 1.0
            try:
                start = time.monotonic()
                result = git_worktree._git("status", cwd=tmpdir)
                elapsed = time.monotonic() - start
            finally:
                os.environ["PATH"] = saved_path
                git_worktree._GIT_TIMEOUT_SECONDS = saved_timeout
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        self.assertIsInstance(result, subprocess.CompletedProcess)
        self.assertEqual(result.returncode, 124,
                         f"expected the timeout returncode: {result}")
        self.assertLess(elapsed, 30,
                        f"_git blocked for {elapsed:.1f}s despite a 1s budget")

    def test_normal_git_still_succeeds(self) -> None:
        """The timeout protection does not disturb a healthy command."""
        from kiss.agents.sorcar.git_worktree import _git

        tmpdir = tempfile.mkdtemp(prefix="kiss-m1-ok-")
        try:
            self.assertEqual(_git("init", "-q", cwd=tmpdir).returncode, 0)
            self.assertEqual(
                _git("status", "--porcelain", cwd=tmpdir).returncode, 0,
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestM4AwaitUserResponseEmptyQueue(unittest.TestCase):
    """When the tab has no answer queue (e.g. closed mid-question), the
    wait method must raise ``KeyboardInterrupt`` instead of looping forever."""

    def test_returns_promptly_when_queue_is_none(self) -> None:
        from kiss.server import task_runner as tr

        class FakePrinter:
            class TL:
                pass
            _thread_local = TL()
            _lock = threading.Lock()
            _subscribers: dict[str, set[str]] = {}

        class FakeServer(tr._TaskRunnerMixin):
            def __init__(self) -> None:
                self.printer = FakePrinter()  # type: ignore[assignment]
                self.printer._thread_local.stop_event = threading.Event()
                self.printer._thread_local.task_id = "ghost-tab"
                self._state_lock = threading.RLock()

        srv = FakeServer()
        t0 = time.time()
        with self.assertRaises(KeyboardInterrupt):
            srv._await_user_response()
        dt = time.time() - t0
        self.assertLess(dt, 1.0,
                        f"_await_user_response took {dt:.2f}s with no queue — "
                        "must raise immediately, not loop")





if __name__ == "__main__":
    unittest.main()
