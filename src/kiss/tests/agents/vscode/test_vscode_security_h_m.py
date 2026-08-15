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
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


@unittest.skipIf(sys.platform == "win32", "POSIX-only file permissions test")
class TestH3RcFilePermissionsAndQuoting(unittest.TestCase):
    """``save_api_key_to_shell`` writes RC with mode 0600 and shell-quotes value."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self._home_patch = mock.patch.dict(
            os.environ, {"HOME": str(self.home), "SHELL": "/bin/bash"},
        )
        self._home_patch.start()
        from kiss.core import vscode_config as vc

        self._vc = vc
        self._orig_rc_path = vc._shell_rc_path
        vc._shell_rc_path = lambda shell: self.home / ".bashrc"  # type: ignore[assignment]
        self._refresh_patch = mock.patch.object(vc, "_refresh_config", lambda: None)
        self._refresh_patch.start()

    def tearDown(self) -> None:
        self._vc._shell_rc_path = self._orig_rc_path  # type: ignore[assignment]
        self._refresh_patch.stop()
        self._home_patch.stop()
        self._tmp.cleanup()

    def test_rc_file_is_mode_0600_after_write(self) -> None:
        """RC file must be created with 0600 permissions, not 0644."""
        self._vc.save_api_key_to_shell("OPENAI_API_KEY", "sk-secret-12345")
        rc = self.home / ".bashrc"
        self.assertTrue(rc.exists())
        mode = stat.S_IMODE(rc.stat().st_mode)
        self.assertEqual(mode, 0o600,
                         f"RC file mode should be 0600, got {oct(mode)}")

    def test_rc_file_mode_preserved_when_overwriting_existing_key(self) -> None:
        """A pre-existing entry update keeps file mode at 0600 (or stricter)."""
        self._vc.save_api_key_to_shell("OPENAI_API_KEY", "old-key")
        self._vc.save_api_key_to_shell("OPENAI_API_KEY", "new-key")
        rc = self.home / ".bashrc"
        mode = stat.S_IMODE(rc.stat().st_mode)
        self.assertFalse(mode & 0o077,
                         f"RC mode {oct(mode)} leaks group/other read bits")

    def test_value_with_double_quote_is_quoted_safely(self) -> None:
        """A key value containing `"` must not break out of its quotes."""
        evil = 'a"b$IFS$(echo pwned > /tmp/h3-pwned)c'
        self._vc.save_api_key_to_shell("OPENAI_API_KEY", evil)
        rc_text = (self.home / ".bashrc").read_text()
        proc = subprocess.run(
            ["bash", "-c", f"source '{self.home / '.bashrc'}' && printf '%s' \"$OPENAI_API_KEY\""],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertEqual(proc.stdout, evil,
                         f"Value did not round-trip; rc was:\n{rc_text}")
        self.assertFalse(Path("/tmp/h3-pwned").exists(),
                         "Command substitution executed during source!")

    def test_value_with_backslash_round_trips(self) -> None:
        """A key value with backslashes must round-trip exactly."""
        evil = "a\\b\\$\\\"c"
        self._vc.save_api_key_to_shell("ANTHROPIC_API_KEY", evil)
        proc = subprocess.run(
            ["bash", "-c",
             f"source '{self.home / '.bashrc'}' && "
             "printf '%s' \"$ANTHROPIC_API_KEY\""],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertEqual(proc.stdout, evil)



class TestH9AutocompleteNonBlocking(unittest.TestCase):
    """``_get_files`` must return promptly without running a synchronous scan."""

    def test_get_files_does_not_block_on_empty_cache(self) -> None:
        from kiss.server import autocomplete as ac

        broadcasts: list[dict] = []

        class StubPrinter:
            def broadcast(self, msg: dict) -> None:
                broadcasts.append(msg)

        class FakeServer(ac._AutocompleteMixin):
            def __init__(self) -> None:
                self.printer = StubPrinter()  # type: ignore[assignment]
                self.work_dir = "/"
                self._state_lock = threading.RLock()
                self._complete_queue = None
                self._complete_worker = None
                self._complete_seq_latest = {}
                self._file_cache = {}

        srv = FakeServer()
        from kiss.server import diff_merge as dm

        slow_scan_started = threading.Event()
        slow_scan_done = threading.Event()

        def slow_scan(work_dir: str) -> list[str]:
            slow_scan_started.set()
            time.sleep(2.0)
            slow_scan_done.set()
            return ["a.py", "b/c.py"]

        with mock.patch.object(dm, "_scan_files", slow_scan):
            t0 = time.time()
            srv._get_files("a")
            dt = time.time() - t0
        self.assertLess(dt, 0.5,
                        f"_get_files blocked for {dt:.2f}s — scan ran on caller thread")
        self.assertTrue(slow_scan_started.wait(2.0),
                        "Background scan was never started")



class TestM1GitHasTimeout(unittest.TestCase):
    """``_git`` must abort a hung git instead of blocking forever.

    Asserted through behaviour rather than through the shape of the
    subprocess call: the helper now delegates to the single hardened
    runner in ``git_worktree`` (which uses ``Popen`` + ``killpg``), so
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

    def test_hanging_git_is_abandoned_within_the_timeout(self) -> None:
        """A hung git yields returncode 124 well before it exits."""
        from kiss.agents.sorcar import git_worktree
        from kiss.server import diff_merge as dm

        tmpdir = tempfile.mkdtemp(prefix="kiss-m1-timeout-")
        try:
            bin_dir = self._install_hanging_git(Path(tmpdir))
            saved_path = os.environ["PATH"]
            saved_timeout = git_worktree._GIT_TIMEOUT_SECONDS
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{saved_path}"
            git_worktree._GIT_TIMEOUT_SECONDS = 1.0
            try:
                start = time.monotonic()
                result = dm._git(tmpdir, "status")
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
        from kiss.server import diff_merge as dm

        tmpdir = tempfile.mkdtemp(prefix="kiss-m1-ok-")
        try:
            self.assertEqual(dm._git(tmpdir, "init", "-q").returncode, 0)
            self.assertEqual(
                dm._git(tmpdir, "status", "--porcelain").returncode, 0,
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



class TestH3PropertyFuzz(unittest.TestCase):
    """Fuzz arbitrary key values through ``save_api_key_to_shell`` and
    require round-trip equality after sourcing the RC."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        from kiss.core import vscode_config as vc

        self._vc = vc
        self._orig = vc._shell_rc_path
        vc._shell_rc_path = lambda shell: self.home / ".bashrc"  # type: ignore[assignment]
        self._refresh_patch = mock.patch.object(vc, "_refresh_config", lambda: None)
        self._refresh_patch.start()
        self._home_patch = mock.patch.dict(
            os.environ, {"HOME": str(self.home), "SHELL": "/bin/bash"},
        )
        self._home_patch.start()

    def tearDown(self) -> None:
        self._vc._shell_rc_path = self._orig  # type: ignore[assignment]
        self._refresh_patch.stop()
        self._home_patch.stop()
        self._tmp.cleanup()

    def _round_trip(self, value: str) -> str:
        self._vc.save_api_key_to_shell("OPENAI_API_KEY", value)
        proc = subprocess.run(
            ["bash", "-c",
             f"source '{self.home / '.bashrc'}' && printf '%s' \"$OPENAI_API_KEY\""],
            capture_output=True, text=True, timeout=10,
        )
        return proc.stdout

    def test_fuzz_random_shell_metachars(self) -> None:
        """50 random values containing shell metachars must round-trip."""
        import random
        rng = random.Random(0xC0FFEE)
        meta = list("\"'`$\\;|&<>(){}*?[]!#%^~ \t")
        for _ in range(50):
            length = rng.randint(1, 40)
            value = "".join(rng.choice(meta + ["a", "b", "c", "1"])
                            for _ in range(length))
            if "\n" in value or "\r" in value or "\0" in value:
                continue
            got = self._round_trip(value)
            self.assertEqual(
                got, value,
                f"round-trip failed for {value!r} → {got!r}",
            )


if __name__ == "__main__":
    unittest.main()
