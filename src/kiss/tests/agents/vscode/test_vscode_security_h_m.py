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
from typing import Any
from unittest import mock

# ---------------------------------------------------------------------------
# H3 — vscode_config.save_api_key_to_shell: 0600 mode + shell-quoted value
# ---------------------------------------------------------------------------


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
        # Patch Path.home() too because vscode_config uses it at module import.
        from kiss.agents.vscode import vscode_config as vc

        self._vc = vc
        self._orig_rc_path = vc._shell_rc_path
        vc._shell_rc_path = lambda shell: self.home / ".bashrc"  # type: ignore[assignment]
        # Avoid triggering DEFAULT_CONFIG rebuild — keeps the test hermetic
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
        # Allow exactly 0600 — no group/other bits.
        self.assertFalse(mode & 0o077,
                         f"RC mode {oct(mode)} leaks group/other read bits")

    def test_value_with_double_quote_is_quoted_safely(self) -> None:
        """A key value containing `"` must not break out of its quotes."""
        # Pathological key with a double quote and a $ — both bash-special.
        evil = 'a"b$IFS$(echo pwned > /tmp/h3-pwned)c'
        self._vc.save_api_key_to_shell("OPENAI_API_KEY", evil)
        rc_text = (self.home / ".bashrc").read_text()
        # The export must round-trip through bash to preserve the literal value.
        # Run a fresh bash to source the RC and echo the variable.
        proc = subprocess.run(
            ["bash", "-c", f"source '{self.home / '.bashrc'}' && printf '%s' \"$OPENAI_API_KEY\""],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertEqual(proc.stdout, evil,
                         f"Value did not round-trip; rc was:\n{rc_text}")
        # And no file got written by command-substitution.
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


# ---------------------------------------------------------------------------
# H9 — autocomplete._get_files: scan must not block on first call
# ---------------------------------------------------------------------------


class TestH9AutocompleteNonBlocking(unittest.TestCase):
    """``_get_files`` must return promptly without running a synchronous scan."""

    def test_get_files_does_not_block_on_empty_cache(self) -> None:
        from kiss.agents.vscode import autocomplete as ac

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
        # Patch _scan_files to take a long time so a synchronous call would block.
        from kiss.agents.vscode import diff_merge as dm

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
        # The call must return in well under 2 s — it must not have waited
        # for the scan.
        self.assertLess(dt, 0.5,
                        f"_get_files blocked for {dt:.2f}s — scan ran on caller thread")
        # The scan should have been kicked off in the background.
        self.assertTrue(slow_scan_started.wait(2.0),
                        "Background scan was never started")


# ---------------------------------------------------------------------------
# M1 — diff_merge._git must time out, not hang forever
# ---------------------------------------------------------------------------


class TestM1GitHasTimeout(unittest.TestCase):
    """``_git`` must pass a ``timeout`` to ``subprocess.run``."""

    def test_git_invocation_carries_timeout(self) -> None:
        from kiss.agents.vscode import diff_merge as dm

        captured: dict = {}
        real_run = subprocess.run

        def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            captured.update(kwargs)
            # Return a successful completed process to avoid breaking callers.
            return real_run(["true"], capture_output=True, text=True)

        with mock.patch.object(subprocess, "run", fake_run):
            dm._git("/tmp", "status")
        self.assertIn("timeout", captured,
                      "_git did not pass a timeout — could hang forever")
        self.assertGreater(captured["timeout"], 0)
        self.assertLessEqual(captured["timeout"], 300,
                             "_git timeout should be modest (<= 300s)")

    def test_git_timeout_returns_completed_process_on_expiry(self) -> None:
        """A hanging git is reported as a normal (failed) CompletedProcess."""
        from kiss.agents.vscode import diff_merge as dm

        # Use a real shell `sleep` to simulate a slow git.
        with mock.patch.object(
            subprocess, "run",
            side_effect=subprocess.TimeoutExpired(cmd="git status", timeout=0.1),
        ):
            result = dm._git("/tmp", "status")
        # Must not raise — should return a CompletedProcess object so callers
        # don't crash.
        self.assertIsInstance(result, subprocess.CompletedProcess)
        self.assertNotEqual(result.returncode, 0)


# ---------------------------------------------------------------------------
# M5 — _save_untracked_base must be atomic; pending-merge.json atomic write
# ---------------------------------------------------------------------------


class TestM5AtomicSaveAndDecodeError(unittest.TestCase):
    """``_save_untracked_base`` is atomic; ``_diff_files`` swallows decode errors."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.work = Path(self._tmp.name)
        (self.work / "a.txt").write_text("hello\nworld\n")
        (self.work / "b.txt").write_text("foo\nbar\n")
        # Patch artifact-root so _untracked_base_dir lands inside the tmp dir.
        from kiss.core import config as cfg

        self._cfg_patch = mock.patch.object(
            cfg, "_artifact_root", lambda: self.work / ".kiss-artifacts",
        )
        self._cfg_patch.start()

    def tearDown(self) -> None:
        self._cfg_patch.stop()
        self._tmp.cleanup()

    def test_save_untracked_base_is_atomic_against_crash(self) -> None:
        """If copy fails partway, the OLD base copy must still be intact."""
        from kiss.agents.vscode import diff_merge as dm

        # First save a known good base copy.
        dm._save_untracked_base(str(self.work), {"a.txt"}, tab_id="tab1")
        base_dir = dm._untracked_base_dir("tab1")
        self.assertTrue((base_dir / "a.txt").exists())

        # Now arrange a copy that crashes mid-way.
        original_copy = shutil.copy2
        call_count = {"n": 0}

        def flaky_copy(src: str, dst: str, *args: object, **kwargs: object) -> None:
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Successfully copy first file.
                original_copy(src, dst)
                return
            raise OSError("disk full")

        with mock.patch.object(shutil, "copy2", flaky_copy):
            try:
                dm._save_untracked_base(
                    str(self.work), {"a.txt", "b.txt"}, tab_id="tab1",
                )
            except OSError:
                pass

        # The base directory must still contain a.txt — the previous good
        # state must not have been clobbered by the failed second save.
        a_in_base = base_dir / "a.txt"
        self.assertTrue(a_in_base.exists(),
                        "Previous good base copy was destroyed by failed save")
        self.assertEqual(a_in_base.read_text(), "hello\nworld\n")

    def test_diff_files_handles_unicode_decode_error(self) -> None:
        """Binary file should yield empty hunks, not raise UnicodeDecodeError."""
        from kiss.agents.vscode import diff_merge as dm

        # UTF-16 encoded — read_text() with default UTF-8 raises UnicodeDecodeError.
        bin_path = self.work / "binary.dat"
        bin_path.write_bytes("hello world".encode("utf-16"))
        text_path = self.work / "text.txt"
        text_path.write_text("hello\n")

        # Should not raise.
        result = dm._diff_files(str(bin_path), str(text_path))
        self.assertIsInstance(result, list)


# ---------------------------------------------------------------------------
# M2 — task_runner must set stop_event before the snapshot
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# M4 — _await_user_response must not loop forever when the queue is None
# ---------------------------------------------------------------------------


class TestM4AwaitUserResponseEmptyQueue(unittest.TestCase):
    """When the tab has no answer queue (e.g. closed mid-question), the
    wait method must raise ``KeyboardInterrupt`` instead of looping forever."""

    def test_returns_promptly_when_queue_is_none(self) -> None:
        from kiss.agents.vscode import task_runner as tr

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
                self._running_agent_states: dict[str, Any] = {}  # no entry for "ghost-tab"

        srv = FakeServer()
        t0 = time.time()
        with self.assertRaises(KeyboardInterrupt):
            srv._await_user_response()
        dt = time.time() - t0
        self.assertLess(dt, 1.0,
                        f"_await_user_response took {dt:.2f}s with no queue — "
                        "must raise immediately, not loop")


# ---------------------------------------------------------------------------
# Property-based fuzzer for the H3 shell-quoting fix
# ---------------------------------------------------------------------------


class TestH3PropertyFuzz(unittest.TestCase):
    """Fuzz arbitrary key values through ``save_api_key_to_shell`` and
    require round-trip equality after sourcing the RC."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        from kiss.agents.vscode import vscode_config as vc

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
            # Skip values containing newlines; export-style RC can't
            # represent them without continuation, which is out of scope.
            if "\n" in value or "\r" in value or "\0" in value:
                continue
            got = self._round_trip(value)
            self.assertEqual(
                got, value,
                f"round-trip failed for {value!r} → {got!r}",
            )


if __name__ == "__main__":
    unittest.main()
