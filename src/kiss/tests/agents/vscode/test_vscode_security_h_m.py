"""Integration tests for HIGH (H1-H10) and MEDIUM (M1-M5) severity fixes
in src/kiss/agents/vscode/.  Each Python-side fix has a behavioural test
that fails when the fix is reverted.

TS-side fixes (DependencyInstaller, SorcarSidebarView, AgentProcess,
SorcarTab) are spot-checked via source-grep tests because the test
harness has no TypeScript runtime.
"""

from __future__ import annotations

import inspect
import os
import re
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
                self.printer = StubPrinter()
                self.work_dir = "/"
                self._state_lock = threading.Lock()
                self._complete_queue = None
                self._complete_worker = None
                self._complete_seq_latest = 0
                self._file_cache = None

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


class TestM2StopEventOrder(unittest.TestCase):
    """The stop event must be plumbed through ``_thread_local`` before
    the long-running ``_capture_pre_snapshot`` runs, so a Stop click can
    abort a hanging ``repo_lock``."""

    def test_thread_local_stop_event_set_early(self) -> None:
        """Source-level: stop_event must be installed before snapshot."""
        from kiss.agents.vscode import task_runner as tr

        src = inspect.getsource(tr._TaskRunnerMixin._run_task_inner)
        # Find the lines for stop_event assignment and capture_pre_snapshot.
        lines = src.splitlines()
        stop_idx = next(
            (i for i, ln in enumerate(lines)
             if "_thread_local.stop_event" in ln and "=" in ln),
            None,
        )
        snap_idx = next(
            (i for i, ln in enumerate(lines)
             if "_capture_pre_snapshot" in ln),
            None,
        )
        self.assertIsNotNone(stop_idx, "stop_event assignment not found")
        self.assertIsNotNone(snap_idx, "_capture_pre_snapshot not found")
        self.assertLess(stop_idx, snap_idx,
                        f"stop_event installed at line {stop_idx} but "
                        f"_capture_pre_snapshot runs at line {snap_idx}; "
                        "swap them so Stop works during snapshot")


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

        class FakeServer(tr._TaskRunnerMixin):
            def __init__(self) -> None:
                self.printer = FakePrinter()
                self.printer._thread_local.stop_event = threading.Event()
                self.printer._thread_local.tab_id = "ghost-tab"
                self._state_lock = threading.Lock()
                self._tab_states = {}  # no entry for "ghost-tab"

        srv = FakeServer()
        t0 = time.time()
        with self.assertRaises(KeyboardInterrupt):
            srv._await_user_response()
        dt = time.time() - t0
        self.assertLess(dt, 1.0,
                        f"_await_user_response took {dt:.2f}s with no queue — "
                        "must raise immediately, not loop")


# ---------------------------------------------------------------------------
# H4 — webview path-traversal guard (TS source check)
# ---------------------------------------------------------------------------


def _ts_path(name: str) -> Path:
    # tests/agents/vscode/X.py → kiss/agents/vscode/src/<name>
    return Path(__file__).resolve().parents[3] / "agents" / "vscode" / "src" / name


def _media_path(name: str) -> Path:
    return Path(__file__).resolve().parents[3] / "agents" / "vscode" / "media" / name


class TestH4OpenFileGuard(unittest.TestCase):
    """The ``openFile`` (and ``submit`` file-shortcut) handlers must
    refuse paths that escape workDir."""

    def test_open_file_handler_validates_path_under_workdir(self) -> None:
        src = _ts_path("SorcarSidebarView.ts").read_text()
        # Find the case 'openFile' block — must contain a guard rejecting
        # paths outside _getWorkDir().  We accept any of: relative()/
        # startsWith()/normalize-and-compare patterns provided they
        # mention the work dir and reject the operation.
        idx = src.find("case 'openFile':")
        self.assertGreater(idx, 0, "openFile case not found")
        # Look at the next ~600 chars after the case marker.
        block = src[idx: idx + 800]
        markers = ("isPathInside", "startsWith", "relative(", "isWithinWorkDir")
        self.assertTrue(any(m in block for m in markers),
                        f"openFile block has no traversal guard:\n{block}")

    def test_submit_file_shortcut_validates_path_under_workdir(self) -> None:
        src = _ts_path("SorcarSidebarView.ts").read_text()
        idx = src.find("case 'submit':")
        self.assertGreater(idx, 0)
        # The file-shortcut detection block lives within the first ~1000
        # chars after the case marker.
        block = src[idx: idx + 1200]
        markers = ("isPathInside", "startsWith", "relative(", "isWithinWorkDir")
        self.assertTrue(any(m in block for m in markers),
                        "submit file-shortcut has no traversal guard")


# ---------------------------------------------------------------------------
# H5 — workspace-trust gate for env / setting in AgentProcess.findKissProject
# ---------------------------------------------------------------------------


class TestH5WorkspaceTrustGate(unittest.TestCase):
    """``findKissProject`` must refuse to honour env var / setting in
    untrusted workspaces."""

    def test_findkissproject_consults_workspace_trust(self) -> None:
        src = _ts_path("AgentProcess.ts").read_text()
        # The function must reference vscode.workspace.isTrusted (or use
        # the trust API equivalent) before returning the env / setting path.
        self.assertRegex(
            src, r"isTrusted",
            "findKissProject does not check vscode.workspace.isTrusted",
        )
        # Find findKissProject body.
        idx = src.find("export function findKissProject")
        self.assertGreater(idx, 0)
        body = src[idx: idx + 1500]
        self.assertIn("isTrusted", body,
                      "findKissProject body does not gate on isTrusted")


# ---------------------------------------------------------------------------
# H7 — CSP nonce must come from a CSPRNG, not Math.random()
# ---------------------------------------------------------------------------


class TestH7CspNonce(unittest.TestCase):
    """Nonce must be generated via Node's crypto module."""

    def test_nonce_uses_crypto_random_bytes(self) -> None:
        src = _ts_path("SorcarTab.ts").read_text()
        # Math.random must NOT appear inside the getNonce function.
        idx = src.find("export function getNonce")
        self.assertGreater(idx, 0)
        body = src[idx: idx + 500]
        self.assertNotIn("Math.random", body,
                         "getNonce still uses Math.random — not CSPRNG-safe")
        # Must use crypto.randomBytes (or webcrypto getRandomValues).
        self.assertTrue(
            "randomBytes" in body or "getRandomValues" in body,
            "getNonce does not use a CSPRNG (crypto.randomBytes / "
            "getRandomValues)",
        )


# ---------------------------------------------------------------------------
# H6 — CSP must include form-action / frame-src / object-src / base-uri
# ---------------------------------------------------------------------------


class TestH6CspTightening(unittest.TestCase):
    """The CSP meta tag must restrict form/frame/object/base-uri."""

    def test_csp_locks_down_dangerous_sinks(self) -> None:
        src = _ts_path("SorcarTab.ts").read_text()
        idx = src.find("Content-Security-Policy")
        self.assertGreater(idx, 0)
        # The CSP string ends at the closing quote of the content attribute.
        # Look at the next ~600 chars.
        csp_block = src[idx: idx + 600]
        for directive in ("form-action", "frame-src", "object-src", "base-uri"):
            self.assertIn(directive, csp_block,
                          f"CSP missing {directive} directive: {csp_block}")
            # Ensure each is locked to 'none' (only safe value here).
            self.assertRegex(
                csp_block,
                rf"{directive}\s+'none'",
                f"CSP {directive} is not 'none'",
            )


# ---------------------------------------------------------------------------
# H6 (frontend) — marked.parse output must be sanitized before innerHTML
# ---------------------------------------------------------------------------


class TestH6MarkdownSanitized(unittest.TestCase):
    """Every ``marked.parse(...)`` call site that flows into ``innerHTML``
    must wrap the result in a sanitizer call."""

    def test_main_js_does_not_assign_marked_parse_directly(self) -> None:
        src = _media_path("main.js").read_text()
        # Direct innerHTML = marked.parse(...) is forbidden — must go
        # through sanitizeHtml() / kissSanitize() / similar wrapper.
        bad = re.findall(r"innerHTML\s*=\s*marked\.parse\(", src)
        self.assertFalse(
            bad,
            f"main.js has {len(bad)} direct innerHTML = marked.parse(...) "
            "assignments without a sanitize wrapper",
        )

    def test_main_js_has_sanitizer_function(self) -> None:
        src = _media_path("main.js").read_text()
        # The sanitizer must exist and strip <script>, <iframe>, javascript:,
        # and on*= attributes.
        markers = ("function kissSanitize", "function sanitizeHtml")
        self.assertTrue(any(m in src for m in markers),
                        "main.js has no sanitizer function for marked output")
        # The sanitizer must remove on*= attributes.
        sanitizer_idx = max(src.find(m) for m in markers if m in src)
        sanitizer_body = src[sanitizer_idx: sanitizer_idx + 2000]
        self.assertRegex(
            sanitizer_body, r"on\\w\\+\\s\*=|on\[a-z\]|removeAttribute",
            f"sanitizer doesn't strip event-handler attributes:\n{sanitizer_body}",
        )


# ---------------------------------------------------------------------------
# H1 — DependencyInstaller must spawn external commands without a shell
# ---------------------------------------------------------------------------


class TestH1DependencyInstallerNoShellInjection(unittest.TestCase):
    """No path-interpolated string is fed into ``exec`` / ``execSync``
    / ``execPromise`` in DependencyInstaller for installation flows."""

    def test_installer_does_not_use_pkill_dash_f(self) -> None:
        src = _ts_path("DependencyInstaller.ts").read_text()
        # H8 — pkill -f matches argv substrings; we must use exact match (-x)
        # or PID-based shutdown.
        self.assertNotRegex(
            src, r"pkill\s+-f",
            "DependencyInstaller still uses pkill -f (matches unrelated processes)",
        )

    def test_installer_does_not_pipe_curl_into_tar_via_shell(self) -> None:
        """The `curl ... | tar xz` chain must be replaced by a no-shell flow."""
        src = _ts_path("DependencyInstaller.ts").read_text()
        # The old vulnerable pattern: a single execPromise carrying
        # 'curl -fsSL ... | tar xz -C ...' — the pipe forces shell execution.
        bad = re.findall(r"execPromise\([^)]*\|\s*tar", src)
        self.assertFalse(
            bad,
            "DependencyInstaller still pipes curl into tar through a shell — "
            "this enables shell-injection from $HOME and other env vars",
        )

    def test_installer_verifies_downloaded_binary_hashes(self) -> None:
        """H2 — downloads must be verified against an expected SHA256."""
        src = _ts_path("DependencyInstaller.ts").read_text()
        # Some integrity-check call must exist alongside the download flows.
        self.assertRegex(
            src, r"createHash\('sha256'\)|sha256",
            "DependencyInstaller has no SHA256 verification of downloads",
        )


# ---------------------------------------------------------------------------
# H10 — plist / systemd unit content must escape XML / unit special chars
# ---------------------------------------------------------------------------


class TestH10PlistSystemdEscape(unittest.TestCase):
    """The macOS plist and Linux systemd unit must escape user-controlled paths."""

    def test_restartkisswebdaemon_escapes_xml(self) -> None:
        src = _ts_path("DependencyInstaller.ts").read_text()
        idx = src.find("function restartKissWebDaemon")
        self.assertGreater(idx, 0)
        body = src[idx: idx + 5000]
        # Must call an XML-escape helper before interpolating into `<string>`.
        self.assertRegex(
            body, r"xmlEscape|escapeXml|replace\(/&/",
            "restartKissWebDaemon does not XML-escape paths in the plist",
        )


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
