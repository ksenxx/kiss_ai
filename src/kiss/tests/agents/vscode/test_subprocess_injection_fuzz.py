# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Property-based / fuzzing tests for every subprocess and shell
command path in ``src/kiss/agents/vscode/``.

These tests are the regression net for the H1, H3, H8 fixes
(DependencyInstaller shell-injection hardening, RC file shell-quoting,
exact ``pkill -x`` rather than the substring-match ``-f`` flag).

Strategy
--------
Every command path that takes user-controlled data — paths,
environment variables, RC values, file names, queries — is fuzzed
either:

  1. *Behaviourally*, by feeding many random shell-metacharacter
     payloads through the real code path and asserting no
     command-substitution fires (e.g. no marker file is created), and

  2. *Structurally*, by source-grepping the TypeScript files for any
     pattern that interpolates a non-constant variable into a shell
     string passed to ``execSync`` / ``execPromise``.  This catches new
     regressions introduced by future edits, even though we have no
     TypeScript runtime in the test harness.

Each test class corresponds to one subprocess/shell call site.  The
fuzzers use a fixed RNG seed for reproducibility but cover enough of
the metachar surface that an injection regression has near-100%
probability of being detected.
"""

from __future__ import annotations

import os
import random
import shutil
import stat
import string
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VSCODE_TS_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "vscode" / "src"
)
VSCODE_PY_DIR = Path(__file__).resolve().parents[3] / "agents" / "vscode"

SHELL_METACHARS = list("\"'`$\\;|&<>(){}*?[]!#%^~ \t")


def _ts(name: str) -> str:
    return (VSCODE_TS_DIR / name).read_text()


def _rng_payload(rng: random.Random, *, length_max: int = 40,
                 forbid: str = "\n\r\0") -> str:
    """Return a random string of shell metacharacters and ASCII fillers.

    Excludes characters in ``forbid`` because RC-file export lines
    can't represent newlines without continuation, and NUL is rejected
    by every UNIX exec().
    """
    pool = SHELL_METACHARS + list("abcXYZ012")
    pool = [c for c in pool if c not in forbid]
    return "".join(rng.choice(pool) for _ in range(rng.randint(1, length_max)))


# ---------------------------------------------------------------------------
# 1. Behavioral fuzz — vscode_config.save_api_key_to_shell across shells.
#    Every payload must round-trip through a real shell without any side
#    effect.  H3 fix property.
# ---------------------------------------------------------------------------


@unittest.skipIf(sys.platform == "win32",
                 "POSIX shells required for round-trip fuzzing")
class TestFuzzSaveApiKeyRoundTripBash(unittest.TestCase):
    """200 random metachar payloads must round-trip via ``bash -c source``."""

    SHELL = "bash"
    RC_NAME = ".bashrc"

    def setUp(self) -> None:
        if not shutil.which(self.SHELL):
            self.skipTest(f"{self.SHELL} not installed")
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        from kiss.core import vscode_config as vc
        self._vc = vc
        self._orig_rc = vc._shell_rc_path
        vc._shell_rc_path = lambda shell: self.home / self.RC_NAME  # type: ignore[assignment]
        self._orig_get_shell = vc._get_user_shell
        vc._get_user_shell = lambda: self.SHELL  # type: ignore[assignment]
        self._refresh_patch = mock.patch.object(vc, "_refresh_config",
                                                lambda: None)
        self._refresh_patch.start()
        self._env_patch = mock.patch.dict(os.environ,
                                          {"HOME": str(self.home),
                                           "SHELL": f"/bin/{self.SHELL}"})
        self._env_patch.start()
        self._marker = Path(tempfile.gettempdir()) / f"fuzz-pwned-{os.getpid()}"
        if self._marker.exists():
            self._marker.unlink()

    def tearDown(self) -> None:
        self._vc._shell_rc_path = self._orig_rc  # type: ignore[assignment]
        self._vc._get_user_shell = self._orig_get_shell  # type: ignore[assignment]
        self._refresh_patch.stop()
        self._env_patch.stop()
        if self._marker.exists():
            self._marker.unlink()
        self._tmp.cleanup()

    def _round_trip(self, value: str) -> str:
        self._vc.save_api_key_to_shell("OPENAI_API_KEY", value)
        rc = self.home / self.RC_NAME
        proc = subprocess.run(
            [self.SHELL, "-c",
             f"source '{rc}' && printf '%s' \"$OPENAI_API_KEY\""],
            capture_output=True, text=True, timeout=10,
        )
        return proc.stdout

    def test_fuzz_200_payloads_round_trip(self) -> None:
        rng = random.Random(0xFAFA)
        for _ in range(200):
            value = _rng_payload(rng)
            with self.subTest(value=value):
                got = self._round_trip(value)
                self.assertEqual(got, value,
                                 f"payload {value!r} → {got!r}")
                self.assertFalse(self._marker.exists(),
                                 f"command substitution fired for {value!r}")

    def test_specific_dangerous_payloads(self) -> None:
        m = self._marker
        # Each payload tries a different injection technique.
        for payload in [
            f'$(touch {m})',
            f'`touch {m}`',
            f'"; touch {m}; #',
            f'"$(touch {m})"',
            f"'\";touch {m};echo '",
            f'\\";touch {m};\\"',
            "$IFS$9touch$IFS" + str(m),
            f'${{IFS}}touch${{IFS}}{m}',
        ]:
            with self.subTest(payload=payload):
                got = self._round_trip(payload)
                self.assertEqual(got, payload)
                self.assertFalse(m.exists(),
                                 f"injection fired: {payload}")


@unittest.skipIf(sys.platform == "win32",
                 "POSIX shells required for round-trip fuzzing")
class TestFuzzSaveApiKeyRoundTripZsh(TestFuzzSaveApiKeyRoundTripBash):
    """Same payload fuzz under zsh."""

    SHELL = "zsh"
    RC_NAME = ".zshrc"


@unittest.skipIf(sys.platform == "win32",
                 "POSIX shells required for round-trip fuzzing")


# ---------------------------------------------------------------------------
# 2. Behavioral fuzz — diff_merge._git
# ---------------------------------------------------------------------------


class TestFuzzGitCwdNoInjection(unittest.TestCase):
    """``_git`` must run via argv (no shell), so fuzzed cwd values that
    contain shell metacharacters are passed verbatim and cannot inject
    shell commands."""

    def test_fuzz_cwd_paths_with_metacharacters(self) -> None:
        from kiss.server import diff_merge as dm

        rng = random.Random(0x617)
        marker = Path(tempfile.gettempdir()) / f"git-pwned-{os.getpid()}"
        if marker.exists():
            marker.unlink()
        try:
            for _ in range(30):
                tmpdir = Path(tempfile.mkdtemp(
                    prefix="kiss-git-fuzz-", suffix=_rng_payload(
                        rng, length_max=8, forbid="\n\r\0/")))
                # Initialise an empty repo so ``git status`` has work to do.
                subprocess.run(["git", "init", "-q", str(tmpdir)],
                               capture_output=True, timeout=20)
                # Pre-existing payloads in the working dir tree must not
                # be evaluated as shell.
                bad_name = f"$(touch '{marker}')"
                # Don't actually create that file — we only need
                # ``_git`` to receive the (possibly weird) cwd as data.
                cp = dm._git(str(tmpdir), "status", "--porcelain")
                self.assertEqual(cp.returncode, 0,
                                 msg=cp.stderr)
                self.assertFalse(marker.exists(),
                                 f"_git executed shell for cwd {tmpdir}; "
                                 f"bad_name {bad_name!r}")
                shutil.rmtree(tmpdir, ignore_errors=True)
        finally:
            if marker.exists():
                marker.unlink()

    def test_fuzz_args_are_passed_verbatim(self) -> None:
        """A fuzzed ``*args`` value must arrive at git unmangled (no
        shell expansion)."""
        from kiss.server import diff_merge as dm

        captured: list[list[str]] = []
        real_run = subprocess.run

        def spy_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
            captured.append(list(cmd))
            return real_run(["true"], capture_output=True, text=True)

        rng = random.Random(0x914)
        with mock.patch.object(subprocess, "run", spy_run):
            for _ in range(20):
                arg = _rng_payload(rng, forbid="\0")
                dm._git("/tmp", "log", arg, "--oneline")
                self.assertEqual(captured[-1][0], "git")
                self.assertIn(arg, captured[-1],
                              f"arg {arg!r} not passed verbatim "
                              f"to git: {captured[-1]}")


# ---------------------------------------------------------------------------
# 3. Behavioral fuzz — _save_untracked_base file names
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# 4. Behavioral fuzz — vscode_config.source_shell_env paths
# ---------------------------------------------------------------------------


@unittest.skipIf(sys.platform == "win32",
                 "POSIX shells required for source-RC fuzzing")
class TestFuzzSourceShellEnvPaths(unittest.TestCase):
    """``source_shell_env`` shell-quotes the RC path so a HOME containing
    metacharacters cannot inject commands into the sourced shell."""

    def setUp(self) -> None:
        if not shutil.which("bash"):
            self.skipTest("bash required")
        self._tmp = tempfile.TemporaryDirectory()
        # ``source_shell_env`` imports every ``export`` in the sourced
        # RC into ``os.environ`` — including the sentinel
        # ``OPENAI_API_KEY=present`` planted below.  Snapshot/restore
        # the environment so the sentinel does not leak into later
        # tests (it made live OpenAI calls fail 401 with "Incorrect
        # API key provided: present").
        self._env_patch = mock.patch.dict(os.environ)
        self._env_patch.start()
        self._marker = (Path(tempfile.gettempdir())
                        / f"source-pwned-{os.getpid()}")
        if self._marker.exists():
            self._marker.unlink()

    def tearDown(self) -> None:
        if self._marker.exists():
            self._marker.unlink()
        self._env_patch.stop()
        self._tmp.cleanup()

    def test_fuzz_rc_paths_with_metacharacters(self) -> None:
        from kiss.core import vscode_config as vc

        rng = random.Random(0xCAFE)
        for _ in range(20):
            # Build a directory name with shell metacharacters under tmp
            # so we can plant the RC file at the resulting weird path
            # and call ``source_shell_env``.
            payload = _rng_payload(rng, length_max=10,
                                   forbid="\n\r\0/")
            sub = Path(self._tmp.name) / f"d-{payload}"
            try:
                sub.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            rc = sub / ".bashrc"
            # Write a minimal RC that exports a known key — we want to
            # detect whether sourcing this file would also execute the
            # injected payload that lives in the file *path*.
            rc.write_text('export OPENAI_API_KEY=present\n')
            with mock.patch.object(vc, "_shell_rc_path", lambda s: rc), \
                    mock.patch.object(vc, "_get_user_shell", lambda: "bash"), \
                    mock.patch.object(vc, "_refresh_config", lambda: None):
                vc.source_shell_env()
            self.assertFalse(self._marker.exists(),
                             f"source_shell_env injected for path {sub}")


# ---------------------------------------------------------------------------
# 5. Source-grep fuzz — DependencyInstaller.ts must not regress
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# 6. Source-grep fuzz — kissPaths.ts must consult workspace trust
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# 7. Source-grep fuzz — webview path-traversal / CSP / nonce
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# 8. Behavioral fuzz — markdown sanitizer must strip dangerous content
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# 9. Behavioral fuzz — autocomplete prefix injection
# ---------------------------------------------------------------------------


class TestFuzzAutocompletePrefix(unittest.TestCase):
    """Autocomplete must never invoke a shell.  Fuzz the prefix with
    every shell metachar — must complete without side effects."""

    def test_fuzz_prefix_metachars(self) -> None:
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
                self._file_cache = {"/": ["a.py", "b.py", "x/y.txt"]}

        srv = FakeServer()
        marker = Path(tempfile.gettempdir()) / f"ac-pwned-{os.getpid()}"
        if marker.exists():
            marker.unlink()
        rng = random.Random(0xACAC)
        try:
            for _ in range(50):
                prefix = _rng_payload(rng, length_max=20)
                broadcasts.clear()
                srv._get_files(prefix)
                self.assertFalse(marker.exists(),
                                 f"autocomplete fired shell for {prefix!r}")
                # Every call yields exactly one broadcast.
                self.assertEqual(len(broadcasts), 1)
                self.assertEqual(broadcasts[0]["type"], "files")
        finally:
            if marker.exists():
                marker.unlink()


# ---------------------------------------------------------------------------
# 10. Source-grep fuzz — every Python subprocess.run uses an argv list
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# 11. Behavioral fuzz — chmod 0600 round-trip on RC under random umasks
# ---------------------------------------------------------------------------


@unittest.skipIf(sys.platform == "win32", "POSIX chmod test")
class TestFuzzRcModeUnderRandomUmasks(unittest.TestCase):
    """Under any umask, the resulting RC file must be 0600."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        from kiss.core import vscode_config as vc
        self._vc = vc
        self._orig_rc = vc._shell_rc_path
        vc._shell_rc_path = lambda shell: self.home / ".bashrc"  # type: ignore[assignment]
        self._refresh_patch = mock.patch.object(vc, "_refresh_config",
                                                lambda: None)
        self._refresh_patch.start()
        self._orig_umask = os.umask(0o000)
        self._env_patch = mock.patch.dict(
            os.environ,
            {"HOME": str(self.home), "SHELL": "/bin/bash"})
        self._env_patch.start()

    def tearDown(self) -> None:
        os.umask(self._orig_umask)
        self._vc._shell_rc_path = self._orig_rc  # type: ignore[assignment]
        self._refresh_patch.stop()
        self._env_patch.stop()
        self._tmp.cleanup()

    def test_rc_mode_0600_under_each_umask(self) -> None:
        rng = random.Random(0xC0DE)
        rc = self.home / ".bashrc"
        for umask in [0o000, 0o022, 0o027, 0o077, 0o002, 0o007]:
            os.umask(umask)
            value = "secret-" + "".join(
                rng.choice(string.ascii_letters) for _ in range(20))
            self._vc.save_api_key_to_shell("OPENAI_API_KEY", value)
            mode = stat.S_IMODE(rc.stat().st_mode)
            self.assertEqual(
                mode, 0o600,
                f"umask={oct(umask)} → RC mode {oct(mode)} "
                "(expected 0o600)")


# ---------------------------------------------------------------------------
# 12. Behavioral fuzz — DependencyInstaller xmlEscape / unitEscape
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# 13. Behavioral — exhaustive injection-payload corpus must round-trip
# ---------------------------------------------------------------------------


@unittest.skipIf(sys.platform == "win32",
                 "POSIX shells required for round-trip fuzzing")
class TestKnownInjectionCorpus(unittest.TestCase):
    """A curated corpus of injection payloads against
    ``save_api_key_to_shell``.  Each must round-trip and never fire."""

    def setUp(self) -> None:
        if not shutil.which("bash"):
            self.skipTest("bash required")
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        from kiss.core import vscode_config as vc
        self._vc = vc
        self._orig_rc = vc._shell_rc_path
        vc._shell_rc_path = lambda shell: self.home / ".bashrc"  # type: ignore[assignment]
        self._refresh_patch = mock.patch.object(vc, "_refresh_config",
                                                lambda: None)
        self._refresh_patch.start()
        # ``save_api_key_to_shell`` exports the saved value into
        # ``os.environ[key_name]``.  Snapshot/restore the environment
        # (as the sibling fuzz classes do) so the last injection
        # payload does not leak into ``OPENAI_API_KEY`` for the rest of
        # the pytest process — that leak made every later live OpenAI
        # call (e.g. audio transcription in test_multimodal) fail 401.
        self._env_patch = mock.patch.dict(
            os.environ,
            {"HOME": str(self.home), "SHELL": "/bin/bash"})
        self._env_patch.start()
        self._marker = (Path(tempfile.gettempdir())
                        / f"corpus-pwned-{os.getpid()}")
        if self._marker.exists():
            self._marker.unlink()

    def tearDown(self) -> None:
        if self._marker.exists():
            self._marker.unlink()
        self._vc._shell_rc_path = self._orig_rc  # type: ignore[assignment]
        self._refresh_patch.stop()
        self._env_patch.stop()
        self._tmp.cleanup()

    def test_known_payloads(self) -> None:
        m = str(self._marker)
        payloads = [
            f'$(touch {m})',
            f'`touch {m}`',
            f'"; touch {m}; echo "',
            f"'; touch {m}; echo '",
            f"\"; touch \"{m}\"; echo \"",
            f"\\\";touch {m};\\\"",
            "$IFS",
            "${IFS}",
            "&& touch " + m,
            "; touch " + m,
            "| touch " + m,
            ">/dev/null && touch " + m,
            "<(touch " + m + ")",
            ">(touch " + m + ")",
            "$((touch " + m + "))",
            f"$'\\x60touch\\x20{m}\\x60'",
        ]
        for p in payloads:
            with self.subTest(payload=p):
                self._vc.save_api_key_to_shell("OPENAI_API_KEY", p)
                rc = self.home / ".bashrc"
                proc = subprocess.run(
                    ["bash", "-c",
                     f"source '{rc}' && printf '%s' \"$OPENAI_API_KEY\""],
                    capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(proc.returncode, 0, msg=proc.stderr)
                self.assertEqual(proc.stdout, p,
                                 f"payload {p!r} round-tripped to {proc.stdout!r}")
                self.assertFalse(self._marker.exists(),
                                 f"INJECTION FIRED: {p!r}")


if __name__ == "__main__":
    unittest.main()
