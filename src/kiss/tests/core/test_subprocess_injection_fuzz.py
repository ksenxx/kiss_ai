# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Property-based / fuzzing tests for the shell-RC API-key code paths.

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.agents.vscode.test_subprocess_injection_fuzz``; the non-core tests remain there.
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
import unittest
from pathlib import Path
from unittest import mock

SHELL_METACHARS = list("\"'`$\\;|&<>(){}*?[]!#%^~ \t")


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


class TestFuzzSourceShellEnvPaths(unittest.TestCase):
    """``source_shell_env`` shell-quotes the RC path so a HOME containing
    metacharacters cannot inject commands into the sourced shell."""

    def setUp(self) -> None:
        if not shutil.which("bash"):
            self.skipTest("bash required")
        self._tmp = tempfile.TemporaryDirectory()
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
            payload = _rng_payload(rng, length_max=10,
                                   forbid="\n\r\0/")
            sub = Path(self._tmp.name) / f"d-{payload}"
            try:
                sub.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            rc = sub / ".bashrc"
            rc.write_text('export OPENAI_API_KEY=present\n')
            with mock.patch.object(vc, "_shell_rc_path", lambda s: rc), \
                    mock.patch.object(vc, "_get_user_shell", lambda: "bash"), \
                    mock.patch.object(vc, "_refresh_config", lambda: None):
                vc.source_shell_env()
            self.assertFalse(self._marker.exists(),
                             f"source_shell_env injected for path {sub}")


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
