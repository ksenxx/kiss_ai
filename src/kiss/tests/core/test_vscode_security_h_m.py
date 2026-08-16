# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Security fixes: RC-file permissions and shell-quoting (H3).

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.agents.vscode.test_vscode_security_h_m``; the non-core tests remain there.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
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
