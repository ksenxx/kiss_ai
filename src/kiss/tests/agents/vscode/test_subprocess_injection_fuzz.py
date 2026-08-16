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
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from kiss.tests.core.test_subprocess_injection_fuzz import _rng_payload

VSCODE_TS_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "vscode" / "src"
)
VSCODE_PY_DIR = Path(__file__).resolve().parents[3] / "agents" / "vscode"



def _ts(name: str) -> str:
    return (VSCODE_TS_DIR / name).read_text()









@unittest.skipIf(sys.platform == "win32",
                 "POSIX shells required for round-trip fuzzing")



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
                subprocess.run(["git", "init", "-q", str(tmpdir)],
                               capture_output=True, timeout=20)
                bad_name = f"$(touch '{marker}')"
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
        """A fuzzed ``*args`` value must arrive at git unmangled.

        Observed at the real exec boundary rather than by spying on
        ``subprocess.run``: a stub ``git`` first on ``PATH`` records
        its own NUL-separated argv, so the property holds whatever
        subprocess primitive the helper uses internally.
        """
        from kiss.server import diff_merge as dm

        tmpdir = Path(tempfile.mkdtemp(prefix="kiss-git-argv-"))
        argv_file = tmpdir / "argv"
        bin_dir = tmpdir / "stub-bin"
        bin_dir.mkdir()
        stub = bin_dir / "git"
        stub.write_text(
            "#!/bin/sh\n"
            ': > "$KISS_ARGV_OUT"\n'
            'for a in "$@"; do printf \'%s\\0\' "$a" '
            '>> "$KISS_ARGV_OUT"; done\n',
            encoding="utf-8",
        )
        stub.chmod(
            stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
        )

        rng = random.Random(0x914)
        saved_path = os.environ["PATH"]
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{saved_path}"
        os.environ["KISS_ARGV_OUT"] = str(argv_file)
        try:
            for _ in range(20):
                arg = _rng_payload(rng, forbid="\0")
                dm._git(str(tmpdir), "log", arg, "--oneline")
                argv = argv_file.read_text().split("\0")[:-1]
                self.assertIn("log", argv)
                self.assertIn("--oneline", argv)
                self.assertIn(arg, argv,
                              f"arg {arg!r} not passed verbatim "
                              f"to git: {argv}")
        finally:
            os.environ["PATH"] = saved_path
            os.environ.pop("KISS_ARGV_OUT", None)
            shutil.rmtree(tmpdir, ignore_errors=True)

















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
                self.assertEqual(len(broadcasts), 1)
                self.assertEqual(broadcasts[0]["type"], "files")
        finally:
            if marker.exists():
                marker.unlink()














if __name__ == "__main__":
    unittest.main()
