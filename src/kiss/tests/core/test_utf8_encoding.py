# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Core regression test for issue #43: explicit UTF-8 file I/O.

Core-only test (its child process imports nothing but
``kiss.core.base``) moved here from
``kiss.tests.agents.third_party_agents.test_utf8_encoding``, which keeps the
sorcar-side round-trip tests and imports :func:`_run_in_c_locale` back
from this module.

The test runs a child Python interpreter with a forced C locale
(``LC_ALL=C``, ``LANG=C``) and Python's UTF-8 mode disabled
(``PYTHONUTF8=0``) so the platform default text encoding is ASCII.
Before the fix, loading the UTF-8 system prompt in this environment
raised ``UnicodeDecodeError``; after the fix it must succeed.
"""

import subprocess
import sys
from pathlib import Path

import kiss


def _run_in_c_locale(script: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run ``script`` in a child interpreter pinned to the C locale.

    Args:
        script: Python source passed to ``python -c``.
        cwd: Working directory for the child process.

    Returns:
        The completed process with captured stdout/stderr.
    """
    # Located via the installed ``kiss`` package rather than ``__file__``
    # so the path stays correct no matter where this test module lives.
    src_dir = str(Path(kiss.__file__).resolve().parents[1])
    env = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONUTF8": "0",
        "PYTHONPATH": src_dir,
        "HOME": str(cwd),
    }
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


class TestUtf8Encoding:
    def test_system_prompt_loads_in_c_locale(self, tmp_path: Path) -> None:
        script = """
from kiss.core import base

assert isinstance(base.SYSTEM_PROMPT, str)
assert len(base.SYSTEM_PROMPT) > 0
print("OK")
"""
        proc = _run_in_c_locale(script, tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout
