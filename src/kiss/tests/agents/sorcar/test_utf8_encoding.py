# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for issue #43: explicit UTF-8 file I/O.

Every test runs a child Python interpreter with a forced C locale
(``LC_ALL=C``, ``LANG=C``) and Python's UTF-8 mode disabled
(``PYTHONUTF8=0``) so the platform default text encoding is ASCII.
Before the fix, text file I/O that omitted ``encoding="utf-8"``
mis-decoded UTF-8 content or raised ``UnicodeEncodeError`` /
``UnicodeDecodeError`` in this environment; after the fix the
round-trips must succeed byte-for-byte.
"""

import json
import subprocess
import sys
from pathlib import Path

NON_ASCII = "café ☕ — ünïcode"
# Escape the non-ASCII text into pure-ASCII JSON so the script source
# itself never depends on the child interpreter's locale decoding.
NON_ASCII_JSON = json.dumps(NON_ASCII)


def _run_in_c_locale(script: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run ``script`` in a child interpreter pinned to the C locale.

    Args:
        script: Python source passed to ``python -c``.
        cwd: Working directory for the child process.

    Returns:
        The completed process with captured stdout/stderr.
    """
    src_dir = str(Path(__file__).resolve().parents[4])
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
    def test_write_read_edit_tools_round_trip_in_c_locale(self, tmp_path: Path) -> None:
        target = tmp_path / "unicode.txt"
        script = f"""
import json
from kiss.core.useful_tools import UsefulTools

text = json.loads({NON_ASCII_JSON!r})
tools = UsefulTools(work_dir={str(tmp_path)!r})
out = tools.Write({str(target)!r}, text)
assert out.startswith("Successfully"), out
read_back = tools.Read({str(target)!r})
assert text in read_back, repr(read_back)
old = json.loads('"\\u00fcn\\u00efcode"')
new = old + json.loads('"\\u2713"')
out = tools.Edit({str(target)!r}, old, new)
assert out.startswith("Successfully"), out
read_back = tools.Read({str(target)!r})
assert new in read_back, repr(read_back)
print("OK")
"""
        proc = _run_in_c_locale(script, tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout
        on_disk = target.read_text(encoding="utf-8")
        assert on_disk == NON_ASCII.replace("ünïcode", "ünïcode✓")

    def test_task_file_loads_in_c_locale(self, tmp_path: Path) -> None:
        task_file = tmp_path / "task.md"
        task_file.write_bytes(NON_ASCII.encode("utf-8"))
        script = f"""
import argparse, json
from kiss.agents.sorcar.cli_helpers import _resolve_task

args = argparse.Namespace(file={str(task_file)!r}, task=None)
loaded = _resolve_task(args)
expected = json.loads({NON_ASCII_JSON!r})
assert loaded == expected, repr(loaded)
print("OK")
"""
        proc = _run_in_c_locale(script, tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout

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
