# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for issue #43: explicit UTF-8 file I/O.

The core-only system-prompt check moved to
``kiss.tests.core.test_utf8_encoding``, which also owns the
:func:`_run_in_c_locale` helper imported below.

Every test runs a child Python interpreter with a forced C locale
(``LC_ALL=C``, ``LANG=C``) and Python's UTF-8 mode disabled
(``PYTHONUTF8=0``) so the platform default text encoding is ASCII.
Before the fix, text file I/O that omitted ``encoding="utf-8"``
mis-decoded UTF-8 content or raised ``UnicodeEncodeError`` /
``UnicodeDecodeError`` in this environment; after the fix the
round-trips must succeed byte-for-byte.
"""

import json
from pathlib import Path

from kiss.tests.core.test_utf8_encoding import _run_in_c_locale

NON_ASCII = "café ☕ — ünïcode"


NON_ASCII_JSON = json.dumps(NON_ASCII)


class TestUtf8Encoding:

    def test_write_read_edit_tools_round_trip_in_c_locale(self, tmp_path: Path) -> None:
        target = tmp_path / "unicode.txt"
        script = f"""
import json
from kiss.agents.sorcar.useful_tools import UsefulTools

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
