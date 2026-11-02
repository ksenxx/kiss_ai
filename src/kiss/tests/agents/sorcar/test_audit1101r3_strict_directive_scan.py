# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 round 3: the extension's 'use strict' directive scan.

The CSP fallback of the extension's ``evaluate`` prepends a deadline
gate to the script, which ends the script's directive prologue, so a
script that opens with ``'use strict'`` gets the directive restated
ahead of the gate.  The round-2 fix found the directive with a regex
that read a newline alone as the end of the string statement (so
``'use strict'\\n.length`` was taken for a directive and a sloppy
script ran strict), lost a directive followed by a comment before its
semicolon, and backtracked exponentially on a prologue of blank-line
separated string statements (a 317-byte script stalled the service
worker past the bridge's timeout).  The scan is now linear and reads
the prologue as the parser does.  V8 is the oracle here: a script is
strict exactly when ``this`` of a plain call in it is ``undefined``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

BACKGROUND_JS = (
    Path(__file__).parents[3] / "agents" / "sorcar" / "chrome_extension" / "background.js"
)

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")

#: A script prefix and the names it refers to.  V8 decides whether the
#: script is strict; the scan must agree on every one.
CASES: tuple[tuple[str, dict[str, object]], ...] = (
    ("'use strict'; ", {}),
    ('"use strict"; ', {}),
    ("'use strict'\n", {}),
    ("'use strict'", {}),
    ("'use strict';;", {}),
    ("  \n  'use strict'  \n  ", {}),
    ("'use strict' /* comment */; ", {}),
    ("'other' /* comment */; 'use strict'; ", {}),
    ("// strict\n'use strict'\n", {}),
    ("/* c\n */ 'use strict' // c\n", {}),
    ("'use strict' /* multi\nline */ 0\n", {}),
    ("'a'\n'use strict'\n", {}),
    ('"use strict"\r\n', {}),
    ("'use strict'\u2028", {}),
    ("'foo\\\nbar'; 'use strict';", {}),
    ("'foo\\\r\nbar'; 'use strict';", {}),
    ("'it\\'s'; 'use strict';", {}),
    ("'use strict'\n++kissCounter\n", {"kissCounter": 0}),
    ("'use strict'\n--kissCounter\n", {"kissCounter": 0}),
    ("'use strict'\n!0\n", {}),
    ("'use strict'\n~0\n", {}),
    ("'use strict'\n{ }\n", {}),
    ("'use strict'\nkissName\n", {"kissName": 1}),
    ("'use strict'\n$kiss\n", {"$kiss": 1}),
    ("'use strict'\ninX\n", {"inX": 1}),
    ("'use strict'\ninstanceofX\n", {"instanceofX": 1}),
    ("\"use strict\"\n'use strict'.length\n", {}),
    # Round 4 review: a dot before a digit is a number, and in / instanceof
    # continued by a non-ASCII letter or an escape are identifiers.
    ("'use strict'\n.5\n", {}),
    ("'use strict'\ninπ\n", {"inπ": 1}),
    ("'use strict'\ninstanceofπ\n", {"instanceofπ": 1}),
    ("'use strict'\nin\\u0061\n", {"ina": 1}),
    ("'use strict'.length; ", {}),
    ("'use strict'\n.length; ", {}),
    ("'use strict'\n+ 'not a directive'; ", {}),
    ("'use strict'\n- 1\n", {}),
    ("'use strict'\n* 1\n", {}),
    ("'use strict'\n** 1\n", {}),
    ("'use strict'\n/ 2\n", {}),
    ("'use strict'\n% 2\n", {}),
    ("'use strict'\n, 1\n", {}),
    ("'use strict'\n? 1 : 2\n", {}),
    ("'use strict'\n== 'x'\n", {}),
    ("'use strict'\n!= 'x'\n", {}),
    ("'use strict'\n!== 'x'\n", {}),
    ("'use strict'\n< 'x'\n", {}),
    ("'use strict'\n<< 1\n", {}),
    ("'use strict'\n&& 1\n", {}),
    ("'use strict'\n|| 1\n", {}),
    ("'use strict'\n?? 1\n", {}),
    ("'use strict'\n& 1\n", {}),
    ("'use strict'\n| 1\n", {}),
    ("'use strict'\n^ 1\n", {}),
    ("'use strict'\nin kissObj\n", {"kissObj": {}}),
    ("'use strict'\nin\u00a0kissObj\n", {"kissObj": {}}),
    ("'use strict'\ninstanceof Object\n", {}),
    ("'use strict'\n[0]\n", {}),
    ("'use strict'\n(0)\n", {}),
    ("'use strict'\n`x`\n", {}),
    ("'use\\u0020strict'; ", {}),
    ("'use strict' + ''; ", {}),
    ("kissName = 1; 'use strict'; ", {"kissName": 0}),
    (";'use strict'; ", {}),
    ("('use strict'); ", {}),
    ("{ 'use strict'; } ", {}),
    ("", {}),
)

#: Scripts whose directive prologue is long or made of statements the
#: round-2 regex could partition in exponentially many ways.
LONG_SCRIPTS = (
    "'other'\n\n" * 3000 + "42",
    "'other';" * 3000 + "42",
    "/* c */ 'other' // c\n" * 3000 + "'use strict'",
    "'a'\n" * 3000 + ".length",
)

HARNESS = """
const vm = require('vm');
const { performance } = require('perf_hooks');
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const scan = vm.runInNewContext(input.source + '\\nhasUseStrictDirective', {});
const PROBE = '\\n;(function () { return this === undefined; })()';
function strictByV8(script, sandbox) {
  try {
    return vm.runInNewContext(script + PROBE, sandbox);
  } catch (err) {
    // A call expression ('use strict'(0)) is no directive; a syntax error is
    // no script. The error comes from the script's own realm: match its name.
    if (err.name === 'TypeError') return false;
    return null;
  }
}
const cases = input.cases.map(
  ([script, sandbox]) => [script, scan(script), strictByV8(script, sandbox)],
);
const timings = input.long.map((script) => {
  const start = performance.now();
  const found = scan(script);
  return [script.length, found, performance.now() - start];
});
process.stdout.write(JSON.stringify({ cases, timings }));
"""


def _scanner_source() -> str:
    """The directive scan as shipped in background.js, from its tokens to its function."""
    match = re.search(
        r"^const TRIVIA = .*?^function hasUseStrictDirective\(script\) \{\n.*?^\}\n",
        BACKGROUND_JS.read_text(encoding="utf-8"), re.S | re.M,
    )
    assert match, "the directive scan is not in background.js"
    return match.group(0)


def _run_harness() -> dict[str, list]:
    payload = {"source": _scanner_source(), "cases": CASES, "long": LONG_SCRIPTS}
    completed = subprocess.run(
        ["node", "-e", HARNESS], input=json.dumps(payload), capture_output=True, text=True,
        check=True, timeout=60,
    )
    results: dict[str, list] = json.loads(completed.stdout)
    return results


def test_the_scan_finds_a_directive_exactly_when_v8_runs_the_script_strict() -> None:
    results = _run_harness()["cases"]
    assert len(results) == len(CASES)
    assert all(verdict is not None for _script, _found, verdict in results), results
    assert [case for case in results if case[1] != case[2]] == []
    assert sum(1 for _script, found, _verdict in results if found) == 31


def test_the_scan_is_linear_in_the_prologue() -> None:
    """3,000 string statements (the regex stalled at 25) are scanned in milliseconds."""
    timings = _run_harness()["timings"]
    assert [found for _size, found, _ms in timings] == [False, False, True, False]
    assert all(ms < 200 for _size, _found, ms in timings), timings


if __name__ == "__main__":
    pytest.main([__file__])
