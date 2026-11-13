# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``findings-report``: the final report's table is built from one JSON line per finding.

The tests write the files fixers would write and run the command on them.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from kiss.scripts import findings_report

FIXED = {
    "file": "src/kiss/core/a.py", "line": 12, "kind": "bug", "status": "fixed",
    "summary": "A clear <never> fired & the context grew", "test": "tests/test_a.py::test_clear",
}
OPEN = {
    "file": "src/kiss/core/b.py", "line": 0, "kind": "open", "status": "open",
    "summary": "The Codex path has no test", "test": "",
}
REJECTED = {
    "file": "src/kiss/core/a.py", "line": 3, "kind": "doc", "status": "rejected",
    "summary": "The docstring is right as it is", "test": "",
}


def _write(path: Path, *entries: Any) -> Path:
    """Write *entries* to *path*, one JSON value per line with a blank line between files' ends."""
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries) + "\n", encoding="utf-8")
    return path


def test_the_report_counts_and_lists_every_finding_open_ones_first(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = _write(tmp_path / "fixer-clearing.jsonl", FIXED, REJECTED)
    second = _write(tmp_path / "fixer-redaction.jsonl", OPEN)
    empty = _write(tmp_path / "reviewer.jsonl")
    out = tmp_path / "report.html"
    # The installed command is this module's ``main``; run it as a process.
    run = subprocess.run(
        [sys.executable, "-m", "kiss.scripts.findings_report", str(first), str(second),
         str(empty), "-o", str(out)],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert (run.returncode, run.stdout, run.stderr) == (0, "", "")
    report = out.read_text(encoding="utf-8")
    counts = "<p>3 findings: 1 open, 1 fixed, 1 rejected. By kind: 1 bug, 1 doc, 1 open.</p>"
    assert counts in report
    rows = [line for line in report.splitlines() if line.startswith("<tr><td>")]
    assert rows == [
        "<tr><td>open</td><td>open</td><td>src/kiss/core/b.py</td>"
        "<td>The Codex path has no test</td><td></td><td>fixer-redaction</td></tr>",
        "<tr><td>fixed</td><td>bug</td><td>src/kiss/core/a.py:12</td>"
        "<td>A clear &lt;never&gt; fired &amp; the context grew</td>"
        "<td>tests/test_a.py::test_clear</td><td>fixer-clearing</td></tr>",
        "<tr><td>rejected</td><td>doc</td><td>src/kiss/core/a.py:3</td>"
        "<td>The docstring is right as it is</td><td></td><td>fixer-clearing</td></tr>",
    ]
    # Without ``-o`` the same fragment goes to standard output; no finding is a report too.
    assert findings_report.main([str(first), str(second), str(empty)]) == 0
    assert capsys.readouterr().out == report
    assert findings_report.main([str(empty)]) == 0
    assert "<p>0 findings: 0 open, 0 fixed, 0 rejected.</p>" in capsys.readouterr().out
    # A report that cannot be written is one line of error, like a file that cannot be read.
    unwritable = tmp_path / "absent" / "report.html"
    assert findings_report.main([str(first), "-o", str(unwritable)]) == 1
    error = capsys.readouterr().err
    assert error.startswith(f"findings-report: {unwritable}: ") and "No such file" in error
    assert len(error.splitlines()) == 1


def test_a_malformed_line_is_named_and_no_report_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "fixer.jsonl"
    lines = [
        json.dumps(FIXED),
        "{not json",
        json.dumps(["a", "list"]),
        json.dumps({key: value for key, value in FIXED.items() if key != "test"} | {"tset": ""}),
        json.dumps(FIXED | {"line": "12"}),
        json.dumps(FIXED | {"line": True}),
        json.dumps(FIXED | {"line": -1}),
        json.dumps(FIXED | {"kind": "feature"}),
        json.dumps(FIXED | {"status": "done"}),
        json.dumps(FIXED | {"summary": "two\nlines"}),
        json.dumps(FIXED | {"summary": " "}),
        json.dumps(FIXED | {"file": 7}),
        json.dumps(FIXED | {"summary": "two\rlines"}),
        json.dumps(FIXED | {"summary": "ends in a return\r"}),
        json.dumps(FIXED | {"summary": "ends in a newline\n"}),
    ]
    bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
    missing = tmp_path / "absent.jsonl"
    out = tmp_path / "report.html"
    assert findings_report.main([str(bad), str(missing), "-o", str(out)]) == 1
    assert not out.exists()
    errors = capsys.readouterr().err.splitlines()
    assert [error.removeprefix(f"findings-report: {bad}:") for error in errors[:-1]] == [
        "2: not JSON (Expecting property name enclosed in double quotes)",
        "3: not a JSON object",
        "4: missing test; unknown tset",
        "5: line is not an integer",
        "6: line is not an integer",
        "7: line is negative",
        "8: kind is not one of bug, test, doc, lint, open",
        "9: status is not one of open, fixed, rejected",
        "10: summary is not one line of text",
        "11: summary is not one line of text",
        "12: file is not a string",
        "13: summary is not one line of text",
        "14: summary is not one line of text",
        "15: summary is not one line of text",
    ]
    assert errors[-1].startswith(f"findings-report: {missing}: ") and "No such file" in errors[-1]


def test_a_line_ends_at_a_newline_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A string may hold U+2028 unescaped, and a file may end its lines with CR LF."""
    entry = FIXED | {"test": "tests/test_a.py::test_clear[a\u2028b]"}
    path = tmp_path / "fixer.jsonl"
    path.write_bytes((json.dumps(entry, ensure_ascii=False) + "\r\n").encode())
    assert findings_report.main([str(path)]) == 0
    assert "test_clear[a\u2028b]" in capsys.readouterr().out
