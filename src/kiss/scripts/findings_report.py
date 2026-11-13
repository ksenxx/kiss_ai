# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Assemble the findings that fixers and reviewers reported into the table of a final report.

Each child of an orchestrating task appends one JSON object per finding,
on one line, to a file of its own (``SYSTEM.md``, "One line per finding")::

    {"file": "src/x.py", "line": 12, "kind": "bug", "status": "fixed",
     "summary": "one sentence", "test": "path::test_name"}

``uv run findings-report <files> -o <report.html>`` checks every line and
writes the counts and the table as an HTML fragment (to standard output
without ``-o``), so the report is not composed from the children's prose.
A malformed line is named with its file, its number and what is wrong
with it, and the run exits 1 without writing a report.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

KINDS = ("bug", "test", "doc", "lint", "open")
"""What a finding is about."""

STATUSES = ("open", "fixed", "rejected")
"""What became of a finding, in the order the report lists them."""

FIELDS = {"file": str, "line": int, "kind": str, "status": str, "summary": str, "test": str}
"""The keys of a finding and their types; ``line`` is 0 and ``test`` empty when there is none."""


def problem(entry: Any) -> str:
    """Return what is wrong with one parsed line of a findings file, or ``""``."""
    if not isinstance(entry, dict):
        return "not a JSON object"
    if set(entry) != set(FIELDS):
        missing = sorted(set(FIELDS) - set(entry))
        unknown = sorted(set(entry) - set(FIELDS))
        return "; ".join(
            f"{label} {', '.join(keys)}"
            for label, keys in (("missing", missing), ("unknown", unknown)) if keys
        )
    for key, kind in FIELDS.items():
        if type(entry[key]) is not kind:
            return f"{key} is not {'an integer' if kind is int else 'a string'}"
    if entry["line"] < 0:
        return "line is negative"
    if entry["kind"] not in KINDS:
        return f"kind is not one of {', '.join(KINDS)}"
    if entry["status"] not in STATUSES:
        return f"status is not one of {', '.join(STATUSES)}"
    if not entry["summary"].strip() or entry["summary"].splitlines() != [entry["summary"]]:
        return "summary is not one line of text"
    return ""


def read_findings(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Return the findings of *path*, each with the file's stem as ``source``, and its errors.

    A line ends at a newline only (a string may hold any other line
    separator) and a blank line is skipped.  An error names *path*, the
    line's number and what is wrong with it; a file that cannot be read is
    one error.
    """
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except (OSError, UnicodeDecodeError) as error:
        return [], [f"{path}: {error}"]
    findings: list[dict[str, Any]] = []
    errors: list[str] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"{path}:{number}: not JSON ({error.msg})")
            continue
        wrong = problem(entry)
        if wrong:
            errors.append(f"{path}:{number}: {wrong}")
        else:
            findings.append(entry | {"source": path.stem})
    return findings, errors


def _order(finding: dict[str, Any]) -> tuple[int, str, int, str]:
    """Return the sort key of a row: status as in :data:`STATUSES`, then file, line, source."""
    return STATUSES.index(finding["status"]), finding["file"], finding["line"], finding["source"]


def _row(finding: dict[str, Any]) -> str:
    """Return the table row of *finding*."""
    where = f"{finding['file']}:{finding['line']}" if finding["line"] else finding["file"]
    cells = (
        finding["status"], finding["kind"], where, finding["summary"], finding["test"],
        finding["source"],
    )
    return "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells) + "</tr>"


def render(findings: list[dict[str, Any]]) -> str:
    """Return the counts and the table of *findings* as an HTML fragment.

    Open findings come first, then fixed, then rejected; within a status
    the rows follow file and line.
    """
    statuses = Counter(finding["status"] for finding in findings)
    kinds = Counter(finding["kind"] for finding in findings)
    counts = ", ".join(f"{statuses[status]} {status}" for status in STATUSES)
    by_kind = ", ".join(f"{kinds[kind]} {kind}" for kind in KINDS if kinds[kind])
    return "\n".join([
        "<h3>Findings</h3>",
        f"<p>{len(findings)} findings: {counts}." + (f" By kind: {by_kind}." if by_kind else "")
        + "</p>",
        "<table>",
        "<tr><th>Status</th><th>Kind</th><th>Where</th><th>Finding</th><th>Test</th>"
        "<th>Reported by</th></tr>",
        *(_row(finding) for finding in sorted(findings, key=_order)),
        "</table>",
    ]) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write the report of the findings files named on the command line.

    Returns:
        0, or 1 on a malformed line or a report that cannot be written.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("files", nargs="+", type=Path, help="findings files, one JSON line each")
    parser.add_argument("-o", "--output", type=Path, help="write the HTML here (default: stdout)")
    args = parser.parse_args(argv)
    findings: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in args.files:
        found, wrong = read_findings(path)
        findings += found
        errors += wrong
    if errors:
        print("findings-report: " + "\nfindings-report: ".join(errors), file=sys.stderr)
        return 1
    report = render(findings)
    if not args.output:
        sys.stdout.write(report)
        return 0
    try:
        args.output.write_text(report, encoding="utf-8")
    except OSError as error:
        print(f"findings-report: {args.output}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
