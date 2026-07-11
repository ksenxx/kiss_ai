"""Benchmark data model and loader for the end-to-end campaign.

The Cleverest+ campaign operates on Issues (a bug id + a scenario). Each Issue
carries the pre-built before/after binaries, the argv template with a single "@@"
input placeholder, the commit metadata, and an optional reference sanitizer
signature learned from the known crashing input (retained under
signature/reference.txt).

The benchmark layout is:

    benchmark/
      subjects/
        <subject>/
          <bug_id>/
            issue.json
            before/<executable>
            after/<executable>
            diff.patch                     (unified git-diff)
            message.txt                    (commit message)
            changed_lines/before.json      ({file: [line, ...]})
            changed_lines/after.json
            signature/reference.txt        (optional ASan stderr transcript)

The loader validates that every referenced file exists and that executables are
executable. It never invokes a shell.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .core import CrashSignature, crash_signature


@dataclass(frozen=True)
class Issue:
    """One (bug, scenario) pair with pre-built before/after binaries."""

    subject: str
    bug_id: str
    scenario: str
    subject_description: str
    format_name: str
    argv_template: tuple[str, ...]
    before_dir: Path
    after_dir: Path
    commit_short_hash: str
    commit_message: str
    diff: str
    changed_lines_before: dict[str, tuple[int, ...]]
    changed_lines_after: dict[str, tuple[int, ...]]
    reference_signature: CrashSignature | None

    @property
    def identifier(self) -> str:
        """Return a stable string identifier: subject/bug_id/scenario."""
        return f"{self.subject}/{self.bug_id}/{self.scenario}"

    def target_dir(self) -> Path:
        """Directory whose binary must crash for a Bug outcome (per scenario)."""
        return self.after_dir if self.scenario == "BIC" else self.before_dir

    def other_dir(self) -> Path:
        """Directory whose binary must NOT crash for a Bug outcome."""
        return self.before_dir if self.scenario == "BIC" else self.after_dir


def _load_lines(path: Path) -> dict[str, tuple[int, ...]]:
    raw = cast(dict[str, list[int]], json.loads(path.read_text()))
    return {file: tuple(lines) for file, lines in raw.items()}


def _validate_executables(exe_name: str, before: Path, after: Path) -> None:
    for direction, directory in (("before", before), ("after", after)):
        binary = directory / exe_name
        if not binary.is_file():
            raise FileNotFoundError(f"missing {direction} binary: {binary}")
        if not (binary.stat().st_mode & 0o111):
            raise PermissionError(f"binary is not executable: {binary}")


def load_issue(issue_root: Path) -> Iterable[Issue]:
    """Load both BIC and FIX scenarios for one bug directory."""
    meta = cast(dict[str, object], json.loads((issue_root / "issue.json").read_text()))
    argv = tuple(cast(list[str], meta["argv_template"]))
    exe = argv[0]
    before_dir = issue_root / "before"
    after_dir = issue_root / "after"
    _validate_executables(exe, before_dir, after_dir)
    diff = (issue_root / "diff.patch").read_text()
    message = (issue_root / "message.txt").read_text()
    changed_before = _load_lines(issue_root / "changed_lines" / "before.json")
    changed_after = _load_lines(issue_root / "changed_lines" / "after.json")
    ref_path = issue_root / "signature" / "reference.txt"
    ref_sig = crash_signature(ref_path.read_bytes()) if ref_path.is_file() else None
    subject_desc = cast(str, meta["subject_description"])
    fmt = cast(str, meta["format_name"])
    short_hash = cast(str, meta["commit_short_hash"])
    bug_id = cast(str, meta["bug_id"])
    subject = cast(str, meta["subject"])
    scenarios = cast(list[str], meta.get("scenarios", ["BIC", "FIX"]))
    for scenario in scenarios:
        if scenario not in {"BIC", "FIX"}:
            raise ValueError(f"invalid scenario in {issue_root}: {scenario}")
        yield Issue(
            subject=subject,
            bug_id=bug_id,
            scenario=scenario,
            subject_description=subject_desc,
            format_name=fmt,
            argv_template=argv,
            before_dir=before_dir,
            after_dir=after_dir,
            commit_short_hash=short_hash,
            commit_message=message,
            diff=diff,
            changed_lines_before=changed_before,
            changed_lines_after=changed_after,
            reference_signature=ref_sig,
        )


def load_benchmark(root: Path) -> Iterator[Issue]:
    """Enumerate every Issue under root/subjects/*/*/."""
    subjects_root = root / "subjects"
    if not subjects_root.is_dir():
        raise FileNotFoundError(f"no subjects directory at {subjects_root}")
    for subject_dir in sorted(subjects_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        for issue_dir in sorted(subject_dir.iterdir()):
            if not issue_dir.is_dir() or not (issue_dir / "issue.json").is_file():
                continue
            yield from load_issue(issue_dir)


def approximate_reached(
    *,
    covered_before_lines: dict[str, tuple[int, ...]],
    covered_after_lines: dict[str, tuple[int, ...]],
    changed_before: dict[str, tuple[int, ...]],
    changed_after: dict[str, tuple[int, ...]],
) -> bool:
    """Return True when covered lines intersect changed lines on either side.

    This is a coarse fallback used when gcov instrumentation is not available for the
    subject; the paper's Table 1 method uses gcov exclusively.
    """
    def _intersects(covered: dict[str, tuple[int, ...]], changed: dict[str, tuple[int, ...]]) -> bool:
        for file, lines in changed.items():
            covered_lines = set(covered.get(file, ()))
            if covered_lines & set(lines):
                return True
        return False
    return _intersects(covered_before_lines, changed_before) or _intersects(covered_after_lines, changed_after)
