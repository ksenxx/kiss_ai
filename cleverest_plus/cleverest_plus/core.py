"""Hardened building blocks for a commit-directed regression-test generator.

This module is deliberately independent from benchmark issue IDs and truth inputs.  It
implements reusable mechanics that can be wired into Cleverest's prompt/execution loop:
structured candidates, safe command construction, cheap format validation, coverage
novelty selection, and sanitizer-signature-based differential classification.
"""
from __future__ import annotations

import base64
import hashlib
from collections.abc import Iterable, Mapping, Sequence
import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import cast


class CandidateError(ValueError):
    """A model candidate cannot be safely parsed or executed."""


@dataclass(frozen=True)
class Candidate:
    data: bytes
    argv_template: tuple[str, ...]
    rationale: str = ""

    @classmethod
    def from_json(cls, text: str, *, allowed_executables: set[str]) -> "Candidate":
        """Parse the only accepted model-output format.

        Expected JSON: {"input_b64": "...", "argv": ["jq", ".", "@@"],
        "rationale": "..."}.  No shell string is accepted.
        """
        try:
            raw = cast(object, json.loads(text))
        except json.JSONDecodeError as exc:
            raise CandidateError(f"invalid JSON: {exc.msg}") from exc
        if not isinstance(raw, dict):
            raise CandidateError("candidate must be a JSON object")
        obj = cast(dict[str, object], raw)
        if set(obj) - {"input_b64", "argv", "rationale"}:
            raise CandidateError("unexpected structured-output fields")
        input_b64 = obj.get("input_b64")
        argv_raw = obj.get("argv")
        if not isinstance(input_b64, str) or not isinstance(argv_raw, list):
            raise CandidateError("input_b64 must be a string and argv must be an array")
        try:
            data = base64.b64decode(input_b64, validate=True)
        except Exception as exc:
            raise CandidateError("input_b64 is not canonical base64") from exc
        argv = validate_argv(cast(list[object], argv_raw), allowed_executables=allowed_executables)
        rationale = obj.get("rationale", "")
        if not isinstance(rationale, str):
            raise CandidateError("rationale must be a string")
        return cls(data=data, argv_template=argv, rationale=rationale)


_SHELL_META = re.compile(r"[;&|<>`\n\r]|\$\(|\$\{")


def validate_argv(values: Sequence[object], *, allowed_executables: set[str]) -> tuple[str, ...]:
    """Validate an argv vector; shell syntax and executable paths are forbidden."""
    if not values or not all(isinstance(v, str) for v in values):
        raise CandidateError("argv must contain non-empty strings")
    argv = tuple(cast(str, value) for value in values)
    exe = argv[0]
    if exe not in allowed_executables or Path(exe).name != exe:
        raise CandidateError(f"executable is not allowlisted: {exe!r}")
    if argv.count("@@") != 1:
        raise CandidateError("argv must contain exactly one @@ input placeholder")
    for arg in argv:
        if "\x00" in arg or _SHELL_META.search(arg):
            raise CandidateError(f"shell control syntax is forbidden: {arg!r}")
        # Reject a second token stream embedded in one field. Quoted spaces are not
        # needed because JSON already preserves each argument boundary.
        if arg != "@@" and len(shlex.split(arg)) != 1:
            raise CandidateError(f"each argv item must be one argument: {arg!r}")
    return argv


def materialize_argv(candidate: Candidate, executable_dir: Path, input_path: Path) -> list[str]:
    """Construct argv without invoking a shell."""
    exe = (executable_dir / candidate.argv_template[0]).resolve()
    root = executable_dir.resolve()
    if exe.parent != root:
        raise CandidateError("executable escaped its build directory")
    return [str(exe), *(str(input_path) if x == "@@" else x for x in candidate.argv_template[1:])]


@dataclass(frozen=True)
class Execution:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool = False


def execute(argv: Sequence[str], *, timeout_s: float = 30.0, env: Mapping[str, str] | None = None) -> Execution:
    """Execute a validated argv vector directly, with no shell and separated streams."""
    try:
        p = subprocess.run(
            list(argv), input=b"C\n", stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout_s, check=False, env=dict(env) if env is not None else None,
        )
        return Execution(p.returncode, p.stdout, p.stderr)
    except subprocess.TimeoutExpired as exc:
        return Execution(124, exc.stdout or b"", exc.stderr or b"", timed_out=True)


class Outcome(str, Enum):
    BUG = "bug"
    DIFFERENT = "different"
    REACHED = "reached"
    NONE = "none"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class CrashSignature:
    sanitizer: str
    kind: str
    top_frame: str | None


_ASAN = re.compile(rb"ERROR: (AddressSanitizer|UndefinedBehaviorSanitizer): ([A-Za-z0-9_-]+)")
_FRAME = re.compile(rb"(?:^|\n)\s*#0\s+[^\n]*?\s(?:in\s+)?([^\s+]+)")


def crash_signature(stderr: bytes) -> CrashSignature | None:
    """Parse only stderr produced by the process, never target-controlled stdout.

    A useful production deployment should additionally capture the sanitizer report on
    a dedicated inherited file descriptor. stderr-only is still strictly safer than
    Cleverest's combined pseudo-terminal output and makes the spoof POC fail.
    """
    m = _ASAN.search(stderr)
    if not m:
        return None
    f = _FRAME.search(stderr)
    return CrashSignature(m.group(1).decode(), m.group(2).decode(), f.group(1).decode() if f else None)


def classify_differential(
    before: Execution,
    after: Execution,
    *,
    scenario: str,
    changed_lines_reached: bool,
    expected_signature: CrashSignature | None = None,
) -> Outcome:
    """Classify without equating an arbitrary differential crash to the intended bug."""
    if scenario not in {"BIC", "FIX"}:
        raise ValueError("scenario must be BIC or FIX")
    sb, sa = crash_signature(before.stderr), crash_signature(after.stderr)
    target, other = (sa, sb) if scenario == "BIC" else (sb, sa)
    if target is not None:
        if other is not None:
            return Outcome.AMBIGUOUS
        if expected_signature is not None and target != expected_signature:
            return Outcome.AMBIGUOUS
        return Outcome.BUG
    if sb is not None or sa is not None:
        return Outcome.AMBIGUOUS
    if (before.stdout, before.stderr, before.returncode) != (after.stdout, after.stderr, after.returncode):
        return Outcome.DIFFERENT
    return Outcome.REACHED if changed_lines_reached else Outcome.NONE


def validate_format(data: bytes, format_name: str) -> tuple[bool, str]:
    """Cheap deterministic validation before expensive execution.

    Invalid inputs are not universally rejected—malformed-input testing is valuable—but
    the validation result can be fed back and used to maintain separate valid/malformed
    candidate quotas.
    """
    name = format_name.lower()
    try:
        if name == "json":
            json.loads(data.decode("utf-8"))
        elif name == "xml":
            import xml.etree.ElementTree as et
            _ = et.fromstring(data)
        elif name in {"javascript", "python", "smt2"}:
            _ = data.decode("utf-8")
        elif name == "pdf":
            if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-2048:]:
                return False, "missing PDF header or terminal EOF marker"
        else:
            return False, f"no validator registered for {format_name}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "valid"


def coverage_fingerprint(edges: Iterable[int]) -> str:
    canonical = ",".join(str(x) for x in sorted(set(edges))).encode()
    return hashlib.sha256(canonical).hexdigest()


def select_diverse(candidates: Iterable[tuple[Candidate, Iterable[int]]], limit: int) -> list[Candidate]:
    """Keep at most one candidate per observed edge-set fingerprint."""
    selected: list[Candidate] = []
    seen: set[str] = set()
    for candidate, edges in candidates:
        fp = coverage_fingerprint(edges)
        if fp in seen:
            continue
        seen.add(fp)
        selected.append(candidate)
        if len(selected) == limit:
            break
    return selected
