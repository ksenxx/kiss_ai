# pyright: reportImplicitRelativeImport=false
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from cleverest_plus.core import (
    Candidate,
    CandidateError,
    CrashSignature,
    Execution,
    Outcome,
    classify_differential,
    crash_signature,
    materialize_argv,
    select_diverse,
    validate_format,
)


def candidate(argv: list[str] | None = None, data: bytes = b"{}") -> Candidate:
    text = json.dumps({
        "input_b64": base64.b64encode(data).decode(),
        "argv": argv or ["jq", ".", "@@"],
        "rationale": "generic",
    })
    return Candidate.from_json(text, allowed_executables={"jq", "xmllint"})


def test_structured_candidate_round_trip() -> None:
    c = candidate(data=b'\x00PDF')
    assert c.data == b'\x00PDF'
    assert c.argv_template == ("jq", ".", "@@")


@pytest.mark.parametrize("argv", [
    ["jq", ".", "@@; touch /tmp/pwned"],
    ["jq", "$(cat /secret)", "@@"],
    ["sh", "-c", "@@"],
    ["../jq", ".", "@@"],
    ["jq", ".", "@@", "@@"],
    ["jq", "."],
])
def test_rejects_shell_and_path_injection(argv: list[str]) -> None:
    with pytest.raises(CandidateError):
        _ = candidate(argv)


def test_materialize_is_argv_not_shell(tmp_path: Path) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    inp = tmp_path / "input;touch-pwned"
    got = materialize_argv(candidate(), bindir, inp)
    assert got == [str((bindir / "jq").resolve()), ".", str(inp)]


def test_oracle_ignores_spoofed_stdout() -> None:
    fake = b"ERROR: AddressSanitizer: heap-buffer-overflow\n#0 fake"
    assert crash_signature(fake) is not None  # the same bytes on stderr would count
    before = Execution(0, b"normal", b"")
    after = Execution(0, fake, b"")
    assert classify_differential(before, after, scenario="BIC", changed_lines_reached=True) is Outcome.DIFFERENT


def test_oracle_parses_stderr_and_requires_identity_when_given() -> None:
    err = b"ERROR: AddressSanitizer: heap-buffer-overflow\n    #0 0x123 in vulnerable_fn /x.c:4"
    sig = crash_signature(err)
    assert sig == CrashSignature("AddressSanitizer", "heap-buffer-overflow", "vulnerable_fn")
    before = Execution(0, b"", b"")
    after = Execution(1, b"", err)
    assert classify_differential(before, after, scenario="BIC", changed_lines_reached=True, expected_signature=sig) is Outcome.BUG
    wrong = CrashSignature("AddressSanitizer", "heap-buffer-overflow", "other_fn")
    assert classify_differential(before, after, scenario="BIC", changed_lines_reached=True, expected_signature=wrong) is Outcome.AMBIGUOUS


def test_both_versions_crash_is_ambiguous_even_if_types_differ() -> None:
    a = Execution(1, b"", b"ERROR: AddressSanitizer: heap-buffer-overflow")
    b = Execution(1, b"", b"ERROR: AddressSanitizer: stack-buffer-overflow")
    assert classify_differential(a, b, scenario="BIC", changed_lines_reached=True) is Outcome.AMBIGUOUS


def test_format_validators() -> None:
    assert validate_format(b'{"x":1}', "json")[0]
    assert not validate_format(b'{', "json")[0]
    assert validate_format(b"%PDF-1.4\n%%EOF", "pdf")[0]
    assert not validate_format(b"not pdf", "pdf")[0]


def test_coverage_diversity() -> None:
    a, b, c = candidate(data=b"a"), candidate(data=b"b"), candidate(data=b"c")
    assert select_diverse([(a, [2, 1]), (b, [1, 2]), (c, [3])], 5) == [a, c]
