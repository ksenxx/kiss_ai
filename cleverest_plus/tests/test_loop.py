# pyright: reportImplicitRelativeImport=false
"""End-to-end feedback-loop tests using a stub LLM (no network calls)."""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from cleverest_plus.benchmark import load_issue
from cleverest_plus.dispatcher import PortfolioMember, run_issue
from cleverest_plus.llm import ChatMessage, LLMResponse
from cleverest_plus.loop import run_trial


class StubClient:
    """Deterministic client returning a canned sequence of JSON candidate replies."""

    def __init__(self, replies: Sequence[str], model: str = "stub") -> None:
        self._replies = list(replies)
        self._i = 0
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def generate(self, messages: Sequence[ChatMessage], *, seed: int | None = None) -> LLMResponse:
        text = self._replies[min(self._i, len(self._replies) - 1)]
        self._i += 1
        return LLMResponse(text=text, prompt_tokens=1, completion_tokens=1,
                           model=self._model, latency_s=0.0)


@dataclass
class _Fixture:
    benchmark_root: Path


@pytest.fixture(scope="module")
def benchmark_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("bench")
    src = Path(__file__).resolve().parents[1] / "benchmark"
    if not src.is_dir():
        pytest.skip("mini-benchmark not built")
    shutil.copytree(src, root, dirs_exist_ok=True)
    return root


def _issue(benchmark_root: Path, subject: str, bug_id: str):
    return next(iter(load_issue(benchmark_root / "subjects" / subject / bug_id)))


def _reply(argv: list[str], data: bytes, rationale: str = "test") -> str:
    return json.dumps({
        "input_b64": base64.b64encode(data).decode(),
        "argv": argv,
        "rationale": rationale,
    })


def test_parselen_bic_finds_bug_with_correct_candidate(benchmark_root: Path, tmp_path: Path) -> None:
    """A length prefix of 0x10 followed by 16 bytes should crash the buggy after build."""
    if shutil.which("clang") is None:
        pytest.skip("clang not available")
    issue = _issue(benchmark_root, "parselen", "bic-01")
    reply = _reply(["parselen", "@@"], b"\x10" + b"A" * 16)
    client = StubClient([reply])
    result = run_trial(
        issue=issue, portfolio_member="stub", client=client, trial_index=1,
        iteration_cap=1, tmp_root=tmp_path,
    )
    assert result.outcome == "bug"
    assert result.ordinal_code == 3


def test_parselen_fix_finds_bug_on_before_side(benchmark_root: Path, tmp_path: Path) -> None:
    issue = _issue(benchmark_root, "parselen", "fix-01")
    reply = _reply(["parselen", "@@"], b"\x10" + b"B" * 16)
    client = StubClient([reply])
    result = run_trial(
        issue=issue, portfolio_member="stub", client=client, trial_index=1,
        iteration_cap=1, tmp_root=tmp_path,
    )
    assert result.outcome == "bug"


def test_parselen_uses_reference_signature_to_reject_unrelated_crash(
    benchmark_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash whose top-frame differs from the reference is Ambiguous, not Bug."""
    issue = _issue(benchmark_root, "parselen", "bic-01")
    from cleverest_plus.core import CrashSignature
    fake_ref = CrashSignature("AddressSanitizer", "heap-buffer-overflow",
                              "some_other_function_that_does_not_exist")
    # Replace the reference signature.
    from dataclasses import replace
    monkeypatched_issue = replace(issue, reference_signature=fake_ref)
    reply = _reply(["parselen", "@@"], b"\x10" + b"C" * 16)
    client = StubClient([reply])
    result = run_trial(
        issue=monkeypatched_issue, portfolio_member="stub", client=client, trial_index=1,
        iteration_cap=1, tmp_root=tmp_path,
    )
    assert result.iterations[-1].outcome == "ambiguous"
    assert result.ordinal_code == 0


def test_dispatcher_allocates_and_runs_trials(benchmark_root: Path, tmp_path: Path) -> None:
    issue = _issue(benchmark_root, "parselen", "bic-01")
    good = _reply(["parselen", "@@"], b"\x10" + b"D" * 16)
    portfolio = [
        PortfolioMember(name="m1", iteration_cap=1,
                        client_factory=lambda: StubClient([good], model="m1")),
        PortfolioMember(name="m2", iteration_cap=1,
                        client_factory=lambda: StubClient([good], model="m2")),
        PortfolioMember(name="m3", iteration_cap=1,
                        client_factory=lambda: StubClient([good], model="m3")),
    ]
    trials = run_issue(issue=issue, portfolio=portfolio, total_trials=10, tmp_root=tmp_path)
    assert len(trials) == 10
    assert [t.portfolio_member for t in trials].count("m1") == 4
    assert [t.portfolio_member for t in trials].count("m2") == 3
    assert [t.portfolio_member for t in trials].count("m3") == 3
    assert all(t.ordinal_code == 3 for t in trials)


def test_parser_rejects_shell_injection_reply(benchmark_root: Path, tmp_path: Path) -> None:
    issue = _issue(benchmark_root, "parselen", "bic-01")
    reply = _reply(["parselen", ";touch /tmp/pwned", "@@"], b"\x10" + b"E" * 16)
    client = StubClient([reply])
    result = run_trial(
        issue=issue, portfolio_member="stub", client=client, trial_index=1,
        iteration_cap=1, tmp_root=tmp_path,
    )
    assert result.iterations[-1].parsed is False
    assert not Path("/tmp/pwned").exists()


def test_stdout_marker_does_not_trigger_bug(benchmark_root: Path, tmp_path: Path) -> None:
    """A safe input (no bug) must not be flagged as Bug even if stdout mentions ASan."""
    issue = _issue(benchmark_root, "parselen", "bic-01")
    reply = _reply(["parselen", "@@"], b"\x02AB")
    client = StubClient([reply])
    result = run_trial(
        issue=issue, portfolio_member="stub", client=client, trial_index=1,
        iteration_cap=1, tmp_root=tmp_path,
    )
    assert result.outcome != "bug"
