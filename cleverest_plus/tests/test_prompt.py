# pyright: reportImplicitRelativeImport=false
"""Prompt-assembly unit tests."""
from __future__ import annotations

import pytest

from cleverest_plus.prompt import AttemptRecord, CommitContext, build_messages


def _commit() -> CommitContext:
    return CommitContext(
        short_hash="deadbee",
        message="fix off-by-one",
        diff="--- a/x.c\n+++ b/x.c\n@@\n-if(n<8)\n+if(n<=8)\n",
        subject_description="a small parser",
        argv_template=("parselen", "@@"),
        format_name="bytes",
    )


def test_bic_prompt_mentions_target_after_side() -> None:
    msgs = build_messages(scenario="BIC", commit=_commit(), prior_attempts=[])
    assert msgs[0].role == "system"
    assert "sanitizer failure" in msgs[1].content
    assert "AFTER the commit but NOT" in msgs[1].content


def test_fix_prompt_mentions_target_before_side() -> None:
    msgs = build_messages(scenario="FIX", commit=_commit(), prior_attempts=[])
    assert "BEFORE the commit (still buggy)" in msgs[1].content
    assert "NOT in the program AFTER the commit" in msgs[1].content


def test_prior_attempts_history_included() -> None:
    prior = [AttemptRecord(
        input_bytes=b"hello",
        argv=("parselen", "@@"),
        reached_changed_lines=True,
        behavior_changed=False,
        before_stdout_tail="",
        after_stdout_tail="",
        before_stderr_tail="",
        after_stderr_tail="",
        before_returncode=0,
        after_returncode=0,
    )]
    msgs = build_messages(scenario="BIC", commit=_commit(), prior_attempts=prior)
    assert "Previous try #1" in msgs[1].content
    assert "hello" in msgs[1].content


def test_rejects_unknown_scenario() -> None:
    with pytest.raises(ValueError):
        _ = build_messages(scenario="OTHER", commit=_commit(), prior_attempts=[])
