# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: N812, E501
"""End-to-end reproducing tests for Phase-5 ROUND 3 review bugs.

Each test reproduces a CRITICAL or HIGH finding from
``tmp/review_*_r3.md`` against the post-round-3-fix code.  The tests
assert the FIXED behavior; running them on the pre-fix source raises
``AssertionError`` (or, in a few cases, the underlying bug itself).
"""

from __future__ import annotations

from pathlib import Path


def test_task_runner_rejects_non_string_task_id() -> None:
    """Non-string ``taskId`` payloads are rejected by the shared guard.

    The guard was centralised into :func:`_client_task_id_of` (bughunt
    round 9); exercise its behaviour directly instead of asserting on
    source-code text.
    """
    from kiss.server.task_runner import _client_task_id_of

    assert _client_task_id_of({"taskId": "abc123"}) == "abc123"
    assert _client_task_id_of({}) == ""
    for bad in ([1], {"x": 1}, True, 7, 3.5, None):
        assert _client_task_id_of({"taskId": bad}) == ""


def test_server_accepts_legacy_int_parent_task_id() -> None:
    from kiss.server.server import _coerce_id

    src = Path(
        "src/kiss/server/server.py"
    ).read_text()
    assert (
        'parent_tid = _coerce_id(subagent_info.get("parent_task_id"))' in src
    )
    assert 'pid = _coerce_id(sub.get("parent_task_id"))' in src
    assert _coerce_id("a" * 32) == "a" * 32
    assert _coerce_id(99) == "99"
