# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: N812, E501
"""End-to-end reproducing tests for Phase-5 ROUND 4 review bugs.

Each test reproduces a CRITICAL or HIGH finding from
``tmp/review_*_r4.md`` against the post-round-4-fix code.  The tests
assert the FIXED behavior; running them on the pre-fix source raises
``AssertionError`` (or the underlying bug itself).
"""

from __future__ import annotations


def test_vscode_server_source_accepts_int_rebound_task_id() -> None:
    from pathlib import Path

    from kiss.server.server import _coerce_id

    src = Path("src/kiss/server/server.py").read_text()
    assert "rebound_task_id = _coerce_id(" in src, (
        "r4-vscode-H1: ``rebound_task_id`` extraction must accept int"
    )
    assert _coerce_id(42) == "42", (
        "r4-vscode-H1: int rebound task_id must be stringified"
    )


def test_vscode_server_source_accepts_int_entry_id() -> None:
    from pathlib import Path

    from kiss.server.server import _coerce_id

    src = Path("src/kiss/server/server.py").read_text()
    assert 'entry_id = _coerce_id(entry.get("id"))' in src, (
        "r4-vscode-H2: ``entry_id`` must accept int rows from legacy DBs"
    )
    assert _coerce_id(7) == "7", (
        "r4-vscode-H2: int ``entry_id`` must be stringified"
    )
    assert _coerce_id(0) is None
