"""End-to-end reproducing tests for bugs flagged by the gpt-5.5 review.

Each test reproduces a CRITICAL or HIGH bug from
``tmp/review_persistence.md``, ``tmp/review_vscode.md`` or
``tmp/review_sorcar_other.md``.  After the fix is in place every test
should pass; before the fix, each test failed deterministically.
"""

from __future__ import annotations

from pathlib import Path


def test_vs_bug1_replay_uses_str_parent_task_id() -> None:
    """The ``parent_tid`` extraction in ``_replay_session`` must accept str.

    Before the fix: ``isinstance(parent_tid_raw, int)`` — always False
    for the str UUID round-trip.  The coercion is now centralised in
    ``_coerce_id``: str UUIDs pass through unchanged (the primary,
    post-refactor canonical contract) and the r3-vscode-H2 int
    fallback stringifies legacy ids rather than replacing the str path.
    """
    from kiss.server.server import _coerce_id

    src = Path(
        "src/kiss/server/server.py"
    ).read_text()
    assert (
        'parent_tid = _coerce_id(subagent_info.get("parent_task_id"))' in src
    )
    assert _coerce_id("deadbeef" * 4) == "deadbeef" * 4
    assert _coerce_id(5) == "5"
    assert _coerce_id("") is None
    assert _coerce_id([5]) is None


def test_vs_bug2_shutdown_helper_accepts_uuid_strings() -> None:
    """``active_task_history_ids`` must be a ``set[str]`` — no ``int(...)``.

    Reproduces by reading the source and asserting the
    ``set[int]``/``int(th_id)`` patterns are gone.  A direct functional
    repro would require spinning up a live agent-state registry /
    websocket / shutdown sequence — covered by the existing E2E test
    suite's daemon-shutdown tests.
    """
    src = Path(
        "src/kiss/server/web_server.py"
    ).read_text()
    assert "active_task_history_ids: set[int]" not in src
    assert "active_task_history_ids: set[str]" in src
    assert "active_task_history_ids.add(int(" not in src


def test_vs_bug3_commands_reject_non_string_taskid() -> None:
    """A non-string ``taskId`` payload must be dropped before SQL.

    The previous pattern ``str(raw_task_id) if raw_task_id else None``
    accepted dicts and lists and stringified them.  All three relevant
    handlers now validate through the shared ``_opt_str`` guard, which
    rejects every non-string payload.
    """
    from kiss.server.commands import _opt_str

    src = Path(
        "src/kiss/server/commands.py"
    ).read_text()
    occurrences = src.count('task_id = _opt_str(cmd.get("taskId"))')
    assert occurrences == 3, (
        f"expected 3 hardened taskId guards, found {occurrences}"
    )
    assert _opt_str({"a": 1}) is None
    assert _opt_str([1]) is None
    assert _opt_str(7) is None
    assert _opt_str("") is None
    assert _opt_str("tid") == "tid"
