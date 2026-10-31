"""Concurrent first writers of the autorouter routing ledger.

``log_decision`` creates ``$KISS_HOME/MODEL_DECISIONS.md`` with its table
header the first time a row is logged.  Two first writers that both saw the
ledger missing used to both write the header, and the second ``write_text``
truncated the row the first had already appended.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from kiss.agents.seas.autorouter import autorouter_sea

WRITERS = 8
ROUNDS = 40


def _log_after_barrier(barrier: threading.Barrier, index: int, results: list[str]) -> None:
    """Wait for every writer, then log one decision whose unit names *index*."""
    barrier.wait(timeout=30)
    results[index] = autorouter_sea.log_decision(f"unit-{index}", "small", "m", "r")


def _race_first_writers(home: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Run :data:`WRITERS` concurrent first ``log_decision`` calls on a fresh *home*.

    Returns:
        The ledger's lines after every writer finished.
    """
    monkeypatch.setenv("KISS_HOME", str(home))
    barrier = threading.Barrier(WRITERS)
    results = [""] * WRITERS
    threads = [
        threading.Thread(target=_log_after_barrier, args=(barrier, i, results))
        for i in range(WRITERS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()
    assert results == [f"logged to {home / autorouter_sea.LEDGER_NAME}"] * WRITERS
    assert sorted(p.name for p in home.iterdir()) == [autorouter_sea.LEDGER_NAME]
    return (home / autorouter_sea.LEDGER_NAME).read_text(encoding="utf-8").splitlines()


def test_concurrent_first_writers_keep_one_header_and_every_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Racing first writers leave the header once, first, followed by all their rows."""
    header = autorouter_sea.LEDGER_HEADER.splitlines()
    for round_index in range(ROUNDS):
        lines = _race_first_writers(tmp_path / f"home-{round_index}", monkeypatch)
        assert lines[: len(header)] == header
        assert lines.count(header[0]) == 1
        rows = lines[len(header) :]
        units = sorted(row.split(" | ")[2] for row in rows)
        assert units == sorted(f"unit-{i}" for i in range(WRITERS))


def test_a_writer_after_the_ledger_exists_appends_below_the_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later row is appended under the existing ones without a second header."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    autorouter_sea.log_decision("first", "small", "m", "r")
    autorouter_sea.log_decision("second", "frontier", "m", "r", "passed")
    lines = (tmp_path / autorouter_sea.LEDGER_NAME).read_text(encoding="utf-8").splitlines()
    header = autorouter_sea.LEDGER_HEADER.splitlines()
    assert lines[: len(header)] == header
    assert [row.split(" | ")[2] for row in lines[len(header) :]] == ["first", "second"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [autorouter_sea.LEDGER_NAME]
