"""Concurrent first writers of the autorouter routing ledger.

``log_decision`` creates ``$KISS_HOME/MODEL_DECISIONS.md`` with its table
header the first time a row is logged.  Two first writers that both saw the
ledger missing used to both write the header, and the second ``write_text``
truncated the row the first had already appended.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from kiss.agents.seas.autorouter import autorouter_sea
from kiss.core.file_lock import lock_exclusive

WRITERS = 8
ROUNDS = 40


def _log_after_barrier(barrier: threading.Barrier, index: int, results: list[str]) -> None:
    """Wait for every writer, then log one decision whose unit names *index*."""
    barrier.wait(timeout=30)
    results[index] = autorouter_sea.log_decision(f"unit-{index}", "small", "m", "r")


def _race_first_writers(home: Path) -> list[str]:
    """Run :data:`WRITERS` concurrent first ``log_decision`` calls on a fresh *home*.

    Returns:
        The ledger's lines after every writer finished.
    """
    os.environ["KISS_HOME"] = str(home)
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
    assert results == [f"logged to {home / autorouter_sea.LEDGER_NAME}"] * WRITERS, results
    names = sorted(p.name for p in home.iterdir())
    assert names == [autorouter_sea.LEDGER_NAME], names
    return (home / autorouter_sea.LEDGER_NAME).read_text(encoding="utf-8").splitlines()


def _race_every_round(root: str) -> None:
    """Race the first writers :data:`ROUNDS` times, each round on a fresh home under *root*.

    The body of :func:`_racing_child`'s process, which sets ``KISS_HOME`` for good and where
    pytest rewrites no assertion: a failed one names what it read.
    """
    header = autorouter_sea.LEDGER_HEADER.splitlines()
    for round_index in range(ROUNDS):
        lines = _race_first_writers(Path(root) / f"home-{round_index}")
        assert lines[: len(header)] == header, lines
        assert lines.count(header[0]) == 1, lines
        rows = lines[len(header) :]
        units = sorted(row.split(" | ")[2] for row in rows)
        assert units == sorted(f"unit-{i}" for i in range(WRITERS)), lines


@contextmanager
def _racing_child(root: Path) -> Iterator[subprocess.Popen[bytes]]:
    """Run :func:`_race_every_round` on *root* in a child interpreter for the block.

    A writer that never gets the ledger lock is a thread nothing can stop: in the
    pytest process it outlived its failed test and kept the interpreter from exiting.
    The child owns the writers instead: it is killed (when it still runs) and reaped as
    the block ends, however it ends.  It exits with 0 when every assertion of every
    round held; a failed one is on its stderr, which is pytest's.
    """
    code = f"import sys; from {__name__} import _race_every_round; _race_every_round(sys.argv[1])"
    child = subprocess.Popen([sys.executable, "-c", code, str(root)])
    try:
        yield child
    finally:
        child.kill()
        child.wait(timeout=30)


def test_concurrent_first_writers_keep_one_header_and_every_row(tmp_path: Path) -> None:
    """Racing first writers leave the header once, first, followed by all their rows."""
    with _racing_child(tmp_path) as child:
        assert child.wait(timeout=300) == 0


def test_writers_stuck_on_the_ledger_lock_die_with_their_child(tmp_path: Path) -> None:
    """Writers that wait for the ledger lock for ever end with the block that owns their child.

    The second round's ledger is locked here before the child starts: past its first
    round, the child's writers poll that lock until the child is killed.
    """
    first = tmp_path / "home-0" / autorouter_sea.LEDGER_NAME
    stuck = tmp_path / "home-1" / autorouter_sea.LEDGER_NAME
    stuck.parent.mkdir()
    stuck.write_text(autorouter_sea.LEDGER_HEADER, encoding="utf-8")
    with stuck.open("a", encoding="utf-8") as held:
        assert lock_exclusive(held, blocking=False)
        with _racing_child(tmp_path) as child:
            deadline = time.monotonic() + 120
            while not first.exists():
                assert time.monotonic() < deadline, "the child never started its first round"
                time.sleep(0.02)
            with pytest.raises(subprocess.TimeoutExpired):
                child.wait(timeout=1)
    assert child.returncode not in (None, 0)  # killed, then reaped: none of its writers is left
    assert stuck.read_text(encoding="utf-8") == autorouter_sea.LEDGER_HEADER
