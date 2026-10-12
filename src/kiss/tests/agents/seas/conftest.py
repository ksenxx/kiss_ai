# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Fixtures shared by the SEA tests."""

from __future__ import annotations

import pytest

from kiss.agents.sorcar.persistence import _flush_chat_events, _get_db, _rw_lock


@pytest.fixture
def empty_history() -> None:
    """Start the test with no rows in the session's shared task history.

    The whole pytest process shares one ``KISS_HOME`` (see the top-level
    ``conftest.py``), so a test that lists "every recent run" would also
    see the rows earlier tests persisted, in whatever order they ran.
    Earlier agent runs also leave ``events`` rows that reference their
    ``task_history`` rows (the foreign key is enforced), so the writer
    thread is drained and the events go first, under the write lock the
    writer also takes.
    """
    _flush_chat_events()
    db = _get_db()
    with _rw_lock.write_lock():
        db.execute("DELETE FROM events")
        db.execute("DELETE FROM task_history")
        db.commit()
