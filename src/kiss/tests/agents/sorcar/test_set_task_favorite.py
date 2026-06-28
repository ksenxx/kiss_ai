# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the ``is_favorite`` flag on ``task_history.extra``.

Covers :func:`kiss.agents.sorcar.persistence._set_task_favorite`:

* defaulting (rows created without an ``is_favorite`` key have
  ``extra`` missing or no ``is_favorite``),
* setting the flag to ``True`` merges into existing ``extra`` JSON,
* setting the flag to ``False`` overwrites a previously-True value,
* unrelated extra keys (tokens, cost, steps, model, subagent) are
  preserved across favourite toggles,
* a non-existent task_id returns ``False`` and writes nothing.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import kiss.agents.sorcar.persistence as th


def _redirect(tmpdir: str):
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore(saved):
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved


def _read_extra(task_id: str) -> dict:
    """Return the synthesized ``extra`` JSON object for *task_id*."""
    db = th._get_db()
    row = db.execute(
        "SELECT * FROM task_history WHERE id = ?", (task_id,),
    ).fetchone()
    assert row is not None
    raw = th._row_to_extra_json(row)
    if not raw:
        return {}
    parsed: dict = json.loads(raw)
    return parsed


class TestSetTaskFavorite:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_default_is_false_when_no_extra(self) -> None:
        """Tasks created without ``extra`` have no ``is_favorite`` key."""
        task_id, _ = th._add_task("plain task")
        extra = _read_extra(task_id)
        # Empty extra column → empty dict → no is_favorite key, which
        # the server reads as False via dict.get default.
        assert extra.get("is_favorite", False) is False

    def test_set_favorite_true_on_empty_extra(self) -> None:
        """Setting True populates ``is_favorite`` in a fresh JSON object."""
        task_id, _ = th._add_task("starme")
        assert th._set_task_favorite(task_id, True) is True
        extra = _read_extra(task_id)
        assert extra.get("is_favorite") is True

    def test_set_favorite_false_overwrites_true(self) -> None:
        """Setting False after True flips the persisted flag back.

        The synthesized ``extra`` JSON omits ``is_favorite`` when the
        column is 0 (False) — server callers read it via
        ``dict.get("is_favorite", False)`` which yields the same False
        semantics.
        """
        task_id, _ = th._add_task("toggle")
        th._set_task_favorite(task_id, True)
        th._set_task_favorite(task_id, False)
        extra = _read_extra(task_id)
        assert extra.get("is_favorite", False) is False

    def test_set_favorite_preserves_other_keys(self) -> None:
        """Other extra keys must survive a favourite toggle."""
        task_id, _ = th._add_task("with metrics")
        th._save_task_extra(
            {
                "tokens": 1234,
                "cost": 0.05,
                "steps": 7,
                "model": "gpt-4o",
                "subagent": {
                    "parent_task_id": "99999999999999999999999999999999"
                },
            },
            task_id=task_id,
        )
        assert th._set_task_favorite(task_id, True) is True
        extra = _read_extra(task_id)
        assert extra["is_favorite"] is True
        assert extra["tokens"] == 1234
        assert extra["cost"] == 0.05
        assert extra["steps"] == 7
        assert extra["model"] == "gpt-4o"
        assert extra["subagent"] == {
            "parent_task_id": "99999999999999999999999999999999"
        }

    def test_set_favorite_unknown_task_returns_false(self) -> None:
        """A non-existent task_id returns False and writes nothing."""
        assert th._set_task_favorite(987654, True) is False

    def test_set_favorite_recovers_from_corrupt_column_value(self) -> None:
        """Corrupt ``is_favorite`` column value is replaced cleanly.

        The legacy schema stored is_favorite inside a JSON ``extra``
        column where malformed JSON had to be tolerated.  The new
        schema stores it in a dedicated ``is_favorite INTEGER`` column,
        so the corruption surface shrinks to non-0/1 integer values
        (or NULL).  This test asserts ``_set_task_favorite`` overwrites
        such values cleanly.
        """
        task_id, _ = th._add_task("garbage")
        db = th._get_db()
        db.execute(
            "UPDATE task_history SET is_favorite = ? WHERE id = ?",
            (42, task_id),
        )
        db.commit()
        assert th._set_task_favorite(task_id, True) is True
        extra = _read_extra(task_id)
        # r3-H3: ``_row_to_extra_json`` emits every typed column.
        # Pop the defaulted ones so the test focuses on the
        # write-under-test: ``is_favorite`` flips to True.
        for k in (
            "auto_commit_mode", "is_parallel", "is_worktree",
            "model", "work_dir", "version",
            "tokens", "cost", "steps", "startTs", "endTs",
        ):
            extra.pop(k, None)
        assert extra == {"is_favorite": True}
