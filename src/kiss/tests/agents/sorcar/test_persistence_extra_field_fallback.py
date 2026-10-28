"""``_save_task_extra`` coerces each metadata field independently.

A field whose conversion fails (``__str__`` raising) must default on
its own instead of aborting the whole UPDATE, and non-finite floats
handed to the ``is_parallel`` / ``is_cron`` flag columns must store
``0``, exactly as the pre-consolidation per-field guards did.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import _add_task, _save_task_extra


class BadText:
    """Object whose ``str()`` always raises."""

    def __str__(self) -> str:
        raise RuntimeError("no text for you")


class TestSaveTaskExtraPerFieldFallback:
    """Fresh temp SQLite DB per test, fully restored after."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        th._db_conn = None
        th._invalidate_chat_context_cache("")

    def teardown_method(self) -> None:
        th._close_db()
        th._invalidate_chat_context_cache("")
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _row(self, task_id: str, cols: str) -> tuple[object, ...]:
        row = th._get_db().execute(
            f"SELECT {cols} FROM task_history WHERE id = ?", (task_id,)
        ).fetchone()
        return tuple(row)

    def test_failing_str_defaults_only_that_field(self) -> None:
        task_id, _ = _add_task("t", extra={"model": "gpt", "steps": 0})

        _save_task_extra({"steps": 7, "model": BadText()}, task_id=task_id)

        assert self._row(task_id, "steps, model") == (7, "")

    def test_non_finite_flags_store_zero(self) -> None:
        task_id, _ = _add_task("t", extra={"is_parallel": 1, "is_cron": 1})

        _save_task_extra(
            {"is_parallel": float("nan"), "is_cron": float("inf")},
            task_id=task_id,
        )

        assert self._row(task_id, "is_parallel, is_cron") == (0, 0)

    def test_non_finite_string_field_defaults(self) -> None:
        task_id, _ = _add_task("t", extra={"model": "gpt"})

        _save_task_extra({"model": float("nan"), "steps": 3}, task_id=task_id)

        assert self._row(task_id, "steps, model") == (3, "")
