# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 3: ``_HISTORY_NOT_SUBAGENT`` LIKE filter false positives.

``_load_history``, ``_search_history``, ``_get_history_entry`` and
``_prefix_match_task`` filter sub-agent rows with the raw SQL
substring predicate ``extra NOT LIKE '%"subagent"%'`` — with NO JSON
re-validation.  A regular (parent) task whose ``extra`` JSON merely
*contains* the substring ``"subagent"`` — e.g. a NESTED key
(``{"opts": {"subagent": false}}``) or a legacy/malformed non-JSON
value embedding the quoted word — is therefore wrongly hidden from
the history sidebar, history search, and prefix autocomplete.

This is inconsistent with ``_is_subagent_row`` — the canonical
sub-agent detector (proper ``json.loads`` + dict + key check) used by
``_load_chat_context``, ``_list_recent_chats`` and
``_load_latest_chat_events_by_chat_id`` — which correctly classifies
such a row as a normal parent task.  The same row is visible in the
chat context but invisible in the history list.

Runs against a real SQLite database redirected to a temp dir.  No
mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import (
    _add_task,
    _get_history_entry,
    _load_chat_context,
    _load_history,
    _prefix_match_task,
    _search_history,
)


class _TempDbTestBase:
    """Fresh temp SQLite DB per test, fully restored after."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        th._invalidate_chat_context_cache("")

    def teardown_method(self) -> None:
        th._close_db()
        th._invalidate_chat_context_cache("")
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestSubagentLikeFalsePositive(_TempDbTestBase):
    """Rows that merely CONTAIN '"subagent"' in extra must stay visible."""

    # A nested "subagent" key: NOT a sub-agent row per the canonical
    # detector (_is_subagent_row checks only the TOP-LEVEL key), yet
    # its JSON encoding contains the literal substring '"subagent"'.
    EXTRA: dict[str, object] = {"model": "m1", "opts": {"subagent": False}}

    def test_history_readers_keep_false_positive_row(self) -> None:
        task_text = "review subagent documentation"
        task_id, chat_id = _add_task(task_text, extra=self.EXTRA)

        # Canonical detector agrees this is NOT a sub-agent row …
        assert not th._is_subagent_row(
            th._get_db().execute(
                "SELECT extra FROM task_history WHERE id = ?", (task_id,)
            ).fetchone()["extra"]
        )
        # … and the chat context (JSON-validating reader) includes it.
        ctx = _load_chat_context(chat_id)
        assert [e["task"] for e in ctx] == [task_text]

        # The LIKE-only readers must agree and keep the row visible.
        hist = _load_history()
        assert [h["id"] for h in hist] == [task_id]

        found = _search_history("review")
        assert [h["id"] for h in found] == [task_id]

        entry = _get_history_entry(0)
        assert entry is not None
        assert entry["id"] == task_id

        assert _prefix_match_task("review sub") == task_text

    def test_true_subagent_rows_remain_filtered(self) -> None:
        parent_id, chat_id = _add_task("parent task")
        _add_task(
            "child task",
            chat_id=chat_id,
            extra={"subagent": {"parent_task_id": parent_id}},
        )

        hist = _load_history()
        assert [h["id"] for h in hist] == [parent_id]
        assert _search_history("child") == []
        entry = _get_history_entry(0)
        assert entry is not None
        assert entry["id"] == parent_id
        assert _prefix_match_task("child") == ""
        # Chat context also filters genuine sub-agent rows.
        ctx = _load_chat_context(chat_id)
        assert [e["task"] for e in ctx] == ["parent task"]

    def test_adjacent_navigation_does_not_skip_false_positive(self) -> None:
        t1, chat_id = _add_task("step one")
        t2, _ = _add_task("step two", chat_id=chat_id, extra=self.EXTRA)
        t3, _ = _add_task("step three", chat_id=chat_id)

        nxt = th._get_adjacent_task_by_chat_id(chat_id, t1, "next")
        assert nxt is not None
        assert nxt["task_id"] == t2
        prv = th._get_adjacent_task_by_chat_id(chat_id, t3, "prev")
        assert prv is not None
        assert prv["task_id"] == t2

    def test_legacy_malformed_extra_rows_stay_visible(self) -> None:
        # Malformed / non-dict extra must behave like _is_subagent_row
        # (False ⇒ regular row) in the SQL filter too.
        task_id, _ = _add_task("legacy row")
        db = th._get_db()
        with th._rw_lock.write_lock():
            db.execute(
                "UPDATE task_history SET extra = ? WHERE id = ?",
                ('{not json "subagent"', task_id),
            )
            db.commit()
        assert [h["id"] for h in _load_history()] == [task_id]
