# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""The history panel's tag filter.

A ``getHistory`` command may carry ``tag``; the daemon then returns only
the tasks whose ``task_history.tags`` column contains that tag, so a
rare tag is found beyond the first page.  The dropdown in ``chat.html``
offers exactly the tags :func:`classify_task_tags` can emit.
"""

from __future__ import annotations

import re
import shutil
import tempfile

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.task_metadata import ALL_TAGS, classify_task_tags
from kiss.server import web_server

from .test_history_chat_first_task import _history_sessions, _make_server
from .test_history_task_meta_server import _redirect, _restore


def _finished(task: str, result: str = "done") -> str:
    """Add *task* and finish it the way a run does: result, then ``endTs``
    (which is what derives and stores the row's ``tags``)."""
    task_id, _ = th._add_task(task)
    th._save_task_result(result=result, task_id=task_id)
    th._save_task_extra({"endTs": 1_000}, task_id=task_id)
    return task_id


class TestTagFilterQueries:
    """``_load_history`` / ``_search_history`` filter by tag in SQL."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_tag_filter_sql_accepts_only_lowercase_words(self) -> None:
        assert th._tag_filter_sql("") == ("", ())
        assert th._tag_filter_sql("Work") == ("", ())
        assert th._tag_filter_sql("%") == ("", ())
        assert th._tag_filter_sql("a_b") == ("", ())
        sql, params = th._tag_filter_sql("coding")
        assert sql.startswith("AND ") and params == ("%,coding,%",)

    def test_load_history_keeps_only_tagged_rows(self) -> None:
        paper = _finished("Write the paper introduction")
        shop = _finished("Buy groceries for the week")
        failed = _finished("Fix the parser bug", result="Task failed: boom")
        running, _ = th._add_task("Still running, no tags yet")
        assert "paper" in classify_task_tags("Write the paper introduction")

        ids = lambda rows: [str(r["id"]) for r in rows]  # noqa: E731
        assert ids(th._load_history()) == [running, failed, shop, paper]
        assert ids(th._load_history(tag="paper")) == [paper]
        assert ids(th._load_history(tag="personal")) == [shop]
        assert ids(th._load_history(tag="failed")) == [failed]
        # Every finished row is "work" or "personal"; the running row
        # has no tags yet and never matches a tag.
        assert ids(th._load_history(tag="work")) == [failed, paper]
        assert ids(th._load_history(tag="scheduling")) == []
        # A tag that is not a plain word is ignored, not matched.
        assert ids(th._load_history(tag="pa%")) == [running, failed, shop, paper]
        # Limit and offset page through the filtered rows.
        assert ids(th._load_history(limit=1, tag="work")) == [failed]
        assert ids(th._load_history(limit=1, offset=1, tag="work")) == [paper]

    def test_tag_matches_whole_tag_not_substring(self) -> None:
        # "docs" must not match "coding"'s "co" nor "data" match
        # anything but the "data" tag itself.
        docs = _finished("Update the README documentation")
        _finished("Refactor the function to use a class")
        rows = th._load_history(tag="docs")
        assert [str(r["id"]) for r in rows] == [docs]
        assert th._load_history(tag="doc") == []

    def test_search_history_combines_text_and_tag(self) -> None:
        paper = _finished("Write the paper on parsers")
        _finished("Fix the parser bug")
        _finished("Write the paper on lexers")
        both = th._search_history("parser", tag="paper")
        assert [str(r["id"]) for r in both] == [paper]
        assert len(th._search_history("parser")) == 2
        assert len(th._search_history("paper", tag="debugging")) == 0
        # An empty query with a tag goes through the plain load.
        assert len(th._search_history("", tag="paper")) == 2

    def test_subagent_rows_stay_out_of_the_listing(self) -> None:
        parent = _finished("Review the parser change with a sub-agent")
        child_id, _ = th._add_task(
            "Review the change", extra={"subagent": {"parent_task_id": parent}}
        )
        th._save_task_result(result="done", task_id=child_id)
        th._save_task_extra({"endTs": 1_000}, task_id=child_id)
        assert th._load_history(tag="subagent") == []
        assert [str(r["id"]) for r in th._load_history(tag="review")] == [parent]


class TestGetHistoryCommand:
    """The ``getHistory`` command threads ``tag`` down to the query."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_command_filters_and_echoes_offset(self) -> None:
        _finished("Write the paper introduction")
        _finished("Fix the parser bug")
        server, events = _make_server()
        server._handle_command({"type": "getHistory", "tag": "paper", "generation": 3})
        (session,) = _history_sessions(events)
        assert session["title"] == "Write the paper introduction"
        assert "paper" in session["tags"].split(",")
        hist = [e for e in events if e.get("type") == "history"][0]
        assert hist["offset"] == 0 and hist["generation"] == 3

    def test_command_without_tag_or_with_bad_tag_returns_everything(self) -> None:
        _finished("Write the paper introduction")
        _finished("Fix the parser bug")
        server, events = _make_server()
        server._handle_command({"type": "getHistory"})
        assert len(_history_sessions(events)) == 2
        events.clear()
        server._handle_command({"type": "getHistory", "tag": 7})
        assert len(_history_sessions(events)) == 2

    def test_command_combines_query_and_tag(self) -> None:
        _finished("Write the paper on parsers")
        _finished("Fix the parser bug")
        server, events = _make_server()
        server._handle_command({"type": "getHistory", "query": "parser", "tag": "debugging"})
        (session,) = _history_sessions(events)
        assert session["title"] == "Fix the parser bug"


class TestDropdownVocabulary:
    """The ``#hf-tag`` options are exactly the classifier's vocabulary."""

    def test_all_tags_is_the_22_tag_vocabulary(self) -> None:
        # 2 scope + 3 intent + 15 activity + 2 run-status tags.
        assert len(ALL_TAGS) == 22 and len(set(ALL_TAGS)) == 22
        assert ALL_TAGS[:5] == ("work", "personal", "secret", "chore", "question")
        assert ALL_TAGS[-2:] == ("subagent", "failed")
        emitted = set(
            classify_task_tags(
                "Buy the groceries, then research and debug the failing tests",
                is_subagent=True,
                failed=True,
            )
        )
        assert emitted <= set(ALL_TAGS)

    def test_chat_html_options_are_the_listable_tags(self) -> None:
        # Sub-agent rows are never listed in the history panel
        # (``_HISTORY_NOT_SUBAGENT``), so the one tag only they carry
        # would always yield an empty list and is not offered.
        html = (web_server.MEDIA_DIR / "chat.html").read_text(encoding="utf-8")
        select = re.search(r'<select id="hf-tag".*?</select>', html, re.S)
        assert select is not None
        options = re.findall(r'<option value="([^"]*)"', select.group(0))
        assert options == ["", *(t for t in ALL_TAGS if t != "subagent")]
