"""End-to-end tests for the ``tags`` / ``sea`` columns, ``chat_summaries`` and their backfill.

A finished task (``_save_task_extra`` with ``endTs``) gets its ``tags``
classified and its chat's ``chat_summaries`` row refreshed; a run's SEA
is persisted in ``sea``; ``_get_history`` forwards all of it; and
:func:`kiss.agents.sorcar.task_metadata.backfill_task_metadata` fills an
older database, inferring sub-agent SEAs from the parent trajectory's
``run_agent`` tool calls.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.task_metadata import (
    backfill_task_metadata,
    classify_task_tags,
    infer_subagent_seas,
    sea_name_of_agent,
    summarize_chat,
)

from .test_history_chat_first_task import _history_sessions, _make_server
from .test_history_task_meta_server import _redirect, _restore


class TestClassifyTaskTags:
    """The rule-based classifier."""

    def test_work_coding_task(self) -> None:
        tags = classify_task_tags("Can you fix the failing tests in persistence.py?")
        assert tags[0] == "work"
        assert "coding" in tags and "testing" in tags and "debugging" in tags
        assert "question" not in tags  # "Can you ..." is a request, not a question

    def test_personal_shopping_task(self) -> None:
        tags = classify_task_tags("Buy a birthday gift for my wife on amazon under $50")
        assert tags[0] == "personal"
        assert "shopping" in tags

    def test_secret_and_chore_and_question(self) -> None:
        assert "secret" in classify_task_tags("Rotate the API key in .env")
        assert "chore" in classify_task_tags("git pull, merge main and push")
        assert "question" in classify_task_tags("What have the task abc done so far?")

    def test_subagent_and_failed_markers_and_cap(self) -> None:
        tags = classify_task_tags(
            "Research, review, test, debug, implement, deploy, write the paper and the docs",
            is_subagent=True,
            failed=True,
        )
        assert len(tags) <= 6
        assert tags[0] == "work"
        long_text = "x" * 5000 + " buy groceries"
        assert classify_task_tags(long_text)[0] == "work"  # intent beyond the head is ignored

    def test_boilerplate_beyond_head_does_not_tag(self) -> None:
        task = "Rename the variable.\n" + "\n" * 5 + "filler " * 400 + "Thoroughly check whether"
        assert "review" not in classify_task_tags(task)


class TestSummarizeChat:
    """The 6-8 word chat summary."""

    def test_strips_filler_and_trims_trailing_stopwords(self) -> None:
        summary = summarize_chat(
            ["can you modify ~/.kiss/sorcar.db in a backward compatible way in the following way:"]
        )
        assert summary == "modify ~/.kiss/sorcar.db in a backward compatible way"
        assert 6 <= len(summary.split()) <= 8

    def test_short_first_sentence_borrows_from_later_sentences_and_tasks(self) -> None:
        assert summarize_chat(["hi", "add a column to the table and fix the tests"]) == (
            "add a column to the table and fix"
        )
        assert summarize_chat(["run tests", "then commit the changes please"]) == (
            "run tests then commit the changes please"
        )
        assert summarize_chat(["Write the paper introduction. Use LaTeX macros; cite X."]) == (
            "Write the paper introduction Use LaTeX macros"
        )
        # A first sentence of six or more words is the whole summary.
        assert summarize_chat(["Write the paper introduction for me now.", "Then more"]) == (
            "Write the paper introduction for me now"
        )

    def test_slash_command_and_urls_and_empty(self) -> None:
        assert summarize_chat(["/review_paper ./papers/x.pdf"]) == "review paper ./papers/x.pdf"
        assert summarize_chat(["Please give likes to <https://www.linkedin.com/feed/x?y=1>"]) == (
            "give likes to www.linkedin.com"
        )
        assert summarize_chat(["hi", "thanks"]) == ""
        assert summarize_chat([]) == ""
        long_word = "a" * 60
        assert summarize_chat([f"open {long_word} now"]).split()[1].endswith("…")


class TestSeaNameOfAgent:
    """The ``run_agent`` argument to SEA name mapping used by the backfill."""

    def test_mapping(self) -> None:
        channels = ["slack", "homeassistant"]
        assert sea_name_of_agent("/x/seas/write_paper/write_paper_sea.py", channels) == (
            "write_paper_sea"
        )
        assert sea_name_of_agent("", channels) == "dummy_sea"
        assert sea_name_of_agent("Cron", channels) == "cron_agent"
        assert sea_name_of_agent("Home Assistant", channels) == "homeassistant_sea"
        assert sea_name_of_agent("slack", channels) == "slack_sea"
        assert sea_name_of_agent("nonesuch", channels) == ""


class TestFinishHookAndHistory:
    """Live path: finishing a task fills ``tags``/``sea``/``chat_summaries``."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_finish_populates_tags_and_chat_summary(self) -> None:
        task_id, chat_id = th._add_task(
            "Can you fix the failing tests in persistence.py?",
            extra={"sea": "autorouter_sea", "startTs": 1_000_000},
        )
        db = th._get_db()
        row = db.execute("SELECT tags, sea FROM task_history WHERE id = ?", (task_id,)).fetchone()
        assert row["tags"] == "" and row["sea"] == "autorouter_sea"
        assert th._chat_summaries([chat_id]) == {}
        th._save_task_result(result="Task failed: boom", task_id=task_id)
        th._save_task_extra({"endTs": 2_000_000}, task_id=task_id)
        row = db.execute("SELECT tags, sea FROM task_history WHERE id = ?", (task_id,)).fetchone()
        tags = row["tags"].split(",")
        assert tags[0] == "work" and "failed" in tags and "testing" in tags
        assert row["sea"] == "autorouter_sea"
        summaries = th._chat_summaries([chat_id, ""])
        assert summaries[chat_id]["summary"] == "fix the failing tests in persistence.py"
        assert summaries[chat_id]["last_launched"] == 1_000_000

    def test_subagent_finish_does_not_touch_chat_summary_and_extra_without_end_is_inert(
        self,
    ) -> None:
        parent_id, chat_id = th._add_task("top-level task text", extra={"startTs": 5_000})
        child_id, _ = th._add_task(
            "child work", chat_id=chat_id, extra={"subagent": {"parent_task_id": parent_id}}
        )
        th._save_task_extra({"tokens": 5}, task_id=parent_id)
        assert th._chat_summaries([chat_id]) == {}
        th._save_task_extra({"endTs": 6_000}, task_id=child_id)
        db = th._get_db()
        child = db.execute("SELECT tags FROM task_history WHERE id = ?", (child_id,)).fetchone()
        assert "subagent" in child["tags"].split(",")
        assert th._chat_summaries([chat_id]) == {}
        # A second listable task moves last_launched forward, the summary
        # still describes the chat from its first task.
        second_id, _ = th._add_task("second task", chat_id=chat_id, extra={"startTs": 9_000})
        th._save_task_extra({"endTs": 9_500}, task_id=second_id)
        summary = th._chat_summaries([chat_id])[chat_id]
        assert summary["last_launched"] == 9_000
        assert str(summary["summary"]).startswith("top-level task text")

    def test_orphan_recovery_finalizes_metadata(self) -> None:
        """A task killed mid-run is tagged and summarised when the sweep recovers it."""
        task_id, chat_id = th._add_task("fix the parser bug", extra={"startTs": 7_000})
        db = th._get_db()
        # A row left behind by a dead process: its owner marker is gone,
        # so the boot-time sweep rewrites the surviving sentinel.
        db.execute("UPDATE task_history SET owner = '' WHERE id = ?", (task_id,))
        db.commit()
        assert th._recover_orphaned_tasks(set()) == 1
        row = db.execute(
            "SELECT result, tags FROM task_history WHERE id = ?", (task_id,)
        ).fetchone()
        assert row["result"] == "Task terminated unexpectedly (process killed)"
        tags = row["tags"].split(",")
        assert tags[0] == "work" and "failed" in tags and "debugging" in tags
        assert th._chat_summaries([chat_id])[chat_id] == {
            "summary": "fix the parser bug",
            "last_launched": 7_000,
        }

    def test_history_payload_carries_tags_sea_and_chat_summary(self) -> None:
        task_id, chat_id = th._add_task(
            "Write the paper introduction", extra={"sea": "write_paper_sea", "startTs": 42_000}
        )
        th._save_task_result(result="done", task_id=task_id)
        th._save_task_extra({"endTs": 43_000}, task_id=task_id)
        server, events = _make_server()
        server._get_history(query=None)
        (session,) = _history_sessions(events)
        assert session["sea"] == "write_paper_sea"
        assert session["tags"].split(",")[0] == "work"
        assert "paper" in session["tags"].split(",")
        assert session["chat_summary"] == "Write the paper introduction"
        assert session["chat_last_launched"] == 42_000
        # The legacy ``extra`` JSON the history rows synthesise carries
        # both columns too.
        (entry,) = th._load_history(limit=1)
        extra = json.loads(str(entry["extra"]))
        assert extra["tags"] == session["tags"] and extra["sea"] == "write_paper_sea"


def _legacy_db(path: Path) -> None:
    """Create a database in the pre-``tags``/``sea`` schema with sample rows."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE task_history (
            id TEXT PRIMARY KEY, timestamp REAL NOT NULL, task TEXT NOT NULL,
            has_events INTEGER DEFAULT 0, result TEXT DEFAULT '', chat_id CHAR(32) DEFAULT '',
            model TEXT DEFAULT '', work_dir TEXT DEFAULT '', version TEXT DEFAULT '',
            tokens INTEGER DEFAULT 0, cost REAL DEFAULT 0.0, steps INTEGER DEFAULT 0,
            is_parallel INTEGER DEFAULT 1, is_worktree INTEGER DEFAULT 1,
            auto_commit_mode INTEGER DEFAULT 1, start_ts INTEGER DEFAULT 0,
            end_ts INTEGER DEFAULT 0, is_favorite INTEGER DEFAULT 0,
            parent_task_id TEXT DEFAULT '', max_budget REAL DEFAULT 0.0,
            owner TEXT DEFAULT '', is_side_channel INTEGER DEFAULT 0
        );
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, seq INTEGER NOT NULL,
            event_json TEXT NOT NULL, timestamp REAL NOT NULL
        );
        """
    )
    rows = [
        ("p1", 100.0, "Please write the paper on routing.\nMore details.", "ok", "chatA", "", 0),
        ("c1", 101.0, "PREAMBLE: draft section 2 of the paper", "ok", "chatA", "p1", 0),
        ("c2", 102.0, "review the draft", "Task failed: x", "chatA", "p1", 0),
        ("c3", 103.0, "parallel worker", "ok", "chatA", "p1", 0),
        (
            "c4",
            104.0,
            "review /home/u/repo/.kiss-worktrees/kiss_wt-1/papers/a.tex carefully",
            "ok",
            "chatA",
            "p1",
            0,
        ),
        ("c5", 105.0, "same prompt", "ok", "chatA", "p1", 0),
        ("c6", 106.0, "same prompt", "ok", "chatA", "p1", 0),
        ("q1", 200.0, "hi", "ok", "chatB", "", 250_000),
        ("q2", 300.0, "buy groceries for dinner", "ok", "chatB", "", 0),
    ]
    for tid, ts, task, result, chat, parent, start in rows:
        conn.execute(
            "INSERT INTO task_history (id, timestamp, task, result, chat_id, parent_task_id, "
            "start_ts) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (tid, ts, task, result, chat, parent, start),
        )
    calls = [
        {
            "type": "tool_call",
            "name": "run_agent",
            "callId": 1,
            "extras": {
                "agent": "/tmp/seas/write_paper/write_paper_sea.py",
                "task": "draft section 2 of the paper",
            },
        },
        {
            "type": "tool_call",
            "name": "run_agent",
            "callId": 2,
            "extras": {"agent": "cron", "task": "review the draft"},
        },
        {"type": "tool_call", "name": "Bash", "callId": 3, "extras": {"command": "run_agent"}},
        {
            "type": "tool_call",
            "name": "run_agent",
            "callId": 4,
            "extras": {"agent": "nonesuch", "task": "parallel worker"},
        },
        {
            "type": "tool_call",
            "name": "run_agent",
            "callId": 5,
            "extras": {"agent": "", "task": ""},
        },
        # Dispatched with the parent repo's absolute path, which
        # agent_dispatch rewrote to the worktree's before the child ran.
        {
            "type": "tool_call",
            "name": "run_agent",
            "callId": 6,
            "extras": {
                "agent": "/x/review_paper/review_paper_sea.py",
                "task": "review /home/u/repo/papers/a.tex carefully",
            },
        },
        # The same prompt dispatched twice, to two different SEAs: the
        # children pair up in order even though the first is labelled.
        {
            "type": "tool_call",
            "name": "run_agent",
            "callId": 7,
            "extras": {"agent": "/x/first_sea.py", "task": "same prompt"},
        },
        {
            "type": "tool_call",
            "name": "run_agent",
            "callId": 8,
            "extras": {"agent": "/x/second_sea.py", "task": "same prompt"},
        },
    ]
    for seq, call in enumerate(calls):
        conn.execute(
            "INSERT INTO events (task_id, seq, event_json, timestamp) VALUES (?, ?, ?, ?)",
            ("p1", seq, json.dumps(call), 100.0 + seq),
        )
    conn.execute(
        "INSERT INTO events (task_id, seq, event_json, timestamp) VALUES (?, ?, ?, ?)",
        ("p1", 99, "not json run_agent", 199.0),
    )
    conn.commit()
    conn.close()


def _cli(db_path: Path, *flags: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "kiss.scripts.backfill_task_metadata", "--db", str(db_path), *flags],
        capture_output=True,
        text=True,
        timeout=120,
    )


class TestBackfill:
    """``backfill_task_metadata`` and its CLI on a legacy database."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "sorcar.db"
        _legacy_db(self.db_path)

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_migration_and_backfill(self) -> None:
        conn = sqlite3.connect(self.db_path)
        th._init_tables(conn)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(task_history)")}
        assert {"tags", "sea"} <= cols
        assert infer_subagent_seas(conn, "nobody") == {}
        # A child labelled before the backfill (a live run) still takes
        # part in the matching, so its twin pairs with the second call.
        conn.execute("UPDATE task_history SET sea = 'first_sea' WHERE id = 'c5'")
        conn.commit()
        assert infer_subagent_seas(conn, "p1")["c6"] == "second_sea"
        assert "c5" not in infer_subagent_seas(conn, "p1")
        counts = backfill_task_metadata(conn)
        assert counts == {"tags": 9, "sea": 4, "chats": 2}
        rows = {r["id"]: r for r in conn.execute("SELECT id, tags, sea FROM task_history")}
        assert rows["c1"]["sea"] == "write_paper_sea"
        assert rows["c2"]["sea"] == "cron_agent"
        assert rows["c3"]["sea"] == ""
        assert rows["c4"]["sea"] == "review_paper_sea"  # matched despite the path rewrite
        assert rows["c5"]["sea"] == "first_sea" and rows["c6"]["sea"] == "second_sea"
        assert rows["p1"]["sea"] == ""
        assert "paper" in rows["p1"]["tags"].split(",")
        assert "failed" in rows["c2"]["tags"] and "subagent" in rows["c2"]["tags"]
        assert rows["q2"]["tags"].split(",")[0] == "personal"
        summaries = {
            r[0]: (r[1], r[2])
            for r in conn.execute("SELECT chat_id, summary, last_launched FROM chat_summaries")
        }
        assert summaries["chatA"] == ("write the paper on routing More details", 100_000)
        assert summaries["chatB"] == ("buy groceries for dinner", 300_000)
        # Idempotent: a second run finds nothing to fill.
        assert backfill_task_metadata(conn) == {"tags": 0, "sea": 0, "chats": 0}
        conn.close()

    def test_cli(self) -> None:
        started = time.monotonic()
        proc = _cli(self.db_path)
        assert proc.returncode == 0, proc.stderr
        assert "tagged 9 tasks" in proc.stdout
        assert "inferred the SEA of 5 sub-agent tasks" in proc.stdout
        assert "summarised 2 chats" in proc.stdout
        assert time.monotonic() - started < 120
        # Nothing left to fill; --refresh recomputes tags and summaries
        # but keeps the inferred SEAs.
        assert "tagged 0 tasks" in _cli(self.db_path).stdout
        refreshed = _cli(self.db_path, "--refresh").stdout
        assert "tagged 9 tasks" in refreshed and "summarised 2 chats" in refreshed
        assert "inferred the SEA of 0 sub-agent tasks" in refreshed
        missing = _cli(Path(self.tmpdir) / "nope" / "x.db")
        assert missing.returncode == 1
        assert "cannot open" in missing.stderr
