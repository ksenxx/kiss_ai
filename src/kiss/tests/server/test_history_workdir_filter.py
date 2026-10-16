# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""The history panel's WORKDIR filter.

A ``getHistory`` command may carry ``workDir``; the daemon then returns
only the tasks whose ``task_history.work_dir`` equals it, so a project's
whole history is listed beyond the first page.  The page-one ``history``
reply also carries ``workDirs``, the options the dropdown in
``chat.html`` offers: the same directories as the chat header's
working-directory drop-down (the current working directory, then the
directories adopted before, newest first, those that still exist).
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time

import kiss.agents.sorcar.persistence as th
import kiss.core.user_config as vc
from kiss.server import web_server
from kiss.tests.server.conftest import (
    make_capturing_server,
    redirect_persistence,
    restore_persistence,
)

from .test_history_chat_first_task import _history_sessions
from .test_recent_work_dirs import _IsolatedConfig


def _task(task: str, work_dir: str) -> str:
    """Add *task* run in *work_dir* and finish it."""
    task_id, _ = th._add_task(task, extra={"work_dir": work_dir})
    th._save_task_result(result="done", task_id=task_id)
    th._save_task_extra({"endTs": 1_000}, task_id=task_id)
    return task_id


def _ids(rows: list[th._HistoryEntry]) -> list[str]:
    return [str(r["id"]) for r in rows]


class TestWorkDirFilterQueries:
    """``_load_history`` / ``_search_history`` filter by work_dir in SQL."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = redirect_persistence(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_history_keeps_only_rows_of_the_directory(self) -> None:
        kiss_a = _task("Fix the parser", "/home/u/kiss")
        paper = _task("Write the paper", "/home/u/paper")
        kiss_b = _task("Add a test", "/home/u/kiss")
        nowhere, _ = th._add_task("No directory at all")

        assert _ids(th._load_history()) == [nowhere, kiss_b, paper, kiss_a]
        assert _ids(th._load_history(work_dir="/home/u/kiss")) == [kiss_b, kiss_a]
        assert _ids(th._load_history(work_dir="/home/u/paper")) == [paper]
        # The match is exact: neither a prefix nor a LIKE pattern.
        assert th._load_history(work_dir="/home/u") == []
        assert th._load_history(work_dir="/home/u/%") == []
        assert th._load_history(work_dir="/home/u/kiss/") == []
        # Limit and offset page through the filtered rows.
        assert _ids(th._load_history(limit=1, work_dir="/home/u/kiss")) == [kiss_b]
        assert _ids(th._load_history(limit=1, offset=1, work_dir="/home/u/kiss")) == [kiss_a]

    def test_search_history_combines_text_tag_and_work_dir(self) -> None:
        kiss_parser = _task("Fix the parser bug", "/home/u/kiss")
        _task("Fix the parser bug", "/home/u/other")
        kiss_paper = _task("Write the paper on parsers", "/home/u/kiss")
        assert _ids(th._search_history("parser", work_dir="/home/u/kiss")) == [
            kiss_paper,
            kiss_parser,
        ]
        assert _ids(th._search_history("bug", work_dir="/home/u/kiss")) == [kiss_parser]
        assert len(th._search_history("parser")) == 3
        assert _ids(th._search_history("", tag="debugging", work_dir="/home/u/kiss")) == [
            kiss_parser,
        ]
        assert th._search_history("paper", tag="paper", work_dir="/home/u/other") == []



class TestGetHistoryCommand:
    """The ``getHistory`` command threads ``workDir`` down to the query
    and the page-one reply lists the chat header's directories."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = redirect_persistence(self.tmpdir)
        self.cfg = _IsolatedConfig()
        self.cfg.enter(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        self.cfg.exit()
        restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_command_filters_and_lists_work_dirs(self) -> None:
        kiss = os.path.join(self.tmpdir, "kiss")
        paper = os.path.join(self.tmpdir, "paper")
        gone = os.path.join(self.tmpdir, "gone")
        worktree = os.path.join(self.tmpdir, "wt")
        for d in (kiss, paper, gone, worktree):
            os.mkdir(d)
        _task("Fix the parser", kiss)
        _task("Write the paper", paper)
        _task("Add a test", kiss)
        _task("Sub-agent in a worktree", worktree)
        # The directories chosen as the working directory so far, as the
        # chat header's drop-down lists them: paper was chosen last.
        vc.record_recent_work_dir(gone)
        time.sleep(0.01)
        vc.record_recent_work_dir(kiss)
        time.sleep(0.01)
        vc.record_recent_work_dir(paper)
        os.rmdir(gone)
        server, events = make_capturing_server()
        server.work_dir = paper
        server._handle_command(
            {"type": "getHistory", "workDir": paper, "generation": 5},
        )
        (session,) = _history_sessions(events)
        assert session["title"] == "Write the paper"
        assert session["work_dir"] == paper
        hist = [e for e in events if e.get("type") == "history"][0]
        assert hist["offset"] == 0 and hist["generation"] == 5
        # The options are the header drop-down's: the directories
        # chosen before, newest first, those that still exist.  The
        # sub-agent's worktree, which the history merely ran in, is not
        # offered.
        assert hist["workDirs"] == [paper, kiss]
        # A trailing separator on the current directory is the same
        # directory (the header's samePath): no duplicate entry.
        events.clear()
        server.work_dir = paper + os.sep
        server._handle_command({"type": "getHistory"})
        hist = [e for e in events if e.get("type") == "history"][0]
        assert hist["workDirs"] == [paper, kiss]
        # The current working directory leads the list even when it was
        # never recorded (or no longer exists): it is the one in use.
        events.clear()
        server.work_dir = os.path.join(self.tmpdir, "fresh")
        server._handle_command({"type": "getHistory"})
        hist = [e for e in events if e.get("type") == "history"][0]
        assert hist["workDirs"] == [server.work_dir, paper, kiss]
        # A later page of the same view carries no options: the
        # dropdown keeps the ones page one brought.
        events.clear()
        server._handle_command({"type": "getHistory", "offset": 50})
        hist = [e for e in events if e.get("type") == "history"][0]
        assert hist["offset"] == 50 and "workDirs" not in hist

    def test_command_without_work_dir_or_with_bad_value_returns_everything(self) -> None:
        _task("Fix the parser", "/home/u/kiss")
        _task("Write the paper", "/home/u/paper")
        server, events = make_capturing_server()
        server._handle_command({"type": "getHistory"})
        assert len(_history_sessions(events)) == 2
        events.clear()
        server._handle_command({"type": "getHistory", "workDir": 7})
        assert len(_history_sessions(events)) == 2

    def test_command_combines_query_tag_and_work_dir(self) -> None:
        _task("Fix the parser bug", "/home/u/kiss")
        _task("Fix the parser bug", "/home/u/other")
        _task("Write the paper on parsers", "/home/u/kiss")
        server, events = make_capturing_server()
        server._handle_command(
            {
                "type": "getHistory",
                "query": "parser",
                "tag": "debugging",
                "workDir": "/home/u/kiss",
            },
        )
        (session,) = _history_sessions(events)
        assert session["title"] == "Fix the parser bug"
        assert session["work_dir"] == "/home/u/kiss"

    def test_remote_machine_request_with_work_dir_stays_local(self) -> None:
        # A directory is a path on this machine: with one chosen, the
        # page holds this daemon's rows even when a remote machine is
        # selected (no remote fetch, no empty page).
        _task("Fix the parser", "/home/u/kiss")
        server, events = make_capturing_server()
        server._handle_command(
            {"type": "getHistory", "machine": "box", "workDir": "/home/u/kiss"},
        )
        (session,) = _history_sessions(events)
        assert session["title"] == "Fix the parser"


class TestDropdownMarkup:
    """``chat.html`` holds the WORKDIR dropdown with only the All option:
    the daemon's reply fills in the directories."""

    def test_chat_html_has_the_workdir_select(self) -> None:
        html = (web_server.MEDIA_DIR / "chat.html").read_text(encoding="utf-8")
        select = re.search(r'<select id="hf-workdir".*?</select>', html, re.S)
        assert select is not None
        assert re.findall(r'<option value="([^"]*)"', select.group(0)) == [""]
        assert re.search(r'<label for="hf-workdir"[^>]*>WORKDIR</label>', html)
