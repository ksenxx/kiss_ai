# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The history marks the tasks the cron scheduler launched (``is_cron``).

The CHATS panel's Cron chip (off by default) hides those rows, so the
daemon must tell them apart: a persisted ``is_cron`` column, stamped
from the ``cronJob`` wire flag an in-chat job sends, or from the
unattended task text every prompt job (and every sub-task of one)
carries, backfilled once for the rows older releases wrote, and
reported as ``is_cron`` on every ``history`` session.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import cron_agent, local_endpoint
from kiss.agents.sorcar import persistence as th
from kiss.core.kiss_agent import KISSAgent
from kiss.server.server import AgentServer
from kiss.tests.server.test_append_basic_tools import DaemonRunApiHarness


class TestHistoryCronFlag(unittest.TestCase):
    """``getHistory`` reports ``is_cron`` from the persisted column."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self._orig_db_path = th._DB_PATH
        th._close_db()
        th._DB_PATH = Path(self._tmp) / "history.db"
        self.server = AgentServer()
        self.server.work_dir = self._tmp
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        orig = self.server.printer.broadcast

        def capture(e: dict[str, Any]) -> None:
            with self._lock:
                self.events.append(dict(e))
            orig(e)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

    def tearDown(self) -> None:
        th._close_db()
        th._DB_PATH = self._orig_db_path
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _sessions(self) -> dict[str, dict[str, Any]]:
        self.server._handle_command({"type": "getHistory"})
        with self._lock:
            for e in reversed(self.events):
                if e["type"] == "history":
                    return {s["preview"]: s for s in e["sessions"]}
        return {}

    def test_persisted_flag_reaches_the_history_rows(self) -> None:
        cron_id, _ = th._add_task("cron task", extra={"is_cron": True})
        th._save_task_result(result="done", task_id=cron_id)
        plain_id, _ = th._add_task("plain task", extra={"is_cron": False})
        th._save_task_result(result="done", task_id=plain_id)
        bare_id, _ = th._add_task("bare task")
        th._save_task_result(result="done", task_id=bare_id)
        # The final save of a run (``_save_task_extra``) keeps the flag
        # through its typed-column map.
        th._save_task_extra({"is_cron": True, "steps": 3}, task_id=bare_id)

        by_task = self._sessions()
        self.assertTrue(by_task["cron task"]["is_cron"])
        self.assertFalse(by_task["plain task"]["is_cron"])
        self.assertTrue(by_task["bare task"]["is_cron"])
        self.assertEqual(by_task["bare task"]["steps"], 3)

    def test_rows_of_older_releases_are_backfilled_once(self) -> None:
        """A database without the column gets it, and the rows a cron
        launch left recognisable traces on are stamped: a task text
        that starts with the prompt-job preamble, or a run directory
        under ``cron/runs``.  Anything else stays a plain row."""
        th._close_db()
        th._get_db()  # create the schema, then recreate the table without the column
        th._close_db()
        conn = sqlite3.connect(str(th._DB_PATH))
        # Not ``ALTER TABLE ... DROP COLUMN``: SQLite rewrites the stored
        # DDL textually, and with the comment lines that precede the
        # column some versions (3.46) leave a dangling comma behind.  The
        # column is dropped wherever it sits: later releases append
        # columns after it.
        (ddl,) = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'task_history'"
        ).fetchone()
        legacy_ddl = re.sub(
            r",\s*(?:--[^\n]*\n\s*)*is_cron INTEGER DEFAULT 0(?=\s*[,)])", "", ddl,
        )
        self.assertNotIn("is_cron", legacy_ddl)
        conn.executescript(f"DROP TABLE task_history; {legacy_ddl};")
        rows = [
            ("preamble", cron_agent.PROMPT_PREAMBLE + "check the mail", "/home/u/proj"),
            ("rundir", "/good_morning now", "/home/u/.kiss/cron/runs/good-abc"),
            ("quoted", "Explain why '" + cron_agent.PROMPT_PREAMBLE.strip() + "' is used", "/x"),
            ("plain", "a plain task", "/home/u/proj"),
        ]
        for tid, task, work_dir in rows:
            conn.execute(
                "INSERT INTO task_history (id, timestamp, task, work_dir, result) "
                "VALUES (?, 1.0, ?, ?, 'done')",
                (tid, task, work_dir),
            )
        # Sub-tasks of a cron run carry no trace of their own: they are
        # stamped through the parent chain, however deep; a sub-task of
        # a plain run is not.
        for tid, parent, task in [
            ("child", "preamble", "child of the prompt job"),
            ("grandchild", "child", "grandchild of the prompt job"),
            ("plainchild", "plain", "child of the plain task"),
        ]:
            conn.execute(
                "INSERT INTO task_history (id, timestamp, task, work_dir, result, "
                "parent_task_id) VALUES (?, 1.0, ?, '/project', 'done', ?)",
                (tid, task, parent),
            )
        conn.commit()
        conn.close()

        by_task = self._sessions()
        self.assertTrue(by_task[rows[0][1]]["is_cron"])
        self.assertTrue(by_task["/good_morning now"]["is_cron"])
        self.assertFalse(by_task[rows[2][1]]["is_cron"])
        self.assertFalse(by_task["a plain task"]["is_cron"])
        # Sub-task rows are not history sessions: read their column.
        conn = sqlite3.connect(str(th._DB_PATH))
        try:
            flags = dict(conn.execute(
                "SELECT id, is_cron FROM task_history WHERE parent_task_id != ''"
            ))
        finally:
            conn.close()
        self.assertEqual(flags, {"child": 1, "grandchild": 1, "plainchild": 0})
        # Re-opening the database leaves the stamps alone (no second
        # backfill, no column error).
        th._close_db()
        by_task = self._sessions()
        self.assertTrue(by_task["/good_morning now"]["is_cron"])
        self.assertFalse(by_task["a plain task"]["is_cron"])


class DaemonStampsCronRuns(DaemonRunApiHarness):
    """A real daemon stamps ``is_cron`` on the rows of cron launches."""

    def _run_raw(self, extra_cmd: dict[str, Any]) -> None:
        """Send one raw ``run`` command and wait for its end."""
        tab_id = f"raw-{uuid.uuid4().hex}"
        ws = local_endpoint.connect(Path(self.endpoint_file), open_timeout=60)
        try:
            ws.send(json.dumps({
                "type": "run",
                "prompt": "raw task",
                "tabId": tab_id,
                "taskId": uuid.uuid4().hex,
                "workDir": self.repo,
                "model": "",
                "useWebTools": False,
                "classifyTasks": False,
                **extra_cmd,
            }))
            started = False
            while True:
                event = json.loads(ws.recv(timeout=60))
                if event.get("tabId") != tab_id or event.get("type") != "status":
                    continue
                if event.get("running"):
                    started = True
                elif started:
                    return
        finally:
            ws.close()

    def _is_cron_by_task(self) -> dict[str, int]:
        conn = sqlite3.connect(str(th._DB_PATH))
        try:
            return {
                task: is_cron
                for task, is_cron in conn.execute(
                    "SELECT task, is_cron FROM task_history"
                )
            }
        finally:
            conn.close()

    def test_wire_flag_and_unattended_text_mark_the_row(self) -> None:
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        assert KISSAgent.run is not self._original_executor_run
        self._run_raw({"prompt": "in-chat job", "cronJob": True})
        self._run_raw({"prompt": cron_agent.PROMPT_PREAMBLE + "prompt job"})
        self._run_raw({
            "prompt": "child of a prompt job",
            "appendToPrompt": cron_agent.unattended_child_suffix(""),
        })
        self._run_raw({"prompt": "a user's own task", "cronJob": False})
        self._run_raw({"prompt": "a task without the field"})
        flags = self._is_cron_by_task()
        self.assertEqual(
            {k: v for k, v in flags.items() if k in (
                "in-chat job", "a user's own task", "a task without the field",
            )},
            {"in-chat job": 1, "a user's own task": 0, "a task without the field": 0},
        )
        self.assertEqual(
            [v for k, v in flags.items() if k.endswith("prompt job")], [1],
        )
        self.assertEqual(
            [v for k, v in flags.items() if k.startswith("child of a prompt job")],
            [1],
        )

    def test_a_sea_that_replaces_the_prompt_keeps_the_mark(self) -> None:
        """A scheduled slash command runs an SEA that may drop the task
        text (``/recommendations now`` does): the mark is decided on the
        text as submitted, so the row is still a cron row."""
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        sea_path = self._write_py(
            "replaces_prompt_sea.py",
            '''
            """SEA that ignores the task text."""

            from kiss.agents.seas.base.base_sea import BaseSea


            class Sea(BaseSea):
                def prompt(self, task: str) -> str:
                    """Replace the task, keeping its last word to tell runs apart."""
                    return "fresh text, no preamble: " + task.rsplit(" ", 1)[-1]
            ''',
        )
        self._run_raw({
            "prompt": cron_agent.PROMPT_PREAMBLE + "/replaces_prompt_sea scheduled",
            "seaPath": sea_path,
        })
        self._run_raw({"prompt": "a user's own slash run: typed", "seaPath": sea_path})
        flags = self._is_cron_by_task()
        self.assertEqual(
            {k: v for k, v in flags.items() if k.startswith("fresh text")},
            {"fresh text, no preamble: scheduled": 1, "fresh text, no preamble: typed": 0},
        )


if __name__ == "__main__":
    unittest.main()
