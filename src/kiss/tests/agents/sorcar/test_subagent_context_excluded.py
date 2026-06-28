# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: sub-agent task rows must not appear in parent chat context.

Reproduces the bug where ``run_parallel`` sub-agent tasks (which share
the parent's ``chat_id`` so they all get persisted to the same chat
session) leaked into the parent agent's "Previous tasks and results"
prompt augmentation built by :meth:`ChatSorcarAgent.build_chat_prompt`.

User-visible symptom: in a chat tab that previously ran
``run_parallel`` with N sub-tasks, the next turn's prompt for the
LLM contained the original parent task PLUS N extra "Task M" / "Result
M" panels for each sub-agent's internal task.  These panels also
showed up in the parent tab's history because
:func:`_load_chat_context` returned every row whose ``chat_id``
matched, without filtering on the ``subagent`` marker stored in the
``extra`` column.

The fix filters rows whose ``extra`` JSON contains a ``subagent`` key.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import (
    _add_task,
    _load_chat_context,
    _save_task_result,
)


def _finish_body() -> bytes:
    """Minimal OpenAI tool-call response invoking ``finish``."""
    return json.dumps({
        "id": "x",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "c",
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "arguments": json.dumps(
                            {"success": "true", "summary": "parent-done"},
                        ),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }).encode()


class _Handler(BaseHTTPRequestHandler):
    """HTTP handler that always returns a ``finish`` tool-call response."""

    def do_POST(self) -> None:  # noqa: N802
        cl = int(self.headers.get("Content-Length", 0))
        if cl:
            self.rfile.read(cl)
        body = _finish_body()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def _start_server() -> tuple[ThreadingHTTPServer, str]:
    """Start a local HTTP server returning ``finish`` for every request."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"


class TestSubagentRowsExcludedFromChatContext:
    """``_load_chat_context`` must filter out sub-agent rows."""

    def setup_method(self) -> None:
        """Redirect persistence at a fresh temp DB before each test."""
        self.tmpdir = tempfile.mkdtemp()
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        self._saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        self.srv, self.url = _start_server()

    def teardown_method(self) -> None:
        """Restore persistence globals and tear down the temp DB / server."""
        self.srv.shutdown()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        (th._DB_PATH, th._db_conn, th._KISS_DIR) = self._saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _insert_subagent_row(
        self, parent_task_id: int, chat_id: str, task: str, result: str,
    ) -> int:
        """Persist a row that simulates one ``_run_tasks_parallel`` worker.

        Matches the ``extra`` payload written by
        :meth:`ChatSorcarAgent._run_tasks_parallel`'s worker thread:
        ``{"subagent": {"parent_task_id": <id>}, ...}``.
        """
        sub_id, _ = _add_task(
            task,
            chat_id=chat_id,
            extra={
                "model": "gpt-4o-mini",
                "subagent": {"parent_task_id": parent_task_id},
            },
        )
        _save_task_result(result, task_id=sub_id)
        return sub_id

    def test_load_chat_context_excludes_subagent_rows(self) -> None:
        """Manually-inserted sub-agent rows must not appear in chat context."""
        parent_id, chat_id = _add_task("parent task", chat_id="")
        _save_task_result("parent-result", task_id=parent_id)

        # Three sub-agents, all sharing the parent's chat_id and tagged
        # with ``extra.subagent.parent_task_id`` like ``run_parallel`` does.
        for i in range(3):
            self._insert_subagent_row(
                parent_id, chat_id, f"sub task {i}", f"sub result {i}",
            )

        # The DB has 4 rows under this chat_id ...
        with th._rw_lock.read_lock():
            db = th._get_db()
            total = db.execute(
                "SELECT COUNT(*) AS n FROM task_history WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()["n"]
        assert total == 4

        # ... but _load_chat_context filters out the 3 sub-agent rows.
        context = _load_chat_context(chat_id)
        assert len(context) == 1, (
            f"Expected only the parent row, got {len(context)}: {context}"
        )
        assert context[0]["task"] == "parent task"
        assert context[0]["result"] == "parent-result"

    def test_build_chat_prompt_excludes_subagent_panels(self) -> None:
        """``build_chat_prompt`` must not surface sub-agent Task/Result panels."""
        parent_id, chat_id = _add_task("orig parent prompt", chat_id="")
        _save_task_result("orig parent result", task_id=parent_id)

        for i in range(3):
            self._insert_subagent_row(
                parent_id, chat_id,
                f"hidden sub task {i}",
                f"hidden sub result {i}",
            )

        agent = ChatSorcarAgent("resume")
        agent.resume_chat_by_id(chat_id)
        augmented = agent.build_chat_prompt("next user message")

        # The parent task / result appears exactly once.
        assert "orig parent prompt" in augmented
        assert "orig parent result" in augmented
        # No sub-agent task or result text leaks into the LLM prompt.
        for i in range(3):
            assert f"hidden sub task {i}" not in augmented, (
                "Sub-agent task leaked into build_chat_prompt output"
            )
            assert f"hidden sub result {i}" not in augmented, (
                "Sub-agent result leaked into build_chat_prompt output"
            )
        # Only one "### Task" / "### Result" pair (the parent's).
        assert augmented.count("### Task 1") == 1
        assert "### Task 2" not in augmented

    def test_run_parallel_end_to_end_excludes_subagent_rows(self) -> None:
        """End-to-end: parent agent + 3 real sub-agents → context has 1 entry.

        Simulates the exact flow ``_run_tasks_parallel`` uses (without the
        thread pool / printer) by spawning sub-agents that
        :meth:`resume_chat_by_id` the parent's chat and set the same
        ``_subagent_info`` field the production worker sets.
        """
        model_config: dict[str, Any] = {
            "base_url": self.url, "api_key": "test-key",
        }

        parent = ChatSorcarAgent("parent")
        parent.run(
            prompt_template="kick off run_parallel",
            model_name="gpt-4o-mini",
            model_config=model_config,
            work_dir=self.tmpdir,
        )
        chat_id = parent.chat_id
        assert chat_id
        parent_task_id = parent._last_task_id
        assert isinstance(parent_task_id, str)

        for i in range(3):
            sub = ChatSorcarAgent(f"sub-{i}")
            sub.resume_chat_by_id(chat_id)
            sub._subagent_info = {"parent_task_id": parent_task_id}
            sub.run(
                prompt_template=f"sub task {i}",
                model_name="gpt-4o-mini",
                model_config=model_config,
                work_dir=self.tmpdir,
            )

        # All 4 rows exist in the DB ...
        with th._rw_lock.read_lock():
            db = th._get_db()
            total = db.execute(
                "SELECT COUNT(*) AS n FROM task_history WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()["n"]
        assert total == 4, f"Expected 4 rows under chat, got {total}"

        # ... but the chat context only returns the parent.
        context = _load_chat_context(chat_id)
        assert len(context) == 1, (
            f"Expected only parent row in context, got {len(context)}: "
            f"{[e['task'] for e in context]}"
        )
        assert context[0]["task"] == "kick off run_parallel"

        # And the next turn's prompt augmentation is parent-only.
        nxt = ChatSorcarAgent("next-turn")
        nxt.resume_chat_by_id(chat_id)
        augmented = nxt.build_chat_prompt("follow-up")
        for i in range(3):
            assert f"sub task {i}" not in augmented
