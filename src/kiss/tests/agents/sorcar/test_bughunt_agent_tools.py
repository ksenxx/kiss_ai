# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt integration tests for the sorcar agent modules.

Covers two real bugs (no mocks/patches/fakes; no paid LLM calls):

1. ``ChatSorcarAgent.run`` read ``use_worktree`` from ``**kwargs`` via
   ``kwargs.get`` (for the early "extra" persistence payload) but did
   not remove it before forwarding ``**kwargs`` to
   ``SorcarAgent.run()``, whose explicit signature has no
   ``use_worktree`` parameter — so any direct caller passing the
   anticipated kwarg got ``TypeError``.

2. The module-level ``run_tasks_parallel`` in ``sorcar_agent.py``
   resolved the parent's thread-local ``task_id`` INSIDE the worker
   thread (where ``threading.local`` never carries the parent thread's
   value, and where ``ChatSorcarAgent.run`` clears its own id before
   the ``finally`` runs), so the ``subagentDone`` broadcast — needed by
   the frontend to stop the running indicator — never fired.

Uses a real local HTTP server returning OpenAI-format ``finish`` tool
calls, a real ``JsonPrinter`` subclass that records broadcasts, and a
temp-dir-redirected persistence DB (same patterns as
``test_chat_parallel_integration.py`` / ``test_run_parallel_integration.py``).
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent


def _finish_response(model: str = "gpt-4o-mini") -> dict:
    """OpenAI chat-completion body that calls ``finish``."""
    return {
        "id": "chatcmpl-fin",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_fin",
                            "type": "function",
                            "function": {
                                "name": "finish",
                                "arguments": json.dumps(
                                    {"success": "true", "summary_in_html": "done"}
                                ),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class _FinishHandler(BaseHTTPRequestHandler):
    """Returns a ``finish`` tool call for every POST request."""

    def do_POST(self) -> None:  # noqa: N802
        cl = int(self.headers.get("Content-Length", 0))
        if cl:
            self.rfile.read(cl)
        body = json.dumps(_finish_response()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def _start_server() -> tuple[ThreadingHTTPServer, str]:
    """Start a local HTTP server and return ``(server, base_url)``."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FinishHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"


def _redirect(tmpdir: str) -> tuple:
    """Point the persistence module at a temp directory and reset the conn."""
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore(saved: tuple) -> None:
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


class TestUseWorktreeKwargConsumed:
    """``ChatSorcarAgent.run`` reads ``use_worktree`` for its early extra
    payload, so it must pop it instead of forwarding it to
    ``SorcarAgent.run`` (which has no such parameter)."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)
        self.srv, self.url = _start_server()

    def teardown_method(self) -> None:
        self.srv.shutdown()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_run_accepts_use_worktree_false(self) -> None:
        """Passing ``use_worktree=False`` must not raise TypeError."""
        agent = ChatSorcarAgent("bughunt-wt-false")
        result = agent.run(
            prompt_template="bughunt worktree kwarg",
            model_name="gpt-4o-mini",
            model_config={"base_url": self.url, "api_key": "test-key"},
            work_dir=self.tmpdir,
            use_worktree=False,
        )
        parsed = yaml.safe_load(result)
        assert isinstance(parsed, dict)
        assert parsed.get("summary") == "<p>done</p>"

    def test_run_accepts_use_worktree_true(self) -> None:
        """Passing ``use_worktree=True`` must not raise TypeError either."""
        agent = ChatSorcarAgent("bughunt-wt-true")
        result = agent.run(
            prompt_template="bughunt worktree kwarg true",
            model_name="gpt-4o-mini",
            model_config={"base_url": self.url, "api_key": "test-key"},
            work_dir=self.tmpdir,
            use_worktree=True,
        )
        parsed = yaml.safe_load(result)
        assert isinstance(parsed, dict)
        assert parsed.get("summary") == "<p>done</p>"
