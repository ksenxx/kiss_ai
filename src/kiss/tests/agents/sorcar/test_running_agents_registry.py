"""Regression: ``ChatSorcarAgent.run()`` publishes itself in
``ChatSorcarAgent.running_agents`` keyed by ``task_history.id``.

The class attribute ``ChatSorcarAgent.running_agents`` is a process-
global map ``task_id -> ChatSorcarAgent`` that lets external observers
find the live agent driving a given persisted task row.  An entry MUST
be inserted as soon as the task row is written to ``sorcar.db`` (i.e.
right after :func:`_add_task` returns inside :meth:`ChatSorcarAgent.run`)
and removed in the same method's ``finally`` block once ``run()``
returns or raises.

These tests use a real local HTTP server that always returns a
``finish`` tool call -- no mocks, no patches.  Persistence is
redirected to a temp directory so the SQLite database is isolated.
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


def _finish_body() -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl-fin",
            "object": "chat.completion",
            "model": "gpt-4o-mini",
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
                                        {"success": "true", "summary": "done"}
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
    ).encode()


class _FinishHandler(BaseHTTPRequestHandler):
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
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FinishHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"


def _redirect(tmpdir: str) -> tuple[Any, Any, Any]:
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore(saved: tuple[Any, Any, Any]) -> None:
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


class TestRunningAgentsRegistry:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)
        self.srv, self.url = _start_server()
        ChatSorcarAgent.running_agents.clear()

    def teardown_method(self) -> None:
        self.srv.shutdown()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        ChatSorcarAgent.running_agents.clear()

    def test_entry_removed_after_run_returns(self) -> None:
        """An entry exists during ``run()`` and is cleaned up afterwards."""
        cfg = {"base_url": self.url, "api_key": "test-key"}
        agent = ChatSorcarAgent("standalone")

        agent.run(
            prompt_template="hello",
            model_name="gpt-4o-mini",
            model_config=cfg,
            work_dir=self.tmpdir,
        )

        # After ``run()`` returns, the agent's last task id must be
        # set and the registry must NOT still contain that key.
        assert agent._last_task_id is not None
        assert agent._last_task_id not in ChatSorcarAgent.running_agents
        assert ChatSorcarAgent.running_agents == {}

    def test_entry_present_mid_run_keyed_by_task_id(self) -> None:
        """Mid-run, ``running_agents[task_id]`` is ``agent`` itself."""
        cfg = {"base_url": self.url, "api_key": "test-key"}
        agent = ChatSorcarAgent("live-check")

        observed: dict[str, Any] = {}
        finished = threading.Event()

        def observer_poll() -> None:
            for _ in range(2000):  # up to ~2s
                # Snapshot a copy because the worker may pop concurrently.
                snapshot = dict(ChatSorcarAgent.running_agents)
                if snapshot:
                    task_id, live = next(iter(snapshot.items()))
                    observed["task_id"] = task_id
                    observed["agent_is_self"] = live is agent
                    break
                threading.Event().wait(0.001)
            finished.set()

        def worker() -> None:
            agent.run(
                prompt_template="hello live",
                model_name="gpt-4o-mini",
                model_config=cfg,
                work_dir=self.tmpdir,
            )

        t_obs = threading.Thread(target=observer_poll)
        t_work = threading.Thread(target=worker)
        t_obs.start()
        t_work.start()
        t_work.join(timeout=30)
        t_obs.join(timeout=5)

        assert observed.get("task_id") is not None, (
            "no entry was ever observed in ChatSorcarAgent.running_agents "
            "while run() was executing"
        )
        assert observed["agent_is_self"], (
            "running_agents[task_id] must be the agent that is running, "
            "not some other instance"
        )
        # ``agent._last_task_id`` was assigned right before the
        # registry insert, so the keys must match.
        assert observed["task_id"] == agent._last_task_id
        # And on exit the registry is empty.
        assert ChatSorcarAgent.running_agents == {}

    def test_entry_removed_when_run_raises(self) -> None:
        """``finally`` cleanup runs even when the inner ``super().run`` errors.

        Replaces ``SorcarAgent.run`` on the class for the duration of
        the test with a function that raises a known exception.  This
        forces ``ChatSorcarAgent.run``'s ``super().run(...)`` call to
        raise so we can confirm the registry's ``finally`` cleanup
        still pops the entry.
        """
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent

        cfg = {"base_url": self.url, "api_key": "test-key"}
        agent = ChatSorcarAgent("error-path")
        boom = RuntimeError("synthetic failure")

        def _raise(_self: Any, *_args: Any, **_kwargs: Any) -> str:
            raise boom

        original = SorcarAgent.run
        SorcarAgent.run = _raise  # type: ignore[method-assign,assignment]
        try:
            raised: Exception | None = None
            try:
                agent.run(
                    prompt_template="will fail",
                    model_name="gpt-4o-mini",
                    model_config=cfg,
                    work_dir=self.tmpdir,
                )
            except Exception as e:  # noqa: BLE001
                raised = e
        finally:
            SorcarAgent.run = original  # type: ignore[method-assign]

        assert raised is boom, "the original exception must propagate"
        # Even though run() raised, the finally must have popped the
        # entry so it does not leak.
        assert ChatSorcarAgent.running_agents == {}
