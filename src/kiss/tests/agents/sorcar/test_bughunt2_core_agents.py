# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bughunt2 (core agents): genuine bugs in the sorcar core agent files.

Bug 2 — ``set_model`` never swaps the live model mid-run
--------------------------------------------------------
``SorcarAgent`` is a :class:`RelentlessAgent`: the model that actually
performs LLM calls lives on the inner per-session executor
(``self._current_executor``, a fresh ``KISSAgent`` per sub-session).
``RelentlessAgent._reset`` never sets ``self.model`` on the parent, so
the ``set_model`` tool's ``getattr(self, "model", None)`` was ALWAYS
``None`` during a real run — the tool silently took the "deferred"
branch (only updating ``self.model_name``, which is read again only at
the NEXT sub-session, which usually never starts).  The documented
contract — "swaps the agent's live LLM model instance in place so the
very next LLM call goes to the changed model" — was never met in
production.  Fixed by targeting ``self._current_executor`` (when live)
so the swap lands on the model instance the ReAct loop reads on each
step.

Bug 1 — ``is_worktree`` persisted wrongly for non-worktree runs
---------------------------------------------------------------
``ChatSorcarAgent.run`` computed the persisted ``is_worktree`` extra
flag as::

    bool(kwargs.pop("use_worktree", False)) or self.uses_worktree

and the end-of-run save used ``self.uses_worktree`` directly.  Both are
wrong for :class:`WorktreeSorcarAgent` (``uses_worktree = True`` at
class level), whose ``run()`` pops the ``use_worktree`` kwarg BEFORE
delegating here:

1. A ``WorktreeSorcarAgent`` invoked with ``use_worktree=False`` (the
   CLI ``--no-worktree`` flag, or the VS Code Worktree toggle OFF)
   executes directly on the main working tree, yet ``task_history``
   permanently records ``is_worktree = 1``.  The same mis-record
   happens when worktree setup falls back to direct execution (e.g.
   ``work_dir`` is not a git repo).
2. A plain ``ChatSorcarAgent`` invoked with an explicit
   ``use_worktree=True`` records ``is_worktree = 1`` in the early save
   but the final save overwrites the column back to ``0`` — the early
   and final saves disagree.

Consumers of the flag (``_replay_session`` seeding ``tab.use_worktree``,
the History sidebar metadata) therefore show worktree state that never
existed.

Fix
---
``ChatSorcarAgent.run`` now resolves the flag ONCE: an explicit
``use_worktree`` kwarg wins; otherwise a worktree-capable subclass is
probed for whether the effective ``work_dir`` was actually redirected
into its own worktree directory (``self._wt_dir``).  The same resolved
value feeds both the early and the final extra saves.

No mocks / no patches: a real local HTTP server plays the LLM (always
returning a ``finish`` tool call) and persistence is redirected to a
temp directory.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

# ---------------------------------------------------------------------------
# Fake OpenAI-compatible server (always returns a ``finish`` tool call)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# DB redirect helpers (same convention as the other sorcar tests)
# ---------------------------------------------------------------------------


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


def _load_is_worktree(task_id: str) -> bool:
    """Read the persisted ``is_worktree`` column for *task_id* directly."""
    conn = sqlite3.connect(th._DB_PATH)
    try:
        row = conn.execute(
            "SELECT is_worktree FROM task_history WHERE id = ?", (task_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, f"task_history row {task_id!r} not found"
    return bool(row[0])


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "HOME": str(repo),
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIsWorktreeExtraFlag:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt2-core-")
        self.saved = _redirect(self.tmpdir)
        self.srv, self.url = _start_server()
        _RunningAgentState.running_agent_states.clear()

    def teardown_method(self) -> None:
        self.srv.shutdown()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        _RunningAgentState.running_agent_states.clear()

    def _run(self, agent: ChatSorcarAgent, **kwargs: Any) -> str:
        cfg = {"base_url": self.url, "api_key": "test-key"}
        return agent.run(
            prompt_template="hello",
            model_name="gpt-4o-mini",
            model_config=cfg,
            printer=None,
            web_tools=False,
            **kwargs,
        )

    def test_worktree_agent_use_worktree_false_records_false(self) -> None:
        """A ``WorktreeSorcarAgent`` run with ``use_worktree=False`` runs on
        the main working tree, so ``is_worktree`` must persist as False."""
        work = Path(self.tmpdir) / "plain"
        work.mkdir()
        agent = WorktreeSorcarAgent("no-wt")
        self._run(agent, work_dir=str(work), use_worktree=False)
        task_id = agent._last_task_id
        assert task_id is not None
        assert _load_is_worktree(task_id) is False

    def test_worktree_agent_non_git_fallback_records_false(self) -> None:
        """When worktree setup falls back to direct execution (work_dir is
        not a git repo), ``is_worktree`` must persist as False."""
        work = Path(self.tmpdir) / "notgit"
        work.mkdir()
        agent = WorktreeSorcarAgent("fallback")
        self._run(agent, work_dir=str(work))  # use_worktree defaults True
        task_id = agent._last_task_id
        assert task_id is not None
        assert _load_is_worktree(task_id) is False

    def test_plain_chat_agent_explicit_true_kept_by_final_save(self) -> None:
        """An explicit ``use_worktree=True`` on a plain ``ChatSorcarAgent``
        must survive the end-of-run extra save (early save recorded True;
        the final save used to flip it back to False)."""
        work = Path(self.tmpdir) / "chat"
        work.mkdir()
        agent = ChatSorcarAgent("explicit-true")
        self._run(agent, work_dir=str(work), use_worktree=True)
        task_id = agent._last_task_id
        assert task_id is not None
        assert _load_is_worktree(task_id) is True

    def test_worktree_agent_real_worktree_records_true(self) -> None:
        """Regression guard: a real worktree run still records True."""
        repo = Path(self.tmpdir) / "repo"
        repo.mkdir()
        _git(repo, "init", "-b", "main")
        _git(repo, "config", "user.email", "t@t.t")
        _git(repo, "config", "user.name", "t")
        (repo / "f.txt").write_text("hello\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-m", "init")
        agent = WorktreeSorcarAgent("real-wt")
        self._run(agent, work_dir=str(repo))
        task_id = agent._last_task_id
        assert task_id is not None
        assert _load_is_worktree(task_id) is True


# ---------------------------------------------------------------------------
# Bug 2: set_model must swap the LIVE model so the next LLM call uses it
# ---------------------------------------------------------------------------


class _SetModelHandler(BaseHTTPRequestHandler):
    """First request returns a ``set_model`` tool call; later ones ``finish``.

    Records the ``model`` field of every incoming request body on the
    server object so the test can assert which model each LLM call was
    addressed to.
    """

    def do_POST(self) -> None:  # noqa: N802
        cl = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(cl) if cl else b"{}"
        try:
            req_model = json.loads(raw).get("model", "")
        except Exception:
            req_model = ""
        server: Any = self.server
        with server.state_lock:
            server.request_models.append(req_model)
            first = len(server.request_models) == 1
        if first:
            tool = {
                "id": "call_sm",
                "type": "function",
                "function": {
                    "name": "set_model",
                    "arguments": json.dumps({"model_name": "swapped-model"}),
                },
            }
        else:
            tool = {
                "id": "call_fin",
                "type": "function",
                "function": {
                    "name": "finish",
                    "arguments": json.dumps(
                        {"success": "true", "summary": "done"}
                    ),
                },
            }
        body = json.dumps(
            {
                "id": "chatcmpl-x",
                "object": "chat.completion",
                "model": req_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [tool],
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
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class TestSetModelSwapsLiveModel:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt2-setmodel-")
        self.saved = _redirect(self.tmpdir)
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), _SetModelHandler)
        self.srv.request_models = []  # type: ignore[attr-defined]
        self.srv.state_lock = threading.Lock()  # type: ignore[attr-defined]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}/v1"
        _RunningAgentState.running_agent_states.clear()

    def teardown_method(self) -> None:
        self.srv.shutdown()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        _RunningAgentState.running_agent_states.clear()

    def test_next_llm_call_uses_swapped_model(self) -> None:
        """After the agent calls ``set_model('swapped-model')`` mid-run, the
        very next LLM request must be addressed to ``swapped-model``."""
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent

        work = Path(self.tmpdir) / "w"
        work.mkdir()
        agent = SorcarAgent("set-model-live")
        result = agent.run(
            prompt_template="please switch models then finish",
            model_name="gpt-4o-mini",
            model_config={"base_url": self.url, "api_key": "test-key"},
            work_dir=str(work),
            web_tools=False,
            max_steps=5,
        )
        assert "success" in result
        models = list(self.srv.request_models)  # type: ignore[attr-defined]
        assert len(models) >= 2
        assert models[0] == "gpt-4o-mini"
        # The bug: the second request still went to gpt-4o-mini because
        # set_model only touched the relentless parent (which has no
        # live model), never the executor actually making the calls.
        assert models[1] == "swapped-model"
        # The parent's model_name must also reflect the change so any
        # subsequent sub-session stays on the new model.
        assert agent.model_name == "swapped-model"
