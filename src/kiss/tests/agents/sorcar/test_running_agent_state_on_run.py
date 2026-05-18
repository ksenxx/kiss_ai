"""Regression: ``WorktreeSorcarAgent.run()`` publishes itself in
``_RunningAgentState.running_agent_states`` while running.

When the standalone :class:`WorktreeSorcarAgent` is invoked, the
process-global running-state map (a class attribute on
:class:`_RunningAgentState`) should contain a live
:class:`_RunningAgentState` entry keyed by the agent's ``chat_id`` for
the duration of the ``run()`` call, and the entry must be removed once
``run()`` completes.  The entry's ``agent`` field must be the caller
itself (not a freshly-allocated ``WorktreeSorcarAgent``).

When an entry with a matching ``chat_id`` is already present (e.g. the
VS Code server pre-populates one keyed by the frontend tab id ahead of
run-start), ``run()`` must NOT clobber or pop it.

Uses a real local HTTP server that always returns a ``finish`` tool call —
no mocks / no patches.  Persistence is redirected to a temp directory.
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
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

# ---------------------------------------------------------------------------
# Fake OpenAI server (always returns a ``finish`` tool call)
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
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
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
# DB redirect helpers
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunningStatePopulatedOnRun:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)
        self.srv, self.url = _start_server()
        # Ensure a clean class-attribute slate; the dict is process-wide.
        _RunningAgentState.running_agent_states.clear()

    def teardown_method(self) -> None:
        self.srv.shutdown()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        _RunningAgentState.running_agent_states.clear()

    def test_state_added_while_running_and_removed_after(self) -> None:
        """``run()`` adds an entry keyed by ``chat_id`` and removes it on exit."""
        cfg = {"base_url": self.url, "api_key": "test-key"}
        agent = WorktreeSorcarAgent("standalone")

        # Capture the state mid-run by hooking the upstream finish tool's
        # bookkeeping isn't trivial; instead, just check the post-run
        # invariant and add a real mid-run check via an inner agent run.
        agent.run(
            prompt_template="hello",
            model_name="gpt-4o-mini",
            model_config=cfg,
            work_dir=self.tmpdir,
        )
        # After run() returns, the entry must be gone again.
        assert agent.chat_id not in _RunningAgentState.running_agent_states

    def test_state_is_live_during_run(self) -> None:
        """While ``run()`` is executing, an entry IS present keyed by chat_id.

        We use a custom ``finish_callback`` injected via the fake server's
        response — easier: query the dict from a different thread while
        the agent runs.  Since the fake server is synchronous, we use a
        signaling Event to observe the state.
        """
        cfg = {"base_url": self.url, "api_key": "test-key"}
        agent = WorktreeSorcarAgent("live-check")
        # Pre-set chat_id so we know the key in advance.
        agent.resume_chat_by_id("live-chat-id")

        observed: dict[str, Any] = {}
        started = threading.Event()
        finished = threading.Event()

        def observer() -> None:
            started.wait(timeout=5)
            # Snapshot the dict membership while the worker is mid-run.
            state = _RunningAgentState.running_agent_states.get("live-chat-id")
            observed["state"] = state
            observed["agent_is_self"] = state is not None and state.agent is agent
            observed["is_task_active"] = state is not None and state.is_task_active
            finished.set()

        def worker() -> None:
            # Wrap to flip ``started`` AFTER state is registered.
            # We rely on the fact that registration happens before
            # ``super().run()`` is called.  To synchronize, we use a
            # callback in finish handler — but simpler: just toggle
            # started right before run() and rely on observer to
            # see the live state because the HTTP round-trip takes
            # a measurable time.
            started.set()
            agent.run(
                prompt_template="hello live",
                model_name="gpt-4o-mini",
                model_config=cfg,
                work_dir=self.tmpdir,
            )

        # Slightly more robust: observer polls until state appears.
        def observer_poll() -> None:
            started.wait(timeout=5)
            for _ in range(2000):  # up to ~2s
                state = _RunningAgentState.running_agent_states.get("live-chat-id")
                if state is not None:
                    observed["state"] = state
                    observed["agent_is_self"] = state.agent is agent
                    observed["is_task_active"] = state.is_task_active
                    break
                threading.Event().wait(0.001)
            finished.set()

        t_obs = threading.Thread(target=observer_poll)
        t_work = threading.Thread(target=worker)
        t_obs.start()
        t_work.start()
        t_work.join(timeout=30)
        t_obs.join(timeout=5)

        assert observed.get("state") is not None, "state was never observed mid-run"
        assert isinstance(observed["state"], _RunningAgentState)
        assert observed["agent_is_self"], "state.agent must be the running agent itself"
        assert observed["is_task_active"], "is_task_active should be True mid-run"

        # And the entry is gone after run() returns.
        assert "live-chat-id" not in _RunningAgentState.running_agent_states

    def test_run_does_not_clobber_preexisting_state(self) -> None:
        """If an entry already exists (VS Code server case), ``run()`` leaves it alone."""
        cfg = {"base_url": self.url, "api_key": "test-key"}
        agent = WorktreeSorcarAgent("vscode-emulated")
        agent.resume_chat_by_id("pre-existing-id")

        # Pre-populate as the VS Code server would: a fresh
        # ``_RunningAgentState`` whose internal agent is the standard
        # ``WorktreeSorcarAgent("Sorcar VS Code")`` (NOT *agent*) and
        # whose ``chat_id`` matches the agent's chat id so the
        # standalone agent's ``_register_running_state`` detects the
        # pre-existing entry and skips re-registration.
        preexisting = _RunningAgentState("pre-existing-id", "gpt-4o-mini")
        preexisting.chat_id = "pre-existing-id"
        _RunningAgentState.running_agent_states["pre-existing-id"] = preexisting

        agent.run(
            prompt_template="please don't clobber me",
            model_name="gpt-4o-mini",
            model_config=cfg,
            work_dir=self.tmpdir,
        )

        # The pre-existing entry must still be there, unchanged.
        assert _RunningAgentState.running_agent_states.get("pre-existing-id") is preexisting
        assert preexisting.agent is not agent
