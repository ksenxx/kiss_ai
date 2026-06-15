# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Invariant: every running ``ChatSorcarAgent`` is registered with the server.

"Registered with the server" means there is an entry in the process-
global :attr:`_RunningAgentState.running_agent_states` registry such
that ``state.agent is the_agent``.  Consumers that rely on this
invariant include:

- :meth:`VSCodeServer._reattach_running_chat` (subscribes a freshly
  opened viewer tab to an in-flight chat).
- :meth:`VSCodeServer._get_running_task_ids` (renders the running
  indicator next to in-flight tasks in the History sidebar).
- :meth:`ChatSorcarAgent._run_tasks_parallel` (looks up the parent's
  ``tab_id`` so each sub-agent's ``new_tab`` broadcast carries a
  ``parent_tab_id`` routing hint that keeps phantom sub-agent tabs
  out of unrelated webviews).

The invariant is upheld for:

- UI-launched top-level tasks — registered by
  :meth:`VSCodeServer._TaskRunnerMixin._run_task_inner` before
  ``agent.run()``.
- Parallel sub-agents — registered by
  :meth:`ChatSorcarAgent._run_tasks_parallel` before
  ``agent.run()``.
- Worktree-CLI top-level tasks — registered by
  :meth:`WorktreeSorcarAgent._register_running_state` before
  ``super().run()``.

The invariant was BROKEN for plain :class:`ChatSorcarAgent` runs that
do not go through any of the above paths — i.e. CLI / third-party /
remote-webapp invocations of ``ChatSorcarAgent.run()`` directly.
These tests reproduce that gap and lock in the fix.

These tests use a real local HTTP server returning a ``finish`` tool
call (no mocks).  Persistence is redirected to a temp directory so the
SQLite database is isolated.
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
from kiss.agents.sorcar.running_agent_state import _RunningAgentState


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


class TestChatSorcarAgentStateRegistration:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)
        self.srv, self.url = _start_server()
        # Start from a clean registry so concurrent test pollution
        # cannot mask the invariant being checked here.
        with _RunningAgentState._registry_lock:
            _RunningAgentState.running_agent_states.clear()
        ChatSorcarAgent.running_agents.clear()

    def teardown_method(self) -> None:
        self.srv.shutdown()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        with _RunningAgentState._registry_lock:
            _RunningAgentState.running_agent_states.clear()
        ChatSorcarAgent.running_agents.clear()

    def test_chat_agent_self_registers_during_run(self) -> None:
        """During ``ChatSorcarAgent.run()`` an entry must point at ``self``.

        Mirrors the consumer at
        :meth:`ChatSorcarAgent._run_tasks_parallel` L240-244 which
        scans the registry for ``state.agent is self`` to resolve
        the parent tab id for sub-agent ``new_tab`` routing.  Before
        the fix the standalone CLI / third-party path never inserted
        such an entry, so this scan returned an empty ``parent_tab_id``
        and webviews materialised phantom sub-agent tabs.
        """
        cfg = {"base_url": self.url, "api_key": "test-key"}
        agent = ChatSorcarAgent("cli-standalone")

        observed: dict[str, Any] = {}

        def observer_poll() -> None:
            for _ in range(2000):  # up to ~2s
                with _RunningAgentState._registry_lock:
                    snapshot = list(
                        _RunningAgentState.running_agent_states.items()
                    )
                hit = next(
                    (
                        (tab_id, st)
                        for tab_id, st in snapshot
                        if st.agent is agent
                    ),
                    None,
                )
                if hit is not None:
                    tab_id, st = hit
                    observed["tab_id"] = tab_id
                    observed["chat_id"] = st.chat_id
                    observed["is_task_active"] = st.is_task_active
                    return
                threading.Event().wait(0.001)

        def worker() -> None:
            agent.run(
                prompt_template="hello",
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

        assert observed.get("tab_id") is not None, (
            "ChatSorcarAgent did not register itself in "
            "_RunningAgentState.running_agent_states while run() was "
            "executing — the per-tab state registry invariant is broken"
        )
        # ``is_task_active`` must be True so the live-task consumer
        # paths (``_reattach_running_chat``, ``_get_running_task_ids``)
        # treat the agent as in-flight.
        assert observed["is_task_active"] is True
        # The agent's own ``chat_id`` must propagate into the state so
        # by-chat lookups (e.g. multi-viewer subscribe) can find it.
        assert observed["chat_id"] != ""
        assert observed["chat_id"] == agent.chat_id

    def test_registry_cleaned_up_after_run(self) -> None:
        """After ``run()`` returns the self-registered entry must be gone."""
        cfg = {"base_url": self.url, "api_key": "test-key"}
        agent = ChatSorcarAgent("cleanup-check")

        agent.run(
            prompt_template="hello",
            model_name="gpt-4o-mini",
            model_config=cfg,
            work_dir=self.tmpdir,
        )

        with _RunningAgentState._registry_lock:
            leaked = [
                (tab_id, st)
                for tab_id, st in (
                    _RunningAgentState.running_agent_states.items()
                )
                if st.agent is agent
            ]
        assert leaked == [], (
            "ChatSorcarAgent.run()'s finally must remove its own "
            "_RunningAgentState entry; leaked entries persist as "
            "phantom in-flight rows in the History sidebar"
        )

    def test_registry_cleaned_up_when_run_raises(self) -> None:
        """The cleanup must still happen on the error path."""
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent

        cfg = {"base_url": self.url, "api_key": "test-key"}
        agent = ChatSorcarAgent("error-path")
        boom = RuntimeError("synthetic failure")

        def _raise(_self: Any, *_a: Any, **_kw: Any) -> str:
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

        assert raised is boom
        with _RunningAgentState._registry_lock:
            leaked = [
                (tab_id, st)
                for tab_id, st in (
                    _RunningAgentState.running_agent_states.items()
                )
                if st.agent is agent
            ]
        assert leaked == [], (
            "_RunningAgentState entry must be removed even when "
            "run() raises"
        )

    def test_parent_tab_id_lookup_succeeds_for_cli_parent(self) -> None:
        """The ``_run_tasks_parallel`` parent-tab-id scan finds an entry.

        This is the concrete observable bug caused by the broken
        invariant: when a CLI ``ChatSorcarAgent`` parent calls
        ``_run_tasks_parallel``, the parent-tab-id lookup at L240-244
        scans :attr:`_RunningAgentState.running_agent_states` for a
        ``state.agent is self`` match.  Before the fix the standalone
        run never registered such an entry, so the scan returned an
        empty ``parent_tab_id`` and sub-agent ``new_tab`` broadcasts
        lost their routing hint.

        Verifies the production scan code path returns a non-empty
        parent tab id while the agent's ``run()`` is active.
        """
        cfg = {"base_url": self.url, "api_key": "test-key"}
        agent = ChatSorcarAgent("parent-tab-id-check")

        observed: dict[str, Any] = {}

        def observer_poll() -> None:
            for _ in range(2000):  # up to ~2s
                # Exact production scan from
                # ``ChatSorcarAgent._run_tasks_parallel``.
                parent_tab_id = ""
                with _RunningAgentState._registry_lock:
                    for tid, st in (
                        _RunningAgentState.running_agent_states.items()
                    ):
                        if st.agent is agent:
                            parent_tab_id = tid
                            break
                if parent_tab_id:
                    observed["parent_tab_id"] = parent_tab_id
                    return
                threading.Event().wait(0.001)

        def worker() -> None:
            agent.run(
                prompt_template="hello",
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

        assert observed.get("parent_tab_id"), (
            "_run_tasks_parallel's parent-tab-id lookup returned "
            "empty for a CLI ChatSorcarAgent parent — sub-agent "
            "new_tab broadcasts lose their parent_tab_id routing hint"
        )
