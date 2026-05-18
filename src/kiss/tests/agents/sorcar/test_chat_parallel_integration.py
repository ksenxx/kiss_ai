"""Integration tests for parallel ChatSorcarAgent sub-agents with shared chat_id.

Verifies that sub-agents sharing a parent's chat_id correctly
accumulate tasks and results in the same chat session history.  Uses a
real HTTP server (no mocks/patches) that returns ``finish`` tool-call
responses in the OpenAI chat-completion format.

Includes both sequential and concurrent (``ThreadPoolExecutor``) tests.
The persistence module uses per-thread SQLite connections with a
read-write lock, making concurrent access safe.
"""

from __future__ import annotations

import json
import random
import shutil
import sqlite3
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import (
    _add_task,
    _allocate_chat_id,
    _load_chat_context,
    _save_task_result,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.vscode.browser_ui import BaseBrowserPrinter

def _retry_on_busy[_T](op: Callable[[], _T], attempts: int = 8) -> _T:
    """Retry ``op`` on transient SQLite "database is locked" errors.

    The persistence layer uses WAL + a 5s ``busy_timeout``, but under
    aggressive thread-pool contention with many short-lived writes
    SQLite still occasionally returns ``SQLITE_BUSY`` because the WAL
    writer holds the file lock while committing.  The concurrency
    invariants the tests verify are unaffected by retrying these
    transient errors at the call site.
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            return op()
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower():
                raise
            last = exc
            time.sleep(0.05 * (i + 1))
    assert last is not None
    raise last

# ---------------------------------------------------------------------------
# Fake OpenAI-compatible server that always returns a ``finish`` tool call
# ---------------------------------------------------------------------------

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
                                    {"success": "true", "summary": "done"}
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


# ---------------------------------------------------------------------------
# DB-redirect helpers (same pattern as test_stateful_sorcar_agent.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSequentialSharedChatId:
    """Multiple sequential ChatSorcarAgents with the same chat_id share history."""

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

    def test_sequential_agents_accumulate_in_same_chat(self) -> None:
        """Three agents run sequentially with the same chat_id.

        All three tasks should appear under one chat session.
        """
        model_config = {"base_url": self.url, "api_key": "test-key"}

        # Agent 1 — starts a new chat session
        agent1 = ChatSorcarAgent("seq-1")
        agent1.run(
            prompt_template="task alpha",
            model_name="gpt-4o-mini",
            model_config=model_config,
            work_dir=self.tmpdir,
        )
        chat_id = agent1.chat_id
        assert chat_id, "agent1 should have received a chat_id"

        # Agent 2 — resumes the same session
        agent2 = ChatSorcarAgent("seq-2")
        agent2.resume_chat_by_id(chat_id)
        agent2.run(
            prompt_template="task bravo",
            model_name="gpt-4o-mini",
            model_config=model_config,
            work_dir=self.tmpdir,
        )
        assert agent2.chat_id == chat_id

        # Agent 3 — also resumes
        agent3 = ChatSorcarAgent("seq-3")
        agent3.resume_chat_by_id(chat_id)
        agent3.run(
            prompt_template="task charlie",
            model_name="gpt-4o-mini",
            model_config=model_config,
            work_dir=self.tmpdir,
        )
        assert agent3.chat_id == chat_id

        # Verify all 3 tasks are in the same chat history
        context = _load_chat_context(chat_id)
        assert len(context) == 3
        tasks = [e["task"] for e in context]
        assert tasks == ["task alpha", "task bravo", "task charlie"]

        # Verify results were saved (finish tool returns "done")
        for entry in context:
            assert entry["result"] == "done"

    def test_results_are_persisted_for_each_agent(self) -> None:
        """Each agent's result is individually persisted under the shared chat."""
        model_config = {"base_url": self.url, "api_key": "test-key"}

        agent1 = ChatSorcarAgent("r-1")
        agent1.run(
            prompt_template="compute alpha",
            model_name="gpt-4o-mini",
            model_config=model_config,
            work_dir=self.tmpdir,
        )
        chat_id = agent1.chat_id

        agent2 = ChatSorcarAgent("r-2")
        agent2.resume_chat_by_id(chat_id)
        agent2.run(
            prompt_template="compute bravo",
            model_name="gpt-4o-mini",
            model_config=model_config,
            work_dir=self.tmpdir,
        )

        context = _load_chat_context(chat_id)
        assert len(context) == 2
        assert all(e["result"] == "done" for e in context)


class TestParallelFlowSimulation:
    """Simulate the _run_tasks_parallel flow: parent + sequential sub-agents.

    The _run_tasks_parallel method creates ChatSorcarAgent sub-agents
    that share the parent's chat_id.  Here we simulate that flow
    sequentially to verify chat_id propagation without SQLite
    concurrency issues.
    """

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

    def test_parent_then_sub_agents_share_chat_id(self) -> None:
        """Parent runs, then sub-agents with same chat_id accumulate in session.

        Replicates the _run_tasks_parallel flow: parent establishes
        chat_id, sub-agents resume it.
        """
        model_config = {"base_url": self.url, "api_key": "test-key"}

        # Parent runs first
        parent = ChatSorcarAgent("parent")
        parent.run(
            prompt_template="parent task",
            model_name="gpt-4o-mini",
            model_config=model_config,
            work_dir=self.tmpdir,
        )
        chat_id = parent.chat_id
        assert chat_id

        context_before = _load_chat_context(chat_id)
        assert len(context_before) == 1
        assert context_before[0]["task"] == "parent task"

        # Sub-agents resume the same chat (as _run_tasks_parallel does)
        sub_tasks = ["sub-task-one", "sub-task-two", "sub-task-three"]
        for task_name in sub_tasks:
            agent = ChatSorcarAgent(f"Parallel-{task_name[:40]}")
            agent.resume_chat_by_id(chat_id)
            agent.run(
                prompt_template=task_name,
                model_name="gpt-4o-mini",
                model_config=model_config,
                work_dir=self.tmpdir,
            )

        # All tasks should be in the same chat session
        context_after = _load_chat_context(chat_id)
        assert len(context_after) == 4  # parent + 3 sub-tasks
        recorded_tasks = {e["task"] for e in context_after}
        assert "parent task" in recorded_tasks
        for st in sub_tasks:
            assert st in recorded_tasks

    def test_no_parent_chat_id_means_separate_sessions(self) -> None:
        """Without a shared chat_id, each agent starts its own session."""
        model_config = {"base_url": self.url, "api_key": "test-key"}
        task_names = ["independent-a", "independent-b"]

        agents = []
        for task in task_names:
            agent = ChatSorcarAgent(f"ind-{task}")
            agent.run(
                prompt_template=task,
                model_name="gpt-4o-mini",
                model_config=model_config,
                work_dir=self.tmpdir,
            )
            agents.append(agent)

        # Each agent should have its own chat_id
        chat_ids = {a.chat_id for a in agents}
        assert len(chat_ids) == 2, (
            f"Expected 2 distinct chat_ids, got {len(chat_ids)}: {chat_ids}"
        )

    def test_sub_agent_sees_parent_context_in_prompt(self) -> None:
        """Sub-agent's prompt includes parent's task/result from chat history."""
        model_config = {"base_url": self.url, "api_key": "test-key"}

        parent = ChatSorcarAgent("parent")
        parent.run(
            prompt_template="initial setup task",
            model_name="gpt-4o-mini",
            model_config=model_config,
            work_dir=self.tmpdir,
        )
        chat_id = parent.chat_id

        # Sub-agent resumes and builds prompt
        sub = ChatSorcarAgent("sub")
        sub.resume_chat_by_id(chat_id)
        prompt = sub.build_chat_prompt("follow-up task")

        assert "initial setup task" in prompt
        assert "done" in prompt  # parent's result
        assert "follow-up task" in prompt


class TestChatContextVisibleToSubAgents:
    """Sub-agents can see prior chat context when building their prompts."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_sub_agent_sees_prior_tasks_in_prompt(self) -> None:
        """A sub-agent with a shared chat_id sees prior tasks in build_chat_prompt."""
        # Seed a chat with 2 tasks
        chat_id = _allocate_chat_id()
        t1, chat_id = _add_task("first task", chat_id=chat_id)
        _save_task_result(result="first result", task_id=t1)
        t2, chat_id = _add_task("second task", chat_id=chat_id)
        _save_task_result(result="second result", task_id=t2)

        # Create a sub-agent with the same chat_id
        sub_agent = ChatSorcarAgent("sub")
        sub_agent.resume_chat_by_id(chat_id)

        # build_chat_prompt should include prior tasks
        prompt = sub_agent.build_chat_prompt("new task for sub-agent")
        assert "first task" in prompt
        assert "first result" in prompt
        assert "second task" in prompt
        assert "second result" in prompt
        assert "new task for sub-agent" in prompt

    def test_multiple_sub_agents_see_growing_context(self) -> None:
        """Each subsequent sub-agent sees all prior tasks in context."""
        chat_id = _allocate_chat_id()
        t1, chat_id = _add_task("task-a", chat_id=chat_id)
        _save_task_result(result="result-a", task_id=t1)

        # First sub-agent sees 1 prior task
        sub1 = ChatSorcarAgent("sub1")
        sub1.resume_chat_by_id(chat_id)
        prompt1 = sub1.build_chat_prompt("sub-task-1")
        assert "task-a" in prompt1
        assert "result-a" in prompt1

        # Add another task
        t2, chat_id = _add_task("task-b", chat_id=chat_id)
        _save_task_result(result="result-b", task_id=t2)

        # Second sub-agent sees 2 prior tasks
        sub2 = ChatSorcarAgent("sub2")
        sub2.resume_chat_by_id(chat_id)
        prompt2 = sub2.build_chat_prompt("sub-task-2")
        assert "task-a" in prompt2
        assert "task-b" in prompt2
        assert "result-a" in prompt2
        assert "result-b" in prompt2


class TestConcurrentThreadPoolExecutor:
    """Concurrent sub-agents via ThreadPoolExecutor with shared chat_id.

    These tests verify that the persistence module's per-thread SQLite
    connections and read-write lock correctly handle concurrent reads
    and writes from multiple agent threads.
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)
        self.srv, self.url = _start_server()

    def teardown_method(self) -> None:
        self.srv.shutdown()
        th._close_db()
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_concurrent_agents_accumulate_in_shared_chat(self) -> None:
        """Multiple agents run concurrently via ThreadPoolExecutor.

        All tasks must appear in the shared chat_id's history.
        """
        model_config = {"base_url": self.url, "api_key": "test-key"}

        # Parent establishes the chat session
        parent = ChatSorcarAgent("parent")
        parent.run(
            prompt_template="parent task",
            model_name="gpt-4o-mini",
            model_config=model_config,
            work_dir=self.tmpdir,
        )
        chat_id = parent.chat_id
        assert chat_id

        sub_tasks = [f"concurrent-task-{i}" for i in range(5)]

        def run_sub(task_name: str) -> str:
            # Random sleep < 0.1s to surface race conditions
            time.sleep(random.uniform(0.01, 0.08))
            agent = ChatSorcarAgent(f"P-{task_name}")
            agent.resume_chat_by_id(chat_id)
            return agent.run(
                prompt_template=task_name,
                model_name="gpt-4o-mini",
                model_config=model_config,
                work_dir=self.tmpdir,
            )

        with ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(run_sub, sub_tasks))

        # All sub-agents should have returned results
        assert len(results) == 5

        # All tasks (parent + 5 concurrent) in the same chat
        context = _load_chat_context(chat_id)
        assert len(context) == 6  # 1 parent + 5 sub-tasks
        recorded_tasks = {e["task"] for e in context}
        assert "parent task" in recorded_tasks
        for st in sub_tasks:
            assert st in recorded_tasks
        # All results should be "done"
        for entry in context:
            assert entry["result"] == "done"

    def test_concurrent_db_writes_no_errors(self) -> None:
        """Direct concurrent writes to the DB don't raise errors.

        Exercises the write lock under contention by adding tasks and
        saving results from multiple threads simultaneously.
        """
        chat_id = _allocate_chat_id()
        errors: list[str] = []

        def writer(idx: int) -> None:
            time.sleep(random.uniform(0.01, 0.05))
            try:
                task_id, _ = _retry_on_busy(
                    lambda: _add_task(
                        f"concurrent-write-{idx}", chat_id=chat_id,
                    )
                )
                time.sleep(random.uniform(0.005, 0.02))
                _retry_on_busy(
                    lambda: _save_task_result(
                        result=f"result-{idx}", task_id=task_id,
                    )
                )
            except Exception as exc:
                errors.append(f"writer-{idx}: {exc}")

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(writer, range(10)))

        assert not errors, f"Concurrent write errors: {errors}"

        context = _load_chat_context(chat_id)
        assert len(context) == 10
        results = {e["result"] for e in context}
        assert results == {f"result-{i}" for i in range(10)}

    def test_concurrent_reads_while_writing(self) -> None:
        """Concurrent reads don't fail while writes are happening.

        Spawns writer and reader threads simultaneously to verify the
        read-write lock allows safe concurrent access.
        """
        chat_id = _allocate_chat_id()
        read_errors: list[str] = []
        write_errors: list[str] = []

        def writer(idx: int) -> None:
            time.sleep(random.uniform(0.005, 0.03))
            try:
                task_id, _ = _retry_on_busy(
                    lambda: _add_task(
                        f"rw-task-{idx}", chat_id=chat_id,
                    )
                )
                _retry_on_busy(
                    lambda: _save_task_result(
                        result=f"rw-result-{idx}", task_id=task_id,
                    )
                )
            except Exception as exc:
                write_errors.append(f"writer-{idx}: {exc}")

        def reader(idx: int) -> None:
            time.sleep(random.uniform(0.005, 0.03))
            try:
                ctx = _retry_on_busy(lambda: _load_chat_context(chat_id))
                # Every entry that has been committed should have
                # consistent task/result pairs
                for entry in ctx:
                    assert entry["task"] is not None
            except Exception as exc:
                read_errors.append(f"reader-{idx}: {exc}")

        with ThreadPoolExecutor(max_workers=12) as pool:
            # 8 writers + 8 readers interleaved
            futures = []
            for i in range(8):
                futures.append(pool.submit(writer, i))
                futures.append(pool.submit(reader, i))
            for f in futures:
                f.result()

        assert not write_errors, f"Write errors: {write_errors}"
        assert not read_errors, f"Read errors: {read_errors}"

        context = _load_chat_context(chat_id)
        assert len(context) == 8

    def test_concurrent_agents_with_random_sleep(self) -> None:
        """Stress test: 8 concurrent agents with random delays.

        Each agent adds a random sleep before persisting to maximize
        the chance of interleaved DB operations.
        """
        model_config = {"base_url": self.url, "api_key": "test-key"}
        chat_id = _allocate_chat_id()
        # Seed one task so the chat exists
        t0, chat_id = _add_task("seed-task", chat_id=chat_id)
        _save_task_result(result="seed-result", task_id=t0)

        task_names = [f"stress-{i}" for i in range(8)]
        errors: list[str] = []

        def run_agent(name: str) -> None:
            time.sleep(random.uniform(0.01, 0.08))
            try:
                agent = ChatSorcarAgent(f"S-{name}")
                agent.resume_chat_by_id(chat_id)
                agent.run(
                    prompt_template=name,
                    model_name="gpt-4o-mini",
                    model_config=model_config,
                    work_dir=self.tmpdir,
                )
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(run_agent, task_names))

        assert not errors, f"Agent errors: {errors}"

        context = _load_chat_context(chat_id)
        # seed-task + 8 stress tasks
        assert len(context) == 9
        recorded = {e["task"] for e in context}
        assert "seed-task" in recorded
        for tn in task_names:
            assert tn in recorded


# ---------------------------------------------------------------------------
# Regression: ``_run_tasks_parallel`` must reject bare-string ``tasks``
# instead of iterating it character-by-character and creating one tab
# per character.  The old ``ChatSorcarAgent._run_tasks_parallel`` override
# lacked the coercion guard that ``run_tasks_parallel`` has.
# ---------------------------------------------------------------------------


class _CapturePrinter(BaseBrowserPrinter):
    """``BaseBrowserPrinter`` that records every broadcast event."""

    def __init__(self) -> None:
        super().__init__()
        self.captured: list[dict[str, Any]] = []
        self._capture_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        with self._capture_lock:
            self.captured.append(event)
        super().broadcast(event)


class TestBareStringTasksDoesNotSpawnPerCharacterTabs:
    """Passing ``tasks`` as a bare string must not create one tab per character.

    Reproduces the bug where ``ChatSorcarAgent._run_tasks_parallel`` (and
    ``SorcarAgent._run_tasks_parallel``) accepted ``tasks="hello"`` and
    iterated ``enumerate("hello")`` → 5 ``openSubagentTab`` events.
    """

    def _count_open_events(self, printer: _CapturePrinter) -> int:
        return sum(
            1 for e in printer.captured if e.get("type") == "openSubagentTab"
        )

    def test_chat_sorcar_string_tasks_yields_single_tab(self) -> None:
        """``ChatSorcarAgent`` coerces a bare string into one task."""
        printer = _CapturePrinter()
        printer._thread_local.tab_id = "parent-chat"
        agent = ChatSorcarAgent("regression-chat")
        agent.printer = printer

        # ``max_workers=0`` makes ``ThreadPoolExecutor`` raise ``ValueError``
        # AFTER the broadcast loop runs, so we still observe whatever tabs
        # the (buggy) broadcast loop opened.
        with pytest.raises(ValueError):
            agent._run_tasks_parallel("hello", max_workers=0)  # type: ignore[arg-type]

        opened = self._count_open_events(printer)
        assert opened == 1, (
            f"Expected exactly 1 openSubagentTab for a bare-string task, "
            f"got {opened} (would be len('hello')=5 if iterated per-char)"
        )

    def test_chat_sorcar_invalid_tasks_raises_typeerror(self) -> None:
        """Non-string, non-list-of-str inputs raise ``TypeError``."""
        printer = _CapturePrinter()
        printer._thread_local.tab_id = "parent-chat-bad"
        agent = ChatSorcarAgent("regression-chat-bad")
        agent.printer = printer

        with pytest.raises(TypeError):
            agent._run_tasks_parallel([1, 2, 3], max_workers=0)  # type: ignore[list-item]
        assert self._count_open_events(printer) == 0

    def test_sorcar_string_tasks_yields_single_tab(self) -> None:
        """``SorcarAgent`` base method also coerces a bare string."""
        printer = _CapturePrinter()
        printer._thread_local.tab_id = "parent-base"
        agent = SorcarAgent("regression-base")
        agent.printer = printer

        with pytest.raises(ValueError):
            agent._run_tasks_parallel("world!", max_workers=0)  # type: ignore[arg-type]

        opened = self._count_open_events(printer)
        assert opened == 1, (
            f"Expected 1 openSubagentTab for bare-string task, got {opened}"
        )
