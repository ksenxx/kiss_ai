# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``run_parallel`` is N ``run_agent`` calls with one ``timeout`` meaning.

A real daemon runs a parent whose scripted model calls ``run_parallel``.
Each child is a daemon sub-task of the parent (a history row with the
parent's ``parent_task_id``) that shares the parent's remaining budget
as remaining / (N + 1).  The call's ``timeout`` is how long the parent
blocks: when it expires no child is stopped — the parent gets one
``agent_job`` notice per unfinished child and ``agent_job(id, "wait")``
collects the result the child finishes with afterwards.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from kiss.server import sorcar
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    request_text,
    tool_call_response,
)

JOB_ID = re.compile(r"\bagent-[0-9a-f]{8}\b")


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    home = IsolatedKissHome(prefix="kiss-run-parallel-n-run-agent-")
    home.write_config(is_worktree=False, auto_commit_mode=False, classify_tasks=False)
    try:
        yield home
    finally:
        home.cleanup()


@pytest.fixture
def daemon(env: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    endpoint_file = str(env.tmpdir / "sorcar-local.json")
    monkeypatch.setenv("KISS_SORCAR_LOCAL", endpoint_file)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    server = RemoteAccessServer(local_endpoint_file=endpoint_file, work_dir=str(env.repo))
    asyncio.run_coroutine_threadsafe(server.start_private_async(), loop).result(timeout=30)
    try:
        yield endpoint_file
    finally:
        asyncio.run_coroutine_threadsafe(server.stop_async(), loop).result(timeout=15)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


def _task_rows(env: IsolatedKissHome) -> list[tuple[str, str, str]]:
    """Return ``(task_id, parent_task_id, task)`` of every persisted task."""
    with sqlite3.connect(env.kiss_home / "history.db") as db:
        return [
            (str(a), str(b or ""), str(c))
            for a, b, c in db.execute("SELECT id, parent_task_id, task FROM task_history")
        ]


def test_children_are_sub_tasks_sharing_the_budget_and_results_keep_task_order(
    env: IsolatedKissHome, daemon: str,
) -> None:
    parent_requests: list[dict[str, Any]] = []
    children: list[str] = []

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        text = request_text(request)
        for marker in ("KID-1", "KID-2", "KID-3"):
            if text.rstrip().endswith(f"{marker} say done"):
                children.append(marker)
                return finish_response(f"done-{marker}")
        parent_requests.append(request)
        if len(parent_requests) == 1:
            return tool_call_response(
                "run_parallel",
                {"tasks": '["KID-1 say done", "KID-2 say done", "KID-3 say done"]'},
            )
        return finish_response("parent-done")

    model = StandInModelServer(responder)
    try:
        result = sorcar.run(
            "PARENT-TASK fan out",
            work_dir=str(env.repo),
            model=STANDIN_MODEL,
            model_config=model.model_config,
            max_budget=8.0,
            use_worktree=False,
            auto_commit=False,
            endpoint_file=daemon,
            timeout=300,
        )
    finally:
        model.stop()
    assert result.success is True, result
    assert sorted(children) == ["KID-1", "KID-2", "KID-3"]
    fanout = request_text(parent_requests[1])
    # One entry per task, in task order, each a run_agent result.
    positions = [fanout.index(f"done-KID-{n}") for n in (1, 2, 3)]
    assert positions == sorted(positions), fanout[-1500:]
    assert fanout.count("ran: sorcar model=") == 3, fanout[-1500:]
    # The parent's remaining budget ($8, nothing spent on the stand-in)
    # shared among the three children plus the parent: 8 / 4.
    assert fanout.count("budget=$2.00") == 3, fanout[-1500:]
    assert fanout.count("inherited=model,max_budget,") == 3, fanout[-1500:]
    rows = _task_rows(env)
    parents = [row for row in rows if "PARENT-TASK" in row[2]]
    assert len(parents) == 1, rows
    parent_id = parents[0][0]
    kids = sorted(row[2][:5] for row in rows if row[1] == parent_id)
    assert kids == ["KID-1", "KID-2", "KID-3"], rows


def test_timeout_returns_job_notices_without_stopping_the_children(
    env: IsolatedKissHome, daemon: str,
) -> None:
    release = threading.Event()
    parent_requests: list[dict[str, Any]] = []
    finished: list[str] = []

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        text = request_text(request)
        for marker in ("KID-1", "KID-2"):
            if text.rstrip().endswith(f"{marker} wait"):
                # The child blocks past the parent's one-second wait and
                # finishes only once the parent has seen the job notices.
                assert release.wait(timeout=120)
                finished.append(marker)
                return finish_response(f"done-{marker}")
        parent_requests.append(request)
        step = len(parent_requests)
        if step == 1:
            return tool_call_response(
                "run_parallel", {"tasks": '["KID-1 wait", "KID-2 wait"]', "timeout": "1"},
            )
        jobs = sorted(set(JOB_ID.findall(request_text(parent_requests[1]))))
        if step == 2:
            release.set()
            return tool_call_response("agent_job", {"job_id": jobs[0], "action": "wait"})
        if step == 3:
            return tool_call_response("agent_job", {"job_id": jobs[1], "action": "wait"})
        return finish_response("parent-done")

    model = StandInModelServer(responder)
    try:
        result = sorcar.run(
            "PARENT-TASK fan out and wait",
            work_dir=str(env.repo),
            model=STANDIN_MODEL,
            model_config=model.model_config,
            use_worktree=False,
            auto_commit=False,
            endpoint_file=daemon,
            timeout=300,
        )
    finally:
        model.stop()
    assert result.success is True, result
    notices = request_text(parent_requests[1])
    assert len(set(JOB_ID.findall(notices))) == 2, notices[-1500:]
    assert "done-KID" not in notices, notices[-1500:]
    # Neither child was stopped at the timeout: both finished on their own
    # and agent_job collected each result.
    assert sorted(finished) == ["KID-1", "KID-2"]
    collected = request_text(parent_requests[2]) + request_text(parent_requests[3])
    assert "done-KID-1" in collected and "done-KID-2" in collected, collected[-2000:]
