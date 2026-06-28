# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt integration tests for task_runner.py / server.py / merge_flow.py.

BUG-1 (server.py ``_replay_session``): when a tab opens a chat whose
task is STILL RUNNING, the resumed-running ``status`` broadcast reads
``startTs`` from the persisted ``extra`` JSON — but ``startTs`` is only
written into ``extra`` at task END (``_save_task_extra`` in
``_run_task_inner``'s cleanup finally).  The early ``extra`` payload
written at task INSERT (``ChatSorcarAgent._build_extra_payload``) has
no ``startTs``, so every live resume broadcast carried ``startTs: 0``
and the frontend (which requires ``ev.startTs > 0``) anchored the
"Running …" timer at the client's ``Date.now()`` — showing "Running 0s"
for a task that had been running for minutes.  This contradicts the
method's own docstring, which claims the broadcast echoes the agent's
true start time.

BUG-2 (merge_flow.py ``_handle_autocommit_action``): the persisted
``autocommit_done`` event is appended under ``tab.last_task_id`` — but
when auto-commit mode is ON the handler runs from the task thread's
post-task cleanup (``_run_task_inner``'s finally) BEFORE ``_run_task``'s
outer finally refreshes ``last_task_id``.  For the second (and every
later) task on the same tab, ``last_task_id`` still holds the PREVIOUS
task's id, so the commit confirmation is persisted into the prior
task's event stream and never shows up when the current task is
replayed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as _persistence
import kiss.agents.vscode.merge_flow as _merge_flow_module
import kiss.agents.vscode.server as _server_module
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.vscode.server import VSCodeServer


def _run_git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )


def _init_repo(repo: str) -> None:
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "config", "commit.gpgsign", "false")
    Path(repo, "seed.txt").write_text("seed\n")
    _run_git(repo, "add", "seed.txt")
    _run_git(repo, "commit", "-q", "-m", "seed")


class _BugHuntBase(unittest.TestCase):
    """Shared scaffolding: stub the inner agent run, capture broadcasts."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt-")
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []

        def capture(event: dict[str, Any]) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

        # Patch out the two LLM-calling helpers (same precedent as
        # test_autocommit_persistence.py) so no paid API is hit.
        self._orig_gen = _merge_flow_module.generate_commit_message_from_diff
        self._orig_followup = _server_module.generate_followup_text

        def fake_commit_msg(
            diff_text: str, user_prompt: str | None = None,
        ) -> str:
            return "test: bughunt autocommit"

        def fake_followup(task: str, result: str, model: str) -> str:
            return ""

        _merge_flow_module.generate_commit_message_from_diff = (  # type: ignore[assignment]
            fake_commit_msg
        )
        _server_module.generate_followup_text = fake_followup  # type: ignore[assignment]

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run
        _merge_flow_module.generate_commit_message_from_diff = self._orig_gen
        _server_module.generate_followup_text = self._orig_followup
        _RunningAgentState.running_agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestResumeRunningStartTs(_BugHuntBase):
    """BUG-1: resuming a still-running task must broadcast a real startTs."""

    def test_replay_session_of_running_task_sends_nonzero_start_ts(self) -> None:
        work_dir = str(Path(self.tmpdir) / "plain")
        Path(work_dir).mkdir()
        release = threading.Event()

        def stub_run(self_agent: object, **kwargs: object) -> str:
            # Simulate a long-running agent: block until the test
            # releases us, keeping the task thread alive.
            release.wait(timeout=30)
            return "success: true\nsummary: long task done\n"

        self._parent_class.run = stub_run

        src_tab = "bughunt-src-tab"
        self.server._cmd_run({
            "type": "run",
            "prompt": "bughunt long-running task",
            "tabId": src_tab,
            "workDir": work_dir,
            "useWorktree": False,
            "autoCommit": False,
            "model": "",
        })
        try:
            # Wait until the run has allocated its task_history row.
            task_id: str | None = None
            deadline = time.time() + 30
            while time.time() < deadline:
                tab = _RunningAgentState.running_agent_states.get(src_tab)
                if (
                    tab is not None
                    and tab.agent is not None
                    and tab.agent._last_task_id is not None
                ):
                    task_id = tab.agent._last_task_id
                    break
                time.sleep(0.02)
            assert task_id is not None, "task row was never allocated"

            # A second tab opens the running task from history.
            self.events.clear()
            self.server._cmd_resume_session({
                "taskId": task_id,
                "tabId": "bughunt-viewer-tab",
            })

            status_events = [
                e for e in self.events
                if e.get("type") == "status"
                and e.get("running") is True
                and e.get("tabId") == "bughunt-viewer-tab"
            ]
            assert status_events, (
                "resume of a running task must broadcast a "
                "status running=True event to the viewer tab"
            )
            start_ts = status_events[-1].get("startTs", 0)
            assert isinstance(start_ts, int) and start_ts > 0, (
                "BUG: status broadcast for a resumed RUNNING task "
                f"carried startTs={start_ts!r} — the frontend ignores "
                "startTs <= 0 and mis-anchors the 'Running …' timer at "
                "the client's Date.now()"
            )
            # Sanity: the anchor is the agent's actual start time
            # (within the last minute), not some garbage value.
            now_ms = int(time.time() * 1000)
            assert now_ms - 60_000 <= start_ts <= now_ms, (
                f"startTs {start_ts} is not a plausible recent ms-epoch "
                f"timestamp (now={now_ms})"
            )
        finally:
            release.set()
            tab = _RunningAgentState.running_agent_states.get(src_tab)
            if tab is not None and tab.task_thread is not None:
                tab.task_thread.join(timeout=30)


class TestAutocommitDonePersistedToCurrentTask(_BugHuntBase):
    """BUG-2: in-task auto-commit must persist under the CURRENT task id."""

    def _events_for_task(self, task_id: str) -> list[dict[str, Any]]:
        _persistence._flush_chat_events()
        db = _persistence._get_db()
        rows = db.execute(
            "SELECT event_json FROM events WHERE task_id = ? ORDER BY seq",
            (task_id,),
        ).fetchall()
        return [json.loads(r["event_json"]) for r in rows]

    def test_second_task_autocommit_done_not_persisted_under_first_task(
        self,
    ) -> None:
        repo = str(Path(self.tmpdir) / "repo")
        Path(repo).mkdir()
        _init_repo(repo)
        self.server.work_dir = repo
        run_count = [0]

        def stub_run(self_agent: object, **kwargs: object) -> str:
            run_count[0] += 1
            Path(repo, "out.txt").write_text(f"change {run_count[0]}\n")
            return "success: true\nsummary: made a change\n"

        self._parent_class.run = stub_run

        tab_id = "bughunt-ac-tab"
        cmd = {
            "type": "run",
            "prompt": "bughunt autocommit task",
            "tabId": tab_id,
            "workDir": repo,
            "useWorktree": False,
            "autoCommit": True,
            "model": "",
        }
        self.server._run_task(dict(cmd))
        tab = _RunningAgentState.running_agent_states[tab_id]
        task1_id = tab.last_task_id
        assert task1_id is not None, "first task id missing"

        self.server._run_task(dict(cmd))
        task2_id = tab.last_task_id
        assert task2_id is not None and task2_id != task1_id, (
            f"second run did not allocate a new task row "
            f"(task1={task1_id}, task2={task2_id})"
        )

        task1_done = [
            e for e in self._events_for_task(task1_id)
            if e.get("type") == "autocommit_done"
        ]
        task2_done = [
            e for e in self._events_for_task(task2_id)
            if e.get("type") == "autocommit_done"
        ]
        assert len(task1_done) == 1, (
            f"BUG: task 1 has {len(task1_done)} persisted autocommit_done "
            "events — the second task's commit confirmation was appended "
            "to the FIRST task's event stream"
        )
        assert len(task2_done) == 1, (
            f"BUG: task 2 has {len(task2_done)} persisted autocommit_done "
            "events — its commit confirmation is missing from its own "
            "event stream (it was persisted under the previous task)"
        )


if __name__ == "__main__":
    unittest.main()
