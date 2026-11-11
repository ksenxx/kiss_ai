# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A task whose daemon is killed comes back under the same task id.

The first two tests are the whole path.  A child daemon
(``remote_daemon``) runs a task against the stand-in model: the script
creates a worktree, writes a file there and starts a long ``Bash``
call.  The daemon is killed with SIGKILL (an OOM, a lost node) or
stopped with SIGTERM (a pod replacement, a daemon restart) while that
call runs, and a new daemon starts on the same ``KISS_HOME``.  Its
orphan sweep must find the task's resume
state under ``<KISS_HOME>/resume/<task_id>``, leave the row open, and
start the task again under the same id: the model's next request
carries the "interrupted, outcome unknown" result and the note naming
the Bash call, the task takes its worktree back, merges it and
finishes.  One ``task_history`` row, with the final result; a
``task_resumed`` event and a banner in its transcript; no resume
state left behind.  In between, a reclaim run from this process must
leave the dead task's worktree alone (its owner task is waiting to be
resumed), which is the Phase 1 ``kiss-owner-task`` guard.

The other tests boot an in-process ``AgentServer`` on homes holding
resume state the sweep must refuse or discard.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from unittest import TestCase

from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.resume_state import (
    MAX_RESUMES,
    read_launch,
    resume_dir,
    write_launch,
)
from kiss.tests.agents.sorcar.test_orphan_task_recovery import (
    _disown,
    _insert_sentinel_row,
    _row_result,
)
from kiss.tests.local_ws import open_local_connection
from kiss.tests.remote_daemon import start_remote_daemon
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    OfflineFastModel,
    StandInModelServer,
    finish_response,
    tool_call_response,
)

_CONFIG = {"max_budget": 5.0, "use_web_browser": False, "classify_tasks": False}


def _run_command(work_dir: str, model_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "run",
        "prompt": "change the file, then wait",
        "workDir": work_dir,
        "tabId": "tab-before-the-kill",
        "model": STANDIN_MODEL,
        "modelConfig": model_config,
        "isParallel": False,
        "useWebTools": False,
        "classifyTasks": False,
        "useMemory": False,
        "maxBudget": 5.0,
    }


async def _submit(home: Path, cmd: dict[str, Any]) -> None:
    reader, writer = await open_local_connection(home)
    writer.write((json.dumps(cmd) + "\n").encode())
    await writer.drain()
    # Let the daemon take the command before the connection drops.
    await asyncio.wait_for(reader.readline(), timeout=30)
    writer.close()


def _pending_checkpoint(home: Path, timeout: float = 90.0) -> str:
    """Wait for a checkpoint whose first pending call is the blocking Bash; return its task id."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for directory in (home / "resume").glob("*"):
            checkpoint = directory / "checkpoint-0.json"
            if checkpoint.exists():
                state = json.loads(checkpoint.read_text())
                if state["calls"] and not state["done"] and state["calls"][0]["name"] == "Bash":
                    return directory.name
        time.sleep(0.1)
    raise AssertionError("no checkpoint with the blocking Bash call appeared")


def _row(task_id: str) -> tuple[str, int | None]:
    db = persistence._get_db()
    row = db.execute(
        "SELECT result, success FROM task_history WHERE id = ?", (task_id,),
    ).fetchone()
    assert row is not None
    return str(row["result"]), row["success"]


def _event_types(task_id: str) -> list[str]:
    db = persistence._get_db()
    rows = db.execute(
        "SELECT event_json FROM events WHERE task_id = ? ORDER BY seq", (task_id,),
    ).fetchall()
    return [str(json.loads(r["event_json"]).get("type")) for r in rows]


class _Script:
    """create_worktree, write a file and block, then (resumed) merge and finish.

    The blocking process is a Python sleep whose command line carries
    *marker*: it outlives the killed daemon (a killed process does not
    take its children along), so the test kills it by that marker.
    """

    def __init__(self, marker: str) -> None:
        self.requests: list[dict[str, Any]] = []
        self.marker = marker

    def respond(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        turn = len(self.requests)
        if turn == 1:
            return tool_call_response("create_worktree", {})
        if turn == 2:
            return tool_call_response("Bash", {
                "command": (
                    "printf 'resumed work\\n' > resumed.txt && "
                    f"{sys.executable} -c 'import time; time.sleep(300)' {self.marker}"
                ),
                "description": "write and wait",
            })
        if turn == 3:
            return tool_call_response("commit_and_merge", {})
        return finish_response("merged after the restart")


class TestTaskResumesAfterDaemonKill(TestCase):
    """Kill or stop the daemon mid tool call; the next daemon finishes the task."""

    def setUp(self) -> None:
        self.home = IsolatedKissHome(prefix="kiss-daemon-resume-")
        self.repo = self.home.repo
        self.offline = OfflineFastModel()
        self.offline.__enter__()
        self.script = _Script(f"kiss-resume-test-{uuid.uuid4().hex}")
        self.standin = StandInModelServer(self.script.respond)
        self.daemons: list[Any] = []
        self.env = {"HOME": str(self.home.tmpdir), "KISS_WORKDIR": str(self.repo)}

    def tearDown(self) -> None:
        for daemon in self.daemons:
            daemon.stop()
        subprocess.run(["pkill", "-f", self.script.marker], check=False)
        self.standin.stop()
        self.offline.__exit__(None, None, None)
        self.home.cleanup()

    def _start_daemon(self, port: int = 0) -> Any:
        daemon = start_remote_daemon(
            self.home.kiss_home, config=_CONFIG, env=self.env, port=port,
        )
        self.daemons.append(daemon)
        return daemon

    def test_task_killed_with_its_daemon_finishes_under_the_same_row(self) -> None:
        self._restart_mid_call(signal.SIGKILL, "Agent Failed Abruptly")

    def test_task_stopped_by_a_graceful_shutdown_finishes_under_the_same_row(self) -> None:
        # SIGTERM is what a pod replacement or a daemon restart sends: the
        # daemon stops the task and writes the shutdown label on its row,
        # and the task must come back all the same.
        self._restart_mid_call(signal.SIGTERM, persistence.SHUTDOWN_INTERRUPTED_RESULT)

    def _restart_mid_call(self, sig: int, result_before_restart: str) -> None:
        """Run the scenario of the module docstring, ending the first daemon with *sig*.

        Args:
            sig: The signal sent to the first daemon while the Bash call runs.
            result_before_restart: The row's result once that daemon is gone.
        """
        first = self._start_daemon()
        asyncio.run(_submit(
            self.home.kiss_home, _run_command(str(self.repo), self.standin.model_config),
        ))
        task_id = _pending_checkpoint(self.home.kiss_home)
        launch = read_launch(task_id)
        assert launch is not None and launch["resume_count"] == 0
        assert launch["cmd"]["prompt"] == "change the file, then wait"
        assert "_launch" not in launch["cmd"] and "_state_key" not in launch["cmd"]
        worktrees = json.loads((resume_dir(task_id) / "worktrees.json").read_text())
        assert len(worktrees) == 1 and worktrees[0]["entered"] is True
        branch = worktrees[0]["branch"]
        assert GitWorktreeOps.load_owner_task(self.repo, branch) == task_id
        # The checkpoint precedes the call: kill once the call's side
        # effect is on disk, the case the resumed model must check for.
        written = Path(worktrees[0]["work_dir"]) / "resumed.txt"
        deadline = time.monotonic() + 60
        while not written.exists():
            assert time.monotonic() < deadline, "the Bash call never wrote its file"
            time.sleep(0.05)

        os.kill(first.pid, sig)
        first.process.wait(timeout=60)
        assert _row(task_id)[0] == result_before_restart
        assert resume_dir(task_id).is_dir()

        # Phase 1: another process's reclaim must not merge the dead
        # task's worktree while the task is waiting to be resumed.
        assert GitWorktreeOps.reclaim_orphaned_worktrees(self.repo) == 0
        assert branch in {b for _d, b in GitWorktreeOps.registered_worktrees(self.repo)}
        assert not (self.repo / "resumed.txt").exists()

        second = self._start_daemon(port=first.port)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and _row(task_id)[0] in (
            result_before_restart, "Agent Failed Abruptly",
        ):
            time.sleep(0.2)
        result, success = _row(task_id)
        assert "merged after the restart" in result, result
        assert success == 1
        assert second.process.poll() is None, "the second daemon died"

        db = persistence._get_db()
        assert db.execute("SELECT COUNT(*) AS n FROM task_history").fetchone()["n"] == 1
        types = _event_types(task_id)
        assert "task_resumed" in types and "warning" in types, types
        assert types[-1] == "task_done"
        assert types.index("task_resumed") > types.index("tool_call")

        resumed = self.script.requests[2]["messages"]
        interrupted = [m for m in resumed if m.get("role") == "tool"]
        assert interrupted[-1]["content"].startswith("Interrupted: the process running this task")
        assert "printf 'resumed work" in resumed[-1]["content"]
        assert len(self.script.requests) == 4

        # The resumed task took its worktree back and merged it.
        merge_result = self.script.requests[3]["messages"][-1]["content"]
        assert (self.repo / "resumed.txt").exists(), merge_result
        assert (self.repo / "resumed.txt").read_text() == "resumed work\n"
        assert branch not in {b for _d, b in GitWorktreeOps.registered_worktrees(self.repo)}
        assert not resume_dir(task_id).exists()
        assert read_launch(task_id) is None


def _boot_server(printer: Any = None) -> Any:
    """Boot an in-process ``AgentServer`` and wait for its sweep thread."""
    from kiss.server.server import AgentServer

    server = AgentServer(printer=printer) if printer is not None else AgentServer()
    thread = server._orphan_sweep_thread
    assert thread is not None
    thread.join(timeout=60)
    assert not thread.is_alive(), "orphan sweep did not finish"
    return server


class TestResumeRefusals(TestCase):
    """Rows the boot sweep must not resume, and what it does with them."""

    def setUp(self) -> None:
        self.home = IsolatedKissHome(prefix="kiss-resume-refusal-")
        self.home.write_config(**_CONFIG)
        os.environ.setdefault("KISS_WORKDIR", str(self.home.repo))

    def tearDown(self) -> None:
        self.home.cleanup()

    def _dead_task(self, **launch_kwargs: Any) -> str:
        task_id = _insert_sentinel_row("interrupted task")
        _disown(task_id)
        write_launch(task_id, {"type": "run", "prompt": "interrupted task"}, **launch_kwargs)
        return task_id

    def test_restart_cap_closes_the_row_with_the_reason(self) -> None:
        task_id = self._dead_task(resume_count=MAX_RESUMES)
        _boot_server()
        assert _row_result(task_id) == (
            f"Not resumed after a restart: it was stopped by {MAX_RESUMES} restarts in a row"
        )
        assert not resume_dir(task_id).exists()

    def test_old_resume_state_is_refused(self) -> None:
        task_id = self._dead_task()
        old = time.time() - 7 * 3600
        for path in resume_dir(task_id).iterdir():
            os.utime(path, (old, old))
        _boot_server()
        assert _row_result(task_id).startswith("Not resumed after a restart: its last activity was")
        assert not resume_dir(task_id).exists()

    def test_setting_off_refuses_every_resume(self) -> None:
        self.home.write_config(resume_on_restart=False)
        task_id = self._dead_task()
        _boot_server()
        assert _row_result(task_id) == (
            "Not resumed after a restart: the resume_on_restart setting is off"
        )

    def test_finished_task_only_loses_its_stale_state(self) -> None:
        task_id = self._dead_task()
        persistence._save_task_result("<p>done before the restart</p>", task_id=task_id)
        _boot_server()
        assert _row_result(task_id) == "<p>done before the restart</p>"
        assert not resume_dir(task_id).exists()

    def test_reopen_takes_only_a_dead_sentinel_row(self) -> None:
        live_id = _insert_sentinel_row("still running elsewhere")
        assert persistence._reopen_task(live_id) is None
        assert persistence._resume_row_status(live_id) == "live"
        dead_id = self._dead_task()
        assert persistence._resume_row_status(dead_id) == "dead"
        assert persistence._reopen_task(dead_id) == "orphan-chat"
        assert persistence._resume_row_status(dead_id) == "live"
        assert persistence._reopen_task("no-such-row") is None
        assert persistence._resume_row_status("no-such-row") == "missing"
        # A row the daemon closed on its way down is resumable too, and
        # reopening it makes it a running row again (the sentinel result).
        stopped_id = self._dead_task()
        persistence._save_task_result(
            persistence.SHUTDOWN_INTERRUPTED_RESULT, task_id=stopped_id,
        )
        assert persistence._resume_row_status(stopped_id) == "dead"
        assert persistence._reopen_task(stopped_id) == "orphan-chat"
        assert _row_result(stopped_id) == "Agent Failed Abruptly"
        assert persistence._resume_row_status(stopped_id) == "live"


class TestFollowUpOfAResumedRun(TestCase):
    """A message typed while a resumed task ran becomes a task of its own."""

    def setUp(self) -> None:
        self.home = IsolatedKissHome(prefix="kiss-resume-followup-")
        self.home.write_config(**_CONFIG)
        os.environ.setdefault("KISS_WORKDIR", str(self.home.repo))
        self.standin = StandInModelServer(lambda request: finish_response("done"))

    def tearDown(self) -> None:
        self.standin.stop()
        self.home.cleanup()

    def _wait_for_end(self, printer: CapturePrinter, tab_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            for event in printer.events_of_type("task_done") + printer.events_of_type("task_error"):
                if event.get("tabId") == tab_id:
                    return event
            time.sleep(0.1)
        raise AssertionError(f"no end event on tab {tab_id}")

    def test_follow_up_is_redispatched_without_the_resume_marker(self) -> None:
        printer = CapturePrinter()
        server = _boot_server(printer)
        server.work_dir = str(self.home.repo)
        base = {
            "type": "run", "model": STANDIN_MODEL, "workDir": str(self.home.repo),
            "isParallel": False, "useWebTools": False, "maxBudget": 5.0,
            "modelConfig": self.standin.model_config,
        }
        server._cmd_run(dict(base, tabId="tab-a", prompt="the resumed task"))
        assert self._wait_for_end(printer, "tab-a")["type"] == "task_done"
        _join_task_threads()
        persistence._flush_chat_events()
        db = persistence._get_db()
        task_id = db.execute(
            "SELECT id FROM task_history WHERE task = ?", ("the resumed task",),
        ).fetchone()["id"]

        # What the runner does when the resumed run ends with a queued
        # message: re-dispatch the submitted command with the new prompt.
        resumed_cmd = dict(base, prompt="the resumed task", resumeTaskId=task_id)
        server._redispatch_leftover_prompts(
            dict(resumed_cmd, tabId=f"resume-{task_id}"), [("a follow-up the user typed", [])],
        )
        end = self._wait_for_end(printer, f"resume-{task_id}")
        _join_task_threads()
        persistence._flush_chat_events()
        assert end["type"] == "task_done", end
        rows = db.execute(
            "SELECT id, result FROM task_history WHERE task = ?", ("a follow-up the user typed",),
        ).fetchall()
        assert len(rows) == 1 and rows[0]["id"] != task_id
        assert "done" in str(rows[0]["result"])
        assert _row_result(task_id) != "Agent Failed Abruptly"


def _join_task_threads() -> None:
    from kiss.server import agent_state

    with agent_state.STATE_LOCK:
        threads = [s.task_thread for s in agent_state.agent_states.values() if s.task_thread]
    for thread in threads:
        thread.join(timeout=60)
