# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for ChatSorcarAgent: worktree lifecycle, blocking, crash recovery."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import FrameType
from typing import Any, cast

import pytest
from websockets.exceptions import ConnectionClosed

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar import agent_dispatch, cron_agent
from kiss.agents.sorcar.agent_dispatch import (
    DEFAULT_AGENT_PATH,
    live_agent_jobs,
    make_run_agent_tool,
)
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import (
    GitWorktreeOps,
    _git,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.tests.agents.third_party_agents.recording_daemon import RecordingDaemon


def _redirect_db(tmpdir: str) -> tuple:
    """Redirect persistence DB to a temp dir."""
    old = (th._DB_PATH, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    return old


def _restore_db(saved: tuple) -> None:
    th._close_thread_db()
    (th._DB_PATH, th._KISS_DIR) = saved


def _make_repo(path: Path) -> Path:
    """Create a minimal git repo with one commit and return its path."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@test.com"],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        capture_output=True, check=True,
    )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "-C", str(path), "add", "."], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True, check=True,
    )
    return path


def _patch_super_run(return_value: str = "success: true\nsummary: test done\n") -> Any:
    """Monkey-patch RelentlessAgent.run to skip the LLM call.

    The fake task does what a real one does: it resolves ``work_dir``
    (``RelentlessAgent._reset``) and calls the agent's
    ``create_worktree`` tool before its first edit.
    """
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original = parent_class.run

    def fake_run(self_agent: Any, **kwargs: object) -> str:
        self_agent.work_dir = str(Path(str(kwargs.get("work_dir") or ".")).resolve())
        self_agent.create_worktree()
        return return_value

    parent_class.run = fake_run
    return original


def _unpatch_super_run(original: Any) -> None:
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    parent_class.run = original


class TestChatSorcarAgent:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.original_run = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.original_run)
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _agent(self, chat_id: str | None = None) -> ChatSorcarAgent:
        agent = ChatSorcarAgent("test")
        if chat_id:
            agent.resume_chat_by_id(chat_id)
        return agent


    def test_consecutive_tasks_same_session(self) -> None:
        """The next task retires a worktree still pending and gets a fresh one.

        The retired worktree changed nothing, so its branch is cleaned
        up rather than parked (a dirty one is committed and kept).
        """
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        first_branch = agent._wt_branch
        assert first_branch is not None
        assert GitWorktreeOps.load_preserve_marker(self.repo, first_branch)

        result = agent.run(prompt_template="task2", work_dir=str(self.repo))
        assert "test done" in result
        assert agent._wt_branch not in (None, first_branch)
        assert not GitWorktreeOps.branch_exists(self.repo, first_branch)
        agent.discard()


    def test_not_a_git_repo(self) -> None:
        no_repo = Path(self.tmpdir) / "no_repo"
        no_repo.mkdir()
        agent = self._agent()
        result = agent.run(prompt_template="task1", work_dir=str(no_repo))
        assert "test done" in result
        assert agent._wt_branch is None


    def test_idempotent_merge(self) -> None:
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        msg1 = agent.merge()
        assert "Successfully merged" in msg1
        with pytest.raises(RuntimeError, match="No pending"):
            agent.merge()

    def test_idempotent_discard(self) -> None:
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        msg1 = agent.discard()
        assert "Discarded" in msg1
        with pytest.raises(RuntimeError, match="No pending"):
            agent.discard()


    def test_super_run_raises(self) -> None:
        _unpatch_super_run(self.original_run)
        parent_class = cast(Any, SorcarAgent.__mro__[1])
        orig = parent_class.run

        def raising_run(self_agent: Any, **kwargs: object) -> str:
            self_agent.work_dir = str(self.repo)
            self_agent.create_worktree()
            raise RuntimeError("LLM crashed")

        parent_class.run = raising_run
        try:
            agent = self._agent()
            result = agent.run(prompt_template="task1", work_dir=str(self.repo))
            assert "Task failed" in result
            assert agent._wt_pending
            agent.discard()
        finally:
            parent_class.run = orig
            self.original_run = _patch_super_run()


    def test_empty_repo(self) -> None:
        empty = Path(self.tmpdir) / "empty_repo"
        empty.mkdir()
        subprocess.run(["git", "init", str(empty)], capture_output=True, check=True)
        agent = self._agent()
        result = agent.run(prompt_template="task1", work_dir=str(empty))
        assert "test done" in result
        assert agent._wt_branch is None

    def test_git_with_cwd(self) -> None:
        result = _git("--version", cwd=self.repo)
        assert result.returncode == 0
        assert "git version" in result.stdout

    def test_ensure_excluded_no_file(self) -> None:
        exclude_file = self.repo / ".git" / "info" / "exclude"
        if exclude_file.exists():
            exclude_file.unlink()
        if exclude_file.parent.exists():
            exclude_file.parent.rmdir()
        GitWorktreeOps.ensure_excluded(self.repo)
        assert exclude_file.exists()
        assert ".kiss-worktrees/" in exclude_file.read_text()

    def test_worktree_add_failure(self) -> None:
        agent = self._agent()
        wt_base = self.repo / ".kiss-worktrees"
        wt_base.mkdir(exist_ok=True)
        blocker = wt_base / f"kiss_wt-{agent.chat_id}-99999999999"
        blocker.write_text("blocker")
        blocker.unlink()

        import time as t
        ts = int(t.time())
        branch_name = f"kiss/wt-{agent.chat_id}-{ts}"
        slug = branch_name.replace("/", "_")
        target = wt_base / slug
        target.mkdir(parents=True, exist_ok=True)
        (target / "blocker").write_text("x")

        result = agent.run(prompt_template="task1", work_dir=str(self.repo))
        assert "test done" in result
        shutil.rmtree(wt_base, ignore_errors=True)

    def test_run_without_work_dir(self) -> None:
        import os
        old_cwd = os.getcwd()
        os.chdir(str(self.repo))
        try:
            agent = self._agent()
            result = agent.run(prompt_template="task1")
            assert "test done" in result
            if agent._wt_pending:
                agent.discard()
        finally:
            os.chdir(old_cwd)

    def test_cleanup_partial_worktree_no_dir(self) -> None:
        branch = "kiss/wt-nocleanup"
        _git("branch", branch, cwd=self.repo)
        nonexistent = self.repo / ".kiss-worktrees" / "nonexistent"
        GitWorktreeOps.cleanup_partial(self.repo, branch, nonexistent)
        check = _git("rev-parse", "--verify", f"refs/heads/{branch}",
                      cwd=self.repo)
        assert check.returncode != 0


    def test_new_chat_preserves_pending_worktree(self) -> None:
        """new_chat() keeps the pending worktree's branch for review.

        Simulates the chat page's "New Chat" button being clicked while
        worktree action buttons are visible.  The agent commits the
        work onto its ``kiss/wt-*`` branch, removes the directory and
        never merges into the user's branch behind their back.
        """
        agent = self._agent()

        agent.run(prompt_template="task1", work_dir=str(self.repo))
        old_branch = agent._wt_branch
        old_wt_dir = agent._wt_dir
        assert old_branch is not None
        assert old_wt_dir is not None
        assert old_wt_dir.exists()

        agent.new_chat()
        assert agent._chat_id == ""
        assert agent._wt is None
        assert not old_wt_dir.exists()

        check = _git(
            "rev-parse", "--verify", f"refs/heads/{old_branch}",
            cwd=self.repo,
        )
        assert check.returncode == 0, (
            "Old branch must be kept for review, never auto-merged"
        )
        assert GitWorktreeOps.load_preserve_marker(self.repo, old_branch)

        agent.run(prompt_template="task2", work_dir=str(self.repo))
        assert agent._wt_branch is not None
        assert agent._wt_branch != old_branch
        agent.discard()

    def test_retire_then_run_preserves_previous_worktree(self) -> None:
        """A run() after the caller retired the pending worktree gets a
        fresh branch; the old one is committed and kept, not merged."""
        agent = self._agent()

        agent.run(prompt_template="task1", work_dir=str(self.repo))
        old_branch = agent._wt_branch
        old_wt_dir = agent._wt_dir
        assert old_branch is not None
        assert old_wt_dir is not None
        assert old_wt_dir.exists()
        (old_wt_dir / "new.txt").write_text("work\n")

        import time as t
        t.sleep(1.1)

        agent._retire_previous_worktree()
        assert agent._wt is None
        agent.run(prompt_template="task2", work_dir=str(self.repo))
        new_branch = agent._wt_branch
        assert new_branch is not None
        assert new_branch != old_branch

        assert not old_wt_dir.exists()
        check = _git(
            "rev-parse", "--verify", f"refs/heads/{old_branch}",
            cwd=self.repo,
        )
        assert check.returncode == 0
        shown = _git("show", f"{old_branch}:new.txt", cwd=self.repo)
        assert shown.stdout == "work\n"
        assert not (self.repo / "new.txt").exists()

        agent.discard()


_WT_PROPERTIES = ("_repo_root", "_wt_branch", "_wt_dir", "_wt_work_dir")


class _PausedPropertyReader(threading.Thread):
    """Reads one worktree property, pausing right after the property's first ``_wt`` lookup."""

    def __init__(self, agent: SorcarAgent, property_name: str) -> None:
        super().__init__(name=f"reader-{property_name}", daemon=True)
        self.agent = agent
        self.getter_code = getattr(SorcarAgent, property_name).fget.__code__
        self.property_name = property_name
        self.paused = threading.Event()
        self.release = threading.Event()
        self.outcome: object = None

    def _trace(self, frame: FrameType, event: str, arg: object) -> Any:
        if (
            event == "return"
            and frame.f_code is SorcarAgent._wt.fget.__code__  # type: ignore[attr-defined]
            and arg is not None
            and frame.f_back is not None
            and frame.f_back.f_code is self.getter_code
            and not self.paused.is_set()
        ):
            self.paused.set()
            self.release.wait(30)
        return self._trace

    def run(self) -> None:
        sys.settrace(self._trace)
        try:
            self.outcome = getattr(self.agent, self.property_name)
        except Exception as exc:
            self.outcome = exc
        finally:
            sys.settrace(None)


def test_worktree_properties_survive_a_merge_between_lookup_and_dereference() -> None:
    """Each derived worktree property reads one ``_wt`` snapshot.

    The server's worktree action reads ``_repo_root`` on its own thread
    while the task thread may be merging: a property that looked ``_wt``
    up once for the check and again for the value raised
    ``AttributeError`` when the merge removed the entry in between
    (audit 2026-11-04, S2 F3).  Every reader here is paused between the
    two moments by a real clean ``commit_and_merge``.
    """
    tmpdir = tempfile.mkdtemp()
    agent = SorcarAgent("test-wt-snapshot")
    try:
        agent.work_dir = str(_make_repo(Path(tmpdir) / "repo"))
        assert agent.create_worktree().startswith("Created worktree")
        wt = agent._wt
        assert wt is not None
        expected = [wt.repo_root, wt.branch, wt.wt_dir, SorcarAgent._wt_work_dir_of(wt)]
        readers = [_PausedPropertyReader(agent, name) for name in _WT_PROPERTIES]
        for reader in readers:
            reader.start()
        for reader in readers:
            assert reader.paused.wait(30), f"{reader.property_name} never looked _wt up"
        assert "Successfully merged" in agent.commit_and_merge()
        assert agent._wt is None
        for reader in readers:
            reader.release.set()
            reader.join(30)
            assert not reader.is_alive()
        assert [reader.outcome for reader in readers] == expected
    finally:
        if agent._wt_pending:
            agent.discard()
        shutil.rmtree(tmpdir, ignore_errors=True)


class _HoldingDaemon(RecordingDaemon):
    """A daemon stand-in whose sub-task runs until :meth:`release`.

    The ``run_agent(wait="false")`` job of the parent stays live (its
    result unrecorded) for as long as the test wants: the daemon sends
    the running status, then the result only once released.
    """

    def __init__(self) -> None:
        self.running = threading.Event()
        self._release = threading.Event()
        super().__init__()

    async def _handle(self, ws: Any) -> None:
        try:
            command = json.loads(await ws.recv())
        except (ConnectionClosed, ValueError):
            return
        self.commands.append(command)
        tab_id = command.get("tabId", "")
        try:
            await ws.send(json.dumps({"type": "status", "running": True, "tabId": tab_id}))
            self.running.set()
            await asyncio.get_running_loop().run_in_executor(None, self._release.wait)
            for event in (self.result, {"type": "status", "running": False}):
                await ws.send(json.dumps({**event, "tabId": tab_id}))
            async for line in ws:
                self.commands.append(json.loads(line))
        except ConnectionClosed:
            pass

    def release(self) -> None:
        """Let the held sub-task finish."""
        self._release.set()


def test_commit_and_merge_is_refused_once_while_a_sub_agent_job_is_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live ``run_agent`` job holds the worktree: ``commit_and_merge`` is refused once.

    A sub-task inherits its caller's worktree and may still be writing
    there; a merge now commits without those writes and removes the
    directory under it (audit 2026-11-07, A13_astra-1).  The first
    ``commit_and_merge`` with live jobs is refused the way ``finish``
    is, naming the jobs; a second call passes so the model is never
    trapped, and once the job finished the merge goes through unasked.
    """
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(tmp_path / "no-daemon.json"))
    daemon = _HoldingDaemon()
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", str(daemon.endpoint_file))
    agent = SorcarAgent("test-merge-live-job")
    try:
        agent.work_dir = str(_make_repo(tmp_path / "repo"))
        assert agent.create_worktree().startswith("Created worktree")
        wt = agent._wt
        assert wt is not None
        run_agent = make_run_agent_tool(agent.work_dir, agent)
        notice = run_agent("child", DEFAULT_AGENT_PATH, wait="false", timeout="60")
        assert daemon.running.wait(30), notice
        (job,) = live_agent_jobs(agent)
        refusal = agent._guard_tool_call("commit_and_merge", {})
        assert isinstance(refusal, str), refusal
        assert refusal.startswith(
            "Error: commit_and_merge rejected — run_agent jobs are still running: "
        ), refusal
        assert f"{job.job_id} (sorcar, running " in refusal, refusal
        assert "agent_job(id, 'wait') collects a job's result and 'kill' stops it" in refusal
        assert agent._wt is wt and wt.wt_dir.is_dir()
        # One refusal per run: the second call passes with the job still live.
        assert agent._guard_tool_call("commit_and_merge", {}) is None
        # A finished job no longer holds the worktree.
        daemon.release()
        assert agent_dispatch.join_agent_job(job, 30), job.job_id
        assert live_agent_jobs(agent) == []
        assert "Successfully merged" in agent.commit_and_merge()
        assert agent._wt is None
    finally:
        daemon.release()
        agent_dispatch.kill_jobs_of(agent)
        if agent._wt_pending:
            agent.discard()
        daemon.close()



