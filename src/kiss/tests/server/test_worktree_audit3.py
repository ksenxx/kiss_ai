# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests verifying fixes for worktree bugs BUG-8 through BUG-11.

Each test verifies the CORRECT behavior after the fix was applied.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.git_worktree import _git
from kiss.agents.sorcar.persistence import _append_chat_event
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state


def _attach_worktree_state(
    agent: WorktreeSorcarAgent, tab_id: str = "0",
) -> agent_state.AgentState:
    """Register a server-owned worktree AgentState for *agent* on *tab_id*.

    Mirrors what the server does for a UI-launched worktree run: the
    state is keyed by a minted task id, carries the launching tab id,
    and has ``use_worktree=True`` so ``agent_state.find_by_tab`` used
    by the merge-flow methods resolves the agent.
    """
    st = agent_state.AgentState(
        uuid.uuid4().hex,
        agent=agent,
        tab_id=tab_id,
        server_owned=True,
    )
    st.use_worktree = True
    agent_state.register(st)
    return st


def _redirect_db(tmpdir: str) -> tuple:
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore_db(saved: tuple) -> None:
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@test.com"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        capture_output=True,
        check=True,
    )
    (path / "README.md").write_text("# Test\n")
    (path / "fileA.txt").write_text("original A\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True,
        check=True,
    )
    return path


def _patch_super_run(
    return_value: str = "success: true\nsummary: test done\n",
) -> Any:
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original = parent_class.run

    def fake_run(self_agent: object, **kwargs: object) -> str:
        return return_value

    parent_class.run = fake_run
    return original


def _unpatch_super_run(original: Any) -> None:
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    parent_class.run = original


def _make_server(repo: Path) -> tuple:
    from kiss.server.server import VSCodeServer

    server = VSCodeServer()
    events: list[dict] = []

    def capture(event: dict) -> None:
        events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    server.work_dir = str(repo)
    return server, events


class TestBug8Fix:
    """After fix, _get_worktree_changed_files only reports files the agent
    actually changed, even when the original branch has advanced.
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.original_run = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.original_run)
        agent_state.agent_states.clear()
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_changed_files_excludes_unrelated_after_main_advances(self) -> None:
        """Only agent-modified files appear, not unrelated files from main."""

        server, events = _make_server(self.repo)
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        _attach_worktree_state(agent)

        agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        assert wt_dir is not None and wt_dir.exists()

        (wt_dir / "fileA.txt").write_text("agent modified A\n")

        original_branch = agent._original_branch
        assert original_branch is not None

        tmp_wt = self.repo / ".kiss-worktrees" / "tmp_advance"
        _git("worktree", "add", "-b", "tmp-advance", str(tmp_wt), cwd=self.repo)
        (tmp_wt / "unrelated_file.txt").write_text("unrelated content\n")
        _git("add", "-A", cwd=tmp_wt)
        _git("commit", "-m", "advance with unrelated file", cwd=tmp_wt)
        _git("worktree", "remove", str(tmp_wt), "--force", cwd=self.repo)
        _git("checkout", original_branch, cwd=self.repo)
        _git("merge", "--ff-only", "tmp-advance", cwd=self.repo)
        _git("branch", "-d", "tmp-advance", cwd=self.repo)

        changed = server._get_worktree_changed_files("0")

        assert "fileA.txt" in changed
        assert "unrelated_file.txt" not in changed, (
            "BUG-8 FIX: unrelated files from main advancement "
            "should NOT appear as changed"
        )

        agent.discard()

    def test_changed_files_still_reports_agent_changes(self) -> None:
        """Sanity check: agent-modified files are still reported correctly."""

        server, events = _make_server(self.repo)
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        _attach_worktree_state(agent)

        agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        assert wt_dir is not None

        (wt_dir / "fileA.txt").write_text("agent modified A\n")
        (wt_dir / "new_file.txt").write_text("brand new\n")

        changed = server._get_worktree_changed_files("0")
        assert "fileA.txt" in changed
        assert "new_file.txt" in changed

        agent.discard()


class TestBug9Fix:
    """After fix, _check_merge_conflict does NOT commit worktree changes."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.original_run = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.original_run)
        agent_state.agent_states.clear()
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_check_conflict_does_not_commit(self) -> None:
        """_check_merge_conflict must not create any commits."""

        server, events = _make_server(self.repo)
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        _attach_worktree_state(agent)

        agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        branch = agent._wt_branch
        original = agent._original_branch
        assert wt_dir is not None and branch is not None and original is not None

        (wt_dir / "agent_output.txt").write_text("important work\n")

        r = subprocess.run(
            ["git", "-C", str(self.repo), "rev-list", "--count",
             f"{original}..{branch}"],
            capture_output=True, text=True,
        )
        assert r.stdout.strip() == "0", "No commits before check"

        server._check_merge_conflict("0")

        r = subprocess.run(
            ["git", "-C", str(self.repo), "rev-list", "--count",
             f"{original}..{branch}"],
            capture_output=True, text=True,
        )
        assert r.stdout.strip() == "0", (
            "BUG-9 FIX: _check_merge_conflict must not create commits"
        )

        agent.discard()

    def test_present_pending_worktree_does_not_commit(self) -> None:
        """_present_pending_worktree must not auto-commit via conflict check."""

        server, events = _make_server(self.repo)
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        _attach_worktree_state(agent)

        agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        branch = agent._wt_branch
        original = agent._original_branch
        assert wt_dir is not None and branch is not None and original is not None

        (wt_dir / "agent_output.txt").write_text("work\n")

        server._present_pending_worktree("0")

        r = subprocess.run(
            ["git", "-C", str(self.repo), "rev-list", "--count",
             f"{original}..{branch}"],
            capture_output=True, text=True,
        )
        assert r.stdout.strip() == "0", (
            "BUG-9 FIX: _present_pending_worktree must not auto-commit"
        )

        agent.discard()

    def test_conflict_detected_when_both_sides_modify_same_file(self) -> None:
        """Conflict is reported when the same file is changed on both sides."""

        server, events = _make_server(self.repo)
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        _attach_worktree_state(agent)

        agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        original = agent._original_branch
        assert wt_dir is not None and original is not None

        (wt_dir / "fileA.txt").write_text("agent version\n")

        tmp_wt = self.repo / ".kiss-worktrees" / "tmp_conflict"
        _git("worktree", "add", "-b", "tmp-conflict", str(tmp_wt), cwd=self.repo)
        (tmp_wt / "fileA.txt").write_text("main version\n")
        _git("add", "-A", cwd=tmp_wt)
        _git("commit", "-m", "conflicting change", cwd=tmp_wt)
        _git("worktree", "remove", str(tmp_wt), "--force", cwd=self.repo)
        _git("checkout", original, cwd=self.repo)
        _git("merge", "--ff-only", "tmp-conflict", cwd=self.repo)
        _git("branch", "-d", "tmp-conflict", cwd=self.repo)

        assert server._check_merge_conflict("0") is True

        agent.discard()

    def test_no_conflict_when_different_files_changed(self) -> None:
        """No conflict when original and worktree modify different files."""

        server, events = _make_server(self.repo)
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        _attach_worktree_state(agent)

        agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        original = agent._original_branch
        assert wt_dir is not None and original is not None

        (wt_dir / "fileA.txt").write_text("agent version\n")

        tmp_wt = self.repo / ".kiss-worktrees" / "tmp_noconflict"
        _git("worktree", "add", "-b", "tmp-noconflict", str(tmp_wt), cwd=self.repo)
        (tmp_wt / "other_file.txt").write_text("other content\n")
        _git("add", "-A", cwd=tmp_wt)
        _git("commit", "-m", "non-conflicting change", cwd=tmp_wt)
        _git("worktree", "remove", str(tmp_wt), "--force", cwd=self.repo)
        _git("checkout", original, cwd=self.repo)
        _git("merge", "--ff-only", "tmp-noconflict", cwd=self.repo)
        _git("branch", "-d", "tmp-noconflict", cwd=self.repo)

        assert server._check_merge_conflict("0") is False

        agent.discard()


class TestBug10Fix:
    """Replaying a session never flips a tab's worktree state on."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.original_run = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.original_run)
        agent_state.agent_states.clear()
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_replay_session_without_worktree_keeps_false(self) -> None:
        """When extra doesn't have is_worktree, use_worktree stays False."""

        server1, events1 = _make_server(self.repo)
        agent1 = WorktreeSorcarAgent("Sorcar VS Code")
        agent1.run(prompt_template="task1", work_dir=str(self.repo))
        chat_id = agent1.chat_id
        task_id = agent1._last_task_id
        assert task_id is not None

        _append_chat_event(
            {"type": "text_delta", "text": "working..."},
            task_id=task_id,
        )
        th._save_task_extra(
            {"is_worktree": False, "model": "test"},
            task_id=task_id,
        )

        server2, events2 = _make_server(self.repo)
        server2._replay_session(chat_id, "0")

        state = agent_state.find_by_tab("0")
        assert state is None or state.use_worktree is False



