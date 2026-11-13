# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end: a pending worktree whose changes git cannot enumerate is never auto-discarded.

``_present_pending_worktree`` auto-discards a pending worktree whose
changed-file listing is empty.  ``_changed_files_of`` answered ``[]``
both for a verified clean worktree and for a comparison git could not
make, so two real repository states lost work:

* the worktree directory was removed (branch-only pending work) and
  the original branch renamed: neither the base nor the branch diff
  resolved, the listing was ``[]`` and the still-existing task branch
  was force-deleted;
* the base commit object is missing while the index and files are
  readable: ``diff``, ``status`` and ``log`` all fail, and the
  directory holding a modified tracked file is deleted.

A failed query now keeps the worktree pending and presents it to the
user instead.  Real pieces throughout: real git repositories and
worktrees, the production ``AgentServer`` presentation and discard
paths, the real agent-state registry.
"""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import GitWorktree
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.conftest import seed_worktree
from kiss.tests.server.worktree_helpers import worktree_changed_files


@pytest.fixture(autouse=True)
def _clean_agent_registry() -> Any:
    """Drop any agent states a test registered, pass or fail."""
    yield
    agent_state.agent_states.clear()


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=check,
    )


def _repo_with_worktree(root: Path) -> GitWorktree:
    """A repository on ``main`` with one commit and a worktree on branch ``task-work``."""
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "tracked.txt").write_text("base\n")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "base")
    wt_dir = root / "worktree"
    _git(repo, "worktree", "add", "-b", "task-work", str(wt_dir))
    return GitWorktree(
        repo_root=repo, branch="task-work", original_branch="main",
        wt_dir=wt_dir, work_dir=wt_dir,
    )


def _server_with_pending(wt: GitWorktree, tab_id: str) -> tuple[AgentServer, list[dict[str, Any]]]:
    """A server whose *tab_id* holds the pending worktree *wt*; returns it with its events."""
    server = AgentServer()
    events: list[dict[str, Any]] = []
    server.printer.broadcast = events.append  # type: ignore[assignment]
    agent = ChatSorcarAgent("pending-worktree")
    seed_worktree(agent, wt)
    agent.work_dir = str(wt.work_dir)
    state = agent_state.AgentState(f"task-{tab_id}", agent=agent, tab_id=tab_id, server_owned=True)
    agent_state.register(state)
    return server, events


def _branch_exists(repo: Path, branch: str) -> bool:
    return _git(repo, "show-ref", "--verify", f"refs/heads/{branch}", check=False).returncode == 0


def test_branch_only_work_survives_a_renamed_base_branch(tmp_path: Path) -> None:
    """Removed directory + renamed original branch: the committed task branch is kept."""
    wt = _repo_with_worktree(tmp_path)
    (wt.wt_dir / "only-task-copy.txt").write_text("COMMITTED TASK WORK\n")
    _git(wt.wt_dir, "add", "only-task-copy.txt")
    _git(wt.wt_dir, "commit", "-m", "task work")
    _git(wt.repo_root, "worktree", "remove", str(wt.wt_dir))
    _git(wt.repo_root, "branch", "-m", "main", "renamed-main")
    shown = _git(wt.repo_root, "show", "task-work:only-task-copy.txt").stdout
    assert shown == "COMMITTED TASK WORK\n"
    server, events = _server_with_pending(wt, "branch-only-tab")
    agent = agent_state.find_by_tab("branch-only-tab").agent  # type: ignore[union-attr]
    assert agent is not None

    presented = server._present_pending_worktree("branch-only-tab")

    assert _branch_exists(wt.repo_root, "task-work"), "the task branch was deleted"
    assert agent.worktrees == [wt], "the worktree is no longer pending"
    assert presented is True
    done = [e for e in events if e["type"] == "worktree_done"]
    assert len(done) == 1
    assert done[0]["tabId"] == "branch-only-tab"
    # The committed work is listed from the branch's own commits.
    assert worktree_changed_files(server, "branch-only-tab") == ["only-task-copy.txt"]


def test_unreadable_history_keeps_the_worktree_and_its_modified_file(tmp_path: Path) -> None:
    """Missing base commit object: diff/status/log fail; the directory and its edit stay."""
    wt = _repo_with_worktree(tmp_path)
    (wt.wt_dir / "tracked.txt").write_text("UNIQUE UNCOMMITTED WORK\n")
    base = _git(wt.repo_root, "rev-parse", "HEAD").stdout.strip()
    blob = wt.repo_root / ".git" / "objects" / base[:2] / base[2:]
    blob.chmod(0o600)  # git stores objects read-only; Windows refuses to unlink such a file
    blob.unlink()
    for args in (
        ["diff", "--name-only", "-z", "main"],
        ["status", "--porcelain", "-z"],
        ["log", "-z", "--pretty=format:", "--name-only", "HEAD"],
    ):
        assert _git(wt.wt_dir, *args, check=False).returncode != 0, args
    assert _git(
        wt.wt_dir, "ls-files", "--others", "--ignored", "--exclude-standard", "-z", check=False,
    ).returncode == 0
    server, events = _server_with_pending(wt, "missing-object-tab")
    agent = agent_state.find_by_tab("missing-object-tab").agent  # type: ignore[union-attr]
    assert agent is not None

    presented = server._present_pending_worktree("missing-object-tab")

    assert wt.wt_dir.exists(), "the worktree directory was deleted"
    assert (wt.wt_dir / "tracked.txt").read_text() == "UNIQUE UNCOMMITTED WORK\n"
    assert agent.worktrees == [wt], "the worktree is no longer pending"
    assert presented is True
    done = [e for e in events if e["type"] == "worktree_done"]
    assert len(done) == 1 and done[0]["tabId"] == "missing-object-tab"
    # The post-task path (the runner discards itself when nothing was
    # presented) is told the worktree was presented too.
    assert server._present_pending_worktree("missing-object-tab", discard_if_empty=False) is True
    assert wt.wt_dir.exists()


def test_index_only_work_is_listed_and_kept(tmp_path: Path) -> None:
    """Staged text whose working file is back at the base (``MM``): listed, never discarded."""
    wt = _repo_with_worktree(tmp_path)
    (wt.wt_dir / "tracked.txt").write_text("UNIQUE STAGED WORK\n")
    _git(wt.wt_dir, "add", "tracked.txt")
    (wt.wt_dir / "tracked.txt").write_text("base\n")
    assert _git(wt.wt_dir, "status", "--porcelain").stdout == "MM tracked.txt\n"
    # The working tree equals the base: only the index holds the work.
    assert _git(wt.wt_dir, "diff", "--name-only", "main").stdout == ""
    server, events = _server_with_pending(wt, "staged-tab")
    agent = agent_state.find_by_tab("staged-tab").agent  # type: ignore[union-attr]
    assert agent is not None

    assert worktree_changed_files(server, "staged-tab") == ["tracked.txt"]
    # Post-task path (the runner discards when nothing was presented),
    # then the session-resume path (which discards itself).
    assert server._present_pending_worktree("staged-tab", discard_if_empty=False) is True
    presented = server._present_pending_worktree("staged-tab")

    assert presented is True
    assert wt.wt_dir.exists(), "the worktree directory was deleted"
    assert _git(wt.wt_dir, "show", ":tracked.txt").stdout == "UNIQUE STAGED WORK\n"
    assert _branch_exists(wt.repo_root, "task-work")
    assert agent.worktrees == [wt], "the worktree is no longer pending"
    done = [e for e in events if e["type"] == "worktree_done"]
    assert [e["tabId"] for e in done] == ["staged-tab", "staged-tab"]


def test_work_of_a_worktree_with_an_unknown_original_branch_is_kept(tmp_path: Path) -> None:
    """No original branch to compare with: the changes are unknown, the committed work stays."""
    known = _repo_with_worktree(tmp_path)
    (known.wt_dir / "only-task-copy.txt").write_text("COMMITTED TASK WORK\n")
    _git(known.wt_dir, "add", "only-task-copy.txt")
    _git(known.wt_dir, "commit", "-m", "task work")
    wt = dataclasses.replace(known, original_branch=None)
    server, events = _server_with_pending(wt, "unknown-base-tab")
    agent = agent_state.find_by_tab("unknown-base-tab").agent  # type: ignore[union-attr]
    assert agent is not None

    presented = server._present_pending_worktree("unknown-base-tab")

    assert (wt.wt_dir / "only-task-copy.txt").read_text() == "COMMITTED TASK WORK\n"
    assert _branch_exists(wt.repo_root, "task-work"), "the task branch was deleted"
    assert agent.worktrees == [wt], "the worktree is no longer pending"
    assert presented is True
    assert [e["tabId"] for e in events if e["type"] == "worktree_done"] == ["unknown-base-tab"]


def test_a_verified_clean_worktree_is_still_auto_discarded(tmp_path: Path) -> None:
    """Control: a readable worktree without changes is discarded as before."""
    wt = _repo_with_worktree(tmp_path)
    server, events = _server_with_pending(wt, "clean-tab")
    agent = agent_state.find_by_tab("clean-tab").agent  # type: ignore[union-attr]
    assert agent is not None

    presented = server._present_pending_worktree("clean-tab")

    assert presented is False
    assert not wt.wt_dir.exists()
    assert not _branch_exists(wt.repo_root, "task-work")
    assert agent.worktrees == []
    assert [e["type"] for e in events if e["type"] == "worktree_done"] == []
