# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The worktree path contract shared by the sorcar tools.

F5: ``WebUseTool.screenshot`` applied ``_active_worktree_remap`` but not the
``_stale_worktree_fallback`` that Read/Write/Edit apply.  After the framework
squash-merges and removes the worktree, a model that remembers an old
worktree path made ``screenshot`` **recreate the deleted worktree directory**
and write the PNG into that zombie tree, where it is never merged and is
deleted by the next ``git worktree prune``.

F1: the ``.kiss-worktrees/kiss_wt-*`` layout was re-implemented with
hard-coded literals in three places in ``useful_tools`` while
``git_worktree`` owns the constants.  These tests build every path from the
**imported** constants, so the guards are pinned to the real layout rather
than to a copy of it.

Real git repositories, real ``git worktree add`` subprocesses, real files,
and a real headless Chromium.  No mocks, patches or doubles.
"""

import subprocess
from pathlib import Path

import pytest

from kiss.agents.sorcar.git_worktree import (
    _WORKTREE_SLUG_PREFIX,
    _WORKTREE_SUBDIR,
)
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.agents.sorcar.web_use_tool import WebUseTool


def _git(repo: Path, *args: str) -> None:
    """Run a real git command in *repo*, failing loudly."""
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _worktree_path(repo: Path, slug: str) -> Path:
    """Build a worktree directory path from the canonical constants."""
    return repo / _WORKTREE_SUBDIR / f"{_WORKTREE_SLUG_PREFIX}{slug}"


@pytest.fixture
def repo_with_worktree(tmp_path):
    """A real git repo with a real live worktree under the canonical layout."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")
    wt = _worktree_path(repo, "g1")
    _git(repo, "worktree", "add", "-b", "kiss/wt-g1", str(wt))
    return repo, wt


def test_write_to_parent_repo_lands_in_the_live_worktree(repo_with_worktree):
    """F1: an absolute parent-repo path is remapped into the live worktree."""
    repo, wt = repo_with_worktree
    tools = UsefulTools(work_dir=str(wt))
    result = tools.Write(str(repo / "notes.txt"), "from the agent")
    assert "Successfully wrote" in result
    assert (wt / "notes.txt").read_text(encoding="utf-8") == "from the agent"
    assert not (repo / "notes.txt").exists()


def test_bash_refuses_a_parent_repo_path(repo_with_worktree):
    """F1: Bash refuses to mutate the user's main checkout."""
    repo, wt = repo_with_worktree
    tools = UsefulTools(work_dir=str(wt))
    result = tools.Bash(f"echo x > {repo}/leaked.txt", "leak")
    assert "outside the active worktree" in result
    assert not (repo / "leaked.txt").exists()


def test_bash_allows_a_worktree_path(repo_with_worktree):
    """F1: the guard does not block legitimate writes inside the worktree."""
    repo, wt = repo_with_worktree
    tools = UsefulTools(work_dir=str(wt))
    tools.Bash(f"echo x > {wt}/inside.txt", "inside")
    assert (wt / "inside.txt").is_file()
    assert not (repo / "inside.txt").exists()


def test_read_falls_back_when_the_worktree_is_gone(repo_with_worktree):
    """F1: a remembered path in a removed worktree reads from the parent repo."""
    repo, wt = repo_with_worktree
    _git(repo, "worktree", "remove", "--force", str(wt))
    tools = UsefulTools(work_dir=str(wt))
    assert "hello" in tools.Read(str(wt / "README.md"))


@pytest.fixture
def stale_worktree(tmp_path):
    """A repo whose worktree was created and then torn down (post-merge)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")
    wt = _worktree_path(repo, "g2")
    _git(repo, "worktree", "add", "-b", "kiss/wt-g2", str(wt))
    _git(repo, "worktree", "remove", "--force", str(wt))
    assert not wt.exists()
    return repo, wt


def test_write_to_a_stale_worktree_path_lands_in_the_parent_repo(stale_worktree):
    """Control: Write already applies the stale-worktree fallback."""
    repo, wt = stale_worktree
    tools = UsefulTools(work_dir=str(wt))
    tools.Write(str(wt / "out.txt"), "x")
    assert (repo / "out.txt").read_text(encoding="utf-8") == "x"
    assert not wt.exists()


def test_bash_from_a_stale_worktree_runs_in_the_parent_repo(stale_worktree):
    """With the worktree gone the guard stands down and Bash still works."""
    repo, wt = stale_worktree
    tools = UsefulTools(work_dir=str(wt))
    assert str(repo) in tools.Bash("pwd", "where am i")
    result = tools.Bash(f"echo x > {repo}/allowed.txt", "write to parent")
    assert "outside the active worktree" not in result
    assert (repo / "allowed.txt").is_file()


def test_plain_work_dir_is_never_remapped_or_refused(tmp_path):
    """Outside any worktree the guards do nothing at all."""
    plain = tmp_path / "plain"
    plain.mkdir()
    tools = UsefulTools(work_dir=str(plain))
    target = tmp_path / "elsewhere" / "note.txt"
    assert "Successfully wrote" in tools.Write(str(target), "hi")
    assert target.read_text(encoding="utf-8") == "hi"
    assert "outside the active worktree" not in tools.Bash(
        f"echo ok > {tmp_path}/plain_ok.txt", "write",
    )
    assert (tmp_path / "plain_ok.txt").is_file()


def test_screenshot_to_a_stale_worktree_path_lands_in_the_parent_repo(
    stale_worktree,
):
    """F5: screenshot must not resurrect a deleted worktree directory."""
    repo, wt = stale_worktree
    tool = WebUseTool(headless=True, user_data_dir=None, work_dir=str(wt))
    try:
        tool.go_to_url("data:text/html,<h1>stale worktree</h1>")
        result = tool.screenshot(str(wt / "reports" / "after.png"))
    finally:
        tool.close()

    assert result.startswith("Screenshot saved to "), result
    assert (repo / "reports" / "after.png").is_file(), result
    assert not wt.exists(), (
        "screenshot resurrected the deleted worktree directory: " + result
    )


def test_screenshot_still_remaps_into_a_live_worktree(repo_with_worktree):
    """Regression guard: the live-worktree remap is unchanged by the fallback."""
    repo, wt = repo_with_worktree
    tool = WebUseTool(headless=True, user_data_dir=None, work_dir=str(wt))
    try:
        tool.go_to_url("data:text/html,<h1>live worktree</h1>")
        result = tool.screenshot(str(repo / "shots" / "page.png"))
    finally:
        tool.close()

    assert (wt / "shots" / "page.png").is_file(), result
    assert not (repo / "shots").exists(), result

def test_created_worktree_uses_the_canonical_layout(tmp_path):
    """The producer's layout is the one the guards recognise.

    ``WorktreeSorcarAgent`` creates the worktree; ``UsefulTools`` and
    the reclaim sweep recognise it.  Both sides must read the layout
    from the same constants, so this builds every expectation from the
    imported ones and then proves the real producer's output is
    accepted by the real consumer.
    """
    from kiss.agents.sorcar.git_worktree import _WORKTREE_BRANCH_PREFIX
    from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

    repo = tmp_path / "producer-repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")

    agent = WorktreeSorcarAgent("layout-producer")
    wt_dir = agent._try_setup_worktree(repo, str(repo))
    assert wt_dir is not None
    wt = agent._wt
    assert wt is not None
    try:
        assert wt.branch.startswith(_WORKTREE_BRANCH_PREFIX), wt.branch
        assert wt_dir.parent == repo / _WORKTREE_SUBDIR, wt_dir
        assert wt_dir.name.startswith(_WORKTREE_SLUG_PREFIX), wt_dir.name

        # The consumer guard agrees: a parent-repo write from inside
        # the produced worktree is remapped into it.
        tools = UsefulTools(work_dir=str(wt_dir))
        assert "Successfully wrote" in tools.Write(
            str(repo / "note.txt"), "hi",
        )
        assert (wt_dir / "note.txt").is_file()
        assert not (repo / "note.txt").exists()
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(wt_dir)],
            cwd=repo, check=False, capture_output=True, text=True,
        )
