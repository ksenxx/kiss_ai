# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the active-worktree path remapping contract.

When the agent runs a task inside a git worktree under
``<repo>/.kiss-worktrees/kiss_wt-*/``, the LLM is told (via the system
prompt's ``Work dir:`` line) to operate from that directory.  But models
routinely *ignore* the work_dir hint and instead emit absolute paths
that point at the parent repo (e.g. ``/abs/repo/README.md`` instead of
``/abs/repo/.kiss-worktrees/kiss_wt-X/README.md``).

When that happens, the unprotected ``Read``/``Write``/``Edit`` tools
operate on the **main repository's working tree**, which means:

1.  The worktree stays empty — so ``_auto_commit_worktree`` finds
    nothing to commit and the squash-merge has no commits to merge.
2.  The user's main checkout ends up with uncommitted edits that the
    framework neither attributes to the task nor cleans up.

These tests reproduce that exact failure mode and pin the fix: when
``work_dir`` is inside a live ``.kiss-worktrees/kiss_wt-*`` directory,
file paths that resolve to the parent repo (outside any worktree) are
transparently remapped to the equivalent path *inside* the active
worktree.

Each test exercises a real on-disk git repository plus a real git
``worktree add`` — no mocks, no patches.  These are end-to-end checks
of the actual user-observable behavior reported in the issue
"why didn't you run the last task in worktree and why didn't you commit
the changes?".
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.core.useful_tools import UsefulTools


def _run(*args: str, cwd: Path) -> None:
    """Run a command, raising on failure (helper for fixture setup)."""
    subprocess.run(
        list(args),
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def repo_with_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """Create a real git repo with one committed README and a live worktree.

    Returns ``(repo, wt_dir)`` where:

    * ``repo`` is the main repository root (initial commit on ``main``).
    * ``wt_dir`` is a freshly created git worktree under
      ``repo / ".kiss-worktrees" / "kiss_wt-test-xxxxxxxx"`` on a fresh
      branch ``kiss/wt-test-xxxxxxxx``.

    The worktree starts with the same content as the main repo.
    """
    repo = tmp_path / "myrepo"
    repo.mkdir()
    _run("git", "init", "-b", "main", ".", cwd=repo)
    _run("git", "config", "user.email", "test@example.com", cwd=repo)
    _run("git", "config", "user.name", "Test", cwd=repo)
    (repo / "README.md").write_text("MAIN README v1\n")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("# main app\n")
    _run("git", "add", "-A", cwd=repo)
    _run("git", "commit", "-m", "init", cwd=repo)
    # Mirror the real framework: ``.kiss-worktrees/`` is never tracked,
    # so it must not pollute the main repo's ``git status``.
    GitWorktreeOps.ensure_excluded(repo)

    wt_dir = repo / ".kiss-worktrees" / "kiss_wt-test-deadbeef"
    _run(
        "git", "worktree", "add",
        "-b", "kiss/wt-test-deadbeef", str(wt_dir),
        cwd=repo,
    )
    return repo, wt_dir


def test_edit_via_main_repo_path_lands_in_worktree(repo_with_worktree) -> None:
    """``Edit("/main/repo/README.md")`` while ``work_dir`` is a worktree
    must mutate the *worktree's* README, never the main repo's.

    This is the exact failure that caused the Task 2 regression: the
    LLM ignored the worktree work-dir hint and emitted main-repo
    absolute paths, so the README edit landed in the main checkout
    and the worktree stayed empty (no auto-commit, no merge).
    """
    repo, wt_dir = repo_with_worktree
    tools = UsefulTools(work_dir=str(wt_dir))

    out = tools.Edit(str(repo / "README.md"), "MAIN README v1", "EDITED v2")

    assert "Successfully replaced" in out, out
    # The worktree's README must have changed.
    assert (wt_dir / "README.md").read_text() == "EDITED v2\n"
    # The main repo's README must NOT have changed.
    assert (repo / "README.md").read_text() == "MAIN README v1\n"


def test_write_via_main_repo_path_lands_in_worktree(repo_with_worktree) -> None:
    """``Write("/main/repo/notes.md")`` writes the file inside the worktree.

    Brand-new files (not previously checked in) must also land in the
    worktree so that ``git status`` of the worktree picks them up as
    untracked and the auto-commit stages them.
    """
    repo, wt_dir = repo_with_worktree
    tools = UsefulTools(work_dir=str(wt_dir))

    out = tools.Write(str(repo / "notes.md"), "hello\n")

    assert "Successfully wrote" in out, out
    assert (wt_dir / "notes.md").read_text() == "hello\n"
    assert not (repo / "notes.md").exists()


def test_read_via_main_repo_path_returns_worktree_content(
    repo_with_worktree,
) -> None:
    """``Read("/main/repo/X")`` returns the worktree's copy, not main's.

    A model that already edited a file via the remapped Edit path
    must subsequently see its own edits via the same path; otherwise
    it would observe stale main-repo content and overwrite its own
    work.
    """
    repo, wt_dir = repo_with_worktree
    # Diverge the two copies so the test can tell them apart.
    (wt_dir / "README.md").write_text("WT README v9\n")
    tools = UsefulTools(work_dir=str(wt_dir))

    out = tools.Read(str(repo / "README.md"))

    assert out == "WT README v9\n"


def test_remap_preserves_subdirectory_paths(repo_with_worktree) -> None:
    """Remap works for nested files, not just files at the repo root."""
    repo, wt_dir = repo_with_worktree
    tools = UsefulTools(work_dir=str(wt_dir))

    out = tools.Edit(str(repo / "src" / "app.py"), "# main app", "# patched")

    assert "Successfully replaced" in out, out
    assert (wt_dir / "src" / "app.py").read_text() == "# patched\n"
    assert (repo / "src" / "app.py").read_text() == "# main app\n"


def test_remap_does_not_touch_paths_outside_main_repo(
    repo_with_worktree, tmp_path: Path,
) -> None:
    """Absolute paths *outside* the main repo are read/written verbatim.

    The remap must never reroute ``/etc/passwd``, ``/tmp/scratch``,
    or any other path that has nothing to do with the worktree's
    parent repository.
    """
    repo, wt_dir = repo_with_worktree
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n")
    tools = UsefulTools(work_dir=str(wt_dir))

    assert tools.Read(str(outside)) == "outside\n"

    tools.Edit(str(outside), "outside", "patched_outside")
    assert outside.read_text() == "patched_outside\n"


def test_remap_leaves_worktree_paths_alone(repo_with_worktree) -> None:
    """Paths that already resolve into the active worktree pass through."""
    repo, wt_dir = repo_with_worktree
    tools = UsefulTools(work_dir=str(wt_dir))

    out = tools.Edit(str(wt_dir / "README.md"), "MAIN README v1", "WT v2")

    assert "Successfully replaced" in out, out
    assert (wt_dir / "README.md").read_text() == "WT v2\n"
    assert (repo / "README.md").read_text() == "MAIN README v1\n"


def test_remap_leaves_other_worktree_paths_alone(repo_with_worktree) -> None:
    """A path inside a *different* worktree must not be rerouted.

    If the user has multiple concurrent worktrees (e.g. two tabs),
    the active agent must not silently redirect a sibling worktree's
    path into its own worktree.
    """
    repo, wt_dir = repo_with_worktree
    other_wt = repo / ".kiss-worktrees" / "kiss_wt-other-abcd1234"
    _run(
        "git", "worktree", "add",
        "-b", "kiss/wt-other-abcd1234", str(other_wt),
        cwd=repo,
    )
    (other_wt / "README.md").write_text("OTHER WT v1\n")
    tools = UsefulTools(work_dir=str(wt_dir))

    out = tools.Read(str(other_wt / "README.md"))
    assert out == "OTHER WT v1\n"


def test_remap_keeps_auto_commit_working(repo_with_worktree) -> None:
    """End-to-end: after a remapped Edit, the worktree auto-commit fires.

    The original Task 2 bug surfaced as "no commit was created" because
    the LLM's main-repo Edit left the worktree's working tree clean.
    With the remap in place, the same LLM-style call now dirties the
    *worktree*, and the worktree's git status shows the change ready
    for the framework's auto-commit step.
    """
    repo, wt_dir = repo_with_worktree
    tools = UsefulTools(work_dir=str(wt_dir))

    tools.Edit(str(repo / "README.md"), "MAIN README v1", "DONE\n")

    # The worktree now has uncommitted changes; the main repo does not.
    assert GitWorktreeOps.has_uncommitted_changes(wt_dir)
    assert not GitWorktreeOps.has_uncommitted_changes(repo)
