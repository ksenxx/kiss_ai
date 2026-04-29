"""Tests for USER_PREFS.md copy to/from worktree.

Verifies that USER_PREFS.md is copied into the worktree on creation
and copied back to the repo root when the worktree is finalized.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


def _make_repo(path: Path) -> Path:
    """Create a git repo with one initial commit at *path*."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@t.com"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "T"],
        capture_output=True,
        check=True,
    )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True,
        check=True,
    )
    return path


class TestUserPrefsCopyToWorktree:
    """USER_PREFS.md is copied into the worktree on setup."""

    def test_copy_when_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            prefs_content = "## Preferences\n- Pref A\n"
            (repo / "USER_PREFS.md").write_text(prefs_content)

            agent = WorktreeSorcarAgent("test")
            agent._chat_id = "testchat"
            agent._try_setup_worktree(repo, str(repo))

            assert agent._wt is not None
            wt_prefs = agent._wt.wt_dir / "USER_PREFS.md"
            assert wt_prefs.is_file()
            assert wt_prefs.read_text() == prefs_content

    def test_no_copy_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            # No USER_PREFS.md in repo

            agent = WorktreeSorcarAgent("test")
            agent._chat_id = "testchat2"
            agent._try_setup_worktree(repo, str(repo))

            assert agent._wt is not None
            wt_prefs = agent._wt.wt_dir / "USER_PREFS.md"
            assert not wt_prefs.exists()


class TestUserPrefsCopyBackOnFinalize:
    """USER_PREFS.md is copied back to repo root when worktree is finalized."""

    def test_copy_back_on_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            original_content = "## Preferences\n- Pref A\n"
            (repo / "USER_PREFS.md").write_text(original_content)

            agent = WorktreeSorcarAgent("test")
            agent._chat_id = "testchat3"
            agent._try_setup_worktree(repo, str(repo))
            assert agent._wt is not None

            # Simulate agent updating USER_PREFS.md in the worktree
            updated_content = "## Preferences\n- Pref A\n- Pref B (new)\n"
            (agent._wt.wt_dir / "USER_PREFS.md").write_text(updated_content)

            result = agent._finalize_worktree()
            assert result is True

            # USER_PREFS.md in repo root should have the updated content
            assert (repo / "USER_PREFS.md").read_text() == updated_content

    def test_no_copy_back_when_absent_in_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            # No USER_PREFS.md anywhere

            agent = WorktreeSorcarAgent("test")
            agent._chat_id = "testchat4"
            agent._try_setup_worktree(repo, str(repo))
            assert agent._wt is not None

            result = agent._finalize_worktree()
            assert result is True

            # No USER_PREFS.md should appear
            assert not (repo / "USER_PREFS.md").exists()

    def test_copy_back_preserves_on_merge(self) -> None:
        """Full merge flow: prefs updated in worktree survive merge."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            # Gitignore USER_PREFS.md so it doesn't get committed
            # and cause merge conflicts
            (repo / ".gitignore").write_text("USER_PREFS.md\n")
            subprocess.run(
                ["git", "-C", str(repo), "add", ".gitignore"],
                capture_output=True, check=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "commit", "-m", "add gitignore"],
                capture_output=True, check=True,
            )

            original_content = "## Prefs\n"
            (repo / "USER_PREFS.md").write_text(original_content)

            agent = WorktreeSorcarAgent("test")
            agent._chat_id = "testchat5"
            agent._try_setup_worktree(repo, str(repo))
            assert agent._wt is not None

            # Simulate agent updating prefs and making a code change
            updated_prefs = "## Prefs\n- Updated pref\n"
            (agent._wt.wt_dir / "USER_PREFS.md").write_text(updated_prefs)
            (agent._wt.wt_dir / "new_file.txt").write_text("hello\n")
            GitWorktreeOps.commit_all(agent._wt.wt_dir, "agent work")

            result = agent.merge()
            assert "Successfully merged" in result

            # USER_PREFS.md should have been copied back before removal
            assert (repo / "USER_PREFS.md").read_text() == updated_prefs
