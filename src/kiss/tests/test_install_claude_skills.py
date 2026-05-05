"""Integration tests for Claude skills steps in release.sh and copy-kiss.sh.

Note: ``install.sh`` was intentionally removed in commit 71a9d0c5
("feat!: security hardening …" / "Remove install.sh — installation now
handled by extension").  Its tests (formerly ``TestInstallShClaudeSkillsStep``)
are deleted as obsolete; remaining classes still cover ``release.sh``
and ``copy-kiss.sh``, which are the active install paths.
"""

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_SH = REPO_ROOT / "scripts" / "release.sh"
COPY_KISS_SH = REPO_ROOT / "src" / "kiss" / "agents" / "vscode" / "copy-kiss.sh"


class TestCopyKissIncludesClaudeSkills(unittest.TestCase):
    """Verify copy-kiss.sh copies claude_skills into the extension bundle."""

    def test_copy_kiss_copies_claude_skills(self) -> None:
        text = COPY_KISS_SH.read_text()
        self.assertIn(
            "claude_skills",
            text,
            "copy-kiss.sh must copy claude_skills into kiss_project",
        )

    def test_copy_kiss_checks_dir_exists(self) -> None:
        text = COPY_KISS_SH.read_text()
        self.assertIn(
            '-d "$CLAUDE_SKILLS_SRC"',
            text,
            "copy-kiss.sh must check if claude_skills directory exists before copying",
        )


class TestReleaseShClaudeSkillsStep(unittest.TestCase):
    """Verify release.sh has the Claude skills download step."""

    def test_release_sh_has_claude_skills_step(self) -> None:
        text = RELEASE_SH.read_text()
        self.assertIn(
            "Downloading official Claude Code skills",
            text,
            "release.sh must contain the Claude skills download step",
        )

    def test_release_sh_clones_anthropics_repo(self) -> None:
        text = RELEASE_SH.read_text()
        self.assertIn(
            "anthropics/claude-code.git",
            text,
            "release.sh must clone from the anthropics/claude-code repo",
        )

    def test_release_sh_targets_claude_skills_dir(self) -> None:
        text = RELEASE_SH.read_text()
        self.assertIn(
            "src/kiss/agents/claude_skills",
            text,
            "release.sh must target the claude_skills directory",
        )

    def test_release_sh_uses_sparse_checkout(self) -> None:
        text = RELEASE_SH.read_text()
        self.assertIn(
            "sparse-checkout set plugins",
            text,
            "release.sh must use sparse checkout to download only the plugins dir",
        )

    def test_release_sh_has_idempotency_guard(self) -> None:
        text = RELEASE_SH.read_text()
        self.assertIn(
            "Claude skills already present",
            text,
            "release.sh must skip download when skills are already present",
        )

    def test_claude_skills_downloaded_before_extension_build(self) -> None:
        """Claude skills step (3) must come before Build VS Code extension (4)."""
        text = RELEASE_SH.read_text()
        skills_pos = text.index("Step 3: Download official Claude Code skills")
        build_pos = text.index("Step 4: Build VS Code extension")
        self.assertLess(
            skills_pos,
            build_pos,
            "Claude skills download must precede VS Code extension build in release.sh",
        )

    def test_claude_skills_deleted_after_build_before_commit(self) -> None:
        """Skills dir must be deleted after build (4), before commit (5)."""
        text = RELEASE_SH.read_text()
        build_pos = text.index("Step 4: Build VS Code extension")
        cleanup_pos = text.index("Cleaned up $CLAUDE_SKILLS_DIR (bundled in extension)")
        commit_pos = text.index("Step 5: Commit")
        self.assertLess(
            build_pos,
            cleanup_pos,
            "claude_skills cleanup must come after extension build in release.sh",
        )
        self.assertLess(
            cleanup_pos,
            commit_pos,
            "claude_skills cleanup must come before git commit in release.sh",
        )

    def test_claude_skills_cleanup_uses_rm_rf(self) -> None:
        """Cleanup must use rm -rf to remove the directory."""
        text = RELEASE_SH.read_text()
        self.assertIn('rm -rf "$CLAUDE_SKILLS_DIR"', text)

    def test_release_sh_workflow_comment_includes_claude_skills(self) -> None:
        """Header workflow comment must list the Claude skills step."""
        text = RELEASE_SH.read_text()
        self.assertIn(
            "# 4. Download official Claude Code skills",
            text,
            "release.sh workflow comment must include Claude skills step",
        )

    def test_release_sh_claude_skills_dir_is_absolute(self) -> None:
        """CLAUDE_SKILLS_DIR must be an absolute path so cp works after cd."""
        text = RELEASE_SH.read_text()
        # The assignment must use $(pwd) or similar to make the path absolute;
        # a bare relative path like CLAUDE_SKILLS_DIR="src/..." would break
        # after cd into the sparse-checkout temp directory.
        import re

        match = re.search(r'CLAUDE_SKILLS_DIR="([^"]*)"', text)
        assert match is not None, "CLAUDE_SKILLS_DIR assignment not found"
        value = match.group(1)
        self.assertTrue(
            value.startswith("$(pwd)") or value.startswith("/"),
            f"CLAUDE_SKILLS_DIR must be absolute, got: {value}",
        )

    def test_release_sh_workflow_has_13_steps(self) -> None:
        """Header workflow comment must have 13 steps after adding Claude skills."""
        text = RELEASE_SH.read_text()
        self.assertIn(
            "# 13. Restore stashed changes",
            text,
            "release.sh workflow must have 13 steps",
        )


if __name__ == "__main__":
    unittest.main()
