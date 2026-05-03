"""Integration tests for the install.sh Claude skills download step."""

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALL_SH = REPO_ROOT / "install.sh"
COPY_KISS_SH = REPO_ROOT / "src" / "kiss" / "agents" / "vscode" / "copy-kiss.sh"


class TestInstallShClaudeSkillsStep(unittest.TestCase):
    """Verify install.sh has the Claude skills download step."""

    def test_install_sh_has_claude_skills_step(self) -> None:
        text = INSTALL_SH.read_text()
        self.assertIn(
            "Downloading official Claude Code skills",
            text,
            "install.sh must contain the Claude skills download step",
        )

    def test_install_sh_clones_anthropics_repo(self) -> None:
        text = INSTALL_SH.read_text()
        self.assertIn(
            "anthropics/claude-code.git",
            text,
            "install.sh must clone from the anthropics/claude-code repo",
        )

    def test_install_sh_targets_claude_skills_dir(self) -> None:
        text = INSTALL_SH.read_text()
        self.assertIn(
            "src/kiss/agents/claude_skills",
            text,
            "install.sh must target the claude_skills directory",
        )

    def test_install_sh_uses_sparse_checkout(self) -> None:
        text = INSTALL_SH.read_text()
        self.assertIn(
            "sparse-checkout set plugins",
            text,
            "install.sh must use sparse checkout to download only the plugins dir",
        )

    def test_install_sh_step_numbering(self) -> None:
        text = INSTALL_SH.read_text()
        # Ensure all 11 steps are present
        for i in range(1, 12):
            self.assertIn(
                f"[{i}/11]",
                text,
                f"install.sh must have step [{i}/11]",
            )

    def test_install_sh_has_idempotency_guard(self) -> None:
        text = INSTALL_SH.read_text()
        self.assertIn(
            "Claude skills already present",
            text,
            "install.sh must skip download when skills are already present",
        )

    def test_claude_skills_downloaded_before_extension_build(self) -> None:
        """Claude skills step (8) must come before Build VS Code extension (9)."""
        text = INSTALL_SH.read_text()
        skills_pos = text.index("[8/11] Downloading official Claude Code skills")
        build_pos = text.index("[9/11] Building VS Code extension")
        self.assertLess(
            skills_pos,
            build_pos,
            "Claude skills download must precede VS Code extension build",
        )

    def test_claude_skills_deleted_after_extension_install(self) -> None:
        """claude_skills directory must be deleted after extension install (step 10)."""
        text = INSTALL_SH.read_text()
        install_pos = text.index("[10/11] Installing VS Code extension")
        cleanup_pos = text.index("Cleaned up $CLAUDE_SKILLS_DIR (bundled in extension)")
        daemon_pos = text.index("[11/11] Setting up kiss-web daemon service")
        self.assertLess(
            install_pos,
            cleanup_pos,
            "claude_skills cleanup must come after extension install",
        )
        self.assertLess(
            cleanup_pos,
            daemon_pos,
            "claude_skills cleanup must come before daemon setup",
        )

    def test_claude_skills_cleanup_uses_rm_rf(self) -> None:
        """Cleanup must use rm -rf to remove the directory."""
        text = INSTALL_SH.read_text()
        self.assertIn('rm -rf "$CLAUDE_SKILLS_DIR"', text)


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


if __name__ == "__main__":
    unittest.main()
