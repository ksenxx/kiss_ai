"""Integration tests for the install.sh Claude skills download step."""

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALL_SH = REPO_ROOT / "install.sh"
CLAUDE_SKILLS_DIR = REPO_ROOT / "src" / "kiss" / "agents" / "claude_skills"

# All 13 official Claude Code plugins from anthropics/claude-code
EXPECTED_SKILLS = [
    "agent-sdk-dev",
    "claude-opus-4-5-migration",
    "code-review",
    "commit-commands",
    "explanatory-output-style",
    "feature-dev",
    "frontend-design",
    "hookify",
    "learning-output-style",
    "plugin-dev",
    "pr-review-toolkit",
    "ralph-wiggum",
    "security-guidance",
]


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
        self.assertIn("[10/11]", text)
        self.assertIn("[11/11]", text)
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


class TestClaudeSkillsDownloaded(unittest.TestCase):
    """Verify all official Claude skills were downloaded to the target dir."""

    def test_claude_skills_dir_exists(self) -> None:
        self.assertTrue(
            CLAUDE_SKILLS_DIR.is_dir(),
            f"{CLAUDE_SKILLS_DIR} must exist",
        )

    def test_all_expected_skills_present(self) -> None:
        for skill_name in EXPECTED_SKILLS:
            skill_dir = CLAUDE_SKILLS_DIR / skill_name
            self.assertTrue(
                skill_dir.is_dir(),
                f"Expected skill directory not found: {skill_name}",
            )

    def test_most_skills_have_claude_plugin_dir(self) -> None:
        with_plugin = [
            s for s in EXPECTED_SKILLS
            if (CLAUDE_SKILLS_DIR / s / ".claude-plugin").is_dir()
        ]
        # At least 12 of 13 official plugins ship with .claude-plugin/
        self.assertGreaterEqual(
            len(with_plugin), 12,
            f"Expected at least 12 skills with .claude-plugin/, got {len(with_plugin)}",
        )

    def test_no_unexpected_extra_directories(self) -> None:
        actual = {
            d.name
            for d in CLAUDE_SKILLS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        }
        expected = set(EXPECTED_SKILLS)
        self.assertEqual(
            actual,
            expected,
            f"Unexpected directories in claude_skills: {actual - expected}",
        )

    def test_skill_count(self) -> None:
        dirs = [
            d for d in CLAUDE_SKILLS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        self.assertEqual(len(dirs), 13, f"Expected 13 skills, got {len(dirs)}")


if __name__ == "__main__":
    unittest.main()
