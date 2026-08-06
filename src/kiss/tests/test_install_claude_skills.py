# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for Claude skills handling in the install/release paths.

Claude skills are owned exclusively by ``release.sh`` (downloads and bundles
them into the release VSIX) and ``copy-kiss.sh`` (performs the bundling, but
only when ``KISS_BUNDLE_CLAUDE_SKILLS`` is set).  ``install.sh`` must never
install, delete, or otherwise touch Claude skills — it does not set the
opt-in variable and contains no skills references at all.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_SH = REPO_ROOT / "scripts" / "release.sh"
INSTALL_SH = REPO_ROOT / "install.sh"
COPY_KISS_SH = REPO_ROOT / "src" / "kiss" / "agents" / "vscode" / "copy-kiss.sh"

SKILLS_BLOCK_BEGIN = "# BEGIN: kiss-claude-skills-bundle"
SKILLS_BLOCK_END = "# END: kiss-claude-skills-bundle"


def extract_skills_block() -> str:
    """Return the Claude-skills bundling block of copy-kiss.sh, verbatim."""
    text = COPY_KISS_SH.read_text()
    begin = text.index(SKILLS_BLOCK_BEGIN)
    end = text.index(SKILLS_BLOCK_END)
    return text[begin:end]


def run_skills_block(bundle_var_set: bool, skills_src_exists: bool) -> bool:
    """Execute the extracted skills block against temp dirs.

    Returns True when the block copied the skills into DEST.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        project_root = tmp_path / "checkout"
        dest = tmp_path / "kiss_project"
        (dest / "src" / "kiss" / "agents").mkdir(parents=True)
        if skills_src_exists:
            skill = project_root / "src" / "kiss" / "agents" / "claude_skills" / "demo"
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text("# demo skill\n")
        script = (
            "set -e\n"
            f'PROJECT_ROOT="{project_root}"\n'
            f'DEST="{dest}"\n' + extract_skills_block()
        )
        env = {"PATH": "/usr/bin:/bin"}
        if bundle_var_set:
            env["KISS_BUNDLE_CLAUDE_SKILLS"] = "1"
        subprocess.run(
            ["bash", "-c", script], env=env, check=True, capture_output=True
        )
        return (
            dest / "src" / "kiss" / "agents" / "claude_skills" / "demo" / "SKILL.md"
        ).is_file()


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


class TestCopyKissClaudeSkillsOptIn(unittest.TestCase):
    """The skills copy must be opt-in so install.sh never touches skills.

    These tests execute the actual bundling block extracted verbatim from
    copy-kiss.sh, so they verify real behavior, not just script text.
    """

    def test_skills_copied_when_bundle_var_set(self) -> None:
        self.assertTrue(
            run_skills_block(bundle_var_set=True, skills_src_exists=True),
            "KISS_BUNDLE_CLAUDE_SKILLS=1 must bundle claude_skills into DEST",
        )

    def test_skills_not_copied_without_bundle_var(self) -> None:
        self.assertFalse(
            run_skills_block(bundle_var_set=False, skills_src_exists=True),
            "without KISS_BUNDLE_CLAUDE_SKILLS the skills must NOT be bundled"
            " (this is the install.sh path)",
        )

    def test_no_copy_when_skills_dir_missing(self) -> None:
        self.assertFalse(
            run_skills_block(bundle_var_set=True, skills_src_exists=False),
            "a missing claude_skills source dir must not fail or copy anything",
        )

    def test_release_sh_sets_bundle_var_for_copy_kiss_and_package(self) -> None:
        """release.sh must opt in for BOTH steps that run copy-kiss.sh.

        `npm run package` re-runs copy-kiss.sh via `vscode:prepublish`, so a
        bare `npm run package` would silently drop the skills from the VSIX.
        """
        text = RELEASE_SH.read_text()
        self.assertIn("KISS_BUNDLE_CLAUDE_SKILLS=1 npm run copy-kiss", text)
        self.assertIn("KISS_BUNDLE_CLAUDE_SKILLS=1 npm run package", text)


class TestInstallShNeverTouchesClaudeSkills(unittest.TestCase):
    """install.sh must not install, delete, or reference Claude skills."""

    def test_install_sh_has_no_claude_skills_references(self) -> None:
        text = INSTALL_SH.read_text().lower()
        for needle in ("claude", "skill"):
            self.assertNotIn(
                needle,
                text,
                f"install.sh must not mention '{needle}' — Claude skills are"
                " owned by release.sh/copy-kiss.sh only",
            )

    def test_install_sh_does_not_opt_into_skills_bundling(self) -> None:
        self.assertNotIn(
            "KISS_BUNDLE_CLAUDE_SKILLS",
            INSTALL_SH.read_text(),
            "install.sh must never set the skills bundling opt-in variable",
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
        """Claude skills step (5) must come before Build VS Code extension (6)."""
        text = RELEASE_SH.read_text()
        skills_pos = text.index("Step 5: Download official Claude Code skills")
        build_pos = text.index("Step 6: Build VS Code extension")
        self.assertLess(
            skills_pos,
            build_pos,
            "Claude skills download must precede VS Code extension build in release.sh",
        )

    def test_claude_skills_deleted_after_build_before_commit(self) -> None:
        """Skills dir must be deleted after build (6), before commit (7)."""
        text = RELEASE_SH.read_text()
        build_pos = text.index("Step 6: Build VS Code extension")
        cleanup_pos = text.index("Cleaned up $CLAUDE_SKILLS_DIR (bundled in extension)")
        commit_pos = text.index("Step 7: Commit")
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
            "# 5. Download official Claude Code skills",
            text,
            "release.sh workflow comment must include Claude skills step",
        )

    def test_release_sh_claude_skills_dir_is_absolute(self) -> None:
        """CLAUDE_SKILLS_DIR must be an absolute path so cp works after cd."""
        text = RELEASE_SH.read_text()
        import re

        match = re.search(r'CLAUDE_SKILLS_DIR="([^"]*)"', text)
        assert match is not None, "CLAUDE_SKILLS_DIR assignment not found"
        value = match.group(1)
        self.assertTrue(
            value.startswith("$(pwd)") or value.startswith("/"),
            f"CLAUDE_SKILLS_DIR must be absolute, got: {value}",
        )

    def test_release_sh_workflow_has_14_steps(self) -> None:
        """Header workflow comment must have 14 steps after adding the history purge."""
        text = RELEASE_SH.read_text()
        self.assertIn(
            "# 14. Restore stashed changes",
            text,
            "release.sh workflow must have 14 steps",
        )


if __name__ == "__main__":
    unittest.main()
