# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for Claude skills handling in the install/release paths.

Claude skills are owned exclusively by ``release.sh``: it downloads them and
opts into bundling by passing ``KISS_BUNDLE_EXTRA_DIRS`` to ``copy-kiss.sh``.
``copy-kiss.sh`` is a fully generic bundler with no Claude references at all,
and ``install.sh`` must never install, delete, or otherwise touch Claude
skills — it does not set the opt-in variable and contains no skills
references.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_SH = REPO_ROOT / "scripts" / "release.sh"
INSTALL_SH = REPO_ROOT / "install.sh"
COPY_KISS_SH = REPO_ROOT / "src" / "kiss" / "agents" / "vscode" / "copy-kiss.sh"

EXTRA_BUNDLE_BLOCK_BEGIN = "# BEGIN: kiss-extra-bundle"
EXTRA_BUNDLE_BLOCK_END = "# END: kiss-extra-bundle"

SKILLS_DIR_REL = "src/kiss/agents/claude_skills"


def extract_extra_bundle_block() -> str:
    """Return the generic extra-dirs bundling block of copy-kiss.sh, verbatim."""
    text = COPY_KISS_SH.read_text()
    begin = text.index(EXTRA_BUNDLE_BLOCK_BEGIN)
    end = text.index(EXTRA_BUNDLE_BLOCK_END)
    return text[begin:end]


def run_extra_bundle_block(bundle_var_set: bool, extra_src_exists: bool) -> bool:
    """Execute the extracted bundling block against temp dirs.

    Returns True when the block copied the extra dir into DEST.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        project_root = tmp_path / "checkout"
        dest = tmp_path / "kiss_project"
        dest.mkdir(parents=True)
        if extra_src_exists:
            skill = project_root / SKILLS_DIR_REL / "demo"
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text("# demo skill\n")
        script = (
            "set -e\n"
            f'PROJECT_ROOT="{project_root}"\n'
            f'DEST="{dest}"\n' + extract_extra_bundle_block()
        )
        env = {"PATH": "/usr/bin:/bin"}
        if bundle_var_set:
            env["KISS_BUNDLE_EXTRA_DIRS"] = SKILLS_DIR_REL
        subprocess.run(
            ["bash", "-c", script], env=env, check=True, capture_output=True
        )
        return (dest / SKILLS_DIR_REL / "demo" / "SKILL.md").is_file()


class TestCopyKissHasNoClaudeReferences(unittest.TestCase):
    """copy-kiss.sh must be fully generic, with no Claude mention at all."""

    def test_copy_kiss_never_mentions_claude(self) -> None:
        text = COPY_KISS_SH.read_text().lower()
        for needle in ("claude", "skill"):
            self.assertNotIn(
                needle,
                text,
                f"copy-kiss.sh must not mention '{needle}' — bundling is generic"
                " via KISS_BUNDLE_EXTRA_DIRS; Claude specifics live in release.sh",
            )

    def test_copy_kiss_has_generic_extra_bundle_block(self) -> None:
        text = COPY_KISS_SH.read_text()
        self.assertIn("KISS_BUNDLE_EXTRA_DIRS", text)
        self.assertIn(EXTRA_BUNDLE_BLOCK_BEGIN, text)
        self.assertIn(EXTRA_BUNDLE_BLOCK_END, text)


class TestCopyKissExtraBundleOptIn(unittest.TestCase):
    """The extra-dirs copy must be opt-in so install.sh bundles nothing extra.

    These tests execute the actual bundling block extracted verbatim from
    copy-kiss.sh, so they verify real behavior, not just script text.
    """

    def test_extra_dir_copied_when_bundle_var_set(self) -> None:
        self.assertTrue(
            run_extra_bundle_block(bundle_var_set=True, extra_src_exists=True),
            "KISS_BUNDLE_EXTRA_DIRS must bundle the listed dirs into DEST",
        )

    def test_extra_dir_not_copied_without_bundle_var(self) -> None:
        self.assertFalse(
            run_extra_bundle_block(bundle_var_set=False, extra_src_exists=True),
            "without KISS_BUNDLE_EXTRA_DIRS nothing extra must be bundled"
            " (this is the install.sh path)",
        )

    def test_no_copy_when_extra_dir_missing(self) -> None:
        self.assertFalse(
            run_extra_bundle_block(bundle_var_set=True, extra_src_exists=False),
            "a missing extra source dir must not fail or copy anything",
        )

    def test_release_sh_sets_bundle_var_for_copy_kiss_and_package(self) -> None:
        """release.sh must opt in for BOTH steps that run copy-kiss.sh.

        `npm run package` re-runs copy-kiss.sh via `vscode:prepublish`, so a
        bare `npm run package` would silently drop the skills from the VSIX.
        """
        text = RELEASE_SH.read_text()
        opt_in = f'KISS_BUNDLE_EXTRA_DIRS="{SKILLS_DIR_REL}"'
        self.assertIn(f"{opt_in} npm run copy-kiss", text)
        self.assertIn(f"{opt_in} npm run package", text)


class TestInstallShNeverTouchesClaudeSkills(unittest.TestCase):
    """install.sh must not install, delete, or reference Claude skills."""

    def test_install_sh_has_no_claude_skills_references(self) -> None:
        text = INSTALL_SH.read_text().lower()
        for needle in ("claude", "skill"):
            self.assertNotIn(
                needle,
                text,
                f"install.sh must not mention '{needle}' — Claude skills are"
                " owned by release.sh only",
            )

    def test_install_sh_does_not_opt_into_extra_bundling(self) -> None:
        self.assertNotIn(
            "KISS_BUNDLE_EXTRA_DIRS",
            INSTALL_SH.read_text(),
            "install.sh must never set the extra-dirs bundling opt-in variable",
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
            SKILLS_DIR_REL,
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
