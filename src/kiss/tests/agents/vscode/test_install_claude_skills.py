# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""copy-kiss.sh tests split out of ``kiss.tests.scripts.test_install_claude_skills``.

``copy-kiss.sh`` lives in ``src/kiss/agents/vscode/`` (it is the VS Code
extension's bundler), so the tests that depend only on it belong here.
The release.sh/install.sh tests remain in
``kiss.tests.scripts.test_install_claude_skills`` (release.sh is under
the repo's ``scripts/`` directory, forcing rule-8 placement), and that
module imports the shared ``SKILLS_DIR_REL`` constant from this one.

``copy-kiss.sh`` must be a fully generic bundler with no Claude
references at all; bundling extra dirs is opt-in via
``KISS_BUNDLE_EXTRA_DIRS`` (which only release.sh sets).
"""

import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
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


if __name__ == "__main__":
    unittest.main()
