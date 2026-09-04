# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end check that copy-kiss.sh bundles only git-tracked ``src/kiss`` files.

Release VSIXes used to ship the official Claude Code skills: release.sh
downloaded them into ``src/kiss/agents/claude_skills`` (gitignored) and
passed ``KISS_BUNDLE_EXTRA_DIRS`` to copy-kiss.sh, which copied the extra
directory into ``kiss_project``.  Both halves are gone.  This test runs the
real copy-kiss.sh against a throwaway git checkout that still has such an
untracked skills directory and the old opt-in variable exported, and
verifies that nothing beyond the tracked runtime lands in ``kiss_project``.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
COPY_KISS_SH = REPO_ROOT / "src" / "kiss" / "agents" / "vscode" / "copy-kiss.sh"
SKILLS_DIR_REL = "src/kiss/agents/claude_skills"


def _git(checkout: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(checkout), *args], check=True, capture_output=True)


def _make_checkout(tmp_path: Path) -> Path:
    """Create a minimal git repo with the layout copy-kiss.sh expects."""
    checkout = tmp_path / "checkout"
    vscode_dir = checkout / "src" / "kiss" / "agents" / "vscode"
    vscode_dir.mkdir(parents=True)
    shutil.copy(COPY_KISS_SH, vscode_dir / "copy-kiss.sh")
    (vscode_dir / "package.json").write_text(json.dumps({"version": "0.0.0"}) + "\n")
    (checkout / "src" / "kiss" / "core").mkdir()
    (checkout / "src" / "kiss" / "core" / "_version.py").write_text('__version__ = "1.2.3"\n')
    (checkout / "src" / "kiss" / "agents" / "sorcar").mkdir()
    (checkout / "src" / "kiss" / "agents" / "sorcar" / "skills.py").write_text("# runtime\n")
    (checkout / "pyproject.toml").write_text('[project]\nname = "demo"\n')
    (checkout / "uv.lock").write_text("")
    (checkout / "README.md").write_text("# demo\n")
    (checkout / "LICENSE").write_text("MIT\n")
    (checkout / ".gitignore").write_text("claude_skills/\nkiss_project/\n")
    _git(checkout, "init", "--quiet")
    _git(checkout, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    _git(checkout, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init")
    # The Claude skills directory a pre-change release.sh left behind: present
    # on disk, ignored by git.
    skill = checkout / SKILLS_DIR_REL / "demo-plugin" / "skills" / "demo" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: demo\ndescription: demo\n---\n")
    return checkout


def test_copy_kiss_ignores_untracked_skills_and_extra_dirs_variable(tmp_path: Path) -> None:
    """copy-kiss.sh must not bundle claude_skills, even with the old opt-in set."""
    checkout = _make_checkout(tmp_path)
    env = dict(os.environ, KISS_BUNDLE_EXTRA_DIRS=SKILLS_DIR_REL)
    result = subprocess.run(
        ["bash", str(checkout / "src" / "kiss" / "agents" / "vscode" / "copy-kiss.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    dest = checkout / "src" / "kiss" / "agents" / "vscode" / "kiss_project"
    assert (dest / "src" / "kiss" / "agents" / "sorcar" / "skills.py").is_file()
    assert (dest / "src" / "kiss" / "core" / "_version.py").is_file()
    assert not (dest / SKILLS_DIR_REL).exists(), (
        f"claude_skills must not be bundled into the extension:\n{result.stdout}"
    )
    assert "Bundled" not in result.stdout
