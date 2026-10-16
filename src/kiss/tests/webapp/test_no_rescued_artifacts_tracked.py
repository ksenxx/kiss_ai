# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""No ``*.kiss-rescued-*`` leftovers may be tracked by git.

The worktree teardown's ``rescue_ignored_files`` (git_worktree.py) lands a
worktree file whose main-tree twin differs under a
``<name>.kiss-rescued-<ns>`` sibling name.  Such a sibling no longer
matches the ``.gitignore`` pattern that covered the original (a build
artefact, say), so the next auto-commit stages the rescued copy with
``git add`` and commits it.  One such zero-byte artefact was committed to this
branch once (the packaged VS Code extension, since removed).

This test pins the cleanup (the artefact is removed from the index) and
guards against the leak re-landing in a commit.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_no_kiss_rescued_files_are_tracked() -> None:
    """`git ls-files` lists no rescue-sibling artifacts anywhere."""
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--", "*.kiss-rescued-*"],
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = [line for line in result.stdout.splitlines() if line.strip()]
    assert tracked == [], (
        "rescued-file artifacts are tracked by git (rescue siblings are "
        "transient recovery copies, never repo content): " + ", ".join(tracked)
    )
