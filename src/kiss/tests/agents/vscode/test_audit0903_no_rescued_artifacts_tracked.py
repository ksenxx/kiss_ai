# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""No ``*.kiss-rescued-*`` leftovers may be tracked by git.

Audit 2026-09-03 (vscode-installer): the worktree teardown's
``rescue_ignored_files`` (git_worktree.py) lands a worktree file whose
main-tree twin differs under a ``<name>.kiss-rescued-<ns>`` sibling name.
For ``kiss-sorcar.vsix`` that sibling — e.g.
``kiss-sorcar.vsix.kiss-rescued-1788470990563603656`` — no longer matches
the ``*.vsix`` pattern in ``.gitignore``, so the very next auto-commit
``git add``\\ ed the rescued binary and committed it, which is exactly
what the guard logic in install.sh exists to prevent for the VSIX
itself.  One such zero-byte artifact was committed to this branch.

This test pins the cleanup (the artifact is removed from the index) and
guards against the leak re-landing in a commit.  The naming/gitignore
root cause lives outside this audit's files and is reported as
NEEDS-CROSS-BOUNDARY in tmp/audit2-report-vscode-installer.md.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]


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
