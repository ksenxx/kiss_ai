# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end check of release.sh's public snapshot: excludes out, vsix in.

Runs ``scripts/test_release_exclude.sh``, which builds a throwaway git
repository with a real bare "public" remote, sources the functions of
``scripts/release.sh``, and verifies that the snapshot pushed to the public
repo drops every path in ``scripts/exclude.json`` and carries the built
``kiss-sorcar.vsix`` while no commit of origin ever contains that file.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
EXCLUDE_TEST_SCRIPT = REPO_ROOT / "scripts" / "test_release_exclude.sh"


def test_public_snapshot_excludes_paths_and_ships_vsix() -> None:
    """The exclude/vsix suite must pass end to end."""
    result = subprocess.run(
        ["bash", str(EXCLUDE_TEST_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, f"exclude suite failed:\n{result.stdout}\n{result.stderr}"
    assert "ALL TESTS PASSED" in result.stdout
    assert "never added or committed to origin" in result.stdout
