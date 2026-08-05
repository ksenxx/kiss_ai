# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end check that release.sh purges excluded paths from public history.

Runs ``scripts/test_release_purge.sh``, which builds throwaway git repositories
with real bare "public" remotes, publishes unfiltered history into them, runs
``purge_public_history`` from ``scripts/release.sh``, and inspects the result.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PURGE_TEST_SCRIPT = REPO_ROOT / "scripts" / "test_release_purge.sh"


def test_release_purges_excluded_paths_from_public_history() -> None:
    """The purge suite must pass: no excluded path may survive in public history."""
    result = subprocess.run(
        ["bash", str(PURGE_TEST_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, f"purge suite failed:\n{result.stdout}\n{result.stderr}"
    assert "ALL TESTS PASSED" in result.stdout
