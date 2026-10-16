# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end check of release.sh's public history: excluded paths stay out.

Runs ``scripts/test_release_exclude.sh``, which builds a throwaway git
repository with a real bare "public" remote, sources the functions of
``scripts/release.sh``, and verifies that the history pushed to the public
repo drops every path in ``scripts/exclude.json`` from every commit, that
``release_needed`` publishes exactly when the public tip is not the filtered
head, and that the leased push replaces older release shapes.
"""

import subprocess
from pathlib import Path

from kiss.tests.conftest import posix_only

REPO_ROOT = Path(__file__).resolve().parents[4]
EXCLUDE_TEST_SCRIPT = REPO_ROOT / "scripts" / "test_release_exclude.sh"

pytestmark = posix_only("runs the bash release exclude suite from scripts/")


def test_public_history_excludes_paths() -> None:
    """The exclude suite must pass end to end."""
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
    assert "in-sync detection: filtered tree matches the public tree" in result.stdout
