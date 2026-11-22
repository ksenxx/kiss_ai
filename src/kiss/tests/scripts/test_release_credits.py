# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end check of who release.sh's public history names.

Runs ``scripts/test_release_credits.sh``, which publishes releases to a real
bare "public" remote under a leaked placeholder git identity and verifies
that ``.mailmap`` replaces that identity on every release commit and tag,
that ``scripts/credits.txt`` puts ``Co-authored-by`` trailers on the named
release only, and that dropping a credit removes its trailer again.
"""

import subprocess
from pathlib import Path

from kiss.tests.conftest import posix_only

REPO_ROOT = Path(__file__).resolve().parents[4]
CREDITS_TEST_SCRIPT = REPO_ROOT / "scripts" / "test_release_credits.sh"

pytestmark = posix_only("runs the bash release credits suite from scripts/")


def test_public_history_identities_and_credits() -> None:
    """The credits suite must pass end to end."""
    result = subprocess.run(
        ["bash", str(CREDITS_TEST_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, f"credits suite failed:\n{result.stdout}\n{result.stderr}"
    assert "ALL TESTS PASSED" in result.stdout
