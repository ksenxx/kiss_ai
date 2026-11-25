# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end check of the installer step of release.sh: Mac package and Windows setup program.

Runs ``scripts/test_release_macos_pkg.sh``.  On macOS it builds the real
package of HEAD minus ``scripts/exclude.json`` (as the release does) and
checks its payload: the app, the version, the commit, and no excluded path.
Where ``makensis`` and 7-Zip are installed it does the same for the real
Windows setup program.  On every platform it then executes the real
``main()`` of ``scripts/release.sh`` in a scratch repository (publishing stubbed, remotes
local, the builder a recorder) and verifies that the package of the bumped
version is committed with the version bump and reaches origin and the public
repo, that it is built from the committed tree minus the excluded paths, that
the signing variables reach the builder, that a commit which reaches origin
after the build is in the pushed package, and that a failing build releases
nothing.
"""

import shutil
import subprocess
import sys
from pathlib import Path

from kiss.tests.conftest import posix_only

REPO_ROOT = Path(__file__).resolve().parents[4]
PKG_TEST_SCRIPT = REPO_ROOT / "scripts" / "test_release_macos_pkg.sh"

pytestmark = posix_only("runs the bash release Mac package suite from scripts/")


def test_release_builds_commits_and_pushes_the_mac_package() -> None:
    """The Mac package suite must pass end to end."""
    result = subprocess.run(
        ["bash", str(PKG_TEST_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, f"Mac package suite failed:\n{result.stdout}\n{result.stderr}"
    assert "ALL TESTS PASSED" in result.stdout
    assert "a failing build releases nothing" in result.stdout
    real_build = "PASS: the real package carries no path of scripts/exclude.json"
    assert (real_build in result.stdout) == (sys.platform == "darwin")
    # The real setup program is built wherever its tools are; the suite says so when not.
    setup_tools = bool(shutil.which("makensis") and (shutil.which("7zz") or shutil.which("7z")))
    real_setup = (
        "PASS: the real setup program holds the app, the version and the commit of HEAD "
        "and no excluded path"
    )
    assert (real_setup in result.stdout) == setup_tools
    skipped_setup = "SKIP: the real setup program needs makensis and 7-Zip"
    assert (skipped_setup in result.stdout) == (not setup_tools)
