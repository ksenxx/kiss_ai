# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end check that the PyPI upload of release.sh stays under 100 MiB.

Runs ``scripts/test_release_pypi_size.sh``, which builds the real sdist and
wheel with ``uv build`` and checks their size and contents (only the wheel's
packages, never benchmark results, papers, reports or node_modules), then
sources ``scripts/release.sh`` and drives ``publish_to_pypi`` against a stub
``uv`` to prove an oversize sdist aborts before anything is uploaded.
"""

import subprocess
from pathlib import Path

from kiss.tests.conftest import posix_only

REPO_ROOT = Path(__file__).resolve().parents[4]
SIZE_TEST_SCRIPT = REPO_ROOT / "scripts" / "test_release_pypi_size.sh"

pytestmark = posix_only("runs the bash release PyPI size suite from scripts/")


def test_pypi_distributions_fit_and_oversize_aborts_upload() -> None:
    """The PyPI size suite must pass end to end."""
    result = subprocess.run(
        ["bash", str(SIZE_TEST_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, f"PyPI size suite failed:\n{result.stdout}\n{result.stderr}"
    assert "ALL TESTS PASSED" in result.stdout
    assert "oversize sdist aborts publish_to_pypi before uv publish" in result.stdout
