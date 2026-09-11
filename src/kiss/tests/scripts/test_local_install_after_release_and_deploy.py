# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end checks that release.sh and rsorcar end by running install.sh.

Runs ``scripts/test_release_local_install.sh``, which executes the real
``main()`` of ``scripts/release.sh`` in a scratch repository (publishing
stubbed, remotes local) and verifies that the release runs ``./install.sh``
non-interactively before restoring the pre-release stash — and that a failing
install aborts the release with the stash restored.

Runs ``scripts/test_rsorcar_local_install.sh``, which executes the real
``./rsorcar`` against a stubbed remote (ssh/scp/rsync/curl answer what a
healthy deploy sees) and verifies that the deploy ends by running
``./install.sh`` on the local machine, after the summary box and before
"Done." — and that a failing local install fails the deploy without hiding
the remote URL and password.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def run_suite(script_name: str) -> None:
    """Run one shell test suite from scripts/ and assert it passes fully."""
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / script_name)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, f"{script_name} failed:\n{result.stdout}\n{result.stderr}"
    assert "ALL TESTS PASSED" in result.stdout


def test_release_runs_local_install_before_finishing() -> None:
    """release.sh must run ./install.sh after publishing, before the stash restore."""
    run_suite("test_release_local_install.sh")


def test_rsorcar_runs_local_install_before_finishing() -> None:
    """rsorcar must run ./install.sh locally after the deploy summary, before Done."""
    run_suite("test_rsorcar_local_install.sh")
