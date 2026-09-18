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
``./rsorcar`` against a stubbed remote (ssh/scp/curl answer what a healthy
deploy sees) and verifies that the deploy ends by running ``./install.sh`` on
the local machine, after the summary box and before "Done." — and that a
failing local install fails the deploy without hiding the remote URL and
password.  The same run copies a fake ``~/.ssh`` as a tar stream through the
ssh stub into the real ``scripts/install-ssh-identity.sh``: the deploy must
not need rsync on the remote (a fresh Debian image has none).

Runs ``scripts/test_install_ssh_identity.sh``, which exercises that remote
half on its own: files land under their relative paths, replaced files with
different content are kept in ``$SSH_BACKUP``, ``authorized_keys`` is never
touched, permissions are fixed, and a missing ``SSH_BACKUP`` or ``tar`` fails
with a clear message.
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


def test_install_ssh_identity_receives_the_ssh_copy() -> None:
    """install-ssh-identity.sh must install a tar stream into ~/.ssh, keeping replaced files."""
    run_suite("test_install_ssh_identity.sh")
