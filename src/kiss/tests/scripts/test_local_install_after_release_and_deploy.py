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
not need rsync on the remote (a fresh Debian image has none).  The same run
also feeds the real ``scripts/install-remote-prereqs.sh`` to the stub's
``bash -s`` before the git sync, and checks that a remote where git cannot be
installed stops the deploy there.

Runs ``scripts/test_install_ssh_identity.sh``, which exercises that remote
half on its own: files land under their relative paths, replaced files with
different content are kept in ``$SSH_BACKUP``, ``authorized_keys`` is never
touched, permissions are fixed, and a missing ``SSH_BACKUP`` or ``tar`` fails
with a clear message.

Runs ``scripts/test_install_remote_prereqs.sh``, which exercises the
prerequisite installer on its own: a complete host is left alone, a real
``debian:13`` container (when docker is usable) gets git, curl, python3 and
the ssh client installed as root, a non-root host goes through ``sudo -n``
and a non-interactive apt-get, and the failure modes (install fails, tool
still missing, sudo wants a password, no package manager) each stop with a
clear message; dnf, pacman and apk get their own package names.

Runs ``scripts/test_check_remote_disk_space.sh`` (the room check rsorcar runs
on the remote before anything of size travels: stale upload files removed, a
full disk refused with the other filesystem and the move command named),
``scripts/test_wait_for_public_url.sh`` (the end of the remote bootstrap: a
tunnel URL the remote cannot resolve itself is a warning with the resolver
cache flushed, not a failed deploy) and ``scripts/test_move_home_to_disk.sh``
(the bind-mount move of a home directory onto a bigger disk, in a privileged
``debian:13`` container; the suite skips itself where docker is unusable).
"""

import subprocess
from pathlib import Path

from kiss.tests.conftest import posix_only

REPO_ROOT = Path(__file__).resolve().parents[4]

pytestmark = posix_only("runs the bash test suites under scripts/")


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
    # A suite that needs docker says "SKIP:" and exits 0 where there is none.
    assert "ALL TESTS PASSED" in result.stdout or result.stdout.startswith("SKIP:"), result.stdout


def test_release_runs_local_install_before_finishing() -> None:
    """release.sh must run ./install.sh after publishing, before the stash restore."""
    run_suite("test_release_local_install.sh")


def test_rsorcar_runs_local_install_before_finishing() -> None:
    """rsorcar must run ./install.sh locally after the deploy summary, before Done."""
    run_suite("test_rsorcar_local_install.sh")


def test_install_ssh_identity_receives_the_ssh_copy() -> None:
    """install-ssh-identity.sh must install a tar stream into ~/.ssh, keeping replaced files."""
    run_suite("test_install_ssh_identity.sh")


def test_install_remote_prereqs_installs_git_before_the_sync() -> None:
    """install-remote-prereqs.sh must install the missing tools (git above all) or stop clearly."""
    run_suite("test_install_remote_prereqs.sh")


def test_check_remote_disk_space_refuses_a_full_disk_with_the_fix_named() -> None:
    """check-remote-disk-space.sh must remove stale uploads and name the disk to move to."""
    run_suite("test_check_remote_disk_space.sh")


def test_wait_for_public_url_does_not_fail_over_the_remotes_dns_cache() -> None:
    """wait-for-public-url.sh must warn, not fail, when only the remote cannot resolve its URL."""
    run_suite("test_wait_for_public_url.sh")


def test_move_home_to_disk_bind_mounts_the_home_onto_the_disk() -> None:
    """move-home-to-disk.sh must move a home onto a disk and bring it back at the same path."""
    run_suite("test_move_home_to_disk.sh")
