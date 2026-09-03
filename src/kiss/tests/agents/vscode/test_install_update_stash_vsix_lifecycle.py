# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The full ``install.sh`` update lifecycle must not brick later updates.

Review 2026-09-03 (gpt-5.6-sol, Update-button follow-up): the
``guard_vsix_tracking`` fix left the repo clean at step [5/5], but the
COMPLETE installer lifecycle broke it again:

1. ``update_repo`` stashes a dirty tree — including a stale rebuilt
   ``kiss-sorcar.vsix`` left by an older install — and registers
   ``restore_stashed_changes`` as an EXIT trap.
2. The pull/reset brings in a release whose tracked VSIX blob differs.
3. ``guard_vsix_tracking`` restores the release VSIX (clean tree).
4. The EXIT trap then runs ``git stash pop``, which CONFLICTS on the
   VSIX, leaving unmerged (UU) stages behind.
5. The NEXT Update-button run hits ``git stash push`` -> ``needs merge``,
   ``update_repo`` warns and returns WITHOUT pulling — the Update button
   installs the same version forever.

The fix is two-sided, and these tests exercise both sides end-to-end
with real git repositories and the real functions extracted verbatim
from ``install.sh`` (no mocks):

* ``update_repo`` normalizes the VSIX (restores HEAD's copy) BEFORE the
  dirty check, so a stale rebuild never enters the stash — and a clone
  already bricked by an older installer (unmerged VSIX stages in the
  index) is healed before the dirty check, so the pull happens again.
* ``restore_stashed_changes`` heals a conflicted pop from a LEGACY stash
  (created by an older install.sh that did stash the VSIX): it restores
  the release VSIX and drops the auto-stash when the VSIX was the only
  conflict.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
INSTALL_SCRIPT = REPO / "install.sh"
VSIX_REL = "src/kiss/agents/vscode/kiss-sorcar.vsix"
RELEASE_V1 = b"released-vsix-bytes-v1"
RELEASE_V2 = b"released-vsix-bytes-v2"
RELEASE_V3 = b"released-vsix-bytes-v3"
REBUILT = b"stale-locally-rebuilt-vsix-bytes"


def extract_function(name: str) -> str:
    """Return the named shell function's source verbatim from install.sh."""
    text = INSTALL_SCRIPT.read_text()
    match = re.search(
        rf"^{name}\(\) \{{\n.*?^\}}$", text, re.MULTILINE | re.DOTALL
    )
    assert match, f"{name}() not found in install.sh"
    return match.group(0)


def git(repo: Path, *args: str) -> str:
    """Run git in *repo*, return stripped stdout, failing loudly."""
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def make_public_clone_with_origin(tmp_path: Path) -> tuple[Path, Path]:
    """A local clone of a bare origin whose release commit ships the VSIX.

    Mirrors a user's ``~/.kiss/kiss_ai``: origin holds release v1 with a
    tracked ``kiss-sorcar.vsix`` (``*.vsix`` gitignored, force-added by
    the release script), and the clone tracks ``origin/main``.
    """
    origin = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "--bare", "-q", "-b", "main", str(origin)],
        check=True,
        capture_output=True,
    )
    seed = tmp_path / "seed"
    (seed / Path(VSIX_REL).parent).mkdir(parents=True)
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(seed)],
        check=True,
        capture_output=True,
    )
    git(seed, "config", "user.email", "test@test.com")
    git(seed, "config", "user.name", "Test")
    (seed / ".gitignore").write_text("*.vsix\n")
    (seed / "README.md").write_text("kiss_ai sandbox\n")
    (seed / VSIX_REL).write_bytes(RELEASE_V1)
    git(seed, "add", ".gitignore", "README.md")
    git(seed, "add", "-f", VSIX_REL)
    git(seed, "commit", "-q", "-m", "Release v1")
    git(seed, "push", "-q", str(origin), "main")
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(origin), str(clone)],
        check=True,
        capture_output=True,
    )
    git(clone, "config", "user.email", "test@test.com")
    git(clone, "config", "user.name", "Test")
    return clone, origin


def push_release(tmp_path: Path, origin: Path, vsix: bytes, tag: str) -> None:
    """Publish a new release commit (new VSIX blob) to *origin*."""
    seed = tmp_path / "seed"
    (seed / VSIX_REL).write_bytes(vsix)
    (seed / "README.md").write_text(f"kiss_ai sandbox {tag}\n")
    git(seed, "add", "-f", VSIX_REL)
    git(seed, "add", "README.md")
    git(seed, "commit", "-q", "-m", f"Release {tag}")
    git(seed, "push", "-q", str(origin), "main")


def run_install_lifecycle(clone: Path) -> subprocess.CompletedProcess[str]:
    """Run install.sh's update lifecycle for real against *clone*.

    Executes, verbatim from install.sh: ``update_repo`` (stash + pull +
    EXIT trap), then ``guard_vsix_tracking`` exactly as step [5/5] does,
    then exits — firing the EXIT trap's ``git stash pop`` last, exactly
    like a real install run.
    """
    script = "\n".join(
        [
            "set -eo pipefail",
            f'PROJECT_DIR="{clone}"',
            "STASHED_CHANGES=0",
            extract_function("guard_vsix_tracking"),
            extract_function("restore_stashed_changes"),
            extract_function("update_repo"),
            "update_repo",
            'guard_vsix_tracking "$PROJECT_DIR"',
        ]
    )
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True
    )


def assert_healthy_at(clone: Path, origin: Path, vsix: bytes) -> None:
    """The clone is at origin's tip, byte-for-byte clean, VSIX restored."""
    assert git(clone, "status", "--porcelain") == "", (
        "repo left dirty after the install lifecycle:\n"
        + git(clone, "status", "--porcelain")
    )
    assert git(clone, "ls-files", "-u") == "", (
        "unmerged index entries survived the install lifecycle"
    )
    assert (clone / VSIX_REL).read_bytes() == vsix
    assert git(clone, "rev-parse", "HEAD") == git(
        origin, "rev-parse", "main"
    ), "clone is not at origin's tip — the update did not update"


def test_stale_rebuilt_vsix_never_conflicts_and_next_update_still_pulls(
    tmp_path: Path,
) -> None:
    """A dirty VSIX + new release must leave a clean repo AND a working
    next update.

    This is the exact lifecycle that bricked the Update button: the
    stale rebuild used to be stashed, the EXIT-trap pop conflicted with
    the freshly restored release VSIX, and every later ``update_repo``
    failed its ``git stash push`` with "needs merge" and skipped the
    pull.
    """
    clone, origin = make_public_clone_with_origin(tmp_path)
    (clone / VSIX_REL).write_bytes(REBUILT)  # stale rebuild, dirty
    push_release(tmp_path, origin, RELEASE_V2, "v2")

    first = run_install_lifecycle(clone)
    assert first.returncode == 0, first.stdout + first.stderr
    assert_healthy_at(clone, origin, RELEASE_V2)
    assert git(clone, "stash", "list") == "", (
        "a leftover stash entry survived the install lifecycle"
    )

    # The next Update-button run must still actually update.
    push_release(tmp_path, origin, RELEASE_V3, "v3")
    second = run_install_lifecycle(clone)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "needs merge" not in second.stdout + second.stderr
    assert_healthy_at(clone, origin, RELEASE_V3)


def test_user_edits_survive_while_vsix_is_normalized(tmp_path: Path) -> None:
    """Real local edits are stashed and restored; the VSIX is not.

    The VSIX normalization must be surgical: an edited tracked file
    (``.gitignore``) and an untracked note go through the stash/pop
    cycle and come back, while the stale VSIX is replaced by the release
    copy without ever entering the stash.
    """
    clone, origin = make_public_clone_with_origin(tmp_path)
    (clone / "notes.txt").write_text("my local untracked notes\n")
    (clone / ".gitignore").write_text("*.vsix\n*.log\n")  # tracked edit
    (clone / VSIX_REL).write_bytes(REBUILT)
    push_release(tmp_path, origin, RELEASE_V2, "v2")

    result = run_install_lifecycle(clone)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (clone / "notes.txt").read_text() == "my local untracked notes\n"
    assert (clone / ".gitignore").read_text() == "*.vsix\n*.log\n", (
        "the tracked local edit was not restored from the stash"
    )
    assert (clone / VSIX_REL).read_bytes() == RELEASE_V2
    assert git(clone, "ls-files", "-u") == ""
    assert git(clone, "rev-parse", "HEAD") == git(origin, "rev-parse", "main")


def test_dev_repo_force_added_vsix_fails_fast(tmp_path: Path) -> None:
    """A development repo with a force-added VSIX must abort loudly.

    The step [5/5] guard hard-errors on that state — but ``update_repo``
    used to stash the staged binary first, hiding it from that guard,
    and the EXIT-trap pop restored it afterwards: the promised error was
    silently bypassed.  The early normalization must therefore fail
    BEFORE the stash, with the guard's own untracking instructions.
    """
    dev = tmp_path / "dev"
    (dev / Path(VSIX_REL).parent).mkdir(parents=True)
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(dev)],
        check=True,
        capture_output=True,
    )
    git(dev, "config", "user.email", "test@test.com")
    git(dev, "config", "user.name", "Test")
    (dev / ".gitignore").write_text("*.vsix\n")
    (dev / "README.md").write_text("dev repo\n")
    git(dev, "add", ".gitignore", "README.md")
    git(dev, "commit", "-q", "-m", "init")
    (dev / VSIX_REL).write_bytes(REBUILT)
    git(dev, "add", "-f", VSIX_REL)  # the accidental force-add

    result = run_install_lifecycle(dev)
    assert result.returncode != 0, (
        "a force-added VSIX in the development repo must abort the "
        f"install:\n{result.stdout}{result.stderr}"
    )
    assert "must remain ignored" in result.stdout + result.stderr
    # The staged binary is untouched for the user to untrack as told.
    assert git(dev, "diff", "--cached", "--name-only") == VSIX_REL


def test_conflicted_pop_keeps_stash_and_heals_vsix(tmp_path: Path) -> None:
    """A conflicted EXIT-trap pop heals the VSIX but NEVER drops the stash.

    Exercises ``restore_stashed_changes``'s recovery branch directly
    (real git, the real function): a stash made by an OLDER installer
    carries a stale VSIX plus an untracked note; by pop time a same-named
    note exists in the tree, so the pop fails without restoring the
    stashed one.  The VSIX must come back byte-for-byte from HEAD, the
    conflicting on-disk note must be untouched, and the stash must be
    KEPT — dropping it would lose the stashed note for good.
    """
    clone, origin = make_public_clone_with_origin(tmp_path)
    (clone / VSIX_REL).write_bytes(REBUILT)
    (clone / "notes.txt").write_text("older stashed notes\n")
    git(clone, "stash", "push", "--include-untracked", "-m", "install.sh auto-stash")
    push_release(tmp_path, origin, RELEASE_V2, "v2")
    git(clone, "pull", "--ff-only", "-q")
    (clone / "notes.txt").write_text("current work, must not be clobbered\n")

    script = "\n".join(
        [
            f'PROJECT_DIR="{clone}"',
            "STASHED_CHANGES=1",
            extract_function("guard_vsix_tracking"),
            extract_function("restore_stashed_changes"),
            "restore_stashed_changes",
        ]
    )
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (clone / VSIX_REL).read_bytes() == RELEASE_V2
    assert git(clone, "ls-files", "-u") == "", "VSIX left unmerged"
    assert (
        (clone / "notes.txt").read_text()
        == "current work, must not be clobbered\n"
    )
    stash_list = git(clone, "stash", "list")
    assert "install.sh auto-stash" in stash_list, (
        "the stash was dropped after a failed pop — the stashed "
        "untracked note is unrecoverable"
    )


def test_clone_bricked_by_older_installer_is_healed_and_updates(
    tmp_path: Path,
) -> None:
    """A clone left with unmerged VSIX stages by an OLD install.sh must
    update again.

    Users hit by the original bug have repos where a conflicted
    ``git stash pop`` already left UU stages on the VSIX plus the kept
    auto-stash.  ``update_repo`` must heal that state up front so its
    dirty-check stash succeeds and the pull happens.
    """
    clone, origin = make_public_clone_with_origin(tmp_path)

    # Reproduce the legacy bricked state for real: dirty VSIX, stash it
    # (what the old update_repo did), pull the new release, restore the
    # release VSIX (what the guard does), then pop -> conflict.
    (clone / VSIX_REL).write_bytes(REBUILT)
    git(clone, "stash", "push", "--include-untracked", "-m", "install.sh auto-stash")
    push_release(tmp_path, origin, RELEASE_V2, "v2")
    git(clone, "pull", "--ff-only", "-q")
    pop = subprocess.run(
        ["git", "-C", str(clone), "stash", "pop"],
        capture_output=True,
        text=True,
    )
    assert pop.returncode != 0, "expected the legacy pop to conflict"
    assert git(clone, "ls-files", "-u") != "", "expected unmerged VSIX stages"

    push_release(tmp_path, origin, RELEASE_V3, "v3")
    result = run_install_lifecycle(clone)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "needs merge" not in result.stdout + result.stderr
    assert_healthy_at(clone, origin, RELEASE_V3)
