# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``install.sh`` must not abort in a public ``kiss_ai`` clone whose release
commits ship ``kiss-sorcar.vsix`` as a tracked file.

Regression context: scripts/release.sh deliberately injects the built VSIX
into every public release commit (``tree_with_vsix``) so docker/code-server
installs work without npm.  install.sh's older defence-in-depth check treated
*any* tracked VSIX as an error, so the VS Code "Update" button failed on every
public clone right after installing the extension::

    ERROR: .../kiss-sorcar.vsix is tracked by git but must remain ignored.

The fixed ``guard_vsix_tracking`` distinguishes the two repo kinds by whether
HEAD contains the VSIX:

* in HEAD (tracked or not) -> public clone, by design: restore HEAD's copy
  into both the index (``git update-index --index-info``: a mode-0 line drops
  every stage of the path, a stage-0 line re-registers HEAD's blob) and the
  working tree (``git checkout-index -f``), so the repo ends byte-for-byte
  clean.  The freshly built VSIX was already installed into VS Code before
  the guard runs, so nothing is lost.  A clean repo is essential: pinning the
  path ``--skip-worktree`` instead was demonstrated to brick every later
  update (``git reset --hard @{upstream}`` fails with "Entry ... not
  uptodate" once a new release changes the blob) and is not inherited by
  linked worktrees; ``git checkout HEAD --`` is also insufficient because it
  silently skips an unmerged entry whose stage-2 blob matches HEAD.  The
  index rewrite heals a staged modification, a staged ``git rm --cached``
  deletion, and unmerged stages left by a conflicted ``git stash pop``.
  Exit 0.
* tracked, NOT in HEAD -> accidental ``git add -f`` in the development repo:
  hard error (exit 1) with untracking instructions.
* untracked, not in HEAD -> healthy development repo: no-op.

The tests below extract ``guard_vsix_tracking`` verbatim from install.sh and
run it for real against sandbox git repositories covering every branch,
including the warning path, which is reached end-to-end by holding a real
``.git/index.lock`` (the state a concurrent git process leaves) while the
guard runs.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
INSTALL_SCRIPT = REPO / "install.sh"
VSIX_REL = "src/kiss/agents/vscode/kiss-sorcar.vsix"
RELEASE_V1 = b"released-vsix-bytes-v1"
RELEASE_V2 = b"released-vsix-bytes-v2"
REBUILT = b"freshly-rebuilt-vsix-bytes"


def extract_guard_function() -> str:
    """Return the ``guard_vsix_tracking`` function body from install.sh."""
    text = INSTALL_SCRIPT.read_text()
    match = re.search(r"^guard_vsix_tracking\(\) \{\n.*?^\}$", text, re.MULTILINE | re.DOTALL)
    assert match, "guard_vsix_tracking() not found in install.sh"
    return match.group(0)


def git(repo: Path, *args: str) -> str:
    """Run a git command in ``repo`` and return stripped stdout, failing loudly."""
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def make_repo(tmp_path: Path) -> Path:
    """Create a sandbox git repo with ``*.vsix`` gitignored, mirroring kiss_ai."""
    repo = tmp_path / "clone"
    (repo / Path(VSIX_REL).parent).mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    git(repo, "config", "user.email", "test@test.com")
    git(repo, "config", "user.name", "Test")
    (repo / ".gitignore").write_text("*.vsix\n")
    (repo / "README.md").write_text("kiss_ai sandbox\n")
    git(repo, "add", ".gitignore", "README.md")
    git(repo, "commit", "-q", "-m", "init")
    return repo


def make_public_clone(tmp_path: Path) -> Path:
    """Sandbox mirroring a public kiss_ai clone: a release commit ships the VSIX."""
    repo = make_repo(tmp_path)
    (repo / VSIX_REL).write_bytes(RELEASE_V1)
    git(repo, "add", "-f", VSIX_REL)
    git(repo, "commit", "-q", "-m", "Release 2026.9.1")
    return repo


def run_guard(repo: Path) -> subprocess.CompletedProcess[str]:
    """Run the extracted guard_vsix_tracking against ``repo`` exactly as install.sh does."""
    script = extract_guard_function() + f'\nguard_vsix_tracking "{repo}"\n'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def git_hash(repo: Path, data: bytes) -> str:
    """Write ``data`` as a blob into ``repo``'s object store; return its sha."""
    result = subprocess.run(
        ["git", "-C", str(repo), "hash-object", "-w", "--stdin"],
        input=data,
        capture_output=True,
        check=True,
    )
    return result.stdout.decode().strip()


def assert_fully_clean(repo: Path) -> None:
    """The guard must leave the repo byte-for-byte clean: empty status, index == HEAD."""
    assert git(repo, "status", "--porcelain") == ""
    assert git(repo, "diff", "--cached", "--name-only") == ""
    # The working tree holds exactly HEAD's release bytes.
    head_blob = git(repo, "rev-parse", f"HEAD:{VSIX_REL}")
    disk_blob = git_hash(repo, (repo / VSIX_REL).read_bytes())
    assert disk_blob == head_blob
    # No skip-worktree pin is left behind (it would break later hard resets).
    assert git(repo, "ls-files", "-v", "--", VSIX_REL).startswith("H ")


class TestPublicCloneShippedVsix:
    """VSIX present in HEAD: the release-shipped case (the regression)."""

    def test_update_button_regression_guard_succeeds(self, tmp_path: Path) -> None:
        """A public clone with a locally rebuilt VSIX passes the guard (exit 0).

        This is the exact state the Update button produced on the 2026.9.0
        machine: HEAD is a release commit shipping the VSIX, and install.sh
        just rebuilt the file on disk.  The pre-fix inline check exited 1 here.
        """
        repo = make_public_clone(tmp_path)
        (repo / VSIX_REL).write_bytes(REBUILT)
        result = run_guard(repo)
        assert result.returncode == 0, result.stderr
        assert "must remain ignored" not in result.stderr
        assert "Restored release-shipped" in result.stdout
        assert_fully_clean(repo)

    def test_auto_commit_flows_see_nothing_to_commit(self, tmp_path: Path) -> None:
        """After the guard, add -A stages nothing and the preflight stash is a no-op."""
        repo = make_public_clone(tmp_path)
        (repo / VSIX_REL).write_bytes(REBUILT)
        assert run_guard(repo).returncode == 0
        # The auto-commit / worktree flow's ``git add -A`` must not stage the binary.
        git(repo, "add", "-A")
        assert git(repo, "diff", "--cached", "--name-only") == ""
        # The Update button's preflight ``git stash push --include-untracked``
        # must find nothing to stash ("No local changes to save").
        stash = subprocess.run(
            ["git", "-C", str(repo), "stash", "push", "--include-untracked"],
            capture_output=True,
            text=True,
        )
        assert "No local changes" in stash.stdout + stash.stderr

    def test_next_update_reset_hard_succeeds(self, tmp_path: Path) -> None:
        """The Update button's preflight works after the guard, across releases.

        Regression check for the skip-worktree design rejected in review: a
        pinned VSIX whose blob changes upstream made ``git reset --hard
        @{upstream}`` fail with "Entry ... not uptodate. Cannot merge.",
        bricking every later update.  The guard must leave the clone in a
        state where fetch + stash + hard reset + stash pop all succeed and
        land on the new release.
        """
        remote = tmp_path / "remote.git"
        # -b main: the bare remote's HEAD must name the branch the clones use,
        # independent of the machine's init.defaultBranch setting.
        subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
        repo = make_public_clone(tmp_path)
        git(repo, "remote", "add", "origin", str(remote))
        git(repo, "push", "-q", "-u", "origin", "main")
        # install.sh rebuilt the VSIX, then the guard cleaned up.
        (repo / VSIX_REL).write_bytes(REBUILT)
        assert run_guard(repo).returncode == 0
        # A new release changes the shipped VSIX blob upstream.
        publisher = tmp_path / "publisher"
        subprocess.run(["git", "clone", "-q", str(remote), str(publisher)], check=True)
        git(publisher, "config", "user.email", "test@test.com")
        git(publisher, "config", "user.name", "Test")
        (publisher / VSIX_REL).write_bytes(RELEASE_V2)
        git(publisher, "add", "-f", VSIX_REL)
        git(publisher, "commit", "-q", "-m", "Release 2026.9.2")
        git(publisher, "push", "-q", "origin", "main")
        # The modern bootstrap (scripts/install.sh) tries a fast-forward pull
        # first; it only works when the guard left the clone byte-for-byte
        # clean (a dirty tracked VSIX aborts with "local changes ... would be
        # overwritten by merge").
        git(repo, "pull", "--ff-only", "origin", "main")
        assert (repo / VSIX_REL).read_bytes() == RELEASE_V2
        # The legacy Update button preflight (SorcarSidebarView.runUpdate
        # fallback): fetch + stash-if-dirty + hard reset must also succeed.
        git(repo, "reset", "--hard", "HEAD~1")  # back to Release 2026.9.1
        git(repo, "fetch", "--force", "--tags", "--prune", "origin")
        if git(repo, "status", "--porcelain"):
            git(repo, "stash", "push", "--include-untracked", "-m", "kiss-update-preflight")
        git(repo, "reset", "--hard", "@{upstream}")
        assert (repo / VSIX_REL).read_bytes() == RELEASE_V2
        assert git(repo, "rev-parse", "HEAD") == git(repo, "rev-parse", "origin/main")

    def test_staged_vsix_change_is_reset_to_head(self, tmp_path: Path) -> None:
        """A staged VSIX modification (e.g. a user ``git add``) is reset to HEAD."""
        repo = make_public_clone(tmp_path)
        (repo / VSIX_REL).write_bytes(REBUILT)
        git(repo, "add", "-f", VSIX_REL)
        assert git(repo, "diff", "--cached", "--name-only") == VSIX_REL
        assert run_guard(repo).returncode == 0
        assert_fully_clean(repo)

    def test_staged_rm_cached_deletion_is_healed(self, tmp_path: Path) -> None:
        """A staged deletion from ``git rm --cached`` (the old error's advice) is undone.

        The pre-fix error message told users to run ``git rm --cached`` on the
        VSIX.  In a public clone that leaves a staged deletion which the
        auto-commit flow would commit, deleting the release-shipped file from
        the user's local main.  The guard must restore the tracked entry.
        """
        repo = make_public_clone(tmp_path)
        git(repo, "rm", "-q", "--cached", VSIX_REL)
        assert git(repo, "diff", "--cached", "--name-only") == VSIX_REL
        assert run_guard(repo).returncode == 0
        assert_fully_clean(repo)

    def test_conflicted_vsix_entry_is_healed(self, tmp_path: Path) -> None:
        """An unmerged VSIX index entry (failed ``git stash pop``) is resolved.

        On a machine that hit the original error, the next update's preflight
        can leave the VSIX in a conflicted (unmerged) index state when the
        stashed local rebuild collides with a new release blob.  The guard
        must restore the entry to HEAD and continue.
        """
        repo = make_public_clone(tmp_path)
        # Build realistic unmerged stages: stage 1 = common base, stage 2 =
        # ours (HEAD, the new release blob), stage 3 = theirs (stashed rebuild).
        blob_base = git_hash(repo, b"released-vsix-bytes-v0")
        blob_ours = git(repo, "rev-parse", f"HEAD:{VSIX_REL}")
        blob_theirs = git_hash(repo, REBUILT)
        index_info = (
            f"100644 {blob_base} 1\t{VSIX_REL}\n"
            f"100644 {blob_ours} 2\t{VSIX_REL}\n"
            f"100644 {blob_theirs} 3\t{VSIX_REL}\n"
        )
        subprocess.run(
            ["git", "-C", str(repo), "update-index", "--index-info"],
            input=index_info,
            text=True,
            check=True,
        )
        assert git(repo, "ls-files", "-u", "--", VSIX_REL) != ""
        result = run_guard(repo)
        assert result.returncode == 0, result.stderr
        assert git(repo, "ls-files", "-u", "--", VSIX_REL) == ""
        assert_fully_clean(repo)

    def test_stale_skip_worktree_pin_is_cleared(self, tmp_path: Path) -> None:
        """A leftover skip-worktree pin on the VSIX is removed by the guard."""
        repo = make_public_clone(tmp_path)
        git(repo, "update-index", "--skip-worktree", "--", VSIX_REL)
        (repo / VSIX_REL).write_bytes(REBUILT)
        assert run_guard(repo).returncode == 0
        assert_fully_clean(repo)

    def test_index_lock_yields_warning_not_false_success(self, tmp_path: Path) -> None:
        """A held ``.git/index.lock`` produces the warning, never false success.

        A concurrent git process holds the index lock, so the guard's index
        rewrite cannot happen and a wrong staged blob survives.  The guard
        must report the warning (and still exit 0: a dirty VSIX only means
        the next update's preflight stash has something to stash).
        """
        repo = make_public_clone(tmp_path)
        (repo / VSIX_REL).write_bytes(REBUILT)
        git(repo, "add", "-f", VSIX_REL)
        (repo / ".git" / "index.lock").touch()
        result = run_guard(repo)
        (repo / ".git" / "index.lock").unlink()
        assert result.returncode == 0
        assert "WARNING" in result.stderr
        assert "Restored release-shipped" not in result.stdout

    def test_stale_pin_with_index_lock_warns(self, tmp_path: Path) -> None:
        """A surviving skip-worktree pin is reported, not hidden as success.

        With the index locked, the pin cannot be cleared and ``git status``
        (which the pin blinds) alone would look clean.  The verification's
        ``ls-files -v`` check must catch the surviving ``S`` flag and emit
        the warning instead of false success.
        """
        repo = make_public_clone(tmp_path)
        git(repo, "update-index", "--skip-worktree", "--", VSIX_REL)
        (repo / VSIX_REL).write_bytes(REBUILT)
        (repo / ".git" / "index.lock").touch()
        result = run_guard(repo)
        (repo / ".git" / "index.lock").unlink()
        assert result.returncode == 0
        assert "WARNING" in result.stderr
        assert "Restored release-shipped" not in result.stdout


class TestDevelopmentRepo:
    """VSIX not shipped by any commit: the development-repo cases."""

    def test_untracked_vsix_is_a_noop(self, tmp_path: Path) -> None:
        """The healthy dev-repo state (gitignored, untracked) passes silently."""
        repo = make_repo(tmp_path)
        (repo / VSIX_REL).write_bytes(b"locally-built-vsix")
        result = run_guard(repo)
        assert result.returncode == 0, result.stderr
        assert result.stdout == ""
        assert result.stderr == ""
        assert git(repo, "status", "--porcelain") == ""
        # The locally built VSIX stays on disk for install retries.
        assert (repo / VSIX_REL).read_bytes() == b"locally-built-vsix"

    def test_accidentally_force_added_vsix_still_errors(self, tmp_path: Path) -> None:
        """``git add -f`` of the VSIX (not in HEAD) keeps the hard error."""
        repo = make_repo(tmp_path)
        (repo / VSIX_REL).write_bytes(b"locally-built-vsix")
        git(repo, "add", "-f", VSIX_REL)
        result = run_guard(repo)
        assert result.returncode == 1
        assert "must remain ignored" in result.stderr
        assert "rm --cached" in result.stderr


def test_install_sh_calls_the_guard() -> None:
    """install.sh step [5/5] must delegate to guard_vsix_tracking and exit on failure."""
    text = INSTALL_SCRIPT.read_text()
    assert re.search(
        r"if ! guard_vsix_tracking \"\$PROJECT_DIR\"; then\n\s*exit 1\n\s*fi", text
    ), "install.sh must call guard_vsix_tracking with PROJECT_DIR and exit 1 on failure"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
