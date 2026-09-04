# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The curl bootstrap must un-brick clones with a pinned or unmerged VSIX.

Audit 2026-09-03 (vscode-installer): the diverged-pull recovery added to
``scripts/install.sh`` by commit 1aad2045b normalizes a stale rebuilt
``kiss-sorcar.vsix`` with ``git checkout HEAD -- <vsix>`` before stashing
and resetting to upstream.  That checkout silently fails (``2>/dev/null
|| true``) on an entry an OLD installer pinned with ``git update-index
--skip-worktree`` — git answers ``pathspec ... did not match any file(s)
known to git`` for a skip-worktree path — and the pin then breaks every
later step of the recovery in sequence:

1. ``git pull --ff-only`` fails ("Please commit your changes"): the
   pinned entry's blob changed upstream.
2. The checkout is skipped, so the pin and the stale bytes survive.
3. ``git status --porcelain`` is EMPTY (the pin hides the dirt), so
   nothing is stashed.
4. ``git reset --hard @{upstream}`` fails with "Entry ... not uptodate.
   Cannot merge." and the script "continues with the current checkout".
5. The OLD ``./install.sh`` (which predates ``guard_vsix_tracking`` —
   that is why the clone is pinned in the first place) runs and cannot
   heal the state either.

The curl one-liner (``curl .../scripts/install.sh | bash``) is the
advertised recovery path for exactly such bricked clones, so the fresh
bootstrap script itself must clear the pin — mirroring the first step of
``guard_vsix_tracking`` in the root ``install.sh``.  These tests run the
REAL ``scripts/install.sh`` against a throwaway ``$HOME`` whose
``~/.kiss/kiss_ai`` is a clone of a local bare origin (no network, no
mocks), following ``test_audit0902_fix_vscode_install_lock.py``.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "install.sh"
VSIX_REL = "src/kiss/agents/vscode/kiss-sorcar.vsix"
RELEASE_V1 = b"released-vsix-bytes-v1"
RELEASE_V2 = b"released-vsix-bytes-v2"
STALE = b"stale-locally-rebuilt-vsix-bytes"

# Stand-in for the clone's ./install.sh: records which release ran it.
STUB_INSTALL_SH = """#!/bin/bash
echo "{tag}" >> "$HOME/.kiss/install-ran"
exit 0
"""


def _git(repo: Path, *args: str, env: dict[str, str]) -> str:
    """Run git in *repo* with the sandbox env, returning stripped stdout."""
    result = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
         "-C", str(repo), *args],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class BootstrapUnbricksVsixTest(unittest.TestCase):
    """scripts/install.sh recovers clones an old installer left bricked."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-bootstrap-vsix-"))
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.env = {
            **{
                k: v
                for k, v in os.environ.items()
                if k not in ("KISS_HOME", "KISS_UPDATE_LOCK_HELD")
            },
            "HOME": str(self.home),
            "GIT_CONFIG_NOSYSTEM": "1",
        }
        self.origin = self.tmp / "origin.git"
        subprocess.run(
            ["git", "init", "--bare", "-q", "-b", "main", str(self.origin)],
            check=True, capture_output=True, env=self.env,
        )
        self.seed = self.tmp / "seed"
        (self.seed / Path(VSIX_REL).parent).mkdir(parents=True)
        _git(self.seed, "init", "-q", "-b", "main", env=self.env)
        (self.seed / ".gitignore").write_text("*.vsix\n")
        (self.seed / "install.sh").write_text(STUB_INSTALL_SH.format(tag="v1"))
        (self.seed / "install.sh").chmod(0o755)
        (self.seed / VSIX_REL).write_bytes(RELEASE_V1)
        _git(self.seed, "add", ".gitignore", "install.sh", env=self.env)
        _git(self.seed, "add", "-f", VSIX_REL, env=self.env)
        _git(self.seed, "commit", "-q", "-m", "Release v1", env=self.env)
        _git(self.seed, "push", "-q", str(self.origin), "main", env=self.env)
        (self.home / ".kiss").mkdir()
        self.clone = self.home / ".kiss" / "kiss_ai"
        subprocess.run(
            ["git", "clone", "-q", str(self.origin), str(self.clone)],
            check=True, capture_output=True, env=self.env,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _push_release_v2(self) -> None:
        """Publish release v2 (new VSIX blob, new install.sh stub tag)."""
        (self.seed / VSIX_REL).write_bytes(RELEASE_V2)
        (self.seed / "install.sh").write_text(STUB_INSTALL_SH.format(tag="v2"))
        _git(self.seed, "add", "install.sh", env=self.env)
        _git(self.seed, "add", "-f", VSIX_REL, env=self.env)
        _git(self.seed, "commit", "-q", "-m", "Release v2", env=self.env)
        _git(self.seed, "push", "-q", str(self.origin), "main", env=self.env)

    def _run_bootstrap(self) -> subprocess.CompletedProcess[str]:
        """Run the real scripts/install.sh in the sandbox HOME."""
        return subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=self.tmp,
            env=self.env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def _assert_updated_to_v2(
        self, result: subprocess.CompletedProcess[str]
    ) -> None:
        """The clone is at origin's v2 tip, clean, and ran v2's install.sh."""
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertEqual(
            _git(self.clone, "rev-parse", "HEAD", env=self.env),
            _git(self.origin, "rev-parse", "main", env=self.env),
            f"clone is not at origin's tip — the bootstrap did not "
            f"update:\n{output}",
        )
        self.assertEqual((self.clone / VSIX_REL).read_bytes(), RELEASE_V2)
        self.assertEqual(
            _git(self.clone, "status", "--porcelain", env=self.env), "",
        )
        self.assertEqual(
            _git(self.clone, "ls-files", "-v", "--", VSIX_REL, env=self.env),
            f"H {VSIX_REL}",
            "the skip-worktree pin must be cleared, or the NEXT "
            "`git reset --hard` bricks again",
        )
        self.assertEqual(
            (self.home / ".kiss" / "install-ran").read_text(),
            "v2\n",
            "the bootstrap handed over to the OLD install.sh — the "
            "update never happened",
        )

    def test_skip_worktree_pinned_stale_vsix_is_unbricked(self) -> None:
        """A clone whose VSIX an old installer pinned must still update.

        Reproduces the bricked state for real: stale rebuilt bytes on
        disk, hidden from git by ``update-index --skip-worktree`` (what
        an ancient installer did to silence the dirty file).  Upstream
        then publishes v2.  Without the fix the recovery's checkout,
        stash and reset ALL fail and the old v1 install.sh runs.
        """
        (self.clone / VSIX_REL).write_bytes(STALE)
        _git(self.clone, "update-index", "--skip-worktree", "--", VSIX_REL,
             env=self.env)
        self._push_release_v2()

        self._assert_updated_to_v2(self._run_bootstrap())

    def test_unmerged_vsix_stages_are_unbricked(self) -> None:
        """A clone with UU stages from an old installer's stash pop updates.

        Reproduces the legacy conflicted-pop state for real: the stale
        rebuild is stashed (what the old update flow did), v2 is pulled,
        and the pop conflicts — leaving unmerged stages that make ``git
        pull`` and ``git stash push`` fail.  The recovery's ``git
        checkout HEAD -- <vsix>`` must drop the stages so the reset (a
        no-op here, already at tip) and the v2 handover happen.  Then a
        v3 push must update again through the normal fast-forward path.
        """
        (self.clone / VSIX_REL).write_bytes(STALE)
        _git(self.clone, "stash", "push", "-q", "--include-untracked",
             "-m", "install.sh auto-stash", env=self.env)
        self._push_release_v2()
        _git(self.clone, "pull", "-q", "--ff-only", env=self.env)
        pop = subprocess.run(
            ["git", "-C", str(self.clone), "stash", "pop"],
            env=self.env, capture_output=True, text=True,
        )
        self.assertNotEqual(pop.returncode, 0, "expected the pop to conflict")
        self.assertNotEqual(
            _git(self.clone, "ls-files", "-u", env=self.env), "",
            "expected unmerged VSIX stages",
        )

        result = self._run_bootstrap()
        self._assert_updated_to_v2(result)
        self.assertEqual(
            _git(self.clone, "ls-files", "-u", env=self.env), "",
            "unmerged stages survived the bootstrap",
        )


if __name__ == "__main__":
    unittest.main()
