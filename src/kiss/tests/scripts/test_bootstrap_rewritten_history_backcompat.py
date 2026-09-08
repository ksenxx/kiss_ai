# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The curl bootstrap must update clones left by v2026.9.0 installs.

Backward compatibility with v2026.9.0 (2026-09-08): every release now
rewrites the public repo's history and re-points EVERY existing tag at
the rewritten commits (``purge_public_history`` in scripts/release.sh)
before force-pushing.  A machine whose last installation was v2026.9.0
(or any pre-rewrite release) therefore holds a clone whose branch AND
tags all diverge from the rewritten public repo.  On such a clone the
bootstrap's diverged-pull recovery used to die in sequence:

1. ``git pull --ff-only`` fails (forced-pushed history, no fast-forward).
2. The recovery's ``git fetch --tags --prune origin`` exits non-zero
   with ``! [rejected] ... (would clobber existing tag)`` for every
   re-pointed tag, which the script misread as "offline".
3. ``git reset --hard @{upstream}`` is therefore SKIPPED.
4. The handover runs the OLD checkout's ./install.sh — the update
   silently never happens, forever.

The fix adds ``--force`` to the recovery fetch: the clone is a managed
install mirror of the public repo, so re-pointing its local tags is
always correct.  These tests run the REAL ``scripts/install.sh``
against a throwaway ``$HOME`` whose ``~/.kiss/kiss_ai`` is a clone of a
local bare origin (no network, no mocks), following
``test_audit0903_bootstrap_pinned_vsix_unbrick.py``.
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
RELEASE_V2 = b"released-vsix-bytes-v2"
LOCAL_BUILD = b"locally-built-vsix-bytes"

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


class BootstrapRewrittenHistoryTest(unittest.TestCase):
    """scripts/install.sh updates v2026.9.0-era clones across a history rewrite."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-bootstrap-90-"))
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
        # Release v1 mirrors the public repo at v2026.9.0: the VSIX is NOT
        # tracked (it was a gitignored local build artifact back then), and
        # version tags point at the pre-rewrite history.
        self.seed = self.tmp / "seed"
        (self.seed / Path(VSIX_REL).parent).mkdir(parents=True)
        _git(self.seed, "init", "-q", "-b", "main", env=self.env)
        (self.seed / ".gitignore").write_text("*.vsix\n")
        (self.seed / "install.sh").write_text(STUB_INSTALL_SH.format(tag="v1"))
        (self.seed / "install.sh").chmod(0o755)
        _git(self.seed, "add", ".gitignore", "install.sh", env=self.env)
        _git(self.seed, "commit", "-q", "-m", "Release 2026.9.0", env=self.env)
        _git(self.seed, "tag", "v0.1.10", env=self.env)
        _git(self.seed, "tag", "v2026.9.0", env=self.env)
        _git(self.seed, "push", "-q", "--tags", str(self.origin), "main",
             env=self.env)
        # The v2026.9.0 install: a clone of that release (tags included)
        # plus the locally built, untracked-and-ignored VSIX on disk.
        (self.home / ".kiss").mkdir()
        self.clone = self.home / ".kiss" / "kiss_ai"
        subprocess.run(
            ["git", "clone", "-q", str(self.origin), str(self.clone)],
            check=True, capture_output=True, env=self.env,
        )
        # Empty directories are not cloned, so recreate the build dir the
        # old installer's npm package step would have created.
        (self.clone / VSIX_REL).parent.mkdir(parents=True)
        (self.clone / VSIX_REL).write_bytes(LOCAL_BUILD)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _publish_rewritten_release(self) -> None:
        """Force-push a rewritten history and re-point every tag at it.

        Mirrors what purge_public_history + a new release do to the public
        repo: main becomes an orphan "Release v2" commit (which now SHIPS
        the VSIX tracked, like every post-9.0 release), and the existing
        tags are re-pointed at rewritten commits.
        """
        _git(self.seed, "checkout", "-q", "--orphan", "rewrite", env=self.env)
        (self.seed / "install.sh").write_text(STUB_INSTALL_SH.format(tag="v2"))
        (self.seed / VSIX_REL).write_bytes(RELEASE_V2)
        _git(self.seed, "add", ".gitignore", "install.sh", env=self.env)
        _git(self.seed, "add", "-f", VSIX_REL, env=self.env)
        _git(self.seed, "commit", "-q", "-m", "Release 2026.9.9", env=self.env)
        _git(self.seed, "tag", "-f", "v0.1.10", env=self.env)
        _git(self.seed, "tag", "-f", "v2026.9.0", env=self.env)
        _git(self.seed, "push", "-q", "--force", str(self.origin),
             "rewrite:main", env=self.env)
        _git(self.seed, "push", "-q", "--force", "--tags", str(self.origin),
             env=self.env)

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

    def test_v2026_9_0_clone_updates_across_history_rewrite(self) -> None:
        """A v2026.9.0-era clone must reset to the rewritten release tip.

        Without ``--force`` on the recovery fetch, the re-pointed tags
        make the fetch fail ("would clobber existing tag"), the reset is
        skipped, and the OLD v1 install.sh runs against the OLD tree.
        """
        self._publish_rewritten_release()

        result = self._run_bootstrap()
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertEqual(
            _git(self.clone, "rev-parse", "HEAD", env=self.env),
            _git(self.origin, "rev-parse", "main", env=self.env),
            f"clone is not at the rewritten tip — the bootstrap did not "
            f"update:\n{output}",
        )
        self.assertEqual(
            (self.clone / VSIX_REL).read_bytes(), RELEASE_V2,
            "the release-shipped VSIX must replace the old local build",
        )
        self.assertEqual(
            _git(self.clone, "status", "--porcelain", env=self.env), "",
        )
        self.assertEqual(
            (self.home / ".kiss" / "install-ran").read_text(),
            "v2\n",
            "the bootstrap handed over to the OLD install.sh — the "
            "update never happened",
        )
        # The re-pointed tags must now match the rewritten remote, or the
        # NEXT release's rewrite clobbers them all over again.
        for tag in ("v0.1.10", "v2026.9.0"):
            self.assertEqual(
                _git(self.clone, "rev-parse", tag, env=self.env),
                _git(self.origin, "rev-parse", tag, env=self.env),
                f"tag {tag} was not re-pointed to the rewritten history",
            )

    def test_unreachable_origin_still_hands_over_to_current_checkout(
        self,
    ) -> None:
        """With origin gone, the bootstrap warns and runs the CURRENT tree.

        This pins the other branch of the recovery's fetch condition: a
        fetch that fails for real (origin unreachable) must keep the
        "continue with the current checkout" behaviour instead of
        aborting the install.
        """
        shutil.rmtree(self.origin)

        result = self._run_bootstrap()
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("git fetch failed", output)
        self.assertEqual(
            (self.home / ".kiss" / "install-ran").read_text(),
            "v1\n",
            "the current checkout's install.sh must still run offline",
        )


if __name__ == "__main__":
    unittest.main()
