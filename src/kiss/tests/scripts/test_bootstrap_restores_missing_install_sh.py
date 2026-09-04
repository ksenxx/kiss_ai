# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The curl bootstrap must heal a checkout whose install.sh is not a file.

The Update buttons (VS Code extension and remote webapp) fall back to
``scripts/install.sh`` whenever ``findInstallScript()`` returns null —
which is true not only when ``~/.kiss/kiss_ai`` is absent, but also when
the checkout EXISTS and its root ``install.sh`` was deleted locally or a
stray directory took its name.  In those states a clean ``git pull
--ff-only`` is a no-op that restores nothing, and the handover would run
a script that is not there.  The bootstrap must restore ``install.sh``
from git HEAD before handing over.

These tests run the REAL ``scripts/install.sh`` against a throwaway
``$HOME`` whose ``~/.kiss/kiss_ai`` is a clone of a local bare origin (no
network, no mocks), following
``test_audit0903_bootstrap_pinned_vsix_unbrick.py``.  The final reclone
branch (git restore impossible AND ``install.sh`` still not a file) is
not covered here: it clones the public GitHub repo and therefore cannot
run without network access.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "install.sh"

# Stand-in for the clone's ./install.sh: records that it ran.
STUB_INSTALL_SH = """#!/bin/bash
echo "ran" >> "$HOME/.kiss/install-ran"
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


class BootstrapRestoresInstallShTest(unittest.TestCase):
    """scripts/install.sh restores a missing/non-file root install.sh."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-bootstrap-missing-"))
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
        seed = self.tmp / "seed"
        seed.mkdir()
        _git(seed, "init", "-q", "-b", "main", env=self.env)
        (seed / "install.sh").write_text(STUB_INSTALL_SH)
        (seed / "install.sh").chmod(0o755)
        _git(seed, "add", "install.sh", env=self.env)
        _git(seed, "commit", "-q", "-m", "Release v1", env=self.env)
        _git(seed, "push", "-q", str(self.origin), "main", env=self.env)
        (self.home / ".kiss").mkdir()
        self.clone = self.home / ".kiss" / "kiss_ai"
        subprocess.run(
            ["git", "clone", "-q", str(self.origin), str(self.clone)],
            check=True, capture_output=True, env=self.env,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_bootstrap_and_assert_healed(self) -> None:
        """Run the real bootstrap; install.sh must be back and have run."""
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=self.tmp,
            env=self.env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertTrue(
            (self.clone / "install.sh").is_file(),
            f"install.sh was not restored:\n{output}",
        )
        self.assertEqual(
            (self.home / ".kiss" / "install-ran").read_text(),
            "ran\n",
            f"the restored install.sh never ran:\n{output}",
        )

    def test_deleted_install_sh_is_restored_from_head(self) -> None:
        """A locally deleted install.sh is restored and executed.

        ``git pull --ff-only`` succeeds (already at tip) without
        recreating the deleted tracked file; before the fix the
        handover then failed on ``./install.sh: No such file``.
        """
        (self.clone / "install.sh").unlink()
        self._run_bootstrap_and_assert_healed()

    def test_directory_named_install_sh_is_replaced(self) -> None:
        """A stray directory named install.sh is replaced by the script."""
        (self.clone / "install.sh").unlink()
        (self.clone / "install.sh").mkdir()
        (self.clone / "install.sh" / "junk.txt").write_text("junk\n")
        self._run_bootstrap_and_assert_healed()


if __name__ == "__main__":
    unittest.main()
