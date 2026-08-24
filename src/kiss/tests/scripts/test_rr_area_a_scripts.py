# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the area-A fixes in kiss.scripts.

Covers:

* A-RC3 — ``remote_config.py`` now stages through ``tempfile.mkstemp``
  (a unique temp file per writer) and serializes its load-modify-replace
  under the shared ``.config.lock`` flock.  Concurrent deploys against
  one ``config.json`` must never publish corrupt JSON, lose unrelated
  settings, or leave staging files behind.
* A-R7 — ``check.py``'s repeated project-root computation collapsed into
  one ``PROJECT_ROOT`` constant; the functions that used the local
  copies must still act relative to the real repository root.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import mkdtemp

from kiss.scripts.check import PROJECT_ROOT, _should_skip_path

_REMOTE_CONFIG = (
    Path(__file__).resolve().parents[4] / "src" / "kiss" / "scripts" / "remote_config.py"
)


class RemoteConfigConcurrentWritersTest(unittest.TestCase):
    """A-RC3: concurrent deploys must serialize, not corrupt each other.

    Before the fix, every run staged its JSON in the same fixed
    ``config.json.sorcar-new`` file: writer B truncated the temp file A
    was still writing, and A then published B's half-written JSON.  The
    read-modify-replace was also lock-free, so one writer's settings
    could vanish under another.  This test runs the real script in
    several simultaneous processes and requires a valid, complete
    configuration file afterwards, every round.
    """

    def setUp(self) -> None:
        self.tmp = Path(mkdtemp())
        self.config = self.tmp / "config.json"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_concurrent_runs_never_corrupt_the_config(self) -> None:
        """Valid JSON, surviving settings, one winning password, no litter."""
        passwords = [f"pw-{i:02d}-abcdef" for i in range(6)]
        for round_no in range(3):
            self.config.write_text(
                json.dumps({"last_model": "claude-opus-5", "round": round_no})
            )
            procs = [
                subprocess.Popen(
                    [
                        sys.executable,
                        str(_REMOTE_CONFIG),
                        str(self.config),
                        "/home/ubuntu/kiss",
                        password,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                for password in passwords
            ]
            for proc in procs:
                _, stderr = proc.communicate(timeout=120)
                self.assertEqual(proc.returncode, 0, stderr.decode())

            # The file parses -- no writer published a half-written temp.
            saved = json.loads(self.config.read_text())
            # Settings no deploy touches survive every interleaving.
            self.assertEqual(saved["last_model"], "claude-opus-5")
            self.assertEqual(saved["round"], round_no)
            self.assertEqual(saved["work_dir"], "/home/ubuntu/kiss")
            # Exactly one writer's password won, and it is one of ours.
            self.assertIn(saved["remote_password"], passwords)
            # No staging file survived any of the six runs.
            self.assertEqual(list(self.tmp.glob("config.json.sorcar-new*")), [])
            # And nothing was "kept as unreadable": no writer ever saw
            # a half-written file through the lock.
            self.assertEqual(list(self.tmp.glob("config.json.unreadable-*")), [])

    def test_single_run_still_behaves(self) -> None:
        """The locking rewrite changes concurrency, not the contract."""
        self.config.write_text(json.dumps({"last_model": "claude-opus-5"}))

        done = subprocess.run(
            [sys.executable, str(_REMOTE_CONFIG), str(self.config),
             "/home/ubuntu/kiss", "chosen-pass"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("SORCAR_PASSWORD=chosen-pass", done.stdout)
        saved = json.loads(self.config.read_text())
        self.assertEqual(saved["last_model"], "claude-opus-5")
        self.assertEqual(saved["remote_password"], "chosen-pass")
        self.assertEqual(self.config.stat().st_mode & 0o077, 0)


class RemoteConfigSymlinkLockTest(unittest.TestCase):
    """The deploy's lock lives beside the link, where save_config locks.

    The supported dotfiles layout is ``~/.kiss/config.json ->
    ~/dotfiles/config.json``.  The web app's ``save_config`` always
    flocks ``<config-dir>/.config.lock`` next to the path it was told
    about (``~/.kiss/.config.lock``); before the fix the script resolved
    the link *first* and locked ``~/dotfiles/.config.lock`` instead, so
    the two writers held different locks and did not exclude each other
    at all.  The lock path must be derived from the path as named, while
    the JSON is still read from and written through the link's target.
    """

    def setUp(self) -> None:
        self.tmp = Path(mkdtemp())
        self.kiss_dir = self.tmp / "kiss"
        self.dotfiles = self.tmp / "dotfiles"
        self.kiss_dir.mkdir()
        self.dotfiles.mkdir()
        self.target = self.dotfiles / "config.json"
        self.target.write_text(json.dumps({"last_model": "claude-opus-5"}))
        self.link = self.kiss_dir / "config.json"
        self.link.symlink_to(self.target)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_deploy_blocks_on_the_link_side_lock(self) -> None:
        """Holding ``<link-dir>/.config.lock`` stalls the whole deploy.

        This is the very flock ``save_config`` takes, so blocking on it
        is what mutual exclusion with the web app *means*.  On the
        unfixed code the script locked the dotfiles side and sailed
        straight past the held lock.
        """
        import fcntl
        import time

        before = self.target.read_text()
        with open(self.kiss_dir / ".config.lock", "w", encoding="utf-8") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            proc = subprocess.Popen(
                [sys.executable, str(_REMOTE_CONFIG), str(self.link),
                 "/home/ubuntu/kiss", "locked-out-pass"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                time.sleep(1.5)
                self.assertIsNone(
                    proc.poll(),
                    "the deploy finished while the web app's lock was held: "
                    "it must have taken a different lock",
                )
                # Blocked before reading: the target is untouched.
                self.assertEqual(self.target.read_text(), before)
            finally:
                fcntl.flock(held, fcntl.LOCK_UN)
                stdout, stderr = proc.communicate(timeout=120)

        self.assertEqual(proc.returncode, 0, stderr.decode())
        self.assertIn("SORCAR_PASSWORD=locked-out-pass", stdout.decode())
        saved = json.loads(self.target.read_text())
        self.assertEqual(saved["last_model"], "claude-opus-5")
        self.assertEqual(saved["remote_password"], "locked-out-pass")
        # The link survived and still leads to the maintained file.
        self.assertTrue(self.link.is_symlink())
        self.assertEqual(json.loads(self.link.read_text()), saved)

    def test_concurrent_deploys_through_the_link_serialize(self) -> None:
        """Simultaneous deploys against the link never corrupt the target."""
        passwords = [f"ln-{i:02d}-abcdef" for i in range(6)]
        procs = [
            subprocess.Popen(
                [sys.executable, str(_REMOTE_CONFIG), str(self.link),
                 "/home/ubuntu/kiss", password],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for password in passwords
        ]
        for proc in procs:
            _, stderr = proc.communicate(timeout=120)
            self.assertEqual(proc.returncode, 0, stderr.decode())

        saved = json.loads(self.target.read_text())
        self.assertEqual(saved["last_model"], "claude-opus-5")
        self.assertEqual(saved["work_dir"], "/home/ubuntu/kiss")
        self.assertIn(saved["remote_password"], passwords)
        self.assertTrue(self.link.is_symlink())
        # The shared lock was created beside the link -- the file the
        # web app locks -- and never beside the target.
        self.assertTrue((self.kiss_dir / ".config.lock").exists())
        self.assertFalse((self.dotfiles / ".config.lock").exists())
        # No staging litter and no "unreadable" backups on either side.
        for directory in (self.kiss_dir, self.dotfiles):
            self.assertEqual(list(directory.glob("config.json.sorcar-new*")), [])
            self.assertEqual(list(directory.glob("config.json.unreadable-*")), [])


class CheckProjectRootTest(unittest.TestCase):
    """A-R7: the shared constant behaves exactly as the four locals did."""

    def test_project_root_is_the_repository_root(self) -> None:
        """The constant points at the checkout the script cleans and checks."""
        self.assertTrue((PROJECT_ROOT / "pyproject.toml").is_file())
        self.assertTrue((PROJECT_ROOT / "src" / "kiss").is_dir())

    def test_should_skip_path_skips_git_internals(self) -> None:
        """Paths inside .git are always skipped."""
        self.assertTrue(_should_skip_path(PROJECT_ROOT / ".git" / "config"))

    def test_should_skip_path_keeps_tracked_files(self) -> None:
        """A tracked source file is not skipped."""
        self.assertFalse(
            _should_skip_path(PROJECT_ROOT / "src" / "kiss" / "scripts" / "check.py")
        )

    def test_should_skip_path_skips_ignored_files(self) -> None:
        """A path .gitignore covers is skipped (cwd-independent via PROJECT_ROOT)."""
        # __pycache__ directories are ignored in this repository.
        ignored = PROJECT_ROOT / "src" / "kiss" / "__pycache__" / "x.pyc"
        self.assertTrue(_should_skip_path(ignored))


if __name__ == "__main__":
    unittest.main()
