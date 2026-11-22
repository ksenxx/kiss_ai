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
  settings, or leave staging files behind.  Through a link it holds the
  lock beside the link and the one beside the file the link names.
* A-R7 — ``check.py``'s repeated project-root computation collapsed into
  one ``PROJECT_ROOT`` constant; the functions that used the local
  copies must still act relative to the real repository root.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from tempfile import mkdtemp

import pytest

from kiss.core.file_lock import exclusive_file_lock, lock_exclusive, unlock
from kiss.scripts import remote_config
from kiss.scripts.check import PROJECT_ROOT
from kiss.tests.conftest import IS_WINDOWS, posix_only

_REMOTE_CONFIG = (
    Path(__file__).resolve().parents[4] / "src" / "kiss" / "scripts" / "remote_config.py"
)


def _configure_into(outcome: list[str | OSError], path: Path) -> None:
    """Deploy to *path* and record the password in force, or the error (a thread's target)."""
    try:
        outcome.append(remote_config.configure(str(path), "/home/ubuntu/kiss", "arrived"))
    except OSError as error:
        outcome.append(error)


def _locked_elsewhere(lock_path: Path) -> bool:
    """Whether another open file holds the lock at *lock_path* right now."""
    with exclusive_file_lock(lock_path, blocking=False) as mine:
        return not mine


def _relink(link: Path, target: Path) -> None:
    """Point the symbolic link *link* at *target*, in one step."""
    staged = link.with_name("relinked")
    staged.symlink_to(target)
    os.replace(staged, link)


# Runs the script (argv[1]) on the arguments from argv[8] on and stops it once, where the
# function named argv[2] returns for the argv[5]-th time with its local argv[3] equal to
# argv[4]: it creates the file argv[6] there and goes on when the file argv[7] exists.  A
# gate, nothing else: every call, path and lock is the script's own.
_PAUSED_DEPLOY = """\
import os, runpy, sys, time
script, function, local, value, nth, ready, go = sys.argv[1:8]
returns = 0
def pause(frame, event, arg):
    global returns
    here = frame.f_code.co_name == function and frame.f_locals.get(local) == value
    if event == "return" and here:
        returns += 1
        if returns == int(nth):
            open(ready, "w").close()
            deadline = time.monotonic() + 120
            while not os.path.exists(go) and time.monotonic() < deadline:
                time.sleep(0.01)
    return pause
sys.settrace(pause)
sys.argv[1:] = sys.argv[8:]
runpy.run_path(script, run_name="__main__")
"""


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
            winners = [p for p in passwords if p == saved["remote_password"]]
            self.assertEqual(len(winners), 1, winners)
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
        if not IS_WINDOWS:  # Windows chmod carries no group/other bits
            self.assertEqual(self.config.stat().st_mode & 0o077, 0)

    @pytest.mark.slow
    def test_a_lock_that_stays_held_fails_the_deploy_instead_of_hanging_it(self) -> None:
        """A wedged holder of ``.config.lock`` costs the script its 30 s bound, not forever.

        It exits 1 naming the lock and leaves the configuration as it was.
        The Windows half (an ``msvcrt`` error other than "held by another
        process" is raised, not retried for good) cannot run here: reaching
        ``msvcrt`` on POSIX takes a stand-in module, which is a test double.
        """
        before = json.dumps({"last_model": "claude-opus-5"})
        self.config.write_text(before)
        with open(self.tmp / ".config.lock", "w", encoding="utf-8") as held:
            lock_exclusive(held)
            done = subprocess.run(
                [sys.executable, str(_REMOTE_CONFIG), str(self.config), "/home/ubuntu/kiss"],
                capture_output=True, text=True, timeout=120, check=False,
            )
        self.assertEqual((done.returncode, done.stdout), (1, ""), done.stderr)
        self.assertIn(".config.lock still locked by another process", done.stderr)
        self.assertEqual(self.config.read_text(), before)
        self.assertEqual(
            sorted(p.name for p in self.tmp.iterdir()), [".config.lock", "config.json"]
        )


class RemoteConfigSymlinkLockTest(unittest.TestCase):
    """The deploy's locks live beside the link and beside its file, where save_config locks.

    The supported dotfiles layout is ``~/.kiss/config.json ->
    ~/dotfiles/config.json``.  The web app's ``save_config`` always
    flocks ``<config-dir>/.config.lock`` next to the path it was told
    about (``~/.kiss/.config.lock``); before the fix the script resolved
    the link *first* and locked ``~/dotfiles/.config.lock`` instead, so
    the two writers held different locks and did not exclude each other
    at all.  The lock path must be derived from the path as named, while
    the JSON is still read from and written through the link's target.
    A writer that reaches that file by another link holds the lock
    beside the file, as ``save_config`` does next to its own: the deploy
    holds that one as well.
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

    def _stored(self) -> dict[str, str]:
        """What the file the link names holds: nothing while it is not there."""
        stored: dict[str, str] = json.loads(self.target.read_text()) if self.target.exists() else {}
        return stored

    def _assert_deploy_waits_for_the_lock_in(self, directory: Path) -> None:
        """Hold ``.config.lock`` in *directory*: the deploy through the link writes after it."""
        before = self._stored()
        with open(directory / ".config.lock", "w", encoding="utf-8") as held:
            lock_exclusive(held)
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
                self.assertEqual(self._stored(), before)
            finally:
                unlock(held)
                stdout, stderr = proc.communicate(timeout=120)

        self.assertEqual(proc.returncode, 0, stderr.decode())
        self.assertIn("SORCAR_PASSWORD=locked-out-pass", stdout.decode())
        saved = self._stored()
        deployed = {"remote_password": "locked-out-pass", "work_dir": "/home/ubuntu/kiss"}
        self.assertEqual(saved, {**before, **deployed})
        # The link survived and still leads to the maintained file.
        self.assertTrue(self.link.is_symlink())
        self.assertEqual(json.loads(self.link.read_text()), saved)

    def test_deploy_blocks_on_the_link_side_lock(self) -> None:
        """Holding ``<link-dir>/.config.lock`` stalls the whole deploy.

        This is the very flock ``save_config`` takes, so blocking on it
        is what mutual exclusion with the web app *means*.  On the
        unfixed code the script locked the dotfiles side and sailed
        straight past the held lock.
        """
        self._assert_deploy_waits_for_the_lock_in(self.kiss_dir)

    def test_deploy_blocks_on_the_lock_beside_the_linked_file(self) -> None:
        """Holding ``.config.lock`` beside the file the link names stalls the deploy too.

        It is the flock of a writer that reaches the file by another
        path: a ``save_config`` whose own ``config.json`` is a second link
        to it, or is the file itself.  The deploy took the link-side lock
        alone and replaced the file from a content that writer was changing.
        """
        self._assert_deploy_waits_for_the_lock_in(self.dotfiles)

    def test_deploy_blocks_on_the_lock_beside_a_linked_file_not_there_yet(self) -> None:
        """The deploy creates the file a link names, so it holds that file's lock as well.

        The web app whose own ``config.json`` that file is saves it for
        the first time under the same lock.
        """
        self.target.unlink()
        self._assert_deploy_waits_for_the_lock_in(self.dotfiles)

    @posix_only("the deploy reaching the lock is read from its emptying the flock-ed file")
    def test_a_missing_directory_of_the_linked_file_is_made_with_the_lock_held(self) -> None:
        """A link into a directory that is not there: the deploy makes nothing while it waits.

        The web app replaces such a link with a regular file, holding the
        lock beside it; a directory the deploy made on its way to that
        lock would be left behind.  With the lock, the deploy makes the
        directory and the file the link names.
        """
        shutil.rmtree(self.dotfiles)
        with open(self.kiss_dir / ".config.lock", "w", encoding="utf-8") as held:
            lock_exclusive(held)
            held.write("held")
            held.flush()
            proc = subprocess.Popen(
                [sys.executable, str(_REMOTE_CONFIG), str(self.link),
                 "/home/ubuntu/kiss", "locked-out-pass"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                # Opening the lock file empties it, and is the last thing the
                # script does before it waits for the lock.
                deadline = time.monotonic() + 60
                while os.path.getsize(held.name) and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(os.path.getsize(held.name), 0)
                self.assertFalse(self.dotfiles.exists())
            finally:
                unlock(held)
                _, stderr = proc.communicate(timeout=120)

        self.assertEqual(proc.returncode, 0, stderr.decode())
        self.assertEqual(
            self._stored(), {"remote_password": "locked-out-pass", "work_dir": "/home/ubuntu/kiss"}
        )
        self.assertTrue(self.link.is_symlink())

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
        # The locks were created beside the link -- the file the web
        # app locks -- and beside the target, for a writer by another link.
        self.assertTrue((self.kiss_dir / ".config.lock").exists())
        self.assertTrue((self.dotfiles / ".config.lock").exists())
        # No staging litter and no "unreadable" backups on either side.
        for directory in (self.kiss_dir, self.dotfiles):
            self.assertEqual(list(directory.glob("config.json.sorcar-new*")), [])
            self.assertEqual(list(directory.glob("config.json.unreadable-*")), [])


@posix_only("creating a symlink needs no privilege")
@pytest.mark.parametrize("moves", [1, 2])
def test_deploy_follows_a_link_pointed_elsewhere_during_its_wait_once(
    moves: int, tmp_path: Path,
) -> None:
    """The file the link names once the locks are held is the one the deploy reads and replaces.

    The lock beside the linked file may come first in the order all
    writers take them in, so the link is followed before the wait.  Here
    it is pointed elsewhere while the deploy waits for the lock beside
    the link, and the lock beside the new file is held: the deploy waits
    for that one too, writes no file until it has it, and keeps what was
    stored under it.  Pointed elsewhere once more during that second
    wait, the link is followed no further: the deploy fails and has
    written no file.
    """
    # The locks are taken in the order of their paths: a-first, kiss, z-second.
    names = ("a-first", "z-second", "third")
    first, second, third = targets = [tmp_path / name / "config.json" for name in names]
    for target in targets:
        target.parent.mkdir()
        target.write_text("{}")
    link = tmp_path / "kiss" / "config.json"
    link.parent.mkdir()
    link.symlink_to(first)
    own, old, new = (path.with_name(".config.lock") for path in (link, first, second))
    outcome: list[str | OSError] = []
    deploy = threading.Thread(target=_configure_into, args=(outcome, link), daemon=True)
    deadline = time.monotonic() + 60
    with exclusive_file_lock(new, timeout=30):
        with exclusive_file_lock(own, timeout=30):
            deploy.start()
            while not _locked_elsewhere(old):  # the deploy has followed the link, and waits
                assert deploy.is_alive() and time.monotonic() < deadline, outcome
                time.sleep(0.01)
            _relink(link, second)
        if moves == 1:
            deploy.join(1.0)  # one that does not wait for the new file's lock is done by now
            assert deploy.is_alive(), outcome
            assert [target.read_text() for target in targets] == ["{}"] * 3
            second.write_text('{"last_model": "kept"}')  # what the holder of that lock stores
        else:
            while _locked_elsewhere(old) or not _locked_elsewhere(own):  # its second wait
                assert deploy.is_alive() and time.monotonic() < deadline, outcome
                time.sleep(0.01)
            _relink(link, third)
    deploy.join(60)
    if moves == 1:
        assert outcome == ["arrived"]
        assert json.loads(second.read_text()) == {
            "last_model": "kept", "remote_password": "arrived", "work_dir": "/home/ubuntu/kiss",
        }
        assert [first.read_text(), third.read_text()] == ["{}", "{}"]
    else:
        assert [type(error) for error in outcome] == [TimeoutError]
        assert "named another file" in str(outcome[0])
        assert [target.read_text() for target in targets] == ["{}"] * 3
    assert link.is_symlink() and link.samefile(targets[moves])


@posix_only("creating a symlink needs no privilege")
@pytest.mark.parametrize("paused", ["read", "made", "checked"])
def test_deploy_holds_the_lock_in_the_directory_it_makes_for_a_linked_file(
    paused: str, tmp_path: Path,
) -> None:
    """A link into a directory that is not there: the deploy makes it and locks in it.

    Nobody holds a lock in a directory before it exists, so the deploy
    took the lock beside the link alone.  Once it had made the directory,
    a web app whose own ``config.json`` the linked file is could save
    there, and the deploy replaced the file from what it had read before.

    ``read``: the deploy is stopped between its read and its replace.  It
    holds the lock in the new directory, a ``save_config`` there waits
    for it, and both writers' settings are stored.  ``made``: it is
    stopped as it has made the directory, and the lock in it is taken
    before the deploy can: the deploy waits for it, in the order all
    writers take the two locks in, and keeps what was stored under it.
    ``checked``: the same, stopped as it has found the link unchanged
    under the lock beside it, and that writer makes the directory: a
    directory somebody else made in time is no lock the deploy holds.
    """
    directory = tmp_path / "dotfiles"
    target, beside = directory / "config.json", directory / ".config.lock"
    link = tmp_path / "kiss" / "config.json"
    link.parent.mkdir()
    link.symlink_to(target)
    ready, go = tmp_path / "ready", tmp_path / "go"
    gate = {
        "read": ("load_config", "path", target, 1),
        "made": ("makedirs", "name", directory, 1),
        "checked": ("_target_and_locks", "path", link, 2),
    }[paused]
    command = [sys.executable, "-c", _PAUSED_DEPLOY, str(_REMOTE_CONFIG), *map(str, gate)]
    command += [str(ready), str(go), str(link), "/home/ubuntu/kiss", "arrived"]
    save = "from kiss.core.user_config import save_config; save_config({'last_model': 'kept'})"
    deploy = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    writer: subprocess.Popen[str] | None = None
    try:
        deadline = time.monotonic() + 60
        while not ready.exists():
            assert deploy.poll() is None and time.monotonic() < deadline, deploy.communicate()
            time.sleep(0.01)
        if paused == "read":
            assert _locked_elsewhere(beside), "the deploy reads and replaces without that lock"
            writer = subprocess.Popen(
                [sys.executable, "-c", save], env={**os.environ, "KISS_HOME": str(directory)},
                stderr=subprocess.PIPE, text=True,
            )
            with pytest.raises(subprocess.TimeoutExpired):
                writer.wait(1.0)  # one that does not wait for the deploy is done by now
            go.touch()
            assert writer.wait(60) == 0, writer.communicate()
        else:
            directory.mkdir(exist_ok=True)
            with exclusive_file_lock(beside, timeout=30):
                go.touch()
                with pytest.raises(subprocess.TimeoutExpired):
                    deploy.wait(1.0)  # one that does not wait for this lock is done by now
                assert not target.exists()
                target.write_text('{"last_model": "kept"}')  # what the holder of that lock stores
        stdout, stderr = deploy.communicate(timeout=120)
    finally:
        go.touch()
        for process in (deploy, writer):
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate()
    assert (deploy.returncode, stdout) == (0, "SORCAR_PASSWORD=arrived\n"), stderr
    assert json.loads(target.read_text()) == {
        "last_model": "kept", "remote_password": "arrived", "work_dir": "/home/ubuntu/kiss",
    }
    assert link.is_symlink() and link.samefile(target)


class CheckProjectRootTest(unittest.TestCase):
    """A-R7: the shared constant behaves exactly as the four locals did."""

    def test_project_root_is_the_repository_root(self) -> None:
        """The constant points at the checkout the script cleans and checks."""
        self.assertTrue((PROJECT_ROOT / "pyproject.toml").is_file())
        self.assertTrue((PROJECT_ROOT / "src" / "kiss").is_dir())


if __name__ == "__main__":
    unittest.main()
