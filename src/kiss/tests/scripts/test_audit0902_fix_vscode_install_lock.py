# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the cross-process update lock in ``scripts/install.sh``.

Review 2026-09-02 (review-vscode.md #1/#2): the VS Code extension's
``runUpdate`` single-flight guard is per sidebar instance, so two VS Code
windows (or a window and the daemon's update endpoint) could run two
installers against the same ``~/.kiss/kiss_ai`` tree, racing each other's
``git reset`` / ``uv sync`` / daemon restart.  The root fix is a lock held
by the installer process itself for its whole lifetime.

Round 2 (review2-vscode.md #1/#2): the first version of that lock was a
``mkdir`` directory with a pid file and stale-lock breaking, and it lost
mutual exclusion two ways: (1) a contender that judged the directory stale
``rm -rf``\\ ed whatever occupied the pathname by then -- eight contenders
plus one pid-less leftover admitted two installers; (2) the lock lived under
``$KISS_HOME`` while the tree it protects lives under ``$HOME``, so a
``KISS_HOME`` override and a default run took different locks on the same
checkout.  The lock is now a kernel advisory lock (``flock(2)``, acquired
through perl, which macOS and Linux both ship; ``flock(1)`` is macOS-less)
on ``$HOME/.kiss/.update.lock``, held by bash's fd 9 for the process's
whole lifetime and released by the kernel when it dies -- there is no
stale state to break.  A second installer prints ``another KISS update is
already running (pid N); exiting.`` and exits 1.

Every test runs the REAL ``scripts/install.sh`` (bash) against a throwaway
``$HOME`` whose ``~/.kiss/kiss_ai`` is a clone of a local bare repository, so
its ``git pull --ff-only`` works offline.  The clone's ``./install.sh`` -- the
program the bootstrap hands over to, and which the lock must outlive -- is a
stub that records that it started and then blocks until the test creates a
release file, which is how the first run is held deterministically while
contenders are launched.  Simultaneous contenders are released together
through a FIFO gate, and "is the lock held?" is probed with a real
``flock(LOCK_EX | LOCK_NB)`` attempt from this process.
"""

import fcntl
import os
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "install.sh"

STUB_INSTALL_SH = """#!/bin/bash
# Stand-in for ~/.kiss/kiss_ai/install.sh: report, then wait to be released.
echo "stub install started pid=$$"
echo "$$" >> "$KISS_TEST_MARK_DIR/started"
if [ -n "${KISS_TEST_SPAWN_DAEMON:-}" ]; then
    # A long-lived process the installer leaves behind (the restarted
    # kiss-web daemon, VS Code): it must not inherit the lock.
    sleep 60 >/dev/null 2>&1 &
    echo "$!" >> "$KISS_TEST_MARK_DIR/daemon"
fi
while [ ! -e "$KISS_TEST_RELEASE" ]; do sleep 0.05; done
echo "done" >> "$KISS_TEST_MARK_DIR/done"
exit "${KISS_TEST_STUB_EXIT:-0}"
"""

REFUSED = "another KISS update is already running (pid "


def _git(*args: str, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
    )


def _lock_is_free(lock_file: Path) -> bool:
    """Probe the installer lock with a real non-blocking flock(2) attempt."""
    fd = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    finally:
        os.close(fd)
    return True


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class InstallLockTest(unittest.TestCase):
    """The bootstrap takes an exclusive, crash-safe, per-$HOME kernel lock."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-install-lock-"))
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.marks = self.tmp / "marks"
        self.marks.mkdir()
        self.release = self.tmp / "release"
        self.env = {
            **{k: v for k, v in os.environ.items() if k != "KISS_HOME"},
            "HOME": str(self.home),
            "KISS_TEST_MARK_DIR": str(self.marks),
            "KISS_TEST_RELEASE": str(self.release),
            "GIT_CONFIG_NOSYSTEM": "1",
        }
        # Local origin + clone at ~/.kiss/kiss_ai, so `git pull --ff-only`
        # in scripts/install.sh needs no network.
        origin = self.tmp / "origin.git"
        subprocess.run(
            ["git", "init", "--bare", "-q", "-b", "main", str(origin)],
            check=True, capture_output=True, env=self.env,
        )
        seed = self.tmp / "seed"
        seed.mkdir()
        _git("init", "-q", "-b", "main", cwd=seed, env=self.env)
        (seed / "install.sh").write_text(STUB_INSTALL_SH)
        (seed / "install.sh").chmod(0o755)
        _git("add", "install.sh", cwd=seed, env=self.env)
        _git("commit", "-q", "-m", "seed", cwd=seed, env=self.env)
        _git("push", "-q", str(origin), "main", cwd=seed, env=self.env)
        kiss_dir = self.home / ".kiss"
        kiss_dir.mkdir()
        subprocess.run(
            ["git", "clone", "-q", str(origin), str(kiss_dir / "kiss_ai")],
            check=True, capture_output=True, env=self.env,
        )
        self.lock_file = kiss_dir / ".update.lock"
        self._procs: list[subprocess.Popen[str]] = []

    def tearDown(self) -> None:
        self.release.write_text("")
        for proc in self._procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        daemon = self.marks / "daemon"
        if daemon.exists():
            for pid in daemon.read_text().split():
                if _alive(int(pid)):
                    os.kill(int(pid), signal.SIGKILL)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _start(
        self, extra_env: dict[str, str] | None = None, gate: Path | None = None,
    ) -> subprocess.Popen[str]:
        # With a gate, the contender reports itself ready and then blocks in
        # open(2) on the FIFO until the test opens it, which releases every
        # waiting reader at once (a reader arriving later passes straight
        # through: the test keeps the FIFO open).
        argv = ["bash", str(SCRIPT)]
        if gate is not None:
            argv = [
                "bash", "-c",
                f'echo $$ >> "{self.marks}/ready"; exec 8<"{gate}"; exec 8<&-; '
                f'exec bash "{SCRIPT}"',
            ]
        proc = subprocess.Popen(
            argv,
            cwd=self.tmp,
            env={**self.env, **(extra_env or {})},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._procs.append(proc)
        return proc

    def _run(self, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT)], cwd=self.tmp, env={**self.env, **(extra_env or {})},
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60,
        )

    def _wait_for(self, pred, what: str, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        while not pred():
            self.assertLess(time.monotonic(), deadline, f"timed out waiting for {what}")
            time.sleep(0.02)

    def _count(self, mark: str) -> int:
        path = self.marks / mark
        return len(path.read_text().split()) if path.exists() else 0

    def _lock_pid(self) -> int:
        return int(self.lock_file.read_text().strip())

    def test_second_concurrent_run_is_refused_and_lock_is_released(self) -> None:
        first = self._start()
        self._wait_for(lambda: self._count("started") == 1, "first installer")
        self.assertFalse(_lock_is_free(self.lock_file), "lock not held while installing")
        self.assertEqual(self._lock_pid(), first.pid, "the lock records the installer's own pid")

        second = self._run()
        self.assertEqual(second.returncode, 1, second.stdout + second.stderr)
        self.assertIn(f"{REFUSED}{first.pid}); exiting.", second.stdout + second.stderr)
        self.assertEqual(self._count("started"), 1, "second run reached install.sh")
        self.assertFalse(_lock_is_free(self.lock_file), "the loser released the winner's lock")
        self.assertEqual(self._lock_pid(), first.pid)

        self.release.write_text("")
        out, _ = first.communicate(timeout=60)
        self.assertEqual(first.returncode, 0, out)
        self.assertTrue((self.marks / "done").exists(), out)
        self.assertTrue(_lock_is_free(self.lock_file), "lock survived a normal exit")

        # With the lock free, the next run proceeds.
        third = self._run()
        self.assertEqual(third.returncode, 0, third.stdout + third.stderr)
        self.assertEqual(self._count("started"), 2)
        self.assertTrue(_lock_is_free(self.lock_file))

    def test_eight_simultaneous_contenders_admit_exactly_one_installer(self) -> None:
        # Legacy leftover of the mkdir protocol (a crash between mkdir and
        # writing the pid): it must be irrelevant, never "recovered".
        legacy = self.home / ".kiss" / ".update.lock.d"
        for trial in range(5):
            legacy.mkdir(exist_ok=True)
            gate = self.tmp / f"gate{trial}"
            os.mkfifo(gate)
            contenders = [self._start(gate=gate) for _ in range(8)]
            self._wait_for(
                lambda: self._count("ready") == 8 * (trial + 1),
                f"trial {trial}: eight contenders at the gate",
            )
            # Opening the FIFO (O_RDWR never blocks) releases every reader.
            gate_fd = os.open(gate, os.O_RDWR)
            self._wait_for(
                lambda: self._count("started") >= trial + 1,
                f"trial {trial}: the winner",
            )
            self._wait_for(
                lambda: sum(p.poll() is not None for p in contenders) == 7,
                f"trial {trial}: seven losers",
            )
            self.assertEqual(
                self._count("started"), trial + 1,
                f"trial {trial}: more than one installer entered",
            )
            winners = [p for p in contenders if p.poll() is None]
            self.assertEqual(len(winners), 1)
            winner = winners[0]
            self.assertEqual(self._lock_pid(), winner.pid)
            self.release.write_text("")
            out, _ = winner.communicate(timeout=60)
            self.assertEqual(winner.returncode, 0, out)
            for loser in contenders:
                if loser is winner:
                    continue
                lost, _ = loser.communicate(timeout=60)
                self.assertEqual(loser.returncode, 1, lost)
                self.assertIn(f"{REFUSED}{winner.pid}); exiting.", lost)
            self.assertEqual(self._count("started"), trial + 1)
            self.assertTrue(_lock_is_free(self.lock_file))
            os.close(gate_fd)
            self.release.unlink()
            self._procs.clear()

    def test_same_home_different_kiss_home_contenders_exclude_each_other(self) -> None:
        # The protected resource (~/.kiss/kiss_ai) follows $HOME, so the
        # lock must too -- a KISS_HOME override must not pick another lock.
        kiss_home = self.tmp / "custom-kiss-home"
        first = self._start({"KISS_HOME": str(kiss_home)})
        self._wait_for(lambda: self._count("started") == 1, "first installer")
        self.assertFalse(_lock_is_free(self.lock_file), "lock must live under $HOME/.kiss")
        self.assertFalse((kiss_home / ".update.lock").exists(), "lock followed KISS_HOME")
        second = self._run()
        self.assertEqual(second.returncode, 1, second.stdout + second.stderr)
        self.assertIn(f"{REFUSED}{first.pid}); exiting.", second.stdout + second.stderr)
        self.assertEqual(self._count("started"), 1, "both contenders entered the same checkout")
        self.release.write_text("")
        first.communicate(timeout=60)
        self.assertEqual(first.returncode, 0)
        self.assertTrue(_lock_is_free(self.lock_file))

    def test_sigterm_releases_the_lock_once_the_installer_chain_exits(self) -> None:
        first = self._start()
        self._wait_for(lambda: self._count("started") == 1, "first installer")
        first.send_signal(signal.SIGTERM)
        # bash runs the trap once its foreground child (the stub) exits, so
        # the lock outlives a stray TERM for as long as the install runs.
        refused = self._run()
        self.assertEqual(refused.returncode, 1, refused.stdout + refused.stderr)
        self.assertIn(REFUSED, refused.stdout + refused.stderr)
        self.assertFalse(_lock_is_free(self.lock_file))
        self.release.write_text("")
        out, _ = first.communicate(timeout=60)
        self.assertEqual(first.returncode, 143, out)
        self.assertTrue(_lock_is_free(self.lock_file), "lock survived SIGTERM")

    def test_failure_under_set_e_releases_the_lock(self) -> None:
        self.release.write_text("")
        result = self._run({"KISS_TEST_STUB_EXIT": "7"})
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertEqual(self._count("started"), 1)
        self.assertTrue(_lock_is_free(self.lock_file), "lock survived a failure exit")

    def test_nested_run_under_a_held_lock_neither_blocks_nor_releases(self) -> None:
        # KISS_UPDATE_LOCK_HELD is exported to the installer the bootstrap
        # hands over to; a nested bootstrap must not deadlock on its own
        # ancestor's lock, nor release it on exit.  This process plays the
        # ancestor: it holds the real lock for the whole test.
        self.lock_file.parent.mkdir(exist_ok=True)
        holder = os.open(self.lock_file, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.write(holder, f"{os.getpid()}\n".encode())
            self.release.write_text("")
            result = self._run({"KISS_UPDATE_LOCK_HELD": "1"})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(self._count("started"), 1)
            self.assertFalse(
                _lock_is_free(self.lock_file), "nested run released its ancestor's lock",
            )
            self.assertEqual(self._lock_pid(), os.getpid())
            # Without the marker the same held lock refuses the run.
            refused = self._run()
            self.assertEqual(refused.returncode, 1)
            self.assertIn(f"{REFUSED}{os.getpid()}); exiting.", refused.stdout + refused.stderr)
            self.assertEqual(self._count("started"), 1)
        finally:
            os.close(holder)
        self.assertTrue(_lock_is_free(self.lock_file))

    def test_launched_background_child_does_not_keep_the_lock(self) -> None:
        # The installer leaves long-lived processes behind (the restarted
        # kiss-web daemon, VS Code).  They must not inherit fd 9, or the
        # lock would stay held until they exit.
        self.release.write_text("")
        result = self._run({"KISS_TEST_SPAWN_DAEMON": "1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        daemon_pid = int((self.marks / "daemon").read_text().split()[0])
        self.assertTrue(_alive(daemon_pid), "the background child should still be running")
        self.assertTrue(
            _lock_is_free(self.lock_file),
            "a still-running background child kept the update lock",
        )
        next_run = self._run()
        self.assertEqual(next_run.returncode, 0, next_run.stdout + next_run.stderr)
        self.assertEqual(self._count("started"), 2)


if __name__ == "__main__":
    unittest.main()
