# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the cross-process update lock in the ROOT ``install.sh``.

Audit 2026-09-02 (fix-crossboundary-vscode.md #1): ``scripts/install.sh``
(the curl bootstrap) holds the update lock for its lifetime and exports
``KISS_UPDATE_LOCK_HELD=1`` before handing over to the root ``install.sh``.
But two callers run the ROOT script directly, outside any lock: the
kiss-web daemon's ``runUpdate`` handler (``Popen(["bash", install.sh,
"--non-interactive"])``) and the VS Code extension's legacy fallback.  So
the root script takes the very same lock itself, skipping it when
``KISS_UPDATE_LOCK_HELD=1`` says an ancestor already holds it.

Round 2 (review2-vscode.md #1/#2): the first lock was a ``mkdir`` directory
with a pid file and stale-lock breaking, which admitted two installers when
eight contenders met a pid-less leftover (the "stale" recovery ``rm -rf``\\ ed
a live replacement lock), and which keyed on ``$KISS_HOME`` although the
checkout and the global extension install follow ``$HOME``.  It is now a
kernel advisory lock (``flock(2)`` through perl, present on macOS and
Linux) on ``$HOME/.kiss/.update.lock`` held by bash's fd 9 for the whole
process lifetime; the kernel releases it on any exit, so there is no stale
state.  Because the root script re-execs itself into a new session through
perl (the ``kiss-new-session-reexec`` block), the lock is taken by the
re-exec'd child that does the work, and the pid recorded is that child's.
Every long-lived process the script launches (VS Code, whatever a build
step leaves behind) gets fd 9 closed so it cannot keep the lock alive.

Every test runs the REAL root ``install.sh --non-interactive`` -- exactly
the daemon's invocation -- against a throwaway ``$HOME``, with a whitelist
``PATH`` that contains only bash, perl, the coreutils the script needs and
these stubs:

* ``git``: ``git --version`` (the first external command the script's body
  runs, before ``update_repo``) records that it started and blocks until the
  test creates a release file; that is how the first run is held while
  contenders are launched.  Every other git invocation fails, and
  ``KISS_SKIP_UPDATE=1`` additionally keeps ``update_repo`` from touching
  the checkout.
* ``curl``: blocks on a second release file, then fails.  With no ``node``
  on the PATH the script reaches ``install_node``, whose download fails, and
  exits 1 at "[2/5] Checking Node.js" -- before any step that could reach
  the network, a package manager or the extension build.

The launch-line test instead runs the whole script to completion against a
throwaway ``PROJECT_DIR`` with stub ``node``/``npm``/``npx``/``code``
binaries (``uname`` reports Linux so no macOS ``open -a`` path is taken):
``npm ci``, ``code --install-extension`` and the final VS Code launch each
leave a background ``sleep`` behind, standing in for the processes a real
install leaves running.

So no test can install anything, modify this repository or touch the real
``~/.kiss``.
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
SCRIPT = REPO_ROOT / "install.sh"

# Everything the root install.sh (and the stubs below) executes from PATH
# before it reaches the Node.js stage.  bash and perl are needed by the
# new-session re-exec (``/usr/bin/env perl`` and ``exec {"bash"}``).
WHITELISTED_TOOLS = (
    "bash",
    "perl",
    "uname",
    "dirname",
    "grep",
    "sed",
    "head",
    "mkdir",
    "date",
    "tee",
    "cat",
    "rm",
    "sleep",
    "tr",
    "tar",
)

STUB_GIT = """#!/bin/bash
# Stand-in for git: the first ``git --version`` is the hold point.
if [ "$1" = "--version" ]; then
    echo "$$" >> "$KISS_TEST_MARK_DIR/started"
    while [ ! -e "$KISS_TEST_RELEASE" ]; do sleep 0.05; done
    echo "git version 2.50.0"
    exit 0
fi
exit 1
"""

STUB_CURL = """#!/bin/bash
# Stand-in for curl (reached by install_node): hold, then fail offline.
echo "$$" >> "$KISS_TEST_MARK_DIR/curl-started"
while [ ! -e "$KISS_TEST_RELEASE2" ]; do sleep 0.05; done
exit 22
"""

# On Darwin the real uname sends install.sh through ensure_xcode_clt and
# ensure_homebrew before the git stage these tests hold at.  Both stubs keep
# that path hermetic: xcode-select reports a fake developer dir (created in
# setUp with a usr/bin/git inside, which the CLT check requires), and a brew
# on PATH short-circuits both the top-level `brew shellenv` eval (which would
# prepend real host directories to the stub PATH) and ensure_homebrew's
# curl-driven bootstrap (which would corrupt the curl-started mark).
STUB_XCODE_SELECT = """#!/bin/bash
[ "$1" = "-p" ] && { echo "$KISS_TEST_DEVDIR"; exit 0; }
exit 1
"""

STUB_BREW = """#!/bin/bash
echo "Homebrew 4.0.0 (test stub)"
exit 0
"""

# Full-run stubs.  Each long-lived leftover records its pid in the
# ``daemon`` mark so the test can prove it is alive and clean it up.
STUB_NODE = """#!/bin/bash
[ "$1" = "--version" ] && echo "v22.16.0"
exit 0
"""

STUB_NPM = """#!/bin/bash
case "$1" in
    --version) echo "10.9.2" ;;
    ci)
        # A build step leaving a daemon behind.
        sleep 60 >/dev/null 2>&1 &
        echo "$!" >> "$KISS_TEST_MARK_DIR/daemon"
        ;;
    run) [ "$2" = "package" ] && : > "$PWD/kiss-sorcar.vsix" ;;
esac
exit 0
"""

STUB_CODE = """#!/bin/bash
case "$1" in
    --version) echo "1.100.0" ;;
    --install-extension)
        sleep 60 >/dev/null 2>&1 &
        echo "$!" >> "$KISS_TEST_MARK_DIR/daemon"
        ;;
    *)
        # ``code <workspace>``: the editor itself, launched via nohup.
        echo "$$" >> "$KISS_TEST_MARK_DIR/daemon"
        exec sleep 60
        ;;
esac
exit 0
"""

STUB_UNAME = """#!/bin/bash
case "$1" in
    -m) echo "x86_64" ;;
    *) echo "Linux" ;;
esac
"""

REFUSED = "another KISS update is already running (pid "
NODE_EXIT = "Node.js, npm, and npx are required to build the extension."


def _parent_pid(pid: int) -> int:
    out = subprocess.run(
        ["ps", "-o", "ppid=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    return int(out.stdout.strip() or "-1")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


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


class RootInstallLockTest(unittest.TestCase):
    """The root installer takes the same exclusive, crash-safe lock as the bootstrap."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-root-install-lock-"))
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.marks = self.tmp / "marks"
        self.marks.mkdir()
        self.release = self.tmp / "release"
        self.release2 = self.tmp / "release2"
        tools = self.tmp / "tools"
        tools.mkdir()
        for name in WHITELISTED_TOOLS:
            real = shutil.which(name)
            self.assertIsNotNone(real, f"{name} is required by this test")
            assert real is not None
            os.symlink(real, tools / name)
        self.stubs = self.tmp / "stubs"
        self.stubs.mkdir()
        for name, body in (
            ("git", STUB_GIT),
            ("curl", STUB_CURL),
            ("xcode-select", STUB_XCODE_SELECT),
            ("brew", STUB_BREW),
        ):
            self._write_stub(name, body)
        # Fake Xcode CLT developer dir: ensure_xcode_clt requires
        # `$(xcode-select -p)/usr/bin/git` to exist.
        self.devdir = self.tmp / "devdir"
        (self.devdir / "usr" / "bin").mkdir(parents=True)
        (self.devdir / "usr" / "bin" / "git").touch()
        self.env = {
            "PATH": f"{self.stubs}:{tools}",
            "HOME": str(self.home),
            "KISS_TEST_MARK_DIR": str(self.marks),
            "KISS_TEST_RELEASE": str(self.release),
            "KISS_TEST_RELEASE2": str(self.release2),
            "KISS_TEST_DEVDIR": str(self.devdir),
            # Belt and braces: never let update_repo touch this checkout.
            "KISS_SKIP_UPDATE": "1",
            "LANG": "C",
        }
        self.lock_file = self.home / ".kiss" / ".update.lock"
        self._procs: list[subprocess.Popen[str]] = []

    def tearDown(self) -> None:
        self._release_all()
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

    def _write_stub(self, name: str, body: str) -> None:
        (self.stubs / name).write_text(body)
        (self.stubs / name).chmod(0o755)

    def _start(
        self,
        extra_env: dict[str, str] | None = None,
        gate: Path | None = None,
        script: Path = SCRIPT,
    ) -> subprocess.Popen[str]:
        # With a gate, the contender reports itself ready and then blocks in
        # open(2) on the FIFO until the test opens it, which releases every
        # waiting reader at once (a reader arriving later passes straight
        # through: the test keeps the FIFO open).
        argv = ["bash", str(script), "--non-interactive"]
        if gate is not None:
            argv = [
                "bash", "-c",
                f'echo $$ >> "{self.marks}/ready"; exec 8<"{gate}"; exec 8<&-; '
                f'exec bash "{script}" --non-interactive',
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
            ["bash", str(SCRIPT), "--non-interactive"],
            cwd=self.tmp,
            env={**self.env, **(extra_env or {})},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def _wait_for(self, pred, what: str, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while not pred():
            self.assertLess(time.monotonic(), deadline, f"timed out waiting for {what}")
            time.sleep(0.02)

    def _count(self, mark: str) -> int:
        path = self.marks / mark
        return len(path.read_text().split()) if path.exists() else 0

    def _lock_pid(self) -> int:
        return int(self.lock_file.read_text().strip())

    def _release_all(self) -> None:
        self.release.write_text("")
        self.release2.write_text("")

    def test_second_concurrent_run_is_refused_and_lock_is_released(self) -> None:
        first = self._start()
        self._wait_for(lambda: self._count("started") == 1, "first installer")
        self.assertFalse(_lock_is_free(self.lock_file), "lock not held while installing")
        holder = self._lock_pid()
        self.assertTrue(_alive(holder), "the lock names a dead pid")
        # The lock is owned by the re-exec'd (setsid) bash that does the
        # work: its parent is the perl supervisor we spawned.
        self.assertEqual(_parent_pid(holder), first.pid)

        second = self._run()
        self.assertEqual(second.returncode, 1, second.stdout + second.stderr)
        self.assertIn(f"{REFUSED}{holder}); exiting.", second.stdout + second.stderr)
        self.assertEqual(self._count("started"), 1, "second run reached the install body")
        self.assertFalse(_lock_is_free(self.lock_file), "the loser released the winner's lock")
        self.assertEqual(self._lock_pid(), holder)

        self._release_all()
        out, _ = first.communicate(timeout=30)
        # The whitelist PATH has no node: the script stops at the Node.js
        # stage, i.e. a failure exit under ``set -e`` must release the lock.
        self.assertEqual(first.returncode, 1, out)
        self.assertIn(NODE_EXIT, out)
        self.assertEqual(self._count("curl-started"), 1, out)
        self.assertTrue(_lock_is_free(self.lock_file), "lock survived a failure exit")

        # With the lock free, the next run proceeds.
        third = self._run()
        self.assertNotIn(REFUSED, third.stdout + third.stderr)
        self.assertIn(NODE_EXIT, third.stdout)
        self.assertEqual(self._count("started"), 2)
        self.assertTrue(_lock_is_free(self.lock_file))

    def test_eight_simultaneous_contenders_admit_exactly_one_installer(self) -> None:
        # Legacy leftover of the mkdir protocol (a crash between mkdir and
        # writing the pid): it must be irrelevant, never "recovered".
        legacy = self.home / ".kiss" / ".update.lock.d"
        for trial in range(5):
            legacy.mkdir(parents=True, exist_ok=True)
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
                f"trial {trial}: more than one install body entered",
            )
            winners = [p for p in contenders if p.poll() is None]
            self.assertEqual(len(winners), 1)
            winner = winners[0]
            holder = self._lock_pid()
            self.assertEqual(_parent_pid(holder), winner.pid)
            self._release_all()
            out, _ = winner.communicate(timeout=30)
            self.assertIn(NODE_EXIT, out)
            for loser in contenders:
                if loser is winner:
                    continue
                lost, _ = loser.communicate(timeout=30)
                self.assertEqual(loser.returncode, 1, lost)
                self.assertIn(f"{REFUSED}{holder}); exiting.", lost)
            self.assertEqual(self._count("started"), trial + 1)
            self.assertTrue(_lock_is_free(self.lock_file))
            os.close(gate_fd)
            self.release.unlink()
            self.release2.unlink()
            self._procs.clear()

    def test_same_home_different_kiss_home_contenders_exclude_each_other(self) -> None:
        # The checkout and the global extension install follow $HOME, so
        # the lock must too -- a KISS_HOME override must not pick another.
        kiss_home = self.tmp / "custom-kiss-home"
        first = self._start({"KISS_HOME": str(kiss_home)})
        self._wait_for(lambda: self._count("started") == 1, "first installer")
        self.assertFalse(_lock_is_free(self.lock_file), "lock must live under $HOME/.kiss")
        self.assertFalse((kiss_home / ".update.lock").exists(), "lock followed KISS_HOME")
        holder = self._lock_pid()
        second = self._run()
        self.assertEqual(second.returncode, 1, second.stdout + second.stderr)
        self.assertIn(f"{REFUSED}{holder}); exiting.", second.stdout + second.stderr)
        self.assertEqual(self._count("started"), 1, "both contenders entered the install body")
        self._release_all()
        first.communicate(timeout=30)
        self.assertTrue(_lock_is_free(self.lock_file))

    def test_sigterm_releases_the_lock(self) -> None:
        first = self._start()
        self._wait_for(lambda: self._count("started") == 1, "first installer")
        holder = self._lock_pid()
        # The perl supervisor (first.pid) ignores TERM by design; the signal
        # has to reach the detached bash that owns the lock.  Its
        # ``handle_interrupt`` trap ignores ONE signal and honours a second
        # one within 3 s; bash runs a trap only after the foreground command
        # (the blocked stub) returns, so each signal is followed by a release.
        os.kill(holder, signal.SIGTERM)
        self.release.write_text("")
        self._wait_for(lambda: self._count("curl-started") == 1, "curl stage")
        self.assertFalse(_lock_is_free(self.lock_file), "lock dropped on the first, ignored TERM")
        os.kill(holder, signal.SIGTERM)
        self.release2.write_text("")
        out, _ = first.communicate(timeout=30)
        self.assertEqual(first.returncode, 130, out)
        self.assertIn("Second interrupt received", out)
        self.assertNotIn(NODE_EXIT, out)
        self.assertTrue(_lock_is_free(self.lock_file), "lock survived SIGTERM")

    def _seed_checkout(self, checkout: Path, git_env: dict[str, str]) -> None:
        """Commit a copy of install.sh into a throwaway clone at *checkout*."""
        real_git = shutil.which("git")
        assert real_git is not None
        origin = self.tmp / "origin.git"
        for args, cwd in (
            (["init", "--bare", "-q", "-b", "main", str(origin)], self.tmp),
            (["clone", "-q", str(origin), str(checkout)], self.tmp),
        ):
            subprocess.run([real_git, *args], cwd=cwd, env=git_env, check=True, capture_output=True)
        shutil.copy(SCRIPT, checkout / "install.sh")
        for args in (
            ["add", "install.sh"],
            ["commit", "-q", "-m", "seed"],
            ["push", "-q", "-u", "origin", "main"],
        ):
            subprocess.run(
                [real_git, *args],
                cwd=checkout,
                env=git_env,
                check=True,
                capture_output=True,
            )

    def test_exit_trap_restores_stash_and_exit_releases_lock(self) -> None:
        # ``update_repo`` installs its own EXIT trap when it stashes a dirty
        # tree.  The lock must not depend on any EXIT trap: it goes away
        # with the process whatever trap ran last.  Run a committed COPY of
        # install.sh from a throwaway git checkout (PROJECT_DIR is dirname
        # "$0") with the real git, dirty it, and hold the run at the curl
        # stub instead of the git stub.
        real_git = shutil.which("git")
        assert real_git is not None
        checkout = self.tmp / "checkout"
        git_env = {
            **self.env,
            "PATH": f"{self.env['PATH']}:{Path(real_git).parent}",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.com",
        }
        (self.stubs / "git").unlink()
        del git_env["KISS_SKIP_UPDATE"]
        self._seed_checkout(checkout, git_env)
        dirty = checkout / "local-edit.txt"
        dirty.write_text("uncommitted\n")

        first = subprocess.Popen(
            ["bash", str(checkout / "install.sh"), "--non-interactive"],
            cwd=self.tmp,
            env=git_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._procs.append(first)
        self._wait_for(lambda: self._count("curl-started") == 1, "curl stage")
        self.assertFalse(_lock_is_free(self.lock_file), "lock not held while installing")
        self.assertFalse(dirty.exists(), "the dirty file was not stashed")
        self.release2.write_text("")
        out, _ = first.communicate(timeout=30)
        self.assertEqual(first.returncode, 1, out)
        self.assertIn("Repository is dirty — stashing local changes", out)
        self.assertIn("Restoring stashed local changes", out)
        self.assertEqual(dirty.read_text(), "uncommitted\n", "stash was not popped")
        self.assertTrue(_lock_is_free(self.lock_file), "lock survived the stash-restoring exit")

    def test_nested_run_under_a_held_lock_neither_blocks_nor_releases(self) -> None:
        # scripts/install.sh exports KISS_UPDATE_LOCK_HELD=1 before handing
        # over to this script; the nested run must not deadlock on its
        # ancestor's lock, nor release it on exit.  This process plays the
        # ancestor: it holds the real lock for the whole test.
        self.lock_file.parent.mkdir(parents=True)
        holder = os.open(self.lock_file, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.write(holder, f"{os.getpid()}\n".encode())
            self._release_all()
            result = self._run({"KISS_UPDATE_LOCK_HELD": "1"})
            self.assertNotIn(REFUSED, result.stdout + result.stderr)
            self.assertIn(NODE_EXIT, result.stdout)
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

    def test_launched_background_children_do_not_keep_the_lock(self) -> None:
        # A complete run: npm ci, code --install-extension and the final
        # VS Code launch each leave a process behind.  None of them may
        # inherit fd 9, or the lock would stay held until they exit.
        for name, body in (
            ("node", STUB_NODE),
            ("npm", STUB_NPM),
            ("npx", STUB_NODE),
            ("code", STUB_CODE),
            ("uname", STUB_UNAME),
        ):
            self._write_stub(name, body)
        project = self.tmp / "project"
        (project / "src" / "kiss" / "agents" / "vscode").mkdir(parents=True)
        shutil.copy(SCRIPT, project / "install.sh")
        tools = self.tmp / "tools"
        for name in ("nohup", "chmod"):
            real = shutil.which(name)
            assert real is not None
            os.symlink(real, tools / name)

        first = self._start(script=project / "install.sh")
        self._wait_for(lambda: self._count("started") == 1, "first installer")
        self.assertFalse(_lock_is_free(self.lock_file), "lock not held while installing")
        self.release.write_text("")
        out, _ = first.communicate(timeout=30)
        self.assertEqual(first.returncode, 0, out)
        self.assertIn("=== Source bootstrap complete ===", out)
        self.assertIn("Launched VS Code from", out)

        daemons = [int(pid) for pid in (self.marks / "daemon").read_text().split()]
        self.assertEqual(len(daemons), 3, out)
        for pid in daemons:
            self.assertTrue(_alive(pid), f"leftover {pid} should still be running")
        self.assertTrue(
            _lock_is_free(self.lock_file),
            "a still-running launched child kept the update lock",
        )
        # And a new update may start right away.
        again = self._run()
        self.assertNotIn(REFUSED, again.stdout + again.stderr)
        self.assertEqual(self._count("started"), 2)


if __name__ == "__main__":
    unittest.main()
