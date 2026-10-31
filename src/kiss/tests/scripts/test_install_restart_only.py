# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test of ``install.sh --restart-only``.

kiss-web runs ``install.sh --restart-only`` (``KISS_HOME`` set, from the
work dir to serve) to finish a restart an earlier run deferred while
tasks were in flight.  The app is installed already; the run must only
(re)start the daemon from it -- no copy, no build, no hooks, no
password box, no webapp window -- keep the problems the deferred run
recorded for the restarted daemon to report, and do what a restart by
step [5/5] does: record the fingerprint, drop the deferred-restart
record and retire the app directory of the old name.

The real ``install.sh`` runs from a seeded fake checkout with a fake
HOME and the whitelisted PATH of ``test_install_update_lock`` (no
``systemctl``: the daemon is started as a detached process).  The
installed app's ``python`` is this test's interpreter and its
``kiss-web`` a stand-in daemon that records how it was started, writes
the endpoint file and listens on ``KISS_WEB_PORT``, as kiss-web does.
"""

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

from kiss.tests.conftest import find_free_port
from kiss.tests.scripts.test_install_brand_homes_vscode_notice import STUB_UV
from kiss.tests.scripts.test_install_update_lock import (
    STUB_BREW,
    STUB_CLOUDFLARED,
    STUB_CURL,
    STUB_GIT,
    STUB_UNAME,
    WHITELISTED_TOOLS,
    _alive,
    _seed_project,
)

pytest.importorskip("fcntl", reason="POSIX-only: bash install.sh and flock")

KISS_WEB = """\
#!/bin/bash
echo "$$" >> "$KISS_TEST_MARK_DIR/daemon"
echo "KISS_HOME=$KISS_HOME APP=$0 PWD=$(pwd -P)" >> "$KISS_TEST_MARK_DIR/daemon-start"
printf '{"url":"ws://127.0.0.1:1/","token":"t"}' > "$KISS_HOME/sorcar-local.json"
exec "$(dirname "$0")/python" -c 'import os, socket, time
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", int(os.environ["KISS_WEB_PORT"])))
s.listen()
time.sleep(60)'
"""


class RestartOnlyTest(unittest.TestCase):
    """``install.sh --restart-only`` restarts kiss-web and does nothing else."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-install-restart-only-"))
        self.home = self.tmp / "home"
        self.marks = self.tmp / "marks"
        self.workdir = self.tmp / "work"
        for directory in (self.home, self.marks, self.workdir):
            directory.mkdir()
        tools = self.tmp / "tools"
        tools.mkdir()
        for name in (*WHITELISTED_TOOLS, "nohup", "chmod", "id", "mv", "env", "basename", "rmdir"):
            real = shutil.which(name)
            assert real is not None, f"{name} is required by this test"
            os.symlink(real, tools / name)
        stubs = self.tmp / "stubs"
        stubs.mkdir()
        for name, body in (
            ("git", STUB_GIT), ("curl", STUB_CURL), ("brew", STUB_BREW), ("uname", STUB_UNAME),
        ):
            (stubs / name).write_text(body)
            (stubs / name).chmod(0o755)
        self.project = self.tmp / "project"
        _seed_project(self.project)
        # The brand home kiss-web runs from, its app named after it.
        self.kiss_home = self.home / ".brand"
        self.app = self.kiss_home / "brand"
        (self.app / "src" / "kiss").mkdir(parents=True)
        (self.app / "src" / "kiss" / "a.pyc").write_bytes(b"")
        bin_dir = self.app / ".venv" / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "python").write_text(f'#!/bin/bash\nexec {sys.executable!r} "$@"\n')
        (bin_dir / "kiss-web").write_text(KISS_WEB)
        for name in ("python", "kiss-web"):
            (bin_dir / name).chmod(0o755)
        self.env = {
            "PATH": f"{stubs}:{tools}",
            "HOME": str(self.home),
            "KISS_HOME": str(self.kiss_home),
            "KISS_WEB_PORT": str(find_free_port()),
            "KISS_WEB_START_TIMEOUT": "20",
            "KISS_TEST_MARK_DIR": str(self.marks),
            "KISS_TEST_RELEASE": str(self.tmp / "release"),
            "KISS_TEST_RELEASE2": str(self.tmp / "release2"),
            "LANG": "C",
        }
        (self.tmp / "release").touch()
        (self.tmp / "release2").touch()

    def tearDown(self) -> None:
        daemon = self.marks / "daemon"
        if daemon.exists():
            for pid in daemon.read_text().split():
                if _alive(int(pid)):
                    os.kill(int(pid), signal.SIGKILL)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _marks(self, name: str) -> list[str]:
        path = self.marks / name
        return path.read_text().splitlines() if path.exists() else []

    def test_restart_only_restarts_kiss_web_from_the_installed_app(self) -> None:
        problems = self.kiss_home / ".install-problems"
        problems.write_text("WARNING: a hook of the deferred run failed\n")
        record = self.kiss_home / ".kiss-web.restart-pending"
        record.write_text("{}\n")
        legacy = self.kiss_home / "kiss"
        (legacy / "src" / "kiss").mkdir(parents=True)
        (legacy / "src" / "kiss" / "old.pyc").write_bytes(b"")

        proc = subprocess.run(
            ["bash", str(self.project / "install.sh"), "--restart-only", "--non-interactive"],
            cwd=self.workdir,
            env=self.env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = proc.stdout + proc.stderr

        self.assertEqual(proc.returncode, 0, out)
        self.assertIn(f">>> Restarting kiss-web onto the app in {self.app}...", out)
        self.assertIn("Starting kiss-web as a detached background process...", out)
        self.assertIn("kiss-web is up", out)
        self.assertEqual(
            self._marks("daemon-start"),
            [
                f"KISS_HOME={self.kiss_home} APP={self.app / '.venv' / 'bin' / 'kiss-web'}"
                f" PWD={self.workdir.resolve()}"
            ],
        )
        # Nothing was installed and nothing else ran.
        for step in ("[1/5]", "[5/5]", "post-install hook", "Remote access password",
                     "Waiting for the kiss-web daemon", "App install complete"):
            self.assertNotIn(step, out)
        self.assertFalse((self.app / "pyproject.toml").exists())
        self.assertFalse((self.app / "src" / "kiss" / "__init__.py").exists())
        # What a restart by step [5/5] leaves behind.
        self.assertTrue((self.kiss_home / ".kiss-web.fingerprint").exists())
        self.assertFalse(record.exists())
        self.assertIn(f"Removed the old app directory {legacy}", out)
        self.assertFalse(legacy.exists())
        # The deferred run's problems stay for the restarted daemon.
        self.assertEqual(
            problems.read_text(), "WARNING: a hook of the deferred run failed\n",
        )
        install_log = (self.home / ".kiss" / "install.log").read_text()
        self.assertIn("Restarting kiss-web onto the app", install_log)

    def test_a_full_install_still_drops_the_previous_runs_problems(self) -> None:
        """Without --restart-only a run starts afresh, as before."""
        problems = self.kiss_home / ".install-problems"
        problems.write_text("WARNING: an earlier run's problem\n")
        stubs = Path(self.env["PATH"].split(":")[0])
        for name, body in (("uv", STUB_UV), ("cloudflared", STUB_CLOUDFLARED)):
            (stubs / name).write_text(body)
            (stubs / name).chmod(0o755)
        proc = subprocess.run(
            ["bash", str(self.project / "install.sh"), "--non-interactive"],
            cwd=self.workdir,
            env={**self.env, "KISS_SKIP_LAUNCH": "1", "KISS_TEST_PYTHON": sys.executable},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, out)
        self.assertIn(f">>> [1/5] Copying KISS Sorcar to {self.app}...", out)
        self.assertNotIn("earlier run's problem", problems.read_text() if problems.exists() else "")
