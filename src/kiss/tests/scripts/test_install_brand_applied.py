# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: after ``install.sh`` a brand home's app is branded,
lives in ``~/.<brand>/<brand>``, and every failure of the run is
recorded for the web app.

Every test runs the real root ``install.sh`` from a seeded fake checkout
with a fake HOME and the stubs of ``test_install_update_lock`` /
``test_install_brand_homes_vscode_notice``, except that this module's
``kiss-web`` stub comes up for real (writes the endpoint file and
listens on ``KISS_WEB_PORT``), so the installer sees the daemon run
from the new app and retires the old app directory.

The brand is Seamless Loop: a home ``~/.s10s`` with a post-install hook
that does what the s10s installer's hook does -- copies the brand's
``brand.json`` over the app's and installs the ``s10s`` CLI next to
``sorcar`` -- run two ways: the stock install (the daemon's Update
button) that repeats itself into the brand home, and a direct
``KISS_HOME=~/.s10s`` install (the s10s installer).

Covered:

* the app is installed and branded in ``~/.s10s/s10s`` (never
  ``~/.s10s/kiss``), the daemon and the CLIs run from there, and the
  stock app stays unbranded;
* the old ``~/.s10s/kiss`` loses the installer's files once the daemon
  runs from the new directory; what else it held stays;
* a failing hook, a failing nested pass and the outer run's own
  warnings land in ``.install-problems`` of the home whose daemon
  runs; a clean run leaves no file.
"""

import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import pytest

from kiss.tests.scripts.test_install_brand_homes_vscode_notice import (
    STUB_CODE,
    STUB_UV,
)
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

# The ``kiss-web`` stub of STUB_UV, made to come up: it writes the
# endpoint and URL files and listens on KISS_WEB_PORT until the test
# kills it.
STUB_UV_UP = STUB_UV.replace(
    "exec sleep 60\n",
    'home="${KISS_HOME:-$HOME/.kiss}"\n'
    "mkdir -p \"$home\"\n"
    "printf '{\"url\":\"wss://127.0.0.1:1/\",\"token\":\"t\"}' > \"$home/sorcar-local.json\"\n"
    "printf '{\"loopback\":\"http://127.0.0.1:%s\",\"tunnel\":\"https://up.trycloudflare.com\"}' "
    '"$KISS_WEB_PORT" > "$home/remote-url.json"\n'
    'exec "$KISS_TEST_PYTHON" -c \'import os, socket, time\n'
    "s = socket.socket()\n"
    "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
    's.bind(("127.0.0.1", int(os.environ["KISS_WEB_PORT"])))\n'
    "s.listen()\n"
    "time.sleep(120)'\n",
)
assert STUB_UV_UP != STUB_UV

# ``uv sync`` fails for the home named by a ``uv-sync-fails-<home>`` mark.
STUB_UV_SYNC_FAILS = STUB_UV_UP.replace(
    "    sync)\n",
    '    sync)\n        [ ! -e "$KISS_TEST_MARK_DIR/uv-sync-fails-${KISS_HOME##*/}" ] || exit 2\n',
)
assert STUB_UV_SYNC_FAILS != STUB_UV_UP

BRAND_JSON = (
    '{\n  "product_name": "Seamless Loop",\n  "short_name": "s10s",\n'
    '  "home_dir": ".s10s"\n}\n'
)

# The brand's post-install hook, as the s10s installer's: brands the app
# it is handed and installs the ``s10s`` CLI as a copy of ``sorcar``.
# A ``hook-fails`` mark makes it fail after branding.
HOOK = """#!/bin/bash
set -e
echo "KISS_HOME=$KISS_HOME APP=$KISS_APP_DIR" >> "$KISS_TEST_MARK_DIR/hooks"
cat "$KISS_HOME/brand.json" > "$KISS_APP_DIR/src/kiss/webapp/media/brand.json"
sed 's/sorcar "\\$@"/s10s "$@"/' "$HOME/.local/bin/sorcar" > "$HOME/.local/bin/s10s"
chmod 755 "$HOME/.local/bin/s10s"
[ ! -e "$KISS_TEST_MARK_DIR/hook-fails" ] || exit 3
"""

LEGACY_FILES = ("src/kiss/__init__.pyc", ".venv/bin/kiss-web", "pyproject.toml", "uv.lock")


class BrandAppliedTest(unittest.TestCase):
    """install.sh brands ~/.s10s/s10s and reports what failed."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-install-branded-"))
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.marks = self.tmp / "marks"
        self.marks.mkdir()
        tools = self.tmp / "tools"
        tools.mkdir()
        for name in (*WHITELISTED_TOOLS, "nohup", "chmod", "id", "cp", "mv", "env", "rmdir"):
            real = shutil.which(name)
            self.assertIsNotNone(real, f"{name} is required by this test")
            assert real is not None
            os.symlink(real, tools / name)
        self.stubs = self.tmp / "stubs"
        self.stubs.mkdir()
        for name, body in (
            ("git", STUB_GIT),
            ("curl", STUB_CURL),
            ("brew", STUB_BREW),
            ("uv", STUB_UV_SYNC_FAILS),
            ("uname", STUB_UNAME),
            ("cloudflared", STUB_CLOUDFLARED),
            ("code", STUB_CODE),
        ):
            (self.stubs / name).write_text(body)
            (self.stubs / name).chmod(0o755)
        self.project = self.tmp / "project"
        _seed_project(self.project)
        (self.project / "src" / "kiss" / "core" / "models").mkdir(parents=True)
        (self.project / "src" / "kiss" / "core" / "models" / "MODEL_INFO.json").write_text("{}\n")
        # The launchers step [4/5] wraps; absent, each is a warning.
        for name in ("rsorcar", "sorcar-docker"):
            (self.project / name).write_text("#!/bin/bash\n")
        (self.tmp / "release").touch()
        (self.tmp / "release2").touch()
        python = shutil.which("python3")
        assert python is not None
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        # No DISPLAY and no systemctl: a remote Linux box whose daemon is
        # a detached process; the stub daemon listens on a free port.
        self.env = {
            "PATH": f"{self.stubs}:{tools}",
            "HOME": str(self.home),
            "KISS_TEST_MARK_DIR": str(self.marks),
            "KISS_TEST_RELEASE": str(self.tmp / "release"),
            "KISS_TEST_RELEASE2": str(self.tmp / "release2"),
            "KISS_TEST_PYTHON": python,
            "KISS_WEB_PORT": str(self.port),
            "KISS_WEB_START_TIMEOUT": "15",
            "KISS_WEBAPP_WAIT_SECS": "10",
            "LANG": "C",
        }
        self.brand = self.home / ".s10s"
        (self.brand / "post-install.d").mkdir(parents=True)
        (self.brand / "post-install.d" / "brand").write_text(HOOK)
        (self.brand / "post-install.d" / "brand").chmod(0o755)
        (self.brand / "brand.json").write_text(BRAND_JSON)
        self.app = self.brand / "s10s"
        self.legacy = self.brand / "kiss"

    def tearDown(self) -> None:
        self._stop_daemons()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _stop_daemons(self) -> None:
        daemon = self.marks / "daemon"
        if not daemon.exists():
            return
        for pid in daemon.read_text().split():
            if _alive(int(pid)):
                os.kill(int(pid), signal.SIGKILL)
        daemon.unlink()
        (self.marks / "daemon-start").unlink(missing_ok=True)

    def _run(self, extra_env: dict[str, str] | None = None, ok: bool = True) -> str:
        """Run install.sh non-interactively; return its combined output."""
        proc = subprocess.run(
            ["bash", str(self.project / "install.sh"), "--non-interactive"],
            cwd=self.tmp,
            env={**self.env, **(extra_env or {})},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=180,
        )
        out = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0 if ok else 1, out)
        return out

    def _daemon_starts(self) -> list[str]:
        deadline = time.monotonic() + 10.0
        path = self.marks / "daemon-start"
        while not path.exists():
            self.assertLess(time.monotonic(), deadline, "the daemon was not started")
            time.sleep(0.05)
        return path.read_text().splitlines()

    def _seed_legacy(self, extra: str | None = None) -> None:
        """Lay out an old ~/.s10s/kiss app, with *extra* (a non-installer file) when given."""
        for rel in LEGACY_FILES:
            (self.legacy / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.legacy / rel).write_text("old\n")
        (self.legacy / ".venv" / "bin" / "kiss-web").chmod(0o755)
        if extra:
            (self.legacy / extra).parent.mkdir(parents=True, exist_ok=True)
            (self.legacy / extra).write_text("mine\n")

    def _assert_branded(self) -> None:
        """The brand's app is in ~/.s10s/s10s, branded, with the CLIs bound to it."""
        brand_json = self.app / "src" / "kiss" / "webapp" / "media" / "brand.json"
        self.assertEqual(brand_json.read_text(), BRAND_JSON)
        self.assertTrue((self.app / "src" / "kiss" / "__init__.pyc").exists())
        self.assertFalse((self.legacy / "src").exists(), "nothing was installed into ~/.s10s/kiss")
        self.assertEqual((self.brand / "MODEL_INFO.json").read_text(), "{}\n")
        s10s = (self.home / ".local" / "bin" / "s10s").read_text()
        self.assertIn(f'--directory "{self.app}" s10s "$@"', s10s)
        self.assertEqual(
            self._daemon_starts(),
            [f"KISS_HOME={self.brand} APP={self.app / '.venv' / 'bin' / 'kiss-web'}"],
        )

    def test_stock_update_brands_the_brand_home_and_retires_the_old_directory(self) -> None:
        # Before: the daemon ran from ~/.s10s/kiss, which also holds
        # artifacts of tasks run from there.
        self._seed_legacy(".kiss.artifacts/jobs/job_1/trajectory.json")
        unit = self.home / ".config" / "systemd" / "user" / "kiss-web.service"
        unit.parent.mkdir(parents=True)
        unit.write_text(
            f"[Service]\nEnvironment=KISS_HOME={self.brand}\n"
            f"ExecStart={self.legacy}/.venv/bin/kiss-web\n"
        )

        out = self._run()

        self.assertIn(f">>> [1/5] Copying KISS Sorcar to {self.home / '.kiss' / 'kiss'}...", out)
        self.assertIn("Installing into brand home ~/.s10s...", out)
        self.assertIn(f"   Installed into brand home {self.brand}", out)
        self.assertIn(
            f"The kiss-web service runs from brand home {self.brand}; restarting that app.", out
        )
        self.assertIn("kiss-web is up", out)
        self._assert_branded()
        self.assertEqual(
            (self.marks / "hooks").read_text(), f"KISS_HOME={self.brand} APP={self.app}\n"
        )
        # The stock app is installed next to it, unbranded.
        stock = self.home / ".kiss" / "kiss"
        self.assertTrue((stock / "src" / "kiss" / "__init__.pyc").exists())
        self.assertFalse((stock / "src" / "kiss" / "webapp" / "media" / "brand.json").exists())
        # The old directory lost the installer's files and kept the rest.
        self.assertIn(
            f"   Removed the old app from {self.legacy} (the app is in {self.app} now); "
            "what else it holds is yours to remove.", out,
        )
        for rel in LEGACY_FILES:
            self.assertFalse((self.legacy / rel).exists(), rel)
        self.assertFalse((self.legacy / ".venv").exists())
        self.assertEqual(
            (self.legacy / ".kiss.artifacts" / "jobs" / "job_1" / "trajectory.json").read_text(),
            "mine\n",
        )
        # Nothing failed: no problems recorded anywhere.
        self.assertNotIn("WARNING", out)
        self.assertFalse((self.brand / ".install-problems").exists())
        self.assertFalse((self.home / ".kiss" / ".install-problems").exists())

    def test_direct_brand_home_install_brands_the_app(self) -> None:
        # The s10s installer's way: KISS_HOME=~/.s10s, nothing but the
        # installer's files in the old directory.
        self._seed_legacy()
        out = self._run({"KISS_HOME": str(self.brand)})
        self.assertIn(f">>> [1/5] Copying KISS Sorcar to {self.app}...", out)
        self.assertNotIn("Installing into brand home", out)
        self.assertIn("kiss-web is up", out)
        self._assert_branded()
        self.assertIn(
            f"   Removed the old app directory {self.legacy} (the app is in {self.app} now).", out
        )
        self.assertFalse(self.legacy.exists())
        self.assertFalse((self.home / ".kiss" / "kiss").exists(), "no stock app: a brand install")
        self.assertFalse((self.brand / ".install-problems").exists())

        # An update of the same install: the app is rebuilt in place.
        self._stop_daemons()
        out = self._run({"KISS_HOME": str(self.brand)})
        self._assert_branded()
        self.assertNotIn("old app", out)

    def test_old_directory_stays_until_the_daemon_runs_from_the_new_one(self) -> None:
        self._seed_legacy()
        out = self._run({"KISS_HOME": str(self.brand), "KISS_WEB_START_TIMEOUT": "0"})
        self.assertIn("WARNING: kiss-web did not come up within 0s", out)
        self.assertIn(
            f"   The old app directory {self.legacy} is removed once kiss-web runs from "
            f"{self.app}.",
            out,
        )
        for rel in LEGACY_FILES:
            self.assertTrue((self.legacy / rel).exists(), rel)
        self.assertEqual(
            (self.brand / ".install-problems").read_text(),
            f"WARNING: kiss-web did not come up within 0s; see {self.brand}/kiss-web-stderr.log\n",
        )

    def test_failures_are_recorded_in_the_home_whose_daemon_runs(self) -> None:
        # A stock install whose brand hook fails and whose launcher step
        # warns (in the nested pass and in the outer run): every line
        # reaches the brand daemon's file, the nested pass's prefixed
        # with its home, since that daemon is the one restarted (the
        # brand is installed and branded); the stock home keeps none.
        (self.marks / "hook-fails").touch()
        (self.project / "rsorcar").unlink()
        out = self._run()
        hook = self.brand / "post-install.d" / "brand"
        self.assertIn(f"WARNING: post-install hook {hook} exited with status 3", out)
        self.assertIn(f"Brand home {self.brand} is installed and branded; starting the daemon", out)
        launcher = f"WARNING: {self.project}/rsorcar not found \u2014 skipping rsorcar launcher."
        self.assertEqual(
            (self.brand / ".install-problems").read_text().splitlines(),
            [
                launcher,
                f"~/.s10s: {launcher}",
                f"~/.s10s: WARNING: post-install hook {hook} exited with status 3",
            ],
        )
        self.assertFalse((self.home / ".kiss" / ".install-problems").exists())
        self._assert_branded()

        # The next run, with nothing failing, ends the report.
        self._stop_daemons()
        (self.marks / "hook-fails").unlink()
        (self.project / "rsorcar").write_text("#!/bin/bash\n")
        out = self._run()
        self.assertNotIn("WARNING", out)
        self.assertFalse((self.brand / ".install-problems").exists())

    def test_failed_nested_pass_and_failed_run_are_recorded(self) -> None:
        # uv sync fails in the brand home: the nested pass stops with an
        # ERROR, the outer run warns about the pass, takes the ERROR over
        # and, since the brand has no app to run, keeps its own file for
        # the stock daemon it starts.
        (self.marks / "uv-sync-fails-.s10s").touch()
        out = self._run()
        self.assertIn(f"   ERROR: uv sync failed in {self.app}", out)
        self.assertIn(f"WARNING: the install into brand home {self.brand} failed (status 1)", out)
        self.assertFalse((self.brand / ".install-problems").exists())
        self.assertEqual(
            (self.home / ".kiss" / ".install-problems").read_text().splitlines(),
            [
                f"WARNING: the install into brand home {self.brand} failed (status 1); "
                "this install carries on.",
                f"~/.s10s: ERROR: uv sync failed in {self.app}",
            ],
        )
        self.assertFalse((self.app / ".venv").exists())

        # The same failure in a direct install of the brand home: the
        # run fails, and the file names the step too when the stop was
        # not the installer's own ERROR.
        self._stop_daemons()
        out = self._run({"KISS_HOME": str(self.brand)}, ok=False)
        self.assertIn(f"   ERROR: uv sync failed in {self.app}", out)
        self.assertEqual(
            (self.brand / ".install-problems").read_text(),
            f"ERROR: uv sync failed in {self.app}\n",
        )
        (self.marks / "uv-sync-fails-.s10s").unlink()
        (self.project / "pyproject.toml").unlink()
        shutil.rmtree(self.project / "src" / "kiss")
        out = self._run({"KISS_HOME": str(self.brand)}, ok=False)
        self.assertIn(f"ERROR: {self.project}/src/kiss not found", out)
        self.assertEqual(
            (self.brand / ".install-problems").read_text(),
            f"ERROR: {self.project}/src/kiss not found; "
            "run this script from a KISS Sorcar checkout.\n",
        )
