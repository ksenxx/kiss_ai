# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: a restart the installer deferred finishes on its own.

The whole chain, with nothing stubbed between its parts:

1. a real kiss-web (``RemoteAccessServer``, in a subprocess whose HOME is a
   sandbox) is running a task;
2. ``install.sh``'s runtime setup (the verbatim block, as
   ``test_install_runtime_setup`` runs it) asks it for tasks in flight
   through the real probe, gets the record file the daemon names, writes
   the deferred restart there and leaves the daemon alone;
3. the task finishes, and the daemon runs the repository's real
   ``install.sh --restart-only`` (its environment is the sandbox's), which
   starts the installed app's kiss-web -- a stand-in that records its work
   dir, writes the endpoint file and listens on ``KISS_WEB_PORT``.

In the second case the installer that deferred is still finishing when the
task ends: it holds the update lock, and the restart waits for it.

The sandbox has no ``systemctl``, ``launchctl``, ``lsof`` or ``fuser``, so
the installer takes its detached-process branch and stops nothing; the
daemon subprocess is killed at the end.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from kiss.server.web_server import _generate_self_signed_cert
from kiss.tests.conftest import posix_only
from kiss.tests.scripts.test_install_update_lock import WHITELISTED_TOOLS
from kiss.tests.test_install_runtime_setup import REPO, Sandbox

pytestmark = posix_only("runs install.sh under bash")

# The busy daemon: a real server with one task in flight until {release}
# exists, polling for deferred restarts every 0.1 s.
DAEMON = """\
import asyncio, sys
from pathlib import Path
import kiss.server.web_server as ws
from kiss.server import agent_state

ws._IDLE_UPDATE_POLL_S = 0.1
ws._PYPI_LATEST_URL = "http://127.0.0.1:1/"
home, release, cert, key = (Path(a) for a in sys.argv[1:5])


async def main():
    server = ws.RemoteAccessServer(
        host="127.0.0.1", port=0, certfile=str(cert), keyfile=str(key),
        url_file=home / "remote-url.json", local_endpoint_file=home / "sorcar-local.json",
        scheduler=False,
    )
    await server.start_async()
    task = agent_state.AgentState(
        "e2e-task", tab_id="e2e-tab", server_owned=True, is_task_active=True
    )
    agent_state.register(task)
    (home / "daemon-ready").touch()
    while not release.exists():
        await asyncio.sleep(0.05)
    agent_state.unregister("e2e-task", task)
    await asyncio.sleep(600)


asyncio.run(main())
"""

# The installer that deferred, still finishing: holds the update lock the
# way install.sh's acquire_update_lock does (its pid in the file) until
# {unlock} exists.
LOCK_HOLDER = """\
import fcntl, os, sys, time
from pathlib import Path
lock, locked, unlock = (Path(a) for a in sys.argv[1:4])
with open(lock, "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    lock.write_text(str(os.getpid()))
    locked.touch()
    while not unlock.exists():
        time.sleep(0.05)
"""


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _wait(predicate, what: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.05)


@pytest.fixture
def sandbox(tmp_path: Path):
    box = Sandbox(tmp_path)
    try:
        yield box
    finally:
        box.kill_daemons()


@pytest.mark.parametrize("installer_still_running", [False, True])
def test_a_deferred_restart_finishes_once_the_task_is_done(
    sandbox: Sandbox, installer_still_running: bool,
) -> None:
    project = sandbox.project()
    sandbox.uname("Linux")
    # install.sh refuses to run without curl; a restart never calls it.
    sandbox.stub("curl", "exit 22\n")
    kiss_home = sandbox.home / ".kiss"
    record = kiss_home / ".kiss-web.restart-pending"
    release = sandbox.root / "release"
    # What the real install.sh needs on PATH beyond the block's tools.
    tools = sandbox.root / "installer-tools"
    tools.mkdir()
    for name in (*WHITELISTED_TOOLS, "nohup", "chmod", "id", "mv", "env", "basename", "touch"):
        real = shutil.which(name)
        assert real is not None, f"{name} is required by this test"
        os.symlink(real, tools / name)
    cert, key = sandbox.root / "cert.pem", sandbox.root / "key.pem"
    _generate_self_signed_cert(cert, key)
    env = {
        "HOME": str(sandbox.home),
        "PATH": f"{sandbox.stubs}:{tools}",
        "KISS_WEB_PORT": str(sandbox.port),
        "LANG": "C",
    }
    daemon = subprocess.Popen(
        [sys.executable, "-c", DAEMON, str(kiss_home), str(release), str(cert), str(key)],
        env=env, cwd=sandbox.workdir, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=(sandbox.root / "daemon.log").open("wb"),
    )
    holder: subprocess.Popen[bytes] | None = None
    fingerprint = kiss_home / ".kiss-web.fingerprint"
    locked, unlock = sandbox.root / "locked", sandbox.root / "unlock"
    try:
        _wait(lambda: (kiss_home / "daemon-ready").exists(), "the daemon to start")

        res = sandbox.run(f'start_kiss_web_daemon "{project}" "{sandbox.workdir}"')

        assert "restart deferred: kiss-web restarts itself onto this build" in res.stdout, res
        assert json.loads(record.read_text(encoding="utf-8")) == {
            "installer": str(REPO / "install.sh"),
            "home": str(kiss_home),
            "workdir": str(sandbox.workdir),
        }
        time.sleep(1.0)  # ten polls: the task is still running
        assert not (kiss_home / "kiss-web.cwd").exists()
        assert record.exists()

        if installer_still_running:
            holder = subprocess.Popen(
                [sys.executable, "-c", LOCK_HOLDER, str(kiss_home / ".update.lock"),
                 str(locked), str(unlock)],
            )
            _wait(locked.exists, "the update lock to be taken")
        release.touch()
        if holder is not None:
            _wait(
                lambda: "Waiting for the KISS update running as pid"
                in _read(kiss_home / "update.log"),
                "the restart to wait for the update lock",
            )
            assert not record.exists()
            time.sleep(1.0)
            assert not fingerprint.exists()
            unlock.touch()
            holder.wait(10)
        # install.sh writes the fingerprint and drops the record before it
        # logs "kiss-web is up", and its log goes through a tee that can lag
        # behind under load: the line is the restart's last trace.
        _wait(
            lambda: "kiss-web is up" in _read(kiss_home / "install.log"),
            "the restart to finish",
        )
        assert fingerprint.exists()
        assert (kiss_home / "kiss-web.cwd").read_text().strip() == str(sandbox.workdir)
        assert not record.exists()
        log = (kiss_home / "install.log").read_text(encoding="utf-8")
        assert f">>> Restarting kiss-web onto the app in {project}..." in log
        assert "[1/5]" not in log
    finally:
        if holder is not None:
            holder.kill()
            holder.wait()
        daemon.kill()
        daemon.wait()
