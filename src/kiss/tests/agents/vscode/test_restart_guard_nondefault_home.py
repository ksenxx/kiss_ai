# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests for the 2026-06-11 00:37:45 self-inflicted daemon kill.

Incident (forensics from ``~/.kiss/kiss-web-stderr.log`` + ``sorcar.db``):
a bug-hunt sub-agent (task_history row 3624) ran a freshly written
pre-fix test that exercised ``_cmd_save_config`` with a junk
``remote_password`` against an in-process ``VSCodeServer``.  The handler
judged it a genuine password change and called
``_restart_kiss_web_daemon()`` — which ran
``launchctl kickstart -k gui/<uid>/com.kiss.web-server`` and SIGTERMed
the developer's REAL kiss-web daemon (pid 2884) at the exact second the
test ran, killing the whole in-flight bug-hunt task tree (rows 3556,
3618-3624) that was running the test in the first place.

Root cause: ``_restart_kiss_web_daemon`` kick-started the system
LaunchAgent unconditionally, even from a process whose ``KISS_HOME``
points at a private/sandboxed home (every pytest process — see
``src/kiss/tests/conftest.py`` — and any sandboxed run).  Such a process
manages a config.json the system daemon never reads, so restarting that
daemon can only destroy unrelated in-flight work.

Fix under test: ``_restart_kiss_web_daemon`` is a no-op (returns False)
whenever ``KISS_HOME`` is set to a non-default location; it dispatches
the restart (returns True) only when the process operates on the
default ``~/.kiss``.

The real restart subprocess is observed via stub ``launchctl`` /
``systemctl`` executables on a private ``PATH`` that record their argv
to a marker file — so even the pre-fix failure mode never touches the
machine's real daemon while the test demonstrates it.
"""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any


class _FakePrinter:
    """Collects broadcast events emitted by the command handler."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def broadcast(self, msg: dict[str, Any]) -> None:
        """Record a broadcast event."""
        self.messages.append(msg)


class TestRestartGuardNonDefaultHome(unittest.TestCase):
    """The kiss-web LaunchAgent must never be restarted from a process
    operating on a non-default ``KISS_HOME`` (e.g. the test suite)."""

    def setUp(self) -> None:
        import kiss.server.vscode_config as vc

        self._tmpdir = tempfile.mkdtemp()
        self._orig_dir = vc.CONFIG_DIR
        self._orig_path = vc.CONFIG_PATH
        vc.CONFIG_DIR = Path(self._tmpdir)
        vc.CONFIG_PATH = Path(self._tmpdir) / "config.json"

        # Stub launchctl/systemctl on a private PATH so a (pre-fix)
        # restart is observable without kicking the real daemon.
        self._stub_bin = Path(self._tmpdir) / "bin"
        self._stub_bin.mkdir()
        self._marker = Path(self._tmpdir) / "restart-invoked.log"
        script = f'#!/bin/sh\necho "$0 $@" >> "{self._marker}"\n'
        for name in ("launchctl", "systemctl"):
            stub = self._stub_bin / name
            stub.write_text(script)
            stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
        self._orig_env_path = os.environ.get("PATH", "")
        os.environ["PATH"] = str(self._stub_bin)

        # conftest.py already points KISS_HOME at a per-run temp dir;
        # pin it explicitly so the test is self-contained.
        self._orig_kiss_home = os.environ.get("KISS_HOME")
        os.environ["KISS_HOME"] = self._tmpdir

    def tearDown(self) -> None:
        import kiss.server.vscode_config as vc

        os.environ["PATH"] = self._orig_env_path
        if self._orig_kiss_home is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = self._orig_kiss_home
        vc.CONFIG_DIR = self._orig_dir
        vc.CONFIG_PATH = self._orig_path
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_server(self) -> Any:
        from kiss.server.commands import _CommandsMixin

        class FakeServer(_CommandsMixin):
            def __init__(self) -> None:
                self.printer = _FakePrinter()  # type: ignore[assignment]
                self.work_dir = "/tmp"
                self._state_lock = threading.RLock()
                self._default_model = ""

            def _get_models(self, conn_id: str = "") -> None:
                pass

        return FakeServer()

    def _restart_invoked(self, timeout: float = 3.0) -> bool:
        """Poll for the stub launchctl/systemctl marker file."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._marker.exists():
                return True
            time.sleep(0.05)
        return False

    def test_password_change_under_custom_home_does_not_restart(self) -> None:
        """A genuine password change in a custom-``KISS_HOME`` process
        must NOT restart the system daemon.

        Pre-fix this was the incident path: the pytest process's
        ``_cmd_save_config`` SIGTERMed the live daemon that was running
        the agent which had launched the test.
        """
        from kiss.server.vscode_config import save_config

        save_config({"remote_password": "old-secret"})
        server = self._make_server()
        server._cmd_save_config(
            {"config": {"remote_password": "brand-new-secret"}, "apiKeys": {}},
        )
        assert not self._restart_invoked(), (
            "saveConfig from a process with a non-default KISS_HOME "
            "restarted the system kiss-web daemon — this is the failure "
            "mode that killed the in-flight bug-hunt task tree (task 3556 "
            "+ sub-tasks 3618-3624) on 2026-06-11 00:37:45"
        )

    def test_helper_skips_and_reports_under_custom_home(self) -> None:
        """``_restart_kiss_web_daemon`` returns False and spawns nothing."""
        from kiss.server.commands import _restart_kiss_web_daemon

        dispatched = _restart_kiss_web_daemon()
        assert dispatched is False, (
            "_restart_kiss_web_daemon dispatched a restart although "
            "KISS_HOME points at a non-default location"
        )
        assert not self._restart_invoked(timeout=1.5)

    def test_helper_restarts_under_default_home(self) -> None:
        """Control: with the default ``~/.kiss`` home the restart still
        dispatches (observed via the PATH stub, never the real daemon)."""
        from kiss.server.commands import _restart_kiss_web_daemon

        os.environ["KISS_HOME"] = str(Path.home() / ".kiss")
        dispatched = _restart_kiss_web_daemon()
        assert dispatched is True
        assert self._restart_invoked(timeout=8.0), (
            "restart subprocess was not spawned under the default home"
        )

    def test_helper_restarts_when_kiss_home_unset(self) -> None:
        """Control: an unset ``KISS_HOME`` means the default home."""
        from kiss.server.commands import _restart_kiss_web_daemon

        os.environ.pop("KISS_HOME", None)
        dispatched = _restart_kiss_web_daemon()
        assert dispatched is True
        assert self._restart_invoked(timeout=8.0)


if __name__ == "__main__":
    unittest.main()
