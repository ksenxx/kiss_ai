# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests: ``saveConfig`` must not restart the kiss-web daemon
unless the remote-access password actually changed.

Background (task 3515, "Task interrupted by server restart/shutdown"):
the chat webview passively flushes the settings form via ``saveConfig``
(on settings-panel close and on blur/change/Enter of the remote-password
inputs), echoing back the already-saved non-empty ``remote_password``.
``_cmd_save_config`` then called ``_restart_kiss_web_daemon()`` whenever
the password was non-empty — even when it was identical to the stored
value — which ran ``launchctl kickstart -k`` and SIGTERMed the daemon,
killing every in-flight agent task with no user action at all.

These tests exercise the real ``_cmd_save_config`` handler with real
config persistence.  The actual ``launchctl``/``systemctl`` binaries are
shadowed by stub executables on a private ``PATH`` that record their
invocation to a marker file, so the genuine restart subprocess is spawned
and observed without ever touching the machine's real kiss-web daemon.
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


class TestSaveConfigDaemonRestart(unittest.TestCase):
    """_cmd_save_config restarts the daemon only on a real password change."""

    def setUp(self) -> None:
        import kiss.server.vscode_config as vc

        self._tmpdir = tempfile.mkdtemp()
        self._orig_dir = vc.CONFIG_DIR
        self._orig_path = vc.CONFIG_PATH
        vc.CONFIG_DIR = Path(self._tmpdir)
        vc.CONFIG_PATH = Path(self._tmpdir) / "config.json"

        # Stub launchctl/systemctl on a private PATH.  The stubs append
        # their argv to a marker file so the test can observe whether the
        # restart subprocess actually ran — without kicking the real
        # daemon (which would kill in-flight work on a dev machine).
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

        # Simulate a production daemon: `_restart_kiss_web_daemon` only
        # dispatches when KISS_HOME is the default ~/.kiss (the guard
        # added after the 2026-06-11 incident where a test process
        # SIGTERMed the live daemon).  Config reads/writes stay fully
        # isolated via the patched vc.CONFIG_DIR/CONFIG_PATH above, and
        # the restart subprocess only ever reaches the PATH stubs.
        self._orig_kiss_home = os.environ.get("KISS_HOME")
        os.environ["KISS_HOME"] = str(Path.home() / ".kiss")

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

    def _save(self, server: Any, config: dict[str, Any]) -> None:
        server._cmd_save_config({"config": config, "apiKeys": {}})

    def _wait_for_restart(self, timeout: float = 10.0) -> bool:
        """Poll for the stub launchctl/systemctl marker file."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._marker.exists():
                return True
            time.sleep(0.05)
        return False

    def test_unchanged_password_does_not_restart_daemon(self) -> None:
        """Echoing back the stored password must NOT restart the daemon.

        This is the exact sequence the webview produces when the settings
        panel is closed (or the password input blurs) with no edits: the
        previously saved password is posted back verbatim.  Pre-fix this
        kickstarted the daemon and interrupted in-flight tasks.
        """
        from kiss.server.vscode_config import save_config

        save_config({"remote_password": "hunter2"})
        server = self._make_server()
        self._save(server, {"remote_password": "hunter2", "max_budget": 100})
        assert not self._wait_for_restart(timeout=4.0), (
            "daemon was restarted although remote_password did not change"
        )

    def test_changed_password_restarts_daemon(self) -> None:
        """A genuinely new password must still restart the daemon."""
        from kiss.server.vscode_config import save_config

        save_config({"remote_password": "old-secret"})
        server = self._make_server()
        self._save(server, {"remote_password": "new-secret"})
        assert self._wait_for_restart(), (
            "daemon was not restarted although remote_password changed"
        )

    def test_first_password_set_restarts_daemon(self) -> None:
        """Setting a password for the first time restarts the daemon."""
        server = self._make_server()
        self._save(server, {"remote_password": "fresh-secret"})
        assert self._wait_for_restart(), (
            "daemon was not restarted when remote_password was first set"
        )

    def test_empty_password_does_not_restart_daemon(self) -> None:
        """Saving a config without a password never restarts the daemon."""
        from kiss.server.vscode_config import save_config

        save_config({"remote_password": "hunter2"})
        server = self._make_server()
        self._save(server, {"remote_password": "", "max_budget": 50})
        assert not self._wait_for_restart(timeout=4.0), (
            "daemon was restarted although the posted password was empty"
        )

    def test_unchanged_password_still_persists_other_fields(self) -> None:
        """The no-restart path must not skip persisting the rest of the form."""
        from kiss.server.vscode_config import load_config, save_config

        save_config({"remote_password": "hunter2", "max_budget": 100})
        server = self._make_server()
        self._save(server, {"remote_password": "hunter2", "max_budget": 42})
        cfg = load_config()
        assert cfg["max_budget"] == 42
        assert cfg["remote_password"] == "hunter2"


if __name__ == "__main__":
    unittest.main()
