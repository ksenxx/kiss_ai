# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here


"""Server-only tests extracted from ``kiss.tests.agents.vscode.test_config_custom_headers``.

Moved here because their full dependency closure touches only
kiss.core, kiss.agents.sorcar and kiss.server (task: relocate
core+sorcar+server-only test methods to tests/server).
"""


from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any


class TestCmdSaveConfigHandlesHeaders(unittest.TestCase):
    """_cmd_save_config persists custom_headers and they appear in configData."""

    def setUp(self) -> None:
        import kiss.core.vscode_config as vc
        from kiss.core import config as config_module

        self._orig_dir = vc.CONFIG_DIR
        self._orig_path = vc.CONFIG_PATH
        self._tmpdir = tempfile.mkdtemp()
        vc.CONFIG_DIR = Path(self._tmpdir)
        vc.CONFIG_PATH = Path(self._tmpdir) / "config.json"
        self._orig_default_config = config_module.DEFAULT_CONFIG
        config_module.DEFAULT_CONFIG = config_module.Config()

    def tearDown(self) -> None:
        import kiss.core.vscode_config as vc
        from kiss.core import config as config_module

        config_module.DEFAULT_CONFIG = self._orig_default_config
        vc.CONFIG_DIR = self._orig_dir
        vc.CONFIG_PATH = self._orig_path
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_save_config_persists_headers(self) -> None:
        from kiss.server.commands import _CommandsMixin

        class FakePrinter:
            def __init__(self) -> None:
                self.messages: list[dict[str, Any]] = []

            def broadcast(self, msg: dict[str, Any]) -> None:
                self.messages.append(msg)

        class FakeServer(_CommandsMixin):
            def __init__(self) -> None:
                self.printer = FakePrinter()  # type: ignore[assignment]
                self.work_dir = "/tmp"
                self._state_lock = threading.RLock()
                self._default_model = ""

            def _get_models(self, conn_id: str = "") -> None:
                pass

        server = FakeServer()
        server._cmd_save_config({
            "config": {
                "custom_headers": "X-Test:123\nAuth:Bearer abc",
                "max_budget": 100,
            },
            "apiKeys": {},
        })

        from kiss.core.vscode_config import load_config

        cfg = load_config()
        assert cfg["custom_headers"] == "X-Test:123\nAuth:Bearer abc"

        config_msgs = [
            m for m in server.printer.messages  # type: ignore[union-attr, attr-defined]
            if m.get("type") == "configData"
        ]
        assert len(config_msgs) == 1
        assert config_msgs[0]["config"]["custom_headers"] == "X-Test:123\nAuth:Bearer abc"
