"""Tests that saving work_dir via the config panel updates all runtime state.

Covers:
- apply_config_to_env sets os.environ["KISS_WORKDIR"]
- _cmd_save_config updates self.work_dir on the server
- save_config persists work_dir to config.json
- load_config returns the saved work_dir
- _run_task_inner picks up the new work_dir for subsequent tasks
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest import TestCase

from kiss.agents.vscode.vscode_config import (
    apply_config_to_env,
    load_config,
    save_config,
)


class TestApplyConfigToEnvUpdatesWorkDir(TestCase):
    """apply_config_to_env must set KISS_WORKDIR when work_dir is non-empty."""

    def setUp(self) -> None:
        self._orig = os.environ.get("KISS_WORKDIR")

    def tearDown(self) -> None:
        if self._orig is not None:
            os.environ["KISS_WORKDIR"] = self._orig
        else:
            os.environ.pop("KISS_WORKDIR", None)

    def test_sets_kiss_workdir_env(self) -> None:
        apply_config_to_env({"work_dir": "/tmp/test-wd-123"})
        assert os.environ.get("KISS_WORKDIR") == "/tmp/test-wd-123"

    def test_does_not_set_when_empty(self) -> None:
        os.environ.pop("KISS_WORKDIR", None)
        apply_config_to_env({"work_dir": ""})
        assert "KISS_WORKDIR" not in os.environ

    def test_does_not_set_when_missing(self) -> None:
        os.environ.pop("KISS_WORKDIR", None)
        apply_config_to_env({"max_budget": 50})
        assert "KISS_WORKDIR" not in os.environ

    def test_overwrites_existing_env(self) -> None:
        os.environ["KISS_WORKDIR"] = "/old/path"
        apply_config_to_env({"work_dir": "/new/path"})
        assert os.environ["KISS_WORKDIR"] == "/new/path"


class TestSaveAndLoadWorkDir(TestCase):
    """save_config persists work_dir and load_config retrieves it."""

    def setUp(self) -> None:
        import kiss.agents.vscode.vscode_config as vc

        self._orig_dir = vc.CONFIG_DIR
        self._orig_path = vc.CONFIG_PATH
        self._tmpdir = tempfile.mkdtemp()
        vc.CONFIG_DIR = Path(self._tmpdir)
        vc.CONFIG_PATH = Path(self._tmpdir) / "config.json"

    def tearDown(self) -> None:
        import kiss.agents.vscode.vscode_config as vc

        vc.CONFIG_DIR = self._orig_dir
        vc.CONFIG_PATH = self._orig_path
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_save_and_load_work_dir(self) -> None:
        save_config({"work_dir": "/Users/test/project"})
        cfg = load_config()
        assert cfg["work_dir"] == "/Users/test/project"

    def test_update_work_dir_preserves_other_keys(self) -> None:
        save_config({"work_dir": "/first", "max_budget": 200})
        save_config({"work_dir": "/second"})
        cfg = load_config()
        assert cfg["work_dir"] == "/second"
        assert cfg["max_budget"] == 200

    def test_preserves_non_default_keys(self) -> None:
        import kiss.agents.vscode.vscode_config as vc

        vc.CONFIG_PATH.write_text(json.dumps({"email": "a@b.com"}))
        save_config({"work_dir": "/new"})
        cfg = load_config()
        assert cfg["work_dir"] == "/new"
        assert cfg["email"] == "a@b.com"

    def test_empty_work_dir_saved_as_empty(self) -> None:
        save_config({"work_dir": "/something"})
        save_config({"work_dir": ""})
        cfg = load_config()
        assert cfg["work_dir"] == ""


class TestCmdSaveConfigUpdatesServerWorkDir(TestCase):
    """_cmd_save_config must update self.work_dir on the server."""

    def setUp(self) -> None:
        import kiss.agents.vscode.vscode_config as vc

        self._orig_dir = vc.CONFIG_DIR
        self._orig_path = vc.CONFIG_PATH
        self._tmpdir = tempfile.mkdtemp()
        vc.CONFIG_DIR = Path(self._tmpdir)
        vc.CONFIG_PATH = Path(self._tmpdir) / "config.json"
        self._orig_env = os.environ.get("KISS_WORKDIR")

    def tearDown(self) -> None:
        import kiss.agents.vscode.vscode_config as vc

        vc.CONFIG_DIR = self._orig_dir
        vc.CONFIG_PATH = self._orig_path
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)
        if self._orig_env is not None:
            os.environ["KISS_WORKDIR"] = self._orig_env
        else:
            os.environ.pop("KISS_WORKDIR", None)

    def test_save_config_updates_server_work_dir(self) -> None:
        from kiss.agents.vscode.commands import _CommandsMixin

        class FakePrinter:
            def __init__(self) -> None:
                self.messages: list[dict[str, Any]] = []

            def broadcast(self, msg: dict[str, Any]) -> None:
                self.messages.append(msg)

        class FakeServer(_CommandsMixin):
            def __init__(self) -> None:
                self.printer = FakePrinter()  # type: ignore[assignment]
                self.work_dir = "/old/dir"
                self._state_lock = __import__("threading").Lock()
                self._tab_states: dict[str, Any] = {}
                self._default_model = ""

            def _get_models(self) -> None:
                pass

        server = FakeServer()
        server._cmd_save_config({
            "config": {"work_dir": "/new/project/dir", "max_budget": 100},
            "apiKeys": {},
        })
        assert server.work_dir == "/new/project/dir"
        assert os.environ.get("KISS_WORKDIR") == "/new/project/dir"

    def test_save_config_does_not_update_if_empty(self) -> None:
        from kiss.agents.vscode.commands import _CommandsMixin

        class FakePrinter:
            def __init__(self) -> None:
                self.messages: list[dict[str, Any]] = []

            def broadcast(self, msg: dict[str, Any]) -> None:
                self.messages.append(msg)

        class FakeServer(_CommandsMixin):
            def __init__(self) -> None:
                self.printer = FakePrinter()  # type: ignore[assignment]
                self.work_dir = "/old/dir"
                self._state_lock = __import__("threading").Lock()
                self._tab_states: dict[str, Any] = {}
                self._default_model = ""

            def _get_models(self) -> None:
                pass

        server = FakeServer()
        server._cmd_save_config({
            "config": {"work_dir": "", "max_budget": 100},
            "apiKeys": {},
        })
        assert server.work_dir == "/old/dir"

    def test_saved_config_broadcasts_configdata(self) -> None:
        from kiss.agents.vscode.commands import _CommandsMixin

        class FakePrinter:
            def __init__(self) -> None:
                self.messages: list[dict[str, Any]] = []

            def broadcast(self, msg: dict[str, Any]) -> None:
                self.messages.append(msg)

        class FakeServer(_CommandsMixin):
            def __init__(self) -> None:
                self.printer = FakePrinter()  # type: ignore[assignment]
                self.work_dir = "/old/dir"
                self._state_lock = __import__("threading").Lock()
                self._tab_states: dict[str, Any] = {}
                self._default_model = ""

            def _get_models(self) -> None:
                pass

        server = FakeServer()
        server._cmd_save_config({
            "config": {"work_dir": "/proj", "max_budget": 50},
            "apiKeys": {},
        })
        config_msgs = [
            m for m in server.printer.messages if m.get("type") == "configData"  # type: ignore[attr-defined,union-attr]
        ]
        assert len(config_msgs) == 1
        assert config_msgs[0]["config"]["work_dir"] == "/proj"


class TestRunTaskInnerUsesUpdatedWorkDir(TestCase):
    """After saving config, _run_task_inner should use the new work_dir."""

    def test_task_runner_picks_up_new_work_dir(self) -> None:
        """When cmd has no workDir, _run_task_inner falls back to self.work_dir."""
        # This is a code-level assertion that the fallback logic is correct
        # `work_dir = cmd.get("workDir") or self.work_dir`
        # If self.work_dir was updated by _cmd_save_config, the fallback works.
        from pathlib import Path

        task_runner_path = (
            Path(__file__).resolve().parents[3]
            / "agents"
            / "vscode"
            / "task_runner.py"
        )
        source = task_runner_path.read_text()
        assert 'cmd.get("workDir") or self.work_dir' in source
