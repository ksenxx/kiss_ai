# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests that custom HTTP headers can be configured via the settings panel.

The settings panel has a textarea (after the custom endpoint field) where
users can enter custom HTTP headers in ``Key:Value`` format, one per line.
These headers flow through to the model via ``model_config["extra_headers"]``.
"""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

_VSCODE_DIR = Path(__file__).resolve().parents[3] / "agents" / "vscode"


# ---------------------------------------------------------------------------
# 1. HTML presence — textarea for headers exists in both templates
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# 2. JavaScript — populateConfigForm and collectConfigForm handle headers
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# 3. Config persistence — custom_headers is saved and loaded
# ---------------------------------------------------------------------------


class TestConfigPersistence(unittest.TestCase):
    """custom_headers is persisted in config.json."""

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

    def test_custom_headers_in_defaults(self) -> None:
        from kiss.agents.vscode.vscode_config import DEFAULTS

        assert "custom_headers" in DEFAULTS

    def test_save_and_load_custom_headers(self) -> None:
        from kiss.agents.vscode.vscode_config import load_config, save_config

        save_config({"custom_headers": "X-Custom:value1\nAuthorization:Bearer tok"})
        cfg = load_config()
        assert cfg["custom_headers"] == "X-Custom:value1\nAuthorization:Bearer tok"

    def test_empty_headers_by_default(self) -> None:
        from kiss.agents.vscode.vscode_config import load_config

        cfg = load_config()
        assert cfg["custom_headers"] == ""

    def test_preserves_other_keys(self) -> None:
        from kiss.agents.vscode.vscode_config import load_config, save_config

        save_config({"custom_headers": "X-Foo:bar", "max_budget": 200})
        save_config({"custom_headers": "X-Baz:qux"})
        cfg = load_config()
        assert cfg["custom_headers"] == "X-Baz:qux"
        assert cfg["max_budget"] == 200


# ---------------------------------------------------------------------------
# 4. get_custom_model_entry includes extra_headers
# ---------------------------------------------------------------------------


class TestCustomModelEntryIncludesHeaders(unittest.TestCase):
    """get_custom_model_entry includes extra_headers from custom_headers config."""

    def test_no_headers_when_empty(self) -> None:
        from kiss.agents.vscode.vscode_config import get_custom_model_entry

        entry = get_custom_model_entry({
            "custom_endpoint": "http://localhost:8080/v1",
            "custom_headers": "",
        })
        assert entry is not None
        assert "extra_headers" not in entry or entry.get("extra_headers") == {}

    def test_headers_parsed_into_dict(self) -> None:
        from kiss.agents.vscode.vscode_config import get_custom_model_entry

        entry = get_custom_model_entry({
            "custom_endpoint": "http://localhost:8080/v1",
            "custom_headers": "X-Custom:value1\nAuthorization:Bearer tok",
        })
        assert entry is not None
        assert entry["extra_headers"] == {
            "X-Custom": "value1",
            "Authorization": "Bearer tok",
        }

    def test_no_headers_when_no_endpoint(self) -> None:
        from kiss.agents.vscode.vscode_config import get_custom_model_entry

        entry = get_custom_model_entry({
            "custom_endpoint": "",
            "custom_headers": "X-Custom:value1",
        })
        assert entry is None

    def test_malformed_header_lines_skipped(self) -> None:
        from kiss.agents.vscode.vscode_config import get_custom_model_entry

        entry = get_custom_model_entry({
            "custom_endpoint": "http://localhost:8080/v1",
            "custom_headers": "X-Good:value\nbadline\n\nX-Also:good",
        })
        assert entry is not None
        assert entry["extra_headers"] == {
            "X-Good": "value",
            "X-Also": "good",
        }


# ---------------------------------------------------------------------------
# 5. task_runner builds model_config with extra_headers from config
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# 6. CLI headers flow — _build_run_kwargs puts headers into model_config
# ---------------------------------------------------------------------------


class TestCLIHeadersFlow(unittest.TestCase):
    """CLI --header option flows into model_config["extra_headers"]."""

    def test_build_run_kwargs_with_headers(self) -> None:
        import argparse

        from kiss.agents.sorcar.cli_helpers import _build_run_kwargs

        args = argparse.Namespace(
            model_name="gpt-4o",
            endpoint="http://localhost:8080/v1",
            header=["X-Custom:value1", "Authorization:Bearer tok"],
            max_budget=100,
            work_dir=None,
            verbose=True,
            no_web=False,
            parallel=False,
            task="test task",
            file=None,
        )
        kwargs = _build_run_kwargs(args)
        assert kwargs["model_config"]["base_url"] == "http://localhost:8080/v1"
        assert kwargs["model_config"]["extra_headers"] == {
            "X-Custom": "value1",
            "Authorization": "Bearer tok",
        }

    def test_build_run_kwargs_without_headers(self) -> None:
        import argparse

        from kiss.agents.sorcar.cli_helpers import _build_run_kwargs

        args = argparse.Namespace(
            model_name="gpt-4o",
            endpoint=None,
            header=None,
            max_budget=100,
            work_dir=None,
            verbose=True,
            no_web=False,
            parallel=False,
            task="test task",
            file=None,
        )
        kwargs = _build_run_kwargs(args)
        assert "extra_headers" not in kwargs.get("model_config", {})


# ---------------------------------------------------------------------------
# 7. _cmd_save_config integration — headers flow through save/load
# ---------------------------------------------------------------------------


class TestCmdSaveConfigHandlesHeaders(unittest.TestCase):
    """_cmd_save_config persists custom_headers and they appear in configData."""

    def setUp(self) -> None:
        import kiss.agents.vscode.vscode_config as vc
        from kiss.core import config as config_module

        self._orig_dir = vc.CONFIG_DIR
        self._orig_path = vc.CONFIG_PATH
        self._tmpdir = tempfile.mkdtemp()
        vc.CONFIG_DIR = Path(self._tmpdir)
        vc.CONFIG_PATH = Path(self._tmpdir) / "config.json"
        # ``_cmd_save_config`` calls ``apply_config_to_env`` which
        # mutates the process-global ``DEFAULT_CONFIG`` (for example
        # ``max_budget``).  Swap in a scratch Config so the mutation
        # cannot leak into later tests, and restore the original
        # object in tearDown.
        self._orig_default_config = config_module.DEFAULT_CONFIG
        config_module.DEFAULT_CONFIG = config_module.Config()

    def tearDown(self) -> None:
        import kiss.agents.vscode.vscode_config as vc
        from kiss.core import config as config_module

        config_module.DEFAULT_CONFIG = self._orig_default_config
        vc.CONFIG_DIR = self._orig_dir
        vc.CONFIG_PATH = self._orig_path
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_save_config_persists_headers(self) -> None:
        from kiss.agents.vscode.commands import _CommandsMixin

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
                self._running_agent_states: dict[str, Any] = {}
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

        from kiss.agents.vscode.vscode_config import load_config

        cfg = load_config()
        assert cfg["custom_headers"] == "X-Test:123\nAuth:Bearer abc"

        # Check that configData broadcast includes custom_headers
        config_msgs = [
            m for m in server.printer.messages  # type: ignore[union-attr, attr-defined]
            if m.get("type") == "configData"
        ]
        assert len(config_msgs) == 1
        assert config_msgs[0]["config"]["custom_headers"] == "X-Test:123\nAuth:Bearer abc"
