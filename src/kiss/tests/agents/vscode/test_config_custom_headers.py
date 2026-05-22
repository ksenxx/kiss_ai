"""Tests that custom HTTP headers can be configured via the settings panel.

The settings panel has a textarea (after the custom endpoint field) where
users can enter custom HTTP headers in ``Key:Value`` format, one per line.
These headers flow through to the model via ``model_config["extra_headers"]``.
"""

from __future__ import annotations

import re
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

_VSCODE_DIR = Path(__file__).resolve().parents[3] / "agents" / "vscode"


# ---------------------------------------------------------------------------
# 1. HTML presence — textarea for headers exists in both templates
# ---------------------------------------------------------------------------


class TestHeadersTextareaInHTML(unittest.TestCase):
    """A textarea with id cfg-custom-headers exists in both HTML templates."""


    def test_web_server_has_headers_textarea(self) -> None:
        py = (_VSCODE_DIR / "web_server.py").read_text()
        assert "cfg-custom-headers" in py


    def test_headers_textarea_is_after_endpoint_web(self) -> None:
        """The headers field appears after the custom endpoint field in web_server."""
        py = (_VSCODE_DIR / "web_server.py").read_text()
        endpoint_pos = py.index("cfg-custom-endpoint")
        headers_pos = py.index("cfg-custom-headers")
        assert headers_pos > endpoint_pos


# ---------------------------------------------------------------------------
# 2. JavaScript — populateConfigForm and collectConfigForm handle headers
# ---------------------------------------------------------------------------


class TestMainJSHandlesHeaders(unittest.TestCase):
    """main.js populates and collects the custom_headers field."""

    _js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls._js = (_VSCODE_DIR / "media" / "main.js").read_text()

    def test_populate_sets_headers(self) -> None:
        """populateConfigForm reads cfg.custom_headers into the textarea."""
        assert "custom_headers" in self._js
        assert "cfg-custom-headers" in self._js

    def test_collect_reads_headers(self) -> None:
        """collectConfigForm reads the textarea value."""
        # Find the collectConfigForm function
        m = re.search(r"function collectConfigForm\(\)", self._js)
        assert m
        # Extract the function body
        start = m.start()
        body_end = self._js.index("\n  }", start) + 4
        body = self._js[start:body_end]
        assert "custom_headers" in body


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


class TestTaskRunnerBuildsModelConfig(unittest.TestCase):
    """task_runner passes model_config with extra_headers to agent.run()."""

    def test_task_runner_reads_custom_headers_from_config(self) -> None:
        """The task_runner source builds model_config from config."""
        src = (_VSCODE_DIR / "task_runner.py").read_text()
        assert "build_model_config" in src
        assert "model_config" in src

    def test_task_runner_passes_model_config_to_agent_run(self) -> None:
        """agent.run() call includes model_config kwarg."""
        src = (_VSCODE_DIR / "task_runner.py").read_text()
        assert "model_config=" in src


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

            def _get_models(self) -> None:
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
