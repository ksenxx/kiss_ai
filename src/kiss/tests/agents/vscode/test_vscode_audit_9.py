"""Cross-language consistency tests for the VS Code agent.

Bug 1: taskDeleted message type missing from ToWebviewMessageBody in types.ts
Bug 2: DependencyInstaller.ts getDefaultModel() had hardcoded model names —
       now calls Python's get_default_model() at runtime.
Bug 3: main.js had hardcoded default selectedModel — now reads from DOM
       (injected by the backend template).
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

_VSCODE_DIR = Path(__file__).resolve().parents[3] / "agents" / "vscode"


class TestTaskDeletedInTypesTs(unittest.TestCase):
    """Bug 1: taskDeleted must be in ToWebviewMessageBody."""

    _types_ts: str = ""
    _main_js: str = ""
    _server_py: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls._types_ts = (_VSCODE_DIR / "src" / "types.ts").read_text()
        cls._main_js = (_VSCODE_DIR / "media" / "main.js").read_text()
        cls._server_py = (_VSCODE_DIR / "server.py").read_text()

    def test_server_sends_task_deleted(self) -> None:
        """server.py sends taskDeleted events."""
        assert '"taskDeleted"' in self._server_py or "'taskDeleted'" in self._server_py

    def test_main_js_handles_task_deleted(self) -> None:
        """main.js has a switch case for taskDeleted."""
        assert "taskDeleted" in self._main_js

    def test_task_deleted_in_to_webview_message(self) -> None:
        """taskDeleted must appear in ToWebviewMessageBody type union."""
        m = re.search(r"type ToWebviewMessageBody\s*=", self._types_ts)
        assert m, "ToWebviewMessageBody type not found"
        body_section = self._types_ts[m.start() :]
        assert "'taskDeleted'" in body_section or '"taskDeleted"' in body_section, (
            "taskDeleted not found in ToWebviewMessageBody"
        )


class TestGetDefaultModelCallsPython(unittest.TestCase):
    """Bug 2: DependencyInstaller.ts getDefaultModel() must call Python,
    not hardcode model names."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls._ts = (_VSCODE_DIR / "src" / "DependencyInstaller.ts").read_text()

    def test_calls_python_get_default_model(self) -> None:
        """getDefaultModel() must invoke Python's get_default_model()."""
        m = re.search(
            r"export function getDefaultModel\(\).*?\{(.*?)\n\}",
            self._ts,
            re.DOTALL,
        )
        assert m, "getDefaultModel() not found in DependencyInstaller.ts"
        body = m.group(1)
        assert "get_default_model" in body, (
            "getDefaultModel() must call Python's get_default_model()"
        )
        assert "process.env" not in body, (
            "getDefaultModel() should not check process.env — "
            "Python's get_default_model() handles API key detection"
        )

    def test_uses_find_uv_path(self) -> None:
        """getDefaultModel() must use findUvPath()."""
        m = re.search(
            r"export function getDefaultModel\(\).*?\{(.*?)\n\}",
            self._ts,
            re.DOTALL,
        )
        assert m
        assert "findUvPath" in m.group(1)


class TestMainJsNoHardcodedModel(unittest.TestCase):
    """Bug 3: main.js must not hardcode default model — reads from DOM."""

    _main_js: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls._main_js = (_VSCODE_DIR / "media" / "main.js").read_text()

    def test_initial_selected_model_is_empty(self) -> None:
        """let selectedModel should be empty string, not a model name."""
        m = re.search(r"let selectedModel\s*=\s*'([^']*)'", self._main_js)
        assert m, "selectedModel declaration not found"
        assert m.group(1) == "", (
            f"selectedModel initialized to '{m.group(1)}' instead of empty string"
        )

    def test_restore_tab_fallback_is_empty(self) -> None:
        """restoreTab fallback should be empty, not a hardcoded model name."""
        # Should not have any hardcoded model name as a fallback
        assert re.search(
            r"modelName.*textContent.*selectedModel|selectedModel.*modelName.*textContent",
            self._main_js,
        ), "selectedModel should be initialized from DOM model-name element"


if __name__ == "__main__":
    unittest.main()
