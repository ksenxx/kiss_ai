"""Cross-language consistency tests for the VS Code agent.

Bug 1: taskDeleted message type missing from ToWebviewMessageBody in types.ts
Bug 2: DependencyInstaller.ts getDefaultModel() has stale model names and wrong
       priority order vs Python get_default_model()
Bug 3: main.js hardcoded default selectedModel is stale ('claude-opus-4-6'
       instead of 'claude-opus-4-7')
Bug 2: DependencyInstaller.ts getDefaultModel() has stale model names and wrong
       priority order vs Python get_default_model()
Bug 3: main.js hardcoded default selectedModel is stale ('claude-opus-4-6'
       instead of 'claude-opus-4-7')
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

    def test_task_deleted_in_to_webview_message(self) -> None:
        """taskDeleted must appear in ToWebviewMessageBody type union."""
        # Extract the ToWebviewMessageBody section
        # Extract the ToWebviewMessageBody section
        m = re.search(r"type ToWebviewMessageBody\s*=", self._types_ts)
        assert m, "ToWebviewMessageBody type not found"
        body_section = self._types_ts[m.start() :]
        assert "'taskDeleted'" in body_section, (
        )


class TestGetDefaultModelConsistency(unittest.TestCase):
    """Bug 2: DependencyInstaller.ts getDefaultModel() must match Python."""
class TestGetDefaultModelConsistency(unittest.TestCase):
    """Bug 2: DependencyInstaller.ts getDefaultModel() must match Python."""
class TestGetDefaultModelCallsPython(unittest.TestCase):
    """Bug 2: DependencyInstaller.ts getDefaultModel() must call Python,
    not hardcode model names."""

    @classmethod
        cls._ts = (_VSCODE_DIR / "src" / "DependencyInstaller.ts").read_text()

        """Parse getDefaultModel() from TypeScript."""
    def _extract_ts_defaults(self) -> dict[str, str]:
        """Parse getDefaultModel() from TypeScript."""
    def test_calls_python_get_default_model(self) -> None:
                ret_match = re.search(r'return\s+"([^"]+)"', lines[i + 1])
                if ret_match:
                    result[key_match.group(1)] = ret_match.group(1)
        return result
            f"Py={py['ANTHROPIC_API_KEY']}"
        lines = body.strip().splitlines()
        result: dict[str, str] = {}
        for i, line in enumerate(lines):
            env_match = re.search(r"process\.env\.(\w+)", line)
            if not env_match:
                continue

    def _get_python_defaults(self) -> dict[str, str]:
        """Get defaults from Python get_default_model()."""
        import inspect

        from kiss.core.models.model_info import get_default_model  # noqa: F811

        result: dict[str, str] = {}
        # The pattern is: "if keys.X:" on one line, "return Y" on the next
            key_match = re.search(r"keys\.(\w+)", line)
            if key_match and i + 1 < len(lines):
                if ret_match:
                    result[key_match.group(1)] = ret_match.group(1)
        return result
        ts = self._extract_ts_defaults()
        py = self._get_python_defaults()
            f"Py={py['OPENAI_API_KEY']}"
    def test_openai_model_matches(self) -> None:
        py = self._get_python_defaults()
        assert ts["OPENAI_API_KEY"] == py["OPENAI_API_KEY"], (
            f"OpenAI model mismatch: TS={ts['OPENAI_API_KEY']}, "
            f"Py={py['OPENAI_API_KEY']}"

        assert ts["OPENROUTER_API_KEY"] == py["OPENROUTER_API_KEY"], (
            f"OpenRouter model mismatch: TS={ts['OPENROUTER_API_KEY']}, "
            f"Py={py['OPENROUTER_API_KEY']}"

    def test_openrouter_model_matches(self) -> None:
        ts = self._extract_ts_defaults()
        py = self._get_python_defaults()
            f"Py={py['OPENROUTER_API_KEY']}"
        assert m
        body = m.group(1)
        assert "process.env" not in body, (
            "getDefaultModel() should not check process.env — "
            "Python's get_default_model() handles API key detection"
        )

    def test_priority_order_matches_python(self) -> None:
        keys_in_order = re.findall(r"process\.env\.(\w+)", body)
        expected_order = [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            f"Priority order mismatch: TS={keys_in_order}, expected={expected_order}"
        )
        assert "findUvPath" in m.group(1)

            r"/\*\*[^*]*getDefaultModel[^/]*/",
            r"/\*\*(.*?)\*/\s*export function getDefaultModel",
            self._ts,
        )
        assert m, "getDefaultModel docstring not found"
        doc = m.group(1)
            f"Docstring should not mention a hardcoded model name: {doc}"
        )


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

    def test_restore_tab_fallback_is_empty(self) -> None:
        """restoreTab fallback should be empty, not a hardcoded model name."""
            self._main_js,
        ), "selectedModel should be initialized from DOM model-name element"
if __name__ == "__main__":
    unittest.main()