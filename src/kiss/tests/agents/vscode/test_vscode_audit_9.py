"""Cross-language consistency tests for the VS Code agent.

Bug 1: taskDeleted message type missing from ToWebviewMessageBody in types.ts
Bug 2: DependencyInstaller.ts getDefaultModel() has stale model names and wrong
       priority order vs Python get_default_model()
Bug 3: main.js hardcoded default selectedModel is stale ('claude-opus-4-6'
       instead of 'claude-opus-4-7')
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
        assert "case 'taskDeleted'" in self._main_js

    def test_task_deleted_in_to_webview_message(self) -> None:
        """taskDeleted must appear in ToWebviewMessageBody type union."""
        # Extract the ToWebviewMessageBody section
        m = re.search(r"type ToWebviewMessageBody\s*=", self._types_ts)
        assert m, "ToWebviewMessageBody type not found"
        body_section = self._types_ts[m.start() :]
        assert "'taskDeleted'" in body_section, (
            "taskDeleted is sent by server.py and handled by main.js "
            "but missing from ToWebviewMessageBody in types.ts"
        )


class TestGetDefaultModelConsistency(unittest.TestCase):
    """Bug 2: DependencyInstaller.ts getDefaultModel() must match Python."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls._ts = (_VSCODE_DIR / "src" / "DependencyInstaller.ts").read_text()

    def _extract_ts_defaults(self) -> dict[str, str]:
        """Parse getDefaultModel() from TypeScript."""
        m = re.search(
            r"export function getDefaultModel\(\).*?\{(.*?)^\}",
            self._ts,
            re.DOTALL | re.MULTILINE,
        )
        assert m, "getDefaultModel not found"
        body = m.group(1)
        lines = body.strip().splitlines()
        result: dict[str, str] = {}
        for i, line in enumerate(lines):
            env_match = re.search(r"process\.env\.(\w+)", line)
            if not env_match:
                continue
            # Return may be on same line or next line
            ret_match = re.search(r"return\s+'([^']+)'", line)
            if not ret_match and i + 1 < len(lines):
                ret_match = re.search(r"return\s+'([^']+)'", lines[i + 1])
            if ret_match:
                result[env_match.group(1)] = ret_match.group(1)
        return result

    def _get_python_defaults(self) -> dict[str, str]:
        """Get defaults from Python get_default_model()."""
        import inspect

        from kiss.core.models.model_info import get_default_model  # noqa: F811

        src = inspect.getsource(get_default_model)
        lines = src.splitlines()
        result: dict[str, str] = {}
        # The pattern is: "if keys.X:" on one line, "return Y" on the next
        for i, line in enumerate(lines):
            key_match = re.search(r"keys\.(\w+)", line)
            if key_match and i + 1 < len(lines):
                ret_match = re.search(r'return\s+"([^"]+)"', lines[i + 1])
                if ret_match:
                    result[key_match.group(1)] = ret_match.group(1)
        return result

    def test_anthropic_model_matches(self) -> None:
        ts = self._extract_ts_defaults()
        py = self._get_python_defaults()
        assert ts["ANTHROPIC_API_KEY"] == py["ANTHROPIC_API_KEY"], (
            f"Anthropic model mismatch: TS={ts['ANTHROPIC_API_KEY']}, "
            f"Py={py['ANTHROPIC_API_KEY']}"
        )

    def test_openai_model_matches(self) -> None:
        ts = self._extract_ts_defaults()
        py = self._get_python_defaults()
        assert ts["OPENAI_API_KEY"] == py["OPENAI_API_KEY"], (
            f"OpenAI model mismatch: TS={ts['OPENAI_API_KEY']}, "
            f"Py={py['OPENAI_API_KEY']}"
        )

    def test_openrouter_model_matches(self) -> None:
        ts = self._extract_ts_defaults()
        py = self._get_python_defaults()
        assert ts["OPENROUTER_API_KEY"] == py["OPENROUTER_API_KEY"], (
            f"OpenRouter model mismatch: TS={ts['OPENROUTER_API_KEY']}, "
            f"Py={py['OPENROUTER_API_KEY']}"
        )

    def test_priority_order_matches_python(self) -> None:
        """TS priority must match Python: Anthropic > OpenAI > Gemini > OpenRouter > Together."""
        m = re.search(
            r"export function getDefaultModel\(\).*?\{(.*?)^\}",
            self._ts,
            re.DOTALL | re.MULTILINE,
        )
        assert m
        body = m.group(1)
        keys_in_order = re.findall(r"process\.env\.(\w+)", body)
        expected_order = [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
            "TOGETHER_API_KEY",
        ]
        assert keys_in_order == expected_order, (
            f"Priority order mismatch: TS={keys_in_order}, expected={expected_order}"
        )

    def test_docstring_matches_priority(self) -> None:
        """The docstring must reflect the correct priority order."""
        m = re.search(
            r"/\*\*[^*]*getDefaultModel[^/]*/",
            self._ts,
            re.DOTALL,
        )
        if not m:
            # Try finding comment just before the function
            m = re.search(
                r"/\*\*(.*?)\*/\s*export function getDefaultModel",
                self._ts,
                re.DOTALL,
            )
        assert m, "getDefaultModel docstring not found"
        doc = m.group(0)
        assert "Anthropic > OpenAI > Gemini > OpenRouter > Together" in doc, (
            f"Docstring has wrong priority order: {doc}"
        )


class TestMainJsDefaultModel(unittest.TestCase):
    """Bug 3: main.js default selectedModel must match Python canonical default."""

    _main_js: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls._main_js = (_VSCODE_DIR / "media" / "main.js").read_text()

    def _get_python_anthropic_default(self) -> str:
        import inspect

        from kiss.core.models.model_info import get_default_model

        src = inspect.getsource(get_default_model)
        # First return value (Anthropic, highest priority)
        m = re.search(r'return\s+"([^"]+)"', src)
        assert m, "No default model found in Python"
        return m.group(1)

    def test_initial_selected_model_matches_python(self) -> None:
        """let selectedModel = '...' must use the Python canonical default."""
        m = re.search(r"let selectedModel\s*=\s*'([^']+)'", self._main_js)
        assert m, "selectedModel declaration not found"
        js_default = m.group(1)
        py_default = self._get_python_anthropic_default()
        assert js_default == py_default, (
            f"main.js default selectedModel '{js_default}' != "
            f"Python default '{py_default}'"
        )

    def test_restore_tab_fallback_matches_python(self) -> None:
        """restoreTab fallback model must match Python canonical default."""
        py_default = self._get_python_anthropic_default()
        # Look for: tab.selectedModel || 'claude-opus-4-X'
        fallbacks = re.findall(
            r"tab\.selectedModel\s*\|\|\s*'([^']+)'", self._main_js
        )
        for fb in fallbacks:
            assert fb == py_default, (
                f"restoreTab fallback '{fb}' != Python default '{py_default}'"
            )


if __name__ == "__main__":
    unittest.main()
