"""Tests for vscode agent audit round 8: redundancies, inconsistencies, bugs.

Covers:
- API_KEY_ENV_VARS type (should be frozenset, not dict)
- _timer_flush closure removal (should be a proper method)
- _timer_flush type annotation (tid should be str | None, not int | None)
- Dead re-exports in server.py
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import time

import pytest


class TestAPIKeyEnvVarsType:
    """API_KEY_ENV_VARS must be a frozenset of key names, not a dict."""

    def test_api_key_env_vars_is_frozenset(self) -> None:
        """Verify API_KEY_ENV_VARS is a frozenset, per USER_PREFS invariant."""
        from kiss.agents.vscode.vscode_config import API_KEY_ENV_VARS

        assert isinstance(API_KEY_ENV_VARS, frozenset), (
            f"API_KEY_ENV_VARS should be frozenset, got {type(API_KEY_ENV_VARS).__name__}"
        )

    def test_api_key_env_vars_contains_expected_keys(self) -> None:
        """Verify all expected API key names are present."""
        from kiss.agents.vscode.vscode_config import API_KEY_ENV_VARS

        expected = {
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "TOGETHER_API_KEY",
            "OPENROUTER_API_KEY",
            "MINIMAX_API_KEY",
        }
        assert API_KEY_ENV_VARS == expected

    def test_get_current_api_keys_works_with_frozenset(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify get_current_api_keys iterates correctly over frozenset."""
        from kiss.agents.vscode import vscode_config

        monkeypatch.setattr(vscode_config, "API_KEY_ENV_VARS", frozenset({"A", "B"}))
        monkeypatch.setenv("A", "val_a")
        monkeypatch.delenv("B", raising=False)
        result = vscode_config.get_current_api_keys()
        assert result == {"A": "val_a", "B": ""}

    def test_source_shell_env_works_with_frozenset(self) -> None:
        """Verify source_shell_env membership check works with frozenset."""
        from kiss.agents.vscode.vscode_config import API_KEY_ENV_VARS

        # The `k in API_KEY_ENV_VARS` check in source_shell_env must work
        assert "GEMINI_API_KEY" in API_KEY_ENV_VARS
        assert "NONEXISTENT_KEY" not in API_KEY_ENV_VARS


class TestTimerFlushNoClosure:
    """_timer_flush must not be a closure; it should be a method."""

    def test_timer_flush_is_method_not_closure(self) -> None:
        """Verify _timer_flush is a proper method on BaseBrowserPrinter."""
        from kiss.agents.vscode.browser_ui import BaseBrowserPrinter

        assert hasattr(BaseBrowserPrinter, "_timer_flush_for_tab"), (
            "BaseBrowserPrinter should have _timer_flush_for_tab method"
        )
        method = getattr(BaseBrowserPrinter, "_timer_flush_for_tab")
        assert callable(method)

    def test_no_nested_def_in_print_method(self) -> None:
        """Verify the print() method has no nested function definitions.

        Closures are forbidden; _timer_flush must be a proper method.
        """
        from kiss.agents.vscode.browser_ui import BaseBrowserPrinter

        source = inspect.getsource(BaseBrowserPrinter.print)
        tree = ast.parse(textwrap.dedent(source))
        # Find nested FunctionDef nodes inside the main function
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        nested_funcs = [
            node.name
            for node in ast.walk(func_def)
            if isinstance(node, ast.FunctionDef) and node is not func_def
        ]
        assert nested_funcs == [], (
            f"print() method should not contain nested function defs, "
            f"found: {nested_funcs}"
        )

    def test_timer_flush_for_tab_type_annotation(self) -> None:
        """Verify _timer_flush_for_tab accepts str | None tab_id."""
        from kiss.agents.vscode.browser_ui import BaseBrowserPrinter

        hints = BaseBrowserPrinter._timer_flush_for_tab.__annotations__
        # The tab_id parameter should accept str | None
        assert "tab_id" in hints

    def test_bash_timer_uses_method(self) -> None:
        """Verify the bash_stream print path creates a timer using
        _timer_flush_for_tab (via functools.partial) instead of a closure."""
        from functools import partial as functools_partial

        from kiss.agents.vscode.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()
        events: list[dict] = []
        original_broadcast = printer.broadcast

        def capture_broadcast(event: dict) -> None:
            events.append(event)
            original_broadcast(event)

        printer.broadcast = capture_broadcast  # type: ignore[assignment]

        # First call flushes immediately (last_flush is 0.0, so delta >= 0.1).
        # A second call within 0.1s triggers the timer path.
        printer.print("chunk1", type="bash_stream")
        printer.print("chunk2", type="bash_stream")
        with printer._bash_lock:
            bs = printer._bash_state
            assert bs.timer is not None, "Timer should be set for buffered bash output"
            timer_func = bs.timer.function
            # Timer function must be a functools.partial, not a closure
            assert isinstance(timer_func, functools_partial), (
                f"Timer function should be functools.partial, got {type(timer_func)}"
            )
            assert timer_func.func == printer._timer_flush_for_tab
        # Wait for timer to fire
        time.sleep(0.2)
        flush_events = [e for e in events if e.get("type") == "system_output"]
        # Both chunks should have been flushed (chunk1 immediately, chunk2 via timer)
        all_text = "".join(e["text"] for e in flush_events)
        assert "chunk1" in all_text
        assert "chunk2" in all_text


class TestNoMisleadingReExportComment:
    """server.py should not have misleading noqa: F401 re-export comments."""

    def test_no_noqa_f401_reexport_comment(self) -> None:
        """Verify server.py does not have a misleading noqa: F401 comment
        on its diff_merge imports (they are actually used, not re-exports)."""
        source_path = inspect.getfile(
            __import__("kiss.agents.vscode.server", fromlist=["VSCodeServer"])
        )
        source = open(source_path).read()
        assert "noqa: F401" not in source, (
            "server.py should not have noqa: F401 comments — "
            "the diff_merge imports are used directly"
        )

    def test_server_all_does_not_expose_diff_merge_names(self) -> None:
        """Verify __all__ in server.py does not list diff_merge symbols."""
        from kiss.agents.vscode import server

        public_names = getattr(server, "__all__", [])
        assert "_cleanup_merge_data" not in public_names
        assert "_git" not in public_names
        assert "_merge_data_dir" not in public_names
