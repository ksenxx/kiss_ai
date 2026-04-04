"""Targeted integration tests to cover partial branches across the codebase.

No mocks, test doubles, or fakes.
"""

from __future__ import annotations

import queue
import threading
from argparse import ArgumentParser
from typing import Any

import pytest
from pydantic import BaseModel

# === core/utils.py partial branches ===


class TestGetConfigValue:
    """Cover branches in get_config_value where value/config_value is not None."""

    def test_explicit_value_returned(self) -> None:
        """When value is not None, it is returned immediately (line 37->38)."""
        from kiss.core.utils import get_config_value

        class Cfg:
            x: int = 99

        result = get_config_value(42, Cfg(), "x", default=0)
        assert result == 42

    def test_config_value_used(self) -> None:
        """When value is None but config has the attr, config value is used (line 40->41)."""
        from kiss.core.utils import get_config_value

        class Cfg:
            x: int = 99

        result = get_config_value(None, Cfg(), "x", default=0)
        assert result == 99


class TestEscapeInvalidTemplateFieldNames:
    """Cover branches for conversion and format_spec in _escape_fragment."""

    def test_with_conversion(self) -> None:
        """Template with !r conversion triggers the conversion branch (line 86->87)."""
        from kiss.core.utils import escape_invalid_template_field_names

        result = escape_invalid_template_field_names("{name!r}", {"name"})
        assert result == "{name!r}"

    def test_with_format_spec(self) -> None:
        """Template with :10d format spec triggers format_spec branch (line 88->89)."""
        from kiss.core.utils import escape_invalid_template_field_names

        result = escape_invalid_template_field_names("{count:10d}", {"count"})
        assert result == "{count:10d}"

    def test_invalid_field_with_conversion_and_spec(self) -> None:
        """Invalid field with conversion+format_spec enters the escape branch (line 92->93)."""
        from kiss.core.utils import escape_invalid_template_field_names

        result = escape_invalid_template_field_names("{bad!r:>10}", set())
        # The invalid field should be double-braced (escaped)
        assert "{{bad!r:>10}}" in result


# === core/config_builder.py partial branches ===


class TestConfigBuilderBoolNoDash:
    """Cover the else branch of 'if arg_name_dashes != arg_name' for bool fields."""

    def test_bool_field_without_underscores(self) -> None:
        """A bool field with no underscores makes arg_name_dashes == arg_name."""
        from kiss.core.config_builder import _add_model_arguments

        class TestModel(BaseModel):
            verbose: bool = False

        parser = ArgumentParser()
        _add_model_arguments(parser, TestModel)
        args = parser.parse_args(["--no-verbose"])
        assert args.verbose is False


# === core/models/__init__.py partial branches ===


class TestLazyImportNotInMap:
    """Cover the else branch where name is NOT in _LAZY_IMPORTS."""

    def test_raises_attribute_error(self) -> None:
        """Accessing a name NOT in _LAZY_IMPORTS raises AttributeError."""
        import kiss.core.models as models_mod

        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(models_mod, "NonExistentModel")


# === channels/_backend_utils.py partial branches ===


class TestWaitForMatchingMessage:
    """Cover branches in wait_for_matching_message."""

    def test_stop_event_cancels_immediately(self) -> None:
        """stop_event.is_set() returns True before polling (line 45->46)."""
        from kiss.channels._backend_utils import wait_for_matching_message

        stop = threading.Event()
        stop.set()
        result = wait_for_matching_message(
            poll=lambda: [],
            matches=lambda m: True,
            extract_text=lambda m: m["text"],
            timeout_seconds=10,
            stop_event=stop,
            poll_interval=0.1,
        )
        assert result is None

    def test_stop_event_during_wait(self) -> None:
        """stop_event fires during the sleep wait (line 55 -> True)."""
        from kiss.channels._backend_utils import wait_for_matching_message

        stop = threading.Event()
        threading.Timer(0.05, stop.set).start()
        result = wait_for_matching_message(
            poll=lambda: [],
            matches=lambda m: True,
            extract_text=lambda m: m["text"],
            timeout_seconds=10,
            stop_event=stop,
            poll_interval=5.0,
        )
        assert result is None

    def test_no_stop_event_timeout(self) -> None:
        """No stop_event, times out via time.sleep (else branch on line 54)."""
        from kiss.channels._backend_utils import wait_for_matching_message

        result = wait_for_matching_message(
            poll=lambda: [],
            matches=lambda m: True,
            extract_text=lambda m: m["text"],
            timeout_seconds=0.05,
            stop_event=None,
            poll_interval=0.01,
        )
        assert result is None

    def test_stop_event_wait_loops_then_timeout(self) -> None:
        """stop_event.wait returns False (loop back line 55->44), then times out."""
        from kiss.channels._backend_utils import wait_for_matching_message

        stop = threading.Event()
        # stop never set: wait() returns False each poll, loops back to top
        result = wait_for_matching_message(
            poll=lambda: [],
            matches=lambda m: True,
            extract_text=lambda m: m["text"],
            timeout_seconds=0.05,
            stop_event=stop,
            poll_interval=0.01,
        )
        assert result is None


class TestDrainQueueMessages:
    """Cover branches in drain_queue_messages."""

    def test_drain_with_filter(self) -> None:
        """Filter keeps some messages, rejects others (line 83 keep branch)."""
        from kiss.channels._backend_utils import drain_queue_messages

        q: queue.Queue[dict[str, Any]] = queue.Queue()
        q.put({"id": 1, "good": True})
        q.put({"id": 2, "good": False})
        q.put({"id": 3, "good": True})
        result = drain_queue_messages(q, limit=10, keep=lambda m: m["good"])
        assert len(result) == 2

    def test_drain_empty_queue(self) -> None:
        """Empty queue hits the except Empty break (line 78 loop exit)."""
        from kiss.channels._backend_utils import drain_queue_messages

        q: queue.Queue[dict[str, Any]] = queue.Queue()
        result = drain_queue_messages(q, limit=10)
        assert result == []

    def test_drain_hits_limit(self) -> None:
        """Queue has more items than limit, while condition exits (line 78->85)."""
        from kiss.channels._backend_utils import drain_queue_messages

        q: queue.Queue[dict[str, Any]] = queue.Queue()
        for i in range(5):
            q.put({"id": i})
        result = drain_queue_messages(q, limit=3)
        assert len(result) == 3


# === scripts/generate_api_docs.py partial branches ===


class TestGenerateApiDocsBranches:
    """Cover partial branches in generate_api_docs.py."""

    def test_parse_multiline_description(self) -> None:
        """Multi-line summary description covers line 139 'if rest'."""
        from kiss.scripts.generate_api_docs import _parse_google_docstring

        doc = "Summary.\n\n    More detail.\n\n    Returns:\n        int: result\n"
        result = _parse_google_docstring(doc)
        assert "More detail" in result.summary

    def test_param_with_type_in_parens(self) -> None:
        """Param with '(str)' covers line 168 elif '(' in param_part."""
        from kiss.scripts.generate_api_docs import _parse_google_docstring

        doc = "Summary.\n\n    Args:\n        name (str): The name.\n"
        result = _parse_google_docstring(doc)
        assert any(n == "name" for n, _ in result.args)

    def test_multiline_param_desc(self) -> None:
        """Continuation line covers line 173 elif current_arg_name."""
        from kiss.scripts.generate_api_docs import _parse_google_docstring

        doc = "Summary.\n\n    Args:\n        x: first line\n            second line.\n"
        result = _parse_google_docstring(doc)
        assert any("second" in d for _, d in result.args)

    def test_parse_all_list_ast(self) -> None:
        """__all__ detection via AST covers line 239."""
        import ast

        from kiss.scripts.generate_api_docs import _parse_all_list

        tree = ast.parse('__all__ = ["foo", "bar"]\n')
        names = _parse_all_list(tree)
        assert names is not None and "foo" in names

    def test_has_decorator(self) -> None:
        """Deprecated decorator detection covers line 326."""
        import ast

        from kiss.scripts.generate_api_docs import _has_decorator

        tree = ast.parse("@deprecated\ndef foo(): pass\n")
        func = tree.body[0]
        assert isinstance(func, ast.FunctionDef)
        assert _has_decorator(func, "deprecated") is True

    def test_module_to_path_package(self) -> None:
        """Module path resolution for a package covers line 261 is_dir check."""
        from kiss.scripts.generate_api_docs import _module_to_path

        # kiss.core is a package directory
        path = _module_to_path("kiss.core")
        # The function resolves to __init__.py, but internal logic checks is_dir
        assert path.exists()


# === scripts/redundancy_analyzer.py partial branches ===


class TestRedundancyAnalyzerBranches:
    """Cover partial branches in redundancy_analyzer.py with real data."""

    def test_analyze_redundancy_returns_list(self) -> None:
        """analyze_redundancy returns a list even without context data."""
        from kiss.scripts.redundancy_analyzer import analyze_redundancy

        result = analyze_redundancy(".coverage")
        assert isinstance(result, list)
