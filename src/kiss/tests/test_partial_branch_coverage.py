"""Targeted integration tests to cover partial branches across the codebase.

No mocks, test doubles, or fakes.
"""

from __future__ import annotations

import queue
from argparse import ArgumentParser
from typing import Any

import pytest
from pydantic import BaseModel

# === core/utils.py partial branches ===


class TestEscapeInvalidTemplateFieldNames:
    """Cover branches for conversion and format_spec in _escape_fragment."""

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

    def test_module_to_path_package(self) -> None:
        """Module path resolution for a package covers line 261 is_dir check."""
        from kiss.scripts.generate_api_docs import _module_to_path

        # kiss.core is a package directory
        path = _module_to_path("kiss.core")
        # The function resolves to __init__.py, but internal logic checks is_dir
        assert path.exists()


# === scripts/redundancy_analyzer.py partial branches ===
