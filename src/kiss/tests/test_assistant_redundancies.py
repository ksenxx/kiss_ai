"""Tests for redundancy fixes in assistant.py."""

from __future__ import annotations


class TestNoSyntaxErrors:
    """Verify the module imports without syntax errors."""

    def test_module_imports(self) -> None:
        import kiss.agents.sorcar.sorcar as mod

        assert hasattr(mod, "run_chatbot")
        assert hasattr(mod, "_read_active_file")
        assert hasattr(mod, "_clean_llm_output")
        assert hasattr(mod, "_model_vendor_order")

    def test_no_get_most_expensive_model_import(self) -> None:
        """Verify removed redundant import doesn't exist in module namespace."""
        import kiss.agents.sorcar.sorcar as mod

        # get_most_expensive_model was removed from imports as it's no longer used
        assert not hasattr(mod, "get_most_expensive_model")
