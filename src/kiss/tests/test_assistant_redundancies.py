"""Tests for redundancy fixes in assistant.py."""

from __future__ import annotations

import json
import os
import tempfile

from kiss.agents.sorcar.sorcar import (
    _clean_llm_output,
    _model_vendor_order,
    _read_active_file,
)


class TestReadActiveFile:
    def test_returns_empty_when_no_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            assert _read_active_file(td) == ""

    def test_returns_path_when_valid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = os.path.join(td, "test.py")
            with open(target, "w") as f:
                f.write("hello")
            af_path = os.path.join(td, "active-file.json")
            with open(af_path, "w") as f:
                json.dump({"path": target}, f)
            assert _read_active_file(td) == target

    def test_returns_empty_when_file_does_not_exist(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            af_path = os.path.join(td, "active-file.json")
            with open(af_path, "w") as f:
                json.dump({"path": "/nonexistent/file.py"}, f)
            assert _read_active_file(td) == ""

    def test_returns_empty_on_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            af_path = os.path.join(td, "active-file.json")
            with open(af_path, "w") as f:
                f.write("not json")
            assert _read_active_file(td) == ""

    def test_returns_empty_when_path_key_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            af_path = os.path.join(td, "active-file.json")
            with open(af_path, "w") as f:
                json.dump({"other": "value"}, f)
            assert _read_active_file(td) == ""

    def test_returns_empty_when_path_is_empty_string(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            af_path = os.path.join(td, "active-file.json")
            with open(af_path, "w") as f:
                json.dump({"path": ""}, f)
            assert _read_active_file(td) == ""


class TestCleanLlmOutput:
    def test_strips_whitespace(self) -> None:
        assert _clean_llm_output("  hello  ") == "hello"

    def test_strips_double_quotes(self) -> None:
        assert _clean_llm_output('"hello"') == "hello"

    def test_strips_single_quotes(self) -> None:
        assert _clean_llm_output("'hello'") == "hello"

    def test_strips_both(self) -> None:
        assert _clean_llm_output('  "hello"  ') == "hello"


class TestModelVendorOrder:
    def test_claude(self) -> None:
        assert _model_vendor_order("claude-opus-4-6") == 0

    def test_openai(self) -> None:
        assert _model_vendor_order("gpt-4o") == 1
        assert _model_vendor_order("o3-mini") == 1

    def test_gemini(self) -> None:
        assert _model_vendor_order("gemini-2.0-flash") == 2

    def test_minimax(self) -> None:
        assert _model_vendor_order("minimax-model") == 3

    def test_openrouter(self) -> None:
        assert _model_vendor_order("openrouter/some-model") == 4

    def test_unknown(self) -> None:
        assert _model_vendor_order("unknown-model") == 5


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
