# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests locking behavior of code paths being simplified.

Covers:
- ``_AutocompleteMixin._active_file_identifier_matches``
  (autocomplete.py).
- ``get_custom_model_entry`` / ``build_model_config`` header parsing
  (vscode_config.py).
- ``JsonPrinter`` cost/token/step offset arithmetic in ``result`` and
  ``usage_info`` events, and ``_coalesce_events`` (json_printer.py).

No mocks or fakes: real functions are called on real objects and temp files.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

from kiss.agents.vscode.autocomplete import _AutocompleteMixin, _ghost_suffix
from kiss.agents.vscode.json_printer import JsonPrinter, _coalesce_events
from kiss.agents.vscode.vscode_config import (
    build_model_config,
    get_custom_model_entry,
)


class _AC(_AutocompleteMixin):
    """Concrete mixin host; the tested methods use no instance state."""


def _server() -> _AC:
    return _AC()


class TestActiveFileIdentifierMatches:
    def test_matches_sorted_longest_first_then_alpha(self) -> None:
        content = "calc calculate calcify calculate_more"
        matches = _server()._active_file_identifier_matches("calc", "", content)
        assert matches == ["calculate_more", "calculate", "calcify"]

    def test_trailing_non_word_returns_empty_list(self) -> None:
        assert _server()._active_file_identifier_matches("hi ", "", "high") == []

    def test_short_partial_returns_empty_list(self) -> None:
        assert _server()._active_file_identifier_matches("a", "", "apple") == []

    def test_includes_dot_chains(self) -> None:
        content = "os.path.join os.path.exists"
        matches = _server()._active_file_identifier_matches(
            "call os.path", "", content,
        )
        assert set(matches) == {"os.path.join", "os.path.exists"}

    def test_no_content_returns_empty_list(self) -> None:
        assert _server()._active_file_identifier_matches("calc", "", "") == []

    def test_case_sensitive_prefix(self) -> None:
        assert (
            _server()._active_file_identifier_matches("Calc", "", "calculate")
            == []
        )

    def test_nonexistent_file_returns_empty_list(self) -> None:
        matches = _server()._active_file_identifier_matches(
            "test", "/nonexistent/no/such/file.py", "",
        )
        assert matches == []

    def test_disk_fallback(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write("frobnicate_widget = 1\n")
            path = f.name
        try:
            matches = _server()._active_file_identifier_matches("frob", path, "")
            assert matches == ["frobnicate_widget"]
        finally:
            os.unlink(path)


class TestGhostSuffix:
    def test_empty_completions(self) -> None:
        assert _ghost_suffix("query", []) == ""

    def test_task_kind_uses_full_query(self) -> None:
        comps = [{"type": "task", "text": "fix the bug in parser"}]
        assert _ghost_suffix("fix the", comps) == " bug in parser"

    def test_identifier_kind_uses_trailing_token(self) -> None:
        comps = [{"type": "identifier", "text": "calculate_total"}]
        assert _ghost_suffix("run calc", comps) == "ulate_total"

    def test_mismatched_prefix_returns_empty(self) -> None:
        comps = [{"type": "identifier", "text": "zzz"}]
        assert _ghost_suffix("run calc", comps) == ""


class TestCustomModelConfig:
    def test_no_endpoint_returns_none(self) -> None:
        assert get_custom_model_entry({"custom_endpoint": ""}) is None
        assert build_model_config({"custom_endpoint": ""}) is None

    def test_entry_with_headers(self) -> None:
        cfg = {
            "custom_endpoint": "https://api.example.com/v1/",
            "custom_api_key": "sk-123",
            "custom_headers": "X-One: alpha\nbad line no colon\nX-Two:  beta ",
        }
        entry = get_custom_model_entry(cfg)
        assert entry is not None
        assert entry["name"] == "custom/v1"
        assert entry["endpoint"] == "https://api.example.com/v1/"
        assert entry["api_key"] == "sk-123"
        assert entry["extra_headers"] == {"X-One": "alpha", "X-Two": "beta"}
        assert entry["vendor"] == "Custom"

    def test_entry_without_headers_has_empty_dict(self) -> None:
        entry = get_custom_model_entry({"custom_endpoint": "http://e/m"})
        assert entry is not None
        assert entry["extra_headers"] == {}

    def test_model_config_with_headers(self) -> None:
        cfg = {
            "custom_endpoint": "http://localhost:8000",
            "custom_api_key": "k",
            "custom_headers": "Authorization: Bearer t",
        }
        mc = build_model_config(cfg)
        assert mc == {
            "base_url": "http://localhost:8000",
            "api_key": "k",
            "extra_headers": {"Authorization": "Bearer t"},
        }

    def test_model_config_omits_empty_headers_and_key(self) -> None:
        mc = build_model_config({
            "custom_endpoint": "http://x",
            "custom_headers": "junk without colon",
        })
        assert mc == {"base_url": "http://x"}


class TestJsonPrinterOffsets:
    def _recording_printer(self) -> tuple[JsonPrinter, str]:
        p = JsonPrinter()
        p._thread_local.task_id = "regr-task"
        p.start_recording()
        return p, "regr-task"

    def test_result_event_applies_offsets(self) -> None:
        p, _ = self._recording_printer()
        p.budget_offset = 0.5
        p.tokens_offset = 100
        p.steps_offset = 3
        p.print(
            "done", type="result", total_tokens=10, cost="$1.2500", step_count=2,
        )
        events = p.stop_recording()
        results = [e for e in events if e["type"] == "result"]
        assert len(results) == 1
        assert results[0]["cost"] == "$1.7500"
        assert results[0]["total_tokens"] == 110
        assert results[0]["step_count"] == 5

    def test_result_malformed_cost_passthrough(self) -> None:
        p, _ = self._recording_printer()
        p.budget_offset = 0.5
        p.print("done", type="result", cost="$abc")
        results = [e for e in p.stop_recording() if e["type"] == "result"]
        assert results[0]["cost"] == "$abc"

    def test_usage_info_applies_offsets(self) -> None:
        p, _ = self._recording_printer()
        p.budget_offset = 0.25
        p.tokens_offset = 7
        p.steps_offset = 1
        broadcasts: list[dict[str, Any]] = []
        orig = p.broadcast

        def capture(event: dict[str, Any]) -> None:
            broadcasts.append(event)
            orig(event)

        p.broadcast = capture  # type: ignore[method-assign]
        p.print(
            "usage", type="usage_info",
            total_tokens=3, cost="$1.0000", total_steps=2,
        )
        usage = [e for e in broadcasts if e["type"] == "usage_info"]
        assert usage[0]["cost"] == "$1.2500"
        assert usage[0]["total_tokens"] == 10
        assert usage[0]["total_steps"] == 3

    def test_usage_info_non_dollar_cost_passthrough(self) -> None:
        p, _ = self._recording_printer()
        broadcasts: list[dict[str, Any]] = []
        orig = p.broadcast

        def capture(event: dict[str, Any]) -> None:
            broadcasts.append(event)
            orig(event)

        p.broadcast = capture  # type: ignore[method-assign]
        p.print("usage", type="usage_info", cost="N/A")
        usage = [e for e in broadcasts if e["type"] == "usage_info"]
        assert usage[0]["cost"] == "N/A"

    def test_result_yaml_parsing(self) -> None:
        p, _ = self._recording_printer()
        text = "success: true\nis_continue: false\nsummary: all good"
        p.print(text, type="result")
        results = [e for e in p.stop_recording() if e["type"] == "result"]
        assert results[0]["success"] is True
        assert results[0]["is_continue"] is False
        assert results[0]["summary"] == "all good"


class TestCoalesceEvents:
    def test_merges_consecutive_deltas(self) -> None:
        events = [
            {"type": "text_delta", "text": "a"},
            {"type": "text_delta", "text": "b"},
            {"type": "tool_call", "name": "Bash"},
            {"type": "text_delta", "text": "c"},
        ]
        out = _coalesce_events(events)
        assert out == [
            {"type": "text_delta", "text": "ab"},
            {"type": "tool_call", "name": "Bash"},
            {"type": "text_delta", "text": "c"},
        ]

    def test_empty_list(self) -> None:
        assert _coalesce_events([]) == []

    def test_non_mergeable_types_untouched(self) -> None:
        events = [
            {"type": "tool_call", "name": "A"},
            {"type": "tool_call", "name": "B"},
        ]
        assert _coalesce_events(events) == events
