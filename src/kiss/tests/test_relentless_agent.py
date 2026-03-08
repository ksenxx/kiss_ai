"""Tests for relentless_agent finish() function and is_continue logic."""

import yaml

from kiss.core.relentless_agent import finish


class TestFinish:
    """Tests for the finish() function."""

    def test_success_true_is_continue_false(self) -> None:
        result = yaml.safe_load(finish(success=True, is_continue=False, summary="done"))
        assert result == {"success": True, "is_continue": False, "summary": "done"}

    def test_success_false_is_continue_true(self) -> None:
        result = yaml.safe_load(finish(success=False, is_continue=True, summary="wip"))
        assert result == {"success": False, "is_continue": True, "summary": "wip"}

    def test_success_false_is_continue_false(self) -> None:
        result = yaml.safe_load(finish(success=False, is_continue=False, summary="failed"))
        assert result == {"success": False, "is_continue": False, "summary": "failed"}

    def test_success_true_is_continue_true(self) -> None:
        result = yaml.safe_load(finish(success=True, is_continue=True, summary="done"))
        assert result == {"success": True, "is_continue": True, "summary": "done"}

    def test_string_success_true(self) -> None:
        result = yaml.safe_load(finish(success="True", is_continue="False", summary="x"))  # type: ignore[arg-type]
        assert result["success"] is True
        assert result["is_continue"] is False

    def test_string_success_false(self) -> None:
        result = yaml.safe_load(finish(success="false", is_continue="true", summary="x"))  # type: ignore[arg-type]
        assert result["success"] is False
        assert result["is_continue"] is True

    def test_string_yes_values(self) -> None:
        result = yaml.safe_load(finish(success="yes", is_continue="1", summary="y"))  # type: ignore[arg-type]
        assert result["success"] is True
        assert result["is_continue"] is True

    def test_string_no_values(self) -> None:
        result = yaml.safe_load(finish(success="no", is_continue="0", summary="n"))  # type: ignore[arg-type]
        assert result["success"] is False
        assert result["is_continue"] is False

    def test_empty_summary(self) -> None:
        result = yaml.safe_load(finish(success=True, is_continue=False, summary=""))
        assert result["summary"] == ""

    def test_multiline_summary(self) -> None:
        summary = "Step 1: did X\nStep 2: did Y\nStep 3: did Z"
        result = yaml.safe_load(finish(success=True, is_continue=False, summary=summary))
        assert result["summary"] == summary
