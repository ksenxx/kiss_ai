"""Integration test: cc/ models calling finish() via text-based tool calling.

Regression test for the bug where cc/opus omitted the ``is_continue``
parameter when calling ``finish()``, producing a TypeError that was
returned as a raw string and then failed YAML parsing downstream.
"""

import shutil

import pytest
import yaml

from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.relentless_agent import finish

_has_claude = shutil.which("claude") is not None
requires_claude_cli = pytest.mark.skipif(not _has_claude, reason="claude CLI not installed")


@requires_claude_cli
class TestCCModelFinishIntegration:
    """Verify cc/ models can call finish() and produce valid YAML."""

    @pytest.mark.timeout(120)
    def test_finish_called_with_valid_params(self) -> None:
        """The model calls finish(); the result is valid YAML with all keys."""
        m = ClaudeCodeModel("cc/haiku")
        m.initialize(
            "Call the finish function with success=True and "
            'summary="task done". Do NOT call any other function.'
        )
        calls, _content, _response = m.generate_and_process_with_tools(
            {"finish": finish}
        )

        finish_calls = [c for c in calls if c["name"] == "finish"]
        assert finish_calls, f"Expected a finish() call, got: {calls}"

        args = finish_calls[0]["arguments"]
        result = finish(**args)

        parsed = yaml.safe_load(result)
        assert isinstance(parsed, dict), f"Expected dict, got: {type(parsed)}"

        assert "success" in parsed, f"Missing 'success' key in: {parsed}"
        assert "is_continue" in parsed, f"Missing 'is_continue' key in: {parsed}"
        assert "summary" in parsed, f"Missing 'summary' key in: {parsed}"

        assert isinstance(parsed["success"], bool)
        assert isinstance(parsed["is_continue"], bool)
        assert isinstance(parsed["summary"], str)

    @pytest.mark.timeout(120)
    def test_finish_without_is_continue_still_works(self) -> None:
        """Even if the model omits is_continue, finish() defaults it to False."""
        m = ClaudeCodeModel("cc/haiku")
        m.initialize(
            "Call the finish function with success=True and "
            'summary="done". Do NOT pass is_continue. '
            "Do NOT call any other function."
        )
        calls, _content, _response = m.generate_and_process_with_tools(
            {"finish": finish}
        )

        finish_calls = [c for c in calls if c["name"] == "finish"]
        assert finish_calls, f"Expected a finish() call, got: {calls}"

        args = finish_calls[0]["arguments"]
        result = finish(**args)
        parsed = yaml.safe_load(result)
        assert isinstance(parsed, dict)
        assert "success" in parsed
        assert "is_continue" in parsed
        assert parsed["is_continue"] is False


def main() -> None:
    """Run the integration tests directly."""
    test = TestCCModelFinishIntegration()
    print("Running test_finish_called_with_valid_params...")
    test.test_finish_called_with_valid_params()
    print("PASSED")
    print("Running test_finish_without_is_continue_still_works...")
    test.test_finish_without_is_continue_still_works()
    print("PASSED")
    print("\nAll integration tests passed!")


if __name__ == "__main__":
    main()
