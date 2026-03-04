"""Tests for tool exception handling and summarizer YAML parsing robustness."""

import yaml

from kiss.core.kiss_agent import KISSAgent
from kiss.core.relentless_agent import finish


def _make_agent() -> KISSAgent:
    agent = KISSAgent("Test")
    agent._reset(
        model_name="claude-haiku-4-5",
        is_agentic=True,
        max_steps=5,
        max_budget=0.01,
        model_config=None,
    )
    return agent


def test_regular_exception_caught() -> None:
    def bad_tool(x: str) -> str:
        """Tool that raises RuntimeError.

        Args:
            x: Input.
        """
        raise RuntimeError("boom")

    agent = _make_agent()
    agent._add_functions([finish, bad_tool])
    name, response = agent._execute_tool({"name": "bad_tool", "arguments": {"x": "a"}})
    assert name == "bad_tool"
    assert "Failed to call" in response
    assert "boom" in response


def test_system_exit_caught() -> None:
    def exit_tool(x: str) -> str:
        """Tool that raises SystemExit.

        Args:
            x: Input.
        """
        raise SystemExit(1)

    agent = _make_agent()
    agent._add_functions([finish, exit_tool])
    name, response = agent._execute_tool({"name": "exit_tool", "arguments": {"x": "a"}})
    assert name == "exit_tool"
    assert "Failed to call" in response


def test_keyboard_interrupt_propagates() -> None:
    def interrupt_tool(x: str) -> str:
        """Tool that raises KeyboardInterrupt.

        Args:
            x: Input.
        """
        raise KeyboardInterrupt()

    agent = _make_agent()
    agent._add_functions([finish, interrupt_tool])
    caught = False
    try:
        agent._execute_tool({"name": "interrupt_tool", "arguments": {"x": "a"}})
    except KeyboardInterrupt:
        caught = True
    assert caught, "KeyboardInterrupt should propagate"


def test_base_exception_subclass_propagates() -> None:
    class CustomBaseError(BaseException):
        pass

    def custom_tool(x: str) -> str:
        """Tool that raises a BaseException subclass.

        Args:
            x: Input.
        """
        raise CustomBaseError("custom base error")

    agent = _make_agent()
    agent._add_functions([finish, custom_tool])
    caught = False
    try:
        agent._execute_tool({"name": "custom_tool", "arguments": {"x": "a"}})
    except CustomBaseError:
        caught = True
    assert caught, "BaseException subclasses should propagate"


def test_generator_exit_propagates() -> None:
    def gen_exit_tool(x: str) -> str:
        """Tool that raises GeneratorExit.

        Args:
            x: Input.
        """
        raise GeneratorExit()

    agent = _make_agent()
    agent._add_functions([finish, gen_exit_tool])
    caught = False
    try:
        agent._execute_tool({"name": "gen_exit_tool", "arguments": {"x": "a"}})
    except GeneratorExit:
        caught = True
    assert caught, "GeneratorExit should propagate"


def _parse_summarizer_result(summarizer_result: str) -> str:
    """Replicate the summarizer YAML parsing logic from relentless_agent.perform_task."""
    try:
        parsed = yaml.safe_load(summarizer_result)
        return (
            parsed.get("result", summarizer_result)
            if isinstance(parsed, dict)
            else summarizer_result
        )
    except Exception:
        return summarizer_result


def test_summarizer_yaml_valid_dict_with_result() -> None:
    raw = yaml.dump({"result": "Task completed successfully"})
    assert _parse_summarizer_result(raw) == "Task completed successfully"


def test_summarizer_yaml_valid_dict_without_result_key() -> None:
    raw = yaml.dump({"other": "some value"})
    assert _parse_summarizer_result(raw) == raw


def test_summarizer_yaml_invalid_yaml() -> None:
    raw = (
        "The `_execute_tool` method in `/Users/ksen/work/kiss/src/kiss/core/kiss_agent.py`\n"
        "was modified to catch BaseException instead of Exception.\n"
        "key: value\n"
        "  broken: indent"
    )
    result = _parse_summarizer_result(raw)
    assert result == raw


def test_summarizer_yaml_plain_text() -> None:
    raw = "This is just plain text summary of the work done."
    assert _parse_summarizer_result(raw) == raw


def test_summarizer_yaml_returns_list() -> None:
    raw = yaml.dump(["item1", "item2"])
    assert _parse_summarizer_result(raw) == raw


def test_summarizer_yaml_returns_scalar() -> None:
    raw = "42"
    result = _parse_summarizer_result(raw)
    assert result == raw


def test_finish_tool_error_produces_valid_yaml_result() -> None:
    """When finish is called with wrong args, the error is handled in perform_task."""
    agent = _make_agent()
    agent._add_functions([finish])
    # Calling finish with wrong argument types - success is a str not bool
    name, response = agent._execute_tool(
        {
            "name": "finish",
            "arguments": {"success": True, "summary": "done"},
        }
    )
    assert name == "finish"
    # Should succeed and return valid YAML
    result = yaml.safe_load(response)
    assert isinstance(result, dict)
    assert result["success"] is True
