"""Test that KISSError is raised when max_steps is exceeded."""

import pytest
from unittest.mock import Mock, patch

from kiss.agents.coding_agents.claude_coding_agent import ClaudeCodingAgent
from kiss.core.kiss_error import KISSError
from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock, UserMessage, ToolResultBlock


async def mock_query_many_steps(prompt, options):
    """Mock query that generates many assistant messages."""
    for step in range(20):
        mock_message = Mock(spec=AssistantMessage)
        mock_message.content = [
            TextBlock(text=f"Step {step + 1}"),
            ToolUseBlock(name="Read", input={"file_path": f"file{step}.py"}, id=f"tool_{step}")
        ]
        mock_message.usage = {"input_tokens": 100, "output_tokens": 50}
        yield mock_message

        mock_user_msg = Mock(spec=UserMessage)
        mock_result = Mock(spec=ToolResultBlock)
        mock_result.content = f"Content {step}"
        mock_result.is_error = False
        mock_result.tool_use_id = f"tool_{step}"
        mock_user_msg.content = [mock_result]
        yield mock_user_msg


def test_raises_kiss_error_at_max_steps():
    """Test that KISSError is raised when max_steps is reached."""
    print(f"\n{'='*80}")
    print("TESTING KISSERROR RAISED AT MAX_STEPS")
    print(f"{'='*80}")

    agent = ClaudeCodingAgent("test_agent")
    max_steps = 3

    with patch('kiss.agents.coding_agents.claude_coding_agent.query', new=mock_query_many_steps):
        with pytest.raises(KISSError) as exc_info:
            result = agent.run(
                prompt_template="Test task",
                model_name="claude-sonnet-4-5",
                max_steps=max_steps,
                use_browser=False,
            )

    error_message = str(exc_info.value)
    print(f"\n✅ KISSError raised as expected")
    print(f"\nError message:\n{error_message}")

    # Verify error message contains relevant info
    assert f"Maximum steps ({max_steps})" in error_message, \
        "Error message should mention max_steps"
    assert f"step {max_steps}" in error_message, \
        "Error message should mention actual step count"

    # Verify agent state
    print(f"\nAgent final state:")
    print(f"  Steps executed: {agent.step_count}")
    print(f"  Max steps: {agent.max_steps}")
    print(f"  Budget used: ${agent.budget_used:.4f}")

    assert agent.step_count == max_steps, \
        f"Agent should have executed exactly {max_steps} steps"

    print(f"\n✅ Agent executed exactly {max_steps} steps before raising error")
    return True


def test_error_raised_at_different_max_steps():
    """Test KISSError is raised at different max_steps values."""
    print(f"\n{'='*80}")
    print("TESTING KISSERROR AT DIFFERENT MAX_STEPS VALUES")
    print(f"{'='*80}")

    test_cases = [1, 2, 5, 10]

    for max_steps in test_cases:
        print(f"\nTesting max_steps={max_steps}")
        agent = ClaudeCodingAgent(f"test_agent_{max_steps}")

        with patch('kiss.agents.coding_agents.claude_coding_agent.query', new=mock_query_many_steps):
            with pytest.raises(KISSError) as exc_info:
                result = agent.run(
                    prompt_template="Test task",
                    model_name="claude-sonnet-4-5",
                    max_steps=max_steps,
                    use_browser=False,
                )

        assert agent.step_count == max_steps, \
            f"Expected {max_steps} steps, got {agent.step_count}"

        error_msg = str(exc_info.value)
        assert f"Maximum steps ({max_steps})" in error_msg

        print(f"  ✅ Raised KISSError at step {max_steps}")

    print(f"\n✅ All max_steps values correctly raise KISSError")
    return True


def test_no_error_when_under_max_steps():
    """Test that no error is raised when steps don't exceed max."""
    print(f"\n{'='*80}")
    print("TESTING NO ERROR WHEN UNDER MAX_STEPS")
    print(f"{'='*80}")

    async def mock_query_few_steps(prompt, options):
        """Mock query with only 2 steps."""
        for step in range(2):
            mock_message = Mock(spec=AssistantMessage)
            mock_message.content = [
                TextBlock(text=f"Step {step + 1}"),
                ToolUseBlock(name="Read", input={"file_path": f"file{step}.py"}, id=f"tool_{step}")
            ]
            mock_message.usage = {"input_tokens": 100, "output_tokens": 50}
            yield mock_message

            mock_user_msg = Mock(spec=UserMessage)
            mock_result = Mock(spec=ToolResultBlock)
            mock_result.content = f"Content {step}"
            mock_result.is_error = False
            mock_result.tool_use_id = f"tool_{step}"
            mock_user_msg.content = [mock_result]
            yield mock_user_msg

    agent = ClaudeCodingAgent("test_agent")

    # Should NOT raise error when steps (2) < max_steps (5)
    with patch('kiss.agents.coding_agents.claude_coding_agent.query', new=mock_query_few_steps):
        result = agent.run(
            prompt_template="Test task",
            model_name="claude-sonnet-4-5",
            max_steps=5,  # More than the 2 steps we'll execute
            use_browser=False,
        )

    print(f"\nAgent steps: {agent.step_count}")
    print(f"Max steps: {agent.max_steps}")

    assert agent.step_count < agent.max_steps, \
        "Steps should be less than max_steps"

    print(f"\n✅ No error raised when steps ({agent.step_count}) < max_steps ({agent.max_steps})")
    return True


if __name__ == "__main__":
    try:
        test_raises_kiss_error_at_max_steps()
        print()
        test_error_raised_at_different_max_steps()
        print()
        test_no_error_when_under_max_steps()

        print(f"\n{'='*80}")
        print("ALL TESTS PASSED ✅")
        print(f"{'='*80}")
        print("\nSummary:")
        print("- KISSError raised when max_steps exceeded")
        print("- Error message includes step count info")
        print("- No error when steps under limit")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
