"""Test that KISSError is raised when budget and token limits are exceeded."""

import pytest
from unittest.mock import Mock, patch

from kiss.agents.coding_agents.claude_coding_agent import ClaudeCodingAgent
from kiss.core.kiss_error import KISSError
from kiss.core.base import Base
from kiss.core import config as config_module
from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock, UserMessage, ToolResultBlock


async def mock_query_high_token_usage(prompt, options):
    """Mock query that generates high token usage."""
    for step in range(5):
        mock_message = Mock(spec=AssistantMessage)
        mock_message.content = [
            TextBlock(text=f"Step {step + 1}"),
            ToolUseBlock(name="Read", input={"file_path": f"file{step}.py"}, id=f"tool_{step}")
        ]
        # Very high token usage
        mock_message.usage = {"input_tokens": 50000, "output_tokens": 50000}
        yield mock_message

        mock_user_msg = Mock(spec=UserMessage)
        mock_result = Mock(spec=ToolResultBlock)
        mock_result.content = f"Content {step}"
        mock_result.is_error = False
        mock_result.tool_use_id = f"tool_{step}"
        mock_user_msg.content = [mock_result]
        yield mock_user_msg


async def mock_query_high_cost(prompt, options):
    """Mock query that generates high cost."""
    for step in range(10):
        mock_message = Mock(spec=AssistantMessage)
        mock_message.content = [
            TextBlock(text=f"Step {step + 1}"),
            ToolUseBlock(name="Read", input={"file_path": f"file{step}.py"}, id=f"tool_{step}")
        ]
        # Moderate tokens but we'll mock high cost
        mock_message.usage = {"input_tokens": 1000, "output_tokens": 1000}
        yield mock_message

        mock_user_msg = Mock(spec=UserMessage)
        mock_result = Mock(spec=ToolResultBlock)
        mock_result.content = f"Content {step}"
        mock_result.is_error = False
        mock_result.tool_use_id = f"tool_{step}"
        mock_user_msg.content = [mock_result]
        yield mock_user_msg


def test_token_limit_exceeded():
    """Test that KISSError is raised when token limit is exceeded."""
    print(f"\n{'='*80}")
    print("TESTING TOKEN LIMIT EXCEEDED")
    print(f"{'='*80}")

    agent = ClaudeCodingAgent("test_agent")
    max_tokens = 50000  # Set low limit

    with patch('kiss.agents.coding_agents.claude_coding_agent.query', new=mock_query_high_token_usage):
        with pytest.raises(KISSError) as exc_info:
            result = agent.run(
                prompt_template="Test task",
                model_name="claude-sonnet-4-5",
                max_steps=100,  # High enough to not hit this limit
                use_browser=False,
            )

    error_message = str(exc_info.value)
    print(f"\n✅ KISSError raised as expected")
    print(f"\nError message:\n{error_message}")

    assert "Token limit exceeded" in error_message, \
        "Error message should mention token limit"

    print(f"\nAgent final state:")
    print(f"  Tokens used: {agent.total_tokens_used}/{agent.max_tokens}")
    print(f"  Budget used: ${agent.budget_used:.4f}")

    assert agent.total_tokens_used > max_tokens, \
        f"Token usage should exceed limit"

    print(f"\n✅ Token limit check working correctly")
    return True


def test_agent_budget_limit_exceeded():
    """Test that KISSError is raised when agent budget limit is exceeded."""
    print(f"\n{'='*80}")
    print("TESTING AGENT BUDGET LIMIT EXCEEDED")
    print(f"{'='*80}")

    agent = ClaudeCodingAgent("test_agent")
    max_budget = 0.01  # Very low budget limit

    # Mock calculate_cost to return high cost
    def mock_calculate_cost(model_name, input_tokens, output_tokens):
        return 0.005  # $0.005 per call

    with patch('kiss.agents.coding_agents.claude_coding_agent.query', new=mock_query_high_cost):
        with patch('kiss.core.models.model_info.calculate_cost', side_effect=mock_calculate_cost):
            with pytest.raises(KISSError) as exc_info:
                result = agent.run(
                    prompt_template="Test task",
                    model_name="claude-sonnet-4-5",
                    max_steps=100,
                    max_budget=max_budget,
                    use_browser=False,
                )

    error_message = str(exc_info.value)
    print(f"\n✅ KISSError raised as expected")
    print(f"\nError message:\n{error_message}")

    assert "Agent budget limit exceeded" in error_message, \
        "Error message should mention agent budget limit"

    print(f"\nAgent final state:")
    print(f"  Budget used: ${agent.budget_used:.4f}/${max_budget:.2f}")
    print(f"  Steps: {agent.step_count}")

    assert agent.budget_used > max_budget, \
        f"Budget should exceed limit"

    print(f"\n✅ Agent budget limit check working correctly")
    return True


def test_global_budget_limit_exceeded():
    """Test that KISSError is raised when global budget limit is exceeded."""
    print(f"\n{'='*80}")
    print("TESTING GLOBAL BUDGET LIMIT EXCEEDED")
    print(f"{'='*80}")

    # Save original global budget
    original_global_budget = Base.global_budget_used

    # Reset global budget for this test
    Base.global_budget_used = 0.0

    agent = ClaudeCodingAgent("test_agent")

    # Mock calculate_cost to return high cost
    def mock_calculate_cost(model_name, input_tokens, output_tokens):
        return 0.1  # $0.1 per call

    # Mock the global_max_budget config
    original_global_max = config_module.DEFAULT_CONFIG.agent.global_max_budget
    config_module.DEFAULT_CONFIG.agent.global_max_budget = 0.15  # Low limit

    try:
        with patch('kiss.agents.coding_agents.claude_coding_agent.query', new=mock_query_high_cost):
            with patch('kiss.core.models.model_info.calculate_cost', side_effect=mock_calculate_cost):
                with pytest.raises(KISSError) as exc_info:
                    result = agent.run(
                        prompt_template="Test task",
                        model_name="claude-sonnet-4-5",
                        max_steps=100,
                        max_budget=100.0,  # High agent budget
                        use_browser=False,
                    )

        error_message = str(exc_info.value)
        print(f"\n✅ KISSError raised as expected")
        print(f"\nError message:\n{error_message}")

        assert "Global budget limit exceeded" in error_message, \
            "Error message should mention global budget limit"

        print(f"\nGlobal budget state:")
        print(f"  Global budget used: ${Base.global_budget_used:.4f}/${0.15:.2f}")

        assert Base.global_budget_used > 0.15, \
            f"Global budget should exceed limit"

        print(f"\n✅ Global budget limit check working correctly")

    finally:
        # Restore original values
        Base.global_budget_used = original_global_budget
        config_module.DEFAULT_CONFIG.agent.global_max_budget = original_global_max

    return True


def test_no_error_within_limits():
    """Test that no error is raised when all limits are respected."""
    print(f"\n{'='*80}")
    print("TESTING NO ERROR WITHIN LIMITS")
    print(f"{'='*80}")

    async def mock_query_low_usage(prompt, options):
        """Mock query with low resource usage."""
        for step in range(2):
            mock_message = Mock(spec=AssistantMessage)
            mock_message.content = [
                TextBlock(text=f"Step {step + 1}"),
                ToolUseBlock(name="Read", input={"file_path": f"file{step}.py"}, id=f"tool_{step}")
            ]
            # Low token usage
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

    # Should NOT raise error when within limits
    with patch('kiss.agents.coding_agents.claude_coding_agent.query', new=mock_query_low_usage):
        result = agent.run(
            prompt_template="Test task",
            model_name="claude-sonnet-4-5",
            max_steps=10,
            max_budget=1.0,
            use_browser=False,
        )

    print(f"\nAgent final state:")
    print(f"  Steps: {agent.step_count}/{agent.max_steps}")
    print(f"  Tokens: {agent.total_tokens_used}/{agent.max_tokens}")
    print(f"  Budget: ${agent.budget_used:.4f}/${agent.max_budget:.2f}")

    assert agent.step_count < agent.max_steps
    assert agent.total_tokens_used < agent.max_tokens
    assert agent.budget_used < agent.max_budget

    print(f"\n✅ No error raised when within all limits")
    return True


if __name__ == "__main__":
    try:
        test_token_limit_exceeded()
        print()
        test_agent_budget_limit_exceeded()
        print()
        test_global_budget_limit_exceeded()
        print()
        test_no_error_within_limits()

        print(f"\n{'='*80}")
        print("ALL TESTS PASSED ✅")
        print(f"{'='*80}")
        print("\nSummary:")
        print("- KISSError raised when token limit exceeded")
        print("- KISSError raised when agent budget limit exceeded")
        print("- KISSError raised when global budget limit exceeded")
        print("- No error when within all limits")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
