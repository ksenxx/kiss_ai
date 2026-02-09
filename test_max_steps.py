"""Test that ClaudeCodingAgent stops when max_steps is reached."""

import time
from unittest.mock import Mock, AsyncMock, patch
from collections.abc import AsyncGenerator

from kiss.agents.coding_agents.claude_coding_agent import ClaudeCodingAgent
from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock, UserMessage, ToolResultBlock


async def mock_query_with_many_steps(prompt, options):
    """Mock query that generates many assistant messages to exceed max_steps."""
    # Simulate 10 steps where each step has a tool call
    for step in range(10):
        # Yield assistant message with tool call
        mock_message = Mock(spec=AssistantMessage)
        mock_message.content = [
            TextBlock(text=f"Step {step + 1}: Processing..."),
            ToolUseBlock(name="Read", input={"file_path": f"file{step}.py"}, id=f"tool_{step}")
        ]
        mock_message.usage = {"input_tokens": 100, "output_tokens": 50}
        yield mock_message

        # Yield user message with tool result
        mock_user_msg = Mock(spec=UserMessage)
        mock_result = Mock(spec=ToolResultBlock)
        mock_result.content = f"File content for file{step}.py"
        mock_result.is_error = False
        mock_result.tool_use_id = f"tool_{step}"
        mock_user_msg.content = [mock_result]
        yield mock_user_msg


def test_stops_at_max_steps():
    """Test that agent stops when max_steps is reached."""
    agent = ClaudeCodingAgent("test_agent")

    with patch('kiss.agents.coding_agents.claude_coding_agent.query', new=mock_query_with_many_steps):
        result = agent.run(
            prompt_template="Test task",
            model_name="claude-sonnet-4-5",
            max_steps=3,  # Set low limit
            use_browser=False,
        )

    print(f"\n{'='*80}")
    print("MAX STEPS TEST RESULTS")
    print(f"{'='*80}")
    print(f"Max steps allowed: 3")
    print(f"Actual steps executed: {agent.step_count}")
    print(f"Budget used: ${agent.budget_used:.4f}")
    print(f"Total tokens: {agent.total_tokens_used}")

    # Verify agent stopped at max_steps
    assert agent.step_count == 3, f"Expected step_count=3, got {agent.step_count}"

    # Check trajectory
    print(f"\n{'='*80}")
    print("TRAJECTORY MESSAGES")
    print(f"{'='*80}")

    model_messages = [msg for msg in agent.messages if msg["role"] == "model"]
    print(f"Total model messages: {len(model_messages)}")

    for i, msg in enumerate(model_messages, 1):
        content_preview = msg["content"][:100].replace("\n", " ")
        print(f"{i}. {content_preview}...")

    # Should have exactly 3 model messages (one per step)
    assert len(model_messages) == 3, f"Expected 3 model messages, got {len(model_messages)}"

    print(f"\n✅ SUCCESS: Agent stopped at max_steps={3}")
    return True


def test_different_max_steps_values():
    """Test with different max_steps values."""
    test_cases = [1, 2, 5]

    print(f"\n{'='*80}")
    print("TESTING DIFFERENT MAX_STEPS VALUES")
    print(f"{'='*80}")

    for max_steps in test_cases:
        agent = ClaudeCodingAgent(f"test_agent_{max_steps}")

        with patch('kiss.agents.coding_agents.claude_coding_agent.query', new=mock_query_with_many_steps):
            result = agent.run(
                prompt_template="Test task",
                model_name="claude-sonnet-4-5",
                max_steps=max_steps,
                use_browser=False,
            )

        print(f"\nTest with max_steps={max_steps}:")
        print(f"  Actual steps: {agent.step_count}")
        print(f"  Model messages: {len([m for m in agent.messages if m['role'] == 'model'])}")

        assert agent.step_count == max_steps, \
            f"Expected step_count={max_steps}, got {agent.step_count}"
        print(f"  ✅ Stopped correctly at {max_steps} steps")

    print(f"\n✅ All max_steps tests passed!")
    return True


if __name__ == "__main__":
    try:
        test_stops_at_max_steps()
        print()
        test_different_max_steps_values()

        print(f"\n{'='*80}")
        print("ALL TESTS PASSED ✅")
        print(f"{'='*80}")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise
