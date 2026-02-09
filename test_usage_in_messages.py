"""Test that usage information is appended to model messages."""

import time
from unittest.mock import Mock

from kiss.agents.coding_agents.claude_coding_agent import ClaudeCodingAgent
from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock, ResultMessage


def test_usage_info_in_assistant_message():
    """Test that assistant messages include usage information."""
    agent = ClaudeCodingAgent("test_agent")

    # Initialize agent state
    agent._init_run_state("claude-sonnet-4-5", ["Read", "Write"])
    agent.base_dir = "."
    agent.max_tokens = 100000
    agent.max_steps = 10
    agent.max_budget = 1.0
    agent.budget_used = 0.25
    agent.total_tokens_used = 5000
    agent.input_tokens_used = 3000
    agent.output_tokens_used = 2000
    agent.step_count = 2
    agent.last_step_input_tokens = 0
    agent.last_step_output_tokens = 0

    # Create a mock assistant message
    mock_message = Mock(spec=AssistantMessage)
    mock_message.content = [
        TextBlock(text="I will help you with that task."),
        ToolUseBlock(name="Read", input={"file_path": "test.py"}, id="tool_1")
    ]
    mock_message.usage = {"input_tokens": 100, "output_tokens": 50}

    # Process the message
    timestamp = int(time.time())
    agent._process_assistant_message(mock_message, timestamp)

    # Check that the message was added with usage info
    assert len(agent.messages) == 1
    message = agent.messages[0]
    assert message["role"] == "model"
    assert "Usage Information" in message["content"]
    assert "Token usage:" in message["content"]
    assert "Agent budget:" in message["content"]
    assert "Global budget:" in message["content"]
    assert "Step:" in message["content"]

    print("✅ Assistant message contains usage information")
    print(f"\nMessage content:\n{message['content']}")


def test_usage_info_in_result_message():
    """Test that result messages include usage information."""
    agent = ClaudeCodingAgent("test_agent")

    # Initialize agent state
    agent._init_run_state("claude-sonnet-4-5", ["Read", "Write"])
    agent.base_dir = "."
    agent.max_tokens = 100000
    agent.max_steps = 10
    agent.max_budget = 1.0
    agent.budget_used = 0.35
    agent.total_tokens_used = 7000
    agent.input_tokens_used = 4000
    agent.output_tokens_used = 3000
    agent.step_count = 3

    # Create a mock result message
    mock_message = Mock(spec=ResultMessage)
    mock_message.result = "Task completed successfully!"
    mock_message.usage = {"input_tokens": 50, "output_tokens": 25}
    mock_message.total_cost_usd = 0.40

    # Process the message
    timestamp = int(time.time())
    result = agent._process_result_message(mock_message, timestamp)

    # Check that the message was added with usage info
    assert len(agent.messages) == 1
    message = agent.messages[0]
    assert message["role"] == "model"
    assert "Usage Information" in message["content"]
    assert "Token usage:" in message["content"]
    assert "Agent budget:" in message["content"]
    assert "Global budget:" in message["content"]
    assert "Step:" in message["content"]
    assert "Task completed successfully!" in message["content"]

    print("✅ Result message contains usage information")
    print(f"\nMessage content:\n{message['content']}")
    print(f"\nResult: {result}")


def test_usage_info_format():
    """Test the format of usage information string."""
    agent = ClaudeCodingAgent("test_agent")

    # Initialize agent state with specific values
    agent._init_run_state("claude-sonnet-4-5", ["Read", "Write"])
    agent.base_dir = "."
    agent.max_tokens = 100000
    agent.max_steps = 10
    agent.max_budget = 1.0
    agent.budget_used = 0.1234
    agent.total_tokens_used = 5432
    agent.step_count = 3

    usage_info = agent._get_usage_info_string()

    print(f"\nUsage info string:\n{usage_info}")

    # Verify the format
    assert "#### Usage Information" in usage_info
    assert "Token usage: 5432/100000" in usage_info
    assert "Agent budget: $0.1234 / $1.00" in usage_info
    assert "Step: 3/10" in usage_info

    print("✅ Usage information format is correct")


if __name__ == "__main__":
    print("="*80)
    print("Testing usage information in messages")
    print("="*80)

    test_usage_info_format()
    print()

    test_usage_info_in_assistant_message()
    print()

    test_usage_info_in_result_message()
    print()

    print("="*80)
    print("All tests passed! ✅")
    print("="*80)
