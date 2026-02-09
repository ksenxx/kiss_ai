"""Test that usage information is printed exactly after tool calls."""

from unittest.mock import Mock, patch
from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock, UserMessage, ToolResultBlock

from kiss.agents.coding_agents.claude_coding_agent import ClaudeCodingAgent


async def mock_query_with_and_without_tool_calls(prompt, options):
    """Mock query with messages that have and don't have tool calls."""
    # Step 1: Assistant message WITH tool call
    msg1 = Mock(spec=AssistantMessage)
    msg1.content = [
        TextBlock(text="I'll read the file."),
        ToolUseBlock(name="Read", input={"file_path": "test.py"}, id="tool_1")
    ]
    msg1.usage = {"input_tokens": 100, "output_tokens": 50}
    yield msg1

    # User response
    user1 = Mock(spec=UserMessage)
    result1 = Mock(spec=ToolResultBlock)
    result1.content = "File content here"
    result1.is_error = False
    result1.tool_use_id = "tool_1"
    user1.content = [result1]
    yield user1

    # Step 2: Assistant message WITHOUT tool call (just text/thinking)
    msg2 = Mock(spec=AssistantMessage)
    msg2.content = [
        TextBlock(text="Let me think about this...")
    ]
    msg2.usage = {"input_tokens": 50, "output_tokens": 30}
    yield msg2

    # Step 3: Assistant message WITH tool call
    msg3 = Mock(spec=AssistantMessage)
    msg3.content = [
        TextBlock(text="Now I'll write."),
        ToolUseBlock(name="Write", input={"file_path": "out.py", "content": "code"}, id="tool_2")
    ]
    msg3.usage = {"input_tokens": 80, "output_tokens": 40}
    yield msg3

    # User response
    user2 = Mock(spec=UserMessage)
    result2 = Mock(spec=ToolResultBlock)
    result2.content = "Write successful"
    result2.is_error = False
    result2.tool_use_id = "tool_2"
    user2.content = [result2]
    yield user2

    # Step 4: Another message WITHOUT tool call
    msg4 = Mock(spec=AssistantMessage)
    msg4.content = [
        TextBlock(text="Done!")
    ]
    msg4.usage = {"input_tokens": 30, "output_tokens": 20}
    yield msg4


def test_usage_only_after_tool_calls():
    """Test that usage info is printed only after messages with tool calls."""
    print(f"\n{'='*80}")
    print("TESTING USAGE INFO PRINTED ONLY AFTER TOOL CALLS")
    print(f"{'='*80}")

    agent = ClaudeCodingAgent("test_agent")

    mock_printer = Mock()
    mock_printer.print_stream_event = Mock()
    mock_printer.print_message = Mock()
    mock_printer.print_usage_info = Mock()
    mock_printer.reset = Mock()

    with patch('kiss.agents.coding_agents.claude_coding_agent.query', new=mock_query_with_and_without_tool_calls):
        with patch('kiss.agents.coding_agents.claude_coding_agent.ConsolePrinter', return_value=mock_printer):
            result = agent.run(
                prompt_template="Test task",
                model_name="claude-sonnet-4-5",
                max_steps=10,
                use_browser=False,
            )

    # Check agent steps and message count
    print(f"\nTotal agent steps: {agent.step_count}")

    model_messages = [msg for msg in agent.messages if msg["role"] == "model"]
    print(f"Total model messages: {len(model_messages)}")

    # Count messages with and without tool calls
    messages_with_tool_calls = 0
    messages_without_tool_calls = 0

    for i, msg in enumerate(model_messages, 1):
        content = msg["content"]
        has_tool = "```python" in content  # Tool calls are formatted with code blocks

        if has_tool:
            messages_with_tool_calls += 1
            print(f"{i}. Has tool call: YES")
        else:
            messages_without_tool_calls += 1
            print(f"{i}. Has tool call: NO")

    print(f"\nMessages with tool calls: {messages_with_tool_calls}")
    print(f"Messages without tool calls: {messages_without_tool_calls}")

    # Check printer calls
    usage_print_calls = mock_printer.print_usage_info.call_count
    print(f"Usage info printed: {usage_print_calls} times")

    # Should print usage info only for messages WITH tool calls
    expected_usage_prints = messages_with_tool_calls
    print(f"Expected usage prints: {expected_usage_prints}")

    assert usage_print_calls == expected_usage_prints, \
        f"Expected {expected_usage_prints} usage prints, got {usage_print_calls}"

    # Verify we have both types of messages
    assert messages_with_tool_calls > 0, "Should have at least one message with tool calls"
    assert messages_without_tool_calls > 0, "Should have at least one message without tool calls"

    print(f"\n✅ SUCCESS: Usage info printed only after {expected_usage_prints} tool calls!")
    print(f"✅ Correctly skipped {messages_without_tool_calls} messages without tool calls")
    return True


def test_all_trajectory_messages_have_usage():
    """Test that ALL trajectory messages contain usage info (even without tool calls)."""
    print(f"\n{'='*80}")
    print("TESTING TRAJECTORY MESSAGES ALWAYS HAVE USAGE INFO")
    print(f"{'='*80}")

    agent = ClaudeCodingAgent("test_agent")

    mock_printer = Mock()
    mock_printer.print_stream_event = Mock()
    mock_printer.print_message = Mock()
    mock_printer.print_usage_info = Mock()
    mock_printer.reset = Mock()

    with patch('kiss.agents.coding_agents.claude_coding_agent.query', new=mock_query_with_and_without_tool_calls):
        with patch('kiss.agents.coding_agents.claude_coding_agent.ConsolePrinter', return_value=mock_printer):
            result = agent.run(
                prompt_template="Test task",
                model_name="claude-sonnet-4-5",
                max_steps=10,
                use_browser=False,
            )

    # All trajectory messages should have usage info in content
    model_messages = [msg for msg in agent.messages if msg["role"] == "model"]
    print(f"\nTotal model messages in trajectory: {len(model_messages)}")

    all_have_usage = True
    for i, msg in enumerate(model_messages, 1):
        if "Usage Information" in msg["content"]:
            print(f"{i}. ✅ Has usage info in trajectory")
        else:
            print(f"{i}. ❌ Missing usage info in trajectory")
            all_have_usage = False

    assert all_have_usage, "All trajectory messages should contain usage info"

    print(f"\n✅ SUCCESS: All {len(model_messages)} trajectory messages contain usage info!")
    return True


if __name__ == "__main__":
    try:
        test_usage_only_after_tool_calls()
        print()
        test_all_trajectory_messages_have_usage()

        print(f"\n{'='*80}")
        print("ALL TESTS PASSED ✅")
        print(f"{'='*80}")
        print("\nSummary:")
        print("- Usage info PRINTED only after tool calls")
        print("- Usage info in TRAJECTORY for all messages")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
