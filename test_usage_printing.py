"""Test that usage information is printed after every assistant message."""

from unittest.mock import Mock, patch, call
from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock, UserMessage, ToolResultBlock

from kiss.agents.coding_agents.claude_coding_agent import ClaudeCodingAgent


async def mock_query_mixed_messages(prompt, options):
    """Mock query with both tool calls and non-tool-call messages."""
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

    # Step 2: Assistant message WITHOUT tool call (just thinking)
    msg2 = Mock(spec=AssistantMessage)
    msg2.content = [
        TextBlock(text="Let me analyze this...")
    ]
    msg2.usage = {"input_tokens": 50, "output_tokens": 30}
    yield msg2

    # Step 3: Assistant message WITH tool call again
    msg3 = Mock(spec=AssistantMessage)
    msg3.content = [
        TextBlock(text="Now I'll write the file."),
        ToolUseBlock(name="Write", input={"file_path": "test.py", "content": "new"}, id="tool_2")
    ]
    msg3.usage = {"input_tokens": 80, "output_tokens": 40}
    yield msg3


def test_usage_printed_for_all_messages():
    """Test that usage info is printed for every AssistantMessage."""
    print(f"\n{'='*80}")
    print("TESTING USAGE INFO PRINTING")
    print(f"{'='*80}")

    agent = ClaudeCodingAgent("test_agent")

    # Mock the printer to track calls
    mock_printer = Mock()
    mock_printer.print_stream_event = Mock()
    mock_printer.print_message = Mock()
    mock_printer.print_usage_info = Mock()
    mock_printer.reset = Mock()

    with patch('kiss.agents.coding_agents.claude_coding_agent.query', new=mock_query_mixed_messages):
        with patch('kiss.agents.coding_agents.claude_coding_agent.ConsolePrinter', return_value=mock_printer):
            result = agent.run(
                prompt_template="Test task",
                model_name="claude-sonnet-4-5",
                max_steps=10,
                use_browser=False,
            )

    # Check how many times print_usage_info was called
    usage_print_calls = mock_printer.print_usage_info.call_count

    print(f"\nAgent steps executed: {agent.step_count}")
    print(f"print_usage_info called: {usage_print_calls} times")

    # We should have 3 assistant messages, so 3 usage info prints
    expected_calls = 3
    print(f"Expected usage prints: {expected_calls}")

    # Verify each call
    print(f"\nUsage print calls:")
    for i, call_args in enumerate(mock_printer.print_usage_info.call_args_list, 1):
        usage_string = call_args[0][0] if call_args[0] else ""
        print(f"{i}. Usage string length: {len(usage_string)} chars")
        if "Usage Information" in usage_string:
            print(f"   ✅ Contains 'Usage Information'")
        else:
            print(f"   ❌ Missing 'Usage Information'")

    # Assertions
    assert usage_print_calls == expected_calls, \
        f"Expected {expected_calls} usage prints, got {usage_print_calls}"

    # Verify all calls contain usage information
    for call_args in mock_printer.print_usage_info.call_args_list:
        usage_string = call_args[0][0] if call_args[0] else ""
        assert "Usage Information" in usage_string, \
            "Usage string missing 'Usage Information'"
        assert "Token usage:" in usage_string, \
            "Usage string missing 'Token usage:'"

    print(f"\n✅ SUCCESS: Usage information printed after all {expected_calls} assistant messages!")
    return True


def test_usage_in_trajectory_and_print():
    """Test that usage info appears in both trajectory and print output."""
    print(f"\n{'='*80}")
    print("TESTING USAGE IN TRAJECTORY AND PRINT")
    print(f"{'='*80}")

    agent = ClaudeCodingAgent("test_agent")

    mock_printer = Mock()
    mock_printer.print_stream_event = Mock()
    mock_printer.print_message = Mock()
    mock_printer.print_usage_info = Mock()
    mock_printer.reset = Mock()

    with patch('kiss.agents.coding_agents.claude_coding_agent.query', new=mock_query_mixed_messages):
        with patch('kiss.agents.coding_agents.claude_coding_agent.ConsolePrinter', return_value=mock_printer):
            result = agent.run(
                prompt_template="Test task",
                model_name="claude-sonnet-4-5",
                max_steps=10,
                use_browser=False,
            )

    # Check trajectory
    model_messages = [msg for msg in agent.messages if msg["role"] == "model"]
    print(f"\nModel messages in trajectory: {len(model_messages)}")

    messages_with_usage_in_content = 0
    for i, msg in enumerate(model_messages, 1):
        if "Usage Information" in msg["content"]:
            messages_with_usage_in_content += 1
            print(f"{i}. ✅ Has usage info in content")
        else:
            print(f"{i}. ❌ Missing usage info in content")

    # Check printer calls
    usage_print_calls = mock_printer.print_usage_info.call_count
    print(f"\nUsage info print calls: {usage_print_calls}")

    # Both should match
    assert messages_with_usage_in_content == len(model_messages), \
        f"Not all trajectory messages have usage info"
    assert usage_print_calls == len(model_messages), \
        f"Print calls ({usage_print_calls}) don't match messages ({len(model_messages)})"

    print(f"\n✅ SUCCESS: Usage info appears in both trajectory and print output!")
    return True


if __name__ == "__main__":
    try:
        test_usage_printed_for_all_messages()
        print()
        test_usage_in_trajectory_and_print()

        print(f"\n{'='*80}")
        print("ALL TESTS PASSED ✅")
        print(f"{'='*80}")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
