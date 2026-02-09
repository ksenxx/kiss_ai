"""Test that max_steps enforcement doesn't cause async errors."""

import sys
import io
from contextlib import redirect_stderr
from unittest.mock import Mock, patch

from kiss.agents.coding_agents.claude_coding_agent import ClaudeCodingAgent
from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock, UserMessage, ToolResultBlock


async def mock_query_many_steps(prompt, options):
    """Mock query that generates many steps."""
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


def test_no_async_runtime_error():
    """Test that breaking at max_steps doesn't cause RuntimeError."""
    print(f"\n{'='*80}")
    print("TESTING FOR ASYNC ERRORS")
    print(f"{'='*80}")

    # Capture stderr to check for async errors
    stderr_capture = io.StringIO()

    agent = ClaudeCodingAgent("test_agent")

    with redirect_stderr(stderr_capture):
        with patch('kiss.agents.coding_agents.claude_coding_agent.query', new=mock_query_many_steps):
            result = agent.run(
                prompt_template="Test task",
                model_name="claude-sonnet-4-5",
                max_steps=5,
                use_browser=False,
            )

    stderr_output = stderr_capture.getvalue()

    print(f"\nAgent executed {agent.step_count} steps")
    print(f"Max steps: 5")

    # Check for async errors in stderr
    has_runtime_error = "RuntimeError" in stderr_output
    has_cancel_scope_error = "cancel scope" in stderr_output
    has_generator_exit_error = "GeneratorExit" in stderr_output
    has_task_exception = "Task exception was never retrieved" in stderr_output

    if stderr_output:
        print(f"\nStderr output ({len(stderr_output)} chars):")
        print("-" * 80)
        print(stderr_output[:500])
        if len(stderr_output) > 500:
            print(f"... (truncated, total {len(stderr_output)} chars)")
        print("-" * 80)

    if has_runtime_error:
        print("\n❌ RuntimeError found in stderr")
        return False

    if has_cancel_scope_error:
        print("\n❌ Cancel scope error found in stderr")
        return False

    if has_generator_exit_error:
        print("\n❌ GeneratorExit error found in stderr")
        return False

    if has_task_exception:
        print("\n❌ Task exception error found in stderr")
        return False

    if stderr_output.strip():
        print(f"\n⚠️  Non-empty stderr (but no async errors detected)")
    else:
        print(f"\n✅ No stderr output - clean execution")

    # Verify agent stopped at max_steps
    assert agent.step_count == 5, f"Expected step_count=5, got {agent.step_count}"

    print(f"✅ Agent stopped correctly at max_steps=5")
    print(f"✅ No async RuntimeError or cancel scope errors!")

    return True


if __name__ == "__main__":
    try:
        success = test_no_async_runtime_error()

        print(f"\n{'='*80}")
        if success:
            print("TEST PASSED ✅ - No async errors detected")
        else:
            print("TEST FAILED ❌ - Async errors found")
        print(f"{'='*80}")

        if not success:
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
