"""Verify that trajectory contains usage information in all model messages."""

import json
import time
from unittest.mock import Mock

from kiss.agents.coding_agents.claude_coding_agent import ClaudeCodingAgent
from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock, ResultMessage


def test_complete_trajectory_with_usage():
    """Simulate a complete agent run and verify trajectory."""
    agent = ClaudeCodingAgent("test_agent")

    # Initialize agent state
    agent._init_run_state("claude-sonnet-4-5", ["Read", "Write"])
    agent.base_dir = "."
    agent.max_tokens = 100000
    agent.max_steps = 10
    agent.max_budget = 1.0
    agent.budget_used = 0.0
    agent.total_tokens_used = 0
    agent.input_tokens_used = 0
    agent.output_tokens_used = 0
    agent.step_count = 0
    agent.last_step_input_tokens = 0
    agent.last_step_output_tokens = 0

    timestamp = int(time.time())

    # Simulate step 1: Assistant with tool call
    mock_message1 = Mock(spec=AssistantMessage)
    mock_message1.content = [
        TextBlock(text="I'll read the file first."),
        ToolUseBlock(name="Read", input={"file_path": "test.py"}, id="tool_1")
    ]
    mock_message1.usage = {"input_tokens": 500, "output_tokens": 100}
    agent._process_assistant_message(mock_message1, timestamp)

    # Simulate step 2: Assistant with another tool call
    timestamp += 5
    mock_message2 = Mock(spec=AssistantMessage)
    mock_message2.content = [
        TextBlock(text="Now I'll write the updated file."),
        ToolUseBlock(name="Write", input={"file_path": "test.py", "content": "..."}, id="tool_2")
    ]
    mock_message2.usage = {"input_tokens": 600, "output_tokens": 150}
    agent._process_assistant_message(mock_message2, timestamp)

    # Simulate step 3: Final result
    timestamp += 5
    mock_result = Mock(spec=ResultMessage)
    mock_result.result = "Task completed! I've updated the file."
    mock_result.usage = {"input_tokens": 550, "output_tokens": 50}
    mock_result.total_cost_usd = None  # Let it calculate naturally
    agent._process_result_message(mock_result, timestamp)

    # Verify trajectory
    print("\n" + "="*80)
    print("TRAJECTORY VERIFICATION")
    print("="*80)

    trajectory_json = agent.get_trajectory()
    trajectory = json.loads(trajectory_json)

    print(f"\nTotal messages in trajectory: {len(trajectory)}")
    print(f"Expected model messages: 3")

    model_messages = [msg for msg in trajectory if msg["role"] == "model"]
    print(f"Actual model messages: {len(model_messages)}")

    # Verify each model message has usage information
    messages_with_usage = 0
    for i, msg in enumerate(model_messages, 1):
        print(f"\n--- Model Message {i} ---")
        print(f"Content length: {len(msg['content'])} chars")

        if "Usage Information" in msg["content"]:
            messages_with_usage += 1
            print("✅ Contains usage information")

            # Verify all required fields are present
            required_fields = ["Token usage:", "Agent budget:", "Global budget:", "Step:"]
            for field in required_fields:
                if field in msg["content"]:
                    print(f"  ✅ {field} present")
                else:
                    print(f"  ❌ {field} MISSING")
        else:
            print("❌ Missing usage information")

        # Show last 300 chars
        print(f"\nLast 300 chars:\n{msg['content'][-300:]}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Model messages: {len(model_messages)}")
    print(f"Messages with usage info: {messages_with_usage}")

    if messages_with_usage == len(model_messages):
        print("\n✅ SUCCESS: All model messages contain usage information!")
        return True
    else:
        print(f"\n❌ FAILURE: {len(model_messages) - messages_with_usage} messages missing usage info")
        return False


if __name__ == "__main__":
    success = test_complete_trajectory_with_usage()
    if not success:
        raise Exception("Trajectory verification failed")
