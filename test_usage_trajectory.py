"""Test that usage information is appended to every model message in trajectory."""

import os
import tempfile
import json
from pathlib import Path

from kiss.agents.coding_agents.claude_coding_agent import ClaudeCodingAgent


def test_usage_in_trajectory():
    """Run agent and verify usage info is in trajectory."""
    agent = ClaudeCodingAgent("test_agent")

    task_description = """Create a simple Python file called hello.py that prints 'Hello, World!'."""

    work_dir = tempfile.mkdtemp()
    old_cwd = os.getcwd()

    try:
        os.chdir(work_dir)
        result = agent.run(
            prompt_template=task_description,
            model_name="claude-sonnet-4-5",
            work_dir=work_dir,
            max_steps=5,
            use_browser=False,
        )

        print(f"\n{'='*80}")
        print("AGENT RUN COMPLETED")
        print(f"{'='*80}")
        print(f"Result: {result}")
        print(f"Budget used: ${agent.budget_used:.4f}")
        print(f"Total tokens: {agent.total_tokens_used}")

        # Check trajectory
        trajectory_file = Path(agent.state_dir) / "trajectory.jsonl"

        if not trajectory_file.exists():
            print(f"\n❌ ERROR: Trajectory file not found at {trajectory_file}")
            return False

        print(f"\n{'='*80}")
        print("CHECKING TRAJECTORY")
        print(f"{'='*80}")

        model_message_count = 0
        messages_with_usage_info = 0

        with open(trajectory_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    msg = json.loads(line.strip())
                    if msg.get("role") == "model":
                        model_message_count += 1
                        content = msg.get("content", "")

                        print(f"\n--- Model Message {model_message_count} (line {line_num}) ---")
                        print(f"Content length: {len(content)} chars")
                        print(f"Content preview (last 500 chars):\n{content[-500:]}")

                        # Check if usage info is present
                        if "Usage Information" in content:
                            messages_with_usage_info += 1
                            print("✅ Contains usage information")
                        else:
                            print("❌ Missing usage information")

                except json.JSONDecodeError as e:
                    print(f"⚠️ Failed to parse line {line_num}: {e}")

        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Total model messages: {model_message_count}")
        print(f"Messages with usage info: {messages_with_usage_info}")

        if model_message_count == messages_with_usage_info and model_message_count > 0:
            print("\n✅ SUCCESS: All model messages contain usage information!")
            return True
        else:
            print(f"\n❌ FAILURE: Not all model messages contain usage information")
            print(f"   Expected: {model_message_count}, Got: {messages_with_usage_info}")
            return False

    finally:
        os.chdir(old_cwd)
        print(f"\nWork directory: {work_dir}")


if __name__ == "__main__":
    success = test_usage_in_trajectory()
    if not success:
        raise Exception("Test failed: Usage information not properly appended to trajectory")
