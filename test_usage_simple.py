"""Simple test to verify usage information is printed."""

import tempfile
from kiss.agents.coding_agents.claude_coding_agent import ClaudeCodingAgent


def test_simple():
    """Test that usage info is printed."""
    agent = ClaudeCodingAgent("test-simple")

    task = "Create a file hello.txt with the text 'Hello, World!'"

    with tempfile.TemporaryDirectory() as work_dir:
        print("\n" + "=" * 80)
        print("RUNNING SIMPLE TEST - WATCH FOR USAGE INFO AT THE END:")
        print("=" * 80 + "\n")

        result = agent.run(
            model_name="claude-sonnet-4-5",
            prompt_template=task,
            work_dir=work_dir,
            max_steps=5,
            max_budget=1.0,
            use_browser=False,
        )

        print("\n" + "=" * 80)
        print("TEST COMPLETE")
        print("=" * 80)
        print(f"Result: {result}")
        print(f"Budget used: ${agent.budget_used:.6f}")
        print(f"Total tokens: {agent.total_tokens_used}")


if __name__ == "__main__":
    test_simple()
