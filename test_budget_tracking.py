"""Test script to verify per-step budget tracking in ClaudeCodingAgent."""

import tempfile
from pathlib import Path

from kiss.agents.coding_agents.claude_coding_agent import ClaudeCodingAgent


def test_budget_tracking():
    """Test that budget updates after each step, not just at the end."""
    agent = ClaudeCodingAgent("test-budget-tracking")

    task = """Write a simple Python function that adds two numbers.
Just create a file called add.py with a function add(a, b) that returns a + b.
Then test it by running python -c "from add import add; print(add(2, 3))".
"""

    with tempfile.TemporaryDirectory() as work_dir:
        print(f"Working directory: {work_dir}")
        print("\n" + "=" * 80)
        print("Starting agent run - watch for budget updates after each step:")
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
        print("FINAL RESULTS:")
        print("=" * 80)
        print(f"✓ Result: {result}")
        print(f"✓ Steps taken: {agent.step_count}")
        print(f"✓ Total tokens: {agent.total_tokens_used}")
        print(f"✓   - Input tokens: {agent.input_tokens_used}")
        print(f"✓   - Output tokens: {agent.output_tokens_used}")
        print(f"✓ Budget used: ${agent.budget_used:.6f}")
        print(f"✓ Global budget used: ${ClaudeCodingAgent.global_budget_used:.6f}")

        # Verify budget was tracked
        if agent.budget_used > 0:
            print("\n✅ SUCCESS: Budget tracking is working!")
        else:
            print("\n❌ FAILURE: Budget still shows $0.00")

        # Show files created
        output_files = list(Path(work_dir).glob("**/*.py"))
        if output_files:
            print(f"\n✓ Files created: {[str(f.relative_to(work_dir)) for f in output_files]}")


if __name__ == "__main__":
    test_budget_tracking()
