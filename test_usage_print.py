"""Test to verify usage information is printed exactly once after each step."""

import os
import tempfile
from kiss.agents.coding_agents.claude_coding_agent import ClaudeCodingAgent


def main() -> None:
    agent = ClaudeCodingAgent("Usage Test Agent")
    task_description = """
Write a simple Python script named 'hello.py' that prints "Hello, World!".
Then run the script to verify it works.
"""

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
        print("\n--- FINAL RESULT ---")
        print(f"Success: {bool(result)}")
        print(f"Steps: {agent.step_count}")
        print(f"Total tokens: {agent.total_tokens_used}")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
