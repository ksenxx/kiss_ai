"""Integration test for AssistantAgent: uses both browser and bash tools."""

import os
import tempfile
import time

import yaml

from kiss.agents.assistant.assistant_agent import AssistantAgent


def test_assistant_agent_web_and_bash() -> None:
    """Test that the agent can use both browser tools and coding tools together."""
    agent = AssistantAgent("Integration Test Agent")
    task = """
**Task:** Research information from a website and create a local report using both
the browser and coding tools.

**Steps:**
1. Use go_to_url() to navigate to https://httpbin.org/html
2. Use get_page_content(text_only=True) to read the full text content of the page
3. Use Bash() to run: echo "Report generated at $(date)" > {work_dir}/report.txt
4. Use go_to_url() to navigate to https://httpbin.org/json
5. Use get_page_content(text_only=True) to read the JSON content
6. Use Write() to append the extracted content to {work_dir}/report.txt
7. Use Bash() to verify the file exists: cat {work_dir}/report.txt
8. Take a screenshot with screenshot("{work_dir}/screenshot.png")
9. Call finish(success=True, summary="Created report with web content and bash commands")

**Important:** You MUST use BOTH browser tools (go_to_url, get_page_content, screenshot)
AND coding tools (Bash, Write) to complete this task.
"""
    work_dir = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    os.chdir(work_dir)
    start_time = time.time()
    try:
        result = agent.run(
            prompt_template=task,
            arguments={"work_dir": work_dir},
            model_name="claude-sonnet-4-5",
            max_steps=15,
            max_budget=2.0,
            work_dir=work_dir,
            headless=True,
            print_to_console=True,
        )
    finally:
        os.chdir(old_cwd)

    elapsed = time.time() - start_time
    result_data = yaml.safe_load(result)

    print(f"\n{'=' * 60}")
    print(f"Test completed in {elapsed:.1f}s")
    print(f"Success: {result_data['success']}")
    print(f"Summary: {result_data['summary']}")
    print(f"Cost: ${agent.budget_used:.4f}")
    print(f"Tokens: {agent.total_tokens_used}")
    print(f"Work dir: {work_dir}")
    print(f"{'=' * 60}")

    assert result_data["success"], f"Agent task failed: {result_data.get('summary')}"

    report_path = os.path.join(work_dir, "report.txt")
    assert os.path.exists(report_path), "report.txt was not created"
    content = open(report_path).read()
    assert len(content) > 0, "report.txt is empty"
    print(f"\nReport content:\n{content[:500]}")


def main() -> None:
    test_assistant_agent_web_and_bash()
    print("\nAll integration tests passed!")


if __name__ == "__main__":
    main()
