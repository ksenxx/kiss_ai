"""Integration test: AssistantAgent goes through multiple sub-sessions."""

import os
import tempfile

import yaml

from kiss.agents.assistant.assistant_agent import AssistantAgent
from kiss.tests.conftest import requires_anthropic_api_key


@requires_anthropic_api_key
def test_assistant_agent_continues_to_second_sub_session() -> None:
    """Test that the agent enters a second sub-session when the first ends with success=False."""
    agent = AssistantAgent("Multi-Session Test Agent")
    work_dir = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    os.chdir(work_dir)

    # The task explicitly instructs the agent to fail session 1 and succeed in session 2.
    # This exercises the multi-sub-session continuation path in perform_task.
    task = (
        "This task has two phases:\n"
        "Phase 1: Create a file called phase1.txt containing 'phase 1 complete'. "
        "Then you MUST call finish(success=False, summary='Phase 1 done: created phase1.txt. "
        "Still need to create phase2.txt.')\n"
        "Phase 2 (after continuation): Create a file called phase2.txt "
        "containing 'phase 2 complete'. "
        "Verify both phase1.txt and phase2.txt exist using Bash. "
        "Then call finish(success=True, summary='Both phases complete')."
    )
    try:
        result = agent.run(
            prompt_template=task,
            model_name="claude-haiku-4-5",
            summarizer_model_name="claude-haiku-4-5",
            max_steps=10,
            max_sub_sessions=3,
            max_budget=1.0,
            work_dir=work_dir,
            headless=True,
            verbose=True,
        )
    finally:
        os.chdir(old_cwd)

    result_data = yaml.safe_load(result)
    print(f"\nResult: {result_data}")
    print(f"Budget used: ${agent.budget_used:.4f}")
    print(f"Tokens used: {agent.total_tokens_used}")

    assert result_data["success"], f"Agent task failed: {result_data.get('summary')}"
    assert os.path.exists(os.path.join(work_dir, "phase1.txt")), "phase1.txt was not created"
    assert os.path.exists(os.path.join(work_dir, "phase2.txt")), "phase2.txt was not created"


if __name__ == "__main__":
    test_assistant_agent_continues_to_second_sub_session()
    print("\nMulti-session integration test passed!")
