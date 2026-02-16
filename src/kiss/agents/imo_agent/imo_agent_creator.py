"""Repo agent that solves tasks in the current project root using RelentlessCodingAgent."""

from __future__ import annotations

from pathlib import Path

from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent

PROJECT_ROOT = str(Path(__file__).resolve().parents[4])


task = """
- can you create an agent in src/kiss/agents/imo_agent/imo_agent.py using
src/kiss/core/kiss_agent.py and 
src/kiss/agents/coding_agents/relentless_coding_agent.py based on the agent
described in the paper https://arxiv.org/abs/2507.15855 .
- First, add all the IMO 2025 problems accurately and their validation criterion and their hardness.
- Second, create the agent imo_agent.py.
- Make sure that that the imo_agent.py does not get to see the validation criterion or the answers.
- In the main function, call the imo_agent.py to solve the problem 4.
- Monitor the output of the imo_agent.py to make sure that it is solving the problem successfully.
  If it fails, modify the imo_agent.py agent code and try again.
- Use an independent KISSagent to validate the results against the validation criterion.
- If validation fails, modify the imo_agent.py agent code and try again.
- Keep repeating the process until the agent can run the task successfully to completion.
- Repeat the process until the agent can solve all the IMO 2025 problems successfully to completion.
- Try using the most powerful models for the solver and the verifier.
"""

def main() -> None:
    agent = RelentlessCodingAgent("IMOAgentCreator")
    result = agent.run(
        prompt_template=task,
        model_name="claude-opus-4-6",
        work_dir=PROJECT_ROOT
    )
    print(result)


if __name__ == "__main__":
    main()
