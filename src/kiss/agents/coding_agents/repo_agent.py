"""Repo agent that solves tasks in the current project root using RelentlessCodingAgent."""

from __future__ import annotations

from pathlib import Path

from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent

PROJECT_ROOT = str(Path(__file__).resolve().parents[4])

TASK = """
can you run 'uv run src/kiss/agents/coding_agents/\
relentless_coding_agent.py' and monitor the output?
If you observe any repeated errors in the output,
please fix them and run the command again.
Once the command succeeds, analyze the output and optimize
src/kiss/agents/coding_agents/relentless_coding_agent.py
so that it runs reliably, faster with less cost.
Keep repeating the process until the running time
and the cost is reduced significantly, such 99%.  Ensure 
that theagent is able to handle the errors and continue
running the task until it is successful.
"""


def main() -> None:
    task = TASK
    agent = RelentlessCodingAgent("RepoAgent")
    result = agent.run(
        prompt_template=task,
        model_name="claude-opus-4-6",
        work_dir=PROJECT_ROOT
    )
    print(result)


if __name__ == "__main__":
    main()
