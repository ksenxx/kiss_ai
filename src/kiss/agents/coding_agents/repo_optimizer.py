"""Repo agent that solves tasks in the current project root using RelentlessCodingAgent."""

from __future__ import annotations

from pathlib import Path

from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent

PROJECT_ROOT = str(Path(__file__).resolve().parents[4])

TASK = """
can you run 'uv run src/kiss/agents/coding_agents/relentless_coding_agent.py' and monitor the output?
If you observe any repeated errors in the output,
please fix them and run the command again.
Once the command succeeds, analyze the output and optimize
src/kiss/agents/coding_agents/relentless_coding_agent.py
so that it runs reliably, faster with less cost.
Keep repeating the process until the running time
and the cost is reduced significantly, such 99%.  Ensure 
that the agent is able to handle the errors and continue
running the task until it is successful.

## Instructions:
1. Do NOT change the agent's interface or streaming mechanism
2. The agent MUST still work correctly on the task above
3. Do NOT use: caching, multiprocessing, async/await, docker

## Strategies
- IMPORTANT: Optimizations must be GENERAL across the task, not task-specific
- Shorter system prompts preserving meaning
- Remove redundant instructions
- Minimize conversation turns

## Time Reduction
- Run short-running commands in bash
- Batch operations where possible
- Use early termination when goals are achieved
- Optimize loops and data structures
- Search the web for time reduction techniques

## Agentic Patterns
- deeply search the web for information about various latest agentic patterns
- patterns that solve long-horizon tasks scalably, efficiently and accurately
- patterns that makes Python code faster
- patterns that make bash commands faster
- patterns that make the agent faster
- patterns that make the agent more reliable
- patterns that make the agent more cost-effective
- deeply invent and implement new agent architectures that are more efficient and reliable
- try some of these patterns in the agent's source code based on your needs

- 
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
