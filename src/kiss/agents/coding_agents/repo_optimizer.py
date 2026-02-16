"""Repo agent that optimizes agent code using RelentlessCodingAgent."""

from __future__ import annotations

import argparse
from pathlib import Path

from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent

DEFAULT_PROJECT_ROOT = str(Path(__file__).resolve().parents[4])
DEFAULT_AGENT_CODE = "src/kiss/agents/coding_agents/relentless_coding_agent.py"
DEFAULT_MODEL = "claude-opus-4-6"

TASK_TEMPLATE = """
Can you run the agent code by executing the command 'uv run {agent_code}'
in the background so that I can see its output and you can continue
to monitor the output in real time, and correct the agent code if needed?
If you observe any repeated errors in the output or the agent is not able
to finish the task successfully, please fix the agent code and run the
command again.  Once the command succeeds and solves the task successfully,
analyze the output and optimize {agent_code}
so that the agent is able to solve the task successfully (1st priority),
faster (2nd priority) and with less cost (3rd priority).
Run the agent on the task to validate that it is successful in solving the task
completely and faster with lower cost.  If validation fails, roll back the changes
and try again.
Keep repeating the process until the agent can solve the task successfully, and
until the running time and the cost are reduced significantly.
DO NOT STOP CORRECTING THE AGENT CODE UNTIL IT IS SUCCESSFUL AT SOLVING THE TASK.
DO NOT KILL the current process running repo_optimizer.py.

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

"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize an agent using RelentlessCodingAgent")
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT,
                        help=f"Project root directory (default: {DEFAULT_PROJECT_ROOT})")
    parser.add_argument("--agent-code", default=DEFAULT_AGENT_CODE,
                        help=f"Path to agent code to optimize (default: {DEFAULT_AGENT_CODE})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model name to use (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    task = TASK_TEMPLATE.format(agent_code=args.agent_code)
    agent = RelentlessCodingAgent("RepoOptimizer")
    result = agent.run(
        prompt_template=task,
        model_name=args.model,
        work_dir=args.project_root
    )
    print(result)


if __name__ == "__main__":
    main()
