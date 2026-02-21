"""Repo agent that optimizes agent code using RelentlessCodingAgent."""

from __future__ import annotations

import argparse
from pathlib import Path

from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent

DEFAULT_PROJECT_ROOT = str(Path(__file__).resolve().parents[4])
DEFAULT_AGENT_CODE = "src/kiss/agents/coding_agents/relentless_coding_agent.py"
DEFAULT_MODEL = "claude-opus-4-6"

# Once the command succeeds and solves the task successfully,
# analyze the output and optimize {agent_code}
# so that the agent is able to solve the task successfully (1st priority),
# faster (2nd priority) and with less cost (3rd priority).

# , and
# until the running time and the cost are reduced significantly.
# DO NOT STOP CORRECTING THE AGENT CODE UNTIL IT IS SUCCESSFUL AT SOLVING THE TASK.
# DO NOT KILL the current process running repo_optimizer.py.


TASK_TEMPLATE = """
Can you run the agent code by executing the command 'uv run {agent_code}'
in the background so that you can continue to monitor the output in real time,
and correct the agent code if needed?  I MUST be able to see the agent output
in real time.

If you observe any repeated errors in the output or the agent is not able
to finish the task successfully, please fix the agent code and run the
command again.  Repeat the process until the agent can solve the task successfully.

After the agent code manages to solve the task successfully, run the agent again
and monitor its output in real time.  Check for opportunities to optimize the agent code
for higher speed and lower cost.  If you find any opportunities, optimize the agent code
and run the agent again.  Repeat the process until the agent can solve the task successfully,
at higer speed and lower cost.

While editing the agent code, make sure that the agent does not get to see any
validation criterion or the answers to the problems in any possible way.
Even feedback from a validation agent must not be used by the agent.  Basically,
make sure that the agent is not aware of the validation criterion or the answers
to the problems in any possible way.


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
