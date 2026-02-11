"""Repo agent that solves tasks in the current project root using RelentlessCodingAgent."""

from __future__ import annotations

import sys
from pathlib import Path

from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent

PROJECT_ROOT = str(Path(__file__).resolve().parents[4])


def main() -> None:
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter task: ")
    if not task.strip():
        raise ValueError("No task provided")

    # task = """
    # can you update README.md based on the latest code in the project.
    # """
    agent = RelentlessCodingAgent("RepoAgent")
    result = agent.run(
        prompt_template=task,
        model_name="claude-opus-4-6",
        work_dir=PROJECT_ROOT
    )
    print(result)


if __name__ == "__main__":
    main()
