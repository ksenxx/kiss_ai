"""Repo agent that solves tasks in the current project root using AssistantAgent."""

from __future__ import annotations

import sys
from pathlib import Path

from kiss.agents.sorkar.assistant_agent import AssistantAgent

PROJECT_ROOT = str(Path(__file__).resolve().parents[4])


def main() -> None:
    """Run an AssistantAgent on the project root with a task from CLI args or stdin."""
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter task: ")
    if not task.strip():
        raise ValueError("No task provided")

    agent = AssistantAgent("RepoAgent")
    result = agent.run(
        prompt_template=task,
        model_name="claude-opus-4-6",
        work_dir=PROJECT_ROOT,
        headless=True,
    )
    print(result)


if __name__ == "__main__":
    main()
