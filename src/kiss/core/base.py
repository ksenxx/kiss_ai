# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Base agent class with common functionality for all KISS agents."""

import json
import sys
import time
from pathlib import Path
from typing import Any, ClassVar

import yaml
from yaml.nodes import ScalarNode

from kiss.core import config as config_module
from kiss.core.models.model_info import get_max_context_length
from kiss.core.print_to_console import ConsolePrinter
from kiss.core.printer import Printer
from kiss.core.utils import config_to_dict


def _str_presenter(dumper: yaml.Dumper, data: str) -> ScalarNode:
    """Use literal block style for multiline strings in YAML output."""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")  # type: ignore[reportUnknownMemberType]


yaml.add_representer(str, _str_presenter)

_KISS_DIR = Path().home() / ".kiss"

_artifact_dir = Path(config_module.DEFAULT_CONFIG.agent.artifact_dir)

SYSTEM_PROMPT = f"""
# Rules
- Write() for new files. Edit() for small changes.
- Use bounded poll loops with output sent to a temporary file for bash commands;
  never use unbounded waits.
- Use go_to_url() for browser tool and internet search or testing an agent/app.
- Look at `{_artifact_dir.parent}/TASK_HISTORY.md` for task history and context.
  Pay more attention to the recent tasks over old tasks. Do not try to finish a
  task from the task history. DO NOT WRITE/EDIT the task history.
- Call finish(success=True, summary="detailed summary of what was accomplished")
  immediately when task is complete.
- YOU **MUST FOLLOW THE INSTRUCTIONS DIRECTLY**

## Code Style Guidelines
- Write simple, clean, and readable code with minimal indirection
- Avoid unnecessary object attributes, local variables, and config variables
- Avoid tight coupling between files and modules.
- Avoid object/struct attribute redirections
- No redundant abstractions or duplicate code
- Each function should do one thing well
- Use clear, descriptive names
- Public methods MUST have full documentation
- Once you finish the task, DO NOT write documentations unless the task specifically requires it.
- You MUST check and test the code you have written except for formatting/typing changes

## Testing Instructions
- Run lint and typecheckers and fix any lint and typecheck errors
- Carefully read the code, find and fix redundancies, duplications,
  inconsistencies, errors, and AI slop in the code
- Generate comprehensive tests so that you achieve 100% branch coverage
- Tests MUST NOT use mocks, patches, or any form of test doubles
- Integration tests are HIGHLY encouraged
- You MUST not add tests that are redundant or duplicate of existing
  tests or does not add new coverage over existing tests
- Generate meaningful stress tests for the code if you are
  optimizing the code for performance
- Each test should be independent and verify actual behavior

## Use web tools when you need to:
- Look up API documentation or library usage from the internet
- Find examples of similar implementations
- Understand existing code in the project
- Augment recent knowledge and to perform web based tasks
- Read papers from the internet to understand concepts and algorithms

### Self-Improvement Loop
- Just before finishing an agent task, update `{_artifact_dir.parent}/LESSONS.md`
  with instructions and rules for yourself ONLY IF you have learned any major lessons
  during the task execution.  Also compact the lessons you have learned
  into concise instructions and rules for you to follow.
- The lessons MUST NOT be specific to a task, but about agent behavior.
- Review lessons when the agent starts

## After you have implemented the task, aggresively and carefully simplify and clean up the code
 - Remove unnecessary conditional checks
 - Make sure that the code is still working correctly
 - Simplify and clean up the test code
"""


class Base:
    """Base class for all KISS agents with common state management and persistence."""

    agent_counter: ClassVar[int] = 1
    global_budget_used: ClassVar[float] = 0.0

    model_name: str
    messages: list[dict[str, Any]]
    function_map: Any
    run_start_timestamp: int
    budget_used: float
    total_tokens_used: int
    step_count: int
    printer: Printer | None

    def __init__(self, name: str) -> None:
        """Initialize a Base agent instance.

        Args:
            name: The name identifier for the agent.
        """
        self.name = name
        self.id = Base.agent_counter
        Base.agent_counter += 1
        self.base_dir = ""
        self.printer: Printer | None = None

    def set_printer(
        self,
        printer: Printer | None = None,
        verbose: bool | None = None,
    ) -> None:
        """Configure the output printer for this agent.

        Args:
            printer: An existing Printer instance to use directly. If provided,
                verbose is ignored.
            verbose: Whether to print to the console. If None,
                uses the verbose config value.
        """
        self.printer: Printer | None = None
        if config_module.DEFAULT_CONFIG.agent.verbose:
            if printer:
                self.printer = printer
            elif verbose is not False:
                self.printer = ConsolePrinter()

    def _build_state_dict(self) -> dict[str, Any]:
        """Build state dictionary for saving.

        Returns:
            dict[str, Any]: A dictionary containing all agent state for persistence.
        """
        try:
            max_tokens = get_max_context_length(self.model_name)
        except Exception:
            max_tokens = None

        return {
            "name": self.name,
            "id": self.id,
            "messages": self.messages,
            "function_map": list(self.function_map),
            "run_start_timestamp": self.run_start_timestamp,
            "run_end_timestamp": int(time.time()),
            "config": config_to_dict(),
            "arguments": getattr(self, "arguments", {}),
            "prompt_template": getattr(self, "prompt_template", ""),
            "is_agentic": getattr(self, "is_agentic", True),
            "model": self.model_name,
            "budget_used": self.budget_used,
            "total_budget": getattr(
                self, "max_budget", config_module.DEFAULT_CONFIG.agent.max_agent_budget
            ),
            "global_budget_used": Base.global_budget_used,
            "global_max_budget": config_module.DEFAULT_CONFIG.agent.global_max_budget,
            "tokens_used": self.total_tokens_used,
            "max_tokens": max_tokens,
            "step_count": self.step_count,
            "max_steps": getattr(self, "max_steps", config_module.DEFAULT_CONFIG.agent.max_steps),
            "command": " ".join(sys.argv),
        }

    def _save(self) -> None:
        """Save the agent's state to a YAML file in the artifacts directory.

        The file is saved to {artifact_dir}/trajectories/trajectory_{name}_{id}_{timestamp}.yaml
        """
        folder_path = Path(config_module.DEFAULT_CONFIG.agent.artifact_dir) / "trajectories"
        folder_path.mkdir(parents=True, exist_ok=True)
        name_safe = self.name.replace(" ", "_").replace("/", "_")
        filename = folder_path / f"trajectory_{name_safe}_{self.id}_{self.run_start_timestamp}.yaml"
        with filename.open("w", encoding="utf-8") as f:
            yaml.dump(self._build_state_dict(), f, indent=2)

    def get_trajectory(self) -> str:
        """Return the trajectory as JSON for visualization.

        Returns:
            str: A JSON-formatted string of all messages in the agent's history.
        """
        return json.dumps(self.messages, indent=2)

    def _add_message(self, role: str, content: Any, timestamp: int | None = None) -> None:
        """Add a message to the history.

        Args:
            role: The role of the message sender (e.g., 'user', 'model').
            content: The content of the message.
            timestamp: Optional Unix timestamp. If None, uses current time.
        """
        self.messages.append(
            {
                "unique_id": len(self.messages),
                "role": role,
                "content": content,
                "timestamp": timestamp if timestamp is not None else int(time.time()),
            }
        )
