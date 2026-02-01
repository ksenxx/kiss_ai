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

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.models.model_info import get_max_context_length
from kiss.core.utils import config_to_dict


def _str_presenter(dumper: yaml.Dumper, data: str) -> ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")  # type: ignore[reportUnknownMemberType]


yaml.add_representer(str, _str_presenter)


DEFAULT_SYSTEM_PROMPT = """
## Code Style Guidelines
- Write simple, readable code with minimal indirection
- Avoid unnecessary object attributes and local variables
- No redundant abstractions or duplicate code
- Each function should do one thing well
- Use clear, descriptive names
- NO need to write documentations or comments unless absolutely necessary

## Testing Requirements
- Run lint and fix any lint errors
- Generate comprehensive tests for EVERY function and feature
- Tests MUST NOT use mocks, patches, or any form of test doubles
- Test with real inputs and verify real outputs
- Test edge cases: empty inputs, None values, boundary conditions
- Test error conditions with actual invalid inputs
- Each test should be independent and verify actual behavior

## Code Structure
- Main implementation code first
- Test code in a separate section using unittest or pytest
- Include a __main__ block to run tests

## Use tools when you need to:
- Look up API documentation or library usage
- Find examples of similar implementations
- Understand existing code in the project

## After you have implemented the task, simplify the code
 - Remove unnessary object/struct attributes, variables, config variables
 - Remove redundant and duplicate code
 - Remove unnecessary comments
 - Make sure that the code is still working correctly

"""


class Base:
    """Base class for all KISS agents with common state management and persistence."""

    agent_counter: ClassVar[int] = 1
    global_budget_used: ClassVar[float] = 0.0

    def __init__(self, name: str) -> None:
        self.name = name
        self.id = Base.agent_counter
        Base.agent_counter += 1
        self.base_dir = ""

    def _init_run_state(self, model_name: str, function_map: list[str]) -> None:
        """Initialize common run state variables."""
        self.model_name = model_name
        self.function_map = function_map
        self.messages: list[dict[str, Any]] = []
        self.step_count = 0
        self.total_tokens_used = 0
        self.budget_used = 0.0
        self.run_start_timestamp = int(time.time())

    def _build_state_dict(self) -> dict[str, Any]:
        """Build state dictionary for saving."""
        try:
            max_tokens = get_max_context_length(self.model_name)
        except Exception:
            max_tokens = None

        return {
            "name": self.name,
            "id": self.id,
            "messages": self.messages,
            "function_map": self.function_map,
            "run_start_timestamp": self.run_start_timestamp,
            "run_end_timestamp": int(time.time()),
            "config": config_to_dict(),
            "arguments": getattr(self, "arguments", {}),
            "prompt_template": getattr(self, "prompt_template", ""),
            "is_agentic": getattr(self, "is_agentic", True),
            "model": self.model_name,
            "budget_used": self.budget_used,
            "total_budget": getattr(self, "max_budget", DEFAULT_CONFIG.agent.max_agent_budget),
            "global_budget_used": Base.global_budget_used,
            "global_max_budget": DEFAULT_CONFIG.agent.global_max_budget,
            "tokens_used": self.total_tokens_used,
            "max_tokens": max_tokens,
            "step_count": self.step_count,
            "max_steps": getattr(self, "max_steps", DEFAULT_CONFIG.agent.max_steps),
            "command": " ".join(sys.argv),
        }

    def _save(self) -> None:
        """Save the agent's state to a file."""
        folder_path = Path(DEFAULT_CONFIG.agent.artifact_dir) / "trajectories"
        folder_path.mkdir(parents=True, exist_ok=True)
        name_safe = self.name.replace(" ", "_").replace("/", "_")
        filename = folder_path / f"trajectory_{name_safe}_{self.id}_{self.run_start_timestamp}.yaml"
        with filename.open("w", encoding="utf-8") as f:
            yaml.dump(self._build_state_dict(), f, indent=2)

    def get_trajectory(self) -> str:
        """Return the trajectory as JSON for visualization."""
        return json.dumps(self.messages, indent=2)

    def _add_message(self, role: str, content: Any, timestamp: int | None = None) -> None:
        """Add a message to the history."""
        self.messages.append(
            {
                "unique_id": len(self.messages),
                "role": role,
                "content": content,
                "timestamp": timestamp if timestamp is not None else int(time.time()),
            }
        )
