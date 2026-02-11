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
from kiss.core.utils import config_to_dict


def _str_presenter(dumper: yaml.Dumper, data: str) -> ScalarNode:
    """Use literal block style for multiline strings in YAML output."""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")  # type: ignore[reportUnknownMemberType]


yaml.add_representer(str, _str_presenter)


CODING_INSTRUCTIONS = """
## Code Style Guidelines
- Write simple, clean, and readable code with minimal indirection
- Avoid unnecessary object attributes and local variables and config variables
- No redundant abstractions or duplicate code and config code
- Each function should do one thing well
- Use clear, descriptive names
- NO need to write documentations or comments unless absolutely necessary
- Check and test the code you have written
"""

# ## Testing Requirements
# - Run lint and typecheckers and fix any lint and typecheck errors
# - Generate comprehensive tests so that you achieve 100% branch coverage
# - Tests MUST NOT use mocks, patches, or any form of test doubles
# - Integration tests are highly encouraged
# - You MUST not add tests that are redundant or duplicate of existing
#   tests or does not add new coverage over existing tests
# - Generate meaningful stress tests for the code if you are
#   optimizing the code for performance
# - Test with real inputs and verify real outputs
# - Test edge cases: empty inputs, None values, boundary conditions
# - Test error conditions with actual invalid inputs
# - Each test should be independent and verify actual behavior

# ## Code Structure
# - Main implementation code first
# - Test code in a separate section using unittest or pytest
# - Include a __main__ block to run tests
# - Do not use 'exit' for early termination, rather throw an exception.

# ## Use tools when you need to:
# - Look up API documentation or library usage
# - Find examples of similar implementations
# - Understand existing code in the project

# ## After you have implemented the task, aggresively and carefully simplify and clean up the code
#  - Remove unnessary object/struct attributes, variables, config variables
#  - Avoid object/struct attribute redirections
#  - Remove unnessary conditional checks
#  - Remove redundant and duplicate code
#  - Remove unnecessary comments
#  - Make sure that the code is still working correctly
#  - Simplify and clean up the test code



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

    def __init__(self, name: str) -> None:
        """Initialize a Base agent instance.

        Args:
            name: The name identifier for the agent.
        """
        self.name = name
        self.id = Base.agent_counter
        Base.agent_counter += 1
        self.base_dir = ""

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
