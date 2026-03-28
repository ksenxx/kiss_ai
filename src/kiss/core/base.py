# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Base agent class with common functionality for all KISS agents."""

import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, ClassVar

import yaml
from yaml.nodes import ScalarNode

from kiss.core import config as config_module
from kiss.core.models.model_info import get_max_context_length
from kiss.core.printer import Printer
from kiss.core.utils import config_to_dict

logger = logging.getLogger(__name__)

def _str_presenter(dumper: yaml.Dumper, data: str) -> ScalarNode:
    """Use literal block style for multiline strings in YAML output."""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")  # type: ignore[reportUnknownMemberType]


yaml.add_representer(str, _str_presenter)

_project_dir = Path(__file__).parent.parent.parent.parent
SYSTEM_PROMPT = (_project_dir / "SYSTEM.md").read_text()

_sorcar_path = Path("SORCAR.md")
if _sorcar_path.exists():
    SYSTEM_PROMPT += "\n" + _sorcar_path.read_text()


class Base:
    """Base class for all KISS agents with common state management and persistence."""

    agent_counter: ClassVar[int] = 1
    global_budget_used: ClassVar[float] = 0.0
    _class_lock: ClassVar[threading.Lock] = threading.Lock()

    model_name: str
    messages: list[dict[str, Any]]
    function_map: dict[str, Any]
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
        with Base._class_lock:
            self.id = Base.agent_counter
            Base.agent_counter += 1
        self.base_dir = ""
        self.printer: Printer | None = None
        self.model_name = ""
        self.messages: list[dict[str, Any]] = []
        self.function_map = {}
        self.run_start_timestamp = 0
        self.budget_used = 0.0
        self.total_tokens_used = 0
        self.step_count = 0

    def set_printer(
        self,
        printer: Printer | None = None,
        verbose: bool | None = None,
    ) -> None:
        """Configure the output printer for this agent.

        If an explicit *printer* is provided, it is always used regardless
        of the verbose setting.  Otherwise a ``ConsolePrinter`` is created
        when verbose output is enabled.

        Args:
            printer: An existing Printer instance to use directly. If provided,
                verbose is ignored.
            verbose: Whether to print to the console. Defaults to True if None.
        """
        if printer:
            self.printer = printer
        elif verbose is not False:
            from kiss.core.print_to_console import ConsolePrinter

            self.printer = ConsolePrinter()
        else:
            self.printer = None

    def _build_state_dict(self) -> dict[str, Any]:
        """Build state dictionary for saving.

        Returns:
            dict[str, Any]: A dictionary containing all agent state for persistence.
        """
        try:
            max_tokens = get_max_context_length(self.model_name)
        except Exception:
            logger.debug("Exception caught", exc_info=True)
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
            "total_budget": getattr(self, "max_budget", 10.0),
            "global_budget_used": Base.global_budget_used,
            "global_max_budget": 200.0,
            "tokens_used": self.total_tokens_used,
            "max_tokens": max_tokens,
            "step_count": self.step_count,
            "max_steps": getattr(self, "max_steps", 100),
            "command": " ".join(sys.argv),
        }

    def _save(self) -> None:
        """Save the agent's state to a YAML file in the artifacts directory.

        The file is saved to {artifact_dir}/trajectories/trajectory_{name}_{id}_{timestamp}.yaml
        """
        folder_path = Path(config_module.artifact_dir) / "trajectories"
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
