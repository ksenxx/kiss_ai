"""Base relentless agent with smart continuation for long tasks."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from kiss.core import config as config_module
from kiss.core.base import Base
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.core.printer import Printer
from kiss.docker.docker_manager import DockerManager

TASK_PROMPT = """# Task

{task_description}

# Rules
- Write() for new files. Edit() for small changes. Bash timeout_seconds=120 for long runs.
- Call finish(success=True, summary="detailed summary of what was accomplished") \
immediately when task is complete.
- If detailed-progress.md exists, it contains a detailed summary of the work done so far. \
Update it when you make large progress.
- At step {step_threshold}: finish(success=False, summary="detailed summary of work done so far")
- Work dir: {work_dir}
{previous_progress}
"""

CONTINUATION_PROMPT = """
# Task Progress

{progress_text}

# Continue
- Complete the rest of the task.
- Don't redo completed work.
"""


def finish(success: bool, summary: str) -> str:
    """Finish execution with status and summary.

    Args:
        success: True if successful, False otherwise.
        summary: Detailed summary of work done so far.
    """
    if isinstance(success, str):
        success = success.strip().lower() not in ("false", "0", "no", "")
    return str(yaml.dump({"success": bool(success), "summary": summary}, indent=2, sort_keys=False))


class RelentlessAgent(Base):
    """Base agent with auto-continuation for long tasks."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _reset(
        self,
        model_name: str | None,
        max_sub_sessions: int | None,
        max_steps: int | None,
        max_budget: float | None,
        work_dir: str | None,
        docker_image: str | None,
        config_path: str,
    ) -> None:
        global_cfg = config_module.DEFAULT_CONFIG
        cfg = global_cfg
        for part in config_path.split("."):
            cfg = getattr(cfg, part)
        default_work_dir = str(Path(global_cfg.agent.artifact_dir).resolve() / "kiss_workdir")

        self.work_dir = str(Path(work_dir or default_work_dir).resolve())
        self.base_dir = self.work_dir
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
        self.is_agentic = True

        self.max_sub_sessions = (
            max_sub_sessions if max_sub_sessions is not None else cfg.max_sub_sessions
        )
        self.max_steps = max_steps if max_steps is not None else cfg.max_steps
        self.max_budget = max_budget if max_budget is not None else cfg.max_budget
        self.model_name = model_name if model_name is not None else cfg.model_name
        self.budget_used: float = 0.0
        self.total_tokens_used: int = 0
        self.docker_image = docker_image
        self.docker_manager: DockerManager | None = None

    def _docker_bash(self, command: str, description: str) -> str:
        if self.docker_manager is None:
            raise KISSError("Docker manager not initialized")
        return self.docker_manager.Bash(command, description)

    def perform_task(self, tools: list[Callable[..., Any]]) -> str:
        """Execute the task with auto-continuation across multiple sub-sessions.

        Args:
            tools: List of callable tools available to the agent during execution.

        Returns:
            YAML string with 'success' and 'summary' keys on successful completion.

        Raises:
            KISSError: If the task fails after exhausting all sub-sessions.
        """
        print(f"Executing task: {self.task_description}")
        all_tools: list[Callable[..., Any]] = [finish, *tools]

        progress_section = ""
        summary = ""
        for trial in range(self.max_sub_sessions):
            executor = KISSAgent(f"{self.name} Trial-{trial}")
            try:
                model_config = {}
                if self.system_instructions:
                    model_config["system_instructions"] = self.system_instructions
                result = executor.run(
                    model_name=self.model_name,
                    prompt_template=TASK_PROMPT,
                    arguments={
                        "task_description": self.task_description,
                        "previous_progress": progress_section,
                        "step_threshold": str(self.max_steps - 2),
                        "work_dir": self.work_dir,
                    },
                    tools=all_tools,
                    max_steps=self.max_steps,
                    max_budget=self.max_budget,
                    model_config=model_config or None,
                    printer=self.printer,
                )
            except Exception as e:
                err_summary = f"{summary}\n# Failure\n- Failed with Error: {e}"
                result = yaml.dump(
                    {"success": False, "summary": err_summary},
                    sort_keys=False,
                )

            self.budget_used += executor.budget_used
            self.total_tokens_used += executor.total_tokens_used

            payload = yaml.safe_load(result)
            if not isinstance(payload, dict):
                payload = {}

            if payload.get("success", False):
                return result

            summary = payload.get("summary", "")
            if summary:
                progress_section = CONTINUATION_PROMPT.format(progress_text=summary)
        raise KISSError(f"Task failed after {self.max_sub_sessions} sub-sessions")

    def run(
        self,
        model_name: str | None = None,
        system_instructions: str = "",
        prompt_template: str = "",
        arguments: dict[str, str] | None = None,
        max_steps: int | None = None,
        max_budget: float | None = None,
        work_dir: str | None = None,
        printer: Printer | None = None,
        max_sub_sessions: int | None = None,
        docker_image: str | None = None,
        print_to_console: bool | None = None,
        print_to_browser: bool | None = None,
        tools_factory: Callable[[], list[Callable[..., Any]]] | None = None,
        config_path: str = "agent",
    ) -> str:
        """Run the agent with tools created by tools_factory (called after _reset).

        Args:
            model_name: LLM model to use. Defaults to config value.
            system_instructions: System-level instructions passed to the underlying LLM
                via model_config. Defaults to empty string (no system instructions).
            prompt_template: Task prompt template with format placeholders.
            arguments: Dictionary of values to fill prompt_template placeholders.
            max_steps: Maximum steps per sub-session. Defaults to config value.
            max_budget: Maximum budget in USD. Defaults to config value.
            work_dir: Working directory for the agent. Defaults to artifact_dir/kiss_workdir.
            printer: Printer instance for output display.
            max_sub_sessions: Maximum continuation sub-sessions. Defaults to config value.
            docker_image: Docker image name to run tools inside a container.
            print_to_console: Whether to print output to console.
            print_to_browser: Whether to print output to browser UI.
            tools_factory: Callable that returns the list of tools for the agent.
            config_path: Dot-separated path to config section (e.g. "agent").

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        self._reset(
            model_name, max_sub_sessions, max_steps, max_budget,
            work_dir, docker_image, config_path,
        )
        self.system_instructions = system_instructions
        self.prompt_template = prompt_template
        self.arguments = arguments or {}
        self.task_description = prompt_template.format(**self.arguments)
        self.set_printer(
            printer, print_to_console=print_to_console, print_to_browser=print_to_browser,
        )

        tools = tools_factory() if tools_factory else []

        if self.docker_image:
            with DockerManager(self.docker_image) as docker_mgr:
                self.docker_manager = docker_mgr
                if self.printer:
                    _printer = self.printer

                    def _docker_stream(text: str) -> None:
                        _printer.print(text, type="bash_stream")

                    docker_mgr.stream_callback = _docker_stream
                try:
                    return self.perform_task(tools)
                finally:
                    self.docker_manager = None
        return self.perform_task(tools)
