"""Base relentless agent with smart continuation for long tasks."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core import config as config_module
from kiss.core.base import Base
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.core.models.model import Attachment
from kiss.core.printer import Printer

logger = logging.getLogger(__name__)

TASK_PROMPT = """
{task_description}

{previous_progress}
"""

IMPORTANT_INSTRUCTIONS = """
# MOST IMPORTANT INSTRUCTIONS
- **At step {step_threshold}: you MUST call finish(success=False, is_continue=True, \
summary="precise chronologically-ordered list of things the agent did \
with the reason for doing that along with relevant code snippets")** or \
if the task is not complete and you are at risk of running out of steps or context length.
- Work dir: {work_dir}
- Current process PID: {current_pid} — NEVER kill this process.
"""

CONTINUATION_PROMPT = """
# Task Progress (Continuation {continuation_number})

{progress_text}

# Continue
- Complete the rest of the task.
- **DON'T** redo completed work.
- If you have been retrying the same approach without progress, step back \
and rethink the strategy from scratch.
"""

SUMMARIZER_PROMPT = """
# Summarizer

The trajectory of the agent is stored in the file: {trajectory_file}

# Instructions
- Read the trajectory file and analyze it.  The trajectory file could be large.
- Return a precise chronologically-ordered list of things the agent did
  with the reason for doing that along with relevant code snippets
"""

def _str_to_bool(value: str | bool) -> bool:
    """Coerce a string or bool to a Python bool.

    Args:
        value: A string ("true", "1", "yes" → True; anything else → False)
            or an already-boolean value.

    Returns:
        The boolean interpretation of *value*.
    """
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


def finish(success: bool, is_continue: bool = False, summary: str = "") -> str:
    """Finish execution with status and summary.

    Args:
        success: True if the agent has successfully completed the task, False otherwise
        is_continue: True if the task is incomplete and should continue, False otherwise
        summary: precise chronologically-ordered list of things the
            agent did with the reason for doing that along with
            relevant code snippets
    """
    result: str = yaml.dump(
        {
            "success": _str_to_bool(success),
            "is_continue": _str_to_bool(is_continue),
            "summary": summary,
        },
        sort_keys=False,
    )
    return result


class RelentlessAgent(Base):
    """Base agent with auto-continuation for long tasks."""

    def _reset(
        self,
        model_name: str | None,
        max_sub_sessions: int | None,
        max_steps: int | None,
        max_budget: float | None,
        work_dir: str | None,
        docker_image: str | None,
        printer: Printer | None = None,
        verbose: bool | None = None,
    ) -> None:
        default_work_dir = str(Path(config_module.artifact_dir).resolve() / "kiss_workdir")

        self.work_dir = str(Path(work_dir or default_work_dir).resolve())
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)

        self.max_sub_sessions = max_sub_sessions if max_sub_sessions is not None else 10000
        self.max_steps = max_steps if max_steps is not None else 100
        self.max_budget = max_budget if max_budget is not None else 200.0
        self.model_name = model_name if model_name is not None else "claude-opus-4-6"
        self.budget_used: float = 0.0
        self.total_tokens_used: int = 0
        self.total_steps: int = 0
        self.docker_image = docker_image
        self.docker_manager: Any = None
        self.task_description: str = ""
        self.system_prompt: str = ""
        self.model_config: dict[str, Any] | None = None
        self.set_printer(printer, verbose=verbose)

    def _docker_bash(self, command: str, description: str) -> str:
        if self.docker_manager is None:
            raise KISSError("Docker manager not initialized")
        return str(self.docker_manager.Bash(command, description))

    def perform_task(
        self,
        tools: list[Callable[..., Any]],
        attachments: list[Attachment] | None = None,
    ) -> str:
        """Execute the task with auto-continuation across multiple sub-sessions.

        Args:
            tools: List of callable tools available to the agent during execution.
            attachments: Optional file attachments (images, PDFs) for the initial prompt.

        Returns:
            YAML string with 'success' and 'summary' keys on successful completion.

        Raises:
            KISSError: If the task fails after exhausting all sub-sessions.
        """
        logger.info("Executing task: %s", self.task_description)
        all_tools: list[Callable[..., Any]] = [finish, *tools]

        progress_section = ""
        summary = ""
        summaries: list[str] = []
        current_pid = str(os.getpid())
        important_instructions = IMPORTANT_INSTRUCTIONS.format(
            step_threshold=str(self.max_steps - 2),
            work_dir=self.work_dir,
            current_pid=current_pid,
        )
        system_prompt = self.system_prompt + important_instructions
        for session in range(self.max_sub_sessions):
            if self.printer:
                self.printer.tokens_offset = self.total_tokens_used  # type: ignore[attr-defined]
                self.printer.budget_offset = self.budget_used  # type: ignore[attr-defined]
                self.printer.steps_offset = self.total_steps  # type: ignore[attr-defined]
            executor = KISSAgent(f"{self.name} Session-{session}")
            try:
                result = executor.run(
                    model_name=self.model_name,
                    prompt_template=TASK_PROMPT,
                    arguments={
                        "task_description": self.task_description,
                        "previous_progress": progress_section,
                    },
                    system_prompt=system_prompt,
                    tools=all_tools,
                    max_steps=self.max_steps,
                    max_budget=self.max_budget,
                    model_config=self.model_config,
                    printer=self.printer,
                    attachments=attachments if session == 0 else None,
                )
            except Exception as exc:
                logger.debug("Exception caught", exc_info=True)
                if (
                    exc.__cause__ is not None
                    or not isinstance(exc, KISSError)
                    or executor.step_count == 0
                ):
                    self.budget_used += executor.budget_used
                    self.total_tokens_used += executor.total_tokens_used
                    self.total_steps += executor.step_count
                    error_result: str = yaml.dump(
                        {"success": False, "is_continue": False, "summary": str(exc)},
                        sort_keys=False,
                    )
                    return error_result
                trajectory_path: Path | None = None
                try:
                    tmp_dir = Path(self.work_dir) / "tmp"
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    trajectory_path = tmp_dir / f"trajectory_{session}.json"
                    trajectory_path.write_text(executor.get_trajectory())
                    _stop_ev = getattr(self.printer, "stop_event", None) if self.printer else None
                    shell_tools = UsefulTools(stop_event=_stop_ev)
                    summarizer_agent = KISSAgent(f"{self.name} Summarizer")
                    summarizer_result = summarizer_agent.run(
                        model_name=self.model_name,
                        prompt_template=SUMMARIZER_PROMPT,
                        tools=[shell_tools.Read, shell_tools.Bash],
                        arguments={
                            "trajectory_file": str(trajectory_path),
                        },
                        max_steps=self.max_steps,
                        max_budget=self.max_budget,
                    )
                    try:
                        parsed = yaml.safe_load(summarizer_result)
                        summary_text = (
                            parsed.get("result", summarizer_result)
                            if isinstance(parsed, dict)
                            else summarizer_result
                        )
                    except Exception:  # pragma: no cover
                        logger.debug("Exception caught", exc_info=True)
                        summary_text = summarizer_result
                except Exception:  # pragma: no cover – requires summarizer LLM failure
                    logger.debug("Exception caught", exc_info=True)
                    summary_text = f"Agent failed: {exc}"
                finally:
                    if trajectory_path and trajectory_path.exists():  # pragma: no branch
                        trajectory_path.unlink()
                result = yaml.dump(
                    {"success": False, "is_continue": True, "summary": summary_text},
                    sort_keys=False,
                )

            self.budget_used += executor.budget_used
            self.total_tokens_used += executor.total_tokens_used
            self.total_steps += executor.step_count

            try:
                payload = yaml.safe_load(result)
            except Exception:  # pragma: no cover
                logger.debug("Exception caught", exc_info=True)
                payload = {}
            if not isinstance(payload, dict):  # pragma: no cover
                payload = {}

            success = _str_to_bool(payload.get("success", False))
            is_continue = _str_to_bool(payload.get("is_continue", False))

            if not is_continue or success:
                return result

            summary = payload.get("summary", "")
            if summary:  # pragma: no branch
                summaries.append(summary)

                all_summaries = "\n\n---\n\n".join(
                    f"### Attempt {i + 1}\n{s}"
                    for i, s in enumerate(summaries)
                )
                progress_section = CONTINUATION_PROMPT.format(
                    progress_text=all_summaries,
                    continuation_number=session + 1,
                )
        raise KISSError(f"Task failed after {self.max_sub_sessions} sub-sessions")

    def run(
        self,
        model_name: str | None = None,
        prompt_template: str = "",
        arguments: dict[str, str] | None = None,
        system_prompt: str = "",
        max_steps: int | None = None,
        max_budget: float | None = None,
        model_config: dict[str, Any] | None = None,
        work_dir: str | None = None,
        printer: Printer | None = None,
        max_sub_sessions: int | None = None,
        docker_image: str | None = None,
        verbose: bool | None = None,
        tools: list[Callable[..., Any]] | None = None,
        attachments: list[Attachment] | None = None,
    ) -> str:
        """Run the agent with the provided tools.

        Args:
            model_name: LLM model to use. Defaults to config value.
            prompt_template: Task prompt template with format placeholders.
            arguments: Dictionary of values to fill prompt_template placeholders.
            system_prompt: System-level instructions passed to the underlying LLM
                via model_config. Defaults to empty string (no system instructions).
            max_steps: Maximum steps per sub-session. Defaults to config value.
            max_budget: Maximum budget in USD. Defaults to config value.
            model_config: Optional dictionary of additional model configuration
                parameters (e.g. temperature, top_p). Defaults to None.
            work_dir: Working directory for the agent. Defaults to artifact_dir/kiss_workdir.
            printer: Printer instance for output display.
            max_sub_sessions: Maximum continuation sub-sessions. Defaults to config value.
            docker_image: Docker image name to run tools inside a container.
            verbose: Whether to print output to console. Defaults to True.
            tools: List of callable tools available to the agent during execution.
            attachments: Optional file attachments (images, PDFs) for the initial prompt.

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        self._reset(
            model_name,
            max_sub_sessions,
            max_steps,
            max_budget,
            work_dir,
            docker_image,
            printer,
            verbose,
        )
        self.system_prompt = system_prompt
        self.model_config = model_config
        args = arguments or {}
        self.task_description = prompt_template.format(**args) if args else prompt_template

        if self.docker_image:
            from kiss.docker.docker_manager import DockerManager

            with DockerManager(self.docker_image) as docker_mgr:
                self.docker_manager = docker_mgr
                if self.printer:
                    _printer = self.printer

                    def _docker_stream(text: str) -> None:
                        _printer.print(text, type="bash_stream")

                    docker_mgr.stream_callback = _docker_stream
                try:
                    return self.perform_task(tools or [], attachments=attachments)
                finally:
                    self.docker_manager = None
        return self.perform_task(tools or [], attachments=attachments)
