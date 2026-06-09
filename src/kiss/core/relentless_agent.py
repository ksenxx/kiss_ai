# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
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

The executor's trajectory is saved at: {trajectory_path}

Read relevant portions of the file using your tools:
- Read the first ~50 lines to understand the task and system instructions.
- Read the last ~200 lines to see the most recent steps and outcomes.
- Do NOT read the entire file; it may be very large.

# Instructions
- Analyze the trajectory file.
- Return a precise chronologically-ordered list of things the agent did
  with the reason for doing that along with relevant code snippets.
- Call finish(success=True, summary="detailed summary of work done so far").
"""

JUDGE_PROMPT = """
# Task Requirements

{task_description}

# Claimed Result

{executor_result}

# Executor Trajectory

The executor's trajectory is saved at: {trajectory_path}

Read relevant portions of the trajectory file using your tools:
- Read the first ~50 lines to understand the task and system instructions.
- Read the last ~200 lines to see the most recent steps and outcomes.
- Do NOT read the entire file; it may be very large.

# Instructions

You are a judge assessing whether the task was fully completed.
- Work dir: {work_dir}
- Use your tools to explore the work dir and verify actual outputs on disk.
- Do NOT redo any work; only inspect and assess.
- When done, call finish() with:
  - success=true if every requirement is met, false otherwise
  - summary: concise explanation of what is missing or why it passes
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

    # ``_reset`` assigns the resolved absolute path on every ``run()``;
    # this default makes ``self.work_dir`` safe to read on a freshly
    # constructed agent (mirroring ``Base.model_name``).
    work_dir: str = ""

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
        self.verbose = verbose
        self.budget_used: float = 0.0
        self.total_tokens_used: int = 0
        self.total_steps: int = 0
        self._current_executor: KISSAgent | None = None
        self.docker_image = docker_image
        self.docker_manager: Any = None
        self.task_description: str = ""
        self.system_prompt: str = ""
        self.model_config: dict[str, Any] | None = None
        # See :attr:`kiss.core.kiss_agent.KISSAgent.pre_step_hook`.
        # Propagated to every inner per-session executor created in
        # :meth:`perform_task`, so subclasses (e.g. ``SorcarAgent``)
        # can hook into model calls without re-implementing the
        # per-session executor.
        self.pre_step_hook: Callable[..., None] | None = None
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
        logger.info(
            "Executing task: agent=%s model=%s max_steps=%d "
            "max_budget=$%.2f pid=%d task=%r",
            self.name,
            self.model_name,
            self.max_steps,
            self.max_budget,
            os.getpid(),
            self.task_description[:200],
        )
        all_tools: list[Callable[..., Any]] = [finish, *tools]

        progress_section = ""
        summary = ""
        summaries: list[str] = []
        current_pid = str(os.getpid())
        # Report ``self.work_dir`` verbatim — in worktree mode this is the
        # real on-disk ``<repo>/.kiss-worktrees/<slug>/...`` path, matching
        # what ``pwd`` / ``os.getcwd()`` return inside the agent's tools.
        important_instructions = IMPORTANT_INSTRUCTIONS.format(
            step_threshold=str(self.max_steps - 2),
            work_dir=self.work_dir,
            current_pid=current_pid,
        )
        system_prompt = self.system_prompt + important_instructions
        for session in range(self.max_sub_sessions):
            remaining_budget = self.max_budget - self.budget_used
            if remaining_budget <= 0:
                raise KISSError(
                    f"Agent {self.name} budget exhausted "
                    f"(${self.budget_used:.4f} / ${self.max_budget:.2f})."
                )
            if self.printer:
                self.printer.tokens_offset = self.total_tokens_used  # type: ignore[attr-defined]
                self.printer.budget_offset = self.budget_used  # type: ignore[attr-defined]
                self.printer.steps_offset = self.total_steps  # type: ignore[attr-defined]
            logger.info(
                "Session %d start: agent=%s budget_remaining=$%.4f "
                "total_tokens=%d total_steps=%d",
                session,
                self.name,
                remaining_budget,
                self.total_tokens_used,
                self.total_steps,
            )
            executor = KISSAgent(f"{self.name} Session-{session}")
            # Propagate any pre-step hook installed on the relentless
            # parent (e.g. ``SorcarAgent.run``'s pending-user-messages
            # drain) to the per-session inner executor — that's the
            # layer that actually calls the model.
            executor.pre_step_hook = getattr(self, "pre_step_hook", None)
            self._current_executor = executor
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
                    max_budget=remaining_budget,
                    model_config=self.model_config,
                    printer=self.printer,
                    verbose=self.verbose,
                    attachments=attachments if session == 0 else None,
                )
            except Exception as exc:
                logger.debug("Exception caught", exc_info=True)
                if (
                    exc.__cause__ is not None
                    or not isinstance(exc, KISSError)
                    or executor.step_count <= 1
                ):
                    self._current_executor = None
                    self.budget_used += executor.budget_used
                    self.total_tokens_used += executor.total_tokens_used
                    self.total_steps += executor.step_count
                    error_result: str = yaml.dump(
                        {
                            "success": False,
                            "is_continue": False,
                            "summary": str(exc),
                        },
                        sort_keys=False,
                    )
                    if self.printer:
                        self.printer.print(
                            error_result,
                            type="result",
                            step_count=executor.step_count,
                            total_tokens=executor.total_tokens_used,
                            cost=f"${executor.budget_used:.4f}",
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
                    summarizer_budget = max(
                        0.01, self.max_budget - self.budget_used - executor.budget_used
                    )
                    summarizer_agent = KISSAgent(f"{self.name} Summarizer")
                    try:
                        summarizer_result = summarizer_agent.run(
                            model_name=self.model_name,
                            prompt_template=SUMMARIZER_PROMPT,
                            tools=[shell_tools.Read, shell_tools.Bash],
                            arguments={
                                # ``SUMMARIZER_PROMPT`` uses
                                # ``{trajectory_path}`` as the
                                # placeholder; the argument key MUST
                                # match or the literal placeholder
                                # leaks into the LLM prompt.
                                "trajectory_path": str(trajectory_path),
                            },
                            max_steps=self.max_steps,
                            max_budget=summarizer_budget,
                            # Inherit the parent's model routing
                            # (e.g. custom ``base_url``/``api_key``)
                            # so the summarizer talks to the same
                            # provider as the executor — otherwise it
                            # silently falls back to the default
                            # OpenAI client and its cost vanishes
                            # from this agent's accounting.
                            model_config=self.model_config,
                            # Inherit verbose/printer so the
                            # summarizer's streaming behaviour
                            # matches the parent's (no surprise
                            # ``ConsolePrinter`` when the parent
                            # asked for ``verbose=False``).
                            printer=self.printer,
                            verbose=self.verbose,
                        )
                    finally:
                        # The summarizer's spend MUST be folded into
                        # the parent's running totals BEFORE we leave
                        # this block — otherwise every model call the
                        # summarizer made (which costs real money on
                        # paid providers) silently disappears from
                        # ``self.budget_used`` and the user is told
                        # they spent less than they actually did.
                        # This must run even when ``summarizer_agent.run``
                        # raises (e.g. budget exceeded mid-summary) so
                        # the partial spend is still attributed.
                        self.budget_used += summarizer_agent.budget_used
                        self.total_tokens_used += (
                            summarizer_agent.total_tokens_used
                        )
                        self.total_steps += summarizer_agent.step_count
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

            self._current_executor = None
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
                if summaries:
                    final_summary = payload.get("summary", "")
                    # ``summaries`` holds every PRIOR session's summary
                    # (collected from ``is_continue=True`` returns).  Prepend
                    # them as historical context BEFORE the terminal session's
                    # summary, which stays the primary payload summary.
                    prior_section = "\n\n---\n\n".join(
                        f"### Previous Session {i + 1}\n{s}"
                        for i, s in enumerate(summaries)
                    )
                    if final_summary:
                        payload["summary"] = (
                            f"{prior_section}\n\n---\n\n### Final Session\n"
                            f"{final_summary}"
                        )
                    else:
                        payload["summary"] = prior_section
                    result = yaml.dump(payload, sort_keys=False)
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
