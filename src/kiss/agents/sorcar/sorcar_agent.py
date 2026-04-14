"""Sorcar agent with both coding tools and browser automation."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar.persistence import _load_last_model, _save_last_model
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.core.base import SYSTEM_PROMPT
from kiss.core.models.model import Attachment
from kiss.core.printer import Printer
from kiss.core.relentless_agent import RelentlessAgent


class SorcarAgent(RelentlessAgent):
    """Agent with both coding tools and browser automation for web + code tasks."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.web_use_tool: WebUseTool | None = None
        self.docker_manager: Any = None
        self._use_web_tools: bool = True
        self._is_parallel: bool = False

    def _get_tools(self) -> list:
        """Build tool list, using DockerTools when docker_manager is active.

        Must be called after docker_manager is set up (i.e., from perform_task,
        not from run() before super().run()).
        """
        def _stream(text: str) -> None:
            if self.printer:
                self.printer.print(text, type="bash_stream")

        def ask_user_question(question: str) -> str:
            """Ask the user a question and wait for their typed response.

            Use when the agent needs clarification, confirmation, or additional
            information from the user in the middle of a task. The user sees
            the question in the chat window, types their answer, and clicks
            "I'm Done". The agent blocks until the answer is provided.

            Args:
                question: The question to display to the user.

            Returns:
                The user's typed response text.
            """
            ask_callback = getattr(self, "_ask_user_question_callback", None)
            if ask_callback:
                return str(ask_callback(question))
            return "(ask_user_question not available in this environment)"

        stop_event = getattr(self, "_stop_event", None)
        useful_tools = UsefulTools(stream_callback=_stream, stop_event=stop_event)
        if self.docker_manager:
            from kiss.docker.docker_tools import DockerTools

            docker_tools = DockerTools(self._docker_bash)
            tools: list = [
                self._docker_bash, docker_tools.Read, docker_tools.Edit, docker_tools.Write,
            ]
        else:
            tools = [useful_tools.Bash, useful_tools.Read, useful_tools.Edit, useful_tools.Write]
        if self._use_web_tools and self.web_use_tool is None:
            self.web_use_tool = WebUseTool()
            tools.extend(self.web_use_tool.get_tools())
        def run_parallel(tasks: list[str], max_workers: int | None = None) -> str:
            """Run multiple independent tasks concurrently using parallel agents.

            Spawns a separate SorcarAgent for each task string and executes
            them in parallel threads.  Use this tool when you have two or
            more **independent** sub-tasks that do not depend on each
            other's results (e.g. summarising several files, researching
            separate topics, running independent code changes).

            Each parallel agent inherits the current model and working
            directory.  Results are returned in the same order as the
            input tasks.

            **When NOT to use**: Do not use this for tasks that must run
            sequentially or that depend on each other's output.

            Args:
                tasks: List of task description strings.  Each string is a
                    complete, self-contained instruction that a fresh
                    SorcarAgent can execute on its own.  Example::

                        [
                            "Read src/foo.py and summarize its purpose",
                            "Read src/bar.py and summarize its purpose",
                            "Find the current weather in San Francisco",
                        ]
                max_workers: Maximum number of concurrent threads.
                    ``None`` (default) lets Python choose automatically.
                    Set to a lower number to limit concurrency.

            Returns:
                A YAML-formatted string containing a list of result
                objects, one per task, in the same order as the input.
                Each result object has ``success`` and ``summary`` keys.
            """
            results = run_tasks_parallel(
                tasks,
                max_workers=max_workers,
                model=getattr(self, "model_name", None),
                work_dir=getattr(self, "work_dir", None),
            )
            result_str: str = yaml.dump(results, sort_keys=False)
            return result_str

        tools.append(ask_user_question)
        if self._is_parallel:
            tools.append(run_parallel)
        return tools

    def perform_task(
        self,
        tools: list,
        attachments: list | None = None,
    ) -> str:
        """Execute the task, building docker-aware tools after docker_manager is set.

        Args:
            tools: Extra tools passed by the caller (from run(tools=...)).
            attachments: Optional file attachments for the initial prompt.

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        all_tools = self._get_tools() + tools
        return super().perform_task(all_tools, attachments=attachments)

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
        resolved_model = model_name or _load_last_model() or "claude-opus-4-6"
        _save_last_model(resolved_model)
        super()._reset(
            model_name=resolved_model,
            max_sub_sessions=max_sub_sessions,
            max_steps=max_steps,
            max_budget=max_budget,
            work_dir=work_dir or ".",
            docker_image=docker_image,
            printer=printer,
            verbose=verbose if verbose is not None else False,
        )

    def run(  # type: ignore[override]
        self,
        model_name: str | None = None,
        prompt_template: str = "",
        arguments: dict[str, str] | None = None,
        system_prompt: str | None = None,
        tools: list[Callable[..., Any]] | None = None,
        max_steps: int | None = None,
        max_budget: float | None = None,
        model_config: dict[str, Any] | None = None,
        work_dir: str | None = None,
        printer: Printer | None = None,
        max_sub_sessions: int | None = None,
        docker_image: str | None = None,
        web_tools: bool = True,
        is_parallel: bool = False,
        verbose: bool | None = None,
        current_editor_file: str | None = None,
        attachments: list[Attachment] | None = None,
        ask_user_question_callback: Callable[[str], str] | None = None,
    ) -> str:
        """Run the assistant agent with coding tools and browser automation.

        Args:
            model_name: LLM model to use. Defaults to config value.
            prompt_template: Task prompt template with format placeholders.
            arguments: Dictionary of values to fill prompt_template placeholders.
            system_prompt: system prompt to be appended to the actual system prompt
            tools: List of tools to be added in addition to bash and web tools.
            max_steps: Maximum steps per sub-session. Defaults to config value.
            max_budget: Maximum budget in USD. Defaults to config value.
            work_dir: Working directory for the agent. Defaults to artifact_dir/kiss_workdir.
            printer: Printer instance for output display.
            max_sub_sessions: Maximum continuation sub-sessions. Defaults to config value.
            docker_image: Docker image name to run tools inside a container.
            web_tools: Whether to include browser/web tools. Defaults to True.
                Set to False for terminal-only environments.
            is_parallel: Whether to include the run_parallel tool. Defaults to False.
                When True, the agent can spawn parallel sub-agents for independent tasks.
            verbose: Whether to print output to console. Defaults to config verbose setting.
            current_editor_file: Path to the currently active editor file, appended to prompt.
            attachments: Optional file attachments (images, PDFs) for the initial prompt.
            ask_user_question_callback: Optional callback used by the ask_user_question
                tool to collect a text response from the user.

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        self._ask_user_question_callback = ask_user_question_callback
        self._use_web_tools = web_tools
        self._is_parallel = is_parallel
        # Lazy-initialized when web tools are first accessed via _get_tools()
        self.web_use_tool = None
        # Extract the per-thread stop event from the printer so UsefulTools
        # can monitor it and kill child processes when the agent is stopped.
        tl = getattr(printer, "_thread_local", None) if printer else None
        self._stop_event = getattr(tl, "stop_event", None) if tl else None

        try:
            system_instructions = (
                SYSTEM_PROMPT
                + (system_prompt if system_prompt else "")
            )
            prompt = prompt_template
            if attachments:
                pdf_count = sum(1 for a in attachments if a.mime_type == "application/pdf")
                img_count = sum(1 for a in attachments if a.mime_type.startswith("image/"))
                audio_count = sum(1 for a in attachments if a.mime_type.startswith("audio/"))
                video_count = sum(1 for a in attachments if a.mime_type.startswith("video/"))
                parts = []
                if img_count:
                    parts.append(f"{img_count} image(s)")
                if pdf_count:
                    parts.append(f"{pdf_count} PDF(s)")
                if audio_count:
                    parts.append(f"{audio_count} audio file(s)")
                if video_count:
                    parts.append(f"{video_count} video file(s)")
                if parts:
                    prompt += (
                        f"\n\n# Important\n - User attached {', '.join(parts)}. "
                        f"The files are included in this message as inline content "
                        f"that you can see directly. "
                        f"Do NOT launch a browser, call screenshot(), go_to_url(), "
                        f"or any other browser tool to view these attachments — "
                        f"you already have them."
                    )
            if current_editor_file:
                system_instructions += (
                    "\n\n- The path of the file open in the editor is "
                    f"{current_editor_file}"
                )
            return super().run(
                model_name=model_name,
                system_prompt=system_instructions,
                prompt_template=prompt,
                arguments=arguments,
                max_steps=max_steps,
                max_budget=max_budget,
                model_config=model_config,
                work_dir=work_dir,
                printer=printer,
                max_sub_sessions=max_sub_sessions,
                docker_image=docker_image,
                verbose=verbose,
                tools=tools or [],
                attachments=attachments,
            )
        finally:
            if self.web_use_tool:
                self.web_use_tool.close()
            self.web_use_tool = None
            self._ask_user_question_callback = None


def run_tasks_parallel(
    tasks: list[str],
    max_workers: int | None = None,
    model: str | None = None,
    work_dir: str | None = None,
) -> list[str]:
    """Execute multiple SorcarAgent tasks concurrently using threads.

    Each task gets its own ``SorcarAgent`` instance and runs in a separate
    thread via :class:`~concurrent.futures.ThreadPoolExecutor`.  This is
    ideal for I/O-bound workloads (LLM API calls, network requests) where
    the GIL is released during I/O waits.

    Args:
        tasks: List of task description strings.  Each string is passed as
            the ``prompt_template`` argument to :meth:`SorcarAgent.run`.
            Example::

                [
                    "Summarize file A",
                    "Summarize file B",
                ]
        max_workers: Maximum number of threads.  ``None`` lets
            :class:`~concurrent.futures.ThreadPoolExecutor` pick a default
            (typically ``min(32, cpu_count + 4)``).
        model: LLM model name for all parallel agents.  ``None`` uses the
            default from persistence (same as :meth:`SorcarAgent.run`).
        work_dir: Working directory for all parallel agents.  ``None`` uses
            the default (``artifact_dir/kiss_workdir``).

    Returns:
        List of YAML result strings in the **same order** as *tasks*.
        Each string contains ``success`` and ``summary`` keys.  If a task
        raises an unhandled exception the corresponding entry is a YAML
        string with ``success: false`` and the traceback in ``summary``.
    """

    def _run_single(task: str) -> str:
        agent = SorcarAgent(f"Parallel-{task[:40]}")
        try:
            result: str = agent.run(
                prompt_template=task,
                model_name=model,
                work_dir=work_dir,
            )
            return result
        except Exception as exc:
            error_result: str = yaml.dump(
                {"success": False, "summary": f"Unhandled exception: {exc}"},
                sort_keys=False,
            )
            return error_result

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_run_single, tasks))
    return results


_DEFAULT_TASK = """
can you find what the current weather is in San Francisco and summarize it?
"""


def _resolve_task(args: argparse.Namespace) -> str:
    """Determine the task description from parsed arguments.

    Priority: -f file > --task string > default task.

    Args:
        args: Parsed argparse namespace with 'f' and 'task' attributes.

    Returns:
        The task description string.

    Raises:
        FileNotFoundError: If -f path does not exist.
    """
    if args.file is not None:
        return Path(args.file).read_text()
    if args.task is not None:
        task: str = args.task
        return task
    return _DEFAULT_TASK


def cli_ask_user_question(question: str) -> str:
    """CLI callback for agent questions (prints and reads from stdin).

    Args:
        question: The question to display to the user.

    Returns:
        The user's typed response text.
    """
    print(f"\n>>> Agent asks: {question}")
    return input("Your answer: ")



