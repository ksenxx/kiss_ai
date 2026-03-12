"""Sorcar agent with both coding tools and browser automation."""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import yaml

from kiss.agents.sorcar.task_history import HISTORY_FILE
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.channels.slack_agent import SlackChannelAgent
from kiss.core import config as config_module
from kiss.core.base import SYSTEM_PROMPT
from kiss.core.models.model import Attachment
from kiss.core.printer import Printer
from kiss.core.relentless_agent import RelentlessAgent
from kiss.docker.docker_manager import DockerManager


class SorcarAgent(RelentlessAgent):
    """Agent with both coding tools and browser automation for web + code tasks."""

    def __init__(
        self,
        name: str,
        wait_for_user_callback: Callable[[str, str], None] | None = None,
        ask_user_question_callback: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__(name)
        self._wait_for_user_callback = wait_for_user_callback
        self._ask_user_question_callback = ask_user_question_callback
        self.web_use_tool: WebUseTool | None = None
        self.docker_manager: DockerManager | None = None
        self.slack_agent = SlackChannelAgent()

    def _get_tools(self) -> list:
        def _stream(text: str) -> None:
            if self.printer:
                self.printer.print(text, type="bash_stream")

        ask_callback = self._ask_user_question_callback

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
            if ask_callback:
                return ask_callback(question)
            return "(ask_user_question not available in this environment)"

        useful_tools = UsefulTools(stream_callback=_stream)
        bash_tool = self._docker_bash if self.docker_manager else useful_tools.Bash
        tools = [bash_tool, useful_tools.Read, useful_tools.Edit, useful_tools.Write]
        if self.web_use_tool:
            tools.extend(self.web_use_tool.get_tools())
        tools.append(ask_user_question)
        tools.extend(self.slack_agent.get_tools())
        return tools

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
        cfg = config_module.DEFAULT_CONFIG.sorcar.sorcar_agent
        super()._reset(
            model_name=model_name if model_name is not None else cfg.model_name,
            max_sub_sessions=(
                max_sub_sessions if max_sub_sessions is not None else cfg.max_sub_sessions
            ),
            max_steps=max_steps if max_steps is not None else cfg.max_steps,
            max_budget=max_budget if max_budget is not None else cfg.max_budget,
            work_dir=work_dir or ".",
            docker_image=docker_image,
            printer=printer,
            verbose=verbose if verbose is not None else cfg.verbose,
        )

    def run(  # type: ignore[override]
        self,
        model_name: str | None = None,
        prompt_template: str = "",
        arguments: dict[str, str] | None = None,
        max_steps: int | None = None,
        max_budget: float | None = None,
        work_dir: str | None = None,
        printer: Printer | None = None,
        max_sub_sessions: int | None = None,
        docker_image: str | None = None,
        headless: bool | None = None,
        verbose: bool | None = None,
        current_editor_file: str | None = None,
        attachments: list[Attachment] | None = None,
    ) -> str:
        """Run the assistant agent with coding tools and browser automation.

        Args:
            model_name: LLM model to use. Defaults to config value.
            prompt_template: Task prompt template with format placeholders.
            arguments: Dictionary of values to fill prompt_template placeholders.
            max_steps: Maximum steps per sub-session. Defaults to config value.
            max_budget: Maximum budget in USD. Defaults to config value.
            work_dir: Working directory for the agent. Defaults to artifact_dir/kiss_workdir.
            printer: Printer instance for output display.
            max_sub_sessions: Maximum continuation sub-sessions. Defaults to config value.
            docker_image: Docker image name to run tools inside a container.
            headless: Whether to run the browser in headless mode. Defaults to config value.
            verbose: Whether to print output to console. Defaults to config verbose setting.
            current_editor_file: Path to the currently active editor file, appended to prompt.
            attachments: Optional file attachments (images, PDFs) for the initial prompt.

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        cfg = config_module.DEFAULT_CONFIG.sorcar.sorcar_agent
        actual_headless = headless if headless is not None else cfg.headless
        self.web_use_tool = WebUseTool(
            headless=actual_headless,
            wait_for_user_callback=self._wait_for_user_callback,
        )

        try:
            system_instructions = SYSTEM_PROMPT + f"\nTask History File: {HISTORY_FILE}\n"
            prompt = prompt_template
            if attachments:
                pdf_count = sum(1 for a in attachments if a.mime_type == "application/pdf")
                img_count = sum(1 for a in attachments if a.mime_type.startswith("image/"))
                parts = []
                if img_count:
                    parts.append(f"{img_count} image(s)")
                if pdf_count:
                    parts.append(f"{pdf_count} PDF(s)")
                if parts:
                    prompt += (
                        f"\n\n# Important\n - User attached {', '.join(parts)}. "
                        f"The files are included in this message. "
                        f"Examine them directly — do NOT use browser tools "
                        f"to view or screenshot these attachments."
                    )
            if current_editor_file:
                prompt += (
                    "\n\n- The path of the file open in the editor is "
                    f"{current_editor_file}"
                )
            return super().run(
                model_name=model_name,
                system_instructions=system_instructions,
                prompt_template=prompt,
                arguments=arguments,
                max_steps=max_steps,
                max_budget=max_budget,
                work_dir=work_dir,
                printer=printer,
                max_sub_sessions=max_sub_sessions,
                docker_image=docker_image,
                verbose=verbose,
                tools=self._get_tools(),
                attachments=attachments,
            )
        finally:
            if self.web_use_tool:
                self.web_use_tool.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for main().

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="Run SorcarAgent demo")
    parser.add_argument(
        "--model_name", type=str, default="claude-opus-4-6", help="LLM model name"
    )
    parser.add_argument("--max_steps", type=int, default=30, help="Maximum number of steps")
    parser.add_argument("--max_budget", type=float, default=5.0, help="Maximum budget in USD")
    parser.add_argument("--work_dir", type=str, default=None, help="Working directory")
    parser.add_argument(
        "--headless",
        type=lambda x: str(x).lower() == "true",
        default=False,
        help="Run browser headless (true/false)",
    )
    parser.add_argument(
        "--verbose",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="Print output to console",
    )
    parser.add_argument("--task", type=str, default=None, help="Prompt template/task description")
    parser.add_argument(
        "-f", type=str, default=None, help="Path to a file whose contents to use as the task"
    )
    return parser


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
    if args.f is not None:
        return Path(args.f).read_text()
    if args.task is not None:
        task: str = args.task
        return task
    return _DEFAULT_TASK


def main() -> None:
    """Run a demo of the SorcarAgent with a sample Gmail task."""
    import time as time_mod

    parser = _build_arg_parser()
    args = parser.parse_args()
    task_description = _resolve_task(args)

    if args.work_dir is not None:
        work_dir = args.work_dir
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    else:
        work_dir = tempfile.mkdtemp()
    agent = SorcarAgent("Sorcar Agent Test")
    old_cwd = os.getcwd()
    os.chdir(work_dir)
    start_time = time_mod.time()
    try:
        result = agent.run(
            prompt_template=task_description,
            model_name=args.model_name,
            max_steps=args.max_steps,
            max_budget=args.max_budget,
            work_dir=work_dir,
            headless=args.headless,
            verbose=args.verbose,
        )
    finally:
        os.chdir(old_cwd)
    elapsed = time_mod.time() - start_time

    print("FINAL RESULT:")
    result_data = yaml.safe_load(result)
    print("Completed successfully: " + str(result_data["success"]))
    print(result_data["summary"])
    print("Work directory was: " + work_dir)
    print(f"Time: {elapsed:.1f}s")
    print(f"Cost: ${agent.budget_used:.4f}")
    print(f"Total tokens: {agent.total_tokens_used}")


if __name__ == "__main__":
    main()
