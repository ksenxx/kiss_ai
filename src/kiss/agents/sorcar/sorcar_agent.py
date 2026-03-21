"""Sorcar agent with both coding tools and browser automation."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar.task_history import _load_last_model, _save_last_model
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.core import config as config_module
from kiss.core.base import SYSTEM_PROMPT
from kiss.core.models.model import Attachment
from kiss.core.printer import Printer
from kiss.core.relentless_agent import RelentlessAgent
from kiss.docker.docker_manager import DockerManager


class SorcarAgent(RelentlessAgent):
    """Agent with both coding tools and browser automation for web + code tasks."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.web_use_tool: WebUseTool | None = None
        self.docker_manager: DockerManager | None = None

    def _get_tools(self) -> list:
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
        bash_tool = self._docker_bash if self.docker_manager else useful_tools.Bash
        tools = [bash_tool, useful_tools.Read, useful_tools.Edit, useful_tools.Write]
        if self.web_use_tool:
            tools.extend(self.web_use_tool.get_tools())
        tools.append(ask_user_question)
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
        resolved_model = model_name or _load_last_model() or cfg.model_name
        _save_last_model(resolved_model)
        super()._reset(
            model_name=resolved_model,
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
        system_prompt: str | None = None,
        tools: list[Callable[..., Any]] | None = None,
        max_steps: int | None = None,
        max_budget: float | None = None,
        model_config: dict[str, Any] | None = None,
        work_dir: str | None = None,
        printer: Printer | None = None,
        max_sub_sessions: int | None = None,
        docker_image: str | None = None,
        headless: bool | None = None,
        verbose: bool | None = None,
        current_editor_file: str | None = None,
        attachments: list[Attachment] | None = None,
        wait_for_user_callback: Callable[[str, str], None] | None = None,
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
            headless: Deprecated, ignored. Browser always runs headless.
            verbose: Whether to print output to console. Defaults to config verbose setting.
            current_editor_file: Path to the currently active editor file, appended to prompt.
            attachments: Optional file attachments (images, PDFs) for the initial prompt.
            wait_for_user_callback: Optional callback used by browser tools when user
                action is required.
            ask_user_question_callback: Optional callback used by the ask_user_question
                tool to collect a text response from the user.

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        self._wait_for_user_callback = wait_for_user_callback
        self._ask_user_question_callback = ask_user_question_callback
        self.web_use_tool = WebUseTool(wait_for_user_callback=wait_for_user_callback)
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
                parts = []
                if img_count:
                    parts.append(f"{img_count} image(s)")
                if pdf_count:
                    parts.append(f"{pdf_count} PDF(s)")
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
                tools=self._get_tools() + (tools if tools else []),
                attachments=attachments,
            )
        finally:
            if self.web_use_tool:
                self.web_use_tool.close()
            self.web_use_tool = None
            self._wait_for_user_callback = None
            self._ask_user_question_callback = None


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for main().

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="Run SorcarAgent demo")
    parser.add_argument(
        "-m", "--model_name", type=str, default="claude-opus-4-6", help="LLM model name"
    )
    parser.add_argument(
        "-e", "--endpoint", type=str, default=None, help="Custom endpoint for local model"
    )
    parser.add_argument(
        "-b", "--max_budget", type=float, default=100.0, help="Maximum budget in USD"
    )
    parser.add_argument("-w", "--work_dir", type=str, default=None, help="Working directory")
    parser.add_argument(
        "--headless",
        type=lambda x: str(x).lower() == "true",
        default=False,
        help="Run browser headless (true/false)",
    )
    parser.add_argument(
        "-v", "--verbose",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="Print output to console",
    )
    parser.add_argument(
        "-t", "--task", type=str, default=None, help="Prompt template/task description"
    )
    parser.add_argument(
        "-f", "--file", type=str, default=None,
        help="Path to a file whose contents to use as the task",
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
    if args.file is not None:
        return Path(args.file).read_text()
    if args.task is not None:
        task: str = args.task
        return task
    return _DEFAULT_TASK


def cli_wait_for_user(instruction: str, url: str) -> None:
    """CLI callback for browser-action prompts (prints and waits for Enter).

    Args:
        instruction: What the user should do.
        url: Current browser URL (printed if non-empty).
    """
    print(f"\n>>> Browser action needed: {instruction}")
    if url:
        print(f"    Current URL: {url}")
    input("Press Enter when done... ")


def cli_ask_user_question(question: str) -> str:
    """CLI callback for agent questions (prints and reads from stdin).

    Args:
        question: The question to display to the user.

    Returns:
        The user's typed response text.
    """
    print(f"\n>>> Agent asks: {question}")
    return input("Your answer: ")


def main() -> None:  # pragma: no cover – CLI entry point requires API
    """Run a demo of the SorcarAgent with a sample Gmail task."""
    import time as time_mod


    if len(sys.argv) < 1:
        print("Usage: sorcar_agent.py [-m MODEL_NAME] [-e ENDPOINT] [-b MAX_BUDGET] "
              "[-w WORK_DIR] [--headless true/false] [-v true/false] "
              "[-t TASK_DESCRIPTION] [-f TASK_FILE]")
        sys.exit(1)
    parser = _build_arg_parser()
    done = False
    agent = SorcarAgent("Sorcar Agent")
    try:
        args = parser.parse_args()
    except Exception:
        task_description = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
        work_dir = tempfile.mkdtemp()
        old_cwd = os.getcwd()
        os.chdir(work_dir)
        start_time = time_mod.time()
        try:
            result = agent.run(
                prompt_template=task_description,
                work_dir=work_dir,
                wait_for_user_callback=cli_wait_for_user,
                ask_user_question_callback=cli_ask_user_question,
            )
        finally:
            os.chdir(old_cwd)
        elapsed = time_mod.time() - start_time
        done = True
    if not done:
        task_description = _resolve_task(args)

        if args.work_dir is not None:
            work_dir = args.work_dir
            Path(work_dir).mkdir(parents=True, exist_ok=True)
        else:
            work_dir = tempfile.mkdtemp()
        model_config = {}
        if args.endpoint:
            model_config["base_url"] = args.endpoint

        old_cwd = os.getcwd()
        os.chdir(work_dir)
        start_time = time_mod.time()
        try:
            result = agent.run(
                prompt_template=task_description,
                model_name=args.model_name,
                max_budget=args.max_budget,
                model_config=model_config,
                work_dir=work_dir,
                headless=args.headless,
                verbose=args.verbose,
                wait_for_user_callback=cli_wait_for_user,
                ask_user_question_callback=cli_ask_user_question,
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
