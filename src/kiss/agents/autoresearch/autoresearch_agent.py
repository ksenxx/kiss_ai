"""Autoresearch agent for autonomous ML experiment iteration.

Implements the autoresearch pattern from https://github.com/karpathy/autoresearch:
an agent reads program.md for instructions, then autonomously modifies train.py,
runs experiments, evaluates results, and iterates.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core import config as config_module
from kiss.core.base import SYSTEM_PROMPT
from kiss.core.printer import Printer
from kiss.core.relentless_agent import RelentlessAgent

_DEFAULT_PROGRAM = "program.md"


class AutoresearchAgent(RelentlessAgent):
    """Agent that autonomously runs ML experiments following a program file.

    Reads a program.md file for instructions, then iterates: modify code,
    run training, evaluate results, keep or discard changes. Provides
    Bash, Read, Edit, and Write tools for code manipulation and experiment
    execution.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _get_tools(self) -> list:
        def _stream(text: str) -> None:
            if self.printer:
                self.printer.print(text, type="bash_stream")

        useful_tools = UsefulTools(stream_callback=_stream)
        return [useful_tools.Bash, useful_tools.Read, useful_tools.Edit, useful_tools.Write]

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
        cfg = config_module.DEFAULT_CONFIG.autoresearch.autoresearch_agent
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
        verbose: bool | None = None,
        program_file: str | None = None,
    ) -> str:
        """Run the autoresearch agent.

        The agent reads the program file for instructions, then autonomously
        iterates: modify train.py, run experiments, evaluate, keep/discard.

        Args:
            model_name: LLM model to use. Defaults to config value.
            prompt_template: Task prompt. If empty, reads from program_file.
            arguments: Dictionary of values to fill prompt_template placeholders.
            max_steps: Maximum steps per sub-session. Defaults to config value.
            max_budget: Maximum budget in USD. Defaults to config value.
            work_dir: Working directory containing the autoresearch repo.
            printer: Printer instance for output display.
            max_sub_sessions: Maximum continuation sub-sessions.
            docker_image: Docker image name to run tools inside a container.
            verbose: Whether to print output to console.
            program_file: Path to program.md file. Defaults to program.md in work_dir.

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        actual_work_dir = work_dir or os.getcwd()
        task = prompt_template
        if not task:
            program_path = program_file or os.path.join(actual_work_dir, _DEFAULT_PROGRAM)
            task = Path(program_path).read_text()

        return super().run(
            model_name=model_name,
            system_instructions=SYSTEM_PROMPT,
            prompt_template=task,
            arguments=arguments,
            max_steps=max_steps,
            max_budget=max_budget,
            work_dir=actual_work_dir,
            printer=printer,
            max_sub_sessions=max_sub_sessions,
            docker_image=docker_image,
            verbose=verbose,
            tools=self._get_tools(),
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for main().

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="Run Autoresearch Agent")
    parser.add_argument(
        "--model_name", type=str, default="claude-opus-4-6", help="LLM model name"
    )
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum steps per session")
    parser.add_argument("--max_budget", type=float, default=200.0, help="Maximum budget in USD")
    parser.add_argument("--work_dir", type=str, default=None, help="Working directory")
    parser.add_argument(
        "--program", type=str, default=None, help="Path to program.md file"
    )
    parser.add_argument(
        "--verbose",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="Print output to console",
    )
    parser.add_argument("--task", type=str, default=None, help="Direct task description")
    return parser


def main() -> None:
    """Run the autoresearch agent from the command line."""
    import time as time_mod

    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.work_dir is not None:
        work_dir = args.work_dir
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    else:
        work_dir = os.getcwd()

    agent = AutoresearchAgent("Autoresearch")
    old_cwd = os.getcwd()
    os.chdir(work_dir)
    start_time = time_mod.time()
    try:
        result = agent.run(
            prompt_template=args.task or "",
            model_name=args.model_name,
            max_steps=args.max_steps,
            max_budget=args.max_budget,
            work_dir=work_dir,
            verbose=args.verbose,
            program_file=args.program,
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
