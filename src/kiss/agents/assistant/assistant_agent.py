"""Assistant agent with both coding tools and browser automation."""

from __future__ import annotations

import os
import tempfile

import yaml

import kiss.agents.assistant.config as _assistant_config  # noqa: F401
from kiss.agents.assistant.relentless_agent import RelentlessAgent
from kiss.core.printer import Printer
from kiss.core.useful_tools import UsefulTools
from kiss.core.web_use_tool import WebUseTool


class AssistantAgent(RelentlessAgent):
    """Agent with both coding tools and browser automation for web + code tasks."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.web_use_tool: WebUseTool | None = None

    def _get_tools(self) -> list:
        printer = self.printer

        def _stream(text: str) -> None:
            if printer:
                printer.print(text, type="bash_stream")

        stream_cb = _stream if printer else None
        useful_tools = UsefulTools(
            base_dir=self.base_dir,
            readable_paths=[str(p) for p in self.readable_paths],
            writable_paths=[str(p) for p in self.writable_paths],
            stream_callback=stream_cb,
        )
        bash_tool = self._docker_bash if self.docker_manager else useful_tools.Bash
        tools = [bash_tool, useful_tools.Read, useful_tools.Edit, useful_tools.Write]
        if self.web_use_tool:
            tools.extend(self.web_use_tool.get_tools())
        return tools

    def run(  # type: ignore[override]
        self,
        model_name: str | None = None,
        prompt_template: str = "",
        arguments: dict[str, str] | None = None,
        max_steps: int | None = None,
        max_budget: float | None = None,
        work_dir: str | None = None,
        base_dir: str | None = None,
        readable_paths: list[str] | None = None,
        writable_paths: list[str] | None = None,
        printer: Printer | None = None,
        max_sub_sessions: int | None = None,
        docker_image: str | None = None,
        headless: bool | None = None,
        print_to_console: bool | None = None,
        print_to_browser: bool | None = None,
    ) -> str:
        """Run the assistant agent with coding tools and browser automation.

        Args:
            model_name: LLM model to use. Defaults to config value.
            prompt_template: Task prompt template with format placeholders.
            arguments: Dictionary of values to fill prompt_template placeholders.
            max_steps: Maximum steps per sub-session. Defaults to config value.
            max_budget: Maximum budget in USD. Defaults to config value.
            work_dir: Working directory for the agent. Defaults to artifact_dir/kiss_workdir.
            base_dir: Base directory for path resolution. Defaults to work_dir.
            readable_paths: Additional paths the agent can read from.
            writable_paths: Additional paths the agent can write to.
            printer: Printer instance for output display.
            max_sub_sessions: Maximum continuation sub-sessions. Defaults to config value.
            docker_image: Docker image name to run tools inside a container.
            headless: Whether to run the browser in headless mode. Defaults to config value.
            print_to_console: Whether to print output to console.
            print_to_browser: Whether to print output to browser UI.

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        from kiss.core import config as config_module

        cfg = config_module.DEFAULT_CONFIG.assistant.assistant_agent
        actual_headless = headless if headless is not None else cfg.headless
        self.web_use_tool = WebUseTool(headless=actual_headless)

        try:
            return super().run(
                model_name=model_name,
                prompt_template=prompt_template,
                arguments=arguments,
                max_steps=max_steps,
                max_budget=max_budget,
                work_dir=work_dir,
                base_dir=base_dir,
                readable_paths=readable_paths,
                writable_paths=writable_paths,
                printer=printer,
                max_sub_sessions=max_sub_sessions,
                docker_image=docker_image,
                print_to_console=print_to_console,
                print_to_browser=print_to_browser,
                tools_factory=self._get_tools,
                config_path="assistant.assistant_agent",
            )
        finally:
            if self.web_use_tool:
                self.web_use_tool.close()


def main() -> None:
    """Run a demo of the AssistantAgent with a sample Gmail task."""
    import argparse
    import time as time_mod

    parser = argparse.ArgumentParser(description="Run AssistantAgent demo")
    parser.add_argument(
        "--model_name", type=str, default="claude-sonnet-4-6", help="LLM model name"
    )
    parser.add_argument("--max_steps", type=int, default=30, help="Maximum number of steps")
    parser.add_argument("--max_budget", type=float, default=5.0, help="Maximum budget in USD")
    parser.add_argument("--work_dir", type=str, default=None, help="Working directory")
    parser.add_argument(
        "--headless",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Run browser headless (true/false)",
    )
    parser.add_argument(
        "--print_to_console",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Print output to console",
    )
    parser.add_argument(
        "--print_to_browser",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Print output to browser UI",
    )
    parser.add_argument("--task", type=str, default=None, help="Prompt template/task description")

    args = parser.parse_args()

    # Simple prompt default if not provided
    if args.task is not None:
        task_description = args.task
    else:
        task_description = """
can you login to gmail using the username 'kissagent1@gmail.com' and
password 'For AI Assistant.' and read the messages.
"""

    if args.work_dir is not None:
        work_dir = args.work_dir
        os.makedirs(work_dir, exist_ok=True)
    else:
        work_dir = tempfile.mkdtemp()
    agent = AssistantAgent("Assistant Agent Test")
    old_cwd = os.getcwd()
    os.chdir(work_dir)
    start_time = time_mod.time()
    try:
        result = agent.run(
            prompt_template=task_description,
            model_name=args.model_name,
            max_steps=args.max_steps,
            max_budget=args.max_budget,
            work_dir=args.work_dir,
            headless=args.headless,
            print_to_browser=args.print_to_browser,
            print_to_console=args.print_to_console,
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
