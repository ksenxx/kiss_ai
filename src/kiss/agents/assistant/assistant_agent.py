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
        useful_tools = UsefulTools(
            base_dir=self.base_dir,
            readable_paths=[str(p) for p in self.readable_paths],
            writable_paths=[str(p) for p in self.writable_paths],
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
        """Run the assistant agent."""
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
    import time as time_mod

    agent = AssistantAgent("Assistant Agent Test")
    task_description = """
can you login to gmail using the username 'kissagent1@gmail.com' and
password 'For AI Assistant.' and read the messages.
"""

    work_dir = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    os.chdir(work_dir)
    start_time = time_mod.time()
    try:
        result = agent.run(
            prompt_template=task_description,
            model_name="claude-sonnet-4-5",
            max_steps=30,
            max_budget=5.0,
            work_dir=work_dir,
            headless=False,
            print_to_browser=True,
            print_to_console=True,
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
