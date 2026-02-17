"""Assistant agent with both coding tools and browser automation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import yaml

import kiss.agents.assistant.config as _assistant_config  # noqa: F401
from kiss.core import config as config_module
from kiss.core.base import Base
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.core.models.model_info import get_max_context_length
from kiss.core.printer import Printer
from kiss.core.useful_tools import UsefulTools
from kiss.core.utils import resolve_path
from kiss.core.web_use_tool import WebUseTool
from kiss.docker.docker_manager import DockerManager

TASK_PROMPT = """# Task
{task_description}

# Tools

## Code: Bash(command, description), Read(file_path), Write(file_path, content)
## Code: Edit(file_path, old_string, new_string)

## Browser
- go_to_url(url): Navigate. Also "tab:list" to list tabs, "tab:N" to switch tab.
- click(element_id, action="click"): Click or hover (action="hover") on element [N].
- type_text(element_id, text, press_enter=False): Type into input [N].
- press_key(key): Press key, e.g. "Escape", "Tab", "PageDown", "Control+a".
- execute_js(code): Run JavaScript. Use for select options, scroll, go back, extract data.
- screenshot(file_path): Save screenshot.
- get_page_content(text_only=False): Get DOM tree, or full text if text_only=True.

DOM tree shows interactive elements as [N] <tag>. Use N with click/type_text.

# Rules
- Write() for new files. Edit() for small changes. Bash timeout_seconds=120 for long runs.
- Call finish(success=True, summary="done") immediately when task is complete.
- At step {step_threshold}: finish(success=False, summary={{"done":[...], "next":[...]}})
- Work dir: {work_dir}
{previous_progress}"""

CONTINUATION_PROMPT = """# CONTINUATION

{progress_text}

Fix remaining issues then call finish. Don't redo completed work."""


def finish(success: bool, summary: str) -> str:
    """Finish execution with status and summary.

    Args:
        success: True if successful, False otherwise.
        summary: Summary of work done and remaining work (JSON for continuation).
    """
    if isinstance(success, str):
        success = success.strip().lower() not in ("false", "0", "no", "")
    return yaml.dump({"success": bool(success), "summary": summary}, indent=2, sort_keys=False)


class AssistantAgent(Base):
    """Agent with both coding tools and browser automation for web + code tasks."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _reset(
        self,
        model_name: str | None,
        max_sub_sessions: int | None,
        max_steps: int | None,
        max_budget: float | None,
        work_dir: str | None,
        base_dir: str | None,
        readable_paths: list[str] | None,
        writable_paths: list[str] | None,
        docker_image: str | None,
        headless: bool | None,
    ) -> None:
        global_cfg = config_module.DEFAULT_CONFIG
        cfg = global_cfg.assistant.assistant_agent
        default_work_dir = str(Path(global_cfg.agent.artifact_dir).resolve() / "kiss_workdir")

        actual_base_dir = base_dir if base_dir is not None else default_work_dir
        actual_work_dir = work_dir if work_dir is not None else default_work_dir

        Path(actual_base_dir).mkdir(parents=True, exist_ok=True)
        Path(actual_work_dir).mkdir(parents=True, exist_ok=True)
        self.base_dir = str(Path(actual_base_dir).resolve())
        self.work_dir = str(Path(actual_work_dir).resolve())
        self.readable_paths = [resolve_path(p, self.base_dir) for p in readable_paths or []]
        self.writable_paths = [resolve_path(p, self.base_dir) for p in writable_paths or []]
        self.readable_paths.append(Path(self.work_dir))
        self.writable_paths.append(Path(self.work_dir))
        self.is_agentic = True

        self.max_sub_sessions = (
            max_sub_sessions if max_sub_sessions is not None else cfg.max_sub_sessions
        )
        self.max_steps = max_steps if max_steps is not None else cfg.max_steps
        self.max_budget = max_budget if max_budget is not None else cfg.max_budget
        self.model_name = model_name if model_name is not None else cfg.model_name
        self.max_tokens = get_max_context_length(self.model_name)

        self.budget_used: float = 0.0
        self.total_tokens_used: int = 0

        self.docker_image = docker_image
        self.docker_manager: DockerManager | None = None
        self.headless = headless if headless is not None else cfg.headless

        self.useful_tools = UsefulTools(
            base_dir=self.base_dir,
            readable_paths=[str(p) for p in self.readable_paths],
            writable_paths=[str(p) for p in self.writable_paths],
        )

        self.web_use_tool = WebUseTool(headless=self.headless)

    def _docker_bash(self, command: str, description: str) -> str:
        if self.docker_manager is None:
            raise KISSError("Docker manager not initialized")
        return self.docker_manager.run_bash_command(command, description)

    def _parse_progress(self, summary: str) -> tuple[list[str], list[str]]:
        try:
            progress = json.loads(summary)
            done = progress.get("done", [])
            next_items = progress.get("next", [])
            if isinstance(done, list) and isinstance(next_items, list):
                done = list(dict.fromkeys(str(d) for d in done if d))
                next_items = list(dict.fromkeys(str(n) for n in next_items if n))
                return done, next_items
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        return [], []

    def _format_progress(self, done_items: list[str], next_items: list[str]) -> str:
        if not done_items:
            return ""
        progress = "## Done\n"
        for item in done_items[-10:]:
            progress += f"- {item}\n"
        if next_items:
            progress += "\n## TODO\n"
            for item in next_items:
                progress += f"- {item}\n"
        return progress

    def _build_continuation_section(
        self, done_items: list[str], next_items: list[str]
    ) -> str:
        progress_text = self._format_progress(done_items, next_items)
        return "\n\n" + CONTINUATION_PROMPT.format(
            progress_text=progress_text,
        )

    def perform_task(self) -> str:
        print(f"Executing task: {self.task_description}")
        bash_tool = self._docker_bash if self.docker_manager else self.useful_tools.Bash

        done_items: list[str] = []
        next_items: list[str] = []

        for trial in range(self.max_sub_sessions):
            step_threshold = self.max_steps - 2

            if trial == 0:
                progress_section = ""
            else:
                progress_section = self._build_continuation_section(
                    done_items, next_items
                )

            executor = KISSAgent(f"{self.name} Trial-{trial}")
            try:
                result = executor.run(
                    model_name=self.model_name,
                    prompt_template=TASK_PROMPT,
                    arguments={
                        "task_description": self.task_description,
                        "previous_progress": progress_section,
                        "step_threshold": str(step_threshold),
                        "work_dir": self.work_dir,
                    },
                    tools=[
                        finish,
                        bash_tool,
                        self.useful_tools.Read,
                        self.useful_tools.Edit,
                        self.useful_tools.Write,
                        *self.web_use_tool.get_tools(),
                    ],
                    max_steps=self.max_steps,
                    max_budget=self.max_budget,
                    printer=self.printer,
                )
            except Exception:
                last_msgs = executor.messages[-2:] if hasattr(executor, "messages") else []
                context = " ".join(
                    str(m.get("content", ""))
                    for m in last_msgs
                    if isinstance(m, dict)
                )
                result = yaml.dump(
                    {
                        "success": False,
                        "summary": json.dumps(
                            {"done": done_items, "next": [f"Continue: {context}"]}
                        ),
                    },
                    sort_keys=False,
                )

            self.budget_used += executor.budget_used
            self.total_tokens_used += executor.total_tokens_used

            ret = yaml.safe_load(result)
            payload = ret if isinstance(ret, dict) else {}

            if payload.get("success", False):
                return result

            summary = payload.get("summary", "")
            trial_done, trial_next = self._parse_progress(summary)

            if trial_done:
                for item in trial_done:
                    if item not in done_items:
                        done_items.append(item)
                next_items = trial_next
            elif summary and summary not in done_items:
                done_items.append(summary)

        raise KISSError(f"Task failed after {self.max_sub_sessions} sub-sessions")

    def run(
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
        self._reset(
            model_name,
            max_sub_sessions,
            max_steps,
            max_budget,
            work_dir,
            base_dir,
            readable_paths,
            writable_paths,
            docker_image,
            headless,
        )
        self.prompt_template = prompt_template
        self.arguments = arguments or {}
        self.task_description = prompt_template.format(**self.arguments)
        self.set_printer(
            printer,
            print_to_console=print_to_console,
            print_to_browser=print_to_browser,
        )

        try:
            if self.docker_image:
                with DockerManager(self.docker_image) as docker_mgr:
                    self.docker_manager = docker_mgr
                    try:
                        return self.perform_task()
                    finally:
                        self.docker_manager = None
            else:
                return self.perform_task()
        finally:
            self.web_use_tool.close()


def main() -> None:
    import time as time_mod

    agent = AssistantAgent("Assistant Agent Test")
    task_description = """
**Task:** Search for the cheapest round-trip flights from San Francisco (SFO) to New York (JFK)
for travel dates March 15-22, 2026 and save a summary report.

**Requirements:**

1. Use the browser to go to Google Flights (https://www.google.com/travel/flights)
2. Search for round-trip flights from SFO to JFK
3. Set departure date to March 15, 2026 and return date to March 22, 2026
4. Browse the results and identify the top 5 cheapest options
5. For each flight option, note:
   - Airline name
   - Departure and arrival times
   - Number of stops
   - Total price
   - Flight duration
6. Use Bash to create a summary report file called "flight_report.txt" in the work directory
7. The report should include:
   - Search parameters (origin, destination, dates)
   - A table of the top 5 cheapest flights
   - The timestamp of when the search was performed
8. Take a screenshot of the search results page and save it as "flight_results.png"

**Important:**
- If Google Flights is difficult to interact with, try alternative sites like
  https://www.kayak.com or https://www.skyscanner.com
- Use the browser tools (go_to_url, click, type_text, etc.) for web interaction
- Use Bash/Write tools for file operations
- Be patient with page loads - use scroll() and get_dom_tree() to explore pages
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
