"""Base relentless agent with smart continuation for long tasks."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from kiss.core import config as config_module
from kiss.core.base import Base
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.core.models.model_info import get_max_context_length
from kiss.core.printer import Printer
from kiss.core.utils import resolve_path
from kiss.docker.docker_manager import DockerManager

TASK_PROMPT = """# Task
{task_description}

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
        base_dir: str | None,
        readable_paths: list[str] | None,
        writable_paths: list[str] | None,
        docker_image: str | None,
        config_path: str,
    ) -> None:
        global_cfg = config_module.DEFAULT_CONFIG
        cfg = global_cfg
        for part in config_path.split("."):
            cfg = getattr(cfg, part)
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

    def perform_task(self, tools: list[Callable[..., Any]]) -> str:
        print(f"Executing task: {self.task_description}")

        done_items: list[str] = []
        next_items: list[str] = []

        all_tools: list[Callable[..., Any]] = [finish, *tools]

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
                    tools=all_tools,
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
        print_to_console: bool | None = None,
        print_to_browser: bool | None = None,
        tools_factory: Callable[[], list[Callable[..., Any]]] | None = None,
        config_path: str = "agent",
    ) -> str:
        """Run the agent with tools created by tools_factory (called after _reset)."""
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
            config_path,
        )
        self.prompt_template = prompt_template
        self.arguments = arguments or {}
        self.task_description = prompt_template.format(**self.arguments)
        self.set_printer(
            printer,
            print_to_console=print_to_console,
            print_to_browser=print_to_browser,
        )

        tools = tools_factory() if tools_factory else []

        if self.docker_image:
            with DockerManager(self.docker_image) as docker_mgr:
                self.docker_manager = docker_mgr
                try:
                    return self.perform_task(tools)
                finally:
                    self.docker_manager = None
        else:
            return self.perform_task(tools)
