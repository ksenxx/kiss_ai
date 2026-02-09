# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Single-agent coding system with smart continuation for long tasks."""

import json
import os
import tempfile
from pathlib import Path

import yaml

from kiss.core import config as config_module
from kiss.core.base import CODING_INSTRUCTIONS, Base
from kiss.core.compact_formatter import CompactFormatter
from kiss.core.formatter import Formatter
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.core.models.model import TokenCallback
from kiss.core.models.model_info import get_max_context_length
from kiss.core.useful_tools import UsefulTools
from kiss.core.utils import resolve_path
from kiss.docker.docker_manager import DockerManager

TASK_PROMPT = """# Task
{task_description}

{coding_instructions}

# Working Directory
All files MUST be created in: {work_dir}
Use relative paths from this directory. Do NOT use /tmp, ~, or any other location.

# Rules
- BATCH: Combine related commands in ONE Bash() call
- SUCCESS: Call finish(success=True, summary="done") when complete
- At step {step_threshold}: finish(success=False,
  summary={{"done":["task A", "task B"], "next":["task C"]}})
  IMPORTANT: Use specific, actionable items in "done" and "next"
  (e.g., "created db.sh with set/get/delete", not "started work")
{previous_progress}"""

CONTINUATION_PROMPT = """# CONTINUATION - Pick up where the previous trial left off
DO NOT recreate files that already exist. Build on existing work efficiently.

{existing_files}

{progress_text}

# Continuation Strategy:
1. FIRST: Quickly verify existing state (ls key files, check if tests pass)
2. Identify what still needs to be done from "Remaining Tasks"
3. Continue implementation or fix any issues found
4. Report progress in structured format at checkpoint

EFFICIENCY: Don't redo completed work. Focus on remaining tasks."""


def finish(success: bool, summary: str) -> str:
    """Finish execution with status and summary.

    Args:
        success: True if successful, False otherwise.
        summary: Summary of work done and remaining work (JSON for continuation).
    """
    return yaml.dump({"success": success, "summary": summary}, indent=2, sort_keys=False)


class RelentlessCodingAgent(Base):
    """Single-agent coding system with auto-continuation for infinite tasks."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _reset(
        self,
        subtasker_model_name: str | None,
        trials: int | None,
        max_steps: int | None,
        max_budget: float | None,
        work_dir: str | None,
        base_dir: str | None,
        readable_paths: list[str] | None,
        writable_paths: list[str] | None,
        docker_image: str | None,
    ) -> None:
        global_cfg = config_module.DEFAULT_CONFIG
        cfg = global_cfg.agent.relentless_coding_agent
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

        self.trials = trials if trials is not None else cfg.trials
        self.max_steps = max_steps if max_steps is not None else cfg.max_steps
        self.max_budget = max_budget if max_budget is not None else cfg.max_budget
        self.subtasker_model_name = (
            subtasker_model_name
            if subtasker_model_name is not None
            else cfg.subtasker_model_name
        )
        self.max_tokens = get_max_context_length(self.subtasker_model_name)

        self.budget_used: float = 0.0
        self.total_tokens_used: int = 0

        self.docker_image = docker_image
        self.docker_manager: DockerManager | None = None

        self.useful_tools = UsefulTools(
            base_dir=self.base_dir,
            readable_paths=[str(p) for p in self.readable_paths],
            writable_paths=[str(p) for p in self.writable_paths],
        )

    def _docker_bash(self, command: str, description: str) -> str:
        if self.docker_manager is None:
            raise KISSError("Docker manager not initialized")
        return self.docker_manager.run_bash_command(command, description)

    def _parse_progress(self, summary: str) -> tuple[list[str], list[str]]:
        """Parse structured progress from summary string with validation."""
        try:
            progress = json.loads(summary)
            done = progress.get("done", [])
            next_items = progress.get("next", [])
            # Validate and clean items
            if isinstance(done, list) and isinstance(next_items, list):
                # Deduplicate while preserving order
                done = list(dict.fromkeys(str(d)[:200] for d in done if d))
                next_items = list(dict.fromkeys(str(n)[:200] for n in next_items if n))
                return done, next_items
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        return [], []

    def _scan_work_dir(self) -> str:
        try:
            files = []
            for p in sorted(Path(self.work_dir).rglob("*")):
                if p.is_file():
                    rel = p.relative_to(self.work_dir)
                    size = p.stat().st_size
                    files.append(f"  {rel} ({size}B)")
            if files:
                return "## Existing Files\n" + "\n".join(files[:50])
        except Exception:
            pass
        return ""

    def _format_progress(self, done_items: list[str], next_items: list[str]) -> str:
        if not done_items:
            return ""
        progress = "## Completed Work\n"
        for item in done_items[-10:]:
            progress += f"- {item}\n"
        if next_items:
            progress += "\n## Remaining Tasks\n"
            for item in next_items[:5]:
                progress += f"- {item}\n"
        return progress

    def _calculate_step_threshold(self, trial: int, done_count: int) -> int:
        """Calculate adaptive step threshold based on progress and trial number."""
        base = self.max_steps - 3

        # For 10K+ steps: use exponential scaling with progress tracking
        if trial == 0:
            # First trial: conservative to establish baseline
            return max(6, base // 2)
        elif trial < 3:
            # Early trials: moderate step count, focus on establishing workflow
            return max(8, int(base * 0.65))
        elif done_count > trial * 2:
            # Good progress: allow more steps per trial
            return max(10, int(base * 0.85))
        elif done_count > trial:
            # Steady progress: moderate steps
            return max(9, int(base * 0.75))
        else:
            # Slow progress: reduce steps to force more frequent checkpoints
            return max(7, int(base * 0.55))

    def _build_continuation_section(
        self, done_items: list[str], next_items: list[str]
    ) -> str:
        existing_files = self._scan_work_dir()
        progress_text = self._format_progress(done_items, next_items)
        return "\n\n" + CONTINUATION_PROMPT.format(
            existing_files=existing_files,
            progress_text=progress_text,
        )

    def perform_task(self) -> str:
        self.formatter.print_status(f"Executing task: {self.task_description}")
        bash_tool = self._docker_bash if self.docker_manager else self.useful_tools.Bash

        done_items: list[str] = []
        next_items: list[str] = []

        for trial in range(self.trials):
            step_threshold = self._calculate_step_threshold(trial, len(done_items))

            if trial == 0:
                progress_section = ""
            else:
                progress_section = self._build_continuation_section(
                    done_items, next_items
                )

            executor = KISSAgent(f"{self.name} Trial-{trial}")
            try:
                result = executor.run(
                    model_name=self.subtasker_model_name,
                    prompt_template=TASK_PROMPT,
                    arguments={
                        "task_description": self.task_description,
                        "coding_instructions": CODING_INSTRUCTIONS,
                        "previous_progress": progress_section,
                        "step_threshold": str(step_threshold),
                        "work_dir": self.work_dir,
                    },
                    tools=[
                        finish,
                        bash_tool,
                        self.useful_tools.Read,
                        self.useful_tools.Edit,
                    ],
                    max_steps=self.max_steps,
                    max_budget=self.max_budget,
                    formatter=self.formatter,
                    token_callback=self.token_callback,
                )
            except KISSError:
                last_msgs = executor.messages[-2:] if hasattr(executor, "messages") else []
                context = " ".join(
                    str(m.get("content", ""))[:100]
                    for m in last_msgs
                    if isinstance(m, dict)
                )[:200]
                result = yaml.dump(
                    {
                        "success": False,
                        "summary": json.dumps(
                            {"done": done_items, "next": [f"Continue: {context}"]}
                        ),
                    },
                    sort_keys=False,
                )

            self.budget_used += executor.budget_used  # type: ignore
            self.total_tokens_used += executor.total_tokens_used  # type: ignore

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
                done_items.append(summary[:100])

        raise KISSError(f"Task failed after {self.trials} trials")

    def run(
        self,
        prompt_template: str,
        arguments: dict[str, str] | None = None,
        subtasker_model_name: str | None = None,
        trials: int | None = None,
        max_steps: int | None = None,
        max_budget: float | None = None,
        work_dir: str | None = None,
        base_dir: str | None = None,
        readable_paths: list[str] | None = None,
        writable_paths: list[str] | None = None,
        docker_image: str | None = None,
        formatter: Formatter | None = None,
        token_callback: TokenCallback | None = None,
    ) -> str:
        """Run the coding agent.

        Args:
            prompt_template: The prompt template for the task.
            arguments: The arguments for the task.
            subtasker_model_name: The name of the model to use.
            trials: The number of continuation trials for long tasks.
            max_steps: The maximum number of steps per trial.
            max_budget: The maximum budget in USD to spend.
            work_dir: The working directory for the agent.
            base_dir: The base directory for expressing readable and writable paths.
            readable_paths: The paths from which the agent is allowed to read.
            writable_paths: The paths to which the agent is allowed to write.
            docker_image: Optional Docker image name to run bash commands in a container.
            formatter: The formatter to use for the agent.
            token_callback: Optional async callback invoked with each streamed text token.
        Returns:
            The result of the task.
        """
        self._reset(
            subtasker_model_name,
            trials,
            max_steps,
            max_budget,
            work_dir,
            base_dir,
            readable_paths,
            writable_paths,
            docker_image,
        )
        self.prompt_template = prompt_template
        self.arguments = arguments or {}
        self.task_description = prompt_template.format(**self.arguments)
        self.formatter = formatter or CompactFormatter()
        self.token_callback = token_callback

        if self.docker_image:
            with DockerManager(self.docker_image) as docker_mgr:
                self.docker_manager = docker_mgr
                try:
                    return self.perform_task()
                finally:
                    self.docker_manager = None
        else:
            return self.perform_task()


def main() -> None:
    import time as time_mod

    agent = RelentlessCodingAgent("Example Multi-Agent")
    task_description = """
 **Task:** Create a robust database engine using only Bash scripts.

 **Requirements:**
 1.  Create a script named `db.sh` that interacts with a local data folder.
 2.  **Basic Operations:** Implement `db.sh set <key> <value>`,
     `db.sh get <key>`, and `db.sh delete <key>`.
 3.  **Atomicity:** Implement transaction support.
     *   `db.sh begin` starts a session where writes are cached but not visible to others.
     *   `db.sh commit` atomically applies all cached changes.
     *   `db.sh rollback` discards pending changes.
 4.  **Concurrency:** Ensure that if two different terminal windows run `db.sh`
     simultaneously, the data is never corrupted (use `mkdir`-based mutex locking).
 5.  **Validation:** Write a test script `test_stress.sh` that launches 10
     concurrent processes to spam the database, verifying no data is lost.

 **Constraints:**
 *   No external database tools (no sqlite3, no python).
 *   Standard Linux utilities only (sed, awk, grep, flock/mkdir).
 *   Safe: Operate entirely within a `./my_db` directory.
 *   No README or docs.
    """

    work_dir = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    start_time = time_mod.time()
    try:
        os.chdir(work_dir)
        result = agent.run(
            prompt_template=task_description,
            subtasker_model_name="claude-sonnet-4-5",
            max_steps=25,
            work_dir=work_dir,
            formatter=CompactFormatter()
        )
    finally:
        os.chdir(old_cwd)
    elapsed = time_mod.time() - start_time

    agent.formatter.print_status("FINAL RESULT:")
    result_data = yaml.safe_load(result)
    agent.formatter.print_status("Completed successfully: " + str(result_data["success"]))
    agent.formatter.print_status(result_data["summary"])
    agent.formatter.print_status("Work directory was: " + work_dir)
    agent.formatter.print_status(f"Time: {elapsed:.1f}s")
    agent.formatter.print_status(f"Cost: ${agent.budget_used:.4f}")
    agent.formatter.print_status(f"Total tokens: {agent.total_tokens_used}")


if __name__ == "__main__":
    main()
