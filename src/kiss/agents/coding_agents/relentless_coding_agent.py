# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Multi-agent coding system with orchestration, and sub-agents using KISSAgent."""

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

IMPORTANT_INSTRUCTIONS = """
 - If you have used 50% of your max_tokens, call 'finish' with 'success'
   set to False and 'summary' set to a summary of the work you have done
   so far and the work you need to do next.
 - Be specific about what you have done and what you need to do next.
 - The summary should have two sections: starting with the headers "Work Done"
   and "Work to Do Next".
 - The summary should be compact, but should be detailed enough so that next
   agent does not repeat the work you have already done.
 - The summary should have the same format as the user's original
   prompt.  The summary should have information about testing, checking, and debugging
   the code you have written so that next agent don't have to repeat testing,
   checking, and debugging the code.
 - The user's original prompt and the summary will be given to a new agent
   to continue the task.
 - DO NOT REPEAT any work (such as testing, checking, debuggings, verifications, etc.)
   that your predecessor agent has already done.
"""

ORCHESTRATOR_PROMPT = """
## Task

{task_description}


{coding_instructions}

 - perform_subtask() will return a yaml encoded dictionary containing the keys
   'success' (boolean) and 'summary' (string).
 - Call perform_subtask() to perform a sub-task.
 - DO NOT call perform_subtask() if you can do the sub-task yourself while
   keeping the cost down.
 - You MUST save time and cost of performing a task.

# Important Instructions
{important_instructions}
"""

TASKING_PROMPT = """
# Main Task

{task_description}

# Sub-task

You need to perform the following sub-task:

Name: {subtask_name}
Description: {description}

{coding_instructions}

# Important Instructions
{important_instructions}
"""


def finish(success: bool, summary: str) -> str:
    """Finishes the current agent execution with success or failure, and summary.

    Args:
        success: True if the task was successful, False otherwise.
        summary: Summary message to return.

    Returns:
        str: A YAML encoded dictionary containing 'success' and 'summary' keys.
    """
    result_str = yaml.dump(
        {
            "success": success,
            "summary": summary,
        },
        indent=2,
        sort_keys=False,
    )
    return result_str


class SubTask:
    """Represents a sub-task in the multi-agent coding system."""

    task_counter: int = 0

    def __init__(self, name: str, description: str) -> None:
        """Initialize a SubTask instance.

        Args:
            name: The name of the sub-task.
            description: A description of what the sub-task should accomplish.
        """
        self.id = SubTask.task_counter
        self.name = name
        self.description = description
        SubTask.task_counter += 1

    def __repr__(self) -> str:
        return f"SubTask(id={self.id}, name={self.name}, description={self.description})"


class RelentlessCodingAgent(Base):
    """Relentless coding agent that uses a multi-agent architecture to solve tasks.

    This agent implements a multi-agent architecture:
    1. Orchestrator: Manages execution and keeps steps below 30
    2. Executor sub-agents: Handle specific sub-tasks efficiently
    3. Token optimization: Uses smaller models for simple tasks
    """

    def __init__(self, name: str) -> None:
        """Initialize a RelentlessCodingAgent instance.

        Args:
            name: The name identifier for the agent.
        """
        super().__init__(name)

    def _reset(
        self,
        orchestrator_model_name: str | None,
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
        """Reset the agent's state for a new run.

        Args:
            orchestrator_model_name: The model name for the orchestrator agent.
            subtasker_model_name: The model name for subtask execution.
            trials: Number of continuation attempts for incomplete tasks.
            max_steps: Maximum steps per agent execution.
            max_budget: Maximum budget in USD.
            work_dir: Working directory for the agent.
            base_dir: Base directory for path resolution.
            readable_paths: Paths allowed for reading.
            writable_paths: Paths allowed for writing.
            docker_image: Optional Docker image for sandboxed execution.
        """
        # Apply config defaults for None values
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
        self.orchestrator_model_name = (
            orchestrator_model_name
            if orchestrator_model_name is not None
            else cfg.orchestrator_model_name
        )
        self.subtasker_model_name = (
            subtasker_model_name
            if subtasker_model_name is not None
            else cfg.subtasker_model_name
        )
        self.max_tokens = max(
            get_max_context_length(self.orchestrator_model_name),
            get_max_context_length(self.subtasker_model_name),
        )

        self.budget_used: float = 0.0
        self.total_tokens_used: int = 0

        # Initialize Docker manager if docker_image is provided
        self.docker_image = docker_image
        self.docker_manager: DockerManager | None = None

        # Initialize UsefulTools instance
        self.useful_tools = UsefulTools(
            base_dir=self.base_dir,
            readable_paths=[str(p) for p in self.readable_paths],
            writable_paths=[str(p) for p in self.writable_paths],
        )


    def _docker_bash(self, command: str, description: str) -> str:
        """Execute a bash command in the Docker container.

        Args:
            command: The bash command to run.
            description: A brief description of the command.

        Returns:
            str: The output of the command.

        Raises:
            KISSError: If Docker manager is not initialized.
        """
        if self.docker_manager is None:
            raise KISSError("Docker manager not initialized")
        return self.docker_manager.run_bash_command(command, description)

    def perform_task(
        self,
    ) -> str:
        """Perform the main task by orchestrating sub-tasks.

        Returns:
            str: A YAML encoded dictionary containing 'success' (boolean)
                and 'summary' (string) keys.

        Raises:
            KISSError: If the task fails after all continuation trials.
        """
        self.formatter.print_status(f"Executing task: {self.task_description}")
        executor = KISSAgent(f"{self.name} Main")
        task_prompt_template = ORCHESTRATOR_PROMPT
        for _ in range(self.trials):
            result = executor.run(
                model_name=self.orchestrator_model_name,
                prompt_template=task_prompt_template,
                arguments={
                    "task_description": self.task_description,
                    "coding_instructions": CODING_INSTRUCTIONS,
                    "important_instructions": IMPORTANT_INSTRUCTIONS,
                },
                tools=[finish, self.perform_subtask],
                max_steps=self.max_steps,
                max_budget=self.max_budget,
                formatter=self.formatter,
                token_callback=self.token_callback,
            )
            self.budget_used += executor.budget_used  # type: ignore
            self.total_tokens_used += executor.total_tokens_used  # type: ignore

            ret = yaml.safe_load(result)
            payload = ret if isinstance(ret, dict) else {}
            success = payload.get("success", False)
            if not success:
                task_prompt_template = ORCHESTRATOR_PROMPT + "\n\n" + payload.get("summary", "")
                continue
            return result
        raise KISSError(f"Task {self.task_description} failed after {self.trials} trials")

    def perform_subtask(
        self,
        subtask_name: str,
        description: str,
    ) -> str:
        """Perform a sub-task.

        Args:
            subtask_name: Name of the sub-task.
            description: Description of the sub-task.

        Returns:
            str: A YAML encoded dictionary containing 'success' (boolean)
                and 'summary' (string) keys.

        Raises:
            KISSError: If the subtask fails after all retry trials.
        """
        subtask = SubTask(subtask_name, description)
        self.formatter.print_status(f"Executing subtask: {subtask.name}")
        executor = KISSAgent(f"{self.name} Executor {subtask.name}")
        task_prompt_template = TASKING_PROMPT

        # Use Docker bash if Docker is enabled, otherwise use local bash
        bash_tool = self._docker_bash if self.docker_manager else self.useful_tools.Bash

        for _ in range(self.trials):
            result = executor.run(
                model_name=self.subtasker_model_name,
                prompt_template=task_prompt_template,
                arguments={
                    "task_description": self.task_description,
                    "subtask_name": subtask.name,
                    "description": subtask.description,
                    "coding_instructions": CODING_INSTRUCTIONS,
                    "important_instructions": IMPORTANT_INSTRUCTIONS,
                },
                tools=[
                    finish,
                    bash_tool,
                    self.useful_tools.Edit,
                    self.useful_tools.MultiEdit,
                ],
                max_steps=self.max_steps,
                max_budget=self.max_budget,
                formatter=self.formatter,
                token_callback=self.token_callback,
            )
            self.budget_used += executor.budget_used  # type: ignore
            self.total_tokens_used += executor.total_tokens_used  # type: ignore

            ret = yaml.safe_load(result)
            payload = ret if isinstance(ret, dict) else {}
            success = payload.get("success", False)
            if not success:
                task_prompt_template = TASKING_PROMPT + "\n\n" + payload.get("summary", "")
                continue
            return result
        raise KISSError(f"Subtask {subtask.name} failed after {self.trials} trials")

    def run(
        self,
        prompt_template: str,
        arguments: dict[str, str] | None = None,
        orchestrator_model_name: str | None = None,
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
        """Run the multi-agent coding system.

        Args:
            orchestrator_model_name: The name of the orchestrator model to use.
            subtasker_model_name: The name of the subtasker model to use.
            trials: The number of trials to attempt for each subtask.
            prompt_template: The prompt template for the task.
            arguments: The arguments for the task.
            tools: Optional tools to provide to executor agents.
            max_steps: The maximum number of total steps per agent.
            max_budget: The maximum budget in USD to spend.
            work_dir: The working directory for the agent.
            base_dir: The base directory for expressing readable and writable paths.
            readable_paths: The paths from which the agent is allowed to read.
                relative paths in readable_paths is resolved against base_dir.
            writable_paths: The paths to which the agent is allowed to write
                relative paths in writable_paths is resolved against base_dir.
            docker_image: Optional Docker image name to run bash commands in a container.
                If provided, bash commands will be executed inside the Docker container.
                Example: "ubuntu:latest", "python:3.11-slim".
            formatter: The formatter to use for the agent. If None, the default formatter is used.
            token_callback: Optional async callback invoked with each streamed text token.
                Default is None.
        Returns:
            The result of the task.
        """
        self._reset(
            orchestrator_model_name,
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

        # Run with Docker container if docker_image is provided
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
    """Example usage of the RelentlessCodingAgent.

    Creates a multi-agent system and runs a sample CSV processing task.
    """

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
    """

    work_dir = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        result = agent.run(
            prompt_template=task_description,
            work_dir=work_dir,
            formatter=CompactFormatter()
        )
    finally:
        os.chdir(old_cwd)

    agent.formatter.print_status("FINAL RESULT:")
    result = yaml.safe_load(result)
    agent.formatter.print_status("Completed successfully: " + str(result["success"]))
    agent.formatter.print_status(result["summary"])
    agent.formatter.print_status("Work directory was: " + work_dir)


if __name__ == "__main__":
    main()
