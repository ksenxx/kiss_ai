# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Multi-agent coding system with orchestration, and sub-agents using KISSAgent."""

import os
import shutil
import tempfile
from pathlib import Path

import yaml

from kiss.core.base import CODING_INSTRUCTIONS, Base
from kiss.core.compact_formatter import CompactFormatter
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.formatter import Formatter
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.core.models.model_info import get_max_context_length
from kiss.core.useful_tools import UsefulTools
from kiss.core.utils import resolve_path
from kiss.docker.docker_manager import DockerManager

ORCHESTRATOR_PROMPT = """
## Task

{task_description}

Call perform_subtask() to perform a sub-task.
perform_subtask() will return a yaml encoded dictionary containing the keys
'success' (boolean) and 'summary' (string).

# **Important**: If you have used 50% of your max_tokens,
call 'finish' with 'success' set to False and 'summary' set to a summary of
the work have done so far and the work you need to do next.
"""

TASKING_PROMPT = """
# Role

You are a software engineer who is expert with
bash commands and coding.

# Sub-task

Name: {subtask_name}
Description: {description}

# Context
This is part of a larger task:

{task_description}

{coding_instructions}

# **Important**: If you have used 50% of your max_tokens,
call 'finish' with 'success' set to False and 'summary' set to a summary of
the work have done so far and the work you need to do next.
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
        """Return a detailed string representation of the SubTask.

        Returns:
            str: A string showing the SubTask's id, name, and description.
        """
        return (
            f"SubTask(id={self.id}, name={self.name}, "
            f"description={self.description})"
        )

    def __str__(self) -> str:
        """Return string representation of the SubTask.

        Returns:
            str: Same as __repr__.
        """
        return self.__repr__()


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
        orchestrator_model_name: str,
        subtasker_model_name: str,
        trials: int,
        max_steps: int,
        max_budget: float,
        work_dir: str,
        base_dir: str,
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
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        self.base_dir = str(Path(base_dir).resolve())
        self.work_dir = str(Path(work_dir).resolve())
        self.readable_paths = [resolve_path(p, self.base_dir) for p in readable_paths or []]
        self.writable_paths = [resolve_path(p, self.base_dir) for p in writable_paths or []]
        self.readable_paths.append(Path(self.work_dir))
        self.writable_paths.append(Path(self.work_dir))
        self.is_agentic = True

        self.trials = trials or DEFAULT_CONFIG.agent.relentless_coding_agent.trials
        self.max_steps = max_steps or DEFAULT_CONFIG.agent.relentless_coding_agent.max_steps
        self.max_budget = max_budget or DEFAULT_CONFIG.agent.relentless_coding_agent.max_budget
        default_orch = DEFAULT_CONFIG.agent.relentless_coding_agent.orchestrator_model_name
        self.orchestrator_model_name = orchestrator_model_name or default_orch
        default_sub = DEFAULT_CONFIG.agent.relentless_coding_agent.subtasker_model_name
        self.subtasker_model_name = subtasker_model_name or default_sub
        self.max_tokens = max(
            get_max_context_length(orchestrator_model_name),
            get_max_context_length(subtasker_model_name),
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
                },
                tools=[finish, self.perform_subtask],
                max_steps=self.max_steps,
                max_budget=self.max_budget,
                formatter=self.formatter,
            )
            self.budget_used += executor.budget_used  # type: ignore
            self.total_tokens_used += executor.total_tokens_used  # type: ignore

            ret = yaml.safe_load(result)
            success = ret.get("success", False)
            if not success:
                task_prompt_template = ORCHESTRATOR_PROMPT + "\n\n" + result
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
                    "subtask_name": subtask.name,
                    "description": subtask.description,
                    "task_description": self.task_description,
                    "max_steps": str(self.max_steps),
                    "coding_instructions": CODING_INSTRUCTIONS,
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
            )
            self.budget_used += executor.budget_used  # type: ignore
            self.total_tokens_used += executor.total_tokens_used  # type: ignore

            ret = yaml.safe_load(result)
            success = ret.get("success", False)
            if not success:
                task_prompt_template = TASKING_PROMPT + "\n\n" + result
                continue
            return result
        raise KISSError(f"Subtask {subtask.name} failed after {self.trials} trials")

    def run(
        self,
        prompt_template: str,
        arguments: dict[str, str] | None = None,
        orchestrator_model_name: str = (
            DEFAULT_CONFIG.agent.kiss_coding_agent.orchestrator_model_name
        ),
        subtasker_model_name: str = (DEFAULT_CONFIG.agent.kiss_coding_agent.subtasker_model_name),
        trials: int = DEFAULT_CONFIG.agent.kiss_coding_agent.trials,
        max_steps: int = DEFAULT_CONFIG.agent.max_steps,
        max_budget: float = DEFAULT_CONFIG.agent.max_agent_budget,
        work_dir: str = str(Path(DEFAULT_CONFIG.agent.artifact_dir).resolve() / "kiss_workdir"),
        base_dir: str = str(Path(DEFAULT_CONFIG.agent.artifact_dir).resolve() / "kiss_workdir"),
        readable_paths: list[str] | None = None,
        writable_paths: list[str] | None = None,
        docker_image: str | None = None,
        formatter: Formatter | None = None,
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

        # Run with Docker container if docker_image is provided
        if self.docker_image:
            with DockerManager(self.docker_image) as docker_mgr:
                self.docker_manager = docker_mgr
                try:
                    return self.perform_task()
                finally:
                    self.docker_manager = None
        else:
            old_cwd = os.getcwd()
            try:
                if self.work_dir:
                    os.chdir(self.work_dir)
                return self.perform_task()
            finally:
                os.chdir(old_cwd)


def main() -> None:
    """Example usage of the RelentlessCodingAgent.

    Creates a multi-agent system and runs a sample CSV processing task.
    """

    agent = RelentlessCodingAgent("Example Multi-Agent")
    task_description = """
    Create, test, and document a Python script that:
    1. Reads a CSV file with two columns (name, age)
    2. Filters rows where age > 18
    3. Writes the filtered results to a new CSV file
    4. Includes error handling for missing files
    Return a summary of your work.
    """

    work_dir = tempfile.mkdtemp()
    try:
        result = agent.run(
            prompt_template=task_description,
            work_dir=work_dir,
            formatter=CompactFormatter()
        )
    finally:
        shutil.rmtree(work_dir)


    agent.formatter.print_status("FINAL RESULT:")
    result = yaml.safe_load(result)
    agent.formatter.print_status("Completed successfully: " + str(result["success"]))
    agent.formatter.print_status(result["summary"])


if __name__ == "__main__":
    main()
