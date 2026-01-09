# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Advanced Coding Agent with planning, error recovery, and dynamic tool creation."""

from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import kiss.agents.self_evolving_multi_agent.config  # noqa: F401
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.core.utils import get_config_value
from kiss.docker.docker_manager import DockerManager


@dataclass
class TodoItem:
    """A single todo item in the task list."""

    id: int
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed
    result: str = ""
    error: str = ""


@dataclass
class AgentState:
    """Mutable state for the coding agent."""

    todos: list[TodoItem] = field(default_factory=list)
    dynamic_tools: dict[str, Callable[..., str]] = field(default_factory=dict)
    error_count: int = 0
    last_error: str = ""
    completed_tasks: list[str] = field(default_factory=list)


# Main orchestrator prompt - kept simple, no output format specification
ORCHESTRATOR_PROMPT = """
## Role ##
You are an advanced coding agent that solves complex programming tasks.

## Task ##
{task}

## Current State ##
Todo List:
{todo_list}

Completed Tasks:
{completed_tasks}

Last Error (if any):
{last_error}

## Available Tools ##
- plan_task: Create a plan by adding todo items
- execute_todo: Execute a specific todo item by ID
- run_bash: Execute a bash command in the Docker container
- create_tool: Create a new reusable tool dynamically
- read_file: Read a file from the workspace
- write_file: Write content to a file
- finish: Complete the task with the final result

## Instructions ##
1. If no plan exists, use plan_task to break down the task into steps
2. Execute todos one by one using execute_todo
3. Use run_bash for shell commands and file operations
4. Create reusable tools with create_tool when you notice repetitive patterns
5. Handle errors by retrying or adjusting your approach
6. Call finish when the task is complete
"""

# Sub-agent prompt for executing individual todos
SUB_AGENT_PROMPT = """
## Role ##
You are a focused coding sub-agent executing a specific task.

## Task ##
{task}

## Context ##
This is part of a larger task. Focus only on completing this specific step.

## Instructions ##
- Use run_bash to execute commands
- Use read_file and write_file for file operations
- Be precise and verify your work
- Report the result clearly when done
"""


class SelfEvolvingMultiAgent:
    """Advanced coding agent with planning, error recovery, and dynamic tools."""

    def __init__(
        self,
        model_name: str | None = None,
        docker_image: str | None = None,
        workdir: str | None = None,
        max_steps: int | None = None,
        max_budget: float | None = None,
        enable_planning: bool | None = None,
        enable_error_recovery: bool | None = None,
        enable_dynamic_tools: bool | None = None,
    ):
        """Initialize the self evolving multi agent.

        Args:
            model_name: LLM model to use
            docker_image: Docker image for execution
            workdir: Working directory in container
            max_steps: Maximum orchestrator steps
            max_budget: Maximum budget in USD
            enable_planning: Enable planning capabilities
            enable_error_recovery: Enable error recovery
            enable_dynamic_tools: Enable dynamic tool creation
        """
        cfg = DEFAULT_CONFIG.self_evolving_multi_agent  # type: ignore[attr-defined]

        # Use helper to reduce repetitive config access pattern
        self.model_name = get_config_value(model_name, cfg, "model")
        self.docker_image = get_config_value(docker_image, cfg, "docker_image")
        self.workdir = get_config_value(workdir, cfg, "workdir")
        self.max_steps = get_config_value(max_steps, cfg, "max_steps")
        self.max_budget = get_config_value(max_budget, cfg, "max_budget")
        self.enable_planning = get_config_value(enable_planning, cfg, "enable_planning")
        self.enable_error_recovery = get_config_value(
            enable_error_recovery, cfg, "enable_error_recovery"
        )
        self.enable_dynamic_tools = get_config_value(
            enable_dynamic_tools, cfg, "enable_dynamic_tools"
        )

        # These are always from config
        self.sub_agent_max_steps = cfg.sub_agent_max_steps
        self.sub_agent_max_budget = cfg.sub_agent_max_budget
        self.max_retries = cfg.max_retries
        self.max_dynamic_tools = cfg.max_dynamic_tools
        self.max_plan_items = cfg.max_plan_items

        self.state = AgentState()
        self.docker: DockerManager | None = None
        self.trajectory: list[dict[str, Any]] = []

    def _format_todo_list(self) -> str:
        """Format the current todo list as a string."""
        if not self.state.todos:
            return "No todos yet. Use plan_task to create a plan."

        lines = []
        for todo in self.state.todos:
            status_icon = {
                "pending": "⬜",
                "in_progress": "🔄",
                "completed": "✅",
                "failed": "❌",
            }.get(todo.status, "❓")
            lines.append(f"  [{todo.id}] {status_icon} {todo.description} ({todo.status})")
            if todo.result:
                lines.append(f"      Result: {todo.result[:100]}...")
            if todo.error:
                lines.append(f"      Error: {todo.error[:100]}...")
        return "\n".join(lines)

    def _format_completed_tasks(self) -> str:
        """Format completed tasks as a string."""
        if not self.state.completed_tasks:
            return "None yet."
        return "\n".join(f"  - {task}" for task in self.state.completed_tasks[-5:])

    def _create_tools(self) -> list[Callable[..., str]]:
        """Create the tool functions for the orchestrator agent."""

        def plan_task(tasks: str) -> str:
            """Create a plan by adding todo items. Each line is a separate task.

            Args:
                tasks: Newline-separated list of task descriptions

            Returns:
                Confirmation of added tasks
            """
            if not self.enable_planning:
                return "Planning is disabled."

            task_list = [t.strip() for t in tasks.strip().split("\n") if t.strip()]
            if len(task_list) > self.max_plan_items:
                task_list = task_list[: self.max_plan_items]

            added = []
            for desc in task_list:
                todo_id = len(self.state.todos) + 1
                self.state.todos.append(TodoItem(id=todo_id, description=desc))
                added.append(f"[{todo_id}] {desc}")

            return f"Added {len(added)} tasks:\n" + "\n".join(added)

        def execute_todo(todo_id: int) -> str:
            """Execute a specific todo item using a sub-agent.

            Args:
                todo_id: The ID of the todo item to execute

            Returns:
                Result of the execution
            """
            todo = next((t for t in self.state.todos if t.id == todo_id), None)
            if not todo:
                return f"Todo item {todo_id} not found."

            if todo.status == "completed":
                return f"Todo {todo_id} already completed: {todo.result}"

            todo.status = "in_progress"

            try:
                # Create sub-agent for this task
                sub_agent = KISSAgent(name=f"SubAgent-{todo_id}")

                # Create sub-agent tools
                sub_tools = self._get_docker_tools()

                result = sub_agent.run(
                    model_name=self.model_name,
                    prompt_template=SUB_AGENT_PROMPT,
                    arguments={"task": todo.description},
                    tools=sub_tools,
                    max_steps=self.sub_agent_max_steps,
                    max_budget=self.sub_agent_max_budget,
                )

                todo.status = "completed"
                todo.result = result
                self.state.completed_tasks.append(f"[{todo_id}] {todo.description}")
                return f"Todo {todo_id} completed: {result}"

            except Exception as e:
                todo.status = "failed"
                todo.error = str(e)
                self.state.error_count += 1
                self.state.last_error = str(e)

                if self.enable_error_recovery and self.state.error_count <= self.max_retries:
                    todo.status = "pending"
                    return f"Todo {todo_id} failed: {e}. Marked for retry."
                return f"Todo {todo_id} failed: {e}"

        def create_tool(name: str, description: str, bash_command_template: str) -> str:
            """Create a new reusable tool that executes a bash command template.

            Args:
                name: Name of the new tool (alphanumeric and underscores only)
                description: What the tool does
                bash_command_template: Bash command with {arg} placeholders

            Returns:
                Confirmation or error message
            """
            if not self.enable_dynamic_tools:
                return "Dynamic tool creation is disabled."

            if len(self.state.dynamic_tools) >= self.max_dynamic_tools:
                return f"Maximum number of dynamic tools ({self.max_dynamic_tools}) reached."

            # Validate name
            if not name.replace("_", "").isalnum():
                return "Tool name must contain only alphanumeric characters and underscores."

            if name in self.state.dynamic_tools:
                return f"Tool '{name}' already exists."

            # Create the dynamic tool
            def dynamic_tool(arg: str = "") -> str:
                """Dynamically created tool."""
                try:
                    cmd = bash_command_template.format(arg=arg)
                    return self._run_bash(cmd, description)
                except Exception as e:
                    return f"Error: {e}"

            dynamic_tool.__name__ = name
            dynamic_tool.__doc__ = (
                f"{description}\n\nArgs:\n    arg: Argument to substitute into the command"
            )

            self.state.dynamic_tools[name] = dynamic_tool
            return f"Created tool '{name}': {description}"

        docker_tools = self._get_docker_tools()
        tools: list[Callable[..., str]] = [
            plan_task,
            execute_todo,
            docker_tools[0],  # run_bash
            create_tool,
            docker_tools[1],  # read_file
            docker_tools[2],  # write_file
        ]

        # Add any dynamic tools
        tools.extend(self.state.dynamic_tools.values())

        return tools

    def _run_bash(self, command: str, description: str = "") -> str:
        """Execute a bash command in the Docker container.

        Args:
            command: The bash command to execute
            description: Brief description of what the command does

        Returns:
            Command output including stdout, stderr, and exit code
        """
        if self.docker is None:
            raise KISSError("Docker container not initialized.")
        return self.docker.run_bash_command(command, description or "Executing command")

    def _read_file(self, path: str) -> str:
        """Read content from a file in the workspace.

        Args:
            path: Path to the file (relative to workspace)

        Returns:
            File content or error message
        """
        return self._run_bash(f"cat {path}", f"Reading {path}")

    def _write_file(self, path: str, content: str) -> str:
        """Write content to a file in the workspace.

        Args:
            path: Path to the file (relative to workspace)
            content: Content to write

        Returns:
            Success message or error
        """
        import base64

        # Use base64 encoding to safely handle any content including special characters
        # and heredoc delimiters
        encoded_content = base64.b64encode(content.encode("utf-8")).decode("ascii")
        cmd = f"echo '{encoded_content}' | base64 -d > {path}"
        return self._run_bash(cmd, f"Writing to {path}")

    def _get_docker_tools(self) -> list[Callable[..., str]]:
        """Get the Docker-related tools (run_bash, read_file, write_file).

        Returns closures that can be used as tools by the agent.
        """
        def run_bash(command: str, description: str = "") -> str:
            """Execute a bash command in the Docker container."""
            return self._run_bash(command, description)

        def read_file(path: str) -> str:
            """Read content from a file in the workspace."""
            return self._read_file(path)

        def write_file(path: str, content: str) -> str:
            """Write content to a file in the workspace."""
            return self._write_file(path, content)

        return [run_bash, read_file, write_file]

    def run(self, task: str) -> str:
        """Run the self evolving multi agent on a task.

        Args:
            task: The coding task to complete

        Returns:
            The final result from the agent
        """
        # Reset state
        self.state = AgentState()
        self.trajectory = []

        with DockerManager(
            self.docker_image,
            workdir="/",  # Start at root, then create workspace
            mount_shared_volume=True,
        ) as docker:
            self.docker = docker

            # Setup workspace - create the directory first
            docker.run_bash_command("mkdir -p /workspace", "Creating workspace directory")
            # Update workdir for subsequent commands
            docker.workdir = self.workdir

            # Create orchestrator agent
            orchestrator = KISSAgent(name="Multi Agent Orchestrator")

            try:
                result = orchestrator.run(
                    model_name=self.model_name,
                    prompt_template=ORCHESTRATOR_PROMPT,
                    arguments={
                        "task": task,
                        "todo_list": self._format_todo_list(),
                        "completed_tasks": self._format_completed_tasks(),
                        "last_error": self.state.last_error or "None",
                    },
                    tools=self._create_tools(),
                    max_steps=self.max_steps,
                    max_budget=self.max_budget,
                )

                self.trajectory = json.loads(orchestrator.get_trajectory())
                return result

            except KISSError as e:
                self.state.last_error = str(e)
                raise

    def get_trajectory(self) -> list[dict[str, Any]]:
        """Get the agent's execution trajectory."""
        return self.trajectory

    def get_stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        completed = sum(1 for t in self.state.todos if t.status == "completed")
        failed = sum(1 for t in self.state.todos if t.status == "failed")
        pending = sum(1 for t in self.state.todos if t.status == "pending")

        return {
            "total_todos": len(self.state.todos),
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "error_count": self.state.error_count,
            "dynamic_tools_created": len(self.state.dynamic_tools),
        }


def run_self_evolving_multi_agent_task(
    task: str,
    model_name: str | None = None,
    docker_image: str | None = None,
    max_steps: int | None = None,
    max_budget: float | None = None,
) -> dict[str, Any]:
    """Convenience function to run a self evolving multi agent task.

    Args:
        task: The self evolving multi agent task to complete
        model_name: LLM model to use
        docker_image: Docker image for execution
        max_steps: Maximum steps
        max_budget: Maximum budget in USD

    Returns:
        Dictionary with result, trajectory, and stats
    """
    agent = SelfEvolvingMultiAgent(
        model_name=model_name,
        docker_image=docker_image,
        max_steps=max_steps,
        max_budget=max_budget,
    )

    try:
        result = agent.run(task)
        return {
            "status": "success",
            "result": result,
            "trajectory": agent.get_trajectory(),
            "stats": agent.get_stats(),
        }
    except Exception as e:
        return {
            "status": "failure",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "trajectory": agent.get_trajectory(),
            "stats": agent.get_stats(),
        }


# Example usage and test
if __name__ == "__main__":
    # Simple test task
    test_task = """
    Create a Python script that:
    1. Generates the first 20 Fibonacci numbers
    2. Saves them to a file called 'fibonacci.txt'
    3. Reads the file back and prints the sum of all numbers
    """

    print("=" * 60)
    print("Self Evolving Multi Agent Test")
    print("=" * 60)

    result = run_self_evolving_multi_agent_task(
        task=test_task,
        model_name="gemini-3-flash-preview",
        max_steps=30,
        max_budget=1.0,
    )

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Status: {result['status']}")
    if result["status"] == "success":
        print(f"Result: {result['result']}")
    else:
        print(f"Error: {result.get('error', 'Unknown')}")
    print(f"Stats: {result['stats']}")
