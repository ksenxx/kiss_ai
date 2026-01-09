from __future__ import annotations

import json
import traceback
import base64
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

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

ORCHESTRATOR_PROMPT = """## Task
{task}

## Current State
Todos: {todo_list}
Done: {completed_tasks}
Last Error: {last_error}

## Tools
- plan_task(tasks: str): Define multiple steps (newline separated). Use for complex projects.
- execute_todo(todo_id: int): Delegate complex logic to a sub-agent.
- complete_todo(todo_id: int, result: str): Mark a task finished after manual work.
- run_bash(command: str): Execute shell. Batch multiple commands with &&.
- create_tool(name: str, description: str, bash_command_template: str): Create a reusable tool.
- read_file(path: str): Read file content.
- write_file(path: str, content: str): Write file content.
- finish(result: str): Task complete.

## Strategy
1. For simple tasks, use run_bash/write_file directly then call finish. 
2. For complex logic, use plan_task followed by execute_todo or run_bash.
3. Batch commands in run_bash to minimize steps.
4. Call finish immediately upon goal completion."""

SUB_AGENT_PROMPT = """## Sub-Task
{task}
Focus ONLY on this. Use run_bash, read_file, and write_file. Be concise and report results."""

class SelfEvolvingMultiAgent:
    """Optimized coding agent with planning and tool-usage efficiency."""

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
        cfg = DEFAULT_CONFIG.self_evolving_multi_agent
        self.model_name = get_config_value(model_name, cfg, "model")
        self.docker_image = get_config_value(docker_image, cfg, "docker_image")
        self.workdir = get_config_value(workdir, cfg, "workdir")
        self.max_steps = get_config_value(max_steps, cfg, "max_steps") or 30
        self.max_budget = get_config_value(max_budget, cfg, "max_budget") or 1.5
        self.enable_planning = get_config_value(enable_planning, cfg, "enable_planning")
        self.enable_error_recovery = get_config_value(enable_error_recovery, cfg, "enable_error_recovery")
        self.enable_dynamic_tools = get_config_value(enable_dynamic_tools, cfg, "enable_dynamic_tools")

        self.sub_agent_max_steps = cfg.sub_agent_max_steps
        self.sub_agent_max_budget = cfg.sub_agent_max_budget
        self.max_retries = cfg.max_retries
        self.max_dynamic_tools = cfg.max_dynamic_tools
        self.max_plan_items = cfg.max_plan_items

        self.state = AgentState()
        self.docker: DockerManager | None = None
        self.trajectory: list[dict[str, Any]] = []

    def _format_todo_list(self) -> str:
        if not self.state.todos:
            return "None."
        return "\n".join(f"#{t.id}: [{t.status}] {t.description}" for t in self.state.todos)

    def _format_completed_tasks(self) -> str:
        if not self.state.completed_tasks:
            return "None."
        return ", ".join(self.state.completed_tasks[-5:])

    def _create_tools(self) -> list[Callable[..., str]]:
        def run_bash(command: str, description: str = "") -> str:
            if not self.docker:
                raise KISSError("Docker not initialized.")
            return self.docker.run_bash_command(command, description or "Command")

        def read_file(path: str) -> str:
            return run_bash(f"cat {path}", f"Read {path}")

        def write_file(path: str, content: str) -> str:
            encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
            return run_bash(f"echo '{encoded}' | base64 -d > {path}", f"Write {path}")

        def plan_task(tasks: str) -> str:
            if not self.enable_planning:
                return "Planning disabled."
            lines = [t.strip() for t in tasks.strip().splitlines() if t.strip()]
            added = []
            for desc in lines[:self.max_plan_items]:
                tid = len(self.state.todos) + 1
                self.state.todos.append(TodoItem(id=tid, description=desc))
                added.append(str(tid))
            return f"Created todos: {', '.join(added)}"

        def complete_todo(todo_id: int, result: str = "Success") -> str:
            todo = next((t for t in self.state.todos if t.id == todo_id), None)
            if not todo:
                return f"Error: Todo {todo_id} not found."
            todo.status, todo.result = "completed", result
            self.state.completed_tasks.append(f"[{todo_id}] {todo.description}")
            return f"Todo {todo_id} completed."

        def execute_todo(todo_id: int) -> str:
            todo = next((t for t in self.state.todos if t.id == todo_id), None)
            if not todo or todo.status == "completed":
                return f"Todo {todo_id} invalid or already done."

            todo.status = "in_progress"
            try:
                sub_agent = KISSAgent(name=f"SubAgent-{todo_id}")
                res = sub_agent.run(
                    model_name=self.model_name,
                    prompt_template=SUB_AGENT_PROMPT,
                    arguments={"task": todo.description},
                    tools=[run_bash, read_file, write_file],
                    max_steps=self.sub_agent_max_steps,
                    max_budget=self.sub_agent_max_budget,
                )
                todo.status, todo.result = "completed", res
                self.state.completed_tasks.append(f"[{todo_id}] {todo.description}")
                return f"Todo {todo_id} finished: {res}"
            except Exception as e:
                self.state.error_count += 1
                self.state.last_error = str(e)
                if self.enable_error_recovery and self.state.error_count <= self.max_retries:
                    todo.status = "pending"
                    return f"Attempt failed, retrying {todo_id}: {e}"
                todo.status, todo.error = "failed", str(e)
                return f"Todo {todo_id} failed: {e}"

        def create_tool(name: str, description: str, bash_command_template: str) -> str:
            if not self.enable_dynamic_tools or len(self.state.dynamic_tools) >= self.max_dynamic_tools:
                return "Tool limit reached."
            if not name.isidentifier():
                return "Invalid identifier."
            
            def dynamic_tool(arg: str = "") -> str:
                try:
                    return run_bash(bash_command_template.format(arg=arg), description)
                except Exception as e:
                    return str(e)

            dynamic_tool.__name__, dynamic_tool.__doc__ = name, description
            self.state.dynamic_tools[name] = dynamic_tool
            return f"Tool '{name}' ready."

        tools = [plan_task, execute_todo, complete_todo, run_bash, create_tool, read_file, write_file]
        tools.extend(self.state.dynamic_tools.values())
        return tools

    def run(self, task: str) -> str:
        """Run the agent on a task."""
        self.state, self.trajectory = AgentState(), []
        if self.docker:
            return self._run_orchestrator(task)
        
        with DockerManager(self.docker_image, workdir="/", mount_shared_volume=True) as docker:
            self.docker = docker
            docker.run_bash_command("mkdir -p /workspace", "Init")
            docker.workdir = self.workdir
            return self._run_orchestrator(task)

    def _run_orchestrator(self, task: str) -> str:
        orchestrator = KISSAgent(name="Orchestrator")
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
        return self.trajectory

    def get_stats(self) -> dict[str, Any]:
        s = self.state
        return {
            "total_todos": len(s.todos),
            "completed": sum(1 for t in s.todos if t.status == "completed"),
            "failed": sum(1 for t in s.todos if t.status == "failed"),
            "error_count": s.error_count,
            "dynamic_tools_created": len(s.dynamic_tools),
        }

def run_task(task: str, model_name: str, docker: DockerManager) -> dict:
    """Entry point for the evolver."""
    agent = SelfEvolvingMultiAgent(model_name=model_name, max_steps=30, max_budget=1.5)
    agent.docker = docker
    docker.workdir = "/workspace"
    try:
        result = agent._run_orchestrator(task)
        return {
            "result": result,
            "metrics": {
                "llm_calls": len(agent.state.completed_tasks) + 1,
                "steps": len(agent.state.completed_tasks),
            },
            "stats": agent.get_stats(),
        }
    except Exception as e:
        return {"result": str(e), "metrics": {"llm_calls": 10, "steps": 0}, "error": str(e)}

def run_self_evolving_multi_agent_task(task, model_name=None, docker_image=None, max_steps=None, max_budget=None):
    agent = SelfEvolvingMultiAgent(model_name=model_name, docker_image=docker_image, max_steps=max_steps, max_budget=max_budget)
    try:
        res = agent.run(task)
        return {"status": "success", "result": res, "trajectory": agent.get_trajectory(), "stats": agent.get_stats()}
    except Exception as e:
        return {"status": "failure", "error": str(e), "traceback": traceback.format_exc(), "trajectory": agent.get_trajectory(), "stats": agent.get_stats()}
