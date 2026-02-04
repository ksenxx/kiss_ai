# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""OpenAI Codex Coding Agent using the OpenAI Agents SDK."""

import time
from pathlib import Path
from typing import Any

import anyio
from agents import Agent, Runner, function_tool
from agents.tool import WebSearchTool

from kiss.core import DEFAULT_CONFIG
from kiss.core.base import CODING_INSTRUCTIONS, Base
from kiss.core.formatter import Formatter
from kiss.core.models.model_info import get_max_context_length
from kiss.core.simple_formatter import SimpleFormatter
from kiss.core.useful_tools import UsefulTools
from kiss.core.utils import is_subpath, resolve_path

DEFAULT_CODEX_MODEL = "gpt-5.2-codex"

SANDBOX_READ_ONLY = "read-only"
SANDBOX_WORKSPACE_WRITE = "workspace-write"
SANDBOX_FULL_ACCESS = "danger-full-access"


class OpenAICodexAgent(Base):
    """OpenAI Codex Coding Agent using the OpenAI Agents SDK."""

    def __init__(self, name: str) -> None:
        """Initialize an OpenAICodexAgent instance.

        Args:
            name: The name identifier for the agent.
        """
        super().__init__(name)

    def _reset(
        self,
        model_name: str,
        readable_paths: list[str] | None,
        writable_paths: list[str] | None,
        base_dir: str,
        max_steps: int,
        max_budget: float,
        formatter: Formatter | None,
    ) -> None:
        """Reset the agent's state for a new run.

        Args:
            model_name: The model name to use.
            readable_paths: Paths allowed for reading.
            writable_paths: Paths allowed for writing.
            base_dir: Base directory for path resolution.
            max_steps: Maximum steps allowed.
            max_budget: Maximum budget in USD.
            formatter: Optional formatter for output display.
        """
        tools = ["read_file", "write_file", "list_dir", "run_shell", "web_search"]
        self._init_run_state(model_name, tools)
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        self.base_dir = base_dir
        base_path = Path(base_dir).resolve()
        resolved_readable = [resolve_path(p, base_dir) for p in readable_paths or []]
        self.readable_paths = resolved_readable + [base_path]
        resolved_writable = [resolve_path(p, base_dir) for p in writable_paths or []]
        self.writable_paths = resolved_writable + [base_path]
        self.max_tokens = get_max_context_length(model_name)
        self.is_agentic = True
        self.max_steps = max_steps
        self.max_budget = max_budget
        self._formatter = formatter or SimpleFormatter()
        self.useful_tools = UsefulTools(
            base_dir=base_dir,
            readable_paths=[str(p) for p in self.readable_paths],
            writable_paths=[str(p) for p in self.writable_paths],
        )

    def _create_tools(self) -> list[Any]:
        """Create tools with path restrictions.

        Returns:
            list[Any]: A list of tool functions for the agent.
        """

        @function_tool
        def read_file(path: str) -> str:
            """Read file contents. Args: path - file path."""
            resolved = resolve_path(path, self.base_dir)
            if not is_subpath(resolved, self.readable_paths):
                return f"Error: Access denied for {path}"
            try:
                return resolved.read_text(encoding="utf-8")
            except Exception as e:
                return f"Error: {e}"

        @function_tool
        def write_file(path: str, content: str) -> str:
            """Write content to file. Args: path - file path, content - text to write."""
            resolved = resolve_path(path, self.base_dir)
            if not is_subpath(resolved, self.writable_paths):
                return f"Error: Access denied for {path}"
            try:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_text(content, encoding="utf-8")
                return f"Wrote {len(content)} chars to {path}"
            except Exception as e:
                return f"Error: {e}"

        @function_tool
        def list_dir(path: str = ".") -> str:
            """List directory contents. Args: path - directory path."""
            resolved = resolve_path(path, self.base_dir)
            if not is_subpath(resolved, self.readable_paths):
                return f"Error: Access denied for {path}"
            try:
                entries = [
                    f"{'[dir]' if e.is_dir() else '[file]'} {e.name}"
                    for e in sorted(resolved.iterdir())
                ]
                return "\n".join(entries) if entries else "(empty)"
            except Exception as e:
                return f"Error: {e}"

        @function_tool
        def run_shell(command: str, timeout: int = 60) -> str:
            """Execute shell command. Args: command - shell command, timeout - seconds."""
            output = self.useful_tools.Bash(
                command=command, description=f"Executing: {command[:50]}..."
            )
            return output

        return [read_file, write_file, list_dir, run_shell, WebSearchTool()]

    def _update_token_usage(self, result: Any) -> None:
        """Update token counts from result.

        Args:
            result: The result object with optional raw_responses.
        """
        if hasattr(result, "raw_responses"):
            for response in result.raw_responses:
                if hasattr(response, "usage") and response.usage:
                    self.total_tokens_used += getattr(response.usage, "input_tokens", 0)
                    self.total_tokens_used += getattr(response.usage, "output_tokens", 0)

    def _process_run_result(self, result: Any, timestamp: int) -> None:
        """Process run result and update state.

        Args:
            result: The run result from the agent.
            timestamp: Unix timestamp for the result.
        """
        for item in result.new_items:
            self.step_count += 1
            item_type = type(item).__name__

            if item_type == "MessageOutputItem":
                content = str(item.raw_item.content[0].text if item.raw_item.content else "")
                self._formatter.print_label_and_value("MESSAGE", f"{content[:200]}...")
                self._add_message("model", content, timestamp)
            elif item_type == "ToolCallItem":
                name = getattr(item.raw_item, "name", "unknown")
                self._formatter.print_label_and_value("TOOL", name)
                self._add_message("model", f"Tool: {name}", timestamp)
            elif item_type == "ToolCallOutputItem":
                self._formatter.print_label_and_value("TOOL RESULT", f"{str(item.output)[:200]}...")
                self._add_message("user", str(item.output), timestamp)
            elif item_type == "ReasoningItem":
                self._formatter.print_label_and_value("REASONING", "...")
                self._add_message("model", "Reasoning", timestamp)

        self._update_token_usage(result)

    def run(
        self,
        model_name: str = DEFAULT_CODEX_MODEL,
        prompt_template: str = "",
        arguments: dict[str, str] | None = None,
        max_steps: int = DEFAULT_CONFIG.agent.max_steps,
        max_budget: float = DEFAULT_CONFIG.agent.max_agent_budget,
        base_dir: str = str(Path(DEFAULT_CONFIG.agent.artifact_dir).resolve() / "codex_workdir"),
        readable_paths: list[str] | None = None,
        writable_paths: list[str] | None = None,
        formatter: Formatter | None = None,
    ) -> str | None:
        """Run the OpenAI Codex agent for a given task.

        Args:
            model_name: The name of the model to use.
            prompt_template: The prompt template for the task.
            arguments: The arguments for the task.
            max_steps: The maximum number of steps to take.
            max_budget: The maximum budget in USD to spend.
            base_dir: The base directory relative to which readable and writable
                paths are resolved if they are not absolute.
            readable_paths: The paths from which the agent is allowed to read from.
            writable_paths: The paths to which the agent is allowed to write to.

        Returns:
            The result of the task.
        """
        self._reset(
            model_name, readable_paths, writable_paths, base_dir, max_steps, max_budget, formatter
        )
        self.prompt_template = prompt_template
        self.arguments = arguments or {}

        async def _run_async() -> str | None:
            task = prompt_template.format(**(arguments or {}))
            timestamp = int(time.time())
            self._add_message("user", task, timestamp)

            agent = Agent(
                name=self.name,
                instructions=CODING_INSTRUCTIONS,
                model=model_name,
                tools=self._create_tools(),
            )

            self._formatter.print_label_and_value("CODEX AGENT", f"Starting in {self.base_dir}")
            self._formatter.print_label_and_value(
                "CODEX AGENT", f"Model: {model_name}, Max turns: {max_steps}"
            )

            try:
                result = await Runner.run(agent, input=task, max_turns=max_steps)
                timestamp = int(time.time())
                self._process_run_result(result, timestamp)

                final_result = str(result.final_output) if result.final_output else None
                self._formatter.print_label_and_value(
                    "CODEX AGENT", f"Output: {str(result.final_output)[:500]}..."
                )

            except Exception as e:
                self._formatter.print_label_and_value("CODEX ERROR", str(e))
                final_result = f"Error: {e}"
                self._add_message("model", f"Error: {e}", timestamp)

            self._save()
            return final_result

        return anyio.run(_run_async)


def main() -> None:
    """Example usage of the OpenAICodexAgent."""
    agent = OpenAICodexAgent("Example agent")
    task_description = """
    can you write, test, and optimize a fibonacci function in Python that is efficient and correct?
    """
    result = agent.run(model_name="gpt-5.2-codex", prompt_template=task_description)

    if result:
        print("\n--- FINAL AGENT REPORT ---")
        print(f"RESULT:\n{result}")


if __name__ == "__main__":
    main()
