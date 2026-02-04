# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Claude Coding Agent using the Claude Agent SDK."""

import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    TextBlock,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)

from kiss.core import DEFAULT_CONFIG
from kiss.core.base import CODING_INSTRUCTIONS, Base
from kiss.core.models.model_info import get_max_context_length
from kiss.core.utils import is_subpath, resolve_path

BUILTIN_TOOLS = [
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "Glob",
    "Grep",
    "Bash",
    "WebSearch",
    "WebFetch",
]

READ_TOOLS = {"Read", "Grep", "Glob"}
WRITE_TOOLS = {"Write", "Edit", "MultiEdit"}


class ClaudeCodingAgent(Base):
    """Claude Coding Agent using the Claude Agent SDK."""

    def __init__(self, name: str) -> None:
        """Initialize a ClaudeCodingAgent instance.

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
    ) -> None:
        """Reset the agent's state for a new run.

        Args:
            model_name: The model name to use.
            readable_paths: Paths allowed for reading.
            writable_paths: Paths allowed for writing.
            base_dir: Base directory for path resolution.
            max_steps: Maximum steps allowed.
            max_budget: Maximum budget in USD.
        """
        self._init_run_state(model_name, BUILTIN_TOOLS)
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        self.base_dir = str(Path(base_dir).resolve())
        self.readable_paths = [resolve_path(p, base_dir) for p in readable_paths or []]
        self.writable_paths = [resolve_path(p, base_dir) for p in writable_paths or []]
        self.max_tokens = get_max_context_length(model_name)
        self.is_agentic = True
        self.max_steps = max_steps
        self.max_budget = max_budget
        self.budget_used: float = 0.0
        self.total_tokens_used: int = 0

    def _check_path_permission(
        self, path_str: str, allowed_paths: list[Path]
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Check if path is allowed, return appropriate permission result.

        Args:
            path_str: The path to check.
            allowed_paths: List of allowed path prefixes.

        Returns:
            PermissionResultAllow if path is allowed, PermissionResultDeny otherwise.
        """
        if is_subpath(Path(path_str).resolve(), allowed_paths):
            return PermissionResultAllow(behavior="allow")
        return PermissionResultDeny(
            behavior="deny", message=f"Access Denied: {path_str} is not in whitelist."
        )

    async def permission_handler(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: ToolPermissionContext,
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Handle permission requests for tool calls.

        Args:
            tool_name: The name of the tool being called.
            tool_input: The input arguments for the tool.
            context: The permission context from the SDK.

        Returns:
            PermissionResultAllow or PermissionResultDeny based on path access.
        """
        path_str = tool_input.get("file_path") or tool_input.get("path")
        if not path_str:
            return PermissionResultAllow(behavior="allow")

        if tool_name in READ_TOOLS:
            return self._check_path_permission(path_str, self.readable_paths)
        if tool_name in WRITE_TOOLS:
            return self._check_path_permission(path_str, self.writable_paths)
        return PermissionResultAllow(behavior="allow")

    def _update_token_usage(self, message: Any) -> None:
        """Update token counts from message usage.

        Args:
            message: A message object with optional usage attribute.
        """
        if hasattr(message, "usage") and message.usage:
            self.total_tokens_used += getattr(message.usage, "input_tokens", 0)
            self.total_tokens_used += getattr(message.usage, "output_tokens", 0)

    def _process_assistant_message(self, message: AssistantMessage, timestamp: int) -> None:
        """Process assistant message and update state.

        Args:
            message: The assistant message from the Claude SDK.
            timestamp: Unix timestamp for the message.
        """
        self.step_count += 1
        self._update_token_usage(message)

        thought, tool_call = "", ""
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                args_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in block.input.items())
                tool_call += f"```python\n{block.name}({args_str})\n```\n"
                print(f"[TOOL] {block.name}({args_str})")
            elif isinstance(block, TextBlock):
                thought += block.text
                print(f"[THOUGHT] {block.text}")

        self._add_message("model", thought + tool_call, timestamp)

    def _process_user_message(self, message: UserMessage, timestamp: int) -> None:
        """Process user message (tool results) and update state.

        Args:
            message: The user message containing tool results.
            timestamp: Unix timestamp for the message.
        """
        result = ""
        for block in message.content:
            if isinstance(block, ToolResultBlock):
                content = block.content if isinstance(block.content, str) else str(block.content)
                display = f"{content[:100]}...{content[-100:]}" if len(content) > 200 else content
                status = "Tool Call Failed" if block.is_error else "Tool Call Succeeded"
                print(f"[TOOL RESULT] {status}: {display.replace(chr(10), chr(92) + 'n')}")
                result += f"{status}\n{content}\n"

        self._add_message("user", result, timestamp)

    def _process_result_message(self, message: ResultMessage, timestamp: int) -> str | None:
        """Process final result message and return the result.

        Args:
            message: The final result message from the Claude SDK.
            timestamp: Unix timestamp for the message.

        Returns:
            str | None: The final result string.
        """
        self._update_token_usage(message)
        if hasattr(message, "cost") and message.total_cost_usd:
            cost = message.total_cost_usd
            self.budget_used += cost
            Base.global_budget_used += cost

        final_result = message.result
        self._add_message("model", final_result, timestamp)
        return final_result

    def run(
        self,
        model_name: str,
        prompt_template: str,
        arguments: dict[str, str] | None = None,
        max_steps: int = DEFAULT_CONFIG.agent.max_steps,
        max_budget: float = DEFAULT_CONFIG.agent.max_agent_budget,
        base_dir: str = str(Path(DEFAULT_CONFIG.agent.artifact_dir).resolve() / "claude_workdir"),
        readable_paths: list[str] | None = None,
        writable_paths: list[str] | None = None,
    ) -> str | None:
        """Run the claude coding agent for a given task.

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
        self._reset(model_name, readable_paths, writable_paths, base_dir, max_steps, max_budget)
        self.prompt_template = prompt_template
        self.arguments = arguments or {}

        async def _run_async() -> str | None:
            options = ClaudeAgentOptions(
                model=model_name,
                system_prompt=CODING_INSTRUCTIONS,
                can_use_tool=self.permission_handler,
                permission_mode="default",
                allowed_tools=BUILTIN_TOOLS,
                cwd=str(self.base_dir),
            )

            async def prompt_stream() -> AsyncGenerator[dict[str, Any]]:
                task = prompt_template.format(**(arguments or {}))
                yield {"type": "user", "message": {"role": "user", "content": task}}

            timestamp = int(time.time())
            final_result: str | None = None

            async for message in query(prompt=prompt_stream(), options=options):
                if isinstance(message, AssistantMessage):
                    self._process_assistant_message(message, timestamp)
                elif isinstance(message, UserMessage):
                    self._process_user_message(message, timestamp)
                elif isinstance(message, ResultMessage):
                    final_result = self._process_result_message(message, timestamp)
                timestamp = int(time.time())

            self._save()
            return final_result

        return anyio.run(_run_async)


def main() -> None:
    """Example usage of the ClaudeCodingAgent."""
    agent = ClaudeCodingAgent("Example agent")
    task_description = """
    can you write, test, and optimize a fibonacci function in Python that is efficient and correct?
    """
    result = agent.run(model_name="claude-sonnet-4-5", prompt_template=task_description)

    if result:
        print("\n--- FINAL AGENT REPORT ---")
        print(f"RESULT:\n{result}")


if __name__ == "__main__":
    main()
