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
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)
from claude_agent_sdk.types import StreamEvent

from kiss.agents.coding_agents.print_to_console import ConsolePrinter
from kiss.core import config as config_module
from kiss.core.base import CODING_INSTRUCTIONS, Base
from kiss.core.models.model import TokenCallback
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

    def __init__(self, name: str, use_browser: bool = False) -> None:
        super().__init__(name)
        self._use_browser = use_browser
        if use_browser:
            from kiss.agents.coding_agents.print_to_browser import BrowserPrinter

            self._printer: Any = BrowserPrinter()
        else:
            self._printer = ConsolePrinter()

    def _reset(
        self,
        model_name: str,
        readable_paths: list[str] | None,
        writable_paths: list[str] | None,
        base_dir: str,
        max_steps: int,
        max_budget: float,
    ) -> None:
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
        path_str = tool_input.get("file_path") or tool_input.get("path")
        if not path_str:
            return PermissionResultAllow(behavior="allow")

        if tool_name in READ_TOOLS:
            return self._check_path_permission(path_str, self.readable_paths)
        if tool_name in WRITE_TOOLS:
            return self._check_path_permission(path_str, self.writable_paths)
        return PermissionResultAllow(behavior="allow")

    def _update_token_usage(self, message: Any) -> None:
        if hasattr(message, "usage") and message.usage:
            self.total_tokens_used += getattr(message.usage, "input_tokens", 0)
            self.total_tokens_used += getattr(message.usage, "output_tokens", 0)

    def _process_assistant_message(self, message: AssistantMessage, timestamp: int) -> None:
        self.step_count += 1
        self._update_token_usage(message)

        thought, tool_call = "", ""
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                args_str = ", ".join(f"{k}={repr(v)}" for k, v in block.input.items())
                tool_call += f"```python\n{block.name}({args_str})\n```\n"
            elif isinstance(block, TextBlock):
                thought += block.text
            elif isinstance(block, ThinkingBlock):
                thought += block.thinking

        self._add_message("model", thought + tool_call, timestamp)

    def _process_user_message(self, message: UserMessage, timestamp: int) -> str:
        result = ""
        for block in message.content:
            if isinstance(block, ToolResultBlock):
                content = block.content if isinstance(block.content, str) else str(block.content)
                status = "Tool Call Failed" if block.is_error else "Tool Call Succeeded"
                result += f"{status}\n{content}\n"

        self._add_message("user", result, timestamp)
        return result

    def _process_result_message(self, message: ResultMessage, timestamp: int) -> str | None:
        self._update_token_usage(message)
        if hasattr(message, "total_cost_usd") and message.total_cost_usd:
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
        max_steps: int | None = None,
        max_budget: float | None = None,
        base_dir: str | None = None,
        readable_paths: list[str] | None = None,
        writable_paths: list[str] | None = None,
        token_callback: TokenCallback | None = None,
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
            token_callback: Optional async callback invoked with each streamed text token.

        Returns:
            The result of the task.
        """
        cfg = config_module.DEFAULT_CONFIG.agent
        actual_max_steps = max_steps if max_steps is not None else cfg.max_steps
        actual_max_budget = max_budget if max_budget is not None else cfg.max_agent_budget
        actual_base_dir = (
            base_dir
            if base_dir is not None
            else str(Path(cfg.artifact_dir).resolve() / "claude_workdir")
        )
        self._reset(
            model_name,
            readable_paths,
            writable_paths,
            actual_base_dir,
            actual_max_steps,
            actual_max_budget,
        )
        self.prompt_template = prompt_template
        self.arguments = arguments or {}
        self.token_callback = token_callback

        async def _run_async() -> str | None:
            if self._use_browser:
                self._printer.start()
            self._printer.reset()
            options = ClaudeAgentOptions(
                model=model_name,
                system_prompt=CODING_INSTRUCTIONS,
                can_use_tool=self.permission_handler,
                permission_mode="default",
                allowed_tools=BUILTIN_TOOLS,
                cwd=str(self.base_dir),
                include_partial_messages=True,
                max_thinking_tokens=10000,
            )

            async def prompt_stream() -> AsyncGenerator[dict[str, Any]]:
                task = prompt_template.format(**(arguments or {}))
                yield {"type": "user", "message": {"role": "user", "content": task}}

            timestamp = int(time.time())
            final_result: str | None = None

            async for message in query(prompt=prompt_stream(), options=options):
                if isinstance(message, StreamEvent):
                    text = self._printer.print_stream_event(message)
                    if self.token_callback and text:
                        await self.token_callback(text)
                elif isinstance(message, SystemMessage):
                    self._printer.print_message(message)
                elif isinstance(message, AssistantMessage):
                    self._process_assistant_message(message, timestamp)
                    timestamp = int(time.time())
                elif isinstance(message, UserMessage):
                    self._printer.print_message(message)
                    text = self._process_user_message(message, timestamp)
                    if self.token_callback and text:
                        await self.token_callback(text)
                    timestamp = int(time.time())
                elif isinstance(message, ResultMessage):
                    final_result = self._process_result_message(message, timestamp)
                    self._printer.print_message(
                        message,
                        step_count=self.step_count,
                        budget_used=self.budget_used,
                        total_tokens_used=self.total_tokens_used,
                    )
                    if self.token_callback and final_result:
                        await self.token_callback(final_result)
                    timestamp = int(time.time())

            self._save()
            if self._use_browser:
                self._printer.stop()
            return final_result

        return anyio.run(_run_async)


def main() -> None:
    """Example usage of the ClaudeCodingAgent with browser output."""
    agent = ClaudeCodingAgent("Example agent", use_browser=True)
    task_description = """
    can you write, test, and optimize a fibonacci function in Python that is efficient and correct?
    """
    result = agent.run(model_name="claude-sonnet-4-5", prompt_template=task_description)

    if result:
        print("\n--- FINAL AGENT REPORT ---")
        print(f"RESULT:\n{result}")


if __name__ == "__main__":
    main()
