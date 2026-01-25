# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Claude Coding Agent using the Claude Agent SDK."""

import json
import re
import time
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
from pydantic import BaseModel, Field

from kiss.core import DEFAULT_CONFIG
from kiss.core.base_agent import BaseAgent
from kiss.core.models.model_info import get_max_context_length

BUILTIN_TOOLS = [
    "Read", "Write", "Edit", "MultiEdit", "Glob", "Grep", "Bash", "WebSearch", "WebFetch"
]

READ_TOOLS = {"Read", "Grep", "Glob"}
WRITE_TOOLS = {"Write", "Edit", "MultiEdit"}

SYSTEMS_PROMPT = """You are an expert Python programmer who writes clean, simple, \
and robust code.

## Code Style Guidelines
- Write simple, readable code with minimal indirection
- Avoid unnecessary object attributes and local variables
- No redundant abstractions or duplicate code
- Each function should do one thing well
- Use clear, descriptive names

## Testing Requirements
- Generate comprehensive tests for EVERY function and feature
- Tests MUST NOT use mocks, patches, or any form of test doubles
- Test with real inputs and verify real outputs
- Test edge cases: empty inputs, None values, boundary conditions
- Test error conditions with actual invalid inputs
- Each test should be independent and verify actual behavior

## Code Structure
- Main implementation code first
- Test code in a separate section using unittest or pytest
- Include a __main__ block to run tests

## Available Tools
You have access to the following tools to help with your task:
- read_project_file: Read files from the project directory
- WebSearch: Search the web for documentation, examples, or solutions
- WebFetch: Fetch content from a specific URL
- Read: Read files from the working directory
- Glob: Find files matching a pattern
- Grep: Search file contents

Use these tools when you need to:
- Look up API documentation or library usage
- Find examples of similar implementations
- Understand existing code in the project

## Output Format
Return a dict of the form by carefully and rigorously introspecting on your work.
```json
{
    "success": bool,
    "result": str,
}
```
result should be a yaml string in the following format:
```yaml
created:
  - file1.py
  - file2.md
modified:
  - file3.ts
  - file4.py
deleted:
  - file5.py
  - file6.py
summary: >
  A summary of the execution of the task.
```
"""


class TaskResult(BaseModel):
    success: bool = Field(description="True if the agent successfully completed the task.")
    result: str = Field(description="The result of the task.")


class ClaudeCodingAgent(BaseAgent):

    def __init__(self, name: str) -> None:
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
        self._init_run_state(model_name, BUILTIN_TOOLS)
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        self.base_dir = base_dir
        self.readable_paths = {Path(p).resolve() for p in (readable_paths or [])}
        self.writable_paths = {Path(p).resolve() for p in (writable_paths or [])}
        self.max_tokens = get_max_context_length(model_name)
        self.is_agentic = True
        self.max_steps = max_steps
        self.max_budget = max_budget

    def _is_subpath(self, target: Path, whitelist: set[Path]) -> bool:
        """Check if target is inside any whitelisted path."""
        return any(target == p or p in target.parents for p in whitelist)

    def _check_path_permission(
        self, path_str: str, allowed_paths: set[Path]
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Check if path is allowed, return appropriate permission result."""
        if not allowed_paths or self._is_subpath(Path(path_str).resolve(), allowed_paths):
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
        """Update token counts from message usage."""
        if hasattr(message, "usage") and message.usage:
            self.total_tokens_used += getattr(message.usage, "input_tokens", 0)
            self.total_tokens_used += getattr(message.usage, "output_tokens", 0)

    def _process_assistant_message(self, message: AssistantMessage, timestamp: int) -> None:
        """Process assistant message and update state."""
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
        """Process user message (tool results) and update state."""
        result = ""
        for block in message.content:
            if isinstance(block, ToolResultBlock):
                content = block.content if isinstance(block.content, str) else str(block.content)
                display = f"{content[:100]}...{content[-100:]}" if len(content) > 200 else content
                status = "Tool Call Failed" if block.is_error else "Tool Call Succeeded"
                print(f"[TOOL RESULT] {status}: {display.replace(chr(10), chr(92) + 'n')}")
                result += f"{status}\n{content}\n"

        self._add_message("user", result, timestamp)

    def _process_result_message(
        self, message: ResultMessage, timestamp: int
    ) -> dict[str, object] | None:
        """Process final result message and return parsed result."""
        self._update_token_usage(message)
        if hasattr(message, "cost") and message.cost:
            self.budget_used += message.cost
            BaseAgent.global_budget_used += message.cost

        if message.structured_output is not None:
            final_result: dict[str, object] | None = message.structured_output  # type: ignore[assignment]
        elif message.result:
            final_result = self._parse_result_json(message.result)
        else:
            final_result = None

        self._add_message("model", final_result, timestamp)
        return final_result

    async def run(
        self,
        model_name: str,
        prompt_template: str,
        arguments: dict[str, str] | None = None,
        max_steps: int = DEFAULT_CONFIG.agent.max_steps,
        max_budget: float = DEFAULT_CONFIG.agent.max_agent_budget,
        base_dir: str = str(
            Path(DEFAULT_CONFIG.agent.artifact_dir).resolve() / "claude_workdir"
        ),
        readable_paths: list[str] | None = None,
        writable_paths: list[str] | None = None,
    ) -> dict[str, object] | None:
        """Run the claude coding agent for a given task."""
        self._reset(model_name, readable_paths, writable_paths, base_dir, max_steps, max_budget)
        self.prompt_template = prompt_template
        self.arguments = arguments or {}

        options = ClaudeAgentOptions(
            model=model_name,
            system_prompt=SYSTEMS_PROMPT,
            output_format=TaskResult.model_json_schema(),
            can_use_tool=self.permission_handler,
            permission_mode="default",
            allowed_tools=BUILTIN_TOOLS,
            cwd=str(self.base_dir),
        )

        async def prompt_stream() -> Any:
            task = self.prompt_template.format(**self.arguments)
            yield {"type": "user", "message": {"role": "user", "content": task}}

        timestamp = int(time.time())
        final_result: dict[str, object] | None = None

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

    def _parse_result_json(self, result: str) -> dict[str, object] | None:
        """Parse JSON from result text, handling markdown code blocks."""
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", result, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())  # type: ignore[return-value, no-any-return]
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(result.strip())  # type: ignore[return-value, no-any-return]
        except json.JSONDecodeError:
            pass
        return {"success": True, "result": result}


async def main() -> None:
    agent = ClaudeCodingAgent("Example agent")
    task_description = """
    can you write, test, and optimize a fibonacci function in Python that is efficient and correct?
    """
    result = await agent.run(model_name="claude-sonnet-4-5", prompt_template=task_description)

    if result:
        print("\n--- FINAL AGENT REPORT ---")
        print(f"SUCCESS: {result['success']}")
        print(f"RESULT:\n{result['result']}")


if __name__ == "__main__":
    anyio.run(main)
