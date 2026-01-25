# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Claude Coding Agent using the Claude Agent SDK.

This module provides a coding agent that uses the Claude Agent SDK to generate
tested Python programs. The agent can use various built-in tools (Read, Bash,
WebSearch, etc.) and custom tools like read_project_file.
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import anyio
import yaml
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
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model_info import get_max_context_length
from kiss.core.utils import config_to_dict

BUILTIN_TOOLS = {
    "Read": "Read files from the working directory",
    "Write": "Create or overwrite files",
    "Edit": "Make precise string-based edits to files",
    "MultiEdit": "Make multiple precise string-based edits to files",
    "Glob": "Find files by glob pattern (e.g., **/*.py)",
    "Grep": "Search file contents with regex",
    "Bash": "Run shell commands",
    "WebSearch": "Search the web for information",
    "WebFetch": "Fetch and process content from a URL",
}


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
    success: bool = Field(
        description=(
            "True if the agent successfully completed the task. "
            "Please introspect on your work to generate the success value."
        )
    )
    result: str = Field(
        description=(
            "The result of the task."
        )
    )

class ClaudeCodingAgent:

    def __init__(self, name: str) -> None:
        self.name = name

    def _reset(
        self,
        model_name: str,
        readable_paths: list[str] | None,
        writable_paths: list[str] | None,
        base_dir: str,
        max_steps: int,
        max_budget: float,
    ) -> None:
        if readable_paths is None:
            readable_paths = []
        if writable_paths is None:
            writable_paths = []
        self.base_dir = base_dir
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)

        self.id = KISSAgent.agent_counter
        KISSAgent.agent_counter += 1
        self.model_name = model_name
        self.readable_paths = {Path(p).resolve() for p in readable_paths}
        self.writable_paths = {Path(p).resolve() for p in writable_paths}
        self.messages: list[dict[str, object]] = []
        self.step_count = 0
        self.total_tokens_used = 0
        self.budget_used = 0.0
        self.run_start_timestamp = int(time.time())
        self.max_tokens = get_max_context_length(model_name)
        self.is_agentic = True
        self.max_steps = max_steps
        self.max_budget = max_budget

    def _is_subpath(self, target: Path, whitelist: set[Path]) -> bool:
        """Checks if the target path is or is inside any of the whitelisted paths."""
        return any(target == p or p in target.parents for p in whitelist)

    async def permission_handler(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: ToolPermissionContext,
    ) -> PermissionResultAllow | PermissionResultDeny:
        path_str = tool_input.get("file_path") or tool_input.get("path")

        if not path_str:
            return PermissionResultAllow(behavior="allow")

        target_path = Path(path_str).resolve()

        if tool_name in ["Read", "Grep", "Glob"]:
            if len(self.readable_paths) == 0 or self._is_subpath(
                target_path, self.readable_paths
            ):
                return PermissionResultAllow(behavior="allow")
            msg = f"Access Denied: {path_str} is not in readable whitelist."
            return PermissionResultDeny(behavior="deny", message=msg)

        if tool_name in ["Write", "Edit", "MultiEdit"]:
            if len(self.writable_paths) == 0 or self._is_subpath(
                target_path, self.writable_paths
            ):
                return PermissionResultAllow(behavior="allow")
            msg = f"Access Denied: {path_str} is not in writable whitelist."
            return PermissionResultDeny(behavior="deny", message=msg)

        return PermissionResultAllow(behavior="allow")

    async def _prompt_stream(self, task: str) -> Any:
        yield {
            "type": "user",
            "message": {"role": "user", "content": task}
        }

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
        """Run the claude coding agent for a given task.

        Args:
            task: The task to run the claude coding agent for.
            model_name: The name of the model to use for the agent.
            base_dir: The base directory to use for the agent.
            readable_paths: The paths to read from.
            writable_paths: The paths to write to.

        Returns:
            The result of the claude coding agent's task.
        """
        self._reset(model_name, readable_paths, writable_paths, base_dir, max_steps, max_budget)
        options = ClaudeAgentOptions(
            model=model_name,
            system_prompt=SYSTEMS_PROMPT,
            output_format=TaskResult.model_json_schema(),
            can_use_tool=self.permission_handler,
            permission_mode="default",
            allowed_tools=list(BUILTIN_TOOLS.keys()),
            cwd=str(self.base_dir)
        )

        timestamp = int(time.time())
        final_result: dict[str, object] | None = None
        self.prompt_template = prompt_template
        self.arguments = arguments or {}
        task = self.prompt_template.format(**self.arguments)

        async for message in query(prompt=self._prompt_stream(task), options=options):
            if isinstance(message, AssistantMessage):
                self.step_count += 1
                # Update token usage if available
                if hasattr(message, "usage") and message.usage is not None:
                    if hasattr(message.usage, "input_tokens"):
                        self.total_tokens_used += message.usage.input_tokens
                    if hasattr(message.usage, "output_tokens"):
                        self.total_tokens_used += message.usage.output_tokens
                thought = ""
                tool_call = ""
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        args_str = ", ".join(
                            f"{k}={repr(v)[:50]}" for k, v in block.input.items()
                        )
                        tool_call += f"```python\n{block.name}({args_str})\n```\n"
                        print(f"[TOOL] {block.name}({args_str})")
                    elif isinstance(block, TextBlock):
                        thought += f"{block.text}"
                        print(f"[THOUGHT] {block.text}")
                msg: dict[str, object] = {
                    "role": "model",
                    "content": thought + tool_call,
                    "timestamp": timestamp,
                    "unique_id": len(self.messages),
                }
                self.messages.append(msg)
            elif isinstance(message, UserMessage):
                result = ""
                full = ""
                for content_block in message.content:
                    if isinstance(content_block, ToolResultBlock):
                        content = content_block.content
                        if isinstance(content, str):
                            if len(content) > 200:
                                display = content[:100] + "..." + content[-100:]
                            else:
                                display = content
                            full = full + content
                            display = display.replace("\n", "\\n")
                        else:
                            content_str = str(content)
                            display = content_str[:100] + "..." + content_str[-100:]
                            full = full + content_str
                        if content_block.is_error:
                            status = "Tool Call Failed"
                        else:
                            status = "Tool Call Succeeded"
                        print(f"[TOOL RESULT] {status}: {display}")
                        result += f"{status}\n{full}\n"
                msg = {
                    "role": "user",
                    "content": result,
                    "timestamp": timestamp,
                    "unique_id": len(self.messages),
                }
                self.messages.append(msg)
            elif isinstance(message, ResultMessage):
                # Update token usage from final result if available
                if hasattr(message, "usage") and message.usage is not None:
                    if hasattr(message.usage, "input_tokens"):
                        self.total_tokens_used += message.usage.input_tokens
                    if hasattr(message.usage, "output_tokens"):
                        self.total_tokens_used += message.usage.output_tokens
                # Update budget_used if cost info is available
                if hasattr(message, "cost") and message.cost is not None:
                    self.budget_used += message.cost
                    KISSAgent.global_budget_used += message.cost
                if message.structured_output is not None:
                    final_result = message.structured_output  # type: ignore[assignment]
                elif message.result:
                    final_result = self._parse_result_json(message.result)
                msg = {
                    "role": "model",
                    "content": final_result,
                    "timestamp": timestamp,
                    "unique_id": len(self.messages),
                }
                self.messages.append(msg)
            timestamp = int(time.time())
        self._save()
        return final_result


    def _build_state_dict(self) -> dict[str, Any]:
        """Builds the state dictionary for saving."""
        try:
            max_tokens = get_max_context_length(self.model_name)
        except Exception:
            max_tokens = None

        return {
            "name": self.name,
            "id": self.id,
            "messages": self.messages,
            "function_map": list(BUILTIN_TOOLS.keys()),
            "run_start_timestamp": self.run_start_timestamp,
            "run_end_timestamp": int(time.time()),
            "config": config_to_dict(),
            "arguments": self.arguments,
            "prompt_template": self.prompt_template,
            "is_agentic": self.is_agentic,
            "model": self.model_name,
            "budget_used": self.budget_used,
            "total_budget": self.max_budget,
            "global_budget_used": KISSAgent.global_budget_used,
            "global_max_budget": DEFAULT_CONFIG.agent.global_max_budget,
            "tokens_used": self.total_tokens_used,
            "max_tokens": max_tokens,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "command": " ".join(sys.argv),
        }

    def _save(self) -> None:
        """Save the agent's state to a file."""
        state = self._build_state_dict()
        folder_path = Path(DEFAULT_CONFIG.agent.artifact_dir) / "trajectories"
        folder_path.mkdir(parents=True, exist_ok=True)
        name_safe = self.name.replace(" ", "_").replace("/", "_")
        filename = folder_path / f"trajectory_{name_safe}_{self.id}_{self.run_start_timestamp}.yaml"
        with filename.open("w", encoding="utf-8") as f:
            yaml.dump(state, f, indent=2)


    def get_trajectory(self) -> str:
        """Returns the trajectory of the agent in standard JSON format for visualization."""
        trajectory = []
        for message in self.messages:
            trajectory.append(message)
        return json.dumps(trajectory, indent=2)

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



