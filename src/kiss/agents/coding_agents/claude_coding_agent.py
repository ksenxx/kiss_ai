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
        usage = getattr(message, "usage", None)
        if not usage:
            return
        if isinstance(usage, dict):
            self.total_tokens_used += usage.get("input_tokens", 0)
            self.total_tokens_used += usage.get("output_tokens", 0)
        else:
            self.total_tokens_used += getattr(usage, "input_tokens", 0)
            self.total_tokens_used += getattr(usage, "output_tokens", 0)

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
        prompt_template: str = "",
        model_name: str | None = None,
        subtasker_model_name: str | None = None,
        arguments: dict[str, str] | None = None,
        max_steps: int | None = None,
        max_budget: float | None = None,
        base_dir: str | None = None,
        work_dir: str | None = None,
        readable_paths: list[str] | None = None,
        writable_paths: list[str] | None = None,
        token_callback: TokenCallback | None = None,
        formatter: Any = None,
        trials: int | None = None,
        max_thinking_tokens: int = 1024,
    ) -> str:
        cfg = config_module.DEFAULT_CONFIG.agent
        actual_model = model_name or subtasker_model_name or "claude-sonnet-4-5"
        actual_max_steps = max_steps if max_steps is not None else cfg.max_steps
        actual_max_budget = max_budget if max_budget is not None else cfg.max_agent_budget
        actual_base_dir = (
            work_dir or base_dir
            or str(Path(cfg.artifact_dir).resolve() / "claude_workdir")
        )
        self._reset(
            actual_model,
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
            system_prompt = (
                CODING_INSTRUCTIONS
                + "\n## Efficiency\n"
                "- Use Write to create complete files in one step\n"
                "- Batch related bash commands with &&\n"
                "- Minimize conversation turns\n"
            )
            options = ClaudeAgentOptions(
                model=actual_model,
                system_prompt=system_prompt,
                can_use_tool=self.permission_handler,
                permission_mode="bypassPermissions",
                allowed_tools=BUILTIN_TOOLS,
                disallowed_tools=["EnterPlanMode"],
                cwd=str(self.base_dir),
                include_partial_messages=True,
                max_thinking_tokens=max_thinking_tokens,
                max_budget_usd=actual_max_budget,
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

        result = anyio.run(_run_async)
        return result or ""


def main() -> None:
    import os
    import tempfile

    agent = ClaudeCodingAgent("Example agent", use_browser=False)
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
    start_time = time.time()
    try:
        os.chdir(work_dir)
        result = agent.run(
            prompt_template=task_description,
            model_name="claude-sonnet-4-5",
            work_dir=work_dir,
            max_steps=25,
        )
    finally:
        os.chdir(old_cwd)
    elapsed = time.time() - start_time

    print("\n--- FINAL AGENT REPORT ---")
    print(f"Success: {bool(result)}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Cost: ${agent.budget_used:.4f}")
    print(f"Total tokens: {agent.total_tokens_used}")
    print(f"Work directory: {work_dir}")
    if result:
        print(f"RESULT:\n{result[:500]}")


if __name__ == "__main__":
    main()
