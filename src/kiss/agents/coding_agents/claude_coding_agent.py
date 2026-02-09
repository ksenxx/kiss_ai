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
from kiss.core.kiss_error import KISSError
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
        self.base_dir = str(Path(base_dir).resolve())
        self.readable_paths = [resolve_path(p, base_dir) for p in readable_paths or []]
        self.writable_paths = [resolve_path(p, base_dir) for p in writable_paths or []]
        self.max_tokens = get_max_context_length(model_name)
        self.is_agentic = True
        self.max_steps = max_steps
        self.max_budget = max_budget
        self.budget_used: float = 0.0
        self.total_tokens_used: int = 0
        self.input_tokens_used: int = 0
        self.output_tokens_used: int = 0
        self.last_step_input_tokens: int = 0
        self.last_step_output_tokens: int = 0

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
        """Update token usage from message usage attribute."""
        usage = getattr(message, "usage", None)
        if not usage:
            return
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
        else:
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)

        self.input_tokens_used += input_tokens
        self.output_tokens_used += output_tokens
        self.total_tokens_used = self.input_tokens_used + self.output_tokens_used

    def _update_token_usage_from_stream_event(self, event: StreamEvent) -> None:
        """Extract and update token usage from stream events."""
        if not hasattr(event, "event"):
            return

        event_data = event.event
        if not isinstance(event_data, dict):
            return

        # Check for usage in message_delta events (output tokens)
        if event_data.get("type") == "message_delta":
            usage = event_data.get("usage", {})
            if usage:
                output_tokens = usage.get("output_tokens", 0)
                self.output_tokens_used += output_tokens
                self.total_tokens_used += output_tokens

        # Check for usage in message_start events (input tokens)
        elif event_data.get("type") == "message_start":
            message = event_data.get("message", {})
            usage = message.get("usage", {})
            if usage:
                input_tokens = usage.get("input_tokens", 0)
                self.input_tokens_used += input_tokens
                self.total_tokens_used += input_tokens

    def _get_usage_info_string(self) -> str:
        return (
            "#### Usage Information\n"
            f"- Token usage: {self.total_tokens_used}/{self.max_tokens}\n"
            f"- Agent budget: ${self.budget_used:.4f} / ${self.max_budget:.2f}\n"
            f"- Global budget: ${Base.global_budget_used:.4f} / "
            f"${config_module.DEFAULT_CONFIG.agent.global_max_budget:.2f}\n"
            f"- Step: {self.step_count}/{self.max_steps}\n"
        )

    def _process_assistant_message(self, message: AssistantMessage, timestamp: int) -> bool:
        from kiss.core.models.model_info import calculate_cost

        self.step_count += 1
        self._update_token_usage(message)

        # Calculate cost for this step based on tokens used since last step
        step_input_tokens = self.input_tokens_used - self.last_step_input_tokens
        step_output_tokens = self.output_tokens_used - self.last_step_output_tokens

        if step_input_tokens > 0 or step_output_tokens > 0:
            step_cost = calculate_cost(
                self.model_name, step_input_tokens, step_output_tokens
            )
            self.budget_used += step_cost
            Base.global_budget_used += step_cost

        # Update last step token counts for next iteration
        self.last_step_input_tokens = self.input_tokens_used
        self.last_step_output_tokens = self.output_tokens_used

        thought, tool_call = "", ""
        has_tool_calls = False
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                has_tool_calls = True
                args_str = ", ".join(f"{k}={repr(v)}" for k, v in block.input.items())
                tool_call += f"```python\n{block.name}({args_str})\n```\n"
            elif isinstance(block, TextBlock):
                thought += block.text
            elif isinstance(block, ThinkingBlock):
                thought += block.thinking

        message_content = thought + tool_call + "\n\n" + self._get_usage_info_string()
        self._add_message("model", message_content, timestamp)
        return has_tool_calls

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

        # If SDK provides total_cost_usd, use it to correct any rounding differences
        # from our incremental calculations
        if hasattr(message, "total_cost_usd") and message.total_cost_usd:
            sdk_total_cost = message.total_cost_usd
            cost_difference = sdk_total_cost - self.budget_used

            # Apply the correction
            self.budget_used = sdk_total_cost
            Base.global_budget_used += cost_difference

        final_result = message.result
        message_content = final_result + "\n\n" + self._get_usage_info_string()
        self._add_message("model", message_content, timestamp)
        return final_result

    def run(
        self,
        model_name: str | None = None,
        prompt_template: str = "",
        arguments: dict[str, str] | None = None,
        max_steps: int | None = None,
        max_budget: float | None = None,
        work_dir: str | None = None,
        base_dir: str | None = None,
        readable_paths: list[str] | None = None,
        writable_paths: list[str] | None = None,
        use_browser: bool = True,
        max_thinking_tokens: int = 1024,
    ) -> str:
        self._use_browser = use_browser
        if use_browser:
            from kiss.agents.coding_agents.print_to_browser import BrowserPrinter

            self._printer: Any = BrowserPrinter()
        else:
            self._printer = ConsolePrinter()


        cfg = config_module.DEFAULT_CONFIG.agent
        actual_model = model_name or "claude-sonnet-4-5"
        actual_max_steps = max_steps if max_steps is not None else cfg.max_steps
        actual_max_budget = max_budget if max_budget is not None else cfg.max_agent_budget
        work_dir = work_dir or str(Path(cfg.artifact_dir).resolve() / "claude_workdir")
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        self._reset(
            actual_model,
            readable_paths,
            writable_paths,
            base_dir or ".",
            actual_max_steps,
            actual_max_budget,
        )
        self.prompt_template = prompt_template
        self.arguments = arguments or {}

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
                # permission_mode="bypassPermissions",
                allowed_tools=BUILTIN_TOOLS,
                disallowed_tools=["EnterPlanMode"],
                cwd=work_dir,
                include_partial_messages=True,
                max_thinking_tokens=max_thinking_tokens,
                max_budget_usd=actual_max_budget,
            )

            async def prompt_stream() -> AsyncGenerator[dict[str, Any]]:
                task = prompt_template.format(**(arguments or {}))
                yield {"type": "user", "message": {"role": "user", "content": task}}

            timestamp = int(time.time())
            final_result: str | None = None
            max_steps_reached = False

            async for message in query(prompt=prompt_stream(), options=options):
                if isinstance(message, StreamEvent):
                    self._update_token_usage_from_stream_event(message)
                    self._printer.print_stream_event(message)
                elif isinstance(message, SystemMessage):
                    self._printer.print_message(message)
                elif isinstance(message, AssistantMessage):
                    if not max_steps_reached:
                        has_tool_calls = self._process_assistant_message(message, timestamp)
                        if has_tool_calls:
                            self._printer.print_usage_info(self._get_usage_info_string())
                        timestamp = int(time.time())
                        if self.step_count >= self.max_steps:
                            max_steps_reached = True
                            self._save()
                            raise KISSError(
                                f"Maximum steps ({self.max_steps}) exceeded. "
                                f"Agent stopped at step {self.step_count}."
                            )
                elif isinstance(message, UserMessage):
                    if not max_steps_reached:
                        self._printer.print_message(message)
                        self._process_user_message(message, timestamp)
                        timestamp = int(time.time())
                elif isinstance(message, ResultMessage):
                    final_result = self._process_result_message(message, timestamp)
                    self._printer.print_message(
                        message,
                        step_count=self.step_count,
                        budget_used=self.budget_used,
                        total_tokens_used=self.total_tokens_used,
                    )
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

    agent = ClaudeCodingAgent("Example agent")
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
            max_steps=50,
            use_browser=False,
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
