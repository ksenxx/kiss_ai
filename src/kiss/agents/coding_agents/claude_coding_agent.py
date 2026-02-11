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

from kiss.core import config as config_module
from kiss.core.base import CODING_INSTRUCTIONS, Base
from kiss.core.kiss_error import KISSError
from kiss.core.models.model_info import calculate_cost, get_max_context_length
from kiss.core.print_to_console import ConsolePrinter
from kiss.core.utils import is_subpath, resolve_path

BUILTIN_TOOLS = [
    "Read", "Write", "Edit", "MultiEdit",
    "Glob", "Grep", "Bash", "WebSearch", "WebFetch",
]

READ_TOOLS = {"Read", "Grep", "Glob"}
WRITE_TOOLS = {"Write", "Edit", "MultiEdit"}


class ClaudeCodingAgent(Base):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _reset(
        self,
        model_name: str | None,
        readable_paths: list[str] | None,
        writable_paths: list[str] | None,
        base_dir: str,
        max_steps: int | None,
        max_budget: float | None,
    ) -> None:
        self.model_name = model_name or "claude-sonnet-4-5"
        self.function_map = list(BUILTIN_TOOLS)
        self.messages: list[dict[str, Any]] = []
        self.step_count = 0
        self.total_tokens_used: int = 0
        self.budget_used: float = 0.0
        self.run_start_timestamp = int(time.time())
        self.base_dir = str(Path(base_dir).resolve())
        self.readable_paths = [resolve_path(p, base_dir) for p in readable_paths or []]
        self.writable_paths = [resolve_path(p, base_dir) for p in writable_paths or []]
        self.max_tokens = get_max_context_length(self.model_name)
        self.is_agentic = True
        self.max_steps = max_steps or config_module.DEFAULT_CONFIG.agent.max_steps
        self.max_budget = max_budget or config_module.DEFAULT_CONFIG.agent.max_agent_budget
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

    def _update_token_usage_from_stream(self, event: StreamEvent) -> None:
        evt = event.event
        evt_type = evt.get("type", "")
        if evt_type == "message_start":
            self.input_tokens_used += evt.get("message", {}).get("usage", {}).get("input_tokens", 0)
        elif evt_type == "message_delta":
            self.output_tokens_used += evt.get("usage", {}).get("output_tokens", 0)
        self.total_tokens_used = self.input_tokens_used + self.output_tokens_used

    def _update_step_cost(self) -> None:
        step_input = self.input_tokens_used - self.last_step_input_tokens
        step_output = self.output_tokens_used - self.last_step_output_tokens
        self.last_step_input_tokens = self.input_tokens_used
        self.last_step_output_tokens = self.output_tokens_used
        if step_input > 0 or step_output > 0:
            step_cost = calculate_cost(self.model_name, step_input, step_output)
            self.budget_used += step_cost
            Base.global_budget_used += step_cost

    def _check_limits(self) -> None:
        if self.step_count > self.max_steps:
            raise KISSError(f"Step limit exceeded: {self.step_count}/{self.max_steps}")
        if self.total_tokens_used > self.max_tokens:
            raise KISSError(f"Token limit exceeded: {self.total_tokens_used}/{self.max_tokens}")
        if self.budget_used > self.max_budget:
            raise KISSError(
                f"Agent budget exceeded: ${self.budget_used:.4f}/${self.max_budget:.2f}"
            )
        global_max = config_module.DEFAULT_CONFIG.agent.global_max_budget
        if Base.global_budget_used > global_max:
            raise KISSError(
                f"Global budget exceeded: ${Base.global_budget_used:.4f}/${global_max:.2f}"
            )

    def _get_usage_info_string(self) -> str:
        global_max = config_module.DEFAULT_CONFIG.agent.global_max_budget
        return (
            f"\nTokens: {self.total_tokens_used}/{self.max_tokens}  "
            f"Budget: ${self.budget_used:.4f}/${self.max_budget:.2f}  "
            f"Global Budget: ${Base.global_budget_used:.4f}/${global_max:.2f}  "
            f"Steps: {self.step_count}/{self.max_steps}\n"
        )

    def _finalize_prev_model_message(self) -> None:
        self._update_step_cost()
        usage_info = self._get_usage_info_string()
        for msg in reversed(self.messages):
            if msg["role"] == "model":
                msg["content"] += usage_info
                return

    def _process_assistant_message(self, message: AssistantMessage, timestamp: int) -> None:
        if self.step_count > 0:
            self._finalize_prev_model_message()
        self.step_count += 1
        self._check_limits()

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
        self._finalize_prev_model_message()

        usage = getattr(message, "usage", None)
        if isinstance(usage, dict):
            self.total_tokens_used = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        if hasattr(message, "total_cost_usd") and message.total_cost_usd:
            cost_diff = message.total_cost_usd - self.budget_used
            self.budget_used = message.total_cost_usd
            Base.global_budget_used += cost_diff

        final_result = message.result or ""
        self._add_message("model", final_result + "\n" + self._get_usage_info_string(), timestamp)
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
        if use_browser:
            from kiss.core.print_to_browser import BrowserPrinter
            printer: Any = BrowserPrinter()
        else:
            printer = ConsolePrinter()

        cfg = config_module.DEFAULT_CONFIG.agent
        work_dir = work_dir or str(Path(cfg.artifact_dir).resolve() / "claude_workdir")
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        self._reset(
            model_name, readable_paths, writable_paths,
            base_dir or ".", max_steps, max_budget,
        )
        self.prompt_template = prompt_template
        self.arguments = arguments or {}

        async def _run_async() -> str | None:
            if use_browser:
                printer.start()
            printer.reset()
            system_prompt = (
                CODING_INSTRUCTIONS
                + "\n## Efficiency\n"
                "- Use Write to create complete files in one step\n"
                "- Batch related bash commands with &&\n"
                "- Minimize conversation turns\n"
            )
            options = ClaudeAgentOptions(
                model=model_name,
                system_prompt=system_prompt,
                can_use_tool=self.permission_handler,
                allowed_tools=BUILTIN_TOOLS,
                cwd=work_dir,
                include_partial_messages=True,
                max_thinking_tokens=max_thinking_tokens,
                max_budget_usd=max_budget,
            )

            async def prompt_stream() -> AsyncGenerator[dict[str, Any]]:
                task = prompt_template.format(**(arguments or {}))
                yield {"type": "user", "message": {"role": "user", "content": task}}

            timestamp = int(time.time())
            final_result: str | None = None
            usage_printed = False

            try:
                async for message in query(prompt=prompt_stream(), options=options):
                    if isinstance(message, StreamEvent):
                        self._update_token_usage_from_stream(message)
                        printer.print(message, type="stream_event")
                    elif isinstance(message, SystemMessage):
                        printer.print(message, type="message")
                    elif isinstance(message, AssistantMessage):
                        self._process_assistant_message(message, timestamp)
                        usage_printed = False
                        timestamp = int(time.time())
                    elif isinstance(message, UserMessage):
                        if not usage_printed:
                            printer.print(
                                self._get_usage_info_string(), type="usage_info",
                            )
                            usage_printed = True
                        printer.print(message, type="message")
                        self._process_user_message(message, timestamp)
                        timestamp = int(time.time())
                    elif isinstance(message, ResultMessage):
                        final_result = self._process_result_message(message, timestamp)
                        printer.print(
                            self._get_usage_info_string(), type="usage_info",
                        )
                        printer.print(
                            message, type="message",
                            step_count=self.step_count,
                            budget_used=self.budget_used,
                            total_tokens_used=self.total_tokens_used,
                        )
                        timestamp = int(time.time())
            finally:
                self._save()
            if use_browser:
                printer.stop()
            return final_result

        return anyio.run(_run_async) or ""


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
            max_steps=100,
            use_browser=True,
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
        print(f"RESULT:\n{result}")


if __name__ == "__main__":
    main()
