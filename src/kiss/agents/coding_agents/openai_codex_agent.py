"""OpenAI Codex Coding Agent using the OpenAI Agents SDK."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import anyio

from agents import Agent, Runner, function_tool
from agents.tool import WebSearchTool

from kiss.core import config as config_module
from kiss.core.base import CODING_INSTRUCTIONS, Base
from kiss.core.models.model_info import get_max_context_length
from kiss.core.useful_tools import UsefulTools
from kiss.core.utils import is_subpath, resolve_path
from kiss.core.print_to_console import ConsolePrinter
from kiss.core.printer import MultiPrinter

DEFAULT_CODEX_MODEL = "gpt-5.3-codex"

SANDBOX_READ_ONLY = "read-only"
SANDBOX_WORKSPACE_WRITE = "workspace-write"
SANDBOX_FULL_ACCESS = "danger-full-access"


class OpenAICodexAgent(Base):
    """OpenAI Codex Coding Agent using the OpenAI Agents SDK."""

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
        self.model_name = model_name
        self.function_map = ["read_file", "write_file", "list_dir", "run_shell", "web_search"]
        self.messages: list[dict[str, Any]] = []
        self.step_count = 0
        self.total_tokens_used = 0
        self.budget_used = 0.0
        self.run_start_timestamp = int(time.time())
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
        self.useful_tools = UsefulTools(
            base_dir=base_dir,
            readable_paths=[str(p) for p in self.readable_paths],
            writable_paths=[str(p) for p in self.writable_paths],
        )

    def _create_tools(self) -> list[Any]:
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
                command=command, description=f"Executing: {command[:50]}...",
                timeout_seconds=float(timeout),
            )
            return output

        return [read_file, write_file, list_dir, run_shell, WebSearchTool()]

    def _update_token_usage(self, result: Any) -> None:
        if hasattr(result, "raw_responses"):
            for response in result.raw_responses:
                if hasattr(response, "usage") and response.usage:
                    self.total_tokens_used += getattr(response.usage, "input_tokens", 0)
                    self.total_tokens_used += getattr(response.usage, "output_tokens", 0)

    def _process_run_result(self, result: Any, timestamp: int) -> list[str]:
        streamed_texts: list[str] = []
        for item in result.new_items:
            self.step_count += 1
            item_type = type(item).__name__

            if item_type == "MessageOutputItem":
                raw = getattr(item, "raw_item", None)
                raw_content = getattr(raw, "content", None)
                text = str(getattr(raw_content[0], "text", "")) if raw_content else ""
                self._add_message("model", text, timestamp)
                if text:
                    streamed_texts.append(text)
            elif item_type == "ToolCallItem":
                name = getattr(item.raw_item, "name", "unknown")
                self._add_message("model", f"Tool: {name}", timestamp)
            elif item_type == "ToolCallOutputItem":
                output = str(item.output)
                self._add_message("user", output, timestamp)
                streamed_texts.append(output)
            elif item_type == "ReasoningItem":
                self._add_message("model", "Reasoning", timestamp)

        self._update_token_usage(result)
        return streamed_texts

    def run(
        self,
        model_name: str = DEFAULT_CODEX_MODEL,
        prompt_template: str = "",
        arguments: dict[str, str] | None = None,
        max_steps: int | None = None,
        max_budget: float | None = None,
        base_dir: str | None = None,
        readable_paths: list[str] | None = None,
        writable_paths: list[str] | None = None,
        use_browser: bool = False,
    ) -> str | None:
        cfg = config_module.DEFAULT_CONFIG.agent
        actual_max_steps = max_steps if max_steps is not None else cfg.max_steps
        actual_max_budget = max_budget if max_budget is not None else cfg.max_agent_budget
        actual_base_dir = (
            base_dir
            if base_dir is not None
            else str(Path(cfg.artifact_dir).resolve() / "codex_workdir")
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
        if use_browser:
            from kiss.core.print_to_browser import BrowserPrinter
            browser_printer = BrowserPrinter()
            browser_printer.start()
            self.printer = MultiPrinter([browser_printer, ConsolePrinter()])
        else:
            self.printer = ConsolePrinter()

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

            try:
                result = await Runner.run(agent, input=task, max_turns=actual_max_steps)
                timestamp = int(time.time())
                streamed_texts = self._process_run_result(result, timestamp)

                if self.printer:
                    for text in streamed_texts:
                        await self.printer.token_callback(text)

                final_result = str(result.final_output) if result.final_output else None

            except Exception as e:
                final_result = f"Error: {e}"
                self._add_message("model", f"Error: {e}", timestamp)

            self._save()
            return final_result

        return anyio.run(_run_async)


def main() -> None:
    import time as time_mod
    work_dir = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    os.chdir(work_dir)
    start_time = time_mod.time()
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

    try:
        agent = OpenAICodexAgent("Example agent")
        result = agent.run(model_name="gpt-5.3-codex", prompt_template=task_description, use_browser=True)
    finally:
        os.chdir(old_cwd)
    elapsed = time_mod.time() - start_time
    print(f"Time: {elapsed:.1f}s")
    print(f"Result: {result}")
    if result:
        print("\n--- FINAL AGENT REPORT ---")
        print(f"RESULT:\n{result}")


if __name__ == "__main__":
    main()
