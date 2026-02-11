# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Gemini CLI Coding Agent using the Google ADK (Agent Development Kit)."""

from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import anyio

from google.adk.agents import Agent
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from kiss.core import config as config_module
from kiss.core.base import CODING_INSTRUCTIONS, Base
from kiss.core.models.model_info import get_max_context_length
from kiss.core.useful_tools import UsefulTools
from kiss.core.utils import is_subpath, resolve_path
from kiss.core.printer import MultiPrinter
from kiss.core.print_to_console import ConsolePrinter

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


class GeminiCliAgent(Base):
    """Gemini CLI Agent using the Google ADK (Agent Development Kit)."""

    def __init__(self, name: str) -> None:
        """Initialize a GeminiCliAgent instance.

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
        """Create tools with path restrictions for the Gemini agent.

        Returns:
            list[Any]: A list of tool functions for the agent.
        """

        def read_file(path: str) -> dict[str, Any]:
            """Read file contents.

            Args:
                path (str): The file path to read.

            Returns:
                dict: A dict with 'status' and 'content' or 'error'.
            """
            resolved = resolve_path(path, self.base_dir)
            if not is_subpath(resolved, self.readable_paths):
                return {"status": "error", "error": f"Access denied for {path}"}
            try:
                content = resolved.read_text(encoding="utf-8")
                return {"status": "success", "content": content}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        def write_file(path: str, content: str) -> dict[str, Any]:
            """Write content to a file.

            Args:
                path (str): The file path to write to.
                content (str): The text content to write.

            Returns:
                dict: A dict with 'status' and 'message' or 'error'.
            """
            resolved = resolve_path(path, self.base_dir)
            if not is_subpath(resolved, self.writable_paths):
                return {"status": "error", "error": f"Access denied for {path}"}
            try:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_text(content, encoding="utf-8")
                return {"status": "success", "message": f"Wrote {len(content)} chars to {path}"}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        def list_dir(path: str = ".") -> dict[str, Any]:
            """List directory contents.

            Args:
                path (str): The directory path to list. Defaults to current directory.

            Returns:
                dict: A dict with 'status' and 'entries' or 'error'.
            """
            resolved = resolve_path(path, self.base_dir)
            if not is_subpath(resolved, self.readable_paths):
                return {"status": "error", "error": f"Access denied for {path}"}
            try:
                entries = [
                    f"{'[dir]' if e.is_dir() else '[file]'} {e.name}"
                    for e in sorted(resolved.iterdir())
                ]
                return {
                    "status": "success",
                    "entries": entries if entries else ["(empty directory)"],
                }
            except Exception as e:
                return {"status": "error", "error": str(e)}

        def run_shell(command: str, timeout: int = 60) -> dict[str, Any]:
            """Execute a shell command.

            Args:
                command (str): The shell command to execute.
                timeout (int): Maximum execution time in seconds. Defaults to 60.

            Returns:
                dict: A dict with 'status', 'stdout', 'stderr', and 'exit_code'.
            """
            output = self.useful_tools.Bash(
                command=command, description=f"Executing: {command[:50]}...",
                timeout_seconds=float(timeout),
            )

            if output.startswith("Error:"):
                return {
                    "status": "error",
                    "stdout": "",
                    "stderr": output,
                    "exit_code": 1,
                }
            else:
                return {
                    "status": "success",
                    "stdout": output or "(no output)",
                    "stderr": "",
                    "exit_code": 0,
                }

        def web_search(query: str) -> dict[str, Any]:
            """Search the web for information (placeholder - returns mock results).

            Args:
                query (str): The search query.

            Returns:
                dict: A dict with 'status' and 'results' or 'error'.
            """
            # This is a placeholder. In a real implementation, you would
            # integrate with a search API like Google Search API or similar.
            return {
                "status": "success",
                "message": f"Web search for '{query}' - feature requires API integration",
                "results": [],
            }

        return [read_file, write_file, list_dir, run_shell, web_search]

    def _process_events(
        self, events: list[Any], timestamp: int
    ) -> tuple[str | None, list[str]]:
        """Process events from the runner and update state.

        Args:
            events: List of events from the ADK runner.
            timestamp: Unix timestamp for the events.

        Returns:
            A tuple of (final_text, streamed_texts) where streamed_texts
            are text chunks suitable for streaming to the printer.
        """
        final_text = None
        streamed_texts: list[str] = []

        for event in events:
            self.step_count += 1

            # Extract token usage from event if available
            if hasattr(event, "usage_metadata") and event.usage_metadata:
                usage = event.usage_metadata
                if hasattr(usage, "prompt_token_count"):
                    self.total_tokens_used += getattr(usage, "prompt_token_count", 0)
                if hasattr(usage, "candidates_token_count"):
                    self.total_tokens_used += getattr(usage, "candidates_token_count", 0)

            # Check if it's the final response
            if hasattr(event, "is_final_response") and event.is_final_response():
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            final_text = part.text
                            self._add_message("model", part.text, timestamp)
                            streamed_texts.append(part.text)

            # Process content parts
            elif event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        self._add_message("model", part.text, timestamp)
                        streamed_texts.append(part.text)
                    elif hasattr(part, "function_call") and part.function_call:
                        name = part.function_call.name
                        args = part.function_call.args
                        self._add_message("model", f"Tool call: {name}({args})", timestamp)
                    elif hasattr(part, "function_response") and part.function_response:
                        response = part.function_response.response
                        self._add_message("user", str(response), timestamp)
                        streamed_texts.append(str(response))

        return final_text, streamed_texts

    def run(
        self,
        model_name: str = DEFAULT_GEMINI_MODEL,
        prompt_template: str = "",
        arguments: dict[str, str] | None = None,
        max_steps: int | None = None,
        max_budget: float | None = None,
        base_dir: str | None = None,
        readable_paths: list[str] | None = None,
        writable_paths: list[str] | None = None,
        use_browser: bool = False,
    ) -> str | None:
        """Run the Gemini CLI agent for a given task."""
        cfg = config_module.DEFAULT_CONFIG.agent
        actual_max_steps = max_steps if max_steps is not None else cfg.max_steps
        actual_max_budget = max_budget if max_budget is not None else cfg.max_agent_budget
        actual_base_dir = (
            base_dir
            if base_dir is not None
            else str(Path(cfg.artifact_dir).resolve() / "gemini_workdir")
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
            timestamp = int(time.time())
            task = prompt_template.format(**(arguments or {}))
            self._add_message("user", task, timestamp)

            # Create the ADK agent with tools
            tools = self._create_tools()
            safe_name = re.sub(r"[^A-Za-z0-9_]", "_", self.name)
            if safe_name and safe_name[0].isdigit():
                safe_name = "_" + safe_name
            agent = Agent(
                model=model_name,
                name=safe_name,
                instruction=CODING_INSTRUCTIONS,
                description="An expert Python programmer that writes clean, simple, robust code.",
                tools=tools,
            )

            try:
                # Set up session service and runner
                session_service = InMemorySessionService()
                app_name = f"kiss_gemini_agent_{self.id}"
                user_id = "kiss_user"
                session_id = f"session_{self.run_start_timestamp}"

                session = await session_service.create_session(
                    app_name=app_name, user_id=user_id, session_id=session_id
                )

                runner = Runner(agent=agent, app_name=app_name, session_service=session_service)

                # Create user message content
                content = types.Content(role="user", parts=[types.Part(text=task)])

                # Run the agent and collect events
                final_result: str | None = None
                events_list: list[Event] = []

                async for event in runner.run_async(
                    user_id=user_id, session_id=session.id, new_message=content
                ):
                    events_list.append(event)
                    timestamp = int(time.time())

                # Process all collected events
                final_result, streamed_texts = self._process_events(events_list, timestamp)

                if self.printer:
                    for text in streamed_texts:
                        await self.printer.token_callback(text)

            except Exception as e:
                final_result = f"Error: {e}"
                self._add_message("model", f"Error: {e}", timestamp)

            self._save()
            return final_result

        return anyio.run(_run_async)


def main() -> None:
    """Example usage of the GeminiCliAgent."""
    import time as time_mod

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
    os.chdir(work_dir)
    start_time = time_mod.time()
    try:
        agent = GeminiCliAgent("example_gemini_agent")
        result = agent.run(model_name=DEFAULT_GEMINI_MODEL, prompt_template=task_description, use_browser=True)
    finally:
        os.chdir(old_cwd)
    elapsed = time_mod.time() - start_time
    print(f"Time: {elapsed:.1f}s")
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
