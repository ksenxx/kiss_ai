"""VS Code extension backend server for Sorcar agent.

This module provides a JSON-based stdio interface between the VS Code
extension and the Sorcar agent. Commands are read from stdin as JSON
lines, and events are written to stdout as JSON lines.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
from typing import Any

from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.task_history import (
    _load_file_usage,
    _load_history,
    _load_model_usage,
    _record_file_usage,
    _search_history,
)
from kiss.core.models.model import Attachment
from kiss.core.models.model_info import MODEL_INFO, get_available_models


def _model_vendor_order(name: str) -> int:
    """Return sort order for model vendor grouping (matches web Sorcar)."""
    openai_prefixes = ("gpt", "o1", "o3", "o4", "codex", "computer-use")
    if name.startswith("claude-"):
        return 0
    if name.startswith(openai_prefixes):
        return 1
    if name.startswith("gemini-"):
        return 2
    if name.startswith("minimax-"):
        return 3
    if name.startswith("openrouter/"):
        return 4
    return 5


def _model_vendor_name(name: str) -> str:
    """Return vendor display name from model name (matches web Sorcar)."""
    if name.startswith("claude-"):
        return "Anthropic"
    openai_prefixes = ("gpt", "o1", "o3", "o4", "codex", "computer-use")
    if name.startswith(openai_prefixes) and not name.startswith("openai/"):
        return "OpenAI"
    if name.startswith("gemini-"):
        return "Gemini"
    if name.startswith("minimax-"):
        return "MiniMax"
    if name.startswith("openrouter/"):
        return "OpenRouter"
    return "Together AI"


class VSCodePrinter(BaseBrowserPrinter):
    """Printer that outputs JSON events to stdout for VS Code extension.

    Inherits from BaseBrowserPrinter to get identical event parsing and
    emission (thinking_start/delta/end, text_delta/end, tool_call,
    tool_result, system_output, result, usage_info). Overrides
    broadcast() to write JSON lines to stdout instead of SSE queues.
    """

    def __init__(self) -> None:
        super().__init__()
        self._stdout_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Write event as a JSON line to stdout and record it.

        Args:
            event: The event dictionary to emit.
        """
        with self._lock:
            for events_list in self._recordings.values():
                events_list.append(event)
        with self._stdout_lock:
            sys.stdout.write(json.dumps(event) + "\n")
            sys.stdout.flush()


class VSCodeServer:
    """Backend server for VS Code extension."""

    def __init__(self) -> None:
        self.printer = VSCodePrinter()
        self.agent = SorcarAgent("Sorcar VS Code")
        self.work_dir = os.environ.get("KISS_WORKDIR", os.getcwd())
        self._stop_event: threading.Event | None = None
        self._user_answer_event: threading.Event | None = None
        self._user_answer: str = ""
        self._selected_model = os.environ.get("KISS_MODEL", "claude-opus-4-6")
        self._file_cache: list[str] = []

    def run(self) -> None:
        """Main loop: read commands from stdin, execute them."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
                self._handle_command(cmd)
            except json.JSONDecodeError as e:
                self.printer.broadcast({"type": "error", "text": f"Invalid JSON: {e}"})
            except Exception as e:  # pragma: no cover
                self.printer.broadcast({"type": "error", "text": str(e)})

    def _handle_command(self, cmd: dict[str, Any]) -> None:
        """Handle a command from VS Code."""
        cmd_type = cmd.get("type")

        if cmd_type == "run":
            self._run_task(cmd)
        elif cmd_type == "stop":
            self._stop_task()
        elif cmd_type == "getModels":
            self._get_models()
        elif cmd_type == "selectModel":
            self._selected_model = cmd.get("model", self._selected_model)
        elif cmd_type == "getHistory":
            self._get_history(cmd.get("query"))
        elif cmd_type == "getFiles":
            self._get_files(cmd.get("prefix", ""))
        elif cmd_type == "refreshFiles":
            self._refresh_file_cache()
        elif cmd_type == "recordFileUsage":
            path = cmd.get("path", "")
            if path:
                _record_file_usage(path)
        elif cmd_type == "userAnswer":
            self._user_answer = cmd.get("answer", "")
            if self._user_answer_event:
                self._user_answer_event.set()
        else:
            self.printer.broadcast({"type": "error", "text": f"Unknown command: {cmd_type}"})

    def _run_task(self, cmd: dict[str, Any]) -> None:
        """Run the agent with the given task."""
        prompt = cmd.get("prompt", "")
        model = cmd.get("model")
        work_dir = cmd.get("workDir") or self.work_dir
        active_file = cmd.get("activeFile")
        raw_attachments = cmd.get("attachments", [])

        # Convert attachments to proper format
        attachments: list[Attachment] | None = None
        if raw_attachments:
            attachments = []
            for att in raw_attachments:
                data = base64.b64decode(att.get("data", ""))
                attachments.append(
                    Attachment(
                        data=data,
                        mime_type=att.get("mimeType", "application/octet-stream"),
                    )
                )

        # Create stop event
        self._stop_event = threading.Event()
        self.printer._thread_local.stop_event = self._stop_event

        # Create user answer event for ask_user_question
        self._user_answer_event = threading.Event()

        self.printer.broadcast({"type": "status", "running": True})
        self.printer.broadcast({"type": "user_msg", "text": prompt})
        self.printer.broadcast({"type": "clear"})

        try:
            self.agent.run(
                prompt_template=prompt,
                model_name=model,
                work_dir=work_dir,
                printer=self.printer,
                current_editor_file=active_file,
                attachments=attachments,
                wait_for_user_callback=self._wait_for_user,
                ask_user_question_callback=self._ask_user_question,
            )
            # BaseBrowserPrinter.print(type="result") already broadcast the
            # result event with parsed summary, tokens, and cost.
            self.printer.broadcast({"type": "task_done"})
        except KeyboardInterrupt:
            self.printer.broadcast({"type": "task_stopped"})
        except Exception as e:  # pragma: no cover
            self.printer.broadcast({"type": "task_error", "text": str(e)})
        finally:
            self.printer.broadcast({"type": "status", "running": False})
            self.printer.reset()
            self._stop_event = None
            self._user_answer_event = None

    def _stop_task(self) -> None:
        """Signal the agent to stop."""
        if self._stop_event:
            self._stop_event.set()

    def _wait_for_user(self, instruction: str, url: str) -> None:
        """Callback for browser action prompts."""
        self.printer.broadcast({
            "type": "waitForUser",
            "instruction": instruction,
            "url": url,
        })
        # Wait for user to signal they're done
        if self._user_answer_event:
            self._user_answer_event.clear()
            self._user_answer_event.wait()

    def _ask_user_question(self, question: str) -> str:
        """Callback for agent questions."""
        self.printer.broadcast({
            "type": "askUser",
            "question": question,
        })
        # Wait for user answer
        if self._user_answer_event:
            self._user_answer_event.clear()
            self._user_answer_event.wait()
        return self._user_answer

    def _get_models(self) -> None:
        """Send available models list with usage counts and pricing.

        Matches the web Sorcar's models_endpoint: includes inp/out prices,
        usage counts, and sorts by vendor order then price descending.
        """
        usage = _load_model_usage()
        models_list: list[dict[str, Any]] = []
        for name in get_available_models():
            info = MODEL_INFO.get(name)
            if info and info.is_function_calling_supported:
                models_list.append({
                    "name": name,
                    "inp": info.input_price_per_1M,
                    "out": info.output_price_per_1M,
                    "uses": usage.get(name, 0),
                    "vendor": _model_vendor_name(name),
                })
        models_list.sort(
            key=lambda m: (
                _model_vendor_order(str(m["name"])),
                -(float(m["inp"]) + float(m["out"])),
            )
        )
        self.printer.broadcast({
            "type": "models",
            "models": models_list,
            "selected": self._selected_model,
        })

    def _get_history(self, query: str | None) -> None:
        """Send conversation history."""
        if query:
            entries = _search_history(query, limit=20)
        else:
            entries = _load_history(limit=20)

        sessions = []
        for entry in entries:
            if isinstance(entry, dict):
                task = str(entry.get("task", ""))
                # Use events_file as ID
                events_file = str(entry.get("events_file", ""))
                sessions.append({
                    "id": events_file,
                    "title": task[:50] + "..." if len(task) > 50 else task,
                    "timestamp": 0,  # Not stored in history
                    "preview": task[:100],
                })
        self.printer.broadcast({"type": "history", "sessions": sessions})

    def _refresh_file_cache(self) -> None:
        """Refresh the file cache from disk."""
        from kiss.agents.sorcar.code_server import _scan_files

        self._file_cache = _scan_files(self.work_dir)

    def _get_files(self, prefix: str) -> None:
        """Send file list for autocomplete with usage-based sorting.

        Matches the web Sorcar's suggestions endpoint: splits files into
        frequent (with usage > 0) and rest, sorts frequent by end-distance,
        recency, and usage count, sorts rest by end-distance.
        """
        if not self._file_cache:
            self._refresh_file_cache()

        q = prefix.lower()
        usage = _load_file_usage()
        frequent: list[dict[str, str]] = []
        rest: list[dict[str, str]] = []

        for path in self._file_cache:
            if not q or q in path.lower():
                item = {"type": "file", "text": path}
                if usage.get(path, 0) > 0:
                    frequent.append(item)
                else:
                    rest.append(item)

        def _end_dist(text: str) -> int:
            if not q:
                return 0
            pos = text.lower().rfind(q)
            if pos < 0:
                return len(text)
            return len(text) - (pos + len(q))

        _usage_keys = list(usage.keys())
        _recency = {k: i for i, k in enumerate(reversed(_usage_keys))}
        _n = len(_usage_keys)
        frequent.sort(
            key=lambda m: (
                _end_dist(m["text"]),
                _recency.get(m["text"], _n),
                -usage.get(m["text"], 0),
            )
        )
        rest.sort(key=lambda m: _end_dist(m["text"]))
        for f in frequent:
            f["type"] = "frequent"
        self.printer.broadcast({"type": "files", "files": (frequent + rest)[:20]})


def main() -> None:
    """Main entry point for VS Code backend server."""
    server = VSCodeServer()
    server.run()


if __name__ == "__main__":
    main()
