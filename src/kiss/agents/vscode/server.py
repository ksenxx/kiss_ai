"""VS Code extension backend server for Sorcar agent.

This module provides a JSON-based stdio interface between the VS Code
extension and the Sorcar agent. Commands are read from stdin as JSON
lines, and events are written to stdout as JSON lines.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import threading
from typing import Any

from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.agents.vscode.diff_merge import (
    _capture_untracked,
    _cleanup_merge_data,
    _git,
    _merge_data_dir,
    _parse_diff_hunks,
    _prepare_merge_view,
    _save_untracked_base,
    _snapshot_files,
)
from kiss.agents.vscode.helpers import (
    clean_llm_output,
    clip_autocomplete_suggestion,
    generate_followup_text,
    model_vendor,
    rank_file_suggestions,
)
from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.agents.sorcar.persistence import (
    _load_file_usage,
    _load_history,
    _load_last_model,
    _load_model_usage,
    _load_task_chat_events,
    _record_file_usage,
    _record_model_usage,
    _save_last_model,
    _search_history,
    _set_latest_chat_events,
)
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model import Attachment
from kiss.core.models.model_info import MODEL_INFO, get_available_models

logger = logging.getLogger(__name__)


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
            self._record_event(event)
        with self._stdout_lock:
            sys.stdout.write(json.dumps(event) + "\n")
            sys.stdout.flush()


class VSCodeServer:
    """Backend server for VS Code extension."""

    def __init__(self) -> None:
        self.printer = VSCodePrinter()
        self.agent = StatefulSorcarAgent("Sorcar VS Code")
        self.work_dir = os.environ.get("KISS_WORKDIR", os.getcwd())
        self._stop_event: threading.Event | None = None
        self._user_answer_event: threading.Event | None = None
        self._user_answer: str = ""
        persisted = _load_last_model()
        self._selected_model = (
            persisted
            or os.environ.get("KISS_MODEL", "")
            or "claude-opus-4-6"
        )
        self._file_cache: list[str] = []
        self._last_active_file: str = ""
        self._task_thread: threading.Thread | None = None
        self._merging = False
        self._remaining_hunks = 0

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
            if self._task_thread and self._task_thread.is_alive():
                self.printer.broadcast({"type": "error", "text": "Task already running"})
                return
            self._task_thread = threading.Thread(target=self._run_task, args=(cmd,), daemon=True)
            self._task_thread.start()
        elif cmd_type == "stop":
            self._stop_task()
        elif cmd_type == "getModels":
            self._get_models()
        elif cmd_type == "selectModel":
            model = cmd.get("model", self._selected_model)
            self._selected_model = model
            _save_last_model(model)
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
        elif cmd_type == "resumeSession":
            task = cmd.get("sessionId", "")
            if task:
                self._replay_session(task)
        elif cmd_type == "getWelcomeSuggestions":
            self._get_welcome_suggestions()
        elif cmd_type == "mergeAction":
            self._handle_merge_action(cmd.get("action", ""))
        elif cmd_type == "getLastSession":
            self._get_last_session()
        elif cmd_type == "newChat":
            self.agent.new_chat()
        elif cmd_type == "complete":
            query = cmd.get("query", "")
            active_file = cmd.get("activeFile")
            if active_file:
                self._last_active_file = active_file
            if query:
                threading.Thread(
                    target=self._complete, args=(query,), daemon=True
                ).start()
        elif cmd_type == "generateCommitMessage":
            model = cmd.get("model") or self._selected_model
            threading.Thread(
                target=self._generate_commit_message, args=(model,), daemon=True
            ).start()
        else:
            self.printer.broadcast({"type": "error", "text": f"Unknown command: {cmd_type}"})

    def _run_task(self, cmd: dict[str, Any]) -> None:
        """Run the agent with the given task."""
        prompt = cmd.get("prompt", "")
        model = cmd.get("model") or self._selected_model
        work_dir = cmd.get("workDir") or self.work_dir
        active_file = cmd.get("activeFile")
        self._last_active_file = active_file or ""
        raw_attachments = cmd.get("attachments", [])

        attachments: list[Attachment] | None = None
        image_urls: list[str] = []
        if raw_attachments:
            attachments = []
            for att in raw_attachments:
                data_b64 = att.get("data", "")
                mime = att.get("mimeType", "application/octet-stream")
                data = base64.b64decode(data_b64)
                attachments.append(Attachment(data=data, mime_type=mime))
                if mime.startswith("image/"):
                    image_urls.append(f"data:{mime};base64,{data_b64}")

        if self._merging:
            self.printer.broadcast(
                {
                    "type": "error",
                    "text": "Cannot run a task while merge review is in progress."
                    " Accept or reject all changes first.",
                }
            )
            return

        self._stop_event = threading.Event()
        self.printer._thread_local.stop_event = self._stop_event
        self._user_answer_event = threading.Event()

        pre_hunks = _parse_diff_hunks(work_dir)
        pre_untracked = _capture_untracked(work_dir)
        pre_file_hashes = _snapshot_files(
            work_dir, set(pre_hunks.keys()) | pre_untracked
        )
        _save_untracked_base(work_dir, pre_untracked | set(pre_hunks.keys()))

        self.printer.broadcast({"type": "status", "running": True})
        self.printer.broadcast({"type": "clear"})

        self.printer.start_recording()
        result_summary = ""
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
            self.printer.broadcast({"type": "task_done"})
            _record_model_usage(model)
            result_summary = self._extract_result_summary() or "No summary available"
            self._generate_followup(prompt, result_summary)
        except KeyboardInterrupt:
            self.printer.broadcast({"type": "task_stopped"})
        except Exception as e:  # pragma: no cover
            self.printer.broadcast({"type": "task_error", "text": str(e)})
        finally:
            chat_events = self.printer.stop_recording()
            _set_latest_chat_events(chat_events, task=prompt, result=result_summary)
            self.printer.broadcast({"type": "tasks_updated"})
            self.printer.broadcast({"type": "status", "running": False})
            self.printer.reset()
            self._stop_event = None
            self._user_answer_event = None
            try:
                merge_dir = str(_merge_data_dir())
                merge_result = _prepare_merge_view(
                    work_dir,
                    merge_dir,
                    pre_hunks,
                    pre_untracked,
                    pre_file_hashes,
                )
                if merge_result.get("status") == "opened":
                    self._merging = True
                    self._remaining_hunks = merge_result.get("hunk_count", 0)
                    merge_json = os.path.join(merge_dir, "pending-merge.json")
                    if os.path.exists(merge_json):
                        with open(merge_json) as f:
                            merge_data = json.load(f)
                        self.printer.broadcast({
                            "type": "merge_data",
                            "data": merge_data,
                            "hunk_count": self._remaining_hunks,
                        })
                    self.printer.broadcast({"type": "merge_started"})
            except Exception:
                logger.debug("Merge view error", exc_info=True)
            self._refresh_file_cache()

    def _handle_merge_action(self, action: str) -> None:
        """Handle merge accept/reject actions from the extension."""
        if action in ("all-done", "accept-all", "reject-all"):
            self._finish_merge()
        elif action in ("accept", "reject"):
            self._remaining_hunks = max(0, self._remaining_hunks - 1)
            if self._remaining_hunks == 0:
                self._finish_merge()

    def _finish_merge(self) -> None:
        """End the merge session: reset state, notify clients, clean up data."""
        self._merging = False
        self._remaining_hunks = 0
        self.printer.broadcast({"type": "merge_ended"})
        _cleanup_merge_data(str(_merge_data_dir()))

    def _stop_task(self) -> None:
        """Signal the agent to stop."""
        if self._stop_event:
            self._stop_event.set()

    def _await_user_response(self) -> None:
        """Block until the user sends a response."""
        if self._user_answer_event:
            self._user_answer_event.clear()
            self._user_answer_event.wait()

    def _wait_for_user(self, instruction: str, url: str) -> None:
        """Callback for browser action prompts."""
        self.printer.broadcast({
            "type": "waitForUser",
            "instruction": instruction,
            "url": url,
        })
        self._await_user_response()

    def _ask_user_question(self, question: str) -> str:
        """Callback for agent questions."""
        self.printer.broadcast({
            "type": "askUser",
            "question": question,
        })
        self._await_user_response()
        return self._user_answer

    def _get_models(self) -> None:
        """Send available models list with usage counts and pricing."""
        usage = _load_model_usage()
        models_list: list[dict[str, Any]] = []
        for name in get_available_models():
            info = MODEL_INFO.get(name)
            if info and info.is_function_calling_supported:
                vendor_name, vendor_order = model_vendor(name)
                models_list.append({
                    "name": name,
                    "inp": info.input_price_per_1M,
                    "out": info.output_price_per_1M,
                    "uses": usage.get(name, 0),
                    "vendor": vendor_name,
                    "_order": vendor_order,
                })
        models_list.sort(
            key=lambda m: (m["_order"], -(float(m["inp"]) + float(m["out"])))
        )
        for m in models_list:
            del m["_order"]
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
            task = str(entry.get("task", ""))
            has_events = bool(entry.get("has_events", False))
            sessions.append({
                "id": task,
                "title": task[:50] + "..." if len(task) > 50 else task,
                "timestamp": entry.get("timestamp", 0),
                "preview": task,
                "text": task,
                "has_events": has_events,
            })
        self.printer.broadcast({"type": "history", "sessions": sessions})

    def _replay_session(self, task: str) -> None:
        """Replay recorded chat events for a previous task."""
        events = _load_task_chat_events(task)
        if not events:
            self.printer.broadcast({"type": "error", "text": "No recorded events for this session"})
            return
        self.agent.resume_chat(task)
        self.printer.broadcast({"type": "task_events", "events": events})

    def _get_last_session(self) -> None:
        """Load the most recent task from history and replay its events."""
        entries = _load_history(limit=1)
        if not entries:
            return
        task = str(entries[0].get("task", ""))
        if not task:
            return
        events = _load_task_chat_events(task)
        self.agent.resume_chat(task)
        self.printer.broadcast({"type": "task_events", "events": events, "task": task})

    def _get_welcome_suggestions(self) -> None:
        """Send recent tasks as welcome screen suggestions."""
        entries = _load_history(limit=10)
        suggestions = []
        for entry in entries:
            task = str(entry.get("task", ""))
            if task:
                suggestions.append({
                    "text": task,
                    "has_events": bool(entry.get("has_events", False)),
                })
        self.printer.broadcast({"type": "welcome_suggestions", "suggestions": suggestions})

    def _generate_followup(self, task: str, result: str) -> None:
        """Generate a follow-up suggestion using LLM after task completion."""
        def _run() -> None:
            suggestion = generate_followup_text(task, result, self._selected_model)
            if suggestion:
                self.printer.broadcast({
                    "type": "followup_suggestion",
                    "text": suggestion,
                })

        threading.Thread(target=_run, daemon=True).start()

    def _extract_result_summary(self) -> str:
        """Extract result summary from the last recorded events."""
        with self.printer._lock:
            for events_list in self.printer._recordings.values():
                for ev in reversed(events_list):
                    if ev.get("type") == "result":
                        summary = ev.get("summary") or ev.get("text") or ""
                        return str(summary)
        return ""

    def _fast_complete(self, query: str) -> str:
        """Local prefix matching against history and active file words.

        Checks task history first, then falls back to matching
        identifiers/words from the currently active editor file.

        Args:
            query: The stripped query string (guaranteed non-empty, len >= 2
                by the caller ``_complete``).

        Returns:
            Continuation string if a match is found, empty string otherwise.
        """
        query_lower = query.lower()
        # Try history match first
        for entry in _load_history(limit=1000):
            task = str(entry.get("task", ""))
            if task.lower().startswith(query_lower) and len(task) > len(query):
                return task[len(query):]
        # Fall back to word/identifier matching from the active file
        return self._complete_from_active_file(query)

    def _complete_from_active_file(self, query: str) -> str:
        """Complete the last word of *query* using words from the active file.

        Reads the active editor file, extracts all identifier-like tokens
        (alphanumeric + underscores, length >= 3), and finds the longest
        match whose prefix matches the last partial word the user is typing.

        Args:
            query: The full query string from the chat input.

        Returns:
            The remaining suffix to append, or empty string if no match.
        """
        import re

        active_path = self._last_active_file
        if not active_path:
            return ""
        try:
            with open(active_path) as f:
                content = f.read(50000)
        except OSError:
            return ""
        # Extract the last partial word from the query
        m = re.search(r"(\w+)$", query)
        if not m:
            return ""
        partial = m.group(1)
        if len(partial) < 2:
            return ""
        partial_lower = partial.lower()
        # Extract unique words/identifiers from file (length >= 3)
        words = set(re.findall(r"\b[A-Za-z_]\w{2,}\b", content))
        best = ""
        for word in words:
            if word.lower().startswith(partial_lower) and len(word) > len(partial):
                suffix = word[len(partial):]
                if not best or len(word) > len(partial) + len(best):
                    best = suffix
        return best

    def _complete(self, query: str) -> None:
        """Ghost text autocomplete via fast local prefix matching."""
        query = query.strip()
        if not query or len(query) < 2:
            self.printer.broadcast({"type": "ghost", "suggestion": "", "query": query})
            return

        fast = clip_autocomplete_suggestion(
            query, self._fast_complete(query)
        )
        self.printer.broadcast({"type": "ghost", "suggestion": fast, "query": query})

    def _refresh_file_cache(self) -> None:
        """Refresh the file cache from disk."""
        from kiss.agents.vscode.diff_merge import _scan_files

        self._file_cache = _scan_files(self.work_dir)

    def _get_files(self, prefix: str) -> None:
        """Send file list for autocomplete with usage-based sorting."""
        if not self._file_cache:
            self._refresh_file_cache()
        usage = _load_file_usage()
        ranked = rank_file_suggestions(self._file_cache, prefix, usage)
        self.printer.broadcast({"type": "files", "files": ranked})

    def _generate_commit_message(self, model: str) -> None:
        """Generate a git commit message from current changes."""
        try:
            diff_result = _git(self.work_dir, "diff")
            cached_result = _git(self.work_dir, "diff", "--cached")
            diff_text = (diff_result.stdout + cached_result.stdout).strip()
            untracked = "\n".join(sorted(_capture_untracked(self.work_dir)))
            if not diff_text and not untracked:
                self.printer.broadcast({
                    "type": "commitMessage",
                    "message": "",
                    "error": "No changes detected",
                })
                return
            context_parts: list[str] = []
            if diff_text:
                context_parts.append(f"Diff:\n{diff_text}")
            if untracked:
                context_parts.append(f"New untracked files:\n{untracked[:500]}")
            agent = KISSAgent("Commit Message Generator")
            raw = agent.run(
                model_name=model,
                prompt_template=(
                    "Generate a nicely markdown formatted, informative git commit message for "
                    "these changes. Use conventional commit format with a clear subject "
                    "line (type: description) and optionally a body with bullet points "
                    "for multiple changes. Return ONLY the commit message text, no "
                    "quotes or markdown fences.\n\n{context}"
                ),
                arguments={"context": "\n\n".join(context_parts)},
                is_agentic=False,
            )
            msg = clean_llm_output(raw)
            self.printer.broadcast({"type": "commitMessage", "message": msg})
        except Exception:
            logger.debug("Commit message generation failed", exc_info=True)
            self.printer.broadcast({
                "type": "commitMessage",
                "message": "",
                "error": "Failed to generate",
            })


def main() -> None:
    """Main entry point for VS Code backend server."""
    server = VSCodeServer()
    server.run()


if __name__ == "__main__":
    main()
