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
from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
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
    generate_followup_text,
    model_vendor,
    rank_file_suggestions,
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
        elif cmd_type == "mergeAction":
            self._handle_merge_action(cmd.get("action", ""))
        elif cmd_type == "getLastSession":
            self._get_last_session()
        elif cmd_type == "newChat":
            self.agent.new_chat()
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
        if raw_attachments:
            attachments = []
            for att in raw_attachments:
                data_b64 = att.get("data", "")
                mime = att.get("mimeType", "application/octet-stream")
                data = base64.b64decode(data_b64)
                attachments.append(Attachment(data=data, mime_type=mime))

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
                    hunk_count = merge_result.get("hunk_count", 0)
                    merge_json = os.path.join(merge_dir, "pending-merge.json")
                    if os.path.exists(merge_json):
                        with open(merge_json) as f:
                            merge_data = json.load(f)
                        self.printer.broadcast({
                            "type": "merge_data",
                            "data": merge_data,
                            "hunk_count": hunk_count,
                        })
                    self.printer.broadcast({"type": "merge_started"})
            except Exception:
                logger.debug("Merge view error", exc_info=True)
            self._refresh_file_cache()

    def _handle_merge_action(self, action: str) -> None:
        """Handle merge accept/reject actions from the extension.

        Only ``all-done`` triggers cleanup. Individual ``accept``/``reject``
        actions are tracked on the TypeScript side; the Python server
        only needs to know when the entire merge session is finished.
        """
        if action == "all-done":
            self._finish_merge()

    def _finish_merge(self) -> None:
        """End the merge session: reset state, notify clients, clean up data."""
        self._merging = False
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
        """Load the most recent task from history and replay its events.

        Also restores any pending merge session from disk so that
        merge-diff buttons reappear after a VS Code restart.
        """
        entries = _load_history(limit=1)
        if entries:
            task = str(entries[0].get("task", ""))
            if task:
                events = _load_task_chat_events(task)
                self.agent.resume_chat(task)
                self.printer.broadcast({"type": "task_events", "events": events, "task": task})
        self._restore_pending_merge()

    def _restore_pending_merge(self) -> None:
        """Restore a pending merge session from disk if one exists.

        Reads ``pending-merge.json`` from the merge data directory and
        sends ``merge_data`` and ``merge_started`` events so the VS Code
        extension re-opens the merge view with decorations and the
        webview shows the accept/reject toolbar.
        """
        merge_json = _merge_data_dir() / "pending-merge.json"
        if not merge_json.is_file():
            return
        try:
            with open(merge_json) as f:
                merge_data = json.load(f)
            files = merge_data.get("files", [])
            if not files:
                return
            total_hunks = sum(len(f.get("hunks", [])) for f in files)
            if total_hunks == 0:
                return
            self._merging = True
            self.printer.broadcast({
                "type": "merge_data",
                "data": merge_data,
                "hunk_count": total_hunks,
            })
            self.printer.broadcast({"type": "merge_started"})
        except (OSError, json.JSONDecodeError, KeyError):
            logger.debug("Failed to restore pending merge", exc_info=True)

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
