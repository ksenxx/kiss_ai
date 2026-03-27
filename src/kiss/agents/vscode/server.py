"""VS Code extension backend server for Sorcar agent.

This module provides a JSON-based stdio interface between the VS Code
extension and the Sorcar agent. Commands are read from stdin as JSON
lines, and events are written to stdout as JSON lines.
"""

from __future__ import annotations

import base64
import ctypes
import itertools
import json
import logging
import os
import re
import sys
import threading
from typing import Any

from kiss.agents.sorcar.persistence import (
    _load_file_usage,
    _load_history,
    _load_last_model,
    _load_model_usage,
    _load_task_chat_events,
    _prefix_match_task,
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
    clip_autocomplete_suggestion,
    fast_model_for,
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
        self._last_active_content: str = ""
        self._state_lock = threading.Lock()
        self._task_thread: threading.Thread | None = None
        self._merging = False
        self._complete_seq = itertools.count()
        self._complete_seq_latest = -1

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
                self.printer.broadcast({"type": "status", "running": False})
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
            self._get_history(cmd.get("query"), cmd.get("offset", 0), cmd.get("generation", 0))
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
            if self._task_thread and self._task_thread.is_alive():
                return
            task = cmd.get("sessionId", "")
            if task:
                self._replay_session(task)
        elif cmd_type == "mergeAction":
            self._handle_merge_action(cmd.get("action", ""))
        elif cmd_type == "getLastSession":
            self._get_last_session()
        elif cmd_type == "newChat":
            if not (self._task_thread and self._task_thread.is_alive()):
                self.agent.new_chat()
        elif cmd_type == "complete":
            query = cmd.get("query", "")
            active_file = cmd.get("activeFile")
            active_content = cmd.get("activeFileContent")
            with self._state_lock:
                if active_file:
                    self._last_active_file = active_file
                if active_content is not None:
                    self._last_active_content = active_content
                snapshot_file = self._last_active_file
                snapshot_content = self._last_active_content
            seq = next(self._complete_seq)
            self._complete_seq_latest = seq
            if query:
                threading.Thread(
                    target=self._complete,
                    args=(query, seq, snapshot_file, snapshot_content),
                    daemon=True,
                ).start()
        elif cmd_type == "getInputHistory":
            self._get_input_history()
        elif cmd_type == "generateCommitMessage":
            threading.Thread(
                target=self._generate_commit_message, daemon=True
            ).start()
        else:
            self.printer.broadcast({"type": "error", "text": f"Unknown command: {cmd_type}"})

    def _run_task(self, cmd: dict[str, Any]) -> None:
        """Run the agent with the given task.

        An outer try/finally guarantees that ``status: running: False``
        is **always** broadcast when this method exits, regardless of
        which code-path is taken.  Previously, early returns (e.g. the
        ``_merging`` guard) and exceptions before the inner try/finally
        could leave the TypeScript ``_isRunning`` flag stuck at ``true``,
        silently dropping all subsequent task submissions.
        """
        try:
            self._run_task_inner(cmd)
        finally:
            self.printer.broadcast({"type": "status", "running": False})

    def _run_task_inner(self, cmd: dict[str, Any]) -> None:
        """Inner implementation of _run_task (without the status guarantee)."""
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

        # Immediate feedback so the user sees the task has started
        self.printer.broadcast({"type": "status", "running": True})
        self.printer.broadcast({"type": "clear"})

        # Git snapshot captures pre-task state (may be slow for large repos)
        pre_hunks = _parse_diff_hunks(work_dir)
        pre_untracked = _capture_untracked(work_dir)
        pre_file_hashes = _snapshot_files(
            work_dir, set(pre_hunks.keys()) | pre_untracked
        )
        _save_untracked_base(work_dir, pre_untracked | set(pre_hunks.keys()))

        self.printer.start_recording()
        result_summary = ""
        task_end_event: dict[str, Any] | None = None
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
            _record_model_usage(model)
            result_summary = self._extract_result_summary() or "No summary available"
            self._generate_followup_sync(prompt, result_summary, model)
            task_end_event = {"type": "task_done"}
        except KeyboardInterrupt:
            task_end_event = {"type": "task_stopped"}
        except Exception as e:  # pragma: no cover
            task_end_event = {"type": "task_error", "text": str(e)}
        finally:
            chat_events = self.printer.stop_recording()
            _set_latest_chat_events(chat_events, task=prompt, result=result_summary)
            self.printer.broadcast({"type": "tasks_updated"})
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
                    merge_json = os.path.join(merge_dir, "pending-merge.json")
                    self._start_merge_session(merge_json)
            except Exception:
                logger.debug("Merge view error", exc_info=True)
            self._refresh_file_cache()
            if task_end_event:
                self.printer.broadcast(task_end_event)

    def _start_merge_session(self, merge_json_path: str) -> bool:
        """Load merge data from disk and broadcast merge_data + merge_started events.

        Args:
            merge_json_path: Path to the pending-merge.json file.

        Returns:
            True if a merge session was started, False otherwise.
        """
        try:
            with open(merge_json_path) as f:
                merge_data = json.load(f)
            files = merge_data.get("files", [])
            if not files:
                return False
            total_hunks = sum(len(f.get("hunks", [])) for f in files)
            if total_hunks == 0:
                return False
            self._merging = True
            self.printer.broadcast({
                "type": "merge_data",
                "data": merge_data,
                "hunk_count": total_hunks,
            })
            self.printer.broadcast({"type": "merge_started"})
            return True
        except (OSError, json.JSONDecodeError, KeyError):
            logger.debug("Failed to load merge data", exc_info=True)
            return False

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
        """Signal the agent to stop.

        Sets the cooperative stop event and, if the task thread doesn't
        exit promptly, forces a ``KeyboardInterrupt`` in the task thread
        using ``ctypes.pythonapi.PyThreadState_SetAsyncExc``.  This
        handles the case where the agent is blocked in an LLM API call
        or other I/O and never reaches a cooperative ``_check_stop()``
        call.
        """
        if self._stop_event:
            self._stop_event.set()
        task_thread = self._task_thread
        if task_thread is not None and task_thread.is_alive():
            threading.Thread(
                target=self._force_stop_thread,
                args=(task_thread,),
                daemon=True,
            ).start()

    @staticmethod
    def _force_stop_thread(task_thread: threading.Thread) -> None:
        """Watchdog that forces ``KeyboardInterrupt`` in *task_thread*.

        Waits 1 second for the cooperative stop-event mechanism to work.
        If the thread is still alive, raises ``KeyboardInterrupt``
        asynchronously in it.  Retries once after 5 seconds in case the
        first exception was swallowed or the thread was in C code.
        """
        task_thread.join(timeout=1)
        for _ in range(2):
            if not task_thread.is_alive():
                return
            tid = task_thread.ident
            if tid is not None:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(tid),
                    ctypes.py_object(KeyboardInterrupt),
                )
            task_thread.join(timeout=5)

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

    def _get_history(self, query: str | None, offset: int = 0, generation: int = 0) -> None:
        """Send conversation history with pagination support."""
        if query:
            entries = _search_history(query, limit=50, offset=offset)
        else:
            entries = _load_history(limit=50, offset=offset)

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
                "chat_id": entry.get("chat_id", ""),
            })
        self.printer.broadcast({
            "type": "history", "sessions": sessions,
            "offset": offset, "generation": generation,
        })

    def _get_input_history(self) -> None:
        """Send deduplicated recent task texts for arrow-key cycling."""
        entries = _load_history(limit=100)
        seen: set[str] = set()
        tasks: list[str] = []
        for e in entries:
            task = str(e.get("task", "")).strip()
            if task and task not in seen:
                seen.add(task)
                tasks.append(task)
        self.printer.broadcast({"type": "inputHistory", "tasks": tasks})

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
        self._start_merge_session(str(_merge_data_dir() / "pending-merge.json"))

    def _generate_followup_sync(self, task: str, result: str, model: str) -> None:
        """Generate and broadcast a follow-up suggestion synchronously.

        Runs before stop_recording() so the event is persisted in saved
        chat events and survives panel re-creation / VS Code restarts.
        """
        suggestion = generate_followup_text(task, result, fast_model_for(model))
        if suggestion:
            self.printer.broadcast({
                "type": "followup_suggestion",
                "text": suggestion,
            })

    def _extract_result_summary(self) -> str:
        """Extract result summary from the last recorded events."""
        with self.printer._lock:
            for events_list in self.printer._recordings.values():
                for ev in reversed(events_list):
                    if ev.get("type") == "result":
                        summary = ev.get("summary") or ev.get("text") or ""
                        return str(summary)
        return ""

    def _fast_complete(
        self, query: str, snapshot_file: str = "", snapshot_content: str = ""
    ) -> str:
        """Local prefix matching against history and active file words.

        Checks task history first, then falls back to matching
        identifiers/words from the currently active editor file.

        Args:
            query: The stripped query string (guaranteed non-empty, len >= 2
                by the caller ``_complete``).
            snapshot_file: Atomically-captured active file path.
            snapshot_content: Atomically-captured active file content.

        Returns:
            Continuation string if a match is found, empty string otherwise.
        """
        # Try history match first (SQL prefix match, uses idx_th_task index)
        match = _prefix_match_task(query)
        if match:
            return match[len(query):]
        # Fall back to word/identifier matching from the active file
        return self._complete_from_active_file(query, snapshot_file, snapshot_content)

    def _complete_from_active_file(
        self, query: str, snapshot_file: str = "", snapshot_content: str = ""
    ) -> str:
        """Complete the trailing token of *query* using identifiers from the active file.

        Extracts single-word identifiers and dot-chained identifiers
        (e.g. ``self.method``, ``os.path.join``) from the active editor
        buffer (or falls back to reading from disk). Matches the trailing
        token of the query — which may contain dots — against all
        candidates via case-sensitive prefix matching.

        Args:
            query: The full query string from the chat input.
            snapshot_file: Atomically-captured active file path.
            snapshot_content: Atomically-captured active file content.

        Returns:
            The remaining suffix to append, or empty string if no match.
        """
        content = snapshot_content
        if not content:
            active_path = snapshot_file
            if not active_path:
                return ""
            try:
                with open(active_path) as f:
                    content = f.read(50000)
            except OSError:
                return ""

        # If the query ends with whitespace the user has moved past the word;
        # don't complete a word that is no longer being typed.
        if query != query.rstrip():
            return ""
        # Extract the trailing token (may include dots for chains)
        m = re.search(r"([\w][\w.]*)[^\w]*$", query)
        if not m:
            return ""
        partial = m.group(1)
        if len(partial) < 2:
            return ""
        # Extract single-word identifiers (length >= 3) and dot-chained identifiers
        words = set(re.findall(r"\b[A-Za-z_]\w{2,}\b", content))
        chains = set(re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b", content))
        candidates = words | chains

        best = ""
        for candidate in candidates:
            if candidate.startswith(partial) and len(candidate) > len(partial):
                suffix = candidate[len(partial):]
                if len(suffix) > len(best):
                    best = suffix
        return best

    def _complete(
        self,
        query: str,
        seq: int = -1,
        snapshot_file: str = "",
        snapshot_content: str = "",
    ) -> None:
        """Ghost text autocomplete via fast local prefix matching.

        Args:
            query: Raw query text from the chat input.
            seq: Sequence number for this request. If a newer request has
                been issued (``seq`` no longer matches the counter), this
                call exits early to avoid broadcasting stale results.
            snapshot_file: Atomically-captured active file path.
            snapshot_content: Atomically-captured active file content.
        """
        if seq >= 0 and seq != self._complete_seq_latest:
            return
        if not query or len(query) < 2:
            self.printer.broadcast({"type": "ghost", "suggestion": "", "query": query})
            return

        fast = clip_autocomplete_suggestion(
            query, self._fast_complete(query, snapshot_file, snapshot_content)
        )
        if seq >= 0 and seq != self._complete_seq_latest:
            return
        self.printer.broadcast({"type": "ghost", "suggestion": fast, "query": query})

    def _refresh_file_cache(self) -> None:
        """Refresh the file cache from disk in a background thread.

        Serves stale cache while refresh is in progress.
        """
        def _do_refresh() -> None:
            from kiss.agents.vscode.diff_merge import _scan_files
            self._file_cache = _scan_files(self.work_dir)

        threading.Thread(target=_do_refresh, daemon=True).start()

    def _get_files(self, prefix: str) -> None:
        """Send file list for autocomplete with usage-based sorting."""
        if not self._file_cache:
            # First call: scan synchronously so we have data to return.
            from kiss.agents.vscode.diff_merge import _scan_files
            self._file_cache = _scan_files(self.work_dir)
        usage = _load_file_usage()
        ranked = rank_file_suggestions(self._file_cache, prefix, usage)
        self.printer.broadcast({"type": "files", "files": ranked})

    def _generate_commit_message(self) -> None:
        """Generate a git commit message from current changes."""
        try:
            cached_result = _git(self.work_dir, "diff", "--cached")
            diff_text = cached_result.stdout.strip()
            if not diff_text:
                self.printer.broadcast({
                    "type": "commitMessage",
                    "message": "Error: No staged files.",
                })
                return
            context_parts: list[str] = []
            context_parts.append(f"Diff:\n{diff_text}")
            agent = KISSAgent("Commit Message Generator")
            raw = agent.run(
                model_name=fast_model_for(self._selected_model),
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
