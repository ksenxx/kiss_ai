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
import queue
import re
import sys
import threading
from typing import Any

from kiss.agents.sorcar.persistence import (
    _append_chat_event,
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
from kiss.agents.vscode.diff_merge import _git
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
from kiss.core.models.model_info import MODEL_INFO, get_available_models, get_default_model

logger = logging.getLogger(__name__)

ctypes.pythonapi.PyThreadState_SetAsyncExc.argtypes = [
    ctypes.c_ulong,
    ctypes.py_object,
]


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
        self._user_answer_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        persisted = _load_last_model()
        self._selected_model = (
            persisted
            or os.environ.get("KISS_MODEL", "")
            or get_default_model()
        )
        self._file_cache: list[str] = []
        self._last_active_file: str = ""
        self._last_active_content: str = ""
        # Lock ordering: _state_lock < printer._lock < printer._stdout_lock < printer._bash_lock
        self._state_lock = threading.Lock()
        self._task_thread: threading.Thread | None = None
        self._task_generation = 0
        self._recording_id = 0
        self._complete_seq = itertools.count()
        self._complete_seq_latest = -1
        self._complete_lock = threading.Lock()
        self._complete_queue: queue.Queue[tuple[str, int, str, str]] = queue.Queue()
        self._complete_worker = threading.Thread(
            target=self._complete_worker_loop, daemon=True
        )
        self._complete_worker.start()
        self._task_history_id: int | None = None
        self._flush_interval: float = 5  # seconds between crash-recovery flushes

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
            with self._state_lock:
                if self._task_thread and self._task_thread.is_alive():
                    self.printer.broadcast({"type": "error", "text": "Task already running"})
                    self.printer.broadcast({"type": "status", "running": False})
                    return
                self._task_thread = threading.Thread(
                    target=self._run_task, args=(cmd,), daemon=True
                )
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
            # Drain any stale answer, then put the new one (P2/D3 fix)
            while not self._user_answer_queue.empty():
                try:
                    self._user_answer_queue.get_nowait()
                except queue.Empty:  # pragma: no cover — race guard
                    break
            self._user_answer_queue.put(cmd.get("answer", ""))
        elif cmd_type == "resumeSession":
            if self._task_thread and self._task_thread.is_alive():
                return
            task = cmd.get("sessionId", "")
            if task:
                self._replay_session(task)
        elif cmd_type == "getLastSession":
            self._get_last_session()
        elif cmd_type == "newChat":
            with self._state_lock:
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
            with self._complete_lock:
                self._complete_seq_latest = seq
            if query:
                self._complete_queue.put((query, seq, snapshot_file, snapshot_content))
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
        which code-path is taken.
        """
        try:
            self.printer.broadcast({"type": "status", "running": True})
            self._run_task_inner(cmd)
        finally:
            with self._state_lock:
                self._task_thread = None
            self.printer.broadcast({"type": "status", "running": False})

    def _periodic_event_flush(self, rec_id: int, stop: threading.Event) -> None:
        """Periodically flush recorded events to DB for crash recovery.

        Runs in a background daemon thread.  Every ``_flush_interval``
        seconds it snapshots the in-memory recording and writes it to the
        database.  If the agent process is killed before the task's
        ``finally`` block runs, the most recent flush ensures partial
        events survive in the DB and can be replayed later.

        Args:
            rec_id: Recording ID to peek at.
            stop: Event signaled when the task completes normally.
        """
        while not stop.wait(self._flush_interval):
            task_id = self.agent._last_task_id
            if task_id is not None:
                events = self.printer.peek_recording(rec_id)
                if events:
                    _set_latest_chat_events(events, task_id=task_id, result=None)

    def _run_task_inner(self, cmd: dict[str, Any]) -> None:
        """Inner implementation of _run_task (without the status guarantee)."""
        prompt = cmd.get("prompt", "")
        model = cmd.get("model") or self._selected_model
        work_dir = cmd.get("workDir") or self.work_dir
        active_file = cmd.get("activeFile")
        with self._state_lock:
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

        self._stop_event = threading.Event()
        self.printer._thread_local.stop_event = self._stop_event
        # Drain stale answers from previous task (P2 fix)
        while not self._user_answer_queue.empty():
            try:
                self._user_answer_queue.get_nowait()
            except queue.Empty:  # pragma: no cover — race guard
                break

        self.printer.broadcast({"type": "clear"})

        # Increment task generation so stale followups are suppressed (P12 fix)
        with self._state_lock:
            self._task_generation += 1
            gen = self._task_generation

        # Use a unique recording ID instead of thread ident (P16 fix)
        self._recording_id += 1
        rec_id = self._recording_id

        # start_recording inside try so stop_recording always runs (P14 fix)
        result_summary = "Agent Failed Abruptly"
        task_end_event: dict[str, Any] | None = None
        flush_stop = threading.Event()
        flush_thread = threading.Thread(
            target=self._periodic_event_flush,
            args=(rec_id, flush_stop),
            daemon=True,
        )
        flush_thread.start()
        try:
            self.printer.start_recording(rec_id)
            self._task_history_id = None
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
                self._task_history_id = self.agent._last_task_id
                result_summary = self._extract_result_summary(rec_id) or "No summary available"
                task_end_event = {"type": "task_done"}
            except KeyboardInterrupt:
                self._task_history_id = self.agent._last_task_id
                result_summary = "Task stopped by user"
                task_end_event = {"type": "task_stopped"}
            except Exception as e:  # pragma: no cover
                self._task_history_id = self.agent._last_task_id
                result_summary = f"Task failed: {e}"
                task_end_event = {"type": "task_error", "text": str(e)}
        except BaseException:  # pragma: no cover — async interrupt before inner try
            # P14: interrupt before inner try — ensure stop_recording runs
            task_end_event = task_end_event or {"type": "task_stopped"}
        finally:
            flush_stop.set()
            flush_thread.join(timeout=2)
            _record_model_usage(model)
            # Entire cleanup wrapped in try/except BaseException (P13 fix)
            try:
                chat_events = self.printer.stop_recording(rec_id)
                if task_end_event:  # pragma: no branch — always set
                    chat_events.append(task_end_event)
                _set_latest_chat_events(
                    chat_events,
                    task_id=self._task_history_id,
                    task=prompt,
                    result=result_summary,
                )
                self.printer.broadcast({"type": "tasks_updated"})
                self.printer.reset()
                self._stop_event = None
                self._refresh_file_cache()
                if task_end_event:  # pragma: no branch — always set
                    self.printer.broadcast(task_end_event)
                if self._task_history_id is not None:
                    self._generate_followup_async(
                        prompt,
                        result_summary,
                        model,
                        gen,
                        self._task_history_id,
                    )
                self._task_history_id = None
            except BaseException:  # pragma: no cover — cleanup interrupted
                logger.debug("Cleanup interrupted", exc_info=True)
                if task_end_event:
                    self.printer.broadcast(task_end_event)

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
        for _ in range(2):  # pragma: no branch — thread always dies within 2 attempts
            if not task_thread.is_alive():
                return
            tid = task_thread.ident
            if tid is not None:  # pragma: no branch — running thread always has ident
                rc = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(tid),
                    ctypes.py_object(KeyboardInterrupt),
                )
                if rc == 0:
                    # Thread ID not found — thread already exited
                    return
                if rc > 1:  # pragma: no cover — rare: exception set in multiple states
                    # Undo: clear the exception in all affected thread states
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_ulong(tid), None
                    )
            task_thread.join(timeout=5)

    def _await_user_response(self) -> str:
        """Block until the user sends a response, checking stop_event periodically.

        Returns:
            The user's answer string.

        Raises:
            KeyboardInterrupt: If the stop event is set before an answer arrives.
        """
        stop = getattr(self.printer._thread_local, "stop_event", None) or self.printer.stop_event
        while True:
            try:
                return self._user_answer_queue.get(timeout=0.5)
            except queue.Empty:
                if stop.is_set():
                    raise KeyboardInterrupt("Stopped while waiting for user")

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
        return self._await_user_response()

    def _get_models(self) -> None:
        """Send available models list with usage counts and pricing."""
        usage = _load_model_usage()
        models_list: list[dict[str, Any]] = []
        sort_keys: dict[str, tuple[int, float]] = {}
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
                })
                price = float(info.input_price_per_1M) + float(info.output_price_per_1M)
                sort_keys[name] = (vendor_order, -price)
        models_list.sort(key=lambda m: sort_keys[m["name"]])
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
        """Send deduplicated task texts for arrow-key cycling.

        Loads the full persisted history so ArrowUp can traverse every
        distinct task stored in ``history.db``, not just an arbitrary
        recent subset.
        """
        entries = _load_history()
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
        self.printer.broadcast({"type": "task_events", "events": events, "task": task})

    def _get_last_session(self) -> None:
        """Load the most recent task from history and replay its events."""
        entries = _load_history(limit=1)
        if entries:
            task = str(entries[0].get("task", ""))
            if task:
                events = _load_task_chat_events(task)
                self.agent.resume_chat(task)
                self.printer.broadcast({"type": "task_events", "events": events, "task": task})

    def _generate_followup_async(
        self, task: str, result: str, model: str, gen: int, task_id: int | None
    ) -> None:
        """Generate and broadcast a follow-up suggestion in a background thread.

        The suggestion is broadcast to the webview and also appended to
        the persisted chat events so it survives panel re-creation.
        Stale followups from a previous task are suppressed by checking
        the generation counter.

        Args:
            task: The completed task description.
            result: The task result summary.
            model: The model used for the task.
            gen: Task generation counter at time of launch.
            task_id: Stable history row id for the completed task.
        """
        def _run() -> None:
            try:
                suggestion = generate_followup_text(
                    task, result, fast_model_for()
                )
                if suggestion:  # pragma: no cover — requires LLM API call
                    # P12 fix: only broadcast if still same task generation
                    if not self._is_current_task_generation(gen):
                        return  # pragma: no cover
                    event: dict[str, object] = {
                        "type": "followup_suggestion",
                        "text": suggestion,
                    }
                    self.printer.broadcast(event)
                    _append_chat_event(event, task_id=task_id, task=task)
            except Exception:  # pragma: no cover — LLM API error handler
                logger.debug("Async followup generation failed", exc_info=True)

        threading.Thread(target=_run, daemon=True).start()

    def _is_current_task_generation(self, gen: int) -> bool:
        """Return whether *gen* still matches the current task generation."""
        with self._state_lock:
            return self._task_generation == gen

    def _extract_result_summary(self, recording_id: int) -> str:
        """Extract result summary from the recorded events for the given recording.

        Args:
            recording_id: The recording ID to extract the summary from.
        """
        events = self.printer.peek_recording(recording_id)
        for ev in reversed(events):
            if ev.get("type") == "result":
                summary = ev.get("summary") or ev.get("text") or ""
                return str(summary)
        return ""

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

    def _complete_worker_loop(self) -> None:
        """Persistent worker that drains the complete queue."""
        while True:
            item = self._complete_queue.get()
            # Drain to latest request (skip stale ones)
            while not self._complete_queue.empty():
                try:
                    item = self._complete_queue.get_nowait()
                except queue.Empty:  # pragma: no cover — race guard
                    break
            query, seq, snapshot_file, snapshot_content = item
            self._complete(query, seq, snapshot_file, snapshot_content)

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
        with self._complete_lock:
            if seq >= 0 and seq != self._complete_seq_latest:
                return
        if not query or len(query) < 2:
            self.printer.broadcast({"type": "ghost", "suggestion": "", "query": query})
            return

        match = _prefix_match_task(query)
        if match:
            fast = match[len(query):]
        else:
            fast = self._complete_from_active_file(query, snapshot_file, snapshot_content)
        fast = clip_autocomplete_suggestion(query, fast)
        self.printer.broadcast({"type": "ghost", "suggestion": fast, "query": query})

    def _refresh_file_cache(self) -> None:
        """Refresh the file cache from disk in a background thread.

        Serves stale cache while refresh is in progress.
        """
        def _do_refresh() -> None:
            from kiss.agents.vscode.diff_merge import _scan_files
            result = _scan_files(self.work_dir)
            with self._state_lock:
                self._file_cache = result

        threading.Thread(target=_do_refresh, daemon=True).start()

    def _get_files(self, prefix: str) -> None:
        """Send file list for autocomplete with usage-based sorting."""
        with self._state_lock:
            cache = self._file_cache
        if not cache:
            from kiss.agents.vscode.diff_merge import _scan_files
            cache = _scan_files(self.work_dir)
            with self._state_lock:
                self._file_cache = cache
        usage = _load_file_usage()
        ranked = rank_file_suggestions(cache, prefix, usage)
        self.printer.broadcast({"type": "files", "files": ranked})

    def _generate_commit_message(self) -> None:
        """Generate a git commit message from current changes."""
        try:
            cached_result = _git(self.work_dir, "diff", "--cached")
            diff_text = cached_result.stdout.strip()
            if not diff_text:  # pragma: no branch — LLM API required for else
                self.printer.broadcast({
                    "type": "commitMessage",
                    "message": "",
                    "error": "No staged changes found. Stage files with 'git add' first.",
                })
                return
            self._generate_commit_message_llm(diff_text)  # pragma: no cover
        except Exception:  # pragma: no cover — LLM API error handler
            logger.debug("Commit message generation failed", exc_info=True)
            self.printer.broadcast({
                "type": "commitMessage",
                "message": "",
                "error": "Failed to generate",
            })

    def _generate_commit_message_llm(self, diff_text: str) -> None:  # pragma: no cover
        """Call LLM to generate commit message from diff text."""
        agent = KISSAgent("Commit Message Generator")
        raw = agent.run(
            model_name=fast_model_for(),
            prompt_template=(
                "Generate a nicely markdown formatted, informative git commit message for "
                "these changes. Use conventional commit format with a clear subject "
                "line (type: description) and optionally a body with bullet points "
                "for multiple changes. Return ONLY the commit message text, no "
                "quotes or markdown fences.\n\n{context}"
            ),
            arguments={"context": f"Diff:\n{diff_text}"},
            is_agentic=False,
            verbose=False,
        )
        msg = clean_llm_output(raw)
        self.printer.broadcast({"type": "commitMessage", "message": msg})


def main() -> None:  # pragma: no cover — CLI entry point
    """Main entry point for VS Code backend server."""
    server = VSCodeServer()
    server.run()


if __name__ == "__main__":
    main()
