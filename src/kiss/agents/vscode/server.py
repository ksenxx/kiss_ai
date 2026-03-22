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

from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.task_history import (
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

_FAST_MODEL = "gemini-2.0-flash"

_OPENAI_PREFIXES = ("gpt", "o1", "o3", "o4", "codex", "computer-use")


def _model_vendor_order(name: str) -> int:
    """Return sort order for model vendor grouping (matches web Sorcar)."""
    if name.startswith("claude-"):
        return 0
    if name.startswith(_OPENAI_PREFIXES):
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
    if name.startswith(_OPENAI_PREFIXES) and not name.startswith("openai/"):
        return "OpenAI"
    if name.startswith("gemini-"):
        return "Gemini"
    if name.startswith("minimax-"):
        return "MiniMax"
    if name.startswith("openrouter/"):
        return "OpenRouter"
    return "Together AI"


def _clean_llm_output(text: str) -> str:
    return text.strip().strip('"').strip("'")


def _clip_autocomplete_suggestion(query: str, suggestion: str) -> str:
    """Return only a short, confident autocomplete continuation."""
    s = _clean_llm_output(suggestion)
    if not s:
        return ""
    if s.lower().startswith(query.lower()):
        s = s[len(query):]
    s = s.lstrip()
    if not s:
        return ""
    words: list[str] = []
    hit_boundary = False
    for token in s.split():
        if any(mark in token for mark in ("\n", ":", ";", "!", "?")):
            hit_boundary = True
            break
        if token.endswith((".", ",")):
            clean = token.rstrip(".,")
            if clean:
                words.append(clean)
            hit_boundary = True
            break
        words.append(token)
        if len(words) >= 4:
            break
    clipped = " ".join(words).strip()
    if len(words) < 1:
        return ""
    if hit_boundary:
        return ""
    if not clipped or len(clipped) > 40:
        return ""
    if s and len(s) > len(clipped) + 20:
        return ""
    return clipped


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
        persisted = _load_last_model()
        self._selected_model = (
            persisted
            or os.environ.get("KISS_MODEL", "")
            or "claude-opus-4-6"
        )
        self._file_cache: list[str] = []
        self._task_thread: threading.Thread | None = None

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
        elif cmd_type == "complete":
            query = cmd.get("query", "")
            if query:
                threading.Thread(
                    target=self._complete, args=(query,), daemon=True
                ).start()
        else:
            self.printer.broadcast({"type": "error", "text": f"Unknown command: {cmd_type}"})

    def _run_task(self, cmd: dict[str, Any]) -> None:
        """Run the agent with the given task."""
        prompt = cmd.get("prompt", "")
        model = cmd.get("model") or self._selected_model
        work_dir = cmd.get("workDir") or self.work_dir
        active_file = cmd.get("activeFile")
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

        self._stop_event = threading.Event()
        self.printer._thread_local.stop_event = self._stop_event
        self._user_answer_event = threading.Event()

        self.printer.broadcast({"type": "status", "running": True})
        user_msg_event: dict[str, Any] = {"type": "user_msg", "text": prompt}
        if image_urls:
            user_msg_event["images"] = image_urls
        self.printer.broadcast(user_msg_event)
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
            result_summary = prompt[:200]
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
            self._refresh_file_cache()

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
        if self._user_answer_event:
            self._user_answer_event.clear()
            self._user_answer_event.wait()

    def _ask_user_question(self, question: str) -> str:
        """Callback for agent questions."""
        self.printer.broadcast({
            "type": "askUser",
            "question": question,
        })
        if self._user_answer_event:
            self._user_answer_event.clear()
            self._user_answer_event.wait()
        return self._user_answer

    def _get_models(self) -> None:
        """Send available models list with usage counts and pricing."""
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
                has_events = bool(entry.get("has_events", False))
                sessions.append({
                    "id": task,
                    "title": task[:50] + "..." if len(task) > 50 else task,
                    "timestamp": 0,
                    "preview": task[:100],
                    "has_events": has_events,
                })
        self.printer.broadcast({"type": "history", "sessions": sessions})

    def _replay_session(self, task: str) -> None:
        """Replay recorded chat events for a previous task."""
        events = _load_task_chat_events(task)
        if not events:
            self.printer.broadcast({"type": "error", "text": "No recorded events for this session"})
            return
        self.printer.broadcast({"type": "task_events", "events": events})

    def _get_welcome_suggestions(self) -> None:
        """Send recent tasks as welcome screen suggestions."""
        entries = _load_history(limit=10)
        suggestions = []
        for entry in entries:
            if isinstance(entry, dict):
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
            try:
                agent = KISSAgent("Followup Proposer")
                raw = agent.run(
                    model_name=_FAST_MODEL,
                    prompt_template=(
                        "A developer just completed this task:\n"
                        "Task: {task}\n"
                        "Result summary: {result}\n\n"
                        "Suggest ONE short, concrete follow-up task they "
                        "might want to do next. Return ONLY the task "
                        "description as a single plain-text sentence."
                    ),
                    arguments={"task": task, "result": result[:500]},
                    is_agentic=False,
                )
                suggestion = _clean_llm_output(raw)
                if suggestion:
                    self.printer.broadcast({
                        "type": "followup_suggestion",
                        "text": suggestion,
                    })
            except Exception:
                logger.debug("Followup generation failed", exc_info=True)

        threading.Thread(target=_run, daemon=True).start()

    def _complete(self, query: str) -> None:
        """Ghost text autocomplete: generate a short continuation via LLM."""
        try:
            entries = _load_history(limit=20)
            task_list = "\n".join(
                f"- {e.get('task', '')}" for e in entries if isinstance(e, dict)
            )[:1000]

            files_list = "\n".join(self._file_cache[:50]) if self._file_cache else ""

            context_parts = []
            if task_list:
                context_parts.append(f"Recent tasks:\n{task_list}")
            if files_list:
                context_parts.append(f"Project files:\n{files_list}")

            agent = KISSAgent("Autocomplete")
            raw = agent.run(
                model_name=_FAST_MODEL,
                prompt_template=(
                    "You are an inline autocomplete engine for a coding assistant. "
                    "Given the user's partial input and context, "
                    "predict only the next few words you are highly confident about. "
                    "Return ONLY the remaining text to insert, never repeating what the "
                    "user already typed. Return at most 4 words. If confidence is not high, "
                    "return empty string.\n\n"
                    + "\n\n".join(context_parts)
                    + '\n\nPartial input: "{query}"\n\n'
                ),
                arguments={"query": query},
                is_agentic=False,
            )
            suggestion = _clip_autocomplete_suggestion(query, raw)
            self.printer.broadcast({"type": "ghost", "suggestion": suggestion})
        except Exception:
            logger.debug("Autocomplete failed", exc_info=True)
            self.printer.broadcast({"type": "ghost", "suggestion": ""})

    def _refresh_file_cache(self) -> None:
        """Refresh the file cache from disk."""
        from kiss.agents.sorcar.code_server import _scan_files

        self._file_cache = _scan_files(self.work_dir)

    def _get_files(self, prefix: str) -> None:
        """Send file list for autocomplete with usage-based sorting."""
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
