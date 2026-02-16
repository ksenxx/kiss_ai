"""Browser-based chatbot for RelentlessCodingAgent."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import socket
import sys
import threading
import webbrowser
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent
from kiss.core.kiss_agent import KISSAgent
from kiss.core.printer import Printer, extract_extras, extract_path_and_lang, truncate_result

HISTORY_FILE = Path.home() / ".kiss_task_history.json"
MAX_HISTORY = 100


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def _load_history() -> list[str]:
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text())
            if isinstance(data, list):
                return [str(t) for t in data[:MAX_HISTORY]]
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_history(tasks: list[str]) -> None:
    try:
        HISTORY_FILE.write_text(json.dumps(tasks[:MAX_HISTORY]))
    except OSError:
        pass


def _find_semantic_duplicates(new_task: str, existing_tasks: list[str]) -> list[int]:
    if not existing_tasks:
        return []
    numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(existing_tasks))
    agent = KISSAgent("Task Deduplicator")
    result = agent.run(
        model_name="gemini-2.0-flash",
        prompt_template=(
            "Which existing tasks are semantically the same as the new task? "
            '"Same" means they ask for essentially the same work, just worded differently.\n\n'
            "New task: {new_task}\n\n"
            "Existing tasks:\n{existing_tasks}\n\n"
            "Return ONLY a JSON array of indices of duplicate tasks. "
            "If none are duplicates, return []. Examples: [2, 5] or []"
        ),
        arguments={"new_task": new_task, "existing_tasks": numbered},
        is_agentic=False,
    )
    try:
        start = result.index("[")
        end = result.index("]", start) + 1
        indices = json.loads(result[start:end])
        return [i for i in indices if isinstance(i, int) and 0 <= i < len(existing_tasks)]
    except (ValueError, json.JSONDecodeError):
        return []


def _add_task(task: str) -> None:
    history = _load_history()
    if task in history:
        history.remove(task)
    else:
        try:
            duplicates = _find_semantic_duplicates(task, history)
            for idx in sorted(duplicates, reverse=True):
                history.pop(idx)
        except Exception:
            pass
    history.insert(0, task)
    _save_history(history[:MAX_HISTORY])


def _scan_files(work_dir: str) -> list[str]:
    paths: list[str] = []
    skip = {
        ".git", "__pycache__", "node_modules", ".venv", "venv",
        ".tox", ".mypy_cache", ".ruff_cache", ".pytest_cache",
    }
    try:
        for root, dirs, files in os.walk(work_dir):
            depth = os.path.relpath(root, work_dir).count(os.sep)
            if depth > 3:
                dirs.clear()
                continue
            dirs[:] = sorted(d for d in dirs if d not in skip and not d.startswith("."))
            for name in sorted(files):
                paths.append(os.path.relpath(os.path.join(root, name), work_dir))
                if len(paths) >= 1000:
                    return paths
    except OSError:
        pass
    return paths


class _ChatbotPrinter(Printer):
    def __init__(self) -> None:
        self._clients: list[queue.Queue[dict[str, Any]]] = []
        self._lock = threading.Lock()
        self._current_block_type = ""
        self._tool_name = ""
        self._tool_json_buffer = ""

    def reset(self) -> None:
        self._current_block_type = ""
        self._tool_name = ""
        self._tool_json_buffer = ""

    @staticmethod
    def _parse_result_yaml(raw: str) -> dict[str, Any] | None:
        try:
            data = yaml.safe_load(raw)
        except Exception:
            return None
        if isinstance(data, dict) and "summary" in data:
            return data
        return None

    def broadcast(self, event: dict[str, Any]) -> None:
        with self._lock:
            for cq in self._clients:
                cq.put(event)

    def add_client(self) -> queue.Queue[dict[str, Any]]:
        cq: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._lock:
            self._clients.append(cq)
        return cq

    def remove_client(self, cq: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if cq in self._clients:
                self._clients.remove(cq)

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:
        if type == "text":
            from io import StringIO

            from rich.console import Console

            buf = StringIO()
            Console(file=buf, highlight=False, width=120, no_color=True).print(content)
            text = buf.getvalue()
            if text.strip():
                self.broadcast({"type": "text_delta", "text": text})
            return ""
        if type == "prompt":
            self.broadcast({"type": "prompt", "text": str(content)})
            return ""
        if type == "stream_event":
            return self._handle_stream_event(content)
        if type == "message":
            self._handle_message(content, **kwargs)
            return ""
        if type == "usage_info":
            self.broadcast({"type": "usage_info", "text": str(content).strip()})
            return ""
        if type == "tool_call":
            self.broadcast({"type": "text_end"})
            self._format_tool_call(str(content), kwargs.get("tool_input", {}))
            return ""
        if type == "tool_result":
            self.broadcast({
                "type": "tool_result",
                "content": truncate_result(str(content)),
                "is_error": kwargs.get("is_error", False),
            })
            return ""
        if type == "result":
            self.broadcast({"type": "text_end"})
            event: dict[str, Any] = {
                "type": "result",
                "text": str(content) or "(no result)",
                "step_count": kwargs.get("step_count", 0),
                "total_tokens": kwargs.get("total_tokens", 0),
                "cost": kwargs.get("cost", "N/A"),
            }
            parsed = self._parse_result_yaml(str(content)) if content else None
            if parsed:
                event["success"] = parsed.get("success")
                event["summary"] = str(parsed["summary"])
            self.broadcast(event)
            return ""
        return ""

    async def token_callback(self, token: str) -> None:
        if token:
            delta_type = (
                "thinking_delta"
                if self._current_block_type == "thinking"
                else "text_delta"
            )
            self.broadcast({"type": delta_type, "text": token})

    def _format_tool_call(self, name: str, tool_input: dict[str, Any]) -> None:
        file_path, lang = extract_path_and_lang(tool_input)
        event: dict[str, Any] = {"type": "tool_call", "name": name}
        if file_path:
            event["path"] = file_path
            event["lang"] = lang
        if desc := tool_input.get("description"):
            event["description"] = str(desc)
        if command := tool_input.get("command"):
            event["command"] = str(command)
        if content := tool_input.get("content"):
            event["content"] = str(content)
        if (old := tool_input.get("old_string")) is not None:
            event["old_string"] = str(old)
        if (new := tool_input.get("new_string")) is not None:
            event["new_string"] = str(new)
        extras = extract_extras(tool_input)
        if extras:
            event["extras"] = extras
        self.broadcast(event)

    def _handle_stream_event(self, event: Any) -> str:
        evt = event.event
        evt_type = evt.get("type", "")
        text = ""
        if evt_type == "content_block_start":
            block = evt.get("content_block", {})
            self._current_block_type = block.get("type", "")
            if self._current_block_type == "thinking":
                self.broadcast({"type": "thinking_start"})
            elif self._current_block_type == "tool_use":
                self._tool_name = block.get("name", "?")
                self._tool_json_buffer = ""
        elif evt_type == "content_block_delta":
            delta = evt.get("delta", {})
            dt = delta.get("type", "")
            if dt == "thinking_delta":
                text = delta.get("thinking", "")
            elif dt == "text_delta":
                text = delta.get("text", "")
            elif dt == "input_json_delta":
                self._tool_json_buffer += delta.get("partial_json", "")
        elif evt_type == "content_block_stop":
            bt = self._current_block_type
            if bt == "thinking":
                self.broadcast({"type": "thinking_end"})
            elif bt == "tool_use":
                try:
                    ti = json.loads(self._tool_json_buffer)
                except (json.JSONDecodeError, ValueError):
                    ti = {"_raw": self._tool_json_buffer}
                self._format_tool_call(self._tool_name, ti)
            else:
                self.broadcast({"type": "text_end"})
            self._current_block_type = ""
        return text

    def _handle_message(self, message: Any, **kwargs: Any) -> None:
        if hasattr(message, "subtype") and hasattr(message, "data"):
            if message.subtype == "tool_output":
                text = message.data.get("content", "")
                if text:
                    self.broadcast({"type": "system_output", "text": text})
        elif hasattr(message, "result"):
            event: dict[str, Any] = {
                "type": "result",
                "text": message.result or "(no result)",
                "step_count": kwargs.get("step_count", 0),
                "total_tokens": kwargs.get("total_tokens_used", 0),
                "cost": (
                    f"${kwargs.get('budget_used', 0.0):.4f}"
                    if kwargs.get("budget_used")
                    else "N/A"
                ),
            }
            parsed = self._parse_result_yaml(message.result) if message.result else None
            if parsed:
                event["success"] = parsed.get("success")
                event["summary"] = str(parsed["summary"])
            self.broadcast(event)
        elif hasattr(message, "content"):
            for block in message.content:
                if hasattr(block, "is_error") and hasattr(block, "content"):
                    c = block.content if isinstance(block.content, str) else str(block.content)
                    self.broadcast({
                        "type": "tool_result",
                        "content": truncate_result(c),
                        "is_error": bool(block.is_error),
                    })


_printer = _ChatbotPrinter()
_running = False
_running_lock = threading.Lock()
_work_dir = os.getcwd()
_file_cache: list[str] = []
_agent_thread: threading.Thread | None = None
_proposed_tasks: list[str] = []
_proposed_lock = threading.Lock()


class _StopRequested(BaseException):
    pass


def _refresh_file_cache() -> None:
    global _file_cache
    _file_cache = _scan_files(_work_dir)


def _refresh_proposed_tasks() -> None:
    global _proposed_tasks
    history = _load_history()
    if not history:
        with _proposed_lock:
            _proposed_tasks = []
        return
    task_list = "\n".join(f"- {t}" for t in history[:20])
    agent = KISSAgent("Task Proposer")
    try:
        result = agent.run(
            model_name="gemini-2.0-flash",
            prompt_template=(
                "Based on these past tasks a developer has worked on, suggest 5 new "
                "tasks they might want to do next. Tasks should be natural follow-ups, "
                "related improvements, or complementary work.\n\n"
                "Past tasks:\n{task_list}\n\n"
                "Return ONLY a JSON array of 5 short task description strings. "
                'Example: ["Add unit tests for X", "Refactor Y module"]'
            ),
            arguments={"task_list": task_list},
            is_agentic=False,
        )
        start = result.index("[")
        end = result.index("]", start) + 1
        proposals = json.loads(result[start:end])
        proposals = [str(p) for p in proposals if isinstance(p, str) and p.strip()][:5]
    except Exception:
        proposals = []
    with _proposed_lock:
        _proposed_tasks = proposals


def _run_agent_thread(task: str) -> None:
    global _running, _agent_thread
    try:
        _add_task(task)
        _printer.broadcast({"type": "clear"})
        agent = RelentlessCodingAgent("Chatbot")
        agent.run(
            prompt_template=task,
            work_dir=_work_dir,
            printer=_printer,
        )
        _printer.broadcast({"type": "task_done"})
    except _StopRequested:
        _printer.broadcast({"type": "task_stopped"})
    except Exception as e:
        _printer.broadcast({"type": "task_error", "text": str(e)})
    finally:
        with _running_lock:
            _running = False
            _agent_thread = None
        _refresh_file_cache()
        try:
            _refresh_proposed_tasks()
        except Exception:
            pass


def _stop_agent() -> bool:
    with _running_lock:
        thread = _agent_thread
    if thread is None or not thread.is_alive():
        return False
    import ctypes

    tid = thread.ident
    if tid is None:
        return False
    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid),
        ctypes.py_object(_StopRequested),
    )
    return True


_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KISS Chatbot</title>
<link rel="stylesheet"
 href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.1/marked.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0d1117;--surface:#161b22;
  --surface2:#1c2128;--border:#30363d;
  --text:#e6edf3;--dim:#8b949e;
  --accent:#58a6ff;--green:#3fb950;
  --red:#f85149;--yellow:#d29922;
  --cyan:#79c0ff;--purple:#bc8cff;
}
body{
  font-family:-apple-system,BlinkMacSystemFont,
    'Segoe UI',Helvetica,Arial,sans-serif;
  background:var(--bg);color:var(--text);
  line-height:1.6;height:100vh;
  display:flex;flex-direction:column;overflow:hidden;
}
header{
  background:linear-gradient(135deg,
    var(--surface) 0%,var(--surface2) 100%);
  border-bottom:1px solid var(--border);
  padding:12px 24px;display:flex;
  align-items:center;
  justify-content:space-between;flex-shrink:0;
}
.logo{
  font-size:18px;font-weight:700;
  color:var(--accent);letter-spacing:-.3px;
}
.logo span{
  color:var(--dim);font-weight:400;
  font-size:14px;margin-left:8px;
}
.status{
  display:flex;align-items:center;
  gap:8px;font-size:13px;color:var(--dim);
}
.dot{
  width:8px;height:8px;
  border-radius:50%;background:var(--dim);
}
.dot.running{
  background:var(--green);animation:pulse 2s infinite;
}
@keyframes pulse{
  0%,100%{opacity:1}50%{opacity:.3}
}
#output{
  flex:2;overflow-y:auto;padding:16px 24px;
  scroll-behavior:smooth;
  border-bottom:1px solid var(--border);min-height:0;
}
.ev{margin-bottom:6px;animation:fadeIn .15s ease}
@keyframes fadeIn{
  from{opacity:0;transform:translateY(3px)}
  to{opacity:1;transform:none}
}
.think{
  border-left:3px solid var(--cyan);
  padding:10px 16px;margin:10px 0;
  background:rgba(121,192,255,.04);
  border-radius:0 8px 8px 0;
  max-height:200px;overflow-y:auto;
}
.think .lbl{
  font-size:11px;font-weight:600;
  text-transform:uppercase;letter-spacing:.06em;
  color:var(--cyan);margin-bottom:4px;
  display:flex;align-items:center;gap:6px;
  cursor:pointer;user-select:none;
}
.think .lbl .arrow{
  transition:transform .2s;display:inline-block;
}
.think .lbl .arrow.collapsed{
  transform:rotate(-90deg);
}
.think .cnt{
  font-size:13px;color:var(--dim);
  font-style:italic;white-space:pre-wrap;
  word-break:break-word;
}
.think .cnt.hidden{display:none}
.txt{
  font-size:14px;white-space:pre-wrap;
  word-break:break-word;padding:2px 0;line-height:1.7;
}
.tc{
  border:1px solid var(--border);border-radius:8px;
  margin:10px 0;overflow:hidden;
  background:var(--surface);transition:box-shadow .2s;
}
.tc:hover{box-shadow:0 2px 12px rgba(0,0,0,.3)}
.tc-h{
  padding:9px 14px;background:var(--surface2);
  display:flex;align-items:center;gap:10px;
  cursor:pointer;user-select:none;
}
.tc-h:hover{background:rgba(48,54,61,.8)}
.tc-h .chv{
  color:var(--dim);transition:transform .2s;
  font-size:11px;flex-shrink:0;
}
.tc-h .chv.open{transform:rotate(90deg)}
.tn{font-weight:600;font-size:13px;color:var(--accent)}
.tp{
  font-size:12px;color:var(--cyan);
  font-family:'SF Mono','Fira Code',monospace;
}
.td{font-size:12px;color:var(--dim);font-style:italic}
.tc-b{
  padding:10px 14px;max-height:300px;overflow-y:auto;
  font-family:'SF Mono','Fira Code',monospace;
  font-size:12px;line-height:1.5;
}
.tc-b.hide{display:none}
.tc-b pre{
  margin:4px 0;white-space:pre-wrap;
  word-break:break-word;
}
.diff-old{
  color:var(--red);background:rgba(248,81,73,.08);
  padding:2px 6px;border-radius:3px;
  display:block;margin:2px 0;
}
.diff-new{
  color:var(--green);background:rgba(63,185,80,.08);
  padding:2px 6px;border-radius:3px;
  display:block;margin:2px 0;
}
.extra{color:var(--dim);margin:2px 0}
.tr{
  border-left:3px solid var(--green);
  padding:8px 14px;margin:6px 0;
  border-radius:0 8px 8px 0;
  font-family:'SF Mono','Fira Code',monospace;
  font-size:12px;max-height:200px;overflow-y:auto;
  white-space:pre-wrap;word-break:break-word;
  background:rgba(63,185,80,.04);
}
.tr.err{
  border-left-color:var(--red);
  background:rgba(248,81,73,.04);
}
.tr .rl{
  font-size:11px;font-weight:600;
  text-transform:uppercase;letter-spacing:.06em;
  margin-bottom:4px;
}
.tr .rl.ok{color:var(--green)}
.tr .rl.fail{color:var(--red)}
.rc{
  border:2px solid var(--green);border-radius:10px;
  margin:20px 0;overflow:hidden;
  background:var(--surface);
}
.rc-h{
  padding:14px 20px;background:rgba(63,185,80,.08);
  display:flex;align-items:center;
  justify-content:space-between;
}
.rc-h h3{
  color:var(--green);font-size:15px;font-weight:600;
}
.rs{
  font-size:12px;color:var(--dim);
  display:flex;gap:18px;
}
.rs b{color:var(--text);font-weight:500}
.rc-body{
  padding:16px 20px;font-size:14px;
  max-height:400px;overflow-y:auto;
  white-space:pre-wrap;word-break:break-word;
  line-height:1.7;
}
.prompt{
  border:1px solid var(--cyan);border-radius:8px;
  margin:10px 0;overflow:hidden;
  background:var(--surface);
}
.prompt-h{
  padding:8px 16px;
  background:rgba(121,192,255,.08);
  font-size:12px;font-weight:600;color:var(--cyan);
  text-transform:uppercase;letter-spacing:.04em;
}
.prompt-body{
  padding:12px 16px;font-size:14px;
  white-space:pre-wrap;word-break:break-word;
  line-height:1.6;max-height:400px;overflow-y:auto;
}
.sys{
  font-size:13px;color:var(--dim);
  font-family:'SF Mono','Fira Code',monospace;
  white-space:pre-wrap;word-break:break-word;
  padding:2px 0;
}
.usage{
  border:1px solid var(--border);border-radius:4px;
  margin:6px 0;padding:4px 12px;
  background:var(--surface);font-size:11px;
  color:var(--dim);font-style:italic;
  font-family:'SF Mono','Fira Code',monospace;
  white-space:nowrap;overflow-x:auto;
}
.empty-msg{
  text-align:center;color:var(--dim);
  padding:80px 20px;font-size:15px;line-height:2;
}
#input-area{
  flex-shrink:0;padding:12px 24px;
  background:var(--surface);
  border-bottom:1px solid var(--border);
  position:relative;
}
#input-row{display:flex;gap:10px;align-items:center}
#task-input{
  flex:1;background:var(--bg);
  border:1px solid var(--border);border-radius:8px;
  padding:10px 14px;color:var(--text);
  font-size:14px;font-family:inherit;outline:none;
  transition:border-color .2s;
}
#task-input:focus{border-color:var(--accent)}
#task-input:disabled{opacity:.5;cursor:not-allowed}
#send-btn{
  background:var(--accent);color:#fff;border:none;
  border-radius:8px;padding:10px 20px;
  font-size:14px;font-weight:600;cursor:pointer;
  transition:background .2s;white-space:nowrap;
}
#send-btn:hover{background:#79b8ff}
#send-btn:disabled{opacity:.5;cursor:not-allowed}
#stop-btn{
  background:var(--red);color:#fff;border:none;
  border-radius:8px;padding:10px 20px;
  font-size:14px;font-weight:600;cursor:pointer;
  transition:background .2s;white-space:nowrap;
  display:none;
}
#stop-btn:hover{background:#f9706a}
.spinner{
  display:flex;align-items:center;gap:10px;
  padding:16px 0;color:var(--dim);font-size:13px;
  animation:fadeIn .3s ease;
}
.spinner::before{
  content:'';width:16px;height:16px;
  border:2px solid var(--border);
  border-top-color:var(--accent);
  border-radius:50%;flex-shrink:0;
  animation:spin .8s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}
#autocomplete{
  position:absolute;bottom:100%;left:24px;right:24px;
  background:var(--surface2);
  border:1px solid var(--border);border-radius:8px;
  max-height:220px;overflow-y:auto;
  display:none;z-index:10;
  box-shadow:0 -4px 16px rgba(0,0,0,.4);
}
.ac-item{
  padding:8px 14px;cursor:pointer;font-size:13px;
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:8px;
}
.ac-item:last-child{border-bottom:none}
.ac-item:hover,.ac-item.sel{
  background:rgba(88,166,255,.1);
}
.ac-type{
  font-size:10px;font-weight:600;
  text-transform:uppercase;padding:2px 6px;
  border-radius:3px;flex-shrink:0;
}
.ac-type.task{
  background:rgba(188,140,255,.15);color:var(--purple);
}
.ac-type.file{
  background:rgba(63,185,80,.15);color:var(--green);
}
.ac-text{
  overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;
}
#bottom-panel{
  flex:1;display:flex;gap:1px;
  background:var(--border);min-height:0;
}
.panel-col{
  flex:1;overflow-y:auto;padding:10px 16px;
  background:var(--bg);min-height:0;
}
.panel-hdr{
  font-size:11px;font-weight:600;
  text-transform:uppercase;letter-spacing:.06em;
  color:var(--dim);margin-bottom:8px;padding:0 2px;
}
.task-item{
  padding:8px 12px;background:var(--surface);
  border:1px solid var(--border);border-radius:6px;
  margin-bottom:6px;cursor:pointer;font-size:13px;
  transition:all .15s;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap;
}
.task-item:hover{
  border-color:var(--accent);background:var(--surface2);
}
.proposed-item{
  padding:8px 12px;background:var(--surface);
  border:1px solid var(--border);border-radius:6px;
  margin-bottom:6px;cursor:pointer;font-size:13px;
  transition:all .15s;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap;
}
.proposed-item:hover{
  border-color:var(--purple);background:var(--surface2);
}
.no-tasks{color:var(--dim);font-size:13px;padding:8px 2px}
.loading-proposals{
  color:var(--dim);font-size:12px;
  font-style:italic;padding:8px 2px;
}
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{
  background:var(--border);border-radius:3px;
}
::-webkit-scrollbar-thumb:hover{background:var(--dim)}
code{font-family:'SF Mono','Fira Code',monospace}
.hljs{
  background:var(--surface2)!important;
  border-radius:6px;padding:10px!important;
}
</style>
</head>
<body>
<header>
  <div class="logo">KISS Chatbot
    <span>Interactive Agent</span></div>
  <div class="status">
    <div class="dot" id="dot"></div>
    <span id="stxt">Ready</span></div>
</header>
<div id="output">
  <div class="empty-msg">
    Enter a task below to start the agent.<br>
    The output will appear here.
  </div>
</div>
<div id="input-area">
  <div id="autocomplete"></div>
  <div id="input-row">
    <input type="text" id="task-input"
      placeholder="Describe your task..."
      autocomplete="off">
    <button id="send-btn">Send</button>
    <button id="stop-btn">Stop</button>
  </div>
</div>
<div id="bottom-panel">
  <div class="panel-col">
    <div class="panel-hdr">Recent Tasks</div>
    <div id="recent-list"></div>
  </div>
  <div class="panel-col">
    <div class="panel-hdr">Proposed Tasks</div>
    <div id="proposed-list"></div>
  </div>
</div>
<script>
var O=document.getElementById('output');
var D=document.getElementById('dot');
var ST=document.getElementById('stxt');
var inp=document.getElementById('task-input');
var btn=document.getElementById('send-btn');
var stopBtn=document.getElementById('stop-btn');
var ac=document.getElementById('autocomplete');
var rl=document.getElementById('recent-list');
var pl=document.getElementById('proposed-list');
var running=false,autoScroll=true;
var thinkEl=null,txtEl=null,scrollRaf=0;
var acIdx=-1,t0=null,timerIv=null,evtSrc=null;
O.addEventListener('scroll',function(){
  autoScroll=O.scrollTop+O.clientHeight
    >=O.scrollHeight-60;
});
function sb(){
  if(!scrollRaf){scrollRaf=requestAnimationFrame(
    function(){
      if(autoScroll)O.scrollTop=O.scrollHeight;
      scrollRaf=0;
    });}
}
function esc(t){
  var d=document.createElement('div');
  d.textContent=t;return d.innerHTML;
}
function toggleTC(el){
  el.nextElementSibling.classList.toggle('hide');
  el.querySelector('.chv').classList.toggle('open');
}
function toggleThink(el){
  var p=el.parentElement;
  p.querySelector('.cnt').classList.toggle('hidden');
  el.querySelector('.arrow').classList.toggle(
    'collapsed');
}
function startTimer(){
  t0=Date.now();
  if(timerIv)clearInterval(timerIv);
  timerIv=setInterval(function(){
    var s=Math.floor((Date.now()-t0)/1000);
    var m=Math.floor(s/60);
    ST.textContent='Running '
      +(m>0?m+'m ':'')+s%60+'s';
  },1000);
}
function stopTimer(){
  if(timerIv){clearInterval(timerIv);timerIv=null;}
}
function removeSpinner(){
  var sp=document.getElementById('wait-spinner');
  if(sp)sp.remove();
}
function showSpinner(){
  removeSpinner();
  var sp=mkEl('div','spinner');
  sp.id='wait-spinner';
  sp.textContent='Waiting for agent output\u2026';
  O.appendChild(sp);sb();
}
function setReady(label){
  running=false;D.classList.remove('running');
  stopTimer();removeSpinner();
  ST.textContent=label||'Ready';
  inp.disabled=false;
  btn.style.display='';
  stopBtn.style.display='none';
  inp.focus();
}
function connectSSE(){
  if(evtSrc)evtSrc.close();
  evtSrc=new EventSource('/events');
  evtSrc.onmessage=function(e){
    var ev;try{ev=JSON.parse(e.data);}catch(x){return;}
    handleEvent(ev);
  };
  evtSrc.onerror=function(){};
}
function mkEl(tag,cls){
  var e=document.createElement(tag);
  if(cls)e.className=cls;return e;
}
function handleEvent(ev){
  var t=ev.type;
  if(t==='thinking_start'||t==='text_delta'
    ||t==='tool_call'||t==='task_done'
    ||t==='task_error'||t==='task_stopped')
    removeSpinner();
  switch(t){
  case'clear':
    O.innerHTML='';thinkEl=null;
    txtEl=null;autoScroll=true;
    showSpinner();break;
  case'thinking_start':
    thinkEl=mkEl('div','ev think');
    thinkEl.innerHTML=
      '<div class="lbl" onclick="toggleThink(this)">'
      +'<span class="arrow">\u25BE</span>'
      +' Thinking</div><div class="cnt"></div>';
    O.appendChild(thinkEl);break;
  case'thinking_delta':
    if(thinkEl){
      var tc=thinkEl.querySelector('.cnt');
      tc.textContent+=ev.text||'';
      thinkEl.scrollTop=thinkEl.scrollHeight;
    }break;
  case'thinking_end':
    if(thinkEl){
      thinkEl.querySelector('.lbl').innerHTML=
        '<span class="arrow collapsed">\u25BE</span>'
        +' Thinking (click to expand)';
      thinkEl.querySelector('.cnt')
        .classList.add('hidden');
    }thinkEl=null;
    if(running)showSpinner();break;
  case'text_delta':
    if(!txtEl){
      txtEl=mkEl('div','txt');O.appendChild(txtEl);
    }txtEl.textContent+=ev.text||'';break;
  case'text_end':txtEl=null;break;
  case'tool_call':{
    var c=mkEl('div','ev tc');
    var h='<span class="chv open">\u25B6</span>'
      +'<span class="tn">'+esc(ev.name)+'</span>';
    if(ev.path)h+='<span class="tp"> '
      +esc(ev.path)+'</span>';
    if(ev.description)h+='<span class="td"> '
      +esc(ev.description)+'</span>';
    var b='';
    if(ev.command)b+='<pre><code class='
      +'"language-bash">'
      +esc(ev.command)+'</code></pre>';
    if(ev.content){
      var lc=ev.lang?'language-'+esc(ev.lang):'';
      b+='<pre><code class="'+lc+'">'
        +esc(ev.content)+'</code></pre>';
    }
    if(ev.old_string!==undefined)
      b+='<div class="diff-old">- '
        +esc(ev.old_string)+'</div>';
    if(ev.new_string!==undefined)
      b+='<div class="diff-new">+ '
        +esc(ev.new_string)+'</div>';
    if(ev.extras){
      for(var k in ev.extras)
        b+='<div class="extra">'
          +esc(k)+': '+esc(ev.extras[k])+'</div>';
    }
    var body=b||('<em style="color:var(--dim)">'
      +'No arguments</em>');
    c.innerHTML=
      '<div class="tc-h" onclick="toggleTC(this)">'
      +h+'</div><div class="tc-b'
      +(b?'':' hide')+'">'+body+'</div>';
    O.appendChild(c);
    c.querySelectorAll('pre code').forEach(
      function(bl){hljs.highlightElement(bl);});
    break;}
  case'tool_result':{
    var r=mkEl('div',
      'ev tr'+(ev.is_error?' err':''));
    var lb=ev.is_error?'FAILED':'OK';
    var lc2=ev.is_error?'fail':'ok';
    r.innerHTML='<div class="rl '+lc2+'">'
      +lb+'</div>'+esc(ev.content);
    O.appendChild(r);
    if(running)showSpinner();break;}
  case'system_output':{
    var s=mkEl('div','ev sys');
    s.textContent=ev.text||'';
    O.appendChild(s);break;}
  case'result':{
    var rc=mkEl('div','ev rc');
    var rb='';
    if(ev.success!==undefined){
      var sl=ev.success
        ?'color:var(--green);font-weight:700'
        :'color:var(--red);font-weight:700';
      var stLabel=ev.success?'PASSED':'FAILED';
      rb+='<div style="'+sl
        +';font-size:16px;margin-bottom:12px">'
        +'Status: '+stLabel+'</div>';
    }
    if(ev.summary){
      rb+=typeof marked!=='undefined'
        ?marked.parse(ev.summary):esc(ev.summary);
    }else{
      rb+=esc(ev.text||'(no result)');
    }
    rc.innerHTML=
      '<div class="rc-h"><h3>Result</h3>'
      +'<div class="rs">'
      +'Steps: <b>'+(ev.step_count||0)+'</b>'
      +' &nbsp; Tokens: <b>'
      +(ev.total_tokens||0)+'</b>'
      +' &nbsp; Cost: <b>'
      +(ev.cost||'N/A')+'</b>'
      +'</div></div>'
      +'<div class="rc-body">'+rb+'</div>';
    rc.querySelectorAll('pre code').forEach(
      function(bl){hljs.highlightElement(bl);});
    O.appendChild(rc);break;}
  case'prompt':{
    var p=mkEl('div','ev prompt');
    p.innerHTML='<div class="prompt-h">Prompt</div>'
      +'<div class="prompt-body">'
      +esc(ev.text||'')+'</div>';
    O.appendChild(p);break;}
  case'usage_info':{
    var u=mkEl('div','ev usage');
    u.textContent=ev.text||'';
    O.appendChild(u);break;}
  case'task_done':{
    var el=t0?Math.floor((Date.now()-t0)/1000):0;
    var em=Math.floor(el/60);
    setReady('Done ('+
      (em>0?em+'m ':'')+el%60+'s)');
    loadTasks();loadProposed();break;}
  case'task_error':{
    var err=mkEl('div','ev tr err');
    err.innerHTML='<div class="rl fail">ERROR</div>'
      +esc(ev.text||'Unknown error');
    O.appendChild(err);
    setReady('Error');loadTasks();loadProposed();break;}
  case'task_stopped':{
    var stEl=mkEl('div','ev tr err');
    stEl.innerHTML=
      '<div class="rl fail">STOPPED</div>'
      +'Agent execution stopped by user';
    O.appendChild(stEl);
    setReady('Stopped');loadTasks();loadProposed();break;}
  }
  sb();
}
function submitTask(){
  var task=inp.value.trim();
  if(!task||running)return;
  running=true;inp.disabled=true;
  btn.style.display='none';
  stopBtn.style.display='inline-block';
  D.classList.add('running');hideAC();startTimer();
  fetch('/run',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({task:task})
  }).then(function(r){
    if(!r.ok){
      r.json().then(function(d){
        setReady('Error');
        alert(d.error||'Failed');
      });return;
    }
    inp.value='';
  }).catch(function(){
    setReady('Error');alert('Network error');
  });
}
btn.addEventListener('click',submitTask);
stopBtn.addEventListener('click',function(){
  fetch('/stop',{method:'POST'})
  .catch(function(){});
});
inp.addEventListener('keydown',function(e){
  if(ac.style.display==='block'){
    var items=ac.querySelectorAll('.ac-item');
    if(e.key==='ArrowDown'){
      e.preventDefault();
      acIdx=Math.min(acIdx+1,items.length-1);
      updateACSel(items);return;
    }
    if(e.key==='ArrowUp'){
      e.preventDefault();
      acIdx=Math.max(acIdx-1,-1);
      updateACSel(items);return;
    }
    if(e.key==='Enter'&&acIdx>=0){
      e.preventDefault();
      items[acIdx].click();return;
    }
    if(e.key==='Escape'){hideAC();return;}
  }
  if(e.key==='Enter'&&!e.shiftKey){
    e.preventDefault();submitTask();
  }
});
var acTimer=null;
inp.addEventListener('input',function(){
  if(acTimer)clearTimeout(acTimer);
  acTimer=setTimeout(fetchAC,200);
});
function fetchAC(){
  var q=inp.value.trim();
  if(!q){hideAC();return;}
  fetch('/suggestions?q='+encodeURIComponent(q))
  .then(function(r){return r.json();})
  .then(function(data){
    if(!data.length){hideAC();return;}
    ac.innerHTML='';acIdx=-1;
    data.forEach(function(item){
      var d=mkEl('div','ac-item');
      d.innerHTML=
        '<span class="ac-type '+item.type+'">'
        +item.type+'</span>'
        +'<span class="ac-text">'
        +esc(item.text)+'</span>';
      d.addEventListener('click',function(){
        if(item.type==='task'){
          inp.value=item.text;
        }else{
          var parts=inp.value.split(/\s/);
          parts[parts.length-1]=item.text;
          inp.value=parts.join(' ');
        }
        hideAC();inp.focus();
      });
      ac.appendChild(d);
    });
    ac.style.display='block';
  }).catch(function(){hideAC();});
}
function hideAC(){ac.style.display='none';acIdx=-1;}
function updateACSel(items){
  items.forEach(function(it,i){
    it.classList.toggle('sel',i===acIdx);
  });
  if(acIdx>=0)
    items[acIdx].scrollIntoView({block:'nearest'});
}
document.addEventListener('click',function(e){
  if(!ac.contains(e.target)&&e.target!==inp)
    hideAC();
});
function loadTasks(){
  fetch('/tasks')
  .then(function(r){return r.json();})
  .then(function(tasks){
    rl.innerHTML='';
    if(!tasks.length){
      rl.innerHTML=
        '<div class="no-tasks">'
        +'No recent tasks yet</div>';
      return;
    }
    tasks.forEach(function(t){
      var d=mkEl('div','task-item');
      d.textContent=t;d.title=t;
      d.addEventListener('click',function(){
        inp.value=t;inp.focus();
      });
      rl.appendChild(d);
    });
  }).catch(function(){});
}
function loadProposed(){
  pl.innerHTML='<div class="loading-proposals">'
    +'Loading suggestions\u2026</div>';
  fetch('/proposed_tasks')
  .then(function(r){return r.json();})
  .then(function(tasks){
    pl.innerHTML='';
    if(!tasks.length){
      pl.innerHTML=
        '<div class="no-tasks">'
        +'No suggestions yet</div>';
      return;
    }
    tasks.forEach(function(t){
      var d=mkEl('div','proposed-item');
      d.textContent=t;d.title=t;
      d.addEventListener('click',function(){
        inp.value=t;inp.focus();
      });
      pl.appendChild(d);
    });
  }).catch(function(){
    pl.innerHTML=
      '<div class="no-tasks">'
      +'Could not load suggestions</div>';
  });
}
connectSSE();loadTasks();loadProposed();inp.focus();
</script>
</body>
</html>"""


def main() -> None:
    global _work_dir

    import uvicorn
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
    from starlette.routing import Route

    _work_dir = str(Path(sys.argv[1]).resolve()) if len(sys.argv) > 1 else os.getcwd()
    _refresh_file_cache()

    async def index(request: Request) -> HTMLResponse:
        return HTMLResponse(_HTML_PAGE)

    async def events(request: Request) -> StreamingResponse:
        cq = _printer.add_client()

        async def generate() -> AsyncGenerator[str]:
            try:
                while True:
                    try:
                        event = cq.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.05)
                        continue
                    yield f"data: {json.dumps(event)}\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                _printer.remove_client(cq)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def run_task(request: Request) -> JSONResponse:
        global _running, _agent_thread
        with _running_lock:
            if _running:
                return JSONResponse(
                    {"error": "Agent is already running"},
                    status_code=409,
                )
            _running = True
        body = await request.json()
        task = body.get("task", "").strip()
        if not task:
            with _running_lock:
                _running = False
            return JSONResponse(
                {"error": "Empty task"}, status_code=400,
            )
        t = threading.Thread(
            target=_run_agent_thread,
            args=(task,),
            daemon=True,
        )
        with _running_lock:
            _agent_thread = t
        t.start()
        return JSONResponse({"status": "started"})

    async def stop_task(request: Request) -> JSONResponse:
        if _stop_agent():
            return JSONResponse({"status": "stopping"})
        return JSONResponse(
            {"error": "No running task"}, status_code=404,
        )

    async def suggestions(request: Request) -> JSONResponse:
        query = request.query_params.get("q", "").strip()
        if not query:
            return JSONResponse([])
        q_lower = query.lower()
        results: list[dict[str, str]] = []
        for task in _load_history():
            if q_lower in task.lower():
                results.append({"type": "task", "text": task})
                if len(results) >= 5:
                    break
        last_word = query.split()[-1].lower() if query.split() else q_lower
        if last_word:
            count = 0
            for path in _file_cache:
                if last_word in path.lower():
                    results.append({"type": "file", "text": path})
                    count += 1
                    if count >= 10:
                        break
        return JSONResponse(results)

    async def tasks(request: Request) -> JSONResponse:
        return JSONResponse(_load_history())

    async def proposed_tasks(request: Request) -> JSONResponse:
        with _proposed_lock:
            return JSONResponse(list(_proposed_tasks))

    app = Starlette(routes=[
        Route("/", index),
        Route("/events", events),
        Route("/run", run_task, methods=["POST"]),
        Route("/stop", stop_task, methods=["POST"]),
        Route("/suggestions", suggestions),
        Route("/tasks", tasks),
        Route("/proposed_tasks", proposed_tasks),
    ])

    threading.Thread(target=_refresh_proposed_tasks, daemon=True).start()

    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"KISS Chatbot running at {url}")
    print(f"Work directory: {_work_dir}")
    webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
