"""Browser-based chatbot for RelentlessAgent-based agents."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import types
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter, find_free_port
from kiss.agents.sorcar.chatbot_ui import _THEME_PRESETS, _build_html
from kiss.agents.sorcar.code_server import (
    _capture_untracked,
    _cleanup_merge_data,
    _git,
    _parse_diff_hunks,
    _prepare_merge_view,
    _restore_merge_files,
    _save_untracked_base,
    _scan_files,
    _setup_code_server,
    _snapshot_files,
)
from kiss.agents.sorcar.shared_utils import (
    FAST_MODEL,
    clean_llm_output,
    clip_autocomplete_suggestion,
    generate_followup_text,
    model_vendor,
    rank_file_suggestions,
)
from kiss.agents.sorcar.task_history import (
    _KISS_DIR,
    _RECENT_CACHE_SIZE,
    _add_task,
    _cleanup_stale_cs_dirs,
    _get_history_entry,
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
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model_info import (
    MODEL_INFO,
    get_available_models,
)
from kiss.core.relentless_agent import RelentlessAgent

logger = logging.getLogger(__name__)


def _log_exc() -> None:
    logger.debug("Exception caught", exc_info=True)

_INTERNAL_MODELS = frozenset({FAST_MODEL})


class _StopRequested(BaseException):
    pass


def _atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* atomically using a temp file and os.replace()."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _read_active_file(sorcar_data_dir: str) -> str:
    try:
        af_path = os.path.join(sorcar_data_dir, "active-file.json")
        with open(af_path) as af:
            path: str = json.loads(af.read()).get("path", "")
        if path and os.path.isfile(path):
            return path
    except (OSError, json.JSONDecodeError):
        _log_exc()
    return ""


def _generate_commit_msg(diff_text: str, *, detailed: bool = False) -> str:
    if detailed:
        prompt = (
            "Generate a nicely markdown formatted, informative git commit message for "
            "these changes. Use conventional commit format with a clear subject "
            "line (type: description) and optionally a body with bullet points "
            "for multiple changes. Return ONLY the commit message text, no "
            "quotes or markdown fences.\n\n{context}"
        )
    else:
        prompt = (
            "Generate a concise git commit message (1-2 lines) for these changes. "
            "Return ONLY the commit message text, no quotes.\n\n{context}"
        )
    agent = KISSAgent("Commit Message Generator")
    try:
        raw = agent.run(
            model_name=FAST_MODEL,
            prompt_template=prompt,
            arguments={"context": diff_text},
            is_agentic=False,
        )
        return clean_llm_output(raw)
    except Exception:  # pragma: no cover – LLM API failure
        _log_exc()
        return ""


def run_chatbot(
    agent_factory: Callable[[str], RelentlessAgent],
    title: str = "KISS Sorcar",
    work_dir: str | None = None,
    default_model: str = "claude-opus-4-6",
    agent_kwargs: dict[str, Any] | None = None,
) -> None:
    """Run a browser-based chatbot UI for any RelentlessAgent-based agent.

    Args:
        agent_factory: Callable that takes a name string and returns a RelentlessAgent instance.
        title: Title displayed in the browser tab.
        work_dir: Working directory for the agent. Defaults to current directory.
        default_model: Default LLM model name for the model selector.
        agent_kwargs: Additional keyword arguments passed to agent.run().
    """
    import uvicorn
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
    from starlette.routing import Route

    printer = BaseBrowserPrinter()
    running = False
    running_lock = threading.Lock()
    shutting_down = threading.Event()
    merging = False
    remaining_hunks = 0
    actual_work_dir = work_dir or os.getcwd()
    file_cache: list[str] = _scan_files(actual_work_dir)
    agent_thread: threading.Thread | None = None
    current_stop_event: threading.Event | None = None
    user_action_event: threading.Event | None = None
    user_question_event: threading.Event | None = None
    user_question_answer: str = ""
    last = _load_last_model()
    selected_model = last if last and last not in _INTERNAL_MODELS else default_model

    # Clean up stale code-server data directories synchronously at startup
    _cleanup_stale_cs_dirs()

    cs_proc: subprocess.Popen[bytes] | None = None
    code_server_url = ""
    sorcar_data_dir = str(_KISS_DIR / "sorcar-data")

    # Remove stale VS Code profile leftovers that may linger from before
    # code-server switched to the shared desktop VS Code data directory.
    for _name in (
        "CachedExtensionVSIXs", "CachedProfilesData",
        "code-server-ipc.sock", "coder.json", "logs",
        "Machine", "serve-web-key-half", "User",
    ):
        _stale = Path(sorcar_data_dir) / _name
        if not _stale.exists():  # pragma: no branch
            continue
        if _stale.is_dir():  # pragma: no cover – only on upgrade
            shutil.rmtree(_stale, ignore_errors=True)
        else:  # pragma: no cover – only on upgrade
            _stale.unlink(missing_ok=True)
    # Use the standard VS Code data directory so code-server shares
    # desktop VS Code's auth sessions, settings, and secret storage.
    vscode_data_dir = str(Path.home() / "Library" / "Application Support" / "Code")
    # All instances share a single extensions directory so extensions
    # are installed once and reused across work dirs.
    _shared_extensions_dir = str(_KISS_DIR / "cs-extensions")

    # Restore files from any stale merge state (e.g., previous crash during merge).
    # If hunks were recovered, the merge view will re-open automatically via
    # pending-merge.json when the extension starts.
    recovered = _restore_merge_files(sorcar_data_dir, actual_work_dir)
    if not recovered:
        # _restore_merge_files only detects merge-current dir.  When Sorcar
        # shuts down cleanly during an active merge, _cleanup already called
        # _restore_merge_files (which deletes merge-current), but left
        # pending-merge.json for the next startup.  Check for it here.
        pending_merge_path = Path(sorcar_data_dir) / "pending-merge.json"
        if pending_merge_path.exists():
            try:
                pm_data = json.loads(pending_merge_path.read_text())
                recovered = sum(
                    len(f.get("hunks", [])) for f in pm_data.get("files", [])
                )
            except (json.JSONDecodeError, OSError):
                _log_exc()
    if recovered:
        merging = True
        remaining_hunks = recovered

    # Read or assign a code-server port.  The port is stored in a
    # persistent file so the browser origin stays stable, preserving
    # localStorage-based secrets.
    cs_port_file = Path(sorcar_data_dir) / "cs-port"
    cs_port_file.parent.mkdir(parents=True, exist_ok=True)
    cs_port = 0
    if cs_port_file.exists():  # pragma: no cover – port file from previous run
        try:
            cs_port = int(cs_port_file.read_text().strip())
        except (ValueError, OSError):
            _log_exc()
    if not cs_port:  # pragma: no branch – cs_port always 0 on fresh start
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
            _s.bind(("", 0))
            cs_port = int(_s.getsockname()[1])
    try:
        _atomic_write_text(cs_port_file, str(cs_port))
    except OSError:  # pragma: no cover – filesystem permission error
        _log_exc()
    cs_url = f"http://127.0.0.1:{cs_port}"
    cs_binary = shutil.which("code-server")

    def _build_cs_env() -> dict[str, str]:  # pragma: no cover – requires code-server binary
        """Build environment dict for code-server with gallery URL."""
        return {
            **os.environ,
            "EXTENSIONS_GALLERY": (
                '{"serviceUrl":"https://marketplace.visualstudio.com/_apis/public/gallery",'
                '"itemUrl":"https://marketplace.visualstudio.com/items"}'
            ),
        }

    def _wait_for_cs() -> bool:  # pragma: no cover – requires code-server binary
        """Wait up to 15s for code-server to accept connections. Sets code_server_url on success."""
        nonlocal code_server_url
        for _ in range(30):
            try:
                with socket.create_connection(("127.0.0.1", cs_port), timeout=0.5):
                    code_server_url = cs_url
                    return True
            except (ConnectionRefusedError, OSError):
                _log_exc()
                time.sleep(0.5)
        return False

    def _code_server_launch_args() -> list[str]:  # pragma: no cover – requires code-server binary
        """Build code-server CLI arguments."""
        assert cs_binary is not None
        return [
            cs_binary,
            "--port",
            str(cs_port),
            "--auth",
            "none",
            "--bind-addr",
            f"127.0.0.1:{cs_port}",
            "--disable-telemetry",
            "--user-data-dir",
            vscode_data_dir,
            "--extensions-dir",
            _shared_extensions_dir,
            "--disable-getting-started-override",
            "--disable-workspace-trust",
            actual_work_dir,
        ]

    def _watch_code_server() -> None:  # pragma: no cover – requires code-server binary
        """Monitor code-server and restart it if it crashes.

        Works whether we started code-server (cs_proc set) or are reusing
        an existing instance (cs_proc is None).
        """
        nonlocal cs_proc, code_server_url
        while not shutting_down.is_set():
            shutting_down.wait(5.0)
            if shutting_down.is_set():
                break
            # Check if code-server is still alive
            if cs_proc is not None:
                ret = cs_proc.poll()
                if ret is None:
                    continue  # Still running
                logger.warning("code-server exited with code %d", ret)
            else:
                # We didn't start it; check if port is reachable
                try:
                    with socket.create_connection(
                        ("127.0.0.1", cs_port), timeout=0.5
                    ):
                        continue  # Still running
                except (ConnectionRefusedError, OSError):
                    pass
                logger.warning("code-server on port %d unreachable", cs_port)
            logger.info("Restarting code-server...")
            try:
                cs_proc = subprocess.Popen(
                    _code_server_launch_args(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=_build_cs_env(),
                    start_new_session=True,
                )
                if _wait_for_cs():
                    logger.info("code-server restarted at %s", code_server_url)
                    printer.broadcast({"type": "code_server_restarted"})
                else:
                    logger.warning("code-server failed to restart")
            except Exception:
                _log_exc()
    if cs_binary:
        ext_changed = _setup_code_server(vscode_data_dir, extensions_dir=_shared_extensions_dir)
        port_in_use = False
        try:
            with socket.create_connection(("127.0.0.1", cs_port), timeout=0.5):
                port_in_use = True  # pragma: no cover – requires pre-existing code-server on port
        except (ConnectionRefusedError, OSError):
            _log_exc()
        # If our stored port is not in use, verify it's still bindable;
        # if another process grabbed it, pick a fresh port.
        if not port_in_use:  # pragma: no branch – port_in_use always False on fresh start
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
                    _s.bind(("127.0.0.1", cs_port))
            except OSError:  # pragma: no cover – port stolen by another process
                cs_port = find_free_port()
                cs_url = f"http://127.0.0.1:{cs_port}"
                try:
                    _atomic_write_text(cs_port_file, str(cs_port))
                except OSError:
                    _log_exc()

        workdir_file = Path(sorcar_data_dir) / "workdir"
        prev_workdir = ""
        try:
            prev_workdir = workdir_file.read_text().strip() if workdir_file.exists() else ""
        except OSError:  # pragma: no cover – filesystem error reading workdir file
            _log_exc()
        workdir_changed = prev_workdir != actual_work_dir

        need_restart = port_in_use and (ext_changed or workdir_changed)
        if need_restart:  # pragma: no cover – requires pre-existing code-server with changed config
            reason = "extension updated" if ext_changed else "work directory changed"
            printer.print(f"Restarting code-server ({reason})...")
            try:
                result = subprocess.run(
                    ["lsof", "-ti", f":{cs_port}", "-sTCP:LISTEN"],
                    capture_output=True,
                    text=True,
                )
                for pid_str in result.stdout.strip().split("\n"):
                    if pid_str.strip():
                        os.kill(int(pid_str.strip()), 15)
                time.sleep(1.5)
            except Exception:
                _log_exc()
            port_in_use = False
        if port_in_use:  # pragma: no cover – requires pre-existing code-server
            code_server_url = cs_url
            printer.print(f"Reusing existing code-server at {code_server_url}")
        else:
            cs_proc = subprocess.Popen(
                _code_server_launch_args(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=_build_cs_env(),
                start_new_session=True,
            )
            if _wait_for_cs():
                printer.print(f"code-server running at {code_server_url}")
            else:  # pragma: no cover – code-server startup failure
                printer.print("Warning: code-server failed to start")
        if code_server_url:  # pragma: no branch – always True after successful startup
            try:
                _atomic_write_text(workdir_file, actual_work_dir)
            except OSError:  # pragma: no cover – filesystem error writing workdir
                _log_exc()

    if cs_binary and code_server_url:
        threading.Thread(target=_watch_code_server, daemon=True).start()

    html_page = _build_html(title, code_server_url, actual_work_dir)
    shutdown_handle: asyncio.TimerHandle | None = None

    def refresh_file_cache() -> None:
        nonlocal file_cache
        file_cache = _scan_files(actual_work_dir)

    def generate_followup(task: str, result: str) -> None:
        try:
            suggestion = generate_followup_text(task, result)
            if suggestion:  # pragma: no branch – LLM always returns non-empty
                printer.broadcast(
                    {
                        "type": "followup_suggestion",
                        "text": suggestion,
                    }
                )
        except Exception:  # pragma: no cover – LLM API failure
            _log_exc()

    def _watch_theme() -> None:
        """Watch for VS Code theme changes and broadcast updates."""
        theme_file = _KISS_DIR / "vscode-theme.json"
        last_mtime = 0.0
        try:
            if theme_file.exists():  # pragma: no branch – depends on system state
                last_mtime = theme_file.stat().st_mtime
        except OSError:  # pragma: no cover – filesystem error
            _log_exc()
        while not shutting_down.is_set():  # pragma: no branch – daemon thread exit
            try:  # pragma: no cover – daemon thread during server run
                if theme_file.exists():
                    mtime = theme_file.stat().st_mtime
                    if mtime > last_mtime:
                        last_mtime = mtime
                        data = json.loads(theme_file.read_text())
                        kind = data.get("kind", "dark")
                        colors = _THEME_PRESETS.get(kind, _THEME_PRESETS["dark"])
                        printer.broadcast({"type": "theme_changed", **colors})
            except (OSError, json.JSONDecodeError):  # pragma: no cover – filesystem/JSON error
                _log_exc()
            shutting_down.wait(1.0)

    threading.Thread(target=_watch_theme, daemon=True).start()

    def _wait_for_user_browser(instruction: str, url: str) -> None:  # pragma: no cover
        nonlocal user_action_event
        event = threading.Event()
        user_action_event = event
        printer.broadcast({
            "type": "user_browser_action",
            "instruction": instruction,
            "url": url,
        })
        while not event.wait(timeout=0.5):
            stop_ev = current_stop_event
            if stop_ev and stop_ev.is_set():
                user_action_event = None
                raise KeyboardInterrupt("Agent stopped while waiting for user")
        user_action_event = None

    def _ask_user_question(question: str) -> str:  # pragma: no cover
        nonlocal user_question_event, user_question_answer
        event = threading.Event()
        user_question_event = event
        user_question_answer = ""
        printer.broadcast({
            "type": "user_question",
            "question": question,
        })
        while not event.wait(timeout=0.5):
            stop_ev = current_stop_event
            if stop_ev and stop_ev.is_set():
                user_question_event = None
                raise KeyboardInterrupt("Agent stopped while waiting for user answer")
        answer = user_question_answer
        user_question_event = None
        user_question_answer = ""
        return answer

    def run_agent_thread(  # pragma: no cover – requires live LLM + server
        task: str,
        model_name: str,
        stop_ev: threading.Event,
        attachments: list | None = None,
    ) -> None:
        nonlocal running, agent_thread, merging, remaining_hunks
        from kiss.core.models.model import Attachment

        # Install per-thread stop event so _check_stop() uses this
        # thread's own event instead of the shared printer.stop_event.
        printer._thread_local.stop_event = stop_ev
        current_thread = threading.current_thread()

        parsed_attachments: list[Attachment] | None = None
        if attachments:
            parsed_attachments = []
            for att in attachments:
                data = base64.b64decode(att["data"])
                parsed_attachments.append(Attachment(data=data, mime_type=att["mime_type"]))

        pre_hunks: dict[str, list[tuple[int, int, int, int]]] = {}
        pre_untracked: set[str] = set()
        pre_file_hashes: dict[str, str] = {}
        result_text = ""
        done_event: dict[str, str] = {}
        try:
            _add_task(task)
            printer.broadcast({"type": "tasks_updated"})
            pre_hunks = _parse_diff_hunks(actual_work_dir)
            pre_untracked = _capture_untracked(actual_work_dir)
            pre_file_hashes = _snapshot_files(
                actual_work_dir, set(pre_hunks.keys()) | pre_untracked
            )
            _save_untracked_base(
                actual_work_dir, pre_untracked | set(pre_hunks.keys())
            )
            active_file = _read_active_file(sorcar_data_dir)
            printer.start_recording()
            printer.broadcast({"type": "clear", "active_file": active_file})
            agent = agent_factory("Chatbot")
            extra_kwargs = dict(agent_kwargs or {})
            if active_file:
                extra_kwargs["current_editor_file"] = active_file
            extra_kwargs.setdefault("wait_for_user_callback", _wait_for_user_browser)
            extra_kwargs.setdefault("ask_user_question_callback", _ask_user_question)
            result = agent.run(
                prompt_template=task,
                work_dir=actual_work_dir,
                printer=printer,
                model_name=model_name,
                attachments=parsed_attachments,
                **extra_kwargs,
            )
            result_text = result or ""
            done_event = {"type": "task_done"}
        except (KeyboardInterrupt, _StopRequested):
            _log_exc()
            result_text = "(stopped by user)"
            done_event = {"type": "task_stopped"}
        except Exception as e:
            _log_exc()
            result_text = f"(error: {e})"
            done_event = {"type": "task_error", "text": str(e)}
        finally:
            # Extract a concise result summary for task history
            result_summary = result_text
            try:
                parsed = yaml.safe_load(result_text)
                if isinstance(parsed, dict) and "summary" in parsed:
                    result_summary = str(parsed["summary"])
            except Exception:
                pass

            printer._thread_local.stop_event = None
            chat_events = printer.stop_recording()
            stopped_externally = False
            with running_lock:
                if agent_thread is not current_thread:
                    # Stopped externally; stop_agent already broadcast
                    # task_stopped which is captured in chat_events.
                    stopped_externally = True
                else:
                    running = False
                    agent_thread = None
            if not stopped_externally:
                # Broadcast AFTER setting running=False so clients can
                # immediately submit a new task without getting a 409.
                if done_event.get("type") == "task_error":
                    # Show the error in a proper Result card so the user
                    # sees it prominently (not just the small error bar).
                    error_result = {
                        "type": "result",
                        "text": done_event.get("text", "Unknown error"),
                        "success": False,
                        "summary": f"Error: {done_event.get('text', 'Unknown error')}",
                        "total_tokens": 0,
                        "cost": "N/A",
                    }
                    chat_events.append(error_result)
                    printer.broadcast(error_result)
                elif done_event.get("type") == "task_done":
                    # If the agent returned a failure result (e.g. non-retryable
                    # API error caught by perform_task) without broadcasting a
                    # result card, show the error in the Results panel.
                    has_result = any(
                        e.get("type") == "result" for e in chat_events
                    )
                    if not has_result:
                        try:
                            pr = yaml.safe_load(result_text)
                            if isinstance(pr, dict) and not pr.get("success", True):
                                summary = str(pr.get("summary", "Unknown error"))
                                fail_result = {
                                    "type": "result",
                                    "text": summary,
                                    "success": False,
                                    "summary": summary,
                                    "total_tokens": 0,
                                    "cost": "N/A",
                                }
                                chat_events.append(fail_result)
                                printer.broadcast(fail_result)
                        except Exception:
                            _log_exc()
                chat_events.append(done_event)
                printer.broadcast(done_event)
                if done_event.get("type") == "task_done":
                    try:
                        generate_followup(task, result_text)
                    except Exception:  # pragma: no cover – LLM API failure
                        _log_exc()
            _set_latest_chat_events(
                chat_events, task=task, result=result_summary,
            )
            try:
                merge_result = _prepare_merge_view(
                    actual_work_dir,
                    sorcar_data_dir,
                    pre_hunks,
                    pre_untracked,
                    pre_file_hashes,
                )
                if merge_result.get("status") == "opened":
                    with running_lock:
                        merging = True
                        remaining_hunks = merge_result.get("hunk_count", 0)
                    printer.broadcast({"type": "merge_started"})
            except Exception:  # pragma: no cover – merge view error
                _log_exc()
            refresh_file_cache()

    def stop_agent() -> bool:
        """Kill the current agent thread and reset state for a new task.

        Sets the thread's per-thread stop event so the agent stops at
        the next printer.print() or token_callback() check.  Also injects
        _StopRequested via PyThreadState_SetAsyncExc as a fallback.
        """
        nonlocal running, agent_thread, current_stop_event
        with running_lock:
            thread = agent_thread
            if thread is None or not thread.is_alive():
                return False
            running = False
            agent_thread = None
            # Set the per-thread stop event so only this thread sees
            # the stop signal.  New threads get their own fresh event.
            stop_ev = current_stop_event
            current_stop_event = None
        if stop_ev is not None:  # pragma: no branch – race: event cleared by thread
            stop_ev.set()
        import ctypes

        tid = thread.ident
        if tid is not None:  # pragma: no branch – race: thread already exited
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(tid),
                ctypes.py_object(_StopRequested),
            )
        printer.broadcast({"type": "task_stopped"})
        return True

    def _finish_merge() -> None:
        """End the merge session: reset state, notify clients, clean up data."""
        nonlocal merging, remaining_hunks
        with running_lock:
            merging = False
            remaining_hunks = 0
        printer.broadcast({"type": "merge_ended"})
        _cleanup_merge_data(sorcar_data_dir)

    def _cleanup() -> None:
        nonlocal merging, remaining_hunks
        with running_lock:
            was_merging = merging
            merging = False
            remaining_hunks = 0
        if was_merging:  # pragma: no cover – cleanup during active merge
            _restore_merge_files(sorcar_data_dir, actual_work_dir)
        stop_agent()
        if cs_proc and cs_proc.poll() is None:  # pragma: no cover – cleanup timing
            # Don't kill code-server if another Sorcar instance is using it.
            _kill_cs = True
            try:
                _cur_port = int(
                    (_KISS_DIR / "assistant-port").read_text().strip()
                )
                if _cur_port != port:
                    with socket.create_connection(
                        ("127.0.0.1", _cur_port), timeout=0.5
                    ):
                        _kill_cs = False  # Another live instance exists
            except (ConnectionRefusedError, OSError, ValueError):
                pass
            if _kill_cs:
                try:
                    os.killpg(cs_proc.pid, 15)  # SIGTERM to process group
                except OSError:
                    cs_proc.terminate()
                try:
                    cs_proc.wait(timeout=5)
                except Exception:
                    _log_exc()
                    try:
                        os.killpg(cs_proc.pid, 9)  # SIGKILL
                    except OSError:
                        cs_proc.kill()

    def _do_shutdown() -> None:  # pragma: no cover – timer-triggered shutdown
        with running_lock:
            if running or printer.has_clients():
                return
            shutting_down.set()
        _cleanup()
        server.should_exit = True

    def _cancel_shutdown() -> None:
        nonlocal shutdown_handle
        if shutdown_handle is not None:  # pragma: no cover – timer race
            shutdown_handle.cancel()
            shutdown_handle = None

    def _schedule_shutdown() -> None:  # pragma: no cover – timer-triggered shutdown
        """Schedule a delayed shutdown from the event loop thread.

        Called only from async handlers (SSE disconnect, /closing), so
        always runs on the event loop thread.
        """
        nonlocal shutdown_handle
        if printer.has_clients():
            return
        with running_lock:
            if running:
                return
        if shutdown_handle is not None:
            shutdown_handle.cancel()
        loop = asyncio.get_event_loop()
        shutdown_handle = loop.call_later(1.0, _do_shutdown)

    async def index(request: Request) -> HTMLResponse:
        return HTMLResponse(html_page)

    async def events(request: Request) -> StreamingResponse:  # pragma: no cover
        cq = printer.add_client()
        if merging:
            cq.put({"type": "merge_started"})
        _cancel_shutdown()

        async def generate() -> AsyncGenerator[str]:
            last_heartbeat = time.monotonic()
            disconnect_check_counter = 0
            try:
                while not shutting_down.is_set():  # pragma: no branch  # noqa: E501
                    disconnect_check_counter += 1
                    if disconnect_check_counter >= 20:
                        disconnect_check_counter = 0
                        if await request.is_disconnected():  # pragma: no cover  # noqa: E501
                            break
                    try:
                        event = cq.get_nowait()
                    except queue.Empty:
                        now = time.monotonic()
                        if now - last_heartbeat >= 5.0:
                            yield ": heartbeat\n\n"
                            last_heartbeat = now
                        await asyncio.sleep(0.05)
                        continue
                    yield f"data: {json.dumps(event)}\n\n"
                    last_heartbeat = time.monotonic()
            except asyncio.CancelledError:
                _log_exc()
            finally:
                printer.remove_client(cq)
                _schedule_shutdown()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    def _create_agent_thread(
        task: str, model: str, attachments: list | None = None,
    ) -> tuple[threading.Thread, JSONResponse | None]:
        """Create an agent thread and register it under the running lock.

        Records model usage, creates the thread (not yet started), and
        acquires the lock.  Returns ``(thread, None)`` on success, or
        ``(thread, JSONResponse)`` on conflict.  Caller must call
        ``thread.start()`` only when error is ``None``.
        """
        nonlocal running, agent_thread, current_stop_event
        _record_model_usage(model)
        stop_ev = threading.Event()
        t = threading.Thread(
            target=run_agent_thread,
            args=(task, model, stop_ev, attachments),
            daemon=True,
        )
        with running_lock:
            if merging:
                return t, JSONResponse(
                    {"error": "Resolve all diffs in the merge view first"},
                    status_code=409,
                )
            if running:
                return t, JSONResponse({"error": "Agent is already running"}, status_code=409)
            current_stop_event = stop_ev
            running = True
            agent_thread = t
        return t, None

    async def run_task(request: Request) -> JSONResponse:
        nonlocal selected_model
        body = await request.json()
        task = body.get("task", "").strip()
        model = body.get("model", "").strip() or selected_model
        attachments = body.get("attachments")
        selected_model = model
        if not task:
            return JSONResponse({"error": "Empty task"}, status_code=400)
        t, err = _create_agent_thread(task, model, attachments)
        if err is not None:
            return err
        t.start()
        return JSONResponse({"status": "started"})

    async def run_selection(request: Request) -> JSONResponse:
        """Run the agent on text selected in the VS Code editor.

        Broadcasts an ``external_run`` event so the chatbox UI enters
        running state and displays the selected text as a user message,
        then starts the agent thread with the selected text as the task.
        """
        body = await request.json()
        text = body.get("text", "").strip()
        if not text:
            return JSONResponse({"error": "No text selected"}, status_code=400)
        t, err = _create_agent_thread(text, selected_model)
        if err is not None:
            return err
        # Broadcast AFTER lock confirms task will start, so clients
        # never see external_run for a task that returns 409.
        printer.broadcast({"type": "external_run", "text": text})
        t.start()
        return JSONResponse({"status": "started"})

    async def stop_task(request: Request) -> JSONResponse:
        if stop_agent():
            return JSONResponse({"status": "stopping"})
        return JSONResponse({"error": "No running task"}, status_code=404)

    async def user_browser_done(request: Request) -> JSONResponse:  # pragma: no cover
        """Signal that the user has finished their browser interaction."""
        ev = user_action_event
        if ev is not None:
            ev.set()
            return JSONResponse({"status": "ok"})
        return JSONResponse({"error": "No pending action"}, status_code=404)

    async def user_question_done(request: Request) -> JSONResponse:  # pragma: no cover
        """Signal that the user has answered the agent's question."""
        nonlocal user_question_answer
        ev = user_question_event
        if ev is not None:
            body = await request.json()
            user_question_answer = body.get("answer", "")
            ev.set()
            return JSONResponse({"status": "ok"})
        return JSONResponse({"error": "No pending question"}, status_code=404)

    async def refresh_files(request: Request) -> JSONResponse:  # pragma: no cover
        """Refresh the file cache on demand (e.g. when user types @)."""
        refresh_file_cache()
        return JSONResponse({"status": "ok"})

    async def suggestions(request: Request) -> JSONResponse:  # pragma: no cover
        query = request.query_params.get("q", "").strip()
        mode = request.query_params.get("mode", "general")
        if mode == "files":
            usage = _load_file_usage()
            return JSONResponse(rank_file_suggestions(file_cache, query, usage))
        if not query:
            return JSONResponse([])
        results = []
        for entry in _search_history(query, limit=5):
            results.append({"type": "task", "text": str(entry["task"])})
        words = query.split()
        last_word = words[-1].lower() if words else query.lower()
        if last_word and len(last_word) >= 2:
            count = 0
            for path in file_cache:
                if last_word in path.lower():
                    results.append({"type": "file", "text": path})
                    count += 1
                    if count >= 8:
                        break
        return JSONResponse(results)

    async def tasks(request: Request) -> JSONResponse:  # pragma: no cover
        """Return task history with optional limit, offset, and search.

        Query params:
            limit: max entries (default 100, 0 = all)
            offset: skip first N entries (default 0)
            q: search substring (case-insensitive)
        """
        try:
            limit = int(request.query_params.get("limit", "100"))
        except (ValueError, TypeError):
            limit = 100
        try:
            offset = int(request.query_params.get("offset", "0"))
        except (ValueError, TypeError):
            offset = 0
        query = request.query_params.get("q", "")
        if query:
            history = _search_history(query, limit=limit + offset)
        else:
            history = _load_history(limit=limit + offset)
        page = history[offset : offset + limit] if limit > 0 else history[offset:]
        return JSONResponse(
            [
                {
                    "task": e["task"],
                    "has_events": bool(e.get("has_events")),
                    "result": e.get("result", ""),
                    "events_file": e.get("events_file", ""),
                }
                for e in page
            ]
        )

    async def task_events(request: Request) -> JSONResponse:  # pragma: no cover
        """Return chat events for a specific task by index."""
        try:
            idx = int(request.query_params.get("idx", "0"))
        except (ValueError, TypeError):
            return JSONResponse({"error": "Invalid index"}, status_code=400)
        entry = _get_history_entry(idx)
        if entry is None:
            return JSONResponse({"error": "Index out of range"}, status_code=404)
        events = _load_task_chat_events(str(entry["task"]))
        return JSONResponse(events)

    def _fast_complete(raw_query: str, query: str) -> str:
        query_lower = query.lower()
        for entry in _load_history(limit=_RECENT_CACHE_SIZE):
            task = str(entry.get("task", ""))
            if task.lower().startswith(query_lower) and len(task) > len(query):
                return task[len(query):]
        words = raw_query.split()
        last_word = words[-1] if words else ""
        if last_word and len(last_word) >= 2:
            lw_lower = last_word.lower()
            for path in file_cache:
                if path.lower().startswith(lw_lower) and len(path) > len(last_word):
                    return path[len(last_word) :]
        return ""

    async def complete(request: Request) -> JSONResponse:  # pragma: no cover
        raw_query = request.query_params.get("q", "")
        query = raw_query.strip()
        if not query or len(query) < 2:
            return JSONResponse({"suggestion": ""})

        fast = clip_autocomplete_suggestion(query, _fast_complete(raw_query, query))
        if fast:
            return JSONResponse({"suggestion": fast})

        def _generate() -> str:
            history = _load_history(limit=20)
            task_list = "\n".join(f"- {e['task']}" for e in history)
            files_list = "\n".join(file_cache[:200])
            active_path = _read_active_file(sorcar_data_dir)
            active_content = ""
            if active_path:
                try:
                    with open(active_path) as f:
                        active_content = f.read(10000)
                except OSError:
                    pass
            context_parts = [
                "Past tasks:\n{task_list}",
                "Files and folders:\n{files_list}",
            ]
            if active_content:
                context_parts.append(
                    "Active file ({active_path}):\n{active_content}"
                )
            agent = KISSAgent("Autocomplete")
            try:
                result = agent.run(
                    model_name=FAST_MODEL,
                    prompt_template=(
                        "You are an inline autocomplete engine for a coding assistant. "
                        "Given the user's partial input, their past task history, "
                        "the list of files/folders in the project, and the content of "
                        "the currently open file in the editor, "
                        "predict only the next few words you are highly confident about. "
                        "Return ONLY the remaining text to insert, never repeating what the "
                        "user already typed. Never complete a full sentence or paragraph. "
                        "Return at most 4 words, with no newline. If confidence is not high, "
                        "return empty string.\n\n"
                        + "\n\n".join(context_parts)
                        + '\n\nPartial input: "{query}"\n\n'
                    ),
                    arguments={
                        "task_list": task_list,
                        "files_list": files_list,
                        "active_path": active_path,
                        "active_content": active_content,
                        "query": query,
                    },
                    is_agentic=False,
                )
                return clip_autocomplete_suggestion(query, result)
            except Exception:  # pragma: no cover – LLM API failure
                _log_exc()
                return ""

        suggestion = await asyncio.to_thread(_generate)  # pragma: no branch
        return JSONResponse({"suggestion": suggestion})

    async def models_endpoint(request: Request) -> JSONResponse:
        usage = _load_model_usage()
        models_list: list[dict[str, Any]] = []
        for name in get_available_models():
            info = MODEL_INFO.get(name)
            if info and info.is_function_calling_supported:
                models_list.append(
                    {
                        "name": name,
                        "inp": info.input_price_per_1M,
                        "out": info.output_price_per_1M,
                        "uses": usage.get(name, 0),
                    }
                )
        models_list.sort(
            key=lambda m: (
                model_vendor(str(m["name"]))[1],
                -(float(m["inp"]) + float(m["out"])),
            )
        )
        return JSONResponse({"models": models_list, "selected": selected_model})

    async def select_model_endpoint(request: Request) -> JSONResponse:  # pragma: no cover
        """Update the selected model when user picks from the dropdown."""
        nonlocal selected_model
        body = await request.json()
        name = body.get("model", "").strip()
        if not name:
            return JSONResponse({"error": "No model"}, status_code=400)
        selected_model = name
        _save_last_model(name)
        return JSONResponse({"status": "ok"})

    async def get_ui_state(request: Request) -> JSONResponse:  # pragma: no cover
        """Return saved UI state (divider position, etc.)."""
        ui_state_file = os.path.join(sorcar_data_dir, "ui-state.json")
        try:
            if os.path.exists(ui_state_file):
                with open(ui_state_file) as f:
                    return JSONResponse(json.load(f))
        except (OSError, json.JSONDecodeError):
            _log_exc()
        return JSONResponse({})

    async def save_ui_state(request: Request) -> JSONResponse:  # pragma: no cover
        """Save UI state (divider position, etc.)."""
        body = await request.json()
        ui_state_file = os.path.join(sorcar_data_dir, "ui-state.json")
        try:
            Path(sorcar_data_dir).mkdir(parents=True, exist_ok=True)
            with open(ui_state_file, "w") as f:
                json.dump(body, f)
        except OSError:
            _log_exc()
        return JSONResponse({"status": "ok"})

    async def closing(request: Request) -> JSONResponse:
        """Handle browser tab/window closing. Schedule a quick shutdown."""
        _schedule_shutdown()
        return JSONResponse({"status": "ok"})

    async def focus_chatbox(request: Request) -> JSONResponse:
        printer.broadcast({"type": "focus_chatbox"})
        return JSONResponse({"status": "ok"})

    async def focus_editor(request: Request) -> JSONResponse:
        pending = os.path.join(sorcar_data_dir, "pending-focus-editor.json")
        with open(pending, "w") as f:
            json.dump({"focus": True}, f)
        return JSONResponse({"status": "ok"})

    async def theme(request: Request) -> JSONResponse:
        theme_file = _KISS_DIR / "vscode-theme.json"
        kind = "dark"
        if theme_file.exists():
            try:
                data = json.loads(theme_file.read_text())
                kind = data.get("kind", "dark")
            except (json.JSONDecodeError, OSError):
                _log_exc()
        return JSONResponse(_THEME_PRESETS.get(kind, _THEME_PRESETS["dark"]))

    async def open_file(request: Request) -> JSONResponse:
        body = await request.json()
        rel = body.get("path", "").strip()
        if not rel:
            return JSONResponse({"error": "No path"}, status_code=400)
        full = rel if rel.startswith("/") else os.path.join(actual_work_dir, rel)
        if not os.path.isfile(full):
            return JSONResponse({"error": "File not found"}, status_code=404)
        pending = os.path.join(sorcar_data_dir, "pending-open.json")
        with open(pending, "w") as f:
            json.dump({"path": full}, f)
        return JSONResponse({"status": "ok"})

    async def merge_action(request: Request) -> JSONResponse:  # pragma: no cover
        nonlocal remaining_hunks
        body = await request.json()
        action = body.get("action", "")
        if action == "all-done":
            _finish_merge()
            return JSONResponse({"status": "ok"})
        if action not in ("prev", "next", "accept-all", "reject-all", "accept", "reject"):
            return JSONResponse({"error": "Invalid action"}, status_code=400)
        pending = os.path.join(sorcar_data_dir, "pending-action.json")
        with open(pending, "w") as f:
            json.dump({"action": action}, f)
        if action in ("accept-all", "reject-all"):
            _finish_merge()
        elif action in ("accept", "reject"):
            done = False
            with running_lock:
                remaining_hunks = max(0, remaining_hunks - 1)
                done = remaining_hunks == 0
            if done:
                _finish_merge()
        return JSONResponse({"status": "ok"})

    async def _thread_json_response(
        fn: Callable[[], dict[str, str]],
        error_status: int = 400,
    ) -> JSONResponse:
        result = await asyncio.to_thread(fn)  # pragma: no branch
        if "error" in result:
            return JSONResponse(result, status_code=error_status)
        return JSONResponse(result)
    async def commit(request: Request) -> JSONResponse:
        def _do_commit() -> dict[str, str]:
            try:
                subprocess.run(["git", "add", "-A"], cwd=actual_work_dir)
                diff_stat = _git(actual_work_dir, "diff", "--cached", "--stat")
                if not diff_stat.stdout.strip():
                    return {"error": "No changes to commit"}
                diff_detail = _git(actual_work_dir, "diff", "--cached")
                message = _generate_commit_msg(diff_detail.stdout)
                commit_env = {
                    **os.environ,
                    "GIT_COMMITTER_NAME": "KISS Sorcar",
                    "GIT_COMMITTER_EMAIL": "ksen@berkeley.edu",
                }
                result = subprocess.run(
                    ["git", "commit", "-m", message, "--author=KISS Sorcar <ksen@berkeley.edu>"],
                    capture_output=True,
                    text=True,
                    cwd=actual_work_dir,
                    env=commit_env,
                )
                if result.returncode != 0:
                    return {"error": result.stderr.strip()}
                return {"status": "ok", "message": message}
            except Exception as e:  # pragma: no cover – git/LLM error
                _log_exc()
                return {"error": str(e)}

        return await _thread_json_response(_do_commit)

    async def record_file_usage_endpoint(
        request: Request,
    ) -> JSONResponse:
        body = await request.json()
        path = body.get("path", "").strip()
        if path:
            _record_file_usage(path)
        return JSONResponse({"status": "ok"})

    async def generate_commit_message(request: Request) -> JSONResponse:
        """Generate a git commit message from current diff and fill the SCM input."""

        def _generate() -> dict[str, str]:
            try:
                diff_result = _git(actual_work_dir, "diff")
                cached_result = _git(actual_work_dir, "diff", "--cached")
                diff_text = (diff_result.stdout + cached_result.stdout).strip()
                untracked_files = "\n".join(sorted(_capture_untracked(actual_work_dir)))
                if not diff_text and not untracked_files:
                    return {"error": "No changes detected"}
                context_parts = []
                if diff_text:  # pragma: no branch – coverage.py asyncio.to_thread tracking bug
                    context_parts.append(f"Diff:\n{diff_text[:4000]}")
                if untracked_files:  # pragma: no branch
                    context_parts.append(f"New untracked files:\n{untracked_files[:500]}")
                msg = _generate_commit_msg("\n\n".join(context_parts), detailed=True)
                scm_pending = os.path.join(sorcar_data_dir, "pending-scm-message.json")
                with open(scm_pending, "w") as f:
                    json.dump({"message": msg}, f)
                return {"message": msg}
            except Exception as e:  # pragma: no cover – git/LLM error
                _log_exc()
                return {"error": str(e)}

        return await _thread_json_response(_generate)

    async def active_file_info(request: Request) -> JSONResponse:
        """Check if the current editor file is a runnable prompt."""
        fpath = _read_active_file(sorcar_data_dir)
        if not fpath or not fpath.lower().endswith(".md"):
            return JSONResponse({"is_prompt": False, "path": fpath})
        return JSONResponse(
            {
                "is_prompt": True,
                "path": fpath,
                "filename": os.path.basename(fpath),
            }
        )

    async def get_file_content(request: Request) -> JSONResponse:
        """Return the text content of a file."""
        fpath = request.query_params.get("path", "").strip()
        if not fpath or not os.path.isfile(fpath):
            return JSONResponse({"error": "File not found"}, status_code=404)
        try:
            with open(fpath, encoding="utf-8") as f:
                content = f.read()
            return JSONResponse({"content": content})
        except Exception as e:  # pragma: no cover – encoding error
            _log_exc()
            return JSONResponse({"error": str(e)}, status_code=500)

    app = Starlette(
        routes=[
            Route("/", index),
            Route("/events", events),
            Route("/run", run_task, methods=["POST"]),
            Route("/run-selection", run_selection, methods=["POST"]),
            Route("/stop", stop_task, methods=["POST"]),
            Route("/user-browser-done", user_browser_done, methods=["POST"]),
            Route("/user-question-done", user_question_done, methods=["POST"]),
            Route("/open-file", open_file, methods=["POST"]),
            Route("/ui-state", get_ui_state),
            Route("/ui-state", save_ui_state, methods=["POST"]),
            Route("/closing", closing, methods=["POST"]),
            Route("/focus-chatbox", focus_chatbox, methods=["POST"]),
            Route("/focus-editor", focus_editor, methods=["POST"]),
            Route("/commit", commit, methods=["POST"]),
            Route("/merge-action", merge_action, methods=["POST"]),
            Route("/record-file-usage", record_file_usage_endpoint, methods=["POST"]),
            Route("/generate-commit-message", generate_commit_message, methods=["POST"]),
            Route("/active-file-info", active_file_info),
            Route("/get-file-content", get_file_content),
            Route("/refresh-files", refresh_files, methods=["POST"]),
            Route("/suggestions", suggestions),
            Route("/complete", complete),
            Route("/tasks", tasks),
            Route("/task-events", task_events),
            Route("/models", models_endpoint),
            Route("/select-model", select_model_endpoint, methods=["POST"]),
            Route("/theme", theme),
        ]
    )

    import atexit

    atexit.register(_cleanup)

    # Read or assign a persistent UI port so the browser origin stays
    # stable across restarts, preserving code-server iframe storage
    # (e.g. auth tokens encrypted in browser storage).
    ui_port_file = Path(sorcar_data_dir) / "ui-port"
    port = 0
    if ui_port_file.exists():
        try:
            port = int(ui_port_file.read_text().strip())
        except (ValueError, OSError):
            _log_exc()
    if port:
        # Verify the saved port is still available (not grabbed by another process).
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
                _s.bind(("127.0.0.1", port))
        except OSError:
            port = 0  # Port in use; pick a fresh one.
    if not port:
        port = find_free_port()
    try:
        Path(sorcar_data_dir).mkdir(parents=True, exist_ok=True)
        _atomic_write_text(ui_port_file, str(port))
        _atomic_write_text(_KISS_DIR / "assistant-port", str(port))
    except OSError:  # pragma: no cover – filesystem permission error
        _log_exc()
    url = f"http://127.0.0.1:{port}"
    print(f"{title} running at {url}", flush=True)
    print(f"Work directory: {actual_work_dir}", flush=True)
    printer.print(f"{title} running at {url}")
    printer.print(f"Work directory: {actual_work_dir}")

    async def _open_browser_async() -> None:  # pragma: no cover – browser launch
        await asyncio.sleep(2)
        WebUseTool._open_in_default_browser(url)

    async def _on_startup() -> None:  # pragma: no cover – browser launch
        asyncio.create_task(_open_browser_async())

    app.add_event_handler("startup", _on_startup)
    logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        timeout_graceful_shutdown=1,
    )
    server = uvicorn.Server(config)
    _orig_handle_exit = server.handle_exit

    def _on_exit(sig: int, frame: types.FrameType | None) -> None:  # pragma: no cover
        shutting_down.set()
        _orig_handle_exit(sig, frame)

    server.handle_exit = _on_exit  # type: ignore[method-assign]
    try:
        server.run()
    except KeyboardInterrupt:  # pragma: no cover – server shutdown signal
        _log_exc()
    _cleanup()


def _auto_update() -> None:  # pragma: no cover – CLI helper
    """Pull latest commits and uv sync, then re-exec if updated."""
    if os.environ.get("_SORCAR_UPDATED"):
        return
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    git_dir = project_root / ".git"
    freshly_initialized = False
    public_url = "https://github.com/ksenxx/kiss_ai.git"
    # Suppress credential prompts — public repo needs no auth.
    git_env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": ""}
    if not git_dir.is_dir():
        print("Sorcar: initializing git repo…")
        try:
            subprocess.run(
                ["git", "init"],
                cwd=project_root, capture_output=True, text=True, timeout=30,
                check=True, env=git_env,
            )
            subprocess.run(
                ["git", "remote", "add", "origin", public_url],
                cwd=project_root, capture_output=True, text=True, timeout=30,
                check=True, env=git_env,
            )
            subprocess.run(
                ["git", "fetch", "origin"],
                cwd=project_root, capture_output=True, text=True, timeout=60,
                check=True, env=git_env,
            )
            subprocess.run(
                ["git", "checkout", "-f", "-B", "main", "origin/main"],
                cwd=project_root, capture_output=True, text=True, timeout=30,
                check=True, env=git_env,
            )
            subprocess.run(
                ["git", "branch", "--set-upstream-to=origin/main", "main"],
                cwd=project_root, capture_output=True, text=True, timeout=30,
                check=True, env=git_env,
            )
            freshly_initialized = True
        except Exception as exc:
            print(f"git init/fetch error: {exc}", file=sys.stderr)
            return
    print("Sorcar: checking for updates…")
    pull_out = ""
    try:
        pull = subprocess.run(
            ["git", "pull", "--ff-only", public_url, "main"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=30,
            env=git_env,
        )
        pull_out = pull.stdout.strip()
        print(pull_out if pull_out else "(no output from git pull)")
        if pull.returncode != 0:
            print(f"git pull failed: {pull.stderr.strip()}", file=sys.stderr)
            if not freshly_initialized:
                return
    except Exception as exc:
        print(f"git pull error: {exc}", file=sys.stderr)
        if not freshly_initialized:
            return
    if not freshly_initialized and pull_out == "Already up to date.":
        print("Sorcar: already up to date.")
        return
    try:
        sync = subprocess.run(
            ["uv", "sync"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if sync.returncode != 0:
            print(f"uv sync failed: {sync.stderr.strip()}", file=sys.stderr)
            return
        print("Sorcar: updated. Relaunching…")
    except Exception as exc:
        print(f"uv sync error: {exc}", file=sys.stderr)
        return
    sys.stdout.flush()
    sys.stderr.flush()
    env = os.environ.copy()
    env["_SORCAR_UPDATED"] = "1"
    sorcar_bin = str(project_root / ".venv" / "bin" / "sorcar")
    os.execve(sorcar_bin, [sorcar_bin, *sys.argv[1:]], env)


def main() -> None:  # pragma: no cover – CLI entry point
    """Launch the KISS Sorcar chatbot UI."""
    import argparse

    _auto_update()

    missing = [k for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY") if not os.environ.get(k)]
    if missing:
        print(
            f"Error: Sorcar requires the following environment variable(s): "
            f"{', '.join(missing)}\n"
            f"Please set them before launching Sorcar.",
            file=sys.stderr,
        )
        sys.exit(1)

    from kiss._version import __version__
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent

    parser = argparse.ArgumentParser(description="KISS Assistant")
    parser.add_argument(
        "work_dir",
        nargs="?",
        default=os.getcwd(),
        help="Working directory for the agent",
    )
    parser.add_argument(
        "--model_name",
        default="claude-opus-4-6",
        help="Default LLM model name",
    )
    args = parser.parse_args()
    work_dir = str(Path(args.work_dir).resolve())

    run_chatbot(
        agent_factory=SorcarAgent,
        title=f"KISS Sorcar: {__version__}",
        work_dir=work_dir,
        default_model=args.model_name,
        agent_kwargs={"headless": False},
    )


if __name__ == "__main__":
    main()
