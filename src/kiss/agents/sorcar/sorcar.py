"""Browser-based chatbot for RelentlessAgent-based agents."""

from __future__ import annotations

import asyncio
import base64
import hashlib
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
import webbrowser
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter, find_free_port
from kiss.agents.sorcar.chatbot_ui import _THEME_PRESETS, _build_html
from kiss.agents.sorcar.code_server import (
    _capture_untracked,
    _cleanup_merge_data,
    _parse_diff_hunks,
    _prepare_merge_view,
    _restore_merge_files,
    _save_untracked_base,
    _scan_files,
    _setup_code_server,
    _snapshot_files,
)
from kiss.agents.sorcar.task_history import (
    _KISS_DIR,
    SAMPLE_TASKS,
    _add_task,
    _append_task_to_md,
    _init_task_history_md,
    _load_file_usage,
    _load_history,
    _load_last_model,
    _load_model_usage,
    _load_proposals,
    _record_file_usage,
    _record_model_usage,
    _save_proposals,
    _set_latest_chat_events,
)
from kiss.core import config as config_module
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model_info import (
    _OPENAI_PREFIXES,
    MODEL_INFO,
    get_available_models,
)
from kiss.core.relentless_agent import RelentlessAgent

logger = logging.getLogger(__name__)

_FAST_MODEL = "gemini-2.0-flash"
_COMMIT_MODEL = "gemini-2.0-flash"
_INTERNAL_MODELS = frozenset({_FAST_MODEL, _COMMIT_MODEL})


class _StopRequested(BaseException):
    pass


def _read_active_file(cs_data_dir: str) -> str:
    try:
        af_path = os.path.join(cs_data_dir, "active-file.json")
        with open(af_path) as af:
            path: str = json.loads(af.read()).get("path", "")
        if path and os.path.isfile(path):
            return path
    except (OSError, json.JSONDecodeError):
        logger.debug("Exception caught", exc_info=True)
    return ""


def _clean_llm_output(text: str) -> str:
    return text.strip().strip('"').strip("'")


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
            model_name=_COMMIT_MODEL,
            prompt_template=prompt,
            arguments={"context": diff_text},
            is_agentic=False,
        )
        return _clean_llm_output(raw)
    except Exception:  # pragma: no cover – LLM API failure
        logger.debug("Exception caught", exc_info=True)
        return ""


def _model_vendor_order(name: str) -> int:
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
    actual_work_dir = work_dir or os.getcwd()
    file_cache: list[str] = _scan_files(actual_work_dir)
    agent_thread: threading.Thread | None = None
    current_stop_event: threading.Event | None = None
    proposed_tasks: list[str] = _load_proposals()
    proposed_lock = threading.Lock()
    selected_model = _load_last_model() or default_model
    last = _load_last_model()
    selected_model = last if last and last not in _INTERNAL_MODELS else default_model

    _init_task_history_md()

    cs_proc: subprocess.Popen[bytes] | None = None
    code_server_url = ""
    wd_hash = hashlib.md5(actual_work_dir.encode()).hexdigest()[:8]
    cs_data_dir = str(_KISS_DIR / f"cs-{wd_hash}")

    # If another sorcar instance is already running with this data dir,
    # use a unique data dir for this instance to avoid collisions
    # (e.g., assistant-port overwrite, shared IPC files).
    _existing_port_file = Path(cs_data_dir) / "assistant-port"
    if _existing_port_file.exists():  # pragma: no cover – requires concurrent instance
        try:
            _existing_port = int(_existing_port_file.read_text().strip())
            with socket.create_connection(
                ("127.0.0.1", _existing_port), timeout=0.5
            ):
                # Another instance is reachable; isolate this instance.
                cs_data_dir = str(
                    _KISS_DIR / f"cs-{wd_hash}-{os.getpid()}"
                )
        except (ConnectionRefusedError, OSError, ValueError):
            pass  # Port not reachable; safe to reuse this data dir.

    # Restore files from any stale merge state (e.g., previous crash during merge)
    _restore_merge_files(cs_data_dir, actual_work_dir)

    # Read or assign a code-server port for this work directory.
    # Use socket.bind directly (not find_free_port) so test patches
    # that override find_free_port for the Starlette port don't collide.
    cs_port_file = Path(cs_data_dir) / "cs-port"
    cs_port_file.parent.mkdir(parents=True, exist_ok=True)
    cs_port = 0
    if cs_port_file.exists():  # pragma: no cover – only on restart with existing data dir
        try:
            cs_port = int(cs_port_file.read_text().strip())
        except (ValueError, OSError):
            logger.debug("Exception caught", exc_info=True)
    if not cs_port:  # pragma: no branch – cs_port always 0 on fresh start
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
            _s.bind(("", 0))
            cs_port = int(_s.getsockname()[1])
        try:
            cs_port_file.write_text(str(cs_port))
        except OSError:  # pragma: no cover – filesystem permission error
            logger.debug("Exception caught", exc_info=True)
    cs_url = f"http://127.0.0.1:{cs_port}"
    cs_binary = shutil.which("code-server")

    def _code_server_launch_args() -> list[str]:  # pragma: no cover – requires code-server binary
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
            cs_data_dir,
            "--extensions-dir",
            str(Path(cs_data_dir) / "extensions"),
            "--disable-getting-started-override",
            "--disable-workspace-trust",
            actual_work_dir,
        ]

    def _watch_code_server() -> None:  # pragma: no cover – requires code-server binary
        """Monitor code-server subprocess and restart it if it crashes."""
        nonlocal cs_proc, code_server_url
        while not shutting_down.is_set():
            shutting_down.wait(5.0)
            if shutting_down.is_set():
                break
            if cs_proc is None:
                continue
            ret = cs_proc.poll()
            if ret is None:
                continue
            logger.warning(
                "code-server exited with code %d, restarting...", ret
            )
            try:
                from kiss.agents.sorcar.code_server import _MS_GALLERY

                cs_env = {**os.environ, "EXTENSIONS_GALLERY": _MS_GALLERY}
                cs_proc = subprocess.Popen(
                    _code_server_launch_args(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=cs_env,
                    start_new_session=True,
                )
                restarted = False
                for _ in range(30):
                    try:
                        with socket.create_connection(
                            ("127.0.0.1", cs_port), timeout=0.5
                        ):
                            code_server_url = cs_url
                            restarted = True
                            break
                    except (ConnectionRefusedError, OSError):
                        logger.debug("Exception caught", exc_info=True)
                        time.sleep(0.5)
                if restarted:
                    logger.info("code-server restarted at %s", code_server_url)
                    printer.broadcast({"type": "code_server_restarted"})
                else:
                    logger.warning("code-server failed to restart")
            except Exception:
                logger.debug("Exception caught", exc_info=True)
    if cs_binary:
        ext_changed = _setup_code_server(cs_data_dir)
        port_in_use = False
        try:
            with socket.create_connection(("127.0.0.1", cs_port), timeout=0.5):
                port_in_use = True  # pragma: no cover – requires pre-existing code-server on port
        except (ConnectionRefusedError, OSError):
            logger.debug("Exception caught", exc_info=True)
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
                    cs_port_file.write_text(str(cs_port))
                except OSError:
                    logger.debug("Exception caught", exc_info=True)

        workdir_file = Path(cs_data_dir) / "workdir"
        prev_workdir = ""
        try:
            prev_workdir = workdir_file.read_text().strip() if workdir_file.exists() else ""
        except OSError:  # pragma: no cover – filesystem error reading workdir file
            logger.debug("Exception caught", exc_info=True)
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
                logger.debug("Exception caught", exc_info=True)
            port_in_use = False
        if port_in_use:  # pragma: no cover – requires pre-existing code-server
            code_server_url = cs_url
            printer.print(f"Reusing existing code-server at {code_server_url}")
        else:
            from kiss.agents.sorcar.code_server import _MS_GALLERY

            cs_env = {**os.environ, "EXTENSIONS_GALLERY": _MS_GALLERY}
            cs_proc = subprocess.Popen(
                [
                    cs_binary,
                    "--port",
                    str(cs_port),
                    "--auth",
                    "none",
                    "--bind-addr",
                    f"127.0.0.1:{cs_port}",
                    "--disable-telemetry",
                    "--user-data-dir",
                    cs_data_dir,
                    "--extensions-dir",
                    str(Path(cs_data_dir) / "extensions"),
                    "--disable-getting-started-override",
                    "--disable-workspace-trust",
                    actual_work_dir,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=cs_env,
                start_new_session=True,
            )
            for _ in range(30):  # pragma: no branch – loop always breaks on success
                try:
                    with socket.create_connection(("127.0.0.1", cs_port), timeout=0.5):
                        code_server_url = cs_url
                        break
                except (ConnectionRefusedError, OSError):
                    logger.debug("Exception caught", exc_info=True)
                    time.sleep(0.5)
            if code_server_url:
                printer.print(f"code-server running at {code_server_url}")
            else:  # pragma: no cover – code-server startup failure
                printer.print("Warning: code-server failed to start")
        if code_server_url:  # pragma: no branch – always True after successful startup
            try:
                workdir_file.write_text(actual_work_dir)
            except OSError:  # pragma: no cover – filesystem error writing workdir
                logger.debug("Exception caught", exc_info=True)

    if cs_binary and code_server_url:
        threading.Thread(target=_watch_code_server, daemon=True).start()

    html_page = _build_html(title, code_server_url, actual_work_dir)
    shutdown_timer: threading.Timer | None = None
    shutdown_lock = threading.Lock()

    def refresh_file_cache() -> None:
        nonlocal file_cache
        file_cache = _scan_files(actual_work_dir)

    def refresh_proposed_tasks() -> None:
        nonlocal proposed_tasks
        history = _load_history()
        if not history:  # pragma: no cover – empty history on fresh install
            with proposed_lock:
                proposed_tasks = []
            printer.broadcast({"type": "proposed_updated"})
            return
        task_list = "\n".join(f"- {e['task']}" for e in history[:20])
        agent = KISSAgent("Task Proposer")
        try:
            result = agent.run(
                model_name=_FAST_MODEL,
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
        except Exception:  # pragma: no cover – LLM API failure
            logger.debug("Exception caught", exc_info=True)
            proposals = []
        with proposed_lock:
            proposed_tasks = proposals
        _save_proposals(proposals)
        printer.broadcast({"type": "proposed_updated"})

    def generate_followup(task: str, result: str) -> None:
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
                arguments={
                    "task": task,
                    "result": result[:500],
                },
                is_agentic=False,
            )
            suggestion = _clean_llm_output(raw)
            if suggestion:  # pragma: no branch – LLM always returns non-empty
                printer.broadcast(
                    {
                        "type": "followup_suggestion",
                        "text": suggestion,
                    }
                )
        except Exception:  # pragma: no cover – LLM API failure
            logger.debug("Exception caught", exc_info=True)

    def _watch_theme_file() -> None:
        theme_file = _KISS_DIR / "vscode-theme.json"
        last_mtime = 0.0
        try:
            if theme_file.exists():  # pragma: no branch – depends on system state
                last_mtime = theme_file.stat().st_mtime
        except OSError:  # pragma: no cover – filesystem error
            logger.debug("Exception caught", exc_info=True)
        while not shutting_down.is_set():  # pragma: no branch – daemon thread exit
            try:
                if theme_file.exists():
                    mtime = theme_file.stat().st_mtime
                    if mtime > last_mtime:
                        last_mtime = mtime
                        data = json.loads(theme_file.read_text())
                        kind = data.get("kind", "dark")
                        colors = _THEME_PRESETS.get(kind, _THEME_PRESETS["dark"])
                        printer.broadcast({"type": "theme_changed", **colors})
            except (OSError, json.JSONDecodeError):  # pragma: no cover – filesystem/JSON error
                logger.debug("Exception caught", exc_info=True)
            shutting_down.wait(1.0)

    threading.Thread(target=_watch_theme_file, daemon=True).start()

    def _watch_no_clients() -> None:
        """Periodically check if all clients have disconnected and schedule shutdown."""
        no_client_since: float | None = None
        while not shutting_down.is_set():  # pragma: no branch – daemon thread exit
            shutting_down.wait(5.0)
            if shutting_down.is_set():
                break
            if not printer.has_clients():
                if no_client_since is None:
                    no_client_since = time.monotonic()
                elif time.monotonic() - no_client_since >= 10.0:
                    _schedule_shutdown()
            else:
                no_client_since = None

    threading.Thread(target=_watch_no_clients, daemon=True).start()

    def run_agent_thread(
        task: str,
        model_name: str,
        stop_ev: threading.Event,
        attachments: list | None = None,
    ) -> None:
        nonlocal running, agent_thread, merging
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
                actual_work_dir, cs_data_dir, pre_untracked | set(pre_hunks.keys())
            )
            active_file = _read_active_file(cs_data_dir)
            printer.start_recording()
            printer.broadcast({"type": "clear", "active_file": active_file})
            agent = agent_factory("Chatbot")
            extra_kwargs = dict(agent_kwargs or {})
            if active_file:
                extra_kwargs["current_editor_file"] = active_file
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
            logger.debug("Exception caught", exc_info=True)
            result_text = "(stopped)"
            done_event = {"type": "task_stopped"}
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            result_text = f"(error: {e})"
            done_event = {"type": "task_error", "text": str(e)}
        finally:
            printer._thread_local.stop_event = None
            chat_events = printer.stop_recording()
            _append_task_to_md(task, result_text)
            with running_lock:
                if agent_thread is not current_thread:
                    # Stopped externally; stop_agent already broadcast
                    # task_stopped which is captured in chat_events.
                    _set_latest_chat_events(chat_events, task=task)
                    return
                running = False
                agent_thread = None
            # Broadcast AFTER setting running=False so clients can
            # immediately submit a new task without getting a 409.
            chat_events.append(done_event)
            printer.broadcast(done_event)
            if done_event.get("type") == "task_done":
                threading.Thread(
                    target=generate_followup,
                    args=(task, result_text),
                    daemon=True,
                ).start()
            _set_latest_chat_events(chat_events, task=task)
            try:
                merge_result = _prepare_merge_view(
                    actual_work_dir,
                    cs_data_dir,
                    pre_hunks,
                    pre_untracked,
                    pre_file_hashes,
                )
                if merge_result.get("status") == "opened":
                    with running_lock:
                        merging = True
                    printer.broadcast({"type": "merge_started"})
            except Exception:  # pragma: no cover – merge view error
                logger.debug("Exception caught", exc_info=True)
            refresh_file_cache()
            try:
                refresh_proposed_tasks()
            except Exception:  # pragma: no cover – proposal generation error
                logger.debug("Exception caught", exc_info=True)

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

    # True when this instance created a PID-specific data dir for isolation.
    _is_isolated = cs_data_dir.endswith(f"-{os.getpid()}")

    def _cleanup() -> None:
        nonlocal merging
        with running_lock:
            was_merging = merging
            merging = False
        if was_merging:  # pragma: no cover – cleanup during active merge
            _restore_merge_files(cs_data_dir, actual_work_dir)
        stop_agent()
        if cs_proc and cs_proc.poll() is None:  # pragma: no cover – cleanup timing
            try:
                os.killpg(cs_proc.pid, 15)  # SIGTERM to process group
            except OSError:
                cs_proc.terminate()
            try:
                cs_proc.wait(timeout=5)
            except Exception:
                logger.debug("Exception caught", exc_info=True)
                try:
                    os.killpg(cs_proc.pid, 9)  # SIGKILL to process group
                except OSError:
                    cs_proc.kill()
        # Remove PID-specific data dir created for instance isolation.
        if _is_isolated:  # pragma: no cover – requires concurrent instance
            try:
                shutil.rmtree(cs_data_dir, ignore_errors=True)
            except Exception:
                logger.debug("Exception caught", exc_info=True)

    def _do_shutdown() -> None:  # pragma: no cover – timer-triggered shutdown
        with running_lock:
            if running or printer.has_clients():
                return
            shutting_down.set()
        _cleanup()
        server.should_exit = True

    def _cancel_shutdown() -> None:
        nonlocal shutdown_timer
        with shutdown_lock:
            if shutdown_timer is not None:  # pragma: no cover – timer race
                shutdown_timer.cancel()
                shutdown_timer = None

    def _schedule_shutdown() -> None:  # pragma: no cover – timer-triggered shutdown
        nonlocal shutdown_timer
        if printer.has_clients():
            return
        with running_lock:
            if running:
                return
        with shutdown_lock:
            if shutdown_timer is not None:
                shutdown_timer.cancel()
            shutdown_timer = threading.Timer(10.0, _do_shutdown)
            shutdown_timer.daemon = True
            shutdown_timer.start()

    async def index(request: Request) -> HTMLResponse:
        return HTMLResponse(html_page)

    async def events(request: Request) -> StreamingResponse:
        cq = printer.add_client()
        _cancel_shutdown()

        async def generate() -> AsyncGenerator[str]:
            last_heartbeat = time.monotonic()
            disconnect_check_counter = 0
            try:
                while not shutting_down.is_set():  # pragma: no branch – SSE loop exits via break/cancel
                    disconnect_check_counter += 1
                    if disconnect_check_counter >= 20:
                        disconnect_check_counter = 0
                        if await request.is_disconnected():  # pragma: no cover – timing-dependent disconnect
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
                logger.debug("Exception caught", exc_info=True)
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

    async def run_task(request: Request) -> JSONResponse:
        nonlocal running, agent_thread, selected_model, current_stop_event
        body = await request.json()
        task = body.get("task", "").strip()
        model = body.get("model", "").strip() or selected_model
        attachments = body.get("attachments")
        selected_model = model
        if not task:
            return JSONResponse({"error": "Empty task"}, status_code=400)
        _record_model_usage(model)
        if model not in _INTERNAL_MODELS:
            _record_model_usage(model)
        stop_ev = threading.Event()
        t = threading.Thread(
            target=run_agent_thread,
            args=(task, model, stop_ev, attachments),
            daemon=True,
        )
        with running_lock:
            if merging:
                return JSONResponse(
                    {"error": "Resolve all diffs in the merge view first"},
                    status_code=409,
                )
            if running:
                return JSONResponse({"error": "Agent is already running"}, status_code=409)
            current_stop_event = stop_ev
            running = True
            agent_thread = t
        t.start()
        return JSONResponse({"status": "started"})

    async def run_selection(request: Request) -> JSONResponse:
        """Run the agent on text selected in the VS Code editor.

        Broadcasts an ``external_run`` event so the chatbox UI enters
        running state and displays the selected text as a user message,
        then starts the agent thread with the selected text as the task.
        """
        nonlocal running, agent_thread, current_stop_event
        body = await request.json()
        text = body.get("text", "").strip()
        if not text:
            return JSONResponse({"error": "No text selected"}, status_code=400)
        model = selected_model
        _record_model_usage(model)
        if model not in _INTERNAL_MODELS:
            _record_model_usage(model)
        stop_ev = threading.Event()
        t = threading.Thread(
            target=run_agent_thread,
            args=(text, model, stop_ev, None),
            daemon=True,
        )
        with running_lock:
            if merging:
                return JSONResponse(
                    {"error": "Resolve all diffs in the merge view first"},
                    status_code=409,
                )
            if running:
                return JSONResponse({"error": "Agent is already running"}, status_code=409)
            current_stop_event = stop_ev
            running = True
            agent_thread = t
        # Broadcast AFTER lock confirms task will start, so clients
        # never see external_run for a task that returns 409.
        printer.broadcast({"type": "external_run", "text": text})
        t.start()
        return JSONResponse({"status": "started"})

    async def stop_task(request: Request) -> JSONResponse:
        if stop_agent():
            return JSONResponse({"status": "stopping"})
        return JSONResponse({"error": "No running task"}, status_code=404)

    async def suggestions(request: Request) -> JSONResponse:
        query = request.query_params.get("q", "").strip()
        mode = request.query_params.get("mode", "general")
        if mode == "files":
            q = query.lower()
            usage = _load_file_usage()
            frequent: list[dict[str, str]] = []
            rest: list[dict[str, str]] = []
            for path in file_cache:
                if not q or q in path.lower():
                    ptype = "dir" if path.endswith("/") else "file"
                    item = {"type": ptype, "text": path}
                    if usage.get(path, 0) > 0:
                        frequent.append(item)
                    else:
                        rest.append(item)
            frequent.sort(key=lambda m: (m["type"] != "file", -usage.get(m["text"], 0)))
            rest.sort(key=lambda m: m["type"] != "file")
            for f in frequent:
                f["type"] = "frequent_" + f["type"]
            return JSONResponse((frequent + rest)[:20])
        if not query:
            return JSONResponse([])
        q_lower = query.lower()
        results = []
        for entry in _load_history():
            task = str(entry["task"])
            if q_lower in task.lower():
                results.append({"type": "task", "text": task})
                if len(results) >= 5:
                    break
        with proposed_lock:
            for t in proposed_tasks:
                if q_lower in t.lower():
                    results.append({"type": "suggested", "text": t})
        words = query.split()
        last_word = words[-1].lower() if words else q_lower
        if last_word and len(last_word) >= 2:
            count = 0
            for path in file_cache:
                if last_word in path.lower():
                    results.append({"type": "file", "text": path})
                    count += 1
                    if count >= 8:
                        break
        return JSONResponse(results)

    async def tasks(request: Request) -> JSONResponse:
        history = _load_history()
        return JSONResponse(
            [
                {"task": e["task"], "has_events": bool(e.get("chat_events"))}
                for e in history
            ]
        )

    async def task_events(request: Request) -> JSONResponse:
        """Return chat events for a specific task by index."""
        try:
            idx = int(request.query_params.get("idx", "0"))
        except (ValueError, TypeError):
            return JSONResponse({"error": "Invalid index"}, status_code=400)
        history = _load_history()
        if idx < 0 or idx >= len(history):
            return JSONResponse({"error": "Index out of range"}, status_code=404)
        events: list[dict[str, object]] = history[idx].get("chat_events", [])  # type: ignore[assignment]
        return JSONResponse(events)

    async def proposed_tasks_endpoint(request: Request) -> JSONResponse:
        with proposed_lock:
            tasks_list = list(proposed_tasks)
        if not tasks_list:  # pragma: no cover – depends on LLM response timing
            tasks_list = [str(t["task"]) for t in SAMPLE_TASKS[:5]]
        return JSONResponse(tasks_list)

    def _fast_complete(raw_query: str, query: str) -> str:
        query_lower = query.lower()
        for entry in _load_history():
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

    async def complete(request: Request) -> JSONResponse:
        raw_query = request.query_params.get("q", "")
        query = raw_query.strip()
        if not query or len(query) < 2:
            return JSONResponse({"suggestion": ""})

        fast = _fast_complete(raw_query, query)
        if fast:
            return JSONResponse({"suggestion": fast})

        def _generate() -> str:
            history = _load_history()
            task_list = "\n".join(f"- {e['task']}" for e in history[:20])
            agent = KISSAgent("Autocomplete")
            try:
                result = agent.run(
                    model_name=_FAST_MODEL,
                    prompt_template=(
                        "You are an inline autocomplete engine for a coding assistant. "
                        "Given the user's partial input and their past task history, "
                        "predict what they want to type and return ONLY the remaining "
                        "text to complete their input. Do NOT repeat the text they already typed. "
                        "Keep the completion concise and natural."
                        "If no good completion, return empty string.\n\n"
                        "Past tasks:\n{task_list}\n\n"
                        'Partial input: "{query}"\n\n'
                    ),
                    arguments={"task_list": task_list, "query": query},
                    is_agentic=False,
                )
                s = _clean_llm_output(result)
                if s.lower().startswith(query.lower()):  # pragma: no branch – LLM output dependent
                    s = s[len(query) :]  # pragma: no cover – coverage.py asyncio.to_thread tracking bug
                return s
            except Exception:  # pragma: no cover – LLM API failure
                logger.debug("Exception caught", exc_info=True)
                return ""

        suggestion = await asyncio.to_thread(_generate)  # pragma: no branch – coverage.py thread tracking bug
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
                _model_vendor_order(str(m["name"])),
                -(float(m["inp"]) + float(m["out"])),
            )
        )
        return JSONResponse({"models": models_list, "selected": selected_model})

    async def closing(request: Request) -> JSONResponse:
        """Handle browser tab/window closing. Schedule a quick shutdown."""
        _schedule_shutdown()
        return JSONResponse({"status": "ok"})

    async def focus_chatbox(request: Request) -> JSONResponse:
        printer.broadcast({"type": "focus_chatbox"})
        return JSONResponse({"status": "ok"})

    async def focus_editor(request: Request) -> JSONResponse:
        pending = os.path.join(cs_data_dir, "pending-focus-editor.json")
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
                logger.debug("Exception caught", exc_info=True)
        return JSONResponse(_THEME_PRESETS.get(kind, _THEME_PRESETS["dark"]))

    async def open_file(request: Request) -> JSONResponse:
        body = await request.json()
        rel = body.get("path", "").strip()
        if not rel:
            return JSONResponse({"error": "No path"}, status_code=400)
        full = rel if rel.startswith("/") else os.path.join(actual_work_dir, rel)
        if not os.path.isfile(full):
            return JSONResponse({"error": "File not found"}, status_code=404)
        pending = os.path.join(cs_data_dir, "pending-open.json")
        with open(pending, "w") as f:
            json.dump({"path": full}, f)
        return JSONResponse({"status": "ok"})

    async def merge_action(request: Request) -> JSONResponse:
        nonlocal merging
        body = await request.json()
        action = body.get("action", "")
        if action == "all-done":
            with running_lock:
                merging = False
            printer.broadcast({"type": "merge_ended"})
            _cleanup_merge_data(cs_data_dir)
            return JSONResponse({"status": "ok"})
        if action not in ("prev", "next", "accept-all", "reject-all", "accept", "reject"):
            return JSONResponse({"error": "Invalid action"}, status_code=400)
        pending = os.path.join(cs_data_dir, "pending-action.json")
        with open(pending, "w") as f:
            json.dump({"action": action}, f)
        return JSONResponse({"status": "ok"})

    async def _thread_json_response(
        fn: Callable[[], dict[str, str]],
        error_status: int = 400,
    ) -> JSONResponse:
        result = await asyncio.to_thread(fn)  # pragma: no branch – coverage.py thread tracking bug
        if "error" in result:
            return JSONResponse(result, status_code=error_status)
        return JSONResponse(result)
    async def commit(request: Request) -> JSONResponse:
        def _do_commit() -> dict[str, str]:
            try:
                subprocess.run(["git", "add", "-A"], cwd=actual_work_dir)
                diff_stat = subprocess.run(
                    ["git", "diff", "--cached", "--stat"],
                    capture_output=True,
                    text=True,
                    cwd=actual_work_dir,
                )
                if not diff_stat.stdout.strip():
                    return {"error": "No changes to commit"}
                diff_detail = subprocess.run(
                    ["git", "diff", "--cached"],
                    capture_output=True,
                    text=True,
                    cwd=actual_work_dir,
                )
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
                logger.debug("Exception caught", exc_info=True)
                return {"error": str(e)}

        return await _thread_json_response(_do_commit)

    async def push(request: Request) -> JSONResponse:
        def _do_push() -> dict[str, str]:
            try:
                result = subprocess.run(
                    ["git", "push"],
                    capture_output=True,
                    text=True,
                    cwd=actual_work_dir,
                )
                if result.returncode != 0:
                    return {"error": result.stderr.strip() or "Push failed"}
                return {"status": "ok"}
            except Exception as e:  # pragma: no cover – git error
                logger.debug("Exception caught", exc_info=True)
                return {"error": str(e)}

        return await _thread_json_response(_do_push)


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
                diff_result = subprocess.run(
                    ["git", "diff"],
                    capture_output=True,
                    text=True,
                    cwd=actual_work_dir,
                )
                cached_result = subprocess.run(
                    ["git", "diff", "--cached"],
                    capture_output=True,
                    text=True,
                    cwd=actual_work_dir,
                )
                diff_text = (diff_result.stdout + cached_result.stdout).strip()
                untracked_files = "\n".join(sorted(_capture_untracked(actual_work_dir)))
                if not diff_text and not untracked_files:
                    return {"error": "No changes detected"}
                context_parts = []
                if diff_text:  # pragma: no branch – coverage.py asyncio.to_thread tracking bug
                    context_parts.append(f"Diff:\n{diff_text[:4000]}")
                if untracked_files:  # pragma: no branch – coverage.py asyncio.to_thread tracking bug
                    context_parts.append(f"New untracked files:\n{untracked_files[:500]}")
                msg = _generate_commit_msg("\n\n".join(context_parts), detailed=True)
                scm_pending = os.path.join(cs_data_dir, "pending-scm-message.json")
                with open(scm_pending, "w") as f:
                    json.dump({"message": msg}, f)
                return {"message": msg}
            except Exception as e:  # pragma: no cover – git/LLM error
                logger.debug("Exception caught", exc_info=True)
                return {"error": str(e)}

        return await _thread_json_response(_generate)

    async def active_file_info(request: Request) -> JSONResponse:
        """Check if the current editor file is a runnable prompt."""
        fpath = _read_active_file(cs_data_dir)
        if not fpath or not fpath.lower().endswith(".md"):
            return JSONResponse({"is_prompt": False, "path": fpath})
        from kiss.agents.sorcar.prompt_detector import PromptDetector

        detector = PromptDetector()
        is_prompt, _score, _reasons = detector.analyze(fpath)
        return JSONResponse(
            {
                "is_prompt": is_prompt,
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
            logger.debug("Exception caught", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    async def generate_config_message(request: Request) -> JSONResponse:
        body = await request.json()
        model = body.get("model", selected_model)

        def _generate() -> dict[str, str]:
            cfg = config_module.DEFAULT_CONFIG
            history = _load_history()
            info_parts = [
                f"Work directory: {actual_work_dir}",
                f"Selected model: {model}",
                f"Max steps: {cfg.sorcar.sorcar_agent.max_steps}",
                f"Max budget: ${cfg.sorcar.sorcar_agent.max_budget:.2f}",
                f"Global max budget: ${cfg.agent.global_max_budget:.2f}",
                f"Headless browser: {cfg.sorcar.sorcar_agent.headless}",
                f"Code-server: {'running' if code_server_url else 'not available'}",
                f"Tasks completed: {len(history)}",
            ]
            recent = [str(e["task"]) for e in history[:5]] if history else []
            if recent:  # pragma: no branch – history always has SAMPLE_TASKS fallback
                info_parts.append("Recent tasks: " + "; ".join(recent))
            config_info = "\n".join(info_parts)

            agent = KISSAgent("Config Message Generator")
            try:
                result = agent.run(
                    model_name=_FAST_MODEL,
                    prompt_template=(
                        "You are a helpful assistant. Given the following configuration "
                        "information about a coding assistant environment, generate a "
                        "short, nicely formatted, informative status message that a "
                        "developer would find useful. Include key details like the "
                        "model, work directory, budget, and recent activity. "
                        "Keep it concise (3-5 lines) and use emoji for visual appeal. "
                        "Return ONLY the message text, no quotes or markdown fences.\n\n"
                        "Configuration:\n{config_info}"
                    ),
                    arguments={"config_info": config_info},
                    is_agentic=False,
                )
                return {"message": result.strip()}
            except Exception as e:  # pragma: no cover – LLM API failure
                logger.debug("Exception caught", exc_info=True)
                return {"error": str(e)}

        return await _thread_json_response(_generate, error_status=500)

    app = Starlette(
        routes=[
            Route("/", index),
            Route("/events", events),
            Route("/run", run_task, methods=["POST"]),
            Route("/run-selection", run_selection, methods=["POST"]),
            Route("/stop", stop_task, methods=["POST"]),
            Route("/open-file", open_file, methods=["POST"]),
            Route("/closing", closing, methods=["POST"]),
            Route("/focus-chatbox", focus_chatbox, methods=["POST"]),
            Route("/focus-editor", focus_editor, methods=["POST"]),
            Route("/commit", commit, methods=["POST"]),
            Route("/push", push, methods=["POST"]),
            Route("/merge-action", merge_action, methods=["POST"]),
            Route("/record-file-usage", record_file_usage_endpoint, methods=["POST"]),
            Route("/generate-commit-message", generate_commit_message, methods=["POST"]),
            Route("/generate-config-message", generate_config_message, methods=["POST"]),
            Route("/active-file-info", active_file_info),
            Route("/get-file-content", get_file_content),
            Route("/suggestions", suggestions),
            Route("/complete", complete),
            Route("/tasks", tasks),
            Route("/task-events", task_events),
            Route("/proposed_tasks", proposed_tasks_endpoint),
            Route("/models", models_endpoint),
            Route("/theme", theme),
        ]
    )

    threading.Thread(target=refresh_proposed_tasks, daemon=True).start()

    import atexit

    atexit.register(_cleanup)

    port = find_free_port()
    try:
        Path(cs_data_dir).mkdir(parents=True, exist_ok=True)
        (Path(cs_data_dir) / "assistant-port").write_text(str(port))
    except OSError:  # pragma: no cover – filesystem permission error
        logger.debug("Exception caught", exc_info=True)
    url = f"http://127.0.0.1:{port}"
    print(f"{title} running at {url}", flush=True)
    print(f"Work directory: {actual_work_dir}", flush=True)
    printer.print(f"{title} running at {url}")
    printer.print(f"Work directory: {actual_work_dir}")

    def _open_browser() -> None:  # pragma: no cover – browser launch
        time.sleep(2)
        try:
            if not webbrowser.open(url):
                logger.warning("webbrowser.open() returned False for %s", url)
        except Exception:
            logger.warning("Failed to open browser", exc_info=True)
            if sys.platform == "darwin":
                try:
                    subprocess.Popen(
                        ["open", url],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    logger.warning("Fallback 'open' command also failed", exc_info=True)

    threading.Thread(target=_open_browser, daemon=True).start()
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

    def _on_exit(sig: int, frame: types.FrameType | None) -> None:  # pragma: no cover – signal handler
        shutting_down.set()
        _orig_handle_exit(sig, frame)

    server.handle_exit = _on_exit  # type: ignore[method-assign]
    try:
        server.run()
    except KeyboardInterrupt:  # pragma: no cover – server shutdown signal
        logger.debug("Exception caught", exc_info=True)
    _cleanup()


def main() -> None:  # pragma: no cover – CLI entry point
    """Launch the KISS chatbot UI in assistant or coding mode based on KISS_MODE env var."""
    import argparse

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

    is_assistant = os.environ.get("KISS_MODE", "assistant").lower() == "assistant"
    run_chatbot(
        agent_factory=SorcarAgent,
        title=f"KISS {'Assistant' if is_assistant else 'Coding Assistant'}: {__version__}",
        work_dir=work_dir,
        default_model=args.model_name,
        agent_kwargs={"headless": not is_assistant},
    )


if __name__ == "__main__":
    main()
