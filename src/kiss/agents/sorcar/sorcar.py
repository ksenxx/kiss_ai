"""Browser-based chatbot for RelentlessAgent-based agents."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import http.server
import json
import logging
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import types
import urllib.parse
import webbrowser
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

import yaml

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
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models import model_info as model_info_module
from kiss.core.models.model_info import (
    _OPENAI_PREFIXES,
    MODEL_INFO,
    get_available_models,
)
from kiss.core.relentless_agent import RelentlessAgent

logger = logging.getLogger(__name__)

try:
    from kiss.core.models.codex_oauth import (
        CODEX_OAUTH_CALLBACK_PORT,
        CODEX_OAUTH_REDIRECT_URI,
        OpenAICodexOAuthManager,
        build_authorization_url,
        generate_code_challenge,
        generate_code_verifier,
        generate_oauth_state,
    )
except ImportError:
    CODEX_OAUTH_CALLBACK_PORT = 1455
    CODEX_OAUTH_REDIRECT_URI = "http://localhost:1455/auth/callback"
    OpenAICodexOAuthManager = None  # type: ignore[assignment,misc]
    build_authorization_url = None  # type: ignore[assignment]
    generate_code_challenge = None  # type: ignore[assignment]
    generate_code_verifier = None  # type: ignore[assignment]
    generate_oauth_state = None  # type: ignore[assignment]

_FAST_MODEL = "gemini-2.0-flash"
_COMMIT_MODEL = "gemini-2.0-flash"
_INTERNAL_MODELS = frozenset({_FAST_MODEL, _COMMIT_MODEL})


class _StopRequested(BaseException):
    pass


def _atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* atomically using a temp file and os.replace()."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


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


def _new_utility_agent(name: str) -> KISSAgent:
    """Create a non-task utility agent without trajectory persistence."""
    agent = KISSAgent(name)
    agent.save_trajectory = False
    return agent


def _get_task_history_md_path() -> Path:
    from kiss.core import config as config_module

    return Path(config_module.DEFAULT_CONFIG.agent.artifact_dir).parent / "TASK_HISTORY.md"


def _append_task_to_md(task: str, result: str) -> None:
    try:
        path = _get_task_history_md_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("# Task History\n\n")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = f"## [{timestamp}] {task}\n\n### Result\n\n{result}\n\n---\n\n"
        with path.open("a") as f:
            f.write(entry)
    except OSError:
        logger.debug("Exception caught", exc_info=True)


def _should_warn_no_changes(
    done_event: dict[str, str] | None,
    merge_result: dict[str, Any],
) -> bool:
    """Return True when task reported completion but produced no file changes."""
    if not done_event or done_event.get("type") != "task_done":
        return False
    return merge_result.get("error") == "No changes"


def _resolve_requested_file_path(requested_path: str, work_dir: str) -> str:
    """Resolve a user-provided file path against the current work directory.

    Accept absolute paths across platforms (including Windows drive/UNC paths)
    and resolve relative paths under work_dir.
    """
    requested_path = _normalize_windows_drive_path(requested_path)
    if os.path.isabs(requested_path):
        return os.path.abspath(requested_path)
    return os.path.abspath(os.path.join(work_dir, requested_path))


def _normalize_windows_drive_path(path: str) -> str:
    """Normalize Git-Bash/WSL-style Windows drive paths to native Windows format.

    Examples on Windows:
    - `/c/Users/me/file.txt` -> `C:\\Users\\me\\file.txt`
    - `/mnt/c/Users/me/file.txt` -> `C:\\Users\\me\\file.txt`
    """
    if os.name != "nt":
        return path
    normalized = path.replace("\\", "/")
    m = re.match(r"^/(?:mnt/)?([a-zA-Z])(?:/(.*))?$", normalized)
    if not m:
        return path
    drive = m.group(1).upper()
    rest = m.group(2) or ""
    if not rest:
        return f"{drive}:{os.sep}"
    return f"{drive}:{os.sep}{rest.replace('/', os.sep)}"


def _collect_listening_pids(port: int) -> set[int]:
    """Return process IDs listening on localhost TCP `port` for current platform."""
    pids: set[int] = set()
    if os.name == "nt":
        try:
            # netstat output is locale-dependent. Avoid state-text matching and
            # identify listeners by foreign address 0 and target local port.
            result = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                proto = parts[0].upper()
                local_addr = parts[1]
                foreign_addr = parts[2]
                pid_str = parts[-1]
                if proto != "TCP":
                    continue
                if not local_addr.endswith(f":{port}"):
                    continue
                if foreign_addr not in {"0.0.0.0:0", "[::]:0", "*:0"}:
                    continue
                try:
                    pids.add(int(pid_str))
                except ValueError:
                    logger.debug("Exception caught", exc_info=True)
                    pass
        except Exception:
            logger.debug("Exception caught", exc_info=True)
            pass
        return pids

    lsof = shutil.which("lsof")
    if not lsof:
        return pids
    try:
        result = subprocess.run(
            [lsof, "-ti", f":{port}", "-sTCP:LISTEN"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        for pid_str in result.stdout.strip().splitlines():
            if not pid_str.strip():
                continue
            try:
                pids.add(int(pid_str.strip()))
            except ValueError:
                logger.debug("Exception caught", exc_info=True)
                pass
    except Exception:
        logger.debug("Exception caught", exc_info=True)
        pass
    return pids


def _terminate_listeners_on_port(port: int) -> None:
    """Terminate processes currently listening on localhost TCP `port`."""
    pids = _collect_listening_pids(port)
    if not pids:
        return
    current_pid = os.getpid()
    for pid in sorted(pids):
        if pid == current_pid:
            continue
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F", "/T"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            else:
                os.kill(pid, signal.SIGTERM)
        except Exception:
            logger.debug("Exception caught", exc_info=True)
            pass


def _clear_codex_auth_caches() -> None:
    """Clear cached Codex auth checks so UI status updates immediately."""
    for fn in (
        model_info_module._is_codex_cli_auth_available,
        model_info_module._is_codex_native_auth_available,
    ):
        clear = getattr(fn, "cache_clear", None)
        if callable(clear):
            clear()


def _mask_auth_id(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) <= 8:
        return value
    return f"{value[:4]}...{value[-4:]}"


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
    agent = _new_utility_agent("Commit Message Generator")
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


def _model_provider_key(name: str) -> str:
    if model_info_module.is_codex_provider_model(name):
        return "codex"
    if name.startswith("claude-"):
        return "anthropic"
    if name.startswith(_OPENAI_PREFIXES) and not name.startswith("openai/gpt-oss"):
        return "openai"
    if name.startswith("gemini-"):
        return "gemini"
    if name.startswith("minimax-"):
        return "minimax"
    if name.startswith("openrouter/"):
        return "openrouter"
    return "together"


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
    user_action_event: threading.Event | None = None
    user_question_event: threading.Event | None = None
    user_question_answer: str = ""
    proposed_tasks: list[str] = []
    proposed_lock = threading.Lock()
    last = _load_last_model()
    selected_model = last if last and last not in _INTERNAL_MODELS else default_model
    oauth_lock = threading.Lock()
    oauth_pending: dict[str, Any] | None = None
    oauth_last_error = ""
    oauth_callback_server: http.server.ThreadingHTTPServer | None = None
    oauth_callback_thread: threading.Thread | None = None

    # Clean up stale code-server data directories synchronously at startup
    _cleanup_stale_cs_dirs()

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

    proposals_file = Path(cs_data_dir) / "proposed-tasks.json"

    def _load_proposals() -> list[str]:
        try:
            if not proposals_file.exists():
                return []
            data = json.loads(proposals_file.read_text())
            if not isinstance(data, list):
                return []
            return [str(item) for item in data if isinstance(item, str) and item.strip()][:5]
        except (OSError, json.JSONDecodeError):
            logger.debug("Exception caught", exc_info=True)
            return []

    def _save_proposals(proposals: list[str]) -> None:
        try:
            proposals_file.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(proposals_file, json.dumps(proposals))
        except OSError:
            logger.debug("Exception caught", exc_info=True)

    proposed_tasks = _load_proposals()

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
            _atomic_write_text(cs_port_file, str(cs_port))
        except OSError:  # pragma: no cover – filesystem permission error
            logger.debug("Exception caught", exc_info=True)
    cs_url = f"http://127.0.0.1:{cs_port}"
    cs_binary = shutil.which("code-server")
    if not cs_binary:
        # Fallback: check the offline installer's well-known paths
        for _cs_path in (
            Path.home() / ".kiss-install" / "bin" / "code-server",
            Path.home() / ".kiss-install" / "code-server" / "bin" / "code-server",
        ):
            if _cs_path.is_file():
                cs_binary = str(_cs_path)
                break

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
                    _atomic_write_text(cs_port_file, str(cs_port))
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
                _terminate_listeners_on_port(cs_port)
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
                _code_server_launch_args(),
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
                _atomic_write_text(workdir_file, actual_work_dir)
            except OSError:  # pragma: no cover – filesystem error writing workdir
                logger.debug("Exception caught", exc_info=True)

    if cs_binary and code_server_url:
        threading.Thread(target=_watch_code_server, daemon=True).start()

    html_page = _build_html(title, code_server_url, actual_work_dir)
    shutdown_handle: asyncio.TimerHandle | None = None

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
        agent = _new_utility_agent("Task Proposer")
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
            agent = _new_utility_agent("Followup Proposer")
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

    def _watch_periodic() -> None:
        """Combined watcher: check theme file every 1s and client count every 5s."""
        theme_file = _KISS_DIR / "vscode-theme.json"
        last_mtime = 0.0
        try:
            if theme_file.exists():  # pragma: no branch – depends on system state
                last_mtime = theme_file.stat().st_mtime
        except OSError:  # pragma: no cover – filesystem error
            logger.debug("Exception caught", exc_info=True)
        no_client_since: float | None = None
        tick = 0
        while not shutting_down.is_set():  # pragma: no branch – daemon thread exit
            # Theme check every 1s
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
            # Client check every 5s (every 5th tick)
            tick += 1
            if tick >= 5:
                tick = 0
                if not printer.has_clients():
                    if no_client_since is None:
                        no_client_since = time.monotonic()
                    elif time.monotonic() - no_client_since >= 2.0:
                        _schedule_shutdown()
                else:
                    no_client_since = None
            shutting_down.wait(1.0)

    threading.Thread(target=_watch_periodic, daemon=True).start()

    def _wait_for_user_browser(instruction: str, url: str) -> None:
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

    def _ask_user_question(question: str) -> str:
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

    def _oauth_html(title: str, message: str, ok: bool) -> bytes:
        color = "#10a37f" if ok else "#d9534f"
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{title}</title>"
            "<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;"
            f"background:{color};color:white;display:flex;align-items:center;justify-content:center;"
            "height:100vh;margin:0}div{max-width:640px;padding:24px;text-align:center}"
            "h1{margin:0 0 8px 0;font-size:28px}p{margin:0;opacity:.95}</style>"
            "</head><body><div>"
            f"<h1>{title}</h1><p>{message}</p>"
            "</div><script>setTimeout(function(){window.close();},3500);</script></body></html>"
        ).encode()

    def _shutdown_oauth_callback_server() -> None:
        nonlocal oauth_callback_server, oauth_callback_thread
        server = oauth_callback_server
        oauth_callback_server = None
        oauth_callback_thread = None
        if server is None:
            return
        try:
            server.shutdown()
        except Exception:
            pass
        try:
            server.server_close()
        except Exception:
            pass

    def _ensure_oauth_callback_server() -> tuple[bool, str]:
        nonlocal oauth_callback_server, oauth_callback_thread, oauth_pending, oauth_last_error
        if oauth_callback_thread is not None and oauth_callback_thread.is_alive():
            return True, ""
        if OpenAICodexOAuthManager is None:
            return False, "Native OAuth manager is unavailable."

        class _OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # noqa: A003
                return

            def do_GET(self) -> None:  # noqa: N802
                nonlocal oauth_pending, oauth_last_error
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path != "/auth/callback":
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b"Not Found")
                    return
                params = urllib.parse.parse_qs(parsed.query)
                error = params.get("error", [""])[0].strip()
                code = params.get("code", [""])[0].strip()
                state = params.get("state", [""])[0].strip()

                with oauth_lock:
                    pending = dict(oauth_pending) if oauth_pending else None

                def _finish(status: int, title: str, message: str, ok: bool) -> None:
                    self.send_response(status)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(_oauth_html(title, message, ok))
                    printer.broadcast({"type": "auth_updated"})
                    threading.Thread(target=_shutdown_oauth_callback_server, daemon=True).start()

                if pending is None:
                    with oauth_lock:
                        oauth_last_error = "No pending login flow. Start login from the auth panel."
                    _finish(400, "No Pending Login", oauth_last_error, False)
                    return
                if error:
                    with oauth_lock:
                        oauth_pending = None
                        oauth_last_error = f"OAuth error: {error}"
                    _finish(400, "Authentication Failed", oauth_last_error, False)
                    return
                if not code or not state:
                    with oauth_lock:
                        oauth_pending = None
                        oauth_last_error = "Missing OAuth code/state in callback."
                    _finish(400, "Authentication Failed", oauth_last_error, False)
                    return
                if state != str(pending.get("state", "")):
                    with oauth_lock:
                        oauth_pending = None
                        oauth_last_error = "OAuth state mismatch. Please retry login."
                    _finish(400, "Authentication Failed", oauth_last_error, False)
                    return

                manager = OpenAICodexOAuthManager()
                token = manager.exchange_authorization_code(
                    code,
                    str(pending.get("code_verifier", "")),
                    redirect_uri=CODEX_OAUTH_REDIRECT_URI,
                )
                if not token:
                    with oauth_lock:
                        oauth_pending = None
                        oauth_last_error = "Token exchange failed. Please retry login."
                    _finish(500, "Authentication Failed", oauth_last_error, False)
                    return

                with oauth_lock:
                    oauth_pending = None
                    oauth_last_error = ""
                _clear_codex_auth_caches()
                _finish(
                    200,
                    "Authentication Successful",
                    "KISS is now authenticated with your ChatGPT plan. You can close this window.",
                    True,
                )

        try:
            server = http.server.ThreadingHTTPServer(
                ("127.0.0.1", CODEX_OAUTH_CALLBACK_PORT),
                _OAuthCallbackHandler,
            )
        except OSError as exc:
            return False, (
                "Cannot start OAuth callback server on "
                f"127.0.0.1:{CODEX_OAUTH_CALLBACK_PORT} ({exc})."
            )

        oauth_callback_server = server

        def _run_server() -> None:
            try:
                server.serve_forever(poll_interval=0.5)
            except Exception:
                pass

        oauth_callback_thread = threading.Thread(target=_run_server, daemon=True)
        oauth_callback_thread.start()
        return True, ""

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
            printer.broadcast(
                {
                    "type": "system_output",
                    "text": f"Task started with model {model_name}. Waiting for tool activity...",
                }
            )
            agent = agent_factory("Chatbot")
            if hasattr(agent, '_wait_for_user_callback'):
                agent._wait_for_user_callback = _wait_for_user_browser  # type: ignore[attr-defined]
            if hasattr(agent, '_ask_user_question_callback'):
                agent._ask_user_question_callback = _ask_user_question  # type: ignore[attr-defined]
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
            result_text = "(stopped by user)"
            done_event = {"type": "task_stopped"}
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
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
            _append_task_to_md(task, result_text)
            should_finalize = False
            with running_lock:
                if agent_thread is not current_thread:
                    # Stopped externally; stop_agent already broadcast
                    # task_stopped which is captured in chat_events.
                    _set_latest_chat_events(
                        chat_events, task=task, result=result_summary,
                    )
                    return
                running = False
                agent_thread = None
            # Broadcast AFTER setting running=False so clients can
            # immediately submit a new task without getting a 409.
            chat_events.append(done_event)
            printer.broadcast(done_event)
            if done_event.get("type") == "task_done":
                try:
                    generate_followup(task, result_text)
                except Exception:  # pragma: no cover – LLM API failure
                    logger.debug("Exception caught", exc_info=True)
            _set_latest_chat_events(
                chat_events, task=task, result=result_summary,
            )
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
                if _should_warn_no_changes(done_event, merge_result):
                    printer.broadcast(
                        {
                            "type": "system_output",
                            "text": (
                                "Task reported completion but no file changes were detected. "
                                "Review the result summary for hallucinated completion."
                            ),
                        }
                    )
            except Exception:  # pragma: no cover – merge view error
                logger.debug("Exception caught", exc_info=True)
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
        _shutdown_oauth_callback_server()
        if cs_proc and cs_proc.poll() is None:  # pragma: no cover – cleanup timing
            try:
                if os.name == "nt":
                    cs_proc.terminate()
                else:
                    os.killpg(cs_proc.pid, signal.SIGTERM)
            except OSError:
                cs_proc.terminate()
            try:
                cs_proc.wait(timeout=5)
            except Exception:
                logger.debug("Exception caught", exc_info=True)
                try:
                    if os.name == "nt":
                        cs_proc.kill()
                    else:
                        os.killpg(cs_proc.pid, signal.SIGKILL)
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
        nonlocal shutdown_handle
        if shutdown_handle is not None:  # pragma: no cover – timer race
            shutdown_handle.cancel()
            shutdown_handle = None

    def _schedule_shutdown_on_loop() -> None:  # pragma: no cover – timer-triggered shutdown
        """Schedule shutdown from the asyncio event loop thread."""
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

    def _schedule_shutdown() -> None:  # pragma: no cover – timer-triggered shutdown
        if printer.has_clients():
            return
        with running_lock:
            if running:
                return
        try:
            loop = asyncio.get_running_loop()
            # Already on the event loop thread (called from async context)
            _schedule_shutdown_on_loop()
        except RuntimeError:
            # Called from a non-async thread; dispatch to the event loop
            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(_schedule_shutdown_on_loop)
            except RuntimeError:  # pragma: no cover – no event loop available
                pass

    async def index(request: Request) -> HTMLResponse:
        return HTMLResponse(html_page)

    async def events(request: Request) -> StreamingResponse:
        cq = printer.add_client()
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

    def _try_start_agent_thread(
        t: threading.Thread, stop_ev: threading.Event
    ) -> JSONResponse | None:
        """Acquire the lock and start the agent thread if not already running.

        Returns None on success, or a JSONResponse error on conflict.
        """
        nonlocal running, agent_thread, current_stop_event
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
        return None

    async def run_task(request: Request) -> JSONResponse:
        nonlocal selected_model
        body = await request.json()
        task = body.get("task", "").strip()
        model = body.get("model", "").strip() or selected_model
        attachments = body.get("attachments")
        selected_model = model
        if not task:
            return JSONResponse({"error": "Empty task"}, status_code=400)
        _record_model_usage(model)
        stop_ev = threading.Event()
        t = threading.Thread(
            target=run_agent_thread,
            args=(task, model, stop_ev, attachments),
            daemon=True,
        )
        err = _try_start_agent_thread(t, stop_ev)
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
        _record_model_usage(selected_model)
        stop_ev = threading.Event()
        t = threading.Thread(
            target=run_agent_thread,
            args=(text, selected_model, stop_ev, None),
            daemon=True,
        )
        err = _try_start_agent_thread(t, stop_ev)
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

    async def user_browser_done(request: Request) -> JSONResponse:
        """Signal that the user has finished their browser interaction."""
        if user_action_event is not None:
            user_action_event.set()
            return JSONResponse({"status": "ok"})
        return JSONResponse({"error": "No pending action"}, status_code=404)

    async def user_question_done(request: Request) -> JSONResponse:
        """Signal that the user has answered the agent's question."""
        nonlocal user_question_answer
        if user_question_event is not None:
            body = await request.json()
            user_question_answer = body.get("answer", "")
            user_question_event.set()
            return JSONResponse({"status": "ok"})
        return JSONResponse({"error": "No pending question"}, status_code=404)

    async def refresh_files(request: Request) -> JSONResponse:
        """Refresh the file cache on demand (e.g. when user types @)."""
        refresh_file_cache()
        return JSONResponse({"status": "ok"})

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
                    item = {"type": "file", "text": path}
                    if usage.get(path, 0) > 0:
                        frequent.append(item)
                    else:
                        rest.append(item)

            def _end_dist(text: str) -> int:
                """Distance from end of path to end of rightmost query match.

                Lower = match is closer to end of path = higher priority.
                Returns 0 when query is empty (all items equal).
                """
                if not q:
                    return 0
                pos = text.lower().rfind(q)
                if pos < 0:
                    return len(text)
                return len(text) - (pos + len(q))

            frequent.sort(
                key=lambda m: (
                    _end_dist(m["text"]),
                    -usage.get(m["text"], 0),
                )
            )
            rest.sort(key=lambda m: _end_dist(m["text"]))
            for f in frequent:
                f["type"] = "frequent"
            return JSONResponse((frequent + rest)[:20])
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

    async def tasks(request: Request) -> JSONResponse:
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

    async def task_events(request: Request) -> JSONResponse:
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

    async def complete(request: Request) -> JSONResponse:
        raw_query = request.query_params.get("q", "")
        query = raw_query.strip()
        if not query or len(query) < 2:
            return JSONResponse({"suggestion": ""})

        fast = _fast_complete(raw_query, query)
        if fast:
            return JSONResponse({"suggestion": fast})

        def _generate() -> str:
            history = _load_history(limit=20)
            task_list = "\n".join(f"- {e['task']}" for e in history)
            files_list = "\n".join(file_cache[:200])
            active_path = _read_active_file(cs_data_dir)
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
            agent = _new_utility_agent("Autocomplete")
            try:
                result = agent.run(
                    model_name=_FAST_MODEL,
                    prompt_template=(
                        "You are an inline autocomplete engine for a coding assistant. "
                        "Given the user's partial input, their past task history, "
                        "the list of files/folders in the project, and the content of "
                        "the currently open file in the editor, "
                        "predict what they want to type and return ONLY the remaining "
                        "text to complete their input. Do NOT repeat the text they already typed. "
                        "Keep the completion concise and natural."
                        "If no good completion, return empty string.\n\n"
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
                s = _clean_llm_output(result)
                if s.lower().startswith(query.lower()):  # pragma: no branch
                    s = s[len(query) :]  # pragma: no cover
                return s
            except Exception:  # pragma: no cover – LLM API failure
                logger.debug("Exception caught", exc_info=True)
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
                        "provider": _model_provider_key(name),
                    }
                )
        models_list.sort(
            key=lambda m: (
                _model_vendor_order(str(m["name"])),
                -(float(m["inp"]) + float(m["out"])),
            )
        )
        return JSONResponse({"models": models_list, "selected": selected_model})

    async def select_model_endpoint(request: Request) -> JSONResponse:
        """Update the selected model when user picks from the dropdown."""
        nonlocal selected_model
        body = await request.json()
        name = body.get("model", "").strip()
        if not name:
            return JSONResponse({"error": "No model"}, status_code=400)
        selected_model = name
        _save_last_model(name)
        return JSONResponse({"status": "ok"})

    async def get_ui_state(request: Request) -> JSONResponse:
        """Return saved UI state (divider position, etc.)."""
        ui_state_file = os.path.join(cs_data_dir, "ui-state.json")
        try:
            if os.path.exists(ui_state_file):
                with open(ui_state_file) as f:
                    return JSONResponse(json.load(f))
        except (OSError, json.JSONDecodeError):
            logger.debug("Exception caught", exc_info=True)
        return JSONResponse({})

    async def save_ui_state(request: Request) -> JSONResponse:
        """Save UI state (divider position, etc.)."""
        body = await request.json()
        ui_state_file = os.path.join(cs_data_dir, "ui-state.json")
        try:
            Path(cs_data_dir).mkdir(parents=True, exist_ok=True)
            with open(ui_state_file, "w") as f:
                json.dump(body, f)
        except OSError:
            logger.debug("Exception caught", exc_info=True)
        return JSONResponse({"status": "ok"})

    async def closing(request: Request) -> JSONResponse:
        """Handle browser tab/window closing. Schedule a quick shutdown."""
        _schedule_shutdown()
        return JSONResponse({"status": "ok"})

    def _auth_status_payload(model_name: str) -> dict[str, Any]:
        nonlocal oauth_pending, oauth_last_error
        requested_model = model_name or selected_model
        keys = config_module.DEFAULT_CONFIG.agent.api_keys
        is_openai_model = (
            requested_model.startswith(_OPENAI_PREFIXES)
            and not requested_model.startswith("openai/gpt-oss")
        )
        codex_native_available = model_info_module._is_codex_native_auth_available()
        codex_cli_available = model_info_module._is_codex_cli_auth_available()
        codex_auth_available = codex_native_available or codex_cli_available

        preferred_auth = "n/a"
        codex_subscription_model = False
        if is_openai_model:
            preferred_auth = model_info_module._resolve_openai_auth_mode(
                requested_model,
                keys.OPENAI_API_KEY,
            )
            codex_subscription_model = model_info_module._is_codex_subscription_model(
                requested_model
            )

        codex_transport = (
            model_info_module._resolve_codex_transport()
            if codex_auth_available
            else "none"
        )
        login_pending = False
        with oauth_lock:
            if oauth_pending:
                expires_at = float(oauth_pending.get("expires_at", 0.0) or 0.0)
                if expires_at and time.time() > expires_at:
                    oauth_pending = None
                    oauth_last_error = "Login timed out. Please start login again."
                else:
                    login_pending = True

        codex_account_id: str | None = None
        codex_cache_file = str(Path("~/.kiss/codex_oauth.json").expanduser())
        codex_source_file = str(Path("~/.codex/auth.json").expanduser())
        if OpenAICodexOAuthManager is not None:
            try:
                manager = OpenAICodexOAuthManager()
                codex_account_id = manager.get_account_id()
                codex_cache_file = str(manager.cache_file)
                codex_source_file = str(manager.source_file)
            except Exception:
                pass

        return {
            "model": requested_model,
            "is_openai_model": is_openai_model,
            "preferred_auth": preferred_auth,
            "codex_subscription_model": codex_subscription_model,
            "openai_api_key_configured": bool(keys.OPENAI_API_KEY),
            "codex_auth_available": codex_auth_available,
            "codex_native_available": codex_native_available,
            "codex_cli_available": codex_cli_available,
            "codex_transport": codex_transport,
            "login_pending": login_pending,
            "login_error": oauth_last_error,
            "oauth_callback_port": CODEX_OAUTH_CALLBACK_PORT,
            "forced_auth": os.environ.get("KISS_OPENAI_AUTH", "").strip().lower() or "auto",
            "forced_transport": os.environ.get("KISS_CODEX_TRANSPORT", "").strip().lower()
            or "auto",
            "codex_account_id": _mask_auth_id(codex_account_id),
            "codex_cache_file": codex_cache_file,
            "codex_source_file": codex_source_file,
        }

    async def auth_endpoint(request: Request) -> JSONResponse:
        nonlocal oauth_pending, oauth_last_error
        if request.method == "GET":
            model_name = request.query_params.get("model", "").strip()
            return JSONResponse(_auth_status_payload(model_name))

        body = await request.json()
        action = str(body.get("action", "refresh")).strip().lower()
        model_name = str(body.get("model", "")).strip()
        if action not in {"refresh", "logout", "login", "cancel_login"}:
            return JSONResponse({"error": "Invalid action"}, status_code=400)

        result: dict[str, Any] = {"status": "ok", "action": action}

        # Always clear caches first so status reflects filesystem/env changes immediately.
        _clear_codex_auth_caches()

        if action == "cancel_login":
            with oauth_lock:
                oauth_pending = None
                oauth_last_error = ""
            _shutdown_oauth_callback_server()

        if action == "login":
            if (
                OpenAICodexOAuthManager is None
                or generate_code_verifier is None
                or generate_code_challenge is None
                or generate_oauth_state is None
                or build_authorization_url is None
            ):
                result["status"] = "error"
                result["error"] = "Native OAuth login is unavailable in this build."
                result["auth"] = _auth_status_payload(model_name)
                return JSONResponse(result, status_code=500)
            code_verifier = generate_code_verifier()
            code_challenge = generate_code_challenge(code_verifier)
            state = generate_oauth_state()
            login_url = build_authorization_url(
                code_challenge,
                state,
                originator="kiss-ai",
                redirect_uri=CODEX_OAUTH_REDIRECT_URI,
            )

            with oauth_lock:
                oauth_pending = {
                    "state": state,
                    "code_verifier": code_verifier,
                    "created_at": time.time(),
                    "expires_at": time.time() + 300.0,
                }
                oauth_last_error = ""

            started, start_error = _ensure_oauth_callback_server()
            if not started:
                with oauth_lock:
                    oauth_pending = None
                    oauth_last_error = start_error
                result["status"] = "error"
                result["error"] = start_error
                result["auth"] = _auth_status_payload(model_name)
                return JSONResponse(result, status_code=500)
            result["login_url"] = login_url
            result["callback_port"] = CODEX_OAUTH_CALLBACK_PORT

        if action == "logout":
            codex_path = shutil.which("codex")
            logout_summary = {"attempted": False, "success": False, "message": ""}
            if codex_path:
                logout_summary["attempted"] = True
                try:
                    proc = subprocess.run(
                        [codex_path, "logout"],
                        capture_output=True,
                        text=True,
                        timeout=15,
                        check=False,
                    )
                    output = (proc.stdout or proc.stderr or "").strip()
                    logout_summary["success"] = proc.returncode == 0
                    logout_summary["message"] = output
                except (OSError, subprocess.SubprocessError) as exc:
                    logout_summary["message"] = str(exc)
            else:
                logout_summary["message"] = "Codex CLI not found on PATH."

            # Remove KISS-managed OAuth cache so native transport is immediately unauthenticated.
            removed_cache_files: list[str] = []
            cache_candidates = {str(Path("~/.kiss/codex_oauth.json").expanduser())}
            source_candidates = {str(Path("~/.codex/auth.json").expanduser())}
            if OpenAICodexOAuthManager is not None:
                try:
                    manager = OpenAICodexOAuthManager()
                    cache_candidates.add(str(manager.cache_file))
                    source_candidates.add(str(manager.source_file))
                except Exception:
                    pass
            auth_file_env = os.environ.get("KISS_CODEX_AUTH_FILE")
            if auth_file_env:
                source_candidates.add(str(Path(auth_file_env).expanduser()))
            for cache_file in sorted(cache_candidates):
                try:
                    path = Path(cache_file)
                    if path.exists():
                        path.unlink()
                        removed_cache_files.append(cache_file)
                except OSError:
                    continue
            removed_source_files: list[str] = []
            for source_file in sorted(source_candidates):
                try:
                    path = Path(source_file)
                    if path.exists():
                        path.unlink()
                        removed_source_files.append(source_file)
                except OSError:
                    continue

            result["logout"] = logout_summary
            result["removed_cache_files"] = removed_cache_files
            result["removed_source_files"] = removed_source_files
            with oauth_lock:
                oauth_pending = None
                oauth_last_error = ""
            _shutdown_oauth_callback_server()

            # Clear once more in case logout modified auth state between checks.
            _clear_codex_auth_caches()

        result["auth"] = _auth_status_payload(model_name)
        return JSONResponse(result)

    async def focus_chatbox(request: Request) -> JSONResponse:
        printer.broadcast({"type": "focus_chatbox"})
        return JSONResponse({"status": "ok"})

    def _write_pending_json(name: str, payload: dict[str, Any]) -> None:
        Path(cs_data_dir).mkdir(parents=True, exist_ok=True)
        pending = Path(cs_data_dir) / name
        pending.write_text(json.dumps(payload))

    async def focus_editor(request: Request) -> JSONResponse:
        _write_pending_json("pending-focus-editor.json", {"focus": True})
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
        # Use OS-aware absolute-path detection so Windows paths like C:\... work.
        full = _resolve_requested_file_path(rel, actual_work_dir)
        if not os.path.isfile(full):
            return JSONResponse({"error": "File not found"}, status_code=404)
        _write_pending_json("pending-open.json", {"path": full})
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
        _write_pending_json("pending-action.json", {"action": action})
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
                    message = result.stderr.strip() or result.stdout.strip() or "git push failed"
                    return {"error": message}
                message = result.stdout.strip() or "Pushed to remote"
                return {"status": "ok", "message": message}
            except Exception as e:  # pragma: no cover – git push error
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
                if untracked_files:  # pragma: no branch
                    context_parts.append(f"New untracked files:\n{untracked_files[:500]}")
                msg = _generate_commit_msg("\n\n".join(context_parts), detailed=True)
                _write_pending_json("pending-scm-message.json", {"message": msg})
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
            logger.debug("Exception caught", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    async def generate_config_message(request: Request) -> JSONResponse:
        body = await request.json()
        model = body.get("model", selected_model)

        def _generate() -> dict[str, str]:
            cfg = config_module.DEFAULT_CONFIG
            total = _count_history()
            recent_entries = _load_history(limit=5)
            info_parts = [
                f"Work directory: {actual_work_dir}",
                f"Selected model: {model}",
                f"Max steps: {cfg.sorcar.sorcar_agent.max_steps}",
                f"Max budget: ${cfg.sorcar.sorcar_agent.max_budget:.2f}",
                f"Global max budget: ${cfg.agent.global_max_budget:.2f}",
                f"Headless browser: {cfg.sorcar.sorcar_agent.headless}",
                f"Code-server: {'running' if code_server_url else 'not available'}",
                f"Tasks completed: {total}",
            ]
            recent = [str(e["task"]) for e in recent_entries]
            if recent:  # pragma: no branch – history always has SAMPLE_TASKS fallback
                info_parts.append("Recent tasks: " + "; ".join(recent))
            config_info = "\n".join(info_parts)

            agent = _new_utility_agent("Config Message Generator")
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
            Route("/user-browser-done", user_browser_done, methods=["POST"]),
            Route("/user-question-done", user_question_done, methods=["POST"]),
            Route("/open-file", open_file, methods=["POST"]),
            Route("/ui-state", get_ui_state),
            Route("/ui-state", save_ui_state, methods=["POST"]),
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
            Route("/refresh-files", refresh_files, methods=["POST"]),
            Route("/suggestions", suggestions),
            Route("/complete", complete),
            Route("/tasks", tasks),
            Route("/task-events", task_events),

            Route("/models", models_endpoint),
            Route("/select-model", select_model_endpoint, methods=["POST"]),
            Route("/auth", auth_endpoint, methods=["GET", "POST"]),
            Route("/select-model", select_model_endpoint, methods=["POST"]),
            Route("/auth", auth_endpoint, methods=["GET", "POST"]),
            Route("/theme", theme),
        ]
    )

    import atexit

    atexit.register(_cleanup)

    port = find_free_port()
    try:
        Path(cs_data_dir).mkdir(parents=True, exist_ok=True)
        _atomic_write_text(Path(cs_data_dir) / "assistant-port", str(port))
    except OSError:  # pragma: no cover – filesystem permission error
        logger.debug("Exception caught", exc_info=True)
    url = f"http://127.0.0.1:{port}"
    print(f"{title} running at {url}", flush=True)
    print(f"Work directory: {actual_work_dir}", flush=True)
    printer.print(f"{title} running at {url}")
    printer.print(f"Work directory: {actual_work_dir}")

    async def _open_browser_async() -> None:  # pragma: no cover – browser launch
        await asyncio.sleep(2)
        try:
            if not webbrowser.open(url):
                logger.warning("webbrowser.open() returned False for %s", url)
        except Exception:
            logger.warning("Failed to open browser", exc_info=True)
            cmd: list[str] = []
            if sys.platform == "darwin":
                cmd = ["open", url]
            elif sys.platform.startswith("linux"):
                cmd = ["xdg-open", url]
            elif sys.platform == "win32":
                cmd = ["cmd", "/c", "start", "", url]
            if cmd:
                try:
                    subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    logger.warning("Fallback command %s also failed", cmd, exc_info=True)

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
        logger.debug("Exception caught", exc_info=True)
    _cleanup()


def main() -> None:  # pragma: no cover – CLI entry point
    """Launch the KISS Sorcar chatbot UI."""
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

    run_chatbot(
        agent_factory=SorcarAgent,
        title=f"KISS Sorcar: {__version__}",
        work_dir=work_dir,
        default_model=args.model_name,
        agent_kwargs={"headless": False},
    )


if __name__ == "__main__":
    main()
