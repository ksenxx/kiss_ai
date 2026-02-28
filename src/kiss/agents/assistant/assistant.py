"""Browser-based chatbot for RelentlessAgent-based agents."""

from __future__ import annotations

import asyncio
import json
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

from kiss.agents.assistant.browser_ui import BaseBrowserPrinter, find_free_port
from kiss.agents.assistant.chatbot_ui import _THEME_PRESETS, _build_html
from kiss.agents.assistant.code_server import (
    _capture_untracked,
    _parse_diff_hunks,
    _prepare_merge_view,
    _scan_files,
    _setup_code_server,
)
from kiss.agents.assistant.relentless_agent import RelentlessAgent
from kiss.agents.assistant.task_history import (
    _KISS_DIR,
    _add_task,
    _append_task_to_md,
    _init_task_history_md,
    _load_history,
    _load_last_model,
    _load_model_usage,
    _load_proposals,
    _record_model_usage,
    _save_proposals,
    _set_latest_result,
)
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model_info import (
    _OPENAI_PREFIXES,
    MODEL_INFO,
    get_available_models,
    get_most_expensive_model,
)


class _StopRequested(BaseException):
    pass


def _model_vendor_order(name: str) -> int:
    if name.startswith("claude-"):
        return 0
    if name.startswith(_OPENAI_PREFIXES) and not name.startswith("openai/"):
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
    title: str = "KISS Assistant",
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
    actual_work_dir = work_dir or os.getcwd()
    file_cache: list[str] = _scan_files(actual_work_dir)
    agent_thread: threading.Thread | None = None
    proposed_tasks: list[str] = _load_proposals()
    proposed_lock = threading.Lock()
    selected_model = (
        _load_last_model() or default_model or get_most_expensive_model() or "claude-opus-4-6"
    )

    _init_task_history_md()

    cs_proc: subprocess.Popen[bytes] | None = None
    code_server_url = ""
    cs_data_dir = str(_KISS_DIR / "code-server-data")
    cs_binary = shutil.which("code-server")
    if cs_binary:
        ext_changed = _setup_code_server(cs_data_dir)
        cs_port = 13338
        port_in_use = False
        try:
            with socket.create_connection(("127.0.0.1", cs_port), timeout=0.5):
                port_in_use = True
        except (ConnectionRefusedError, OSError):
            pass
        if port_in_use and ext_changed:
            print("Extension updated, restarting code-server...")
            try:
                result = subprocess.run(
                    ["lsof", "-ti", f":{cs_port}", "-sTCP:LISTEN"],
                    capture_output=True, text=True,
                )
                for pid_str in result.stdout.strip().split("\n"):
                    if pid_str.strip():
                        os.kill(int(pid_str.strip()), 15)
                time.sleep(1.5)
            except Exception:
                pass
            port_in_use = False
        if port_in_use:
            code_server_url = f"http://127.0.0.1:{cs_port}"
            print(f"Reusing existing code-server at {code_server_url}")
        else:
            cs_proc = subprocess.Popen(
                [
                    cs_binary, "--port", str(cs_port), "--auth", "none",
                    "--bind-addr", f"127.0.0.1:{cs_port}", "--disable-telemetry",
                    "--user-data-dir", cs_data_dir,
                    "--extensions-dir", str(Path(cs_data_dir) / "extensions"),
                    "--disable-getting-started-override",
                    "--disable-workspace-trust",
                    actual_work_dir,
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            for _ in range(30):
                try:
                    with socket.create_connection(("127.0.0.1", cs_port), timeout=0.5):
                        code_server_url = f"http://127.0.0.1:{cs_port}"
                        break
                except (ConnectionRefusedError, OSError):
                    time.sleep(0.5)
            if code_server_url:
                print(f"code-server running at {code_server_url}")
            else:
                print("Warning: code-server failed to start")

    html_page = _build_html(title, code_server_url, actual_work_dir)
    shutdown_timer: threading.Timer | None = None

    def refresh_file_cache() -> None:
        nonlocal file_cache
        file_cache = _scan_files(actual_work_dir)

    def refresh_proposed_tasks() -> None:
        nonlocal proposed_tasks
        history = _load_history()
        if not history:
            with proposed_lock:
                proposed_tasks = []
            printer.broadcast({"type": "proposed_updated"})
            return
        task_list = "\n".join(f"- {e['task']}" for e in history[:20])
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
        with proposed_lock:
            proposed_tasks = proposals
        _save_proposals(proposals)
        printer.broadcast({"type": "proposed_updated"})

    def generate_followup(task: str, result: str) -> None:
        try:
            agent = KISSAgent("Followup Proposer")
            raw = agent.run(
                model_name="gemini-2.0-flash",
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
            suggestion = raw.strip().strip('"').strip("'")
            if suggestion:
                printer.broadcast({
                    "type": "followup_suggestion",
                    "text": suggestion,
                })
        except Exception:
            pass

    def _watch_theme_file() -> None:
        theme_file = _KISS_DIR / "vscode-theme.json"
        last_mtime = 0.0
        try:
            if theme_file.exists():
                last_mtime = theme_file.stat().st_mtime
        except OSError:
            pass
        while not shutting_down.is_set():
            try:
                if theme_file.exists():
                    mtime = theme_file.stat().st_mtime
                    if mtime > last_mtime:
                        last_mtime = mtime
                        data = json.loads(theme_file.read_text())
                        kind = data.get("kind", "dark")
                        colors = _THEME_PRESETS.get(kind, _THEME_PRESETS["dark"])
                        printer.broadcast({"type": "theme_changed", **colors})
            except (OSError, json.JSONDecodeError):
                pass
            shutting_down.wait(1.0)

    threading.Thread(target=_watch_theme_file, daemon=True).start()

    def run_agent_thread(task: str, model_name: str) -> None:
        nonlocal running, agent_thread
        pre_hunks: dict[str, list[tuple[int, int, int, int]]] = {}
        pre_untracked: set[str] = set()
        try:
            _add_task(task)
            printer.broadcast({"type": "tasks_updated"})
            printer.broadcast({"type": "clear"})
            pre_hunks = _parse_diff_hunks(actual_work_dir)
            pre_untracked = _capture_untracked(actual_work_dir)
            agent = agent_factory("Chatbot")
            result = agent.run(
                prompt_template=task,
                work_dir=actual_work_dir,
                printer=printer,
                model_name=model_name,
                **(agent_kwargs or {}),
            )
            _set_latest_result(result or "")
            _append_task_to_md(task, result or "")
            printer.broadcast({"type": "task_done"})
            threading.Thread(
                target=generate_followup,
                args=(task, result or ""),
                daemon=True,
            ).start()
        except _StopRequested:
            _set_latest_result("(stopped)")
            _append_task_to_md(task, "(stopped)")
            printer.broadcast({"type": "task_stopped"})
        except Exception as e:
            _set_latest_result(f"(error: {e})")
            _append_task_to_md(task, f"(error: {e})")
            printer.broadcast({"type": "task_error", "text": str(e)})
        finally:
            with running_lock:
                running = False
                agent_thread = None
            try:
                merge_result = _prepare_merge_view(
                    actual_work_dir, cs_data_dir, pre_hunks, pre_untracked,
                )
                if merge_result.get("status") == "opened":
                    printer.broadcast({"type": "merge_started"})
            except Exception:
                pass
            refresh_file_cache()
            try:
                refresh_proposed_tasks()
            except Exception:
                pass

    def stop_agent() -> bool:
        nonlocal agent_thread
        with running_lock:
            thread = agent_thread
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

    def _cleanup() -> None:
        stop_agent()
        if cs_proc:
            cs_proc.terminate()
            try:
                cs_proc.wait()
            except Exception:
                cs_proc.kill()

    def _do_shutdown() -> None:
        if printer.has_clients():
            return
        _cleanup()
        os._exit(0)

    def _schedule_shutdown() -> None:
        nonlocal shutdown_timer
        if printer.has_clients():
            return
        if shutdown_timer is not None:
            shutdown_timer.cancel()
        shutdown_timer = threading.Timer(1.0, _do_shutdown)
        shutdown_timer.daemon = True
        shutdown_timer.start()

    async def index(request: Request) -> HTMLResponse:
        return HTMLResponse(html_page)

    async def events(request: Request) -> StreamingResponse:
        cq = printer.add_client()

        async def generate() -> AsyncGenerator[str]:
            try:
                while not shutting_down.is_set():
                    try:
                        event = cq.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.05)
                        continue
                    yield f"data: {json.dumps(event)}\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                printer.remove_client(cq)
                _schedule_shutdown()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def run_task(request: Request) -> JSONResponse:
        nonlocal running, agent_thread, selected_model
        with running_lock:
            if running:
                return JSONResponse({"error": "Agent is already running"}, status_code=409)
            running = True
        body = await request.json()
        task = body.get("task", "").strip()
        model = body.get("model", "").strip() or selected_model
        selected_model = model
        if not task:
            with running_lock:
                running = False
            return JSONResponse({"error": "Empty task"}, status_code=400)
        _record_model_usage(model)
        t = threading.Thread(target=run_agent_thread, args=(task, model), daemon=True)
        with running_lock:
            agent_thread = t
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
            results: list[dict[str, str]] = []
            for path in file_cache:
                if not q or q in path.lower():
                    ptype = "dir" if path.endswith("/") else "file"
                    results.append({"type": ptype, "text": path})
                    if len(results) >= 20:
                        break
            return JSONResponse(results)
        if not query:
            return JSONResponse([])
        q_lower = query.lower()
        results = []
        for entry in _load_history():
            task = entry["task"]
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
        return JSONResponse(_load_history())

    async def proposed_tasks_endpoint(request: Request) -> JSONResponse:
        with proposed_lock:
            return JSONResponse(list(proposed_tasks))

    def _fast_complete(raw_query: str, query: str) -> str:
        query_lower = query.lower()
        for entry in _load_history():
            task = entry.get("task", "")
            if task.lower().startswith(query_lower) and len(task) > len(query):
                return task[len(query):]
        words = raw_query.split()
        last_word = words[-1] if words else ""
        if last_word and len(last_word) >= 2:
            lw_lower = last_word.lower()
            for path in file_cache:
                if path.lower().startswith(lw_lower) and len(path) > len(last_word):
                    return path[len(last_word):]
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
                    model_name="gemini-2.0-flash",
                    prompt_template=(
                        "You are an inline autocomplete engine for a coding assistant. "
                        "Given the user's partial input and their past task history, "
                        "predict what they want to type and return ONLY the remaining "
                        "text to complete their input. Do NOT repeat the text they already typed. "
                        "Keep the completion concise and natural.\n\n"
                        "Past tasks:\n{task_list}\n\n"
                        'Partial input: "{query}"\n\n'
                        "Return ONLY the completion text (the part after what they typed). "
                        "If no good completion, return empty string."
                    ),
                    arguments={"task_list": task_list, "query": query},
                    is_agentic=False,
                )
                s = result.strip().strip('"').strip("'")
                if s.lower().startswith(query.lower()):
                    s = s[len(query):]
                if s and not s[0].isspace() and raw_query and not raw_query[-1].isspace():
                    s = " " + s
                return s
            except Exception:
                return ""

        suggestion = await asyncio.to_thread(_generate)
        return JSONResponse({"suggestion": suggestion})

    async def models_endpoint(request: Request) -> JSONResponse:
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
                })
        models_list.sort(key=lambda m: (
            _model_vendor_order(str(m["name"])),
            -(float(m["inp"]) + float(m["out"])),
        ))
        return JSONResponse({"models": models_list, "selected": selected_model})


    async def theme(request: Request) -> JSONResponse:
        theme_file = _KISS_DIR / "vscode-theme.json"
        kind = "dark"
        if theme_file.exists():
            try:
                data = json.loads(theme_file.read_text())
                kind = data.get("kind", "dark")
            except (json.JSONDecodeError, OSError):
                pass
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
        body = await request.json()
        action = body.get("action", "")
        if action == "all-done":
            printer.broadcast({"type": "merge_ended"})
            return JSONResponse({"status": "ok"})
        if action not in ("next", "accept-all", "reject-all"):
            return JSONResponse({"error": "Invalid action"}, status_code=400)
        pending = os.path.join(cs_data_dir, "pending-action.json")
        with open(pending, "w") as f:
            json.dump({"action": action}, f)
        return JSONResponse({"status": "ok"})

    async def commit(request: Request) -> JSONResponse:
        def _do_commit() -> dict[str, str]:
            subprocess.run(["git", "add", "-A"], cwd=actual_work_dir)
            diff_stat = subprocess.run(
                ["git", "diff", "--cached", "--stat"],
                capture_output=True, text=True, cwd=actual_work_dir,
            )
            if not diff_stat.stdout.strip():
                return {"error": "No changes to commit"}
            diff_detail = subprocess.run(
                ["git", "diff", "--cached"],
                capture_output=True, text=True, cwd=actual_work_dir,
            )
            agent = KISSAgent("Commit Message Generator")
            message = agent.run(
                model_name="claude-haiku-4-5",
                prompt_template=(
                    "Generate a concise git commit message (1-2 lines) for these changes. "
                    "Return ONLY the commit message text, no quotes.\n\n{diff}"
                ),
                arguments={"diff": diff_detail.stdout[:4000]},
                is_agentic=False,
            )
            message = message.strip().strip('"').strip("'")
            result = subprocess.run(
                ["git", "commit", "-m", message],
                capture_output=True, text=True, cwd=actual_work_dir,
            )
            if result.returncode != 0:
                return {"error": result.stderr.strip()}
            return {"status": "ok", "message": message}

        result = await asyncio.to_thread(_do_commit)
        if "error" in result:
            return JSONResponse(result, status_code=400)
        return JSONResponse(result)

    app = Starlette(routes=[
        Route("/", index),
        Route("/events", events),
        Route("/run", run_task, methods=["POST"]),
        Route("/stop", stop_task, methods=["POST"]),
        Route("/open-file", open_file, methods=["POST"]),
        Route("/merge-action", merge_action, methods=["POST"]),
        Route("/commit", commit, methods=["POST"]),
        Route("/suggestions", suggestions),
        Route("/complete", complete),
        Route("/tasks", tasks),
        Route("/proposed_tasks", proposed_tasks_endpoint),
        Route("/models", models_endpoint),
        Route("/theme", theme),
    ])

    threading.Thread(target=refresh_proposed_tasks, daemon=True).start()

    import atexit
    atexit.register(_cleanup)

    port = find_free_port()
    try:
        (_KISS_DIR / "assistant-port").write_text(str(port))
    except OSError:
        pass
    url = f"http://127.0.0.1:{port}"
    print(f"{title} running at {url}")
    print(f"Work directory: {actual_work_dir}")

    def _open_browser() -> None:
        time.sleep(2)
        webbrowser.open(url)

    threading.Thread(target=_open_browser, daemon=True).start()
    import logging
    logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning",
        timeout_graceful_shutdown=1,
    )
    server = uvicorn.Server(config)
    _orig_handle_exit = server.handle_exit

    def _on_exit(sig: int, frame: types.FrameType | None) -> None:
        shutting_down.set()
        _orig_handle_exit(sig, frame)

    server.handle_exit = _on_exit  # type: ignore[method-assign]
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    _cleanup()
    os._exit(0)


def main() -> None:
    """Launch the KISS chatbot UI in assistant or coding mode based on KISS_MODE env var."""
    from kiss._version import __version__
    from kiss.agents.assistant.assistant_agent import AssistantAgent
    from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent

    work_dir = str(Path(sys.argv[1]).resolve()) if len(sys.argv) > 1 else os.getcwd()

    mode = os.environ.get("KISS_MODE", "assistant").lower()
    if mode == "assistant":
        run_chatbot(
            agent_factory=AssistantAgent,
            title=f"KISS Assistant: {__version__}",
            work_dir=work_dir,
            agent_kwargs={"headless": False},
        )
    else:
        run_chatbot(
            agent_factory=RelentlessCodingAgent,
            title=f"KISS Coding Assistant: {__version__}",
            work_dir=work_dir,
        )


if __name__ == "__main__":
    main()
