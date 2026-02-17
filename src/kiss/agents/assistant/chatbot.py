"""Browser-based chatbot for AssistantAgent (reuses shared code from coding chatbot)."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import sys
import threading
import webbrowser
from collections.abc import AsyncGenerator
from pathlib import Path

from kiss.agents.assistant.assistant_agent import AssistantAgent
from kiss.agents.coding_agents.chatbot import (
    _HTML_PAGE,
    _add_task,
    _ChatbotPrinter,
    _find_free_port,
    _load_history,
    _scan_files,
    _StopRequested,
)
from kiss.core.kiss_agent import KISSAgent

_printer = _ChatbotPrinter()
_running = False
_running_lock = threading.Lock()
_work_dir = os.getcwd()
_file_cache: list[str] = []
_agent_thread: threading.Thread | None = None
_proposed_tasks: list[str] = []
_proposed_lock = threading.Lock()

_ASSISTANT_HTML = _HTML_PAGE.replace("KISS Chatbot", "KISS Assistant")


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
            model_name="gemini-2.5-flash",
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
        agent = AssistantAgent("Assistant Chatbot")
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
        return HTMLResponse(_ASSISTANT_HTML)

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
        query_str = request.query_params.get("q", "").strip()
        if not query_str:
            return JSONResponse([])
        q_lower = query_str.lower()
        results: list[dict[str, str]] = []
        for task in _load_history():
            if q_lower in task.lower():
                results.append({"type": "task", "text": task})
                if len(results) >= 5:
                    break
        last_word = query_str.split()[-1].lower() if query_str.split() else q_lower
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
    print(f"KISS Assistant running at {url}")
    print(f"Work directory: {_work_dir}")
    webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
