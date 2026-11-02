# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end: Stop on a task-update child's nested tab stops ONLY the child.

The task-info panel's update is a ``/ask`` child
(:func:`kiss.server.task_update.run_task_update_sea`) run on its own
thread.  The child's thread used to be bound to the parent task's very
own stop event, so the registry state the printer bridge created for
the child carried that same object: a Stop on the child's nested tab
(``AgentServer._stop_task`` resolves the viewer tab to the child's
state and sets its event) cancelled the parent task too.  A
``run_parallel`` child has an event of its own
(``test_subagent_only_stop_and_inject.py``); so has the update child
now, observing the parent's event one way only.

Real pieces throughout: a ``TaskUpdateRunner`` running the production
child under the server's printer (which registers the child's state
and streams its model calls) against a local OpenAI-compatible endpoint
whose streamed reply is always one more ``task_context`` call, the real
agent-state registry, the real subscriber mapping and the production
``AgentServer._stop_task``.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import _add_task
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.server.task_update import TaskUpdateRunner
from kiss.tests.agents.sorcar.local_model_server import MODEL

#: Longest the child may outlive the Stop click: one in-flight model
#: call to the local endpoint plus the cooperative check after it.
STOP_DEADLINE_S = 15.0


class _ForeverTaskContextModel(BaseHTTPRequestHandler):
    """An OpenAI-compatible streaming endpoint whose every reply calls ``task_context``.

    The child calls the tool on every step until something stops it.
    ``calls`` counts the requests served; one subclass per test keeps
    the counters apart.
    """

    calls = 0
    task_id = ""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        type(self).calls += 1
        call = {
            "index": 0, "id": f"call-{type(self).calls}", "type": "function",
            "function": {
                "name": "task_context",
                "arguments": json.dumps({"task_id": type(self).task_id}),
            },
        }
        base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": MODEL}
        chunks = [
            {**base, "choices": [{
                "index": 0, "delta": {"role": "assistant", "tool_calls": [call]},
                "finish_reason": None,
            }]},
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            {**base, "choices": [], "usage": {
                "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
            }},
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@contextmanager
def _serve_forever_task_context(
    task_id: str,
) -> Iterator[tuple[str, type[_ForeverTaskContextModel]]]:
    """Serve :class:`_ForeverTaskContextModel` for *task_id* on a local port."""
    handler = cast(
        type[_ForeverTaskContextModel],
        type("Handler", (_ForeverTaskContextModel,), {"calls": 0, "task_id": task_id}),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=30)


def _wait_until(predicate: Any, timeout: float) -> None:
    """Block until *predicate* is true (polling), or fail after *timeout*."""
    deadline = time.time() + timeout
    while not predicate():
        if time.time() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.01)


def _child_state_of(task_id: str) -> agent_state.AgentState | None:
    """Return the registered active update child of task *task_id*, if any."""
    with agent_state.STATE_LOCK:
        children = [
            s for s in agent_state.agent_states.values()
            if s.parent_task_id == task_id and s.is_task_active
        ]
    return children[0] if children else None


def _running_parent(
    tmp_path: Path, tab_id: str, server: AgentServer,
) -> tuple[ChatSorcarAgent, str, threading.Event, agent_state.AgentState]:
    """Leave a parent the way the task runner leaves a running task.

    A persisted row exposed through ``last_task_id``, the server's
    printer (the child runs under it, as in the daemon) and a
    registration in the agent-state registry under *tab_id* with a
    stop event of its own.

    Returns:
        ``(parent, task_id, parent_stop, state)``; the caller
        unregisters *state*.
    """
    parent = ChatSorcarAgent("task-update-child-stop-parent")
    parent.work_dir = str(tmp_path)
    parent.printer = server.printer
    parent._tab_id = tab_id
    task_id, chat_id = _add_task(
        "Parent task prompt", chat_id="",
        extra={
            "model": MODEL, "work_dir": str(tmp_path),
            "startTs": int(time.time() * 1000),
        },
    )
    parent.resume_chat_by_id(chat_id)
    with parent._task_id_lock:
        parent._last_task_id = task_id
    parent_stop = threading.Event()
    state = agent_state.AgentState(
        task_id, agent=parent, tab_id=tab_id, server_owned=True,
        stop_event=parent_stop, is_task_active=True,
    )
    agent_state.register(state)
    return parent, task_id, parent_stop, state


def test_stop_on_the_update_childs_tab_leaves_the_parent_running(tmp_path: Path) -> None:
    """Stop pressed on the child's nested tab: the child ends, the parent's event stays clear."""
    server = AgentServer()
    events: list[dict[str, Any]] = []
    server.printer.broadcast = events.append  # type: ignore[assignment]
    tab_id = "tab-update-child-stop"
    parent, task_id, parent_stop, state = _running_parent(tmp_path, tab_id, server)
    runner = TaskUpdateRunner()
    try:
        with _serve_forever_task_context(task_id) as (url, model):
            parent.model_name = MODEL
            parent.model_config = {"base_url": url, "api_key": "local"}
            assert runner.poll(task_id, parent, force=True).running
            # The child is registered and well into its loop before the click.
            _wait_until(lambda: model.calls >= 3, timeout=STOP_DEADLINE_S)
            _wait_until(lambda: _child_state_of(task_id) is not None, timeout=STOP_DEADLINE_S)
            child = _child_state_of(task_id)
            assert child is not None
            assert child.stop_event is not None
            # A frontend tab viewing the child's transcript, as the
            # server subscribes it when the nested tab opens.
            viewer = f"{tab_id}__sub_{child.task_id}"
            server.printer.subscribe_tab(child.task_id, viewer)

            server._stop_task(viewer)

            acks = [e for e in events if e["type"] == "stop_ack"]
            assert [(a["tabId"], a["accepted"]) for a in acks] == [(viewer, True)]
            assert not parent_stop.is_set(), "stopping the child cancelled the parent"
            assert child.stop_event is not parent_stop, "the child must own its stop event"
            _wait_until(
                lambda: not runner.poll(task_id, parent).running,
                timeout=STOP_DEADLINE_S,
            )
            served = model.calls
            time.sleep(0.3)
            assert model.calls == served, "the stopped child kept calling the model"
            upd = runner.poll(task_id, parent)
            assert upd.error == "The task was stopped before this update finished."
            assert upd.text == ""
            # The parent is untouched: still registered, active, not stopped.
            assert not parent_stop.is_set()
            assert agent_state.get(task_id) is state
            assert state.is_task_active
            # The child's state is gone with its run.
            assert _child_state_of(task_id) is None
    finally:
        agent_state.unregister(task_id, state)


def test_stop_on_the_parent_tab_still_reaches_the_update_child(tmp_path: Path) -> None:
    """The child's own event observes the parent's: a parent Stop still ends the child.

    Complements the test above (and ``test_task_update_stop.py``):
    giving the child an event of its own must not break the one-way
    inheritance, including for a child that is blocked in a wait on its
    event rather than polling ``is_set``.
    """
    server = AgentServer()
    events: list[dict[str, Any]] = []
    server.printer.broadcast = events.append  # type: ignore[assignment]
    tab_id = "tab-update-parent-stop"
    parent, task_id, parent_stop, state = _running_parent(tmp_path, tab_id, server)
    runner = TaskUpdateRunner()
    try:
        with _serve_forever_task_context(task_id) as (url, model):
            parent.model_name = MODEL
            parent.model_config = {"base_url": url, "api_key": "local"}
            assert runner.poll(task_id, parent, force=True).running
            _wait_until(lambda: model.calls >= 3, timeout=STOP_DEADLINE_S)
            _wait_until(lambda: _child_state_of(task_id) is not None, timeout=STOP_DEADLINE_S)
            child = _child_state_of(task_id)
            assert child is not None and child.stop_event is not None
            child_event = child.stop_event
            # A blocking wait on the child's event (the way a model
            # adapter or a shell tool waits) returns once the PARENT's
            # event is set, within its timeout.
            waiter_result: list[bool] = []
            waiter = threading.Thread(
                target=lambda: waiter_result.append(child_event.wait(STOP_DEADLINE_S)),
                daemon=True,
            )
            waiter.start()
            assert not child_event.wait(0.2)

            server._stop_task(tab_id)

            assert parent_stop.is_set()
            assert child_event.is_set()
            waiter.join(timeout=STOP_DEADLINE_S)
            assert waiter_result == [True]
            _wait_until(
                lambda: not runner.poll(task_id, parent).running,
                timeout=STOP_DEADLINE_S,
            )
            served = model.calls
            time.sleep(0.3)
            assert model.calls == served, "the stopped child kept calling the model"
            upd = runner.poll(task_id, parent)
            assert upd.error == "The task was stopped before this update finished."
            acks = [e for e in events if e["type"] == "stop_ack"]
            assert [(a["tabId"], a["accepted"]) for a in acks] == [(tab_id, True)]
    finally:
        agent_state.unregister(task_id, state)
