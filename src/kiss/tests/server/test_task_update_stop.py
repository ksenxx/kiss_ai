# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end: Stop on a task's tab also stops its running update child.

The task-info panel's update is a ``/ask`` child
(:func:`kiss.server.task_update.run_task_update_sea`) run on its own
thread.  ``AgentServer._stop_task`` sets the stop event of the task
that owns the tab; the child used to run on a thread with no stop
event bound at all, so a Stop ended the task while its update child
kept calling the model until its budget cap.  The child now shares
the parent's stop event, the way a ``run_parallel`` child does.

Real pieces throughout: a ``TaskUpdateRunner`` running the production
child against a scripted local model endpoint whose one scripted reply
repeats forever (the child would call ``task_context`` until its
budget ran out), the real agent-state registry and the production
``AgentServer._stop_task``.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import _add_task
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.server.task_update import TaskUpdateRunner, _ChildStopEvent
from kiss.tests.agents.sorcar.local_model_server import MODEL, serve, tool_call_body
from kiss.tests.lock_injection import stop_as_it_takes_its_lock

#: Longest the child may outlive the Stop click: one in-flight model
#: call to the local endpoint plus the cooperative check after it.
STOP_DEADLINE_S = 15.0


def _wait_until(predicate: Any, timeout: float) -> None:
    """Block until *predicate* is true (polling), or fail after *timeout*."""
    deadline = time.time() + timeout
    while not predicate():
        if time.time() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.01)


def test_stop_on_the_parent_tab_ends_its_running_update(tmp_path: Path) -> None:
    """Stop pressed while the update child runs: the child ends within a bound.

    The parent is left the way the task runner leaves a running task:
    a persisted row, exposed through ``last_task_id``, registered in
    the agent-state registry under its tab with its stop event.
    """
    server = AgentServer()
    events: list[dict[str, Any]] = []
    server.printer.broadcast = events.append  # type: ignore[assignment]
    tab_id = "tab-update-stop"
    parent = ChatSorcarAgent("task-update-stop-parent")
    parent.work_dir = str(tmp_path)
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
    stop_event = threading.Event()
    state = agent_state.AgentState(
        task_id, agent=parent, tab_id=tab_id, server_owned=True,
        stop_event=stop_event, is_task_active=True,
    )
    agent_state.register(state)
    # The last scripted body repeats: the child calls ``task_context``
    # on every step until something stops it.
    script = [
        tool_call_body(
            "task_context", {"task_id": task_id}, prompt_tokens=1, completion_tokens=1,
        ),
    ]
    runner = TaskUpdateRunner()
    try:
        with serve(script) as (url, requests):
            parent.model_name = MODEL
            parent.model_config = {"base_url": url, "api_key": "local"}
            assert runner.poll(task_id, parent, force=True).running
            # The child is well into its loop before the click.
            _wait_until(lambda: len(requests) >= 3, timeout=STOP_DEADLINE_S)

            server._stop_task(tab_id)

            assert stop_event.is_set()
            assert [e["type"] for e in events] == ["stop_ack"]
            assert events[0]["accepted"] is True
            _wait_until(
                lambda: not runner.poll(task_id, parent).running,
                timeout=STOP_DEADLINE_S,
            )
            served = len(requests)
            time.sleep(0.3)
            assert len(requests) == served, "the stopped child kept calling the model"
            # The stop is the update's quiet end, not a failure the
            # thread dies of: the panel is told the task was stopped.
            upd = runner.poll(task_id, parent)
            assert upd.error == "The task was stopped before this update finished."
            assert upd.text == ""
    finally:
        agent_state.unregister(task_id, state)


def _wait_under_profile(
    event: threading.Event, profile: Any, raised: list[BaseException],
) -> None:
    """Thread body: wait for *event* under the profile hook, noting what the wait raises."""
    sys.setprofile(profile)
    try:
        event.wait(30)
    except BaseException as exc:  # noqa: BLE001 — the injected Stop is expected
        raised.append(exc)
    finally:
        sys.setprofile(None)


def test_a_stop_as_the_childs_wait_takes_its_lock_leaves_the_event_settable() -> None:
    """The forced Stop landing where a slice of the child's wait has taken its lock leaves it free.

    The update's thread waits on its stop event in slices of
    ``Event.wait``, which enters a ``threading.Condition``: the Stop
    raised as its ``__enter__`` returns left the lock held (gh-74174),
    and the Stop click on the child's tab, which sets that event from a
    command handler, blocked on it for good.
    """
    event = _ChildStopEvent(threading.Event())
    raised: list[BaseException] = []
    profile = stop_as_it_takes_its_lock(threading.Event.wait, _ChildStopEvent.wait)
    waiter = threading.Thread(
        target=_wait_under_profile, args=(event, profile, raised), daemon=True,
    )
    waiter.start()
    waiter.join(20)
    assert not waiter.is_alive()
    assert [type(exc) for exc in raised] == [KeyboardInterrupt]
    setter = threading.Thread(target=event.set, daemon=True)
    setter.start()
    setter.join(5)
    assert not setter.is_alive(), "the Stop left the child's stop event locked"
