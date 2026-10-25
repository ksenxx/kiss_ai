# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""``JsonPrinter.thread_binding`` / ``adopt_thread_binding`` carry the task id across threads.

The printer routes every event by the thread-local ``task_id``, so a
helper thread that prints on a task's behalf must adopt the agent
thread's binding or its events reach no tab.
"""

import threading

from kiss.agents.sorcar._concurrency import locked_call
from kiss.server.json_printer import JsonPrinter


def test_helper_thread_adopts_the_task_thread_binding() -> None:
    printer = JsonPrinter()
    printer._thread_local.task_id = 4242
    binding = printer.thread_binding()
    assert binding == "4242"
    seen: list[str] = []

    def helper() -> None:
        seen.append(printer._task_key())
        printer.adopt_thread_binding(binding)
        seen.append(printer._task_key())

    thread = threading.Thread(target=helper)
    thread.start()
    thread.join()
    assert seen == ["", "4242"]
    # The task thread's own binding is untouched.
    assert printer._task_key() == "4242"


def test_unbound_thread_binding_is_empty() -> None:
    printer = JsonPrinter()
    assert printer.thread_binding() == ""
    printer.adopt_thread_binding(None)
    assert printer._task_key() == ""


def test_task_less_event_is_not_recorded_under_the_thread_task() -> None:
    """``taskId: ""`` marks an event as not attributed to any task.

    ``_inject_task_id`` is the one attribution point: it fills in the
    thread's task only when the event has no ``taskId``.  An emitter
    that stamps ``""`` on purpose (``tasks_updated``, ``new_tab``)
    must not have the event recorded under whatever task the calling
    thread happens to be bound to.
    """
    printer = JsonPrinter()
    printer._thread_local.task_id = "t1"
    printer.start_recording()
    printer.broadcast({"type": "tasks_updated", "taskId": ""})
    printer.broadcast({"type": "text_delta", "text": "mine"})
    # The raw recording, not the display-filtered view ``stop_recording``
    # returns: ``tasks_updated`` is not a display event, so only the raw
    # list shows whether it was filed under the thread's task.
    raw = locked_call(printer._lock, printer._recording_ref, "t1")
    assert [e["type"] for e in raw or []] == ["text_delta"]
