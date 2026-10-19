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
