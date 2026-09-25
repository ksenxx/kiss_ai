# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end: a child's spend banked from a foreign thread reaches the parent's tab.

``_attribute_sub_usage`` folds a finished child's cost into the parent
agent and refreshes the printer offsets that ``usage_info`` /``result``
events are corrected by.  The ``/update`` side channel
(``task_update.TaskUpdateRunner``) and the merge conflict resolver call
it from worker threads that carry no task binding; the thread-keyed
offset setters filed the parent's new total under the worker's empty key,
so the parent's next ``usage_info`` under-counted by the child's cost
until the parent thread rewrote its own offset.  The offsets must land
under the parent's task key, and ``ChatSorcarAgent.run`` must hand a
thread back with the task binding it had before.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import SorcarAgent, _agent_usage, _attribute_sub_usage
from kiss.server.json_printer import JsonPrinter

_PARENT_TASK = "4242"


def _parent_with_printer() -> tuple[ChatSorcarAgent, JsonPrinter, list[dict[str, Any]]]:
    """A parent whose task row is *_PARENT_TASK*, wired to a real printer.

    Returns:
        ``(parent, printer, broadcast_events)``.
    """
    parent = ChatSorcarAgent("offset-key-parent")
    with parent._task_id_lock:
        parent._last_task_id = _PARENT_TASK
    printer = JsonPrinter()
    seen: list[dict[str, Any]] = []
    printer.broadcast = seen.append  # type: ignore[method-assign, assignment]
    parent.printer = printer
    return parent, printer, seen


def _attribute_from_unbound_thread(parent: Any, budget: float) -> None:
    """Bank *budget* on *parent* from a thread with no task binding."""
    thread = threading.Thread(target=_attribute_sub_usage, args=(parent, budget, 100, 1))
    thread.start()
    thread.join(timeout=30)
    assert not thread.is_alive()


def test_side_channel_attribution_updates_the_parents_offsets() -> None:
    parent, printer, seen = _parent_with_printer()
    # The parent's own thread banked $1.00 from earlier sessions.
    printer._thread_local.task_id = _PARENT_TASK
    printer.budget_offset = 1.0

    _attribute_from_unbound_thread(parent, 0.5)

    assert _agent_usage(parent)[0] == pytest.approx(0.5)
    assert printer._budget_offsets == {_PARENT_TASK: pytest.approx(0.5)}
    assert printer._tokens_offsets == {_PARENT_TASK: 100}
    assert printer._steps_offsets == {_PARENT_TASK: 1}
    # The parent's executor reports $0.25 of its own; its tab shows
    # banked + live, including the side channel's spend, immediately.
    printer.print("", type="usage_info", total_tokens=10, cost="$0.2500", total_steps=1)
    usage = [e for e in seen if e.get("type") == "usage_info"][-1]
    assert usage["cost"] == "$0.7500"


def test_attribution_to_a_cleaned_up_task_is_dropped() -> None:
    parent, printer, _seen = _parent_with_printer()
    printer.cleanup_task(_PARENT_TASK)
    _attribute_from_unbound_thread(parent, 0.5)
    assert _agent_usage(parent)[0] == pytest.approx(0.5)
    assert _PARENT_TASK not in printer._budget_offsets


def test_agent_without_a_task_row_keeps_the_thread_keyed_offsets() -> None:
    """A plain ``SorcarAgent`` (no ``last_task_id``) is attributed on its own thread."""
    agent = SorcarAgent("plain-parent")
    printer = JsonPrinter()
    agent.printer = printer
    printer._thread_local.task_id = "plain"
    _attribute_sub_usage(agent, 0.25, 10, 1)
    assert printer._budget_offsets == {"plain": pytest.approx(0.25)}


def test_chat_agent_run_restores_the_threads_previous_task_binding(tmp_path: Any) -> None:
    """A child run on a bound thread leaves the binding as it found it."""
    from kiss.agents.sorcar.persistence import _add_task
    from kiss.tests.agents.sorcar.local_model_server import MODEL, finish_body, serve

    printer = JsonPrinter()
    printer.broadcast = lambda _e: None  # type: ignore[method-assign, assignment]
    printer._thread_local.task_id = _PARENT_TASK
    child = ChatSorcarAgent("restore-binding-child")
    _task_id, chat_id = _add_task("seed", chat_id="", extra={"model": MODEL})
    child.resume_chat_by_id(chat_id)
    with serve([finish_body("<p>ok</p>", prompt_tokens=10)]) as (url, _requests):
        child.run(
            prompt_template="hi",
            model_name=MODEL,
            work_dir=str(tmp_path),
            printer=printer,
            model_config={"base_url": url, "api_key": "local"},
            max_budget=1.0,
        )
    assert printer._thread_local.task_id == _PARENT_TASK


def test_concurrent_attributions_leave_the_offsets_at_the_ledger_totals() -> None:
    """Snapshot-and-publish is serialized: no older snapshot lands last.

    Each attribution appends to the ledger, snapshots the totals and
    publishes them as offsets.  Without ordering, a caller that snapshotted
    before a sibling's append could publish after that sibling and leave
    the offsets below the ledger (review finding).  Run many attributions
    at once and require the final offsets to equal the final ledger.
    """
    parent, printer, _seen = _parent_with_printer()
    workers = 200
    start = threading.Barrier(workers)

    def attribute() -> None:
        start.wait()
        _attribute_sub_usage(parent, 0.01, 7, 1)

    threads = [threading.Thread(target=attribute) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not any(thread.is_alive() for thread in threads)

    budget, tokens, steps = _agent_usage(parent)
    assert budget == pytest.approx(workers * 0.01)
    assert (tokens, steps) == (workers * 7, workers)
    assert printer._budget_offsets[_PARENT_TASK] == pytest.approx(budget)
    assert printer._tokens_offsets[_PARENT_TASK] == tokens
    assert printer._steps_offsets[_PARENT_TASK] == steps
