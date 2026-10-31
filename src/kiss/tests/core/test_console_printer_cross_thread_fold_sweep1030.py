# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A sub-agent's spend folded from its own thread reaches the parent's console panels.

``ConsolePrinter`` keeps each agent's usage offsets per thread, and a
finished sub-agent's spend is folded into its parent
(``sorcar_agent._attribute_sub_usage`` -> ``publish_usage_offsets``) on
the sub-agent's job thread.  The console printer had no task-keyed
``set_usage_offsets``, so the fold went through the plain setters and
landed in the job thread's offsets: the parent's per-session Result
panels under-reported the child's cost, tokens and steps until the
parent's next session-start rebase.
"""

from __future__ import annotations

import io
import threading

import yaml

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.relentless_agent import publish_usage_offsets
from kiss.agents.sorcar.sorcar_agent import _attribute_sub_usage
from kiss.core.print_to_console import ConsolePrinter


def _offsets(printer: ConsolePrinter) -> tuple[float, int, int]:
    return printer.budget_offset, printer.tokens_offset, printer.steps_offset


def _parent(printer: ConsolePrinter, task_id: str) -> ChatSorcarAgent:
    """A parent agent bound to *printer* that runs the persisted task *task_id*."""
    agent = ChatSorcarAgent("parent")
    agent.printer = printer
    agent._last_task_id = task_id
    return agent


def _fold_on_a_job_thread(agent: ChatSorcarAgent, budget: float, tokens: int, steps: int) -> None:
    """Fold a finished child's spend into *agent* from a thread of its own."""
    job = threading.Thread(target=_attribute_sub_usage, args=(agent, budget, tokens, steps))
    job.start()
    job.join()


def test_a_fold_from_the_child_thread_reaches_the_parent_thread() -> None:
    buf = io.StringIO()
    printer = ConsolePrinter(file=buf)
    parent = _parent(printer, "task-parent")
    _attribute_sub_usage(parent, 0.25, 10, 1)  # a bank on the parent's own thread
    _fold_on_a_job_thread(parent, 1.5, 100, 3)
    assert _offsets(printer) == (1.75, 110, 4)
    payload = {"success": True, "is_continue": False, "summary": "<p>done</p>"}
    printer.print(
        yaml.dump(payload), type="result", step_count=1, total_tokens=5, cost="$0.0100"
    )
    assert "tokens=115  cost=$1.7600  steps=5" in buf.getvalue()


def test_parallel_tasks_on_a_shared_printer_each_see_only_their_own_folds() -> None:
    printer = ConsolePrinter(file=io.StringIO())
    first = _parent(printer, "task-first")
    second = _parent(printer, "task-second")
    publish_usage_offsets(first, printer)  # first's session start, this thread
    seen: list[tuple[float, int, int]] = []
    started = threading.Event()
    second_folded = threading.Event()
    checked = threading.Event()
    first_folded = threading.Event()

    def second_runs() -> None:
        publish_usage_offsets(second, printer)  # second's session start, its thread
        started.set()
        if second_folded.wait(10):
            seen.append(_offsets(printer))
        checked.set()
        if first_folded.wait(10):
            seen.append(_offsets(printer))

    runner = threading.Thread(target=second_runs)
    runner.start()
    try:
        assert started.wait(10)
        _fold_on_a_job_thread(second, 2.0, 200, 2)
        assert _offsets(printer) == (0.0, 0, 0), "second's fold leaked into first's thread"
        second_folded.set()
        assert checked.wait(10)
        _fold_on_a_job_thread(first, 0.5, 50, 5)
        assert _offsets(printer) == (0.5, 50, 5)
        first_folded.set()
    finally:
        second_folded.set()
        first_folded.set()
        runner.join(10)
    assert not runner.is_alive()
    assert seen == [(2.0, 200, 2), (2.0, 200, 2)], "first's fold leaked into second's thread"


def test_plain_setters_unbind_the_thread_from_its_task() -> None:
    printer = ConsolePrinter(file=io.StringIO())
    parent = _parent(printer, "task-parent")
    _attribute_sub_usage(parent, 1.0, 10, 1)
    assert _offsets(printer) == (1.0, 10, 1)
    # An agent without a persisted task publishes on this same thread.
    printer.budget_offset, printer.tokens_offset, printer.steps_offset = 0.5, 5, 2
    _fold_on_a_job_thread(parent, 3.0, 30, 3)
    assert _offsets(printer) == (0.5, 5, 2)
