# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""2026-10-16 audit: one offset publisher for both usage writers.

``RelentlessAgent._set_printer_offsets`` (the session-start rebase on
the agent thread) and ``sorcar_agent._attribute_sub_usage`` (a child's
fold on whatever thread finished it) used to carry two copies of the
same snapshot-and-publish step.  Both now go through
``relentless_agent.publish_usage_offsets``, so they file the offsets
under the same key: the agent's persisted task when it has one, else
the calling thread's task (the printer's plain setters).
"""

from __future__ import annotations

import threading

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.relentless_agent import publish_usage_offsets
from kiss.agents.sorcar.sorcar_agent import SorcarAgent, _attribute_sub_usage
from kiss.server.json_printer import JsonPrinter


def test_rebase_and_fold_publish_under_the_agents_task_key() -> None:
    """A chat agent's rebase and a fold from another thread name the same task."""
    agent = ChatSorcarAgent("offset-key")
    printer = JsonPrinter()
    agent.printer = printer
    agent.reset_usage()
    agent._last_task_id = "task-42"
    printer._thread_local.task_id = "task-42"
    agent._attribute_usage(0.25, 10, 2)
    # The agent thread's session-start rebase.
    agent._set_printer_offsets()
    assert printer._budget_offsets == {"task-42": pytest.approx(0.25)}
    assert printer._steps_offsets == {"task-42": 2}
    # A child's fold on an unbound worker thread lands under the same key.
    worker = threading.Thread(target=_attribute_sub_usage, args=(agent, 0.5, 20, 3))
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert printer._budget_offsets == {"task-42": pytest.approx(0.75)}
    assert printer._tokens_offsets == {"task-42": 30}
    assert printer._steps_offsets == {"task-42": 5}


def test_agent_without_a_task_row_uses_the_threads_key() -> None:
    """A plain agent (no ``last_task_id``) publishes under the calling thread's task."""
    agent = SorcarAgent("offset-plain")
    printer = JsonPrinter()
    agent.printer = printer
    agent.reset_usage()
    printer._thread_local.task_id = "thread-task"
    agent._attribute_usage(1.0, 100, 4)
    publish_usage_offsets(agent, printer)
    assert printer._budget_offsets == {"thread-task": pytest.approx(1.0)}
    assert printer._tokens_offsets == {"thread-task": 100}
    assert printer._steps_offsets == {"thread-task": 4}
