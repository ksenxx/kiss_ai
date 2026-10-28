# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A task-update child whose spend charge fails still closes its tab.

:func:`kiss.server.task_update.run_task_update_sea` ends with two
steps: charge the child's spend to the parent
(:func:`~kiss.server.task_update.charge_side_channel_usage`) and tell
every tab watching the parent that the child is done
(``subagentDone``).  They ran back to back in one ``finally``, so a
raise inside the charge — a usage broadcast, the chat-event write of
an updated row (a synchronous SQLite write that can time out while
another thread holds the database) — skipped the completion signal:
every surface kept the nested ``task-…__update-…`` tab open with its
spinner until the next reload.

The real failure needs a SQLite write to exceed the 30 s busy timeout
at exactly that moment, which no bounded test can stage.  The test
below drives the real path (a real :class:`ChatSorcarAgent` child on a
stand-in model endpoint, the real persistence, the real
``_notify_subagent_done`` fan-out) with one substitution: the parent's
printer is a real :class:`~kiss.server.json_printer.JsonPrinter`
(:class:`CapturePrinter`) whose ``broadcast`` raises once, for the
parent's ``usage_info`` the charge publishes.  That is the first call
inside the charge that can raise after the guarded row update, and it
stands in for the database error.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import _add_task
from kiss.server.task_update import run_task_update_sea
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    StandInModelServer,
    finish_response,
    tool_call_response,
)


class _ChargeFailsPrinter(CapturePrinter):
    """A real printer whose broadcast of the parent's ``usage_info`` raises once."""

    def __init__(self, parent_task_id: str) -> None:
        super().__init__()
        self.parent_task_id = parent_task_id
        self.failed = False

    def broadcast(self, event: dict[str, Any]) -> None:
        """Raise on the first ``usage_info`` charged to the parent, then behave."""
        if (
            not self.failed
            and event.get("type") == "usage_info"
            and event.get("taskId") == self.parent_task_id
        ):
            self.failed = True
            raise RuntimeError("usage broadcast failed")
        super().broadcast(event)


def test_subagent_done_is_sent_when_the_spend_charge_raises(
    tmp_path: Path, caplog: Any,
) -> None:
    """``subagentDone`` reaches the tabs even when the charge raises; the run still returns."""
    parent_id, chat_id = _add_task(
        "Parent prompt", chat_id="",
        extra={"model": STANDIN_MODEL, "work_dir": str(tmp_path)},
    )
    printer = _ChargeFailsPrinter(parent_id)
    parent = ChatSorcarAgent("task-update-parent")
    parent.work_dir = str(tmp_path)
    parent.printer = printer
    parent._tab_id = "root-tab"
    parent.resume_chat_by_id(chat_id)
    with parent._task_id_lock:
        parent._last_task_id = parent_id

    def respond(request: dict[str, Any]) -> dict[str, Any]:
        if any(m.get("role") == "tool" for m in request.get("messages", [])):
            return finish_response("<h4>Working</h4>")
        return tool_call_response("task_context", {"task_id": parent_id})

    standin = StandInModelServer(respond)
    try:
        parent.model_name = STANDIN_MODEL
        parent.model_config = standin.model_config
        with caplog.at_level(logging.WARNING, logger="kiss.server.task_update"):
            text, cost = run_task_update_sea(parent, parent_id)
    finally:
        standin.stop()

    assert text == "<h4>Working</h4>"
    assert cost > 0.0
    assert printer.failed, "the charge's usage broadcast never ran"
    assert "not charged" in caplog.text
    (new_tab,) = printer.events_of_type("new_tab")
    done_tabs = [e["tab_id"] for e in printer.events_of_type("subagentDone")]
    assert f"root-tab__sub_{new_tab['task_id']}" in done_tabs
    assert any(t.startswith(f"task-{parent_id}__update-") for t in done_tabs), done_tabs
