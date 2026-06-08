# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Reproduces cross-tab chat pollution between two concurrently running tasks.

Symptom (as reported): "the contents of one tab chat can pollute the
chat of another running task in a different tab."

Mechanism
---------
The server owns a single :class:`WebPrinter`.  Every event the agent
emits is routed by ``task_id``: ``WebPrinter.broadcast`` injects the
agent thread's thread-local ``task_id`` and fans the event out *only*
to the tabs subscribed to that task (each copy stamped with its own
``tabId``).  Events that carry an explicit ``tabId`` are "system"
events sent verbatim.  But an event with **neither** an explicit
``tabId`` **nor** a resolvable thread-local ``task_id`` falls into the
"global" branch and is broadcast *verbatim to every connected client*
(no ``tabId``).  The frontend renders any ``tabId``-less event into
whichever tab is currently active (``main.js`` default case:
``if (ev.tabId !== undefined && ev.tabId !== activeTabId) ...`` only
diverts events that *carry* a foreign ``tabId``).  So a ``tabId``-less
content event emitted while task A runs shows up inside whatever tab
the user is currently viewing — including a *different* tab that is
running task B.  That is the cross-tab pollution.

Real trigger
------------
``TaskRunnerMixin._run_task_inner`` emits a ``result`` panel for the
"No model available" / unknown-model case **before**
``ChatSorcarAgent.run`` sets the thread-local ``task_id``.  On the task
thread at that point the printer's thread-local ``task_id`` is unset,
so the ``result`` is broadcast globally and pollutes whichever tab the
user is viewing.

These tests use the real :class:`WebPrinter` (no mocks of the
broadcast/fan-out/subscription logic) and the real
:class:`VSCodeServer` task lifecycle.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from kiss.agents.vscode.server import VSCodeServer
from kiss.agents.vscode.web_server import WebPrinter

# Content/panel event types that belong to a single task's chat stream.
# If any of these is broadcast without a ``tabId`` while two tabs are
# running, the frontend renders it into the active tab — polluting a
# different running task's chat.
_CONTENT_EVENT_TYPES = frozenset({
    "result", "text_delta", "text_end", "thinking_start", "thinking_delta",
    "thinking_end", "tool_call", "tool_result", "system_output",
    "system_prompt", "prompt", "usage_info", "clear",
})


class _CapturingWebPrinter(WebPrinter):
    """Real :class:`WebPrinter` that captures the wire payloads it sends.

    Overrides only :meth:`_send_to_ws_clients` — the single choke point
    through which ``broadcast`` pushes JSON to every connected client —
    so the full production ``broadcast`` routing (explicit-``tabId``
    verbatim, thread-local ``taskId`` injection, global fallback, and
    per-subscriber fan-out) is exercised unchanged.  The base method
    returns early when no asyncio loop is running; capturing here both
    records the payload and sidesteps that loop dependency.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sent: list[dict[str, Any]] = []
        self._sent_lock = threading.Lock()

    def _send_to_ws_clients(self, data: str) -> None:
        """Record every JSON payload that would be sent to clients."""
        with self._sent_lock:
            self.sent.append(json.loads(data))


def _make_server() -> tuple[VSCodeServer, _CapturingWebPrinter]:
    """Build a real VSCodeServer backed by a capturing WebPrinter."""
    printer = _CapturingWebPrinter()
    server = VSCodeServer(printer=printer)
    return server, printer


class TestCrossTabChatPollution:
    """The no-model ``result`` panel must be scoped to the owning tab."""

    def test_no_model_result_scoped_to_owning_tab(self) -> None:
        """``_run_task_inner``'s no-model result must carry the owning tabId.

        ``tabB`` is running an unrelated task (subscribed to its event
        stream).  We invoke the real ``_run_task_inner`` for ``tabA``
        with an unknown model so it takes the "No model available"
        branch and returns before any agent — and thus before any
        thread-local ``task_id`` — is set.

        Every ``result`` the run emits must be scoped to ``tabA``.
        Before the fix the result is broadcast globally with no
        ``tabId``; the frontend then renders it into whichever tab is
        active, polluting ``tabB``'s running chat.
        """
        server, printer = _make_server()

        # tabB is actively running an unrelated task.
        printer.subscribe_tab("200", "tabB")
        server._get_tab("tabB").task_history_id = 200

        server._run_task_inner({
            "tabId": "tabA",
            "prompt": "hello",
            "model": "__definitely_not_a_real_model__",
        })

        results = [e for e in printer.sent if e.get("type") == "result"]
        assert results, "Expected a no-model result broadcast for tabA"
        for ev in results:
            tab_id = ev.get("tabId")
            assert tab_id == "tabA", (
                "Cross-tab pollution: _run_task_inner broadcast its "
                "no-model 'result' without scoping it to tabA (got "
                f"tabId={tab_id!r}).  A tabId-less result is delivered "
                "to every client and rendered into the active tab, "
                "polluting tabB's running chat."
            )

    def test_no_untagged_content_event_leaks_while_other_tab_runs(self) -> None:
        """No content event may be broadcast globally while two tabs run.

        Any chat-content event broadcast with neither a ``tabId`` nor a
        resolvable task context takes ``WebPrinter.broadcast``'s global
        path — delivered verbatim (no ``tabId``) to every connected
        client and rendered by the frontend into whichever tab is
        active.  While ``tabA`` runs its (unknown-model) task and
        ``tabB`` runs an unrelated one, such an event would surface
        inside ``tabB``'s chat.  Assert none escapes unscoped.
        """
        server, printer = _make_server()

        printer.subscribe_tab("200", "tabB")
        server._get_tab("tabB").task_history_id = 200

        server._run_task_inner({
            "tabId": "tabA",
            "prompt": "hello",
            "model": "__definitely_not_a_real_model__",
        })

        leaked = [
            e
            for e in printer.sent
            if e.get("type") in _CONTENT_EVENT_TYPES and "tabId" not in e
        ]
        assert not leaked, (
            "Cross-tab pollution: content event(s) were broadcast "
            "globally with no tabId while another tab was running a "
            "task, so they render into whichever tab is active "
            f"(e.g. tabB).  Leaked: {leaked}"
        )
