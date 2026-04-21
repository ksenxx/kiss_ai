"""VS Code extension printer — writes JSON events to stdout.

Split out of ``server.py`` for organisation.  Imported and
re-exported from ``server`` for backwards compatibility.
"""

from __future__ import annotations

import json
import sys
import threading
from typing import Any

from kiss.agents.sorcar.persistence import _append_chat_event
from kiss.agents.vscode.browser_ui import _DISPLAY_EVENT_TYPES, BaseBrowserPrinter


class VSCodePrinter(BaseBrowserPrinter):
    """Printer that outputs JSON events to stdout for VS Code extension.

    Inherits from BaseBrowserPrinter to get identical event parsing and
    emission (thinking_start/delta/end, text_delta/end, tool_call,
    tool_result, system_output, result). Overrides
    broadcast() to write JSON lines to stdout instead of SSE queues.
    """

    def __init__(self) -> None:
        super().__init__()
        self._stdout_lock = threading.Lock()
        self._persist_agents: dict[str, Any] = {}

    def broadcast(self, event: dict[str, Any]) -> None:
        """Write event as a JSON line to stdout, record it, and persist to DB.

        Injects ``tabId`` from thread-local storage when available so the
        frontend can route events to the correct chat tab.

        Display events are persisted to the database via
        ``_append_chat_event`` as they are created, provided a
        per-tab agent with a valid ``_last_task_id`` is registered
        in ``_persist_agents``.

        The ``_record_event`` call and the stdout write are performed
        inside a single ``_lock`` critical section so recording order
        is guaranteed to match stdout-write order even under
        concurrent broadcasts.  ``_stdout_lock`` is nested inside
        ``_lock`` for defence-in-depth against any future caller that
        writes to stdout directly.

        Args:
            event: The event dictionary to emit.
        """
        tab_id = getattr(self._thread_local, "tab_id", None)
        if tab_id is not None and "tabId" not in event:
            event = {**event, "tabId": tab_id}
        with self._lock:
            self._record_event(event)
            with self._stdout_lock:
                sys.stdout.write(json.dumps(event) + "\n")
                sys.stdout.flush()
        # Persist display events to the database as they are created
        if event.get("type") in _DISPLAY_EVENT_TYPES:
            evt_tab = event.get("tabId")
            if evt_tab is not None:
                agent = self._persist_agents.get(evt_tab)
                if agent is not None:
                    task_id = agent._last_task_id
                    if task_id is not None:
                        _append_chat_event(event, task_id=task_id)
