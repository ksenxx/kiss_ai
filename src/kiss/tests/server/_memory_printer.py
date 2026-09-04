# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""In-memory browser printer for tests.

A minimal :class:`JsonPrinter` subclass that captures every
broadcast (one stamped copy per subscribed tab) into a list so tests
can assert on emitted events without standing up sockets or capturing
stdout.  The broadcast contract mirrors the production
:class:`WebPrinter`:

* Events that already carry an explicit ``tabId`` (system events) are
  emitted verbatim — exactly once — and, mirroring ``WebPrinter``,
  a ``prompt``/``result`` that ALSO carries a ``taskId`` (e.g. the
  task runner's recorded setup-failure result) is recorded and
  persisted under that task with the ``tabId`` stripped.
* Events without ``tabId`` are task events: ``taskId`` is injected from
  the agent thread's thread-local, the event is recorded under the
  task, and one stamped copy per subscribed tab is appended to
  ``self.emitted``.
"""

from __future__ import annotations

from typing import Any

from kiss.server.json_printer import JsonPrinter


class MemoryPrinter(JsonPrinter):
    """Records every emitted event into ``self.emitted``."""

    def __init__(self) -> None:
        """Initialise an empty in-memory emission buffer."""
        super().__init__()
        self.emitted: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        """Capture broadcast events, mirroring :class:`WebPrinter`.

        Args:
            event: The event dictionary to emit.  When ``tabId`` is
                already present the event is captured once verbatim;
                otherwise ``taskId`` is injected, the event is
                recorded once, and one stamped copy per subscriber
                tab is appended.
        """
        if "tabId" in event:
            if event.get("type") in ("prompt", "result") and event.get("taskId"):
                record = {k: v for k, v in event.items() if k != "tabId"}
                with self._lock:
                    self._record_event(record)
                self._persist_event(record)
            self.emitted.append(event)
            return
        event = self._inject_task_id(event)
        super().broadcast(event)
        targets = self._fanout_targets(event.get("taskId"))
        if not targets:
            return
        for tab_id in targets:
            self.emitted.append({**event, "tabId": tab_id})
