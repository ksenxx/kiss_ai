"""In-memory browser printer for tests.

A minimal :class:`BaseBrowserPrinter` subclass that captures every
broadcast (primary + fan-out copies) into a list so tests can assert
on emitted events without standing up sockets or capturing stdout.
The broadcast contract mirrors the production :class:`WebPrinter`:
inject ``tabId`` from thread-local storage, record once under the
source tab, persist display events, then emit one fan-out copy per
subscribed viewer (with ``tabId`` rewritten; not recorded, not
persisted).
"""

from __future__ import annotations

from typing import Any

from kiss.agents.vscode.browser_ui import BaseBrowserPrinter


class MemoryPrinter(BaseBrowserPrinter):
    """Records every emitted event into ``self.emitted``."""

    def __init__(self) -> None:
        """Initialise an empty in-memory emission buffer."""
        super().__init__()
        self.emitted: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record + persist + capture the primary event, then fan out.

        Args:
            event: The event dictionary to emit.  ``_inject_tab_id``
                inside :meth:`BaseBrowserPrinter.broadcast` rewrites
                it in place with the thread-local ``tabId`` when
                absent before it is recorded; we use the same
                injected reference for the capture buffer so test
                assertions on ``tabId`` see the value the production
                transport would see.
        """
        # ``BaseBrowserPrinter.broadcast`` (re)assigns ``event`` to the
        # injected variant.  We need the same reference for ``emitted``,
        # so inject explicitly first and pass the same dict down.
        event = self._inject_tab_id(event)
        super().broadcast(event)
        self.emitted.append(event)
        for viewer in self._fanout_targets(event.get("tabId")):
            self.emitted.append({**event, "tabId": viewer})
