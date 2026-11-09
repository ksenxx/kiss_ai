# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Probes of a daemon's remote-tab bindings for the tests under this package.

The surfaces learn a tab's machine from the ``tabMachine``
announcement and the server routes through
:meth:`~kiss.server.remote_tabs.RemoteTabs.route`; the tests read the
binding itself, from the mapping those keep.
"""

from __future__ import annotations

from kiss.server.remote_tabs import RemoteTabs


def machine_of(tabs: RemoteTabs, tab_id: str) -> str:
    """Return the name of the machine *tab_id* is bound to in *tabs* ("" when it runs here)."""
    with tabs._lock:
        tab = tabs._tabs.get(tab_id)
    return tab.machine.name if tab is not None else ""
