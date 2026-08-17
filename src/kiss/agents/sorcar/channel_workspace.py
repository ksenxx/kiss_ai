# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Reference-counted publication of the active channel workspace.

Channel agent modules read their multi-account workspace identifier
from the process-global ``KISS_CHANNEL_WORKSPACE`` environment
variable when their ``get_tools()`` runs on the daemon.  Everything
that launches a channel session — the channel CLIs' kiss-web launcher
(``kiss.agents.third_party_agents._kiss_web_launcher``) and the sorcar
``run_agent`` dispatch tool (:mod:`kiss.agents.sorcar.agent_dispatch`)
— publishes the workspace through the helpers here.

The env var is process-global, so it is managed by reference counting
rather than save/restore snapshots: snapshots taken by overlapping
launches restore each other's values out of order, leaving a stale
workspace exported after every task has finished.

The helpers live in the sorcar layer so the dispatch tool can use
them without importing ``kiss.agents.third_party_agents`` (see the
layering invariant in
``kiss.tests.agents.sorcar.test_layering_invariants``); the launcher
imports them back from here, keeping the single shared registry.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

WORKSPACE_ENV_VAR = "KISS_CHANNEL_WORKSPACE"
_WORKSPACE_LOCK = threading.Lock()
_ACTIVE_WORKSPACES: dict[str, int] = {}


def enter_workspace(workspace: str) -> None:
    """Mark a launch's workspace active and publish it to the env var.

    Args:
        workspace: The launching agent's workspace identifier.
    """
    with _WORKSPACE_LOCK:
        _ACTIVE_WORKSPACES[workspace] = _ACTIVE_WORKSPACES.get(workspace, 0) + 1
        os.environ[WORKSPACE_ENV_VAR] = workspace
        if len(_ACTIVE_WORKSPACES) > 1:
            logger.warning(
                "concurrent kiss-web launches use different workspaces %s "
                "but share the single process-global %s environment "
                "variable; their daemon-side get_tools() may see the "
                "wrong workspace",
                sorted(_ACTIVE_WORKSPACES),
                WORKSPACE_ENV_VAR,
            )


def exit_workspace(workspace: str) -> None:
    """Mark a launch's workspace inactive and clean up the env var.

    When the last active launch finishes the env var is removed; while
    other launches remain active the env var is kept pointing at one of
    their workspaces.

    Args:
        workspace: The workspace passed to :func:`enter_workspace`.
    """
    with _WORKSPACE_LOCK:
        count = _ACTIVE_WORKSPACES.get(workspace, 0) - 1
        if count > 0:
            _ACTIVE_WORKSPACES[workspace] = count
        else:
            _ACTIVE_WORKSPACES.pop(workspace, None)
        if not _ACTIVE_WORKSPACES:
            os.environ.pop(WORKSPACE_ENV_VAR, None)
        elif os.environ.get(WORKSPACE_ENV_VAR) not in _ACTIVE_WORKSPACES:
            os.environ[WORKSPACE_ENV_VAR] = next(iter(_ACTIVE_WORKSPACES))
