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
workspace exported after every task has finished.  Overlapping
launches sharing ONE workspace run concurrently; a launch with a
DIFFERENT workspace waits in :func:`enter_workspace` until the others
finish (or its timeout expires), because overwriting the exported
value would hand the running launches the wrong account's
credentials.

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
import time

logger = logging.getLogger(__name__)

WORKSPACE_ENV_VAR = "KISS_CHANNEL_WORKSPACE"
_WORKSPACE_COND = threading.Condition()
_ACTIVE_WORKSPACES: dict[str, int] = {}


def enter_workspace(workspace: str, timeout: float | None = None) -> bool:
    """Mark a launch's workspace active and publish it to the env var.

    The env var is process-global, so a launch that exported a
    DIFFERENT workspace and is still running must finish first:
    overwriting its value would make that launch's daemon-side channel
    ``get_tools()`` read THIS launch's workspace and load the wrong
    account's credentials.  This call therefore blocks until no other
    workspace is active (same-workspace launches overlap freely via
    the reference count) or *timeout* expires.

    Args:
        workspace: The launching agent's workspace identifier.
        timeout: Maximum seconds to wait for conflicting launches to
            finish; ``None`` waits indefinitely.

    Returns:
        ``True`` when the workspace was entered and published (the
        caller MUST pair it with :func:`exit_workspace`); ``False``
        when *timeout* expired while a different workspace was still
        active (nothing was entered — the caller must not launch, and
        must not call :func:`exit_workspace`).
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    with _WORKSPACE_COND:
        while any(active != workspace for active in _ACTIVE_WORKSPACES):
            remaining = (
                None if deadline is None else deadline - time.monotonic()
            )
            if remaining is not None and remaining <= 0:
                logger.error(
                    "timed out waiting for concurrent kiss-web launches "
                    "using workspaces %s to finish before publishing "
                    "workspace %r to the process-global %s environment "
                    "variable",
                    sorted(_ACTIVE_WORKSPACES), workspace, WORKSPACE_ENV_VAR,
                )
                return False
            logger.info(
                "waiting for concurrent kiss-web launches using "
                "workspaces %s to finish before publishing workspace %r",
                sorted(_ACTIVE_WORKSPACES), workspace,
            )
            _WORKSPACE_COND.wait(remaining)
        _ACTIVE_WORKSPACES[workspace] = _ACTIVE_WORKSPACES.get(workspace, 0) + 1
        os.environ[WORKSPACE_ENV_VAR] = workspace
        return True


def exit_workspace(workspace: str) -> None:
    """Mark a launch's workspace inactive and clean up the env var.

    When the last active launch finishes the env var is removed and
    launches blocked in :func:`enter_workspace` are woken up.

    Args:
        workspace: The workspace passed to :func:`enter_workspace`.
    """
    with _WORKSPACE_COND:
        count = _ACTIVE_WORKSPACES.get(workspace, 0) - 1
        if count > 0:
            _ACTIVE_WORKSPACES[workspace] = count
        else:
            _ACTIVE_WORKSPACES.pop(workspace, None)
        if not _ACTIVE_WORKSPACES:
            os.environ.pop(WORKSPACE_ENV_VAR, None)
        _WORKSPACE_COND.notify_all()
