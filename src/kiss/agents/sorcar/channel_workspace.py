# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Reference-counted publication of the active channel workspace.

Channel agent modules read their multi-account workspace identifier
from the process-global ``KISS_CHANNEL_WORKSPACE`` environment
variable when their ``add_to_tools()`` runs on the daemon.  The
daemon's task runner publishes the workspace through the helpers here
for every ``channel``-kind run (``/slack ...``, ``run_agent("slack")``,
``run_agent(".../slack_sea.py")``, a channel CLI's launch; see
``kiss.server.task_runner``), from before the tools are built until
the run ends.

The hold is recorded on the entering THREAD (:func:`held_workspace`)
in the same locked step that takes it, so the task runner's final
cleanup releases exactly what its thread holds — a stop injected
anywhere between the acquisition and the cleanup cannot leak the hold
or release one it does not own.

The env var is process-global, so it is managed by reference counting
rather than save/restore snapshots: snapshots taken by overlapping
launches restore each other's values out of order, leaving a stale
workspace exported after every task has finished.  Overlapping
launches sharing ONE workspace run concurrently; a launch with a
DIFFERENT workspace waits in :func:`enter_workspace` until the others
finish (or its timeout expires), because overwriting the exported
value would hand the running launches the wrong account's
credentials.

The helpers live in the sorcar layer so the daemon and the dispatch
tool can use them without importing
``kiss.agents.third_party_agents`` (see the layering invariant in
``kiss.tests.agents.sorcar.test_layering_invariants``).
"""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

WORKSPACE_ENV_VAR = "KISS_CHANNEL_WORKSPACE"

WORKSPACE_WAIT_TIMEOUT_SECONDS = 900.0
"""Bound on a run's wait for a conflicting channel workspace to free up.

A ``channel``-kind run whose workspace differs from a running channel
task's waits at most this long in :func:`enter_workspace` before it
fails with a diagnostic, so a conflicting task cannot hang it forever.
"""
_WORKSPACE_COND = threading.Condition()
_ACTIVE_WORKSPACES: dict[str, int] = {}
_HELD = threading.local()


def held_workspace() -> str:
    """Return the workspace the calling thread entered and has not exited yet, else ``""``."""
    return str(getattr(_HELD, "workspace", "") or "")


def enter_workspace(workspace: str, timeout: float | None = None) -> bool:
    """Mark a launch's workspace active and publish it to the env var.

    The env var is process-global, so a launch that exported a
    DIFFERENT workspace and is still running must finish first:
    overwriting its value would make that launch's daemon-side channel
    ``add_to_tools()`` read THIS launch's workspace and load the wrong
    account's credentials.  This call therefore blocks until no other
    workspace is active (same-workspace launches overlap freely via
    the reference count) or *timeout* expires.

    Args:
        workspace: The launching agent's workspace identifier.
        timeout: Maximum seconds to wait for conflicting launches to
            finish; ``None`` waits indefinitely.

    Returns:
        ``True`` when the workspace was entered, recorded as the calling
        thread's hold (:func:`held_workspace`) and published (the
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
        # The count and the thread's record are two adjacent stores
        # with no call between them: an asynchronously injected stop
        # cannot land between the hold and its record.
        _ACTIVE_WORKSPACES[workspace] = _ACTIVE_WORKSPACES.get(workspace, 0) + 1
        _HELD.workspace = workspace
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
        if getattr(_HELD, "workspace", "") == workspace:
            _HELD.workspace = ""
        count = _ACTIVE_WORKSPACES.get(workspace, 0) - 1
        if count > 0:
            _ACTIVE_WORKSPACES[workspace] = count
        else:
            _ACTIVE_WORKSPACES.pop(workspace, None)
        if not _ACTIVE_WORKSPACES:
            os.environ.pop(WORKSPACE_ENV_VAR, None)
        _WORKSPACE_COND.notify_all()
