# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sorcar agent with both coding tools and browser automation."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar._concurrency import _race_delay
from kiss.agents.sorcar.persistence import _load_last_model, is_task_history_id
from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.agents.sorcar.skills import make_skill_tool
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.core.base import SYSTEM_PROMPT
from kiss.core.kiss_error import BudgetExceededError, KISSError
from kiss.core.models.model import Attachment
from kiss.core.models.model_info import (
    MODEL_INFO,
    OPENAI_COMPATIBLE_PROVIDERS,
    _match_openai_compatible_provider,
    _strip_provider_prefix,
    get_default_model,
)
from kiss.core.models.model_info import model as _model_factory
from kiss.core.printer import Printer

logger = logging.getLogger(__name__)


def _generate_commit_message(
    commit_dir: Path,
    user_prompt: str | None = None,
    task_result: str | None = None,
) -> str:
    """Generate a commit message for staged changes using an LLM.

    Gets the staged diff and delegates to
    :func:`~kiss.agents.sorcar.commit_message.generate_commit_message_from_diff`.
    When *user_prompt* is provided, it is forwarded so the user's
    task prompt is incorporated into the commit message.  When
    *task_result* is provided, the task's result summary is appended
    to the commit message as well.

    Args:
        commit_dir: The directory containing staged changes.
        user_prompt: The user's task prompt that produced these
            staged changes, or ``None`` when not available.
        task_result: The task's result summary, or ``None`` when
            not available.

    Returns:
        A commit message string.
    """
    from kiss.agents.sorcar.commit_message import generate_commit_message_from_diff
    from kiss.agents.sorcar.git_worktree import GitWorktreeOps

    diff_text = GitWorktreeOps.staged_diff(commit_dir)
    return generate_commit_message_from_diff(
        diff_text, user_prompt=user_prompt, task_result=task_result,
    )


def auto_commit_changes(
    commit_dir: Path,
    user_prompt: str | None,
    message_fn: Callable[[Path, str | None, str | None], str],
    notify_fn: Callable[[str, str], None] | None = None,
    task_result: str | None = None,
) -> bool:
    """Stage all changes, generate a commit message, and commit.

    Stages once so *message_fn* can compute the diff, runs
    *message_fn* (typically a slow LLM call) to generate the commit
    subject/body, then re-stages immediately before the commit so
    any file that appeared in the worktree during the LLM call
    (e.g. ``PROGRESS.md`` rewrites, macOS ``.DS_Store`` materializing
    after an ``open`` of the report, an editor side-channel saving
    swap files) is included in the same commit.  Without the second
    ``stage_all`` those late-arriving files would be left
    uncommitted, ``_finalize_worktree`` would see them via
    ``has_uncommitted_changes`` and abort the auto-merge with the
    misleading "pre-commit hook may have rejected" warning
    (observed in production on 2026-06-26 07:23:14 for worktree
    ``kiss_wt-1782483430-cb03445c`` even though the repo had no
    custom pre-commit hooks installed).

    Falls back to a generic commit message when *message_fn* raises
    (e.g. the LLM-based generator is unavailable).

    Args:
        commit_dir: Directory whose changes are staged and committed.
        user_prompt: The user's task prompt, woven into the commit
            message (or its fallback), or ``None`` when unavailable.
        message_fn: Callable producing a commit message from
            ``(commit_dir, user_prompt, task_result)``.
        task_result: The task's result summary, appended to the
            commit message (or its fallback) under a ``Result:``
            heading, or ``None`` when unavailable.
        notify_fn: Optional UI callback invoked at two life-cycle
            points so the chat webview can render toasts:

            - ``notify_fn("generating", "")`` immediately before
              *message_fn* runs (typically a slow LLM call) so the
              user sees "Generating commit message" while the LLM
              works.
            - ``notify_fn("committed", subject)`` immediately after
              a successful commit, where *subject* is the first
              non-empty line of the committed message.

            Both hooks are SKIPPED when there is nothing to commit
            (no staged diff after the initial ``stage_all``), so the
            webview never sees a misleading "Generating commit
            message" toast without a follow-up.  When
            ``commit_staged`` returns ``False`` after *message_fn*
            (e.g. a pre-commit hook rejected the commit),
            ``notify_fn("failed", "")`` is invoked instead so the
            sticky "generating" toast always gets a terminal update.

            All ``notify_fn`` exceptions are swallowed so a broken
            UI hook can never block the commit itself.

    Returns:
        True if a commit was created, False if nothing to commit.
    """
    from kiss.agents.sorcar.git_worktree import GitWorktreeOps

    GitWorktreeOps.stage_all(commit_dir)
    if not GitWorktreeOps.staged_diff(commit_dir):
        return False
    _safe_notify(notify_fn, "generating", "")
    try:
        msg = message_fn(commit_dir, user_prompt, task_result)
    except Exception:
        logger.debug(
            "LLM commit message generation failed; using fallback", exc_info=True,
        )
        msg = "kiss: auto-commit agent changes"
        if user_prompt:
            from kiss.agents.sorcar.commit_message import _append_user_prompt

            msg = _append_user_prompt(msg, user_prompt)
        if task_result:
            from kiss.agents.sorcar.commit_message import _append_task_result

            msg = _append_task_result(msg, task_result)
    GitWorktreeOps.stage_all(commit_dir)
    committed = GitWorktreeOps.commit_staged(commit_dir, msg)
    if committed:
        _safe_notify(notify_fn, "committed", _commit_subject(msg))
    else:
        # Terminal notification for the failure path: without it the
        # sticky "Generating commit message" toast (emitted above)
        # would linger in the webview forever after e.g. a pre-commit
        # hook rejection.
        _safe_notify(notify_fn, "failed", "")
    return committed


def _commit_subject(message: str) -> str:
    """Return the first non-empty line of a commit *message*.

    Used as the subject the chat webview renders inside the
    "Committed <subject>" toast.  Falls back to an empty string when
    the message has no printable line (defensive: in practice the
    fallback message always starts with ``kiss:``).
    """
    for raw in message.splitlines():
        line = raw.strip()
        if line:
            return line
    return ""


def _safe_notify(
    notify_fn: Callable[[str, str], None] | None,
    stage: str,
    subject: str,
) -> None:
    """Invoke *notify_fn* swallowing any exception.

    Errors in the optional UI hook must never prevent the commit
    itself (and must never poison the surrounding ``except`` block
    that the LLM-failure fallback relies on).
    """
    if notify_fn is None:
        return
    try:
        notify_fn(stage, subject)
    except Exception:
        logger.debug("auto_commit_changes notify_fn raised", exc_info=True)


def _yaml_failure(exc: BaseException) -> str:
    """Return a YAML result string for an unhandled sub-agent exception."""
    failure: str = yaml.dump(
        {"success": False, "summary": f"Unhandled exception: {exc}"},
        sort_keys=False,
    )
    return failure


def _agent_usage(agent: Any) -> tuple[float, int, int]:
    """Return ``(budget_used, total_tokens_used, total_steps)`` for *agent*."""
    return (
        float(getattr(agent, "budget_used", 0.0) or 0.0),
        int(getattr(agent, "total_tokens_used", 0) or 0),
        int(getattr(agent, "total_steps", 0) or 0),
    )


def _broadcast_subagent_done(
    printer: Any, tab_ids: list[str], model: str = "",
) -> None:
    """Broadcast ``subagentDone`` for each tab id so the frontend can
    stop the running indicator on the sub-agent tab.

    A sub-agent that switched models with ``set_model`` also has to hand
    its tab's model picker back to *model* — the model the task was
    launched with — for the same reason a top-level task does.  Errors
    are swallowed (the broadcast is best-effort UI signalling).

    Args:
        printer: The printer to broadcast through.
        tab_ids: The sub-agent's tab plus any tabs viewing it.
        model: The model the sub-agent was launched with, restored into
            those tabs' pickers.
    """
    broadcast = getattr(printer, "broadcast", None)
    if broadcast is None:
        return
    restore = getattr(printer, "restore_model_pick", None)
    for vid in tab_ids:
        try:
            broadcast({"type": "subagentDone", "tab_id": vid, "tabId": ""})
            if callable(restore) and model:
                restore(model, vid)
        except Exception:
            pass


# How long the parent may sit in one wait() before re-reading its stop
# event.  A completed child wakes the wait immediately, so this only
# bounds flag-checking: the abandon path below allows 15s anyway, and
# ``_force_stop_thread`` waits 1s before its first injection and retries
# at +5s.  It must not be much smaller: nested fan-outs put one waiting
# parent on the stack per level, and every one of them wakes on this
# interval, so a 0.1s slice made a deeply nested tree crawl under GIL
# contention.
_SUBAGENT_POLL_SECONDS = 1.0
_SUBAGENT_STOP_GRACE_SECONDS = 15.0


class _SubagentStopEvent(threading.Event):
    """Per-sub-agent stop event chained to the parent task's stop event.

    Each parallel sub-agent worker gets its own instance so the user
    can stop ONLY that sub-agent's task (``VSCodeServer._stop_task``
    resolves the sub-agent's registered ``stop_event`` and
    calls :meth:`set`, which flips just this event).  At the same time
    a stop of the PARENT task must keep killing the whole fan-out, so
    :meth:`is_set` and :meth:`wait` also observe the parent event —
    every consumer (``JsonPrinter._check_stop``'s per-print poll, the
    ``UsefulTools`` bash process-group killer's poll loop, and the
    0.1 s ``stop.wait`` loops) sees the union of the two signals.
    Nested ``run_parallel`` fan-outs chain transitively: the inner
    event's parent is the outer sub-agent's event.
    """

    def __init__(self, parent: threading.Event | None = None) -> None:
        """Create an unset event linked to *parent* (may be ``None``)."""
        super().__init__()
        self._parent_event = parent

    def is_set(self) -> bool:
        """True when this event OR any ancestor parent event is set.

        Walks the parent chain ITERATIVELY: deeply nested
        ``run_parallel`` fan-outs chain one linked event per level, so
        a recursive walk could hit the interpreter recursion limit.
        """
        ev: threading.Event | None = self
        while isinstance(ev, _SubagentStopEvent):
            if threading.Event.is_set(ev):
                return True
            ev = ev._parent_event
        return ev is not None and ev.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait until this event or an ancestor is set.

        Polls the parent chain on a short interval (0.05 s) so a
        parent-task stop wakes waiters promptly even though the parent
        event has no reference back to this child event.

        Args:
            timeout: Maximum seconds to wait; ``None`` waits forever.

        Returns:
            True when the event (or an ancestor) is set, else False
            after *timeout* elapsed.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self.is_set():
                return True
            slice_s = 0.05
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self.is_set()
                slice_s = min(slice_s, remaining)
            if super().wait(slice_s):
                return True


def _await_subagents(
    futures: list[Future[str]],
    stop_event: threading.Event | None,
) -> list[str]:
    """Collect fan-out results without becoming unstoppable.

    ``list(pool.map(...))`` parks the parent thread in a C-level lock,
    where it can neither poll its stop event (a parent prints nothing
    while its children run) nor accept the ``KeyboardInterrupt`` that
    ``VSCodeServer._stop_task`` injects — CPython delivers an injected
    exception only at a bytecode boundary.  That is why the parent of
    task ``709ebce3`` outlived the Stop click by three minutes
    (``reports/stop_button_delay_2026-08-05.html``).  Waiting in short
    slices instead keeps the parent at a bytecode boundary throughout.

    A stopped child normally unwinds in well under a second, and its
    result is still collected so sibling spend and summaries survive.
    Only a child that ignores its stop event for
    ``_SUBAGENT_STOP_GRACE_SECONDS`` is abandoned, so that one wedged
    sub-agent can no longer hold the whole task hostage.

    Args:
        futures: One future per fanned-out sub-agent, in task order.
        stop_event: The parent task's stop event, or ``None`` when the
            fan-out is not running under a stoppable task.

    Returns:
        The sub-agent results, in the order the tasks were given.

    Raises:
        KeyboardInterrupt: When a stop was requested and at least one
            child was still running after the grace period.
    """
    pending = set(futures)
    give_up_at: float | None = None
    while pending:
        _done, pending = wait(pending, timeout=_SUBAGENT_POLL_SECONDS)
        if not pending:
            break
        if stop_event is None or not stop_event.is_set():
            continue
        if give_up_at is None:
            give_up_at = time.monotonic() + _SUBAGENT_STOP_GRACE_SECONDS
        elif time.monotonic() >= give_up_at:
            raise KeyboardInterrupt("Agent stop requested")
    return [f.result() for f in futures]


def _collect_unfinished_usage(
    futures: list[Future[str]],
    sub_agents: list[Any],
    sub_usage: list[tuple[float, int, int]],
    lock: threading.Lock,
) -> None:
    """Fill in the spend of children that never got to report it.

    A child fills its own ``sub_usage`` slot in its ``finally``, so a
    child the parent abandoned (see :func:`_await_subagents`) would leave
    a zero there and its cost, tokens and steps would silently vanish
    from the parent task's totals.  Reading the live figures off the
    child's agent recovers everything it had spent up to this instant —
    without waiting for it, which is the whole point of abandoning it.

    Every slot update — the worker's final write in its ``finally`` and
    this read-modify-write — happens under *lock*, so the two can no
    longer interleave: before, a child that published its final figure
    and completed between this function's read and its write had that
    figure overwritten by the older live read, and, its future now
    being done, it was not registered as abandoned either — the
    difference was never banked.  The component-wise maximum remains
    for a child that has already published but whose future is not yet
    done: a live read can lag its true spend slightly (mid-handoff
    between executor sessions), so a slot is only ever raised, never
    lowered.

    Args:
        futures: One future per fanned-out sub-agent, in task order.
        sub_agents: The children's agents, in the same order; entries are
            ``None`` for children that never started.
        sub_usage: Per-child ``(cost, tokens, steps)`` slots, raised in
            place for unfinished children only.
        lock: The lock the workers hold for their own final slot write.
    """
    for idx, future in enumerate(futures):
        agent = sub_agents[idx]
        if future.done() or agent is None:
            continue
        with lock:
            live = _live_agent_usage(agent)
            current = sub_usage[idx]
            sub_usage[idx] = (
                max(current[0], live[0]),
                max(current[1], live[1]),
                max(current[2], live[2]),
            )


class _AbandonedSubagent:
    """A sub-agent thread its parent gave up waiting for.

    :func:`_await_subagents` abandons a child that ignores its stop
    event for :data:`_SUBAGENT_STOP_GRACE_SECONDS`, but Python cannot
    kill a thread: the child keeps running with ``work_dir`` set to the
    parent's directory (a git worktree, for a server run) and keeps
    spending budget.  Holding on to it lets the parent (a) refuse to
    delete a directory a live thread is still writing to and (b) bank
    the spend the child reports after it was abandoned.
    """

    def __init__(
        self,
        future: Future[str],
        agent: Any,
        counted: tuple[float, int, int],
    ) -> None:
        """Record *future*/*agent* and the usage already attributed."""
        self.future = future
        self.agent = agent
        self.counted = counted

    def unbanked_usage(self) -> tuple[float, int, int]:
        """Return the spend not yet attributed to the parent.

        Returns:
            The ``(budget, tokens, steps)`` delta between the child's
            live figures and what the parent has already counted.
        """
        live = _live_agent_usage(self.agent)
        delta = (
            live[0] - self.counted[0],
            live[1] - self.counted[1],
            live[2] - self.counted[2],
        )
        # Test hook (no-op in production): widens the read-modify-
        # write window so concurrency tests can prove the caller
        # serialises reclaims (see reclaim_abandoned_subagents).
        _race_delay()
        self.counted = live
        return delta


def _persisted_task_id(agent: Any) -> str:
    """Return *agent*'s persisted ``task_history`` row id, or ``""``.

    Only :class:`~kiss.agents.sorcar.chat_sorcar_agent.ChatSorcarAgent`
    and its subclasses persist a row (and expose the ``last_task_id``
    accessor that reads it under the agent's lock), and even they have
    no id before their first ``run``, so every caller must tolerate
    ``""``.

    Args:
        agent: Any agent object, or ``None``.

    Returns:
        The row id, or ``""`` when the agent has none.
    """
    task_id = getattr(agent, "last_task_id", "") if agent is not None else ""
    return task_id if isinstance(task_id, str) else ""


def _register_abandoned(
    parent_agent: Any,
    futures: list[Future[str]],
    sub_agents: list[Any],
    sub_usage: list[tuple[float, int, int]],
) -> None:
    """Hand every still-running child to *parent_agent* for follow-up.

    Called only on the abandon path.  ``sub_usage`` has just been
    refreshed from the live children, so it is exactly what the parent
    has counted for each of them.

    Args:
        parent_agent: The fanning-out agent, or ``None`` for a bare
            functional call (nothing can be reclaimed then).
        futures: One future per child, in task order.
        sub_agents: The children's agents, in the same order.
        sub_usage: Per-child ``(cost, tokens, steps)`` already counted.
    """
    pending = getattr(parent_agent, "_abandoned_subagents", None)
    lock = getattr(parent_agent, "_abandoned_lock", None)
    if pending is None or lock is None:
        return
    with lock:
        for idx, future in enumerate(futures):
            if future.done() or sub_agents[idx] is None:
                continue
            pending.append(
                _AbandonedSubagent(future, sub_agents[idx], sub_usage[idx])
            )


def _executor_usage(agent: Any) -> tuple[float, int, int]:
    """Return the in-flight executor session's ``(budget, tokens, steps)``.

    :class:`~kiss.agents.sorcar.relentless_agent.RelentlessAgent` folds a
    session executor's spend into the agent's totals only when the
    session ends, so mid-session the live spend is visible only on
    ``agent._current_executor``.  The single reader of that executor's
    counters — :func:`_live_agent_usage` and
    :meth:`_LiveUsageMonitor._emit` used to carry drifting copies (the
    executor's step counter is ``step_count``, not ``total_steps``, an
    easy copy to get wrong).

    Args:
        agent: The agent whose live executor to read.

    Returns:
        The executor's spend, or ``(0.0, 0, 0)`` when no session is in
        flight.
    """
    executor = getattr(agent, "_current_executor", None)
    if executor is None:
        return 0.0, 0, 0
    return (
        float(getattr(executor, "budget_used", 0.0) or 0.0),
        int(getattr(executor, "total_tokens_used", 0) or 0),
        int(getattr(executor, "step_count", 0) or 0),
    )


def _live_agent_usage(agent: Any) -> tuple[float, int, int]:
    """Return live ``(budget, tokens, steps)`` for *agent*, including its
    in-flight executor session (see :func:`_executor_usage`).
    """
    budget, tokens, steps = _agent_usage(agent)
    live_budget, live_tokens, live_steps = _executor_usage(agent)
    return budget + live_budget, tokens + live_tokens, steps + live_steps


class _LiveUsageMonitor:
    """Streams the parent task's live cumulative usage while parallel
    sub-agents run.

    Between the moment ``run_parallel`` blocks the parent's turn and the
    moment :func:`_attribute_sub_usage` folds the finished sub-agents'
    spend back into the parent, nothing else emits ``usage_info`` on the
    PARENT task — the cost/tokens header (chat webview top bar)
    would otherwise show a stale figure that excludes
    all live sub-agent spend until every sub-agent finished.  This
    monitor polls every tracked sub-agent and broadcasts a parent-task
    ``usage_info`` whenever the totals change, so the header always
    reflects the agent plus all of its sub-agents at every turn.

    The emitted values are RAW (session-relative), exactly like the
    per-turn ``usage_info`` from ``KISSAgent``: the printer adds the
    parent task's budget/tokens/steps offsets (the parent's cumulative
    spend snapshotted at session start).  :meth:`stop` joins the polling
    thread and is called BEFORE ``_attribute_sub_usage`` bumps those
    offsets, so a late emission can never double-count sub-agent spend.
    """

    def __init__(self, parent: Any, printer: Any, interval: float = 1.0) -> None:
        self._parent = parent
        self._printer = printer
        self._interval = interval
        self._agents_lock = threading.Lock()
        self._agents: list[Any] = []
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_emitted: tuple[float, int, int] | None = None
        thread_local = getattr(printer, "_thread_local", None) if printer else None
        self._parent_task_id = (
            getattr(thread_local, "task_id", "") if thread_local else ""
        )

    def track(self, agent: Any) -> None:
        """Register a spawned sub-agent whose live spend should be polled."""
        with self._agents_lock:
            self._agents.append(agent)

    def start(self) -> None:
        """Start the polling thread (no-op without a printer)."""
        if self._printer is None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="live-usage-monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop and join the polling thread.

        The monitor emits one final snapshot before its thread exits.
        Joining then guarantees no later emission can race with the
        subsequent :func:`_attribute_sub_usage` offset bump (which would
        double-count the sub-agents' spend in the displayed total).
        """
        self._done.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _loop(self) -> None:
        thread_local = getattr(self._printer, "_thread_local", None)
        if thread_local is not None:
            thread_local.task_id = self._parent_task_id
        while True:
            stopping = self._done.wait(self._interval)
            try:
                # A final poll on shutdown captures sub-agents that finished
                # between regular ticks.  It runs on this thread so the event
                # retains the parent task id, and _last_emitted suppresses a
                # duplicate when the preceding regular poll saw the same data.
                self._emit()
            except Exception:
                logger.debug("Live usage emission failed", exc_info=True)
            if stopping:
                return

    def _emit(self) -> None:
        """Broadcast a parent-task ``usage_info`` when the totals changed."""
        # Only the parent's LIVE executor session, never its folded
        # totals: the printer adds the parent's cumulative offsets to
        # every raw usage_info it renders.
        budget, tokens, steps = _executor_usage(self._parent)
        with self._agents_lock:
            agents = list(self._agents)
        for sub in agents:
            try:
                sub_budget, sub_tokens, sub_steps = _live_agent_usage(sub)
            except Exception:
                logger.debug("Live usage poll failed", exc_info=True)
                continue
            budget += sub_budget
            tokens += sub_tokens
            steps += sub_steps
        snapshot = (budget, tokens, steps)
        if snapshot == self._last_emitted:
            return
        if self._last_emitted is not None:
            last_budget, last_tokens, last_steps = self._last_emitted
            if (
                budget < last_budget - 1e-9
                or tokens < last_tokens
                or steps < last_steps
            ):
                # Torn read: at every RelentlessAgent session handoff the
                # executor is detached BEFORE its spend is folded into the
                # agent fields, so a poll in that window sees neither copy.
                # Never emit a total where ANY cumulative dimension
                # (budget, tokens, or steps) regresses — the next poll
                # repairs it.
                return
        self._last_emitted = snapshot
        cost = f"${budget:.4f}"
        self._printer.print(
            f"Tokens: {tokens:,}, Budget: {cost} (live, incl. parallel sub-agents), ",
            type="usage_info",
            total_tokens=tokens,
            cost=cost,
            total_steps=steps,
        )


def _attribute_sub_usage(agent: Any, budget: float, tokens: int, steps: int) -> None:
    """Attribute sub-agents' cost, tokens, and steps to the parent *agent*.

    Without this, sub-agent budgets would be invisible to the parent
    agent's global accounting and UI.  Also updates the printer offsets
    so the live status line in the current sub-session reflects the
    additional spend immediately (the offsets are otherwise
    snapshotted only at session start).

    The whole read-modify-write runs under the agent's ``_usage_lock``
    (see :meth:`RelentlessAgent.__init__`): this function is called
    concurrently by the agent thread (a fan-out's ``finally``, a
    ``talk`` synthesis bank) and by server threads
    (:meth:`SorcarAgent.reclaim_abandoned_subagents` from worktree
    cleanup / teardown / discard), and unserialized increments lost
    updates.  ``reclaim_abandoned_subagents`` calls this while holding
    ``_abandoned_lock``, so the (fixed) lock order is
    ``_abandoned_lock`` → ``_usage_lock``; nothing acquires them in
    the opposite order.  A minimal agent-shaped object without the
    lock attribute gets a throwaway lock (no cross-thread protection,
    but such objects are single-threaded by construction).
    """
    lock = getattr(agent, "_usage_lock", None) or threading.Lock()
    with lock:
        agent.budget_used = float(getattr(agent, "budget_used", 0.0) or 0.0) + budget
        agent.total_tokens_used = (
            int(getattr(agent, "total_tokens_used", 0) or 0) + tokens
        )
        agent.total_steps = int(getattr(agent, "total_steps", 0) or 0) + steps
        if agent.printer is not None:
            try:
                agent.printer.budget_offset = agent.budget_used
                agent.printer.tokens_offset = agent.total_tokens_used
                agent.printer.steps_offset = agent.total_steps
            except Exception:
                pass


def _attribute_tts_usage(agent: Any, usage: dict[str, Any]) -> None:
    """Attribute a ``talk``-tool TTS synthesis call's spend to *agent*.

    ``synthesize_talk_audio`` runs a throwaway single-shot
    ``TalkSynthesisAgent`` (gpt-audio-1.5, whose audio output bills
    $64/M tokens) whose ``budget_used`` would otherwise vanish from the
    task's accounting — the reported per-task cost would lie low (July
    2026 cost audit).  Steps are NOT attributed: the synthesis is one
    non-agentic model call, not an agent step.

    Args:
        agent: The Sorcar agent whose task accounting receives the spend.
        usage: The ``usage_out`` dict filled by ``synthesize_talk_audio``
            (``budget_used`` USD, ``total_tokens_used``); empty when the
            synthesis never issued an API call.
    """
    budget = float(usage.get("budget_used", 0.0) or 0.0)
    tokens = int(usage.get("total_tokens_used", 0) or 0)
    if budget > 0 or tokens > 0:
        _attribute_sub_usage(agent, budget, tokens, 0)


_FACTORY_DEFAULT_BASE_URLS: frozenset[str] = frozenset(
    provider.base_url.rstrip("/") for provider in OPENAI_COMPATIBLE_PROVIDERS
)


_PROVIDER_SPECIFIC_CONFIG_KEYS: dict[str, frozenset[str]] = {
    "openai": frozenset({"reasoning_effort", "use_responses_api"}),
    "anthropic": frozenset({"thinking"}),
    "gemini": frozenset({"thinking_config"}),
}


def _model_family(model_name: str) -> str:
    """Return the provider family *model_name* routes to in the factory.

    Mirrors the routing order of :func:`kiss.core.models.model_info.model`:
    OpenAI-compatible providers first, then Gemini, then Anthropic.

    Args:
        model_name: A model name, possibly carrying a harbor-style
            ``provider/`` prefix.

    Returns:
        One of ``"openai"``, ``"gemini"``, ``"anthropic"``, or
        ``"other"``.
    """
    name = _strip_provider_prefix(model_name)
    if _match_openai_compatible_provider(name) is not None:
        return "openai"
    if name.startswith("gemini-"):
        return "gemini"
    if name.startswith("claude-"):
        return "anthropic"
    return "other"


def _sanitize_model_config_for_switch(
    config: dict[str, Any], old_model_name: str, new_model_name: str,
) -> dict[str, Any]:
    """Drop source-provider request options that the target cannot accept.

    ``set_model`` copies the old adapter's complete ``model_config``
    onto the new one.  Provider-specific request options (Anthropic's
    ``thinking``, Gemini's ``thinking_config``, OpenAI's
    ``reasoning_effort`` / ``use_responses_api``) survive that copy and
    are then sent as unsupported SDK kwargs by the target adapter, so
    the switch reports success but the next model request fails.  When
    the provider family changes, remove every known provider-specific
    key that does not belong to the target family.

    Args:
        config: The config dict to sanitize (mutated in place).
        old_model_name: The model name the config came from.
        new_model_name: The model name the config is being given to.

    Returns:
        The same *config* dict, for chaining.
    """
    new_family = _model_family(new_model_name)
    if new_family == _model_family(old_model_name):
        return config
    for family, keys in _PROVIDER_SPECIFIC_CONFIG_KEYS.items():
        if family == new_family:
            continue
        for key in keys:
            config.pop(key, None)
    return config


_ATTACHMENT_KINDS: tuple[tuple[str, str], ...] = (
    ("image/", "image(s)"),
    ("application/pdf", "PDF(s)"),
    ("audio/", "audio file(s)"),
    ("video/", "video file(s)"),
)


def _attachment_parts(attachments: list[Attachment]) -> list[str]:
    """Return human-readable per-kind attachment counts (e.g. ``"2 image(s)"``)."""
    parts: list[str] = []
    for prefix, label in _ATTACHMENT_KINDS:
        count = sum(1 for a in attachments if a.mime_type.startswith(prefix))
        if count:
            parts.append(f"{count} {label}")
    return parts


class SorcarAgent(RelentlessAgent):
    """Agent with both coding tools and browser automation for web + code tasks."""

    uses_worktree: bool = False

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.web_use_tool: WebUseTool | None = None
        self.docker_manager: Any = None
        self._use_web_tools: bool = True
        self._is_parallel: bool = True
        self._append_basic_tools: bool = True
        # Sub-agent threads this agent stopped waiting for; see
        # :class:`_AbandonedSubagent` and :meth:`reclaim_abandoned_subagents`.
        # Touched by the agent thread and by server threads (worktree
        # cleanup), hence the lock.
        self._abandoned_subagents: list[_AbandonedSubagent] = []
        self._abandoned_lock: threading.Lock = threading.Lock()

    def reclaim_abandoned_subagents(self, timeout: float = 0.0) -> bool:
        """Bank abandoned sub-agents' spend and report whether any live on.

        A child that ignored its stop event is abandoned, not killed
        (see :func:`_await_subagents`), so two things outlive the
        fan-out: the thread — which is still writing into this agent's
        ``work_dir`` — and the budget it keeps spending after the
        parent froze its totals.  This waits up to *timeout* for those
        threads, folds whatever they have spent since they were last
        counted into this agent's totals, and forgets the ones that
        finished.

        Callers use the return value to decide whether it is safe to
        delete the shared working directory.

        Args:
            timeout: Seconds to wait for the abandoned threads.  ``0``
                polls without waiting.

        Returns:
            True when no abandoned sub-agent is still running.
        """
        with self._abandoned_lock:
            pending = list(self._abandoned_subagents)
        if not pending:
            return True
        if timeout > 0:
            # Waiting happens OUTSIDE the lock so a concurrent
            # zero-timeout poll (e.g. server-side worktree cleanup)
            # is never blocked for this caller's full timeout.
            wait([item.future for item in pending], timeout=timeout)
        # The whole bank-and-forget sequence holds the lock:
        # ``item.unbanked_usage()`` (read-modify-write of
        # ``item.counted``) and ``_attribute_sub_usage`` (read-modify-
        # write of this agent's ``budget_used`` totals) would otherwise
        # race a concurrent reclaimer — the agent thread and server-
        # side worktree cleanup call this concurrently — double-
        # counting spend or losing a totals update.  Neither callee
        # acquires ``_abandoned_lock``, so this cannot deadlock.
        with self._abandoned_lock:
            still_running: list[_AbandonedSubagent] = []
            for item in pending:
                budget, tokens, steps = item.unbanked_usage()
                if budget or tokens or steps:
                    _attribute_sub_usage(self, budget, tokens, steps)
                if not item.future.done():
                    still_running.append(item)
            live = {id(item) for item in still_running}
            self._abandoned_subagents = [
                item
                for item in self._abandoned_subagents
                if item not in pending or id(item) in live
            ]
        return not still_running

    def _subagent_budget_share(self, num_tasks: int) -> float | None:
        """Return the ``max_budget`` each parallel sub-agent may spend.

        Splits this task's REMAINING budget — ``max_budget`` minus the
        spend already attributed to this agent minus the live executor
        session's own spend — evenly across *num_tasks* sub-agents PLUS
        one reserved parent share.  Reserving that share leaves the main
        agent enough budget to process the results and finish; importantly,
        even a one-item fan-out cannot consume the parent's entire remainder.

        Args:
            num_tasks: Number of parallel sub-agent tasks about to spawn.

        Returns:
            The per-sub-agent budget share in USD, or ``None`` when this
            agent has no budget context yet (``run``/``_reset`` never
            ran, e.g. direct ``_run_tasks_parallel`` invocations) — the
            sub-agents then fall back to their default budget.

        Raises:
            KISSError: If the task has no remaining budget.
        """
        raw_max_budget = getattr(self, "max_budget", None)
        if raw_max_budget is None:
            return None
        max_budget = float(raw_max_budget)
        executor = getattr(self, "_current_executor", None)
        live = executor.budget_used if executor is not None else 0.0
        remaining = max_budget - float(getattr(self, "budget_used", 0.0) or 0.0) - live
        if remaining <= 0:
            raise BudgetExceededError(
                f"Agent {self.name} has no remaining budget for parallel "
                f"sub-agents (${max_budget - remaining:.4f} / "
                f"${max_budget:.2f})."
            )
        return remaining if num_tasks <= 0 else remaining / (num_tasks + 1)

    def _subagent_parent_tab_id(self) -> str:
        """Return the frontend tab id sub-agents should call their parent.

        Normally this agent's own ``_tab_id``.  When this agent is
        itself a sub-agent (nested ``run_parallel``), its ``_tab_id``
        is the synthetic id its parent invented, which no webview may
        have opened yet; the printer's viewer registry knows which tab
        is really watching this task, so that one wins.

        Returns:
            The tab id, or ``""`` when running headless.
        """
        tab_id = str(getattr(self, "_tab_id", "") or "")
        if getattr(self, "_subagent_info", None) is None or self.printer is None:
            return tab_id
        fanout = getattr(self.printer, "_fanout_targets", None)
        own_task_id = _persisted_task_id(self)
        if fanout is None or not own_task_id:
            return tab_id
        viewer_ids = fanout(own_task_id)
        return sorted(viewer_ids)[0] if viewer_ids else tab_id

    def _run_tasks_parallel(
        self,
        tasks: list[str],
        max_workers: int | None = None,
    ) -> list[str]:
        """Execute multiple independent tasks concurrently using parallel agents.

        Each task gets its own ``ChatSorcarAgent`` instance, resuming
        this agent's chat session and nested under this agent's
        persisted task, via the single fan-out engine
        :func:`run_tasks_parallel`.

        This method owns no frontend concepts (tabs, ``new_tab``
        broadcasts, ...): it only reads this agent's context and hands
        it to the engine.  Sub-agent-specific frontend behaviour is
        owned by the sub-agent itself — see :meth:`ChatSorcarAgent.run`,
        which self-broadcasts a ``new_tab`` message whenever it detects
        ``self._subagent_info`` is set.

        Args:
            tasks: List of self-contained task description strings
                (the ``run_parallel`` tool closure has already coerced
                the LLM's raw argument via :func:`_coerce_tasks`, and
                the :func:`run_tasks_parallel` engine re-coerces
                defensively).
            max_workers: Maximum concurrent threads (``None`` = auto).

        Returns:
            List of YAML result strings in the same order as *tasks*.
        """
        # Bank whatever an earlier fan-out's abandoned children spent
        # after this agent stopped waiting for them, before the budget
        # share below is computed from those totals.
        self.reclaim_abandoned_subagents()
        totals: dict[str, float] = {}
        monitor = _LiveUsageMonitor(self, self.printer)
        monitor.start()
        try:
            results = run_tasks_parallel(
                tasks,
                max_workers=max_workers,
                model_name=self.model_name,
                work_dir=self.work_dir,
                printer=self.printer,
                totals_out=totals,
                usage_monitor=monitor,
                max_budget=self._subagent_budget_share(len(tasks)),
                model_config=getattr(self, "model_config", None),
                parent_agent=self,
                chat_id=str(getattr(self, "_chat_id", "") or ""),
                parent_tab_id=self._subagent_parent_tab_id(),
                base_system_prompt=str(
                    getattr(self, "_base_system_prompt", "") or ""
                ),
                system_prompt_suffix=str(
                    getattr(self, "_system_prompt_suffix", "") or ""
                ),
                web_tools=self._use_web_tools,
            )
        finally:
            # stop() joins the monitor BEFORE the offsets bump below so a
            # late emission can never double-count.  The attribution runs
            # in this finally so an interrupt (user stop) that unwinds
            # the fan-out cannot make the sub-agents' spend disappear
            # from the parent task's budget/token/step totals.
            monitor.stop()
            _attribute_sub_usage(
                self,
                float(totals.get("budget_used", 0.0)),
                int(totals.get("total_tokens_used", 0)),
                int(totals.get("total_steps", 0)),
            )
        return results

    def _docker_bash(
        self,
        command: str,
        description: str,
        timeout_seconds: int = 30,
        max_output_chars: int = 50000,
    ) -> str:
        """Run *command* in the task's container, honouring both limits.

        Widens ``RelentlessAgent._docker_bash``, which forwards only
        the command and its description.  ``DockerManager.Bash``
        honours a timeout and truncates its output, but a two-argument
        forwarder pins both to the manager's defaults, so the model
        could neither raise the 30-second cap for a slow build nor ask
        for more than the default slice of a large output — limits the
        non-docker ``UsefulTools.Bash`` has always exposed.

        Args:
            command: The bash command to run.
            description: A brief description of the command.
            timeout_seconds: Timeout in seconds for the command.
            max_output_chars: Maximum characters in output before truncation.

        Returns:
            The output of the command.

        Raises:
            KISSError: If no docker manager is attached to this agent.
        """
        if self.docker_manager is None:
            raise KISSError("Docker manager not initialized")
        return str(
            self.docker_manager.Bash(
                command, description, timeout_seconds, max_output_chars,
            )
        )

    def _get_tools(self) -> list:
        """Build tool list, using DockerTools when docker_manager is active.

        Must be called after docker_manager is set up (i.e., from perform_task,
        not from run() before super().run()).
        """
        def _stream(text: str) -> None:
            if self.printer:
                self.printer.print(text, type="bash_stream")

        def ask_user_question(question: str) -> str:
            """Ask the user a question and wait for their typed response.

            Use when the agent needs clarification, confirmation, or additional
            information from the user in the middle of a task. The user sees
            the question in the chat window, types their answer, and clicks
            "I'm Done". The agent blocks until the answer is provided.

            Args:
                question: The question to display to the user.

            Returns:
                The user's typed response text.
            """
            ask_callback = getattr(self, "_ask_user_question_callback", None)
            if ask_callback:
                return str(ask_callback(question))
            return "(ask_user_question not available in this environment)"

        def talk(language: str, text: str, emotion: str = "") -> str:
            """Speak text aloud to the user through their device speakers.

            Broadcasts a text-to-speech request to every client tab open
            for the running task (across all connected devices); each
            client plays the text on its default speaker system using
            the given language's voice.  Use this to respond aloud when
            the user speaks to the running task.

            Write *text* the way a warm, engaged human actually talks —
            NEVER like a robot reading a report.  Use contractions
            ("I'm", "let's", "that's"), short varied sentences, and
            natural interjections ("Alright,", "Oh nice —", "Hmm,",
            "Okay, so...").  Punctuation drives the delivery: questions
            rise, exclamations add energy, an ellipsis trails off, and
            sentence breaks become natural breathing pauses.  Pick an
            *emotion* that matches the vibe of the moment instead of
            leaving it flat.

            Args:
                language: BCP-47 language tag for the speech voice
                    (e.g. "en-US", "es", "fr-FR").
                text: The text to synthesize and play aloud, written in
                    a natural, conversational, emotionally expressive
                    style (contractions, interjections, punctuation).
                emotion: Optional vibe for the delivery; the client
                    shapes speech rate and pitch to match.  One of
                    "cheerful", "excited", "playful", "curious", "warm",
                    "proud", "calm", "empathetic", "reassuring",
                    "apologetic", "serious", or "sad".  Empty means
                    neutral (the client may still infer a vibe from the
                    punctuation and wording of *text*).

            Returns:
                A confirmation message, or a note that audio playback is
                unavailable in this environment.
            """
            broadcast = getattr(self.printer, "broadcast", None)
            if not callable(broadcast):
                return "(talk not available in this environment)"
            payload: dict[str, Any] = {
                "type": "talk",
                "language": language,
                "text": text,
                "emotion": emotion,
                "talkId": uuid.uuid4().hex,
            }
            tts_usage: dict[str, Any] = {}
            try:
                from kiss.core.speech_synthesis import synthesize_talk_audio

                synthesized = synthesize_talk_audio(
                    text, language, emotion, usage_out=tts_usage,
                )
            except Exception:
                synthesized = None
            _attribute_tts_usage(self, tts_usage)
            if synthesized:
                payload["audioB64"], payload["audioMime"] = synthesized
            broadcast(payload)
            return f"Spoke to the user in language {language!r}."

        if self.docker_manager:
            from kiss.agents.sorcar.docker_tools import DockerTools

            docker_tools = DockerTools(self._docker_bash)

            def Bash(  # noqa: N802
                command: str,
                description: str,
                timeout_seconds: int = 30,
                max_output_chars: int = 50000,
            ) -> str:
                """Runs a bash command in the task's Docker container and returns its output.

                Args:
                    command: The bash command to run.
                    description: A brief description of the command.
                    timeout_seconds: Timeout in seconds for the command.
                    max_output_chars: Maximum characters in output before truncation.

                Returns:
                    The output of the command.
                """
                return self._docker_bash(
                    command, description, timeout_seconds, max_output_chars,
                )

            tools: list = [
                Bash, docker_tools.Read, docker_tools.Edit, docker_tools.Write,
            ]
        else:
            useful_tools = UsefulTools(
                stream_callback=_stream,
                stop_event=getattr(self, "_stop_event", None),
                work_dir=self.work_dir,
            )
            tools = [
                useful_tools.Bash,
                useful_tools.Read, useful_tools.Edit, useful_tools.Write,
            ]
        if self._use_web_tools and self.web_use_tool is None:
            # Sub-agents run concurrently, so they get a throwaway profile
            # instead of contending for the shared profile's Chromium lock.
            self.web_use_tool = WebUseTool(
                work_dir=self.work_dir,
                ephemeral=getattr(self, "_subagent_info", None) is not None,
            )
            tools.extend(self.web_use_tool.get_tools())
        def run_parallel(tasks: str, max_workers: str = "") -> str:
            """Run multiple independent tasks concurrently using parallel agents.

            Spawns a separate ChatSorcarAgent for each task string and executes
            them in parallel threads.

            **When to call run_parallel:**
            - Multi-source / multi-topic research ("research these 5
              companies", "summarize each of these N PDFs").
            - Codebase exploration across unrelated modules ("look at the
              frontend, backend, db layer, and auth in parallel").
            - Multi-perspective review of one artifact (correctness
              reviewer + security reviewer + style reviewer +
              architecture reviewer, each looking at the same diff with
              a different lens).
            - Generating N alternative candidates for the same problem
              so the orchestrator can pick the best.
            - Independent test suites or validations on disjoint targets.
            - Bulk file generation when each file is independent and the
              API contract between them is already pinned down in a
              spec.


            Args:
                tasks: A JSON-encoded list of task description strings.
                    Example::

                        '["Read src/foo.py and summarize its purpose", '
                        '"Read src/bar.py and summarize its purpose", '
                        '"Find the current weather in San Francisco"]'
                max_workers: Maximum number of concurrent threads, as a
                    string containing an integer (e.g. ``"4"``).  An empty
                    string (default) lets Python choose automatically.
                    Set to a lower number to limit concurrency.

            Returns:
                A YAML-formatted string containing a list of result
                objects, one per task, in the same order as the input.
                Each result object has ``success`` and ``summary`` keys.
            """
            task_list = _coerce_tasks(tasks)
            workers: int | None = int(max_workers) if max_workers else None
            results = self._run_tasks_parallel(task_list, max_workers=workers)
            result_str: str = yaml.dump(results, sort_keys=False)
            return result_str

        def number_of_cores() -> int:
            """Return the number of CPU cores available on the current machine.

            Useful for choosing a reasonable ``max_workers`` value when
            calling :func:`run_parallel`.

            Returns:
                The number of CPU cores available to the process,
                falling back to ``1`` when it cannot be determined.
            """
            return os.process_cpu_count() or 1

        def set_model(model_name: str) -> str:
            """Change only this running agent's LLM model dynamically.

            The tabs watching this task show the new model in their
            picker for as long as the task runs, then revert to the
            user's own choice.  The switch is display-only: it never
            persists ``last_model``, so the user's picker preference
            survives untouched — only an explicit picker selection
            updates that.

            Args:
                model_name: New LLM model name (for example
                    ``"gpt-5.5"``, ``"claude-sonnet-4-8"``,
                    ``"gemini-3.5-flash"``).

            Returns:
                A human-readable confirmation string describing the
                change (or a "no change" message when the requested
                model is already active).
            """
            from kiss.core.models.model_info import (
                model_runs_task_to_completion,
            )

            if getattr(self, "docker_image", None) and model_runs_task_to_completion(
                model_name
            ):
                return (
                    f"Cannot switch to {model_name}: it is a CLI agent "
                    "that runs natively on the host, which would bypass "
                    "this task's docker_image isolation. Pick an API model."
                )
            target = getattr(self, "_current_executor", None) or self
            old_model = getattr(target, "model", None)
            if old_model is None:
                self.model_name = model_name
                self._show_model_in_picker(model_name)
                return (
                    f"Model deferred-changed to {model_name} "
                    "(no live model yet)."
                )
            if old_model.model_name == model_name:
                return f"Model is already {model_name}; no change."

            new_config: dict[str, Any] = dict(old_model.model_config or {})
            old_info = MODEL_INFO.get(old_model.model_name)
            if (
                old_info is not None
                and new_config.get("reasoning_effort") == old_info.thinking
            ):
                new_config.pop("reasoning_effort", None)
            _sanitize_model_config_for_switch(
                new_config, old_model.model_name, model_name,
            )
            old_base_url = getattr(old_model, "base_url", None)
            old_api_key = getattr(old_model, "api_key", None)
            if "use_responses_api" in new_config:
                # `use_responses_api` forces /responses delegation, which
                # only some OpenAI-compatible vendors support.  Keep it
                # only when the switch stays on the SAME vendor endpoint.
                from kiss.core.models.model_info import (
                    openai_compatible_provider_for_base_url,
                )

                old_vendor = openai_compatible_provider_for_base_url(
                    old_base_url or ""
                )
                new_vendor = _match_openai_compatible_provider(
                    _strip_provider_prefix(model_name)
                )
                if old_vendor is None or old_vendor is not new_vendor:
                    new_config.pop("use_responses_api", None)
            if old_base_url and "base_url" not in new_config:
                normalized = old_base_url.rstrip("/")
                if normalized in _FACTORY_DEFAULT_BASE_URLS:
                    # Standard provider endpoint: preserve routing (and
                    # crucially the possibly task-specific api_key) only
                    # when the target model routes to the SAME provider
                    # default — otherwise the factory would silently
                    # replace a per-task key with the process-global one.
                    target_provider = _match_openai_compatible_provider(
                        _strip_provider_prefix(model_name)
                    )
                    preserve = (
                        target_provider is not None
                        and target_provider.base_url.rstrip("/") == normalized
                    )
                else:
                    # Custom endpoint: always carry it (and its key) over.
                    preserve = True
                if preserve:
                    new_config["base_url"] = old_base_url
                    if old_api_key is not None:
                        new_config["api_key"] = old_api_key
            new_model = _model_factory(
                model_name,
                model_config=new_config or None,
                token_callback=old_model.token_callback,
                thinking_callback=old_model.thinking_callback,
            )
            old_family = _model_family(old_model.model_name)
            if (
                old_api_key
                and old_family in ("gemini", "anthropic")
                and old_family == _model_family(model_name)
                and hasattr(new_model, "api_key")
            ):
                # Native same-provider switch (Gemini→Gemini,
                # Claude→Claude): the factory always injects the
                # process-global key, which would silently replace a
                # task-specific one.  Both native adapters build their
                # SDK client from self.api_key inside initialize(), so
                # overriding BEFORE initialize() routes requests with
                # the task's credential.  setattr: the attribute lives
                # on the concrete adapters, not the Model base class
                # (the hasattr guard above ensures it exists).
                setattr(new_model, "api_key", old_api_key)  # noqa: B010
            new_model.initialize("")
            new_model.conversation = old_model.conversation
            new_model.usage_info_for_messages = old_model.usage_info_for_messages
            old_sigs = getattr(old_model, "_thought_signatures", None)
            new_sigs = getattr(new_model, "_thought_signatures", None)
            if isinstance(old_sigs, dict) and isinstance(new_sigs, dict):
                # Gemini-to-Gemini switch: the conversation references
                # historical tool-call ids whose thought signatures live
                # only in this side map (initialize() cleared the new
                # model's copy); without them signature-enforcing Gemini
                # models reject the next request.
                new_sigs.update(old_sigs)

            previous_name = old_model.model_name
            target.model = new_model  # type: ignore[attr-defined, union-attr]
            target.model_name = model_name
            self.model_name = model_name
            if getattr(target, "function_map", None):
                target._cached_tools_schema = new_model._build_openai_tools_schema(  # type: ignore[attr-defined, union-attr]
                    target.function_map,
                )
            self._show_model_in_picker(model_name)
            return f"Model changed from {previous_name} to {model_name}."

        skill_tool = make_skill_tool(self.work_dir or ".")
        if skill_tool is not None:
            tools.append(skill_tool)
        try:
            from kiss.agents.sorcar.mcp_servers import make_mcp_tools

            tools.extend(make_mcp_tools(self.work_dir or "."))
        except Exception:
            logger.warning("MCP tool setup failed", exc_info=True)
        from kiss.agents.sorcar.agent_dispatch import make_run_agent_tool

        # Scheduled automations (cron) are not a built-in tool: the
        # agent dispatches them via run_agent("cron", ...), which runs
        # kiss.agents.sorcar.cron_agent as an agent script.  Passing
        # self makes each dispatched sub-task's cost/tokens/steps fold
        # into THIS task's accounting, so the end-of-task cost shown
        # to the user includes run_agent sub-tasks (like run_parallel).
        tools.append(make_run_agent_tool(self.work_dir or "", self))
        tools.append(ask_user_question)
        tools.append(talk)
        tools.append(set_model)
        if self._is_parallel:
            tools.append(run_parallel)
            tools.append(number_of_cores)
        return tools

    def _show_model_in_picker(self, model_name: str) -> None:
        """Display *model_name* in the picker of every tab watching this task.

        The override lasts only while the task runs — the daemon puts
        the user's own pick back when the task ends — so this is purely
        a live view of what the agent is running right now.  Purely
        cosmetic, hence best-effort: printers without the capability
        (plain console runs) and transport errors are ignored rather
        than allowed to fail the ``set_model`` tool call.

        The printer resolves the watching tabs itself via its
        transient all-watching-tabs primitive
        (``JsonPrinter._transient_targets``, shared with
        ``broadcast_transient``); the agent only supplies its ids —
        ``_last_task_id`` keeps the fan-out working when the call is
        made off the run thread or near teardown, after the printer's
        thread-local task id has been cleared.  The model pick goes
        through ``broadcast_agent_model_pick`` rather than the plain
        primitive because the printer must also remember each target
        for ``restore_model_pick``.

        Args:
            model_name: The model the agent just switched to.
        """
        show = getattr(self.printer, "broadcast_agent_model_pick", None)
        if not callable(show):
            return
        try:
            show(
                model_name,
                getattr(self, "_tab_id", "") or "",
                _persisted_task_id(self) or None,
            )
        except Exception:
            logger.warning("model picker update failed", exc_info=True)

    def perform_task(
        self,
        tools: list,
        attachments: list | None = None,
    ) -> str:
        """Execute the task, building docker-aware tools after docker_manager is set.

        Args:
            tools: Extra tools passed by the caller (from run(tools=...)).
            attachments: Optional file attachments for the initial prompt.

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        # ``run(append_basic_tools=False)`` strips the agent down to
        # ``finish`` (added by ``RelentlessAgent.perform_task``) plus
        # the caller's *tools*: the built-in toolset is never built, so
        # no web profile, MCP server, or run_agent/run_parallel wiring
        # is set up either.
        if self._append_basic_tools:
            all_tools = self._get_tools() + tools
        else:
            all_tools = list(tools)
        # Always install the steering hooks: they are self-guarding
        # no-ops when no follow-up channel exists (a printer without
        # the duck-typed ``drain_pending_user_messages`` bridge), and
        # the server UI's printer bridge must be drained when present.
        self.pre_step_hook = self._drain_pending_user_messages
        self.tool_call_guard = self._block_finish_when_user_message_pending
        return super().perform_task(all_tools, attachments=attachments)

    def _reset(
        self,
        model_name: str | None,
        max_sub_sessions: int | None,
        max_steps: int | None,
        max_budget: float | None,
        work_dir: str | None,
        docker_image: str | None,
        printer: Printer | None = None,
        verbose: bool | None = None,
    ) -> None:
        resolved_model = self._resolve_model_name(model_name)
        self._launch_model_name = resolved_model
        super()._reset(
            model_name=resolved_model,
            max_sub_sessions=max_sub_sessions,
            max_steps=max_steps,
            max_budget=max_budget,
            work_dir=work_dir or ".",
            docker_image=docker_image,
            printer=printer,
            verbose=verbose if verbose is not None else False,
        )

    @staticmethod
    def _resolve_model_name(model_name: str | None) -> str:
        """The model a run asked to use *model_name* actually runs with.

        The same fallback chain ``_reset`` applies: the caller's model,
        else the user's last-selected model, else the configured
        default.  Exposed so callers that record a run's settings
        BEFORE ``_reset`` executes (``ChatSorcarAgent.run``'s early
        history row and ``task_settings`` event) persist the resolved
        value instead of a blank.

        Args:
            model_name: The caller-supplied model name, possibly None.

        Returns:
            The resolved model name.
        """
        return model_name or _load_last_model() or get_default_model()

    def _system_prompt_task_settings(self) -> dict[str, str]:
        """Extend the base settings with this agent's parallel mode.

        Returns:
            The base label → value pairs plus "Parallel mode".
        """
        settings = super()._system_prompt_task_settings()
        settings["Parallel mode"] = (
            "parallel" if self._is_parallel else "sequential"
        )
        return settings

    def run(  # type: ignore[override]
        self,
        model_name: str | None = None,
        prompt_template: str = "",
        arguments: dict[str, str] | None = None,
        system_prompt: str | None = None,
        tools: list[Callable[..., Any]] | None = None,
        max_steps: int | None = None,
        max_budget: float | None = None,
        model_config: dict[str, Any] | None = None,
        work_dir: str | None = None,
        printer: Printer | None = None,
        max_sub_sessions: int | None = None,
        docker_image: str | None = None,
        web_tools: bool = True,
        is_parallel: bool = True,
        verbose: bool | None = None,
        current_editor_file: str | None = None,
        attachments: list[Attachment] | None = None,
        ask_user_question_callback: Callable[[str], str] | None = None,
        base_system_prompt: str = "",
        append_basic_tools: bool = True,
        llm_call_hook: (
            Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None
        ) = None,
        tool_call_hook: Callable[[str, dict[str, Any]], str] | None = None,
    ) -> str:
        """Run the assistant agent with coding tools and browser automation.

        Args:
            model_name: LLM model to use. Defaults to config value.
            prompt_template: Task prompt template with format placeholders.
            arguments: Dictionary of values to fill prompt_template placeholders.
            system_prompt: system prompt to be appended to the actual system
                prompt.  Also forwarded to every sub-agent spawned via
                ``run_parallel``, appended to each sub-agent's own base
                system prompt — like *base_system_prompt*, so the extra
                instructions constrain the whole task tree.
            tools: List of tools to be added in addition to bash and web tools.
            max_steps: Maximum steps per sub-session. Defaults to 10000.
            max_budget: Maximum budget in USD. Defaults to config value.
            work_dir: Working directory for the agent. Defaults to artifact_dir/kiss_workdir.
            printer: Printer instance for output display.
            max_sub_sessions: Maximum continuation sub-sessions. Defaults to config value.
            docker_image: Docker image name to run tools inside a container.
            web_tools: Whether to include browser/web tools. Defaults to True.
                Set to False for terminal-only environments.
            is_parallel: Whether to include the run_parallel tool. Defaults to True.
                When True, the agent can spawn parallel sub-agents for independent tasks.
            verbose: Whether to print output to console. Defaults to config verbose setting.
            current_editor_file: Path to the currently active editor file, appended to prompt.
            attachments: Optional file attachments (images, PDFs) for the initial prompt.
            ask_user_question_callback: Optional callback used by the ask_user_question
                tool to collect a text response from the user.
            base_system_prompt: Custom base system prompt.  When non-blank it
                REPLACES the default ``SYSTEM.md`` system prompt
                (:data:`kiss.core.base.SYSTEM_PROMPT`) for this agent and for
                every sub-agent it spawns via ``run_parallel``.  The
                *system_prompt* suffix, the active-editor-file line, and the
                per-run operational instructions (work dir, PID,
                ``~/.kiss/SORCAR.md``) are still appended.  Blank (default)
                keeps the default system prompt.  Appended after the
                historical positional arguments so every one of them
                keeps its position.
            append_basic_tools: Whether :meth:`perform_task` prepends the
                built-in basic toolset (:meth:`_get_tools`: Bash, Read,
                Edit, Write, browser tools, run_agent,
                ask_user_question, talk, set_model, run_parallel, ...)
                to the caller's *tools*.  Defaults to True.  When False
                the agent runs with ONLY the ``finish`` tool (added by
                ``RelentlessAgent.perform_task``) and the caller's
                *tools* — *web_tools* and *is_parallel* then have no
                effect, since the tools they toggle are never built.
            llm_call_hook: Optional hook forwarded to the underlying
                :meth:`kiss.core.kiss_agent.KISSAgent.run` of every
                sub-session this agent runs (see that docstring): called
                before every LLM call with the new messages about to be
                sent, and its return value replaces them.  Applies to
                this agent only, not to ``run_parallel`` sub-agents.
                Defaults to None (no hook).
            tool_call_hook: Optional hook forwarded to the underlying
                :meth:`kiss.core.kiss_agent.KISSAgent.run` of every
                sub-session this agent runs (see that docstring): called
                before every tool call with the tool's name and
                arguments; any verdict other than ``"OK"`` suppresses
                the call and is returned to the model as the tool's
                result.  Applies to this agent only, not to
                ``run_parallel`` sub-agents.  Defaults to None (no
                hook).

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        self._ask_user_question_callback = ask_user_question_callback
        self._use_web_tools = web_tools
        self._is_parallel = is_parallel
        self._append_basic_tools = append_basic_tools
        # Stored on self (not just a local) so the ``run_parallel``
        # fan-out — which executes DURING ``super().run`` below — can
        # forward the same base system prompt to every sub-agent.
        self._base_system_prompt = (
            base_system_prompt if base_system_prompt.strip() else ""
        )
        # Stored on self for the same reason: the fan-out forwards the
        # append-only *system_prompt* suffix to every sub-agent, so a
        # run's extra system instructions constrain its whole task
        # tree, exactly like a *base_system_prompt* replacement.
        self._system_prompt_suffix = system_prompt if system_prompt else ""
        self.web_use_tool = None
        tl = getattr(printer, "_thread_local", None) if printer else None
        self._stop_event = getattr(tl, "stop_event", None) if tl else None
        try:
            system_instructions = (
                (self._base_system_prompt or SYSTEM_PROMPT)
                + (system_prompt if system_prompt else "")
            )
            prompt = prompt_template
            if attachments:
                parts = _attachment_parts(attachments)
                if parts:
                    prompt += (
                        f"\n\n# Important\n - User attached {', '.join(parts)}. "
                        f"The files are included in this message as inline content "
                        f"that you can see directly. "
                        f"Do NOT launch a browser, call screenshot(), go_to_url(), "
                        f"or any other browser tool to view these attachments — "
                        f"you already have them."
                    )
            if current_editor_file:
                system_instructions += (
                    "\n\n- The path of the file open in the editor is "
                    f"{current_editor_file}"
                )
            return super().run(
                model_name=model_name,
                system_prompt=system_instructions,
                prompt_template=prompt,
                arguments=arguments,
                max_steps=max_steps,
                max_budget=max_budget,
                model_config=model_config,
                work_dir=work_dir,
                printer=printer,
                max_sub_sessions=max_sub_sessions,
                docker_image=docker_image,
                verbose=verbose,
                tools=tools or [],
                attachments=attachments,
                llm_call_hook=llm_call_hook,
                tool_call_hook=tool_call_hook,
            )
        finally:
            if self.web_use_tool:
                self.web_use_tool.close()
            self.web_use_tool = None
            self._ask_user_question_callback = None
            self.pre_step_hook = None
            self.tool_call_guard = None

    def _drain_pending_user_messages(self, model: Any) -> None:
        """Append any queued follow-up prompts to *model*'s conversation.

        Called once at the top of every model step (wired in via
        :attr:`kiss.core.kiss_agent.KISSAgent.pre_step_hook`).  Drains
        the run's queued follow-up prompts through the printer's
        duck-typed ``drain_pending_user_messages`` bridge (the server
        keeps them on the task's registered agent state, keyed by the
        calling thread's task id) and pushes each entry into *model*'s
        conversation as a ``user`` role message.  Each entry is
        wrapped as ``User says: <message>. Take the message into
        account and finish your task.`` so the model treats it as a
        mid-task steering instruction rather than a bare trajectory
        line.  The bridge empties the queue on every drain so the same
        queued message is never injected twice, and emits a durable
        ``recordOnly`` echo for any message whose live echo could not
        be attributed to a task id at queueing time.

        Args:
            model: The live model whose conversation receives the
                queued user messages.
        """
        drain = getattr(
            getattr(self, "printer", None),
            "drain_pending_user_messages",
            None,
        )
        queued: list[str] = drain() if drain is not None else []
        for msg in queued:
            model.add_message_to_conversation(
                "user",
                f"User says: {msg}. "
                "Take the message into account and finish your task.",
            )

    def _block_finish_when_user_message_pending(
        self, name: str, args: dict[str, Any],
    ) -> str | None:
        """Reject ``finish`` while a queued user follow-up is undrained.

        The server accepts ``appendUserMessage`` while a model call is
        in flight, but queued messages are drained only by the
        pre-step hook at the TOP of a step.  Without this guard, a
        prompt queued after the last drain would be silently discarded
        when the in-flight response calls ``finish`` — the user sees
        their follow-up echoed in the UI even though the agent never
        saw it.  Blocking the ``finish`` forces one more step, whose
        pre-step drain injects the queued message.

        Args:
            name: The tool name the model is calling.
            args: The tool call arguments (unused).

        Returns:
            ``None`` to allow the call, or a rejection message when
            ``finish`` was attempted with steering input still queued.
        """
        del args
        if name != "finish":
            return None
        has_pending = getattr(
            getattr(self, "printer", None),
            "has_pending_user_messages",
            None,
        )
        if has_pending is None or not has_pending():
            return None
        return (
            "Error: finish rejected — the user sent a new message while "
            "you were working. It will be appended to the conversation "
            "at the start of your next step; take it into account "
            "before finishing."
        )


def _coerce_tasks(tasks: Any) -> list[str]:
    """Normalize the ``tasks`` argument to a ``list[str]``.

    LLM tool calls sometimes pass ``tasks`` in two malformed shapes that
    we recover from here:

    1. A JSON-encoded list string such as ``'["task A", "task B"]'``.
       Without recovery, the entire JSON string would be treated as one
       task and dispatched to a single sub-agent.  We parse it back into
       a proper ``list[str]``.
    2. A bare task string such as ``"hello"``.  Without this guard,
       ``enumerate(tasks)`` would iterate the string character-by-
       character and create one sub-agent (and one ``openSubagentTab``
       event) per character.  We wrap it into ``["hello"]``.

    Args:
        tasks: Either a ``list[str]``, a JSON-encoded ``list[str]`` string,
            or a single task ``str``.

    Returns:
        A ``list[str]``.  JSON-encoded list strings are parsed; other
        ``str`` inputs are wrapped in a one-element list.

    Raises:
        TypeError: If *tasks* is neither a ``str`` nor a ``list[str]``.
    """
    if isinstance(tasks, str):
        stripped = tasks.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, list):
                return [t if isinstance(t, str) else str(t) for t in parsed]
        return [tasks]
    if isinstance(tasks, list) and all(isinstance(t, str) for t in tasks):
        return tasks
    raise TypeError(
        f"tasks must be list[str], got {type(tasks).__name__}: {tasks!r}"
    )


def run_tasks_parallel(
    tasks: list[str],
    max_workers: int | None = None,
    model_name: str | None = None,
    work_dir: str | None = None,
    printer: Printer | None = None,
    totals_out: dict[str, float] | None = None,
    max_budget: float | None = None,
    model_config: dict[str, Any] | None = None,
    usage_monitor: _LiveUsageMonitor | None = None,
    parent_agent: Any = None,
    chat_id: str = "",
    parent_tab_id: str = "",
    base_system_prompt: str = "",
    system_prompt_suffix: str = "",
    web_tools: bool = True,
) -> list[str]:
    """Execute multiple SorcarAgent tasks concurrently using threads.

    Each task gets its own ``ChatSorcarAgent`` instance and runs in a
    separate thread via :class:`~concurrent.futures.ThreadPoolExecutor`.
    This is ideal for I/O-bound workloads (LLM API calls, network
    requests) where the GIL is released during I/O waits.

    This is the ONE fan-out engine in the codebase.  It used to have a
    near-identical twin in ``ChatSorcarAgent._run_tasks_parallel``, and
    four correctness fixes (per-sub-agent stop event, stopped-child
    recovery, real ``parent_task_id``, chat/tab propagation) had landed
    only in that twin, so every plain :class:`SorcarAgent` subclass —
    which is what the third-party channel agents used to be before they
    became daemon-launched carriers — silently ran the unfixed copy.
    Keep it single.

    The engine still owns no frontend concepts: it marks each spawned
    agent as a sub-agent (via ``_subagent_info``) and the sub-agent
    itself broadcasts its own ``new_tab`` inside ``run()``.

    Args:
        tasks: List of task description strings.  Each string is passed as
            the ``prompt_template`` argument to :meth:`SorcarAgent.run`.
            Example::

                [
                    "Summarize file A",
                    "Summarize file B",
                ]
        max_workers: Maximum number of threads.  ``None`` lets
            :class:`~concurrent.futures.ThreadPoolExecutor` pick a default
            (typically ``min(32, cpu_count + 4)``).
        model_name: LLM model name for all parallel agents.  ``None`` uses the
            default from persistence (same as :meth:`SorcarAgent.run`).
        work_dir: Working directory for all parallel agents.  ``None`` uses
            the default (``artifact_dir/kiss_workdir``).
        printer: Optional printer from the parent agent.  Forwarded
            verbatim to each sub-agent's ``run`` so live events
            continue to flow through the same channel.  The executor
            itself does not call any printer methods.
        totals_out: Optional dict that receives the aggregated usage of
            all sub-agents.  When provided, the summed spend across
            every spawned agent is written into it under the keys
            ``"budget_used"``, ``"total_tokens_used"`` and
            ``"total_steps"`` so the caller can attribute sub-agent
            usage back to the parent task (see
            :func:`_attribute_sub_usage`).
        max_budget: Per-sub-agent budget cap in USD, forwarded to each
            sub-agent's ``run``.  Callers spawning sub-agents on behalf
            of a parent task pass each child one share of the parent's
            remaining budget and reserve one equal share for the parent
            (see :meth:`SorcarAgent._subagent_budget_share`), so even a
            one-child fan-out cannot spend the parent's whole remainder.
            ``None`` uses the sub-agent's default (config value).
        model_config: Model configuration (e.g. custom ``base_url`` /
            ``api_key`` routing) forwarded to each sub-agent's ``run``
            so sub-agents talk to the same provider endpoint as the
            parent.  ``None`` uses default provider routing.
        usage_monitor: Optional :class:`_LiveUsageMonitor` that each
            spawned sub-agent is registered with, so the parent task's
            cost/tokens header can stream live aggregate usage while
            the sub-agents run.  ``None`` disables live tracking.
        parent_agent: The agent that is fanning out, when there is one.
            Its persisted ``task_history`` row id is re-read as each
            worker starts and stamped on the child, so the child is
            stored as a nested sub-task rather than a bogus top-level
            history row.  ``None`` (a bare functional call) falls back
            to the printer's thread-local task id.
        chat_id: Chat session the children resume, so a sub-agent
            starts with the parent's conversation context instead of a
            brand-new empty session.  ``""`` gives each child a fresh
            chat.
        parent_tab_id: Frontend tab id of the parent, forwarded in
            ``_subagent_info`` so the child's ``new_tab`` broadcast
            tells the owning webview which tab spawned it.
        base_system_prompt: Custom base system prompt forwarded to each
            sub-agent's ``run``, so a parent running with a caller-supplied
            system prompt (see :meth:`SorcarAgent.run`) spawns children
            that use the same prompt instead of the default ``SYSTEM.md``.
            ``""`` keeps the default.
        system_prompt_suffix: Extra text appended to each sub-agent's
            base system prompt, forwarded as the ``system_prompt``
            argument of each sub-agent's ``run``.  A parent running
            with an append-only system-prompt suffix (see
            :meth:`SorcarAgent.run`'s *system_prompt*) passes it on so
            the extra instructions constrain the whole task tree,
            mirroring *base_system_prompt*.  ``""`` appends nothing.
        web_tools: Whether each sub-agent gets browser/web tools,
            forwarded to each sub-agent's ``run``.  A parent running
            without web tools (``run(web_tools=False)``) passes False
            so its children cannot re-acquire the browser it was denied.

    Returns:
        List of YAML result strings in the **same order** as *tasks*.
        Each string contains ``success`` and ``summary`` keys.  If a task
        raises an unhandled exception the corresponding entry is a YAML
        string with ``success: false`` and the traceback in ``summary``.

    Raises:
        TypeError: If *tasks* is not a list of strings.  As a convenience
            for LLM tool callers that mistakenly pass a bare string,
            ``str`` is coerced to a one-element list.
    """
    tasks = _coerce_tasks(tasks)

    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

    sub_usage: list[tuple[float, int, int]] = [(0.0, 0, 0)] * len(tasks)
    # Held for every slot write: the workers' final figures and the
    # parent's live refresh of abandoned children must not interleave
    # (see _collect_unfinished_usage).
    sub_usage_lock = threading.Lock()
    # Published as soon as each child exists so an abandoned child's
    # spend can still be read (see _collect_unfinished_usage).
    sub_agents: list[Any] = [None] * len(tasks)

    parent_tl = getattr(printer, "_thread_local", None) if printer else None
    parent_key = str(getattr(parent_tl, "task_id", "") or "") if parent_tl else ""
    parent_stop_event = getattr(parent_tl, "stop_event", None) if parent_tl else None
    persisted_parent_id = _persisted_task_id(parent_agent)
    # Stable for the whole fan-out: the children's synthetic tab ids
    # must not change between submission and the subagentDone
    # broadcast, even though the parent's persisted id can appear late.
    # It is a ROUTING key only — never persisted, because a synthetic
    # id names no row in ``task_history``.
    routing_key = persisted_parent_id or parent_key or uuid.uuid4().hex
    # What the children are PERSISTED under.  A parent that keeps no
    # history row of its own — every third-party channel agent is a
    # plain ``SorcarAgent`` — still must not turn each of its children
    # into a top-level history entry, so the fan-out gets one synthetic
    # parent id in the canonical row-id shape.  It names no row, which
    # is exactly right: the children are grouped together and hidden
    # from the root list, and history keeps only entries a user
    # actually started.
    fanout_parent_id = persisted_parent_id or (
        parent_key if is_task_history_id(parent_key) else uuid.uuid4().hex
    )

    def _run_single(args: tuple[int, str]) -> str:
        idx, task = args
        # A per-child event, chained to the parent's: stopping ONE
        # sub-agent must not stop the parent or its siblings, while a
        # parent stop still reaches every child (_SubagentStopEvent).
        sub_stop_event = _SubagentStopEvent(parent_stop_event)
        tl = getattr(printer, "_thread_local", None) if printer else None
        if tl is not None:
            tl.stop_event = sub_stop_event
        agent = ChatSorcarAgent(f"Parallel-{task[:40]}")
        sub_agents[idx] = agent
        if chat_id:
            agent.resume_chat_by_id(chat_id)
        sub_tab_id = f"task-{routing_key}__sub_{idx}"
        agent._tab_id = sub_tab_id
        # Re-read rather than reuse ``fanout_parent_id``: the parent may
        # persist its own row while this fan-out is being submitted, and
        # a child stamped with "" is stored as a top-level history row.
        agent._subagent_info = {
            "parent_task_id": _persisted_task_id(parent_agent)
            or fanout_parent_id,
            "parent_tab_id": parent_tab_id,
        }
        if usage_monitor is not None:
            usage_monitor.track(agent)
        try:
            result: str = agent.run(
                prompt_template=task,
                model_name=model_name,
                work_dir=work_dir,
                printer=printer,
                is_parallel=True,
                max_budget=max_budget,
                model_config=model_config,
                base_system_prompt=base_system_prompt,
                system_prompt=system_prompt_suffix or None,
                web_tools=web_tools,
            )
            return result
        except KeyboardInterrupt:
            # Only THIS child was stopped: report it as a stopped task
            # so its already-finished siblings' results are still
            # collected.  A stop of the whole parent task keeps
            # propagating, because there is nothing left to preserve.
            if parent_stop_event is not None and parent_stop_event.is_set():
                raise
            stopped: str = yaml.dump(
                {"success": False, "summary": "Sub-agent task stopped by user."},
                sort_keys=False,
            )
            return stopped
        except Exception as exc:
            return _yaml_failure(exc)
        finally:
            # _live_agent_usage (not _agent_usage): an interrupted child
            # never folds its in-flight executor session's spend into the
            # agent totals, so the folded-only read would undercount it.
            with sub_usage_lock:
                sub_usage[idx] = _live_agent_usage(agent)
            if printer is not None:
                # Notify every tab watching the sub-agent: its own
                # synthetic tab plus any other tabs subscribed to the
                # sub-agent's task stream via the printer's fan-out
                # registry.
                try:
                    viewer_ids: list[str] = []
                    fanout = getattr(printer, "_fanout_targets", None)
                    sub_task_id = _persisted_task_id(agent) or None
                    if callable(fanout) and sub_task_id is not None:
                        found = fanout(sub_task_id)
                        if isinstance(found, list):
                            viewer_ids = [v for v in found if v]
                    if sub_tab_id not in viewer_ids:
                        viewer_ids.append(sub_tab_id)
                    _broadcast_subagent_done(
                        printer, viewer_ids, model_name or "",
                    )
                except Exception:
                    logger.debug(
                        "subagentDone broadcast failed", exc_info=True,
                    )
            # Pool workers are reused and the binding is per THREAD, so
            # leaving it behind would let an unrelated sibling inherit a
            # stop meant for this task.
            if tl is not None:
                tl.stop_event = None

    pool: ThreadPoolExecutor | None = None
    futures: list[Future[str]] = []
    abandoned = False
    try:
        pool = ThreadPoolExecutor(max_workers=max_workers)
        futures = [pool.submit(_run_single, item) for item in enumerate(tasks)]
        try:
            results = _await_subagents(futures, parent_stop_event)
        except BaseException:
            abandoned = any(not f.done() for f in futures)
            raise
    finally:
        # Only a child that ignored its stop event is abandoned; every
        # other path joins (and so RECLAIMS the workers) exactly as the
        # old `with ThreadPoolExecutor(...)` block did.
        if pool is not None:
            pool.shutdown(wait=not abandoned, cancel_futures=abandoned)
        # Fill totals_out even when a worker propagates an interrupt, and
        # read the live figures of any child that never got to report its
        # own, so no completed sibling's spend is lost.
        _collect_unfinished_usage(futures, sub_agents, sub_usage, sub_usage_lock)
        # Registration and the totals summation happen under ONE hold of
        # the slot lock: an abandoned worker that unwound in the
        # meantime publishes its FINAL slot value under the same lock,
        # and a publish landing between the two used to make the parent
        # bank the final figure while the registered ``counted``
        # baseline kept the older one — the next reclaim then banked
        # the difference a second time.  Under one hold, the figure
        # summed into ``totals_out`` for a registered child is exactly
        # its ``counted`` baseline, so banked-now plus reclaimed-later
        # is the child's spend exactly once.
        with sub_usage_lock:
            if abandoned:
                # The abandoned threads keep running inside ``work_dir``
                # and keep spending: hand them to the parent so it can
                # refuse to delete that directory and can bank the rest
                # of their spend.
                _register_abandoned(parent_agent, futures, sub_agents, sub_usage)
            # Test hook (no-op in production): widens the window between
            # the registration above and the summation below so
            # concurrency tests can prove a worker's final publish
            # cannot land between them
            # (see test_audit0903_fanout_bank_register_race).
            _race_delay()
            if totals_out is not None:
                totals_out["budget_used"] = sum(u[0] for u in sub_usage)
                totals_out["total_tokens_used"] = sum(u[1] for u in sub_usage)
                totals_out["total_steps"] = sum(u[2] for u in sub_usage)
    return results


def _budget_arg(text: str) -> float:
    """Parse and validate a ``--max-budget`` command-line value.

    Plain ``float`` would accept ``nan`` and infinities, and the budget
    checks compare with ``>=`` / ``<= 0`` — a NaN cap makes both
    comparisons false and silently disables budget enforcement, so
    non-finite and non-positive values are rejected here.

    Args:
        text: The raw command-line value.

    Returns:
        The budget as a positive finite float.

    Raises:
        argparse.ArgumentTypeError: If *text* is not a number or is not
            a positive finite value.
    """
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid budget value: {text!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(
            f"budget must be a positive finite number, got {text!r}"
        )
    return value


def _ask_user_in_terminal(question: str) -> str:
    """Print *question* on the terminal and return the user's typed reply.

    Used as the ``ask_user_question_callback`` of :func:`main` so the
    agent's ``ask_user_question`` tool works when the agent runs from a
    shell instead of the kiss-web UI.

    Args:
        question: The question the agent wants the user to answer.

    Returns:
        The line the user typed, or an empty string on end-of-file.
    """
    print(f"\n{question}")
    try:
        return input("> ")
    except EOFError:
        return ""


def main() -> None:
    """Run a :class:`SorcarAgent` on a task given on the command line.

    Installed as the ``sorcar`` console script.  The task comes from
    exactly one of two required, mutually exclusive options: ``-t
    TASK`` runs the given string, and ``-f FILE`` runs the file's
    content as the task.  The agent works in ``$KISS_WORKDIR`` —
    exported by the ``~/.local/bin/sorcar`` wrapper the VS Code
    extension installs, so the agent acts on the directory the user
    invoked ``sorcar`` from — falling back to the current directory.
    Exits with status 0 when the agent reports success and 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        prog="sorcar",
        description="Run the KISS SorcarAgent on a task.",
        epilog='example: sorcar -t "Summarize README.md"',
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "-t",
        "--task",
        default=None,
        help="the task for the agent",
    )
    source.add_argument(
        "-f",
        "--file",
        default=None,
        help="file whose content is used as the task",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="",
        help="LLM model name (default: best model for the configured API keys)",
    )
    parser.add_argument(
        "-b",
        "--max-budget",
        type=_budget_arg,
        default=None,
        help="maximum budget in USD for the task",
    )
    parser.add_argument(
        "--work-dir",
        default="",
        help="directory the agent works in (default: $KISS_WORKDIR or the"
        " current directory)",
    )
    args = parser.parse_args()

    if args.file is not None:
        try:
            task = Path(args.file).read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            # UnicodeError too: a non-UTF-8 file must surface as the
            # same status-2 usage error as an unreadable one, not as an
            # uncaught UnicodeDecodeError traceback with exit status 1.
            parser.error(f"cannot read task file {args.file!r}: {exc}")
        if not task:
            parser.error(f"task file {args.file!r} is empty")
    else:
        task = args.task.strip()
        if not task:
            parser.error(
                'task must not be empty, e.g.: sorcar -t "Summarize README.md"'
            )

    model_name = args.model or get_default_model()
    if model_name == "No model":
        parser.exit(
            1,
            "sorcar: no model available — set at least one API key"
            " (e.g. ANTHROPIC_API_KEY) in the environment\n",
        )

    work_dir = args.work_dir or os.environ.get("KISS_WORKDIR") or os.getcwd()
    # Interactive terminals get the verbose console printer, which
    # already displays the formatted result at the end of the run —
    # printing the raw YAML again would show it twice.  Piped/redirected
    # stdout gets exactly the raw YAML result and nothing else.
    verbose = sys.stdout.isatty()
    agent = SorcarAgent("Sorcar CLI")
    result = agent.run(
        model_name=model_name,
        prompt_template=task,
        work_dir=work_dir,
        max_budget=args.max_budget,
        verbose=verbose,
        ask_user_question_callback=_ask_user_in_terminal,
    )
    if not verbose:
        print(result)
    try:
        success = bool(yaml.safe_load(result).get("success"))
    except Exception:
        success = False
    raise SystemExit(0 if success else 1)
