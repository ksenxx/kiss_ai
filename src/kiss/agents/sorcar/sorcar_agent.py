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
from typing import Any, NamedTuple, cast

import yaml

from kiss.agents.sorcar._concurrency import _race_delay
from kiss.agents.sorcar.decide_tool import decisions_tool_available, make_decide_tool
from kiss.agents.sorcar.fanout_guard import (
    REVIEW_BUDGET_REFUSAL,
    REVIEW_CAP_REFUSAL,
    REVIEWER_SPAWN_REFUSAL,
    ReviewQuota,
    is_implementation_task,
    is_review_task,
    parse_tasks_json,
    review_budget_for,
)
from kiss.agents.sorcar.persistence import _load_last_model, is_task_history_id
from kiss.agents.sorcar.relentless_agent import DEFAULT_MAX_BUDGET, RelentlessAgent
from kiss.agents.sorcar.skills import make_skill_tool
from kiss.agents.sorcar.task_classifier import (
    TaskClassification,
    classification_enabled,
    classify_task,
)
from kiss.agents.sorcar.useful_tools import (
    BackgroundJob,
    UsefulTools,
    rewrite_parent_repo_paths,
)
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.core import tool_interrupt
from kiss.core.base import SYSTEM_PROMPT, SYSTEM_PROMPT_LITE
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import BudgetExceededError, KISSError
from kiss.core.memoryfield.tools import MEMORY_PROTOCOL, MemoryTools
from kiss.core.models.model import Attachment
from kiss.core.models.model_info import (
    MODEL_INFO,
    OPENAI_COMPATIBLE_PROVIDERS,
    _match_openai_compatible_provider,
    _strip_provider_prefix,
    get_default_model,
    model_runs_task_to_completion,
)
from kiss.core.models.model_info import model as _model_factory
from kiss.core.printer import Printer
from kiss.core.tool_interrupt import ToolCallInterrupted
from kiss.core.utils import substitute_prompt_args

logger = logging.getLogger(__name__)

# Smallest per-child budget a ``run_parallel`` fan-out may hand out.
# Every child resumes the parent's chat context, so even its FIRST LLM
# step can cost a few cents; below this floor a child is killed by the
# budget check after step 1 having produced nothing (observed in
# recursive fan-outs that split a parent's remainder down to ~$0.03).
# See :meth:`SorcarAgent._subagent_budget_share`.
MIN_SUBAGENT_BUDGET = 0.50


TOOL_PROFILES: dict[str, frozenset[str] | None] = {
    # ``None``: every tool the agent can build (today's default).
    "full": None,
    # Reduced reviewer set: it can inspect the tree and run commands
    # (Bash is unrestricted, so this is not a sandbox) but has no file
    # editing, browser, talk, agent dispatch or fan-out tools.
    "review": frozenset({
        "Bash", "bash_job", "Read", "run_commands_parallel", "memory_search",
        "memory_pull", "memory_read", "memory_list", "decide", "summary",
    }),
    # Shell runner: just enough to run commands and read their output.
    "shell": frozenset({"Bash", "bash_job", "Read", "run_commands_parallel"}),
    # Single command runner (the bundled ``/sh`` agent): Bash and nothing else.
    "bash": frozenset({"Bash"}),
}
"""Tool profiles an agent can run with (``finish`` is always added).

Every tool schema is re-sent on every model step, so a reviewer that
carries the browser, channel, cron and fan-out tools pays for ~30
schemas it never calls.  The fan-out engine gives reviewer-marked
children the ``review`` profile; a parent may name a profile explicitly
through ``run_parallel(..., tool_profile=...)``, and a top-level run
through ``run(tool_profile=...)`` (the ``tool_profile`` parameter of
:func:`kiss.server.sorcar.run` / the ``run_agent`` tool, or an agent
script's ``tool_profile()`` getter).
"""


RESTRICTED_PROFILE_NOTE = """

# Restricted tool profile: {profile}
This sub-agent has only these tools plus finish: {tools}. Rules above that
require other tools (Write/Edit files, tmp/PROGRESS.md, browser research,
memory writes, run_parallel, run_agent, talk) do not apply here: do not attempt
them. Report everything in finish(summary_in_html=...).
"""


def summary(description: str) -> str:
    """Every 10 steps: summarize your steps since the last `summary` call.

    Args:
        description: Natural language summary in 5-10 sentences of
            what the agent since the last call to `summary`, written in
            Markdown format (use bullet lists for the steps, and
            ``**bold**`` / backtick code spans).

    Returns:
        A short confirmation string.
    """
    del description
    return "Summary recorded."


def _memory_root_for_run(
    append_basic_tools: bool,
    docker_image: str | None,
    model_name: str,
    caller_system_instruction: bool = False,
    use_memory_override: bool | None = None,
) -> Path | None:
    """The memory directory this run should use, or None for no memory.

    Memory rides with the built-in toolset, so four gates precede the
    user's ``use_memory`` setting (see :func:`_memory_settings`):

    * ``append_basic_tools=False`` strips the run down to ``finish`` plus
      the caller's tools — promising ``memory_*`` tools in the prompt
      would be a lie.
    * Docker runs replace Bash/Read/Edit/Write with container-backed
      tools; the memory tools execute in the host process on host paths,
      so registering them would hand a containerized task read/write
      access to host memory outside the ``docker_image`` boundary.
    * Run-to-completion CLI models (``cc/*``, ``codex/*``) never see
      KISS-registered tools, so they get neither the tools nor a protocol
      demanding them.
    * A caller-supplied ``model_config["system_instruction"]`` replaces
      the whole composed prompt (``KISSAgent.run`` only ``setdefault``-s
      it), so ``MEMORY_PROTOCOL`` would never reach the model; the tools
      must not be registered without the protocol that governs them.

    The gates are hard invariants — an explicit *use_memory_override*
    never bypasses them.  Past the gates, a boolean override is the
    caller's per-run choice and wins over both the ``KISS_USE_MEMORY``
    environment variable and the stored ``use_memory`` setting; ``None``
    (the default) keeps the :func:`_memory_settings` resolution.

    Args:
        append_basic_tools: Whether the run builds the built-in toolset.
        docker_image: The run's Docker image, if any.
        model_name: The resolved model name the run will use.
        caller_system_instruction: Whether the caller's ``model_config``
            carries its own ``system_instruction``.
        use_memory_override: Per-run memory toggle — ``True`` enables
            (subject to the gates above), ``False`` disables, ``None``
            falls back to the environment/config default.

    Returns:
        The memory root when every gate and the effective toggle allow
        it, else None.
    """
    if not append_basic_tools or docker_image or caller_system_instruction:
        return None
    if model_runs_task_to_completion(model_name):
        return None
    enabled, root = _memory_settings()
    if use_memory_override is not None:
        enabled = use_memory_override
    return root if enabled else None


def _memory_settings() -> tuple[bool, Path]:
    """Return whether persistent agent memory is enabled and where it lives.

    The toggle is the ``use_memory`` key of ``~/.kiss/config.json`` (see
    :data:`kiss.core.vscode_config.DEFAULTS`, default on).  The
    ``KISS_USE_MEMORY`` environment variable, when non-empty, wins over
    the stored value — ``0``/``false``/``no``/``off`` (any case) disable,
    anything else enables — so one process or test can flip memory
    without editing the config file.  Pages live in the config's
    ``memory_dir`` when set, else ``$KISS_HOME/memories``
    (``~/.kiss/memories``).

    Returns:
        ``(enabled, root)`` where *root* is the memory page directory
        (created lazily on first write by
        :class:`kiss.core.memoryfield.pages.MemoryDir`).
    """
    from kiss.core.config import kiss_home
    from kiss.core.vscode_config import load_config

    cfg = load_config()
    env = os.environ.get("KISS_USE_MEMORY", "").strip().lower()
    if env:
        enabled = env not in ("0", "false", "no", "off")
    else:
        enabled = bool(cfg.get("use_memory", True))
    raw_dir = str(cfg.get("memory_dir", "")).strip()
    root = Path(raw_dir).expanduser() if raw_dir else kiss_home() / "memories"
    return enabled, root


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
    if not GitWorktreeOps.has_staged_changes(commit_dir):
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
    """Return ``(budget_used, total_tokens_used, total_steps)`` for *agent*.

    A :class:`~kiss.agents.sorcar.relentless_agent.RelentlessAgent`
    derives the triple from one append-only usage ledger; it MUST be
    read once through ``usage_snapshot()``, which sums a single ledger
    reference.  Three separate property reads each sum the ledger
    afresh, so concurrent banks between them tear the triple — a final
    abandoned-child reclaim that read the OLD budget with the NEW
    tokens/steps then discarded the child and permanently lost its
    spend from the parent's accounting.  The per-attribute fallback
    serves agent-shaped objects whose fields are plain attributes.

    A bare :class:`KISSAgent` also exposes ``usage_snapshot()``, but
    its third element is the SESSION dimension ``step_count`` — not
    the folded ``total_steps`` this reader reports — so it goes
    through the attribute fallback on purpose (its per-field property
    reads are each one atomic snapshot load; only ledger-bearing
    agents need the cross-field snapshot here, because only their
    per-property reads re-sum a concurrently growing ledger).
    """
    snapshot = getattr(agent, "usage_snapshot", None)
    if callable(snapshot) and not isinstance(agent, KISSAgent):
        budget, tokens, steps = cast("tuple[float, int, int]", snapshot())
        return float(budget or 0.0), int(tokens or 0), int(steps or 0)
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


def _notify_subagent_done(
    printer: Any, sub_task_id: str | None, sub_tab_id: str, model: str = "",
) -> None:
    """Broadcast ``subagentDone`` to every tab watching a finished sub-agent.

    The targets are the tabs subscribed to *sub_task_id* through the
    printer's fan-out registry plus the sub-agent's own synthetic
    *sub_tab_id*.  An empty *sub_task_id* (no ``task_history`` row was
    allocated, so no ``new_tab`` was ever broadcast) fans out to the
    synthetic tab only.  Shared by ``run_parallel`` children, ``run_agent``
    dispatches, the merge agent and the task-update agent.

    Args:
        printer: The parent task's printer.
        sub_task_id: The sub-agent's persisted task id, or ``None``.
        sub_tab_id: The sub-agent's synthetic tab id (may be empty).
        model: The model the sub-agent was launched with, restored into
            the watching tabs' pickers.
    """
    viewer_ids: list[str] = []
    fanout = getattr(printer, "_fanout_targets", None)
    if callable(fanout) and sub_task_id:
        found = fanout(sub_task_id)
        if isinstance(found, list):
            viewer_ids = [v for v in found if v]
    if sub_tab_id and sub_tab_id not in viewer_ids:
        viewer_ids.append(sub_tab_id)
    _broadcast_subagent_done(printer, viewer_ids, model)


# How long the parent may sit in one wait() before re-reading its stop
# event.  A completed child wakes the wait immediately, so this only
# bounds flag-checking: the abandon path below allows 15s anyway, and
# ``_force_stop_thread`` waits 1s before its first injection and retries
# at +5s.  It must not be much smaller: nested fan-outs put one waiting
# parent on the stack per level, and every one of them wakes on this
# interval, so a 0.1s slice made a deeply nested tree crawl under GIL
# contention.
_SUBAGENT_POLL_SECONDS = 1.0
# The slice while the fan-out runs as a tool call (run_parallel): below
# kiss.core.tool_interrupt's 1 s injection grace, see _await_subagents.
_SUBAGENT_TOOL_POLL_SECONDS = 0.4
_SUBAGENT_STOP_GRACE_SECONDS = 15.0
# Upper bound on how long _LiveUsageMonitor.stop() waits for its polling
# thread; the thread calls printer.print synchronously and a blocked
# printer must not hang the parent task's fan-out unwind.
_LIVE_USAGE_JOIN_TIMEOUT = 2.0


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

    The ``run_parallel`` tool panel's own Stop is honored cooperatively
    too: each wake checks the running tool call's interrupt and raises
    ``ToolCallInterrupted`` at once (the caller then signals the
    children through the fan-out's stop event).

    Raises:
        KeyboardInterrupt: When a stop was requested and at least one
            child was still running after the grace period.
        ToolCallInterrupted: When the user stopped the ``run_parallel``
            tool call while children were still running.
    """
    pending = set(futures)
    give_up_at: float | None = None
    # Under a tool call the slice stays well inside the interrupt's
    # cooperative grace, so the raise below always beats the forced
    # injection (which would otherwise land inside ``wait``).
    poll = (
        _SUBAGENT_TOOL_POLL_SECONDS
        if tool_interrupt.current_tool_call() is not None
        else _SUBAGENT_POLL_SECONDS
    )
    while pending:
        _done, pending = wait(pending, timeout=poll)
        if not pending:
            break
        tool_interrupt.raise_if_interrupted()
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


class _ClassifierSpend(NamedTuple):
    """One pre-run task classification's complete, immutable spend.

    ``SorcarAgent._classify_task_once`` publishes the WHOLE outcome —
    the retry-stable fold key together with all three spend dimensions
    — as one instance with a single ``STORE_ATTR``, so an
    asynchronously injected stop can only leave the agent with the
    complete outcome or with no outcome at all, never with a torn
    subset (round-6 finding 3: a stop between separate key and spend
    stores made the mandatory fold consume a zero triple and clear the
    transaction, silently dropping real provider spend).
    """

    key: str
    budget: float
    tokens: int
    steps: int


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
        epoch: Any = None,
    ) -> None:
        """Record *future*/*agent*, the usage already attributed, and the epoch.

        Args:
            future: The abandoned worker's future.
            agent: The abandoned child's agent.
            counted: The ``(budget, tokens, steps)`` the parent has
                already attributed for this child.
            epoch: The parent's usage-ledger epoch token
                (``RelentlessAgent._usage_epoch()``) at registration,
                or ``None`` when the parent has no ledger.  Every
                reclaim commit is BOUND to this exact object: after a
                ``reset_usage()`` (the next run's task boundary) the
                child's further spend settles into the discarded
                prior-epoch ledger — it belongs to the finished PRIOR
                task and must not corrupt the new task's accounting
                (round-4 finding 3; round-6 finding 2 closed the
                check-then-commit race by binding the commit itself,
                not just checking the token first).  The item stays
                tracked for liveness either way (worktree-deletion
                safety).
        """
        self.future = future
        self.agent = agent
        self.epoch = epoch
        # The transaction identity of this source's reclaim commits:
        # generation ``g``'s ledger record carries the retry-stable
        # source ``"reclaim:<txn_id>"`` with sequence ``g`` —
        # generations are committed in increasing order (the ledger's
        # per-source monotonic-seq contract), so pre- and post-append
        # retries deduplicate to exactly one contribution per
        # generation.
        self.txn_id = uuid.uuid4().hex
        # ``(generation, counted)`` advances in ONE store, only AFTER
        # generation ``generation``'s ledger append: an interrupt
        # anywhere in the commit retries the SAME transaction.
        self.checkpoint: tuple[int, tuple[float, int, int]] = (0, counted)
        # Write-ahead intent ``(generation, live_snapshot)``: fixes the
        # amount generation ``generation`` banks BEFORE the append, so
        # a retry re-appends the identical record (same key, same
        # values) instead of recomputing a different delta that
        # first-record-wins dedup would silently drop.
        self.pending: tuple[int, tuple[float, int, int]] | None = None

    @property
    def counted(self) -> tuple[float, int, int]:
        """The ``(budget, tokens, steps)`` already attributed to the parent."""
        return self.checkpoint[1]

    def bank_unbanked(self, parent: Any) -> tuple[float, int, int]:
        """Attribute the child's spend since the last checkpoint, exactly once.

        The commit order is append-then-advance with a write-ahead
        intent, so an asynchronously injected stop at ANY point makes
        the next reclaim retry the SAME transaction:

        1. Load the checkpoint ``(generation, counted)``.
        2. Reuse (or record — one store) the pending intent
           ``(generation, live)``; the live snapshot is clamped to
           ``counted`` because a mid-handoff read can momentarily
           REGRESS (RelentlessAgent detaches ``_current_executor``
           before folding its spend), and banking a negative delta or
           lowering the checkpoint would double-count later.
        3. Append the delta as ONE keyed ledger record — source
           ``"reclaim:<txn_id>"``, seq ``generation`` — bound to
           :attr:`epoch`, the ledger object captured at registration.
           A retry that finds the intent re-appends the identical
           record and read-side dedup counts it once; the pre-round-5
           order (advance ``counted`` first, append second) let a stop
           between the two permanently drop a finished child's spend.
           Binding the commit to the captured epoch object (round-6
           finding 2) means a ``reset_usage()`` racing this step can
           never divert the delta into the NEW task's ledger: the
           record lands in the discarded prior-epoch object, which
           nobody sums.
        4. Advance the checkpoint in ONE store, then clear the intent.
           A stale intent from a completed generation (stop between 4
           and the clear) is recognized by its generation number and
           discarded.

        Callers serialize on ``_abandoned_lock``; this method is not
        safe for concurrent calls on one item.

        Args:
            parent: The agent whose ledger receives the attribution.

        Returns:
            The ``(budget, tokens, steps)`` delta banked by this call
            (``(0.0, 0, 0)`` when the child reported nothing new) —
            callers use it to surface late spend that settled into a
            superseded epoch.
        """
        generation, counted = self.checkpoint
        pending = self.pending
        if pending is not None and pending[0] != generation:
            # A finished generation's leftover intent (the stop landed
            # between the checkpoint advance and the intent clear).
            self.pending = None
            pending = None
        if pending is None:
            live = _live_agent_usage(self.agent)
            live = (
                max(live[0], counted[0]),
                max(live[1], counted[1]),
                max(live[2], counted[2]),
            )
            if live == counted:
                return (0.0, 0, 0)
            pending = (generation, live)
            self.pending = pending
        live = pending[1]
        delta = (
            live[0] - counted[0],
            live[1] - counted[1],
            live[2] - counted[2],
        )
        # Test hook (no-op in production): widens the commit window so
        # concurrency tests can prove the caller serialises reclaims
        # (see reclaim_abandoned_subagents).
        _race_delay()
        if delta[0] or delta[1] or delta[2]:
            _attribute_sub_usage(
                parent,
                delta[0],
                delta[1],
                delta[2],
                key=f"reclaim:{self.txn_id}",
                seq=generation,
                epoch=self.epoch,
            )
        self.checkpoint = (generation + 1, live)
        self.pending = None
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
    lock = getattr(parent_agent, "_abandoned_lock", None)
    if lock is None:
        return
    # Tag each item with the parent's CURRENT ledger epoch: every
    # reclaim commit for the item is bound to this object, so late
    # spend after the parent's next reset_usage() (a new task) settles
    # into the discarded epoch instead of the new task's accounting.
    epoch_of = getattr(parent_agent, "_usage_epoch", None)
    epoch = epoch_of() if callable(epoch_of) else None
    with lock:
        # The tracking list is fetched INSIDE the lock: reading it
        # before acquisition let a concurrent reclaim (which used to
        # replace the attribute under the lock) strand a live child in
        # a detached list, so worktree cleanup saw no abandoned
        # children while the child's thread still wrote into the
        # directory (round-6 finding 4).  Reclaim also mutates the
        # list in place now, so the object registered into is always
        # the object reclaim scans.
        pending = getattr(parent_agent, "_abandoned_subagents", None)
        if pending is None:
            return
        for idx, future in enumerate(futures):
            if future.done() or sub_agents[idx] is None:
                continue
            pending.append(
                _AbandonedSubagent(
                    future, sub_agents[idx], sub_usage[idx], epoch=epoch,
                )
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
    # ONE coherent snapshot when the executor publishes one (KISSAgent
    # stores its whole triple as one immutable record): this function
    # is polled from other threads (_LiveUsageMonitor, unfinished-usage
    # collection, abandoned-child reclaim) while the executor thread
    # updates the counters, and three separate property reads could
    # pair one response's tokens with the pre-response cost.
    snapshot = getattr(executor, "usage_snapshot", None)
    if callable(snapshot):
        budget, tokens, steps = cast("tuple[float, int, int]", snapshot())
        return float(budget or 0.0), int(tokens or 0), int(steps or 0)
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
        # Set by stop() when the join timed out: _emit() must not START
        # a print after the caller has moved on to the offsets bump.
        self._detached = threading.Event()
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

        Safe to call when :meth:`start` never ran, raised (thread
        exhaustion) or was interrupted by a stop injected while it was
        waiting for the thread to come up: ``join`` would raise on a
        thread that never registered as started, so only a live thread
        is joined — ``_done`` is set regardless, and a thread that did
        start exits at its next tick.

        The join is bounded: the monitor calls ``printer.print``
        synchronously, and a printer whose sink stops consuming (a
        pipe/file nobody reads) would otherwise hang the fan-out's
        ``finally`` — and with it the parent task's unwind and usage
        accounting — forever.  On timeout the monitor is detached:
        ``_detached`` stops any emission that has not yet reached the
        printer, a warning is logged, and the caller proceeds.  A print
        already blocked inside the printer may still complete once
        after detachment; that is the accepted trade-off for never
        hanging the task.
        """
        self._done.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(_LIVE_USAGE_JOIN_TIMEOUT)
            if thread.is_alive():
                self._detached.set()
                logger.warning(
                    "Live usage monitor did not stop within %.1fs "
                    "(printer blocked?); detaching it and continuing",
                    _LIVE_USAGE_JOIN_TIMEOUT,
                )

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
        if self._detached.is_set():
            # stop() gave up waiting for this thread; the parent has
            # already bumped its offsets, so this emission would
            # double-count the sub-agents' spend.
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


def _attribute_sub_usage(
    agent: Any,
    budget: float,
    tokens: int,
    steps: int,
    key: str | None = None,
    seq: int = 0,
    epoch: Any = None,
) -> None:
    """Attribute sub-agents' cost, tokens, and steps to the parent *agent*.

    Without this, sub-agent budgets would be invisible to the parent
    agent's global accounting and UI.  Also updates the printer offsets
    so the live status line in the current sub-session reflects the
    additional spend immediately (the offsets are otherwise
    snapshotted only at session start).

    The delta is committed through the agent's
    :meth:`RelentlessAgent._attribute_usage` — ONE atomic append of an
    immutable record carrying all three dimensions to the agent's
    append-only usage ledger.  No writer can overwrite or lose another
    writer's record: this function is called concurrently by the agent
    thread (a fan-out's ``finally``, a ``talk`` synthesis bank) and by
    server threads (:meth:`SorcarAgent.reclaim_abandoned_subagents`
    from worktree cleanup / teardown / discard), and it can also cross
    a :meth:`RelentlessAgent._reset` — three separate property stores
    used to let the reset land between them and publish an impossible
    mixed state (zero budget, pre-reset tokens/steps), whereas the
    single record now lands wholly in the old epoch (discarded with
    it) or wholly in the new one.  The append takes no lock, so a
    caller holding ``_abandoned_lock`` (``reclaim_abandoned_subagents``)
    can never deadlock here, even after an injected stop.  A minimal
    agent-shaped object without ``_attribute_usage`` gets plain
    attribute increments (no cross-thread protection, but such objects
    are single-threaded by construction).

    Args:
        agent: The parent agent receiving the attribution.
        budget: USD spend to add.
        tokens: Token count to add.
        steps: Step count to add.
        key: Stable transaction source for a retryable adjustment (an
            abandoned-child reclaim); ``None`` for one-shot
            attributions (fan-out totals, TTS), which need no dedup
            identity because they are appended at most once.
        seq: Per-*key* monotonic sequence number (a reclaim's
            generation); ignored for one-shot attributions.
        epoch: The ledger epoch object the commit is bound to (see
            ``RelentlessAgent._attribute_usage``), or ``None`` for the
            agent's current epoch.  A reclaim passes the epoch
            captured when the abandoned child was registered, so a
            concurrent reset can never divert a prior task's spend
            into the new task's ledger.
    """
    attribute = getattr(agent, "_attribute_usage", None)
    if callable(attribute):
        attribute(budget, tokens, steps, key=key, seq=seq, epoch=epoch)
    else:
        agent.budget_used = float(getattr(agent, "budget_used", 0.0) or 0.0) + budget
        agent.total_tokens_used = (
            int(getattr(agent, "total_tokens_used", 0) or 0) + tokens
        )
        agent.total_steps = int(getattr(agent, "total_steps", 0) or 0) + steps
    if agent.printer is not None:
        try:
            # One coherent triple (see _agent_usage): separate property
            # reads could tear across a concurrent snapshot publish.
            budget_total, tokens_total, steps_total = _agent_usage(agent)
            agent.printer.budget_offset = budget_total
            agent.printer.tokens_offset = tokens_total
            agent.printer.steps_offset = steps_total
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
        # Persistent agent memory (kiss.core.memoryfield), built per
        # run by :meth:`run` when the ``use_memory`` config flag (or the
        # KISS_USE_MEMORY environment variable) enables it; None keeps
        # the run memory-free.  :meth:`_get_tools` registers its tools.
        self._memory_tools: MemoryTools | None = None
        # Per-run memory toggle (:meth:`run`'s *use_memory*), kept on
        # self so the ``run_parallel`` fan-out — which executes DURING
        # the run — forwards the same override to every sub-agent.
        self._use_memory_override: bool | None = None
        self._use_web_tools: bool = True
        self._is_parallel: bool = True
        self._append_basic_tools: bool = True
        # Background jobs started by ``Bash(background=True)``, kept on
        # the agent (not the per-run UsefulTools) so a follow-up prompt
        # in the same chat can still wait on, tail or kill them.
        self._background_jobs: dict[str, BackgroundJob] = {}
        # Task-tree-wide budget of review fan-outs (see
        # :class:`fanout_guard.ReviewQuota`).  A top-level :meth:`run`
        # creates a fresh one; the fan-out engine hands the parent's
        # instance to every child, so the whole in-process tree draws
        # from ONE budget of ``MAX_REVIEW_ROUNDS`` rounds.
        self._review_quota: ReviewQuota | None = None
        # Explicit tool profile named by the fan-out (a TOOL_PROFILES key)
        # or "" to let _tool_profile() decide from the reviewer marker.
        self._tool_profile_name: str = ""
        # Pre-run task classification state (see
        # :meth:`_classify_task_once`).  ``_classification_attempted``
        # makes the classifier run at most once per task even though
        # both ``WorktreeSorcarAgent.run`` (for the worktree decision)
        # and :meth:`run` (for the system prompt) consult it; one
        # immutable ``_classifier_spend`` record holds the classifier's
        # spend until :meth:`_fold_classifier_usage` banks it into the
        # run totals.
        self._task_classification: TaskClassification | None = None
        self._classification_attempted: bool = False
        # True when the verdict was established by an external driver
        # (the server's task runner via :meth:`classify_task_for_run`)
        # for the CURRENT submission: internal per-run resets then keep
        # the verdict so every subtask of the submission reuses it and
        # the driver's own worktree bookkeeping never disagrees with
        # the agent's.  Cleared by the next classify_task_for_run call.
        self._classification_preseeded: bool = False
        # The UNFOLDED classifier spend: one immutable record carrying
        # the retry-stable fold key AND all three spend dimensions,
        # published with a single store (see :class:`_ClassifierSpend`)
        # — never a partial subset.  The fold appends its ledger record
        # under the record's key BEFORE clearing this field, so a
        # stop-interrupted fold retried later re-appends the SAME key
        # and read-side dedup makes the spend count exactly once.
        self._classifier_spend: _ClassifierSpend | None = None
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
        # ``item.bank_unbanked()`` (an intent/checkpoint transaction on
        # the item) would otherwise race a concurrent reclaimer — the
        # agent thread and server-side worktree cleanup call this
        # concurrently — double-counting the child's spend.
        # (``_attribute_sub_usage`` itself is one lock-free ledger
        # append and needs no serialization.)  Neither callee acquires
        # ``_abandoned_lock``, so this cannot deadlock.
        #
        # The pass runs over the CURRENT list, not the pre-wait
        # snapshot: a child abandoned by a fan-out that ended during
        # the wait must count towards the return value (callers delete
        # the shared working directory on True), and one that a
        # concurrent reclaimer already banked and forgot must not be
        # banked again.
        epoch_of = getattr(self, "_usage_epoch", None)
        current_epoch = epoch_of() if callable(epoch_of) else None
        with self._abandoned_lock:
            still_running: list[_AbandonedSubagent] = []
            for item in self._abandoned_subagents:
                # Every commit is BOUND to item.epoch (the ledger
                # object captured at registration), so no epoch
                # check-then-commit race with reset_usage() exists: an
                # item registered under a PRIOR epoch settles its
                # further spend into that finished task's discarded
                # ledger, never into the current task's totals.  Late
                # spend settled that way is REAL provider spend that
                # no live task reports — an explicit, documented
                # undercount (the finished task's terminal accounting
                # was already persisted when it ended), surfaced in
                # the log below.  The thread is still tracked so a
                # live child keeps blocking worktree deletion.
                banked = item.bank_unbanked(self)
                if (
                    any(banked)
                    and item.epoch is not None
                    and item.epoch is not current_epoch
                ):
                    logger.info(
                        "Abandoned sub-agent spend "
                        "(budget=%.6f tokens=%d steps=%d) arrived after "
                        "its task's usage epoch ended; settled into the "
                        "finished task's ledger, not the current task.",
                        banked[0],
                        banked[1],
                        banked[2],
                    )
                if not item.future.done():
                    still_running.append(item)
            # In-place update: registration appends to the same list
            # object it fetched under this lock, so replacing the
            # attribute could strand a concurrent registration's items
            # in a detached list (round-6 finding 4).
            self._abandoned_subagents[:] = still_running
        return not still_running

    def _subagent_budget_share(self, num_tasks: int) -> float | None:
        """Return the ``max_budget`` each parallel sub-agent may spend.

        Splits this task's REMAINING budget — ``max_budget`` minus the
        spend already attributed to this agent minus the live executor
        session's own spend — evenly across *num_tasks* sub-agents PLUS
        one reserved parent share.  Reserving that share leaves the main
        agent enough budget to process the results and finish; importantly,
        even a one-item fan-out cannot consume the parent's entire remainder.

        A share below :data:`MIN_SUBAGENT_BUDGET` is refused instead of
        spawned: each sub-agent resumes the parent's whole chat context,
        so its very first LLM step can cost a few cents, and a child
        whose budget cannot cover even that step is killed on step 1
        having done nothing.  Recursive fan-outs used to divide the
        remainder into such doomed slivers (each nesting level splits by
        ``num_tasks + 1``), burning budget on children that all returned
        "Task failed".  Refusing with a plain :class:`KISSError` — not
        :class:`BudgetExceededError`, which aborts the whole agent —
        surfaces an actionable tool error so the parent model does the
        work inline instead.

        Args:
            num_tasks: Number of parallel sub-agent tasks about to spawn.

        Returns:
            The per-sub-agent budget share in USD, or ``None`` when this
            agent has no budget context yet (``run``/``_reset`` never
            ran, e.g. direct ``_run_tasks_parallel`` invocations) — the
            sub-agents then fall back to their default budget.

        Raises:
            BudgetExceededError: If the task has no remaining budget.
            KISSError: If the per-child share would be below
                :data:`MIN_SUBAGENT_BUDGET`.
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
        if num_tasks <= 0:
            return remaining
        share = remaining / (num_tasks + 1)
        # 1e-9 tolerance: budget arithmetic is float subtraction, so a
        # mathematically exact floor share (e.g. (1.13-0.13)/2 per child)
        # can land a few ULPs below 0.50 and must not be refused with a
        # self-contradictory "$0.50 is below the $0.50 minimum" message.
        if share < MIN_SUBAGENT_BUDGET - 1e-9:
            raise KISSError(
                f"Refusing to spawn {num_tasks} parallel sub-agent(s): the "
                f"remaining budget ${remaining:.2f} gives each sub-agent "
                f"only ${share:.2f}, below the ${MIN_SUBAGENT_BUDGET:.2f} "
                "minimum a sub-agent needs to do useful work. Do the work "
                "inline yourself (without run_parallel), or fan out fewer "
                "tasks."
            )
        return share

    def _is_reviewer_subagent(self) -> bool:
        """Return whether this agent runs inside a reviewer's sub-tree.

        The fan-out engine stamps ``_subagent_info["reviewer"]`` on a
        child whose task is a review task or whose parent is itself a
        reviewer, so the flag covers the whole sub-tree under a
        reviewer.

        Returns:
            True for a reviewer sub-agent or any of its descendants.
        """
        info = getattr(self, "_subagent_info", None) or {}
        return bool(info.get("reviewer", False))

    def _subagent_parent_tab_id(self) -> str:
        """Return the frontend tab id sub-agents should call their parent.

        Normally this agent's own ``_tab_id``.  When this agent is
        itself a sub-agent (nested ``run_parallel``, or a ``run_agent``
        dispatch), its ``_tab_id`` is a synthetic id — the fan-out's
        invented one, or the daemon dispatch's ``api-…`` id — which no
        webview has opened; the printer's viewer registry knows which
        tab is really watching this task, so that one wins.  The
        agent's own synthetic id is excluded from the candidates: a
        ``run_agent`` dispatch registers it as a subscriber too (see
        ``register_task_ui``), and picking it would parent the nested
        children under a tab no client has, dropping their tabs.

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
        viewer_ids = sorted(
            v for v in fanout(own_task_id) if v and v != tab_id
        )
        return viewer_ids[0] if viewer_ids else tab_id

    def _run_tasks_parallel(
        self,
        tasks: list[str],
        max_workers: int | None = None,
        model_name: str | None = None,
        tool_profile: str = "",
        review_budget: float | None = None,
        review_flags: list[bool] | None = None,
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
            model_name: Model for the children; ``None`` uses this
                agent's.  A different model is run with default provider
                routing (this agent's ``model_config`` is not forwarded,
                since its endpoint and key belong to this agent's model).
            tool_profile: Explicit tool profile for the children (a key
                of :data:`TOOL_PROFILES`); ``""`` lets each child pick.
            review_budget: USD already reserved from the review quota
                for this fan-out's reviewer children; each reviewer's
                budget is clipped to its share of it and the part the
                reviewers did not spend is released afterwards (also
                when the fan-out raises).
            review_flags: Which of *tasks* are reviewer children (same
                length as *tasks*); ``None`` means none.

        Returns:
            List of YAML result strings in the same order as *tasks*.
        """
        totals: dict[str, float | list[float]] = {}
        flags = review_flags or [False] * len(tasks)
        try:
            return self._run_tasks_parallel_inner(
                tasks, max_workers, model_name, tool_profile, review_budget, flags, totals,
            )
        finally:
            if review_budget is not None and self._review_quota is not None:
                per_task = totals.get("budget_used_per_task")
                spent_by_reviewers = (
                    sum(u for u, flag in zip(per_task, flags, strict=True) if flag)
                    if isinstance(per_task, list) else 0.0
                )
                self._review_quota.release(review_budget - spent_by_reviewers)

    def _run_tasks_parallel_inner(
        self,
        tasks: list[str],
        max_workers: int | None,
        model_name: str | None,
        tool_profile: str,
        review_budget: float | None,
        review_flags: list[bool],
        totals: dict[str, float | list[float]],
    ) -> list[str]:
        """Body of :meth:`_run_tasks_parallel`; *totals* receives the engine's usage."""
        # Bank whatever an earlier fan-out's abandoned children spent
        # after this agent stopped waiting for them, before the budget
        # share below is computed from those totals.
        self.reclaim_abandoned_subagents()
        monitor = _LiveUsageMonitor(self, self.printer)
        share = self._subagent_budget_share(len(tasks))
        child_budgets: list[float | None] | None = None
        if review_budget is not None and share is not None:
            review_share = min(share, review_budget / max(1, sum(review_flags)))
            child_budgets = [review_share if flag else share for flag in review_flags]
        child_model = model_name or self.model_name
        # Sub-agents act in the parent's live container, not on the host
        # and not in a fresh container of their own.
        child_docker_image: str | None = None
        if self.docker_manager is not None and self.docker_manager.container is not None:
            from kiss.agents.sorcar.docker_manager import ATTACH_PREFIX

            child_docker_image = ATTACH_PREFIX + self.docker_manager.container.id
        try:
            # Started inside the try: a stop injected between the start
            # and the try would otherwise leak the polling thread —
            # emitting every second for the rest of the process.
            monitor.start()
            results = run_tasks_parallel(
                tasks,
                max_workers=max_workers,
                model_name=child_model,
                work_dir=self.work_dir,
                docker_image=child_docker_image,
                printer=self.printer,
                totals_out=totals,
                usage_monitor=monitor,
                max_budget=share,
                child_budgets=child_budgets,
                model_config=(
                    getattr(self, "model_config", None)
                    if child_model == self.model_name else None
                ),
                tool_profile=tool_profile,
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
                use_memory=self._use_memory_override,
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
                float(cast(float, totals.get("budget_used", 0.0))),
                int(cast(float, totals.get("total_tokens_used", 0))),
                int(cast(float, totals.get("total_steps", 0))),
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

    def _tool_profile(self, task: str = "") -> str:
        """Return the tool profile this agent runs with.

        A profile named explicitly by the fan-out (``_tool_profile_name``)
        wins.  Otherwise a reviewer sub-agent gets the reduced ``review``
        profile when ``DEFAULT_CONFIG.tool_profiles`` is on and its task
        does not ask for changes (:func:`is_implementation_task`), and
        everything else gets ``full``.

        Args:
            task: The task text to judge; defaults to this agent's
                ``task_description``.

        Returns:
            One of the keys of :data:`TOOL_PROFILES`.
        """
        explicit = str(getattr(self, "_tool_profile_name", "") or "")
        if explicit in TOOL_PROFILES:
            return explicit
        task = task or str(getattr(self, "task_description", "") or "")
        if (
            DEFAULT_CONFIG.tool_profiles
            and self._is_reviewer_subagent()
            and not is_implementation_task(task)
        ):
            return "review"
        return "full"

    def _get_tools(self) -> list:
        """Build tool list, using DockerTools when docker_manager is active.

        Must be called after docker_manager is set up (i.e., from perform_task,
        not from run() before super().run()).

        The list is cut down to the agent's tool profile
        (:meth:`_tool_profile`): a ``review`` or ``shell`` sub-agent
        never builds the browser, MCP, channel-dispatch or fan-out
        tools, so its every step carries only the schemas it can use.
        """
        profile = self._tool_profile()
        allowed = TOOL_PROFILES[profile]

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
            from kiss.agents.sorcar import cron_agent

            if cron_agent.is_unattended(self):
                # A cron run (or its sub-task) has no one to answer; a
                # blocked question would only stall until the timeout.
                return (
                    "Error: this task runs unattended (scheduled automation) and "
                    "nobody can answer. Do not ask again: proceed on the most "
                    "reasonable assumption, or report the blocker in your final "
                    "summary and finish."
                )
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
            self.docker_manager.stop_event = getattr(self, "_stop_event", None)

            def Bash(  # noqa: N802
                command: str,
                description: str,
                timeout_seconds: int = 30,
                max_output_chars: int = 50000,
                background: bool = False,
            ) -> str:
                """Runs a bash command in the task's Docker container and returns its output.

                Background jobs (``background=True``) are not available in
                Docker mode: start long commands yourself with
                ``nohup cmd > /tmp/out.log 2>&1 < /dev/null &`` and poll
                the log file with ``tail``.

                Args:
                    command: The bash command to run.
                    description: A brief description of the command.
                    timeout_seconds: Timeout in seconds for the command.
                    max_output_chars: Maximum characters in output before truncation.
                    background: Must stay false in Docker mode (see above).

                Returns:
                    The output of the command.
                """
                if background:
                    return (
                        "Error: background=True is not available in Docker mode. "
                        "Run the command as `nohup cmd > /tmp/out.log 2>&1 "
                        "< /dev/null &` and poll the log with `tail`."
                    )
                return self._docker_bash(
                    command, description, timeout_seconds, max_output_chars,
                )

            tools: list = [
                Bash, self.docker_manager.run_commands_parallel,
                docker_tools.Read, docker_tools.Edit, docker_tools.Write,
            ]
        else:
            useful_tools = UsefulTools(
                stream_callback=_stream,
                stop_event=getattr(self, "_stop_event", None),
                work_dir=self.work_dir,
                jobs=self._background_jobs,
            )
            tools = [
                useful_tools.Bash, useful_tools.bash_job,
                useful_tools.run_commands_parallel,
                useful_tools.Read, useful_tools.Edit, useful_tools.Write,
            ]
            # The Read dedupe assumes an earlier output is still in the
            # model's context; the executor calls this when it is not.
            self.context_reset_hook = useful_tools.forget_reads
        if allowed is None and self._use_web_tools and self.web_use_tool is None:
            # Sub-agents run concurrently, so they get a throwaway profile
            # instead of contending for the shared profile's Chromium lock.
            self.web_use_tool = WebUseTool(
                work_dir=self.work_dir,
                ephemeral=getattr(self, "_subagent_info", None) is not None,
            )
            tools.extend(self.web_use_tool.get_tools())
        def run_parallel(
            tasks: str, max_workers: str = "", model_name: str = "",
            tool_profile: str = "",
        ) -> str:
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

            **When NOT to call run_parallel:** when each task is just a
            shell command whose output you need (test splits, builds,
            lints).  Use ``run_commands_parallel`` for those: it runs the
            commands concurrently without spawning LLM sub-agents.

            **Hard limits (enforced, not advisory):**
            - ``tasks`` must be a literal JSON array; shell substitutions
              such as ``"$(cat tasks.json)"`` are not expanded and are
              rejected.
            - At most 3 fan-outs per task tree may contain review/audit
              tasks (the budget is shared with every sub-agent); the 4th
              is refused, so verify the last fixes yourself.
            - A reviewer sub-agent (and anything it spawns) may not spawn
              further reviewers.
            - When the user's task names a review share ("at most 40%
              of the budget for reviewing"), reviewers may be handed at
              most that share of the top-level budget in total; a
              review fan-out is clipped to what is left and refused
              once nothing useful is left.

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
                model_name: LLM model for the sub-agents (e.g. a cheaper
                    or a different reviewer model).  Empty (default)
                    uses this agent's model.  Prefer this over asking
                    the sub-agent to call ``set_model`` itself, which
                    costs a whole step on the wrong model.
                tool_profile: ``"review"`` gives the sub-agents the
                    read-only toolset (Bash, bash_job, Read,
                    run_commands_parallel, memory reads, decide, summary);
                    ``"shell"`` just Bash, bash_job, Read and
                    run_commands_parallel.  Empty
                    (default): review tasks get ``"review"``, others
                    the full toolset.

            Returns:
                A YAML-formatted string containing a list of result
                objects, one per task, in the same order as the input.
                Each result object has ``success`` and ``summary`` keys.
                A string starting with ``Error:`` when the call was
                refused by one of the hard limits above.
            """
            try:
                task_list = parse_tasks_json(tasks)
            except ValueError as e:
                return f"Error: {e.args[0]}"
            from kiss.agents.sorcar import cron_agent

            if cron_agent.is_unattended(self):
                task_list = [cron_agent.unattended_child_prompt(t) for t in task_list]
            try:
                workers: int | None = int(max_workers) if max_workers else None
            except ValueError:
                return (
                    f"Error: max_workers must be an integer string, "
                    f"got {max_workers!r}."
                )
            if workers is not None and workers < 1:
                return f"Error: max_workers must be at least 1, got {workers}."
            if tool_profile and tool_profile not in TOOL_PROFILES:
                return (
                    f"Error: tool_profile must be one of "
                    f"{', '.join(TOOL_PROFILES)}, got {tool_profile!r}."
                )
            # Review intent: the task text, or an explicit review profile.
            review_flags = [
                tool_profile == "review" or is_review_task(t) for t in task_list
            ]
            review_budget: float | None = None
            if any(review_flags):
                if self._is_reviewer_subagent():
                    return f"Error: {REVIEWER_SPAWN_REFUSAL}"
                # Zero-child preflight BEFORE reserving a round: a
                # fan-out refused for budget must not burn the review
                # quota (it spawned no reviewer).  Raises the same
                # error the dispatch itself would.
                share = self._subagent_budget_share(len(task_list))
                if self._review_quota is None:
                    self._review_quota = ReviewQuota()
                if share is not None:
                    n_review = sum(review_flags)
                    granted = self._review_quota.reserve_budget(share * n_review)
                    if granted / n_review < MIN_SUBAGENT_BUDGET - 1e-9:
                        self._review_quota.release(granted)
                        return f"Error: {REVIEW_BUDGET_REFUSAL}"
                    review_budget = granted
                if not self._review_quota.try_reserve():
                    if review_budget is not None:
                        self._review_quota.release(review_budget)
                    return f"Error: {REVIEW_CAP_REFUSAL}"
            results = self._run_tasks_parallel(
                task_list, max_workers=workers, model_name=model_name or None,
                tool_profile=tool_profile, review_budget=review_budget,
                review_flags=review_flags,
            )
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

        if self._memory_tools is not None:
            tools.extend(self._memory_tools.tools())
        if allowed is not None:
            # Restricted profile: no skills, MCP servers, channel
            # dispatch, user interaction, model switching or fan-out.
            tools.append(summary)
            if decisions_tool_available():
                tools.append(make_decide_tool(self))
            return [tool for tool in tools if tool.__name__ in allowed]
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
        # agent dispatches them via run_agent(agent="cron", ...), which runs
        # kiss.agents.sorcar.cron_agent as an agent script.  Passing
        # self makes each dispatched sub-task's cost/tokens/steps fold
        # into THIS task's accounting, so the end-of-task cost shown
        # to the user includes run_agent sub-tasks (like run_parallel).
        tools.append(make_run_agent_tool(self.work_dir or "", self))
        tools.append(ask_user_question)
        tools.append(talk)
        tools.append(set_model)
        # Typed classification / routing / scoring through OpenRouter's
        # decisions endpoint (Jev).  Offered only when it can actually
        # run: an OpenRouter key is configured and the catalog has the
        # model, otherwise every call would fail and the tool would
        # only cost prompt tokens.  Its spend folds into this task's
        # accounting like ``talk``'s synthesis.
        if decisions_tool_available():
            tools.append(make_decide_tool(self))
        # No-op tool letting the model periodically condense its recent
        # activity.  Chat-webview runs react to the persisted
        # ``tool_call`` event by nesting and collapsing the preceding
        # event panels (see ``media/main.js``); outside a webview the
        # call is a harmless no-op.  The every-N-steps cadence is
        # requested by the SYSTEM.md instructions and this tool's
        # docstring only — there is no mechanical enforcement.
        tools.append(summary)
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

    def _reset_task_classification(self) -> None:
        """Forget any classification state left over from an earlier run.

        Called at the start of every top-level entry point that
        classifies (``WorktreeSorcarAgent.run``) so a run that crashed
        before :meth:`run`'s cleanup cannot leak a stale verdict into
        the next task on a reused agent instance.  Unfolded classifier
        spend from such a crashed run is dropped with the verdict:
        folding it into an unrelated later run would corrupt that run's
        accounting worse than losing the (≤ classifier-cap) telemetry.

        A verdict pre-seeded by :meth:`classify_task_for_run` survives:
        the external driver established it for the current submission,
        and internal resets (one per subtask run) must not discard it.
        """
        if self._classification_preseeded:
            return
        self._classification_attempted = False
        self._task_classification = None
        self._classifier_spend = None

    def classify_task_for_run(
        self,
        model_name: str | None,
        task: str,
        model_config: dict[str, Any] | None = None,
        enabled: bool | None = None,
    ) -> TaskClassification | None:
        """Classify *task* now and pre-seed the verdict for coming runs.

        For external drivers that must know the run's effective
        worktree mode BEFORE calling :meth:`run` — the server's task
        runner decides its main-tree claims, merge presentation, and
        persistence from ``use_worktree``, so it classifies here, sets
        ``use_worktree = use_worktree and verdict.is_development`` (the
        verdict can only demote a run that asked for a worktree, never
        promote a pinned-off one), and passes that
        value to the run.  The pre-seeded verdict is then reused by the
        run itself (worktree gating and system prompt selection) and by
        every later subtask of the same submission, so the driver and
        the agent can never disagree.  The seed lasts until the next
        ``classify_task_for_run`` call on this agent.

        Args:
            model_name: The model the run will use, possibly None.
            task: The (already substituted) task prompt about to run.
            model_config: The model configuration the run will use.
            enabled: Per-run override of the persisted
                ``classify_tasks`` setting — the ``classifyTasks`` wire
                field of the ``run`` command (the *classify_tasks*
                parameter of :func:`kiss.server.sorcar.run`).  ``True``
                forces classification on, ``False`` skips it (the run
                then behaves exactly as it would without a
                classifier), and ``None`` (the default) follows the
                config.  The ``KISS_DISABLE_TASK_CLASSIFIER``
                environment kill switch wins over any override.

        Returns:
            The task's classification, or ``None`` when classification
            is disabled or failed.
        """
        self._classification_preseeded = False
        self._reset_task_classification()
        verdict = self._classify_task_once(
            model_name, task, model_config, enabled_override=enabled,
        )
        self._classification_preseeded = True
        return verdict

    def _classify_task_once(
        self,
        model_name: str | None,
        task: str,
        model_config: dict[str, Any] | None,
        arguments: dict[str, str] | None = None,
        enabled_override: bool | None = None,
    ) -> TaskClassification | None:
        """Classify *task* at most once per run and return the verdict.

        The first call per run (see :meth:`_reset_task_classification`)
        performs the classification — when
        :func:`~kiss.agents.sorcar.task_classifier.classification_enabled`
        allows it — with the same resolved model and model config the
        main run will use, and banks the classifier's usage counters for
        :meth:`_fold_classifier_usage`.  Every later call returns the
        cached verdict, so ``WorktreeSorcarAgent.run`` (worktree
        decision) and :meth:`run` (system prompt selection) share one
        classification.

        Args:
            model_name: The caller-supplied model name, possibly None.
            task: The task prompt template about to run.
            model_config: The caller-supplied model configuration.
            arguments: The caller-supplied prompt-template arguments;
                substituted into *task* before classification so the
                classifier sees the prompt the run will actually
                execute, not the raw ``{placeholder}`` template.
            enabled_override: Per-run override of the persisted
                ``classify_tasks`` setting (see
                :meth:`classify_task_for_run`); ``None`` follows the
                config.

        Returns:
            The task's classification, or ``None`` when classification
            is disabled or failed (the run then behaves exactly as it
            would without a classifier).
        """
        if self._classification_attempted:
            return self._task_classification
        self._classification_attempted = True
        if not classification_enabled(enabled_override):
            return None
        outcome = classify_task(
            task=substitute_prompt_args(task, arguments),
            model_name=self._resolve_model_name(model_name),
            model_config=model_config,
        )
        # ONE immutable publication: the retry-stable fold key and the
        # whole spend triple become visible together with a single
        # store, so an injected stop leaves either the complete outcome
        # or no outcome — never a key paired with a zero triple that
        # the mandatory fold would consume and clear (round-6
        # finding 3).  A stop between the model call returning and
        # this store loses the whole outcome, which is the documented
        # all-or-nothing semantics: an unpublished classification was
        # never committed, so nothing partial can leak into totals.
        self._classifier_spend = _ClassifierSpend(
            f"classifier:{uuid.uuid4().hex}",
            outcome.budget_used,
            outcome.tokens_used,
            outcome.steps,
        )
        self._task_classification = outcome.classification
        if outcome.classification is not None:
            logger.info(
                "Task classified: is_simple=%s is_development=%s",
                outcome.classification.is_simple,
                outcome.classification.is_development,
            )
        return self._task_classification

    def _fold_classifier_usage(self) -> None:
        """Bank the pre-run classifier's spend into this run's totals.

        Runs from :meth:`run`'s ``finally``, AFTER ``super().run`` — the
        classifier executes before ``RelentlessAgent._reset`` zeroes the
        cumulative counters, so folding earlier would be erased.  Also
        clears the classification state so a reused agent instance
        starts its next run fresh.

        The fold is one :meth:`RelentlessAgent._attribute_usage`
        ledger append: it runs UNCONDITIONALLY on every unwind path,
        including after a stop-injected ``KeyboardInterrupt`` unwound a
        session bank mid-commit.  The append is lock-free, so it always
        terminates — an earlier lock-based design deadlocked ``run``'s
        mandatory ``finally`` forever on a lock the injection leaked.

        The fold consumes ``_classifier_spend`` — ONE immutable record
        carrying the stable transaction key and the whole spend triple
        (see :class:`_ClassifierSpend`), so it can never observe a
        torn subset of the outcome (round-6 finding 3).  The keyed
        append happens BEFORE the record is cleared, so a stop
        injected anywhere in this method is retry-safe: a stop before
        the append leaves the record intact for a later fold to
        commit; a stop after it leaves it intact too, and the retry's
        same-key re-append deduplicates on read — exactly once either
        way (round-4 finding 2b: the previous unkeyed append-once
        design lost the spend on a pre-append stop and could not be
        retried safely).
        """
        spend = self._classifier_spend
        if spend is not None:
            self._attribute_usage(
                spend.budget,
                spend.tokens,
                spend.steps,
                key=spend.key,
            )
            self._classifier_spend = None
        self._reset_task_classification()

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
        use_memory: bool | None = None,
        tool_profile: str = "",
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
                ask_user_question, talk, set_model, decide,
                run_parallel, ...) to the caller's *tools*.  Defaults
                to True.  When False
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
            use_memory: Per-run persistent-memory toggle
                (:mod:`kiss.core.memoryfield`).  ``True`` gives the
                run the ``memory_*`` tools and the ``MEMORY_PROTOCOL``
                prompt block, ``False`` withholds them, and ``None``
                (the default) falls back to the ``KISS_USE_MEMORY``
                environment variable / the stored ``use_memory``
                setting (see :func:`_memory_settings`).  A boolean
                never bypasses the hard gates of
                :func:`_memory_root_for_run` (stripped basic tools,
                Docker runs, run-to-completion CLI models, a caller
                ``model_config["system_instruction"]``).  Forwarded to
                every ``run_parallel`` sub-agent, so one override
                governs the whole task tree.
            tool_profile: Name of the tool profile this run's built-in
                toolset is cut down to — a key of :data:`TOOL_PROFILES`
                (``"full"``, ``"review"``, ``"shell"``, ``"bash"``) —
                or ``""`` (the default) to let :meth:`_tool_profile`
                decide (``full`` for a top-level task, ``review`` for a
                reviewer sub-agent).  Applies to this agent only:
                ``run_parallel`` children pick their own profile.

        Returns:
            YAML string with 'success' and 'summary' keys.

        Raises:
            ValueError: If *tool_profile* is neither ``""`` nor a key of
                :data:`TOOL_PROFILES`.
        """
        if tool_profile and tool_profile not in TOOL_PROFILES:
            raise ValueError(
                f"tool_profile must be one of {', '.join(TOOL_PROFILES)}, "
                f"got {tool_profile!r}."
            )
        self._tool_profile_name = tool_profile
        self._ask_user_question_callback = ask_user_question_callback
        self._use_web_tools = web_tools
        self._use_memory_override = use_memory
        self._is_parallel = is_parallel
        self._append_basic_tools = append_basic_tools
        # A top-level task starts with a fresh review budget; a
        # sub-agent keeps the quota the fan-out engine inherited from
        # its parent (falling back to a fresh one when spawned outside
        # the engine).
        if getattr(self, "_subagent_info", None) is None or self._review_quota is None:
            self._review_quota = ReviewQuota(budget=review_budget_for(
                max_budget if max_budget is not None else DEFAULT_MAX_BUDGET,
                prompt_template,
            ))
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
        self._memory_tools = None
        tl = getattr(printer, "_thread_local", None) if printer else None
        self._stop_event = getattr(tl, "stop_event", None) if tl else None
        try:
            # Pre-run task classification (idempotent per run:
            # WorktreeSorcarAgent.run may have classified already for
            # its worktree decision).  A task the classifier deems
            # simple — no software development, no Internet search —
            # runs on the reduced SYSTEM_LITE.md prompt; everything
            # else (including a failed or disabled classification)
            # keeps the full SYSTEM.md.  A caller-supplied
            # *base_system_prompt* still wins over both.
            classification = self._classify_task_once(
                model_name, prompt_template, model_config,
                arguments=arguments,
            )
            default_base_prompt = (
                SYSTEM_PROMPT_LITE
                if classification is not None and classification.is_simple
                else SYSTEM_PROMPT
            )
            system_instructions = (
                (self._base_system_prompt or default_base_prompt)
                + (system_prompt if system_prompt else "")
            )
            profile = self._tool_profile(prompt_template)
            # No note without a built-in toolset to cut down: a run with
            # ``append_basic_tools=False`` has only ``finish`` and the
            # caller's tools, whatever profile it names.
            if profile != "full" and self._append_basic_tools:
                allowed = TOOL_PROFILES[profile]
                assert allowed is not None
                # The docker toolset has no job registry (see the docker
                # ``Bash`` shim in :meth:`_get_tools`), so the note must
                # not promise ``bash_job`` there.
                offered = set(allowed) - ({"bash_job"} if docker_image else set())
                system_instructions += RESTRICTED_PROFILE_NOTE.format(
                    profile=profile, tools=", ".join(sorted(offered)),
                )
            memory_root = _memory_root_for_run(
                self._append_basic_tools,
                docker_image,
                self._resolve_model_name(model_name),
                caller_system_instruction=bool(
                    (model_config or {}).get("system_instruction")
                ),
                use_memory_override=use_memory,
            )
            if memory_root is not None:
                self._memory_tools = MemoryTools(memory_root)
                system_instructions += "\n\n" + MEMORY_PROTOCOL
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
            self._fold_classifier_usage()
            if self.web_use_tool:
                self.web_use_tool.close()
            self.web_use_tool = None
            self._memory_tools = None
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
    totals_out: dict[str, Any] | None = None,
    max_budget: float | None = None,
    model_config: dict[str, Any] | None = None,
    usage_monitor: _LiveUsageMonitor | None = None,
    parent_agent: Any = None,
    chat_id: str = "",
    parent_tab_id: str = "",
    base_system_prompt: str = "",
    system_prompt_suffix: str = "",
    web_tools: bool = True,
    use_memory: bool | None = None,
    tool_profile: str = "",
    child_budgets: list[float | None] | None = None,
    docker_image: str | None = None,
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
            :func:`_attribute_sub_usage`), plus the per-child spend
            list ``"budget_used_per_task"`` (same order as *tasks*).
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
        use_memory: Per-run persistent-memory toggle forwarded to each
            sub-agent's ``run`` (see :meth:`SorcarAgent.run`'s
            *use_memory*), so a parent run's explicit override governs
            its whole task tree.  ``None`` (the default) lets each
            sub-agent fall back to the environment/config default,
            exactly like the parent did.
        tool_profile: Explicit tool profile for every child (a key of
            :data:`TOOL_PROFILES`).  ``""`` (default) lets each child
            pick its own: ``review`` for reviewer-marked children when
            ``DEFAULT_CONFIG.tool_profiles`` is on, ``full`` otherwise.
        child_budgets: Per-child ``max_budget`` overrides (same length
            as *tasks*); an entry of ``None`` falls back to *max_budget*.
            Used to clip reviewer children to the review allowance
            without touching their non-review siblings.
        docker_image: ``docker_image`` for every child (normally the
            parent's live container as ``container:<id>``, so the
            children's tools act inside the same container); ``None``
            runs the children's tools on the host.

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
    parent_is_reviewer = bool(
        (getattr(parent_agent, "_subagent_info", None) or {}).get("reviewer")
    )
    parent_quota = getattr(parent_agent, "_review_quota", None)
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

    # The whole fan-out's own stop signal, chained to the parent's.  It
    # is set when the user presses the run_parallel panel's own Stop
    # button (ToolCallInterrupted lands in the parent's wait): the
    # parent's stop event stays unset then, so without this the
    # children would keep running, and spending, after the fan-out the
    # user just stopped had returned.  Any other reason the parent
    # unwinds leaves the children alone (they are abandoned, their
    # spend reclaimed later), exactly as before.
    fanout_stop_event = _SubagentStopEvent(parent_stop_event)

    def _run_single(args: tuple[int, str]) -> str:
        idx, task = args
        if DEFAULT_CONFIG.dispatch_path_rewrite:
            # A parent in a worktree keeps writing parent-repo paths into
            # its children's tasks; the children's Bash guard would then
            # refuse every such command.
            task = rewrite_parent_repo_paths(task, work_dir)
        # A per-child event, chained to the fan-out's and through it to
        # the parent's: stopping ONE sub-agent must not stop the parent
        # or its siblings, while a parent stop (or an abandoned
        # fan-out) still reaches every child (_SubagentStopEvent).
        sub_stop_event = _SubagentStopEvent(fanout_stop_event)
        tl = getattr(printer, "_thread_local", None) if printer else None
        if tl is not None:
            tl.stop_event = sub_stop_event
        agent = ChatSorcarAgent(f"Parallel-{task[:40]}")
        # The parent's review budget, BEFORE run() (which keeps an
        # inherited quota): the whole in-process tree shares one cap.
        agent._review_quota = parent_quota
        # Decided here, on the bare task text: the child's own prompt
        # will carry the whole chat history, whose earlier tasks would
        # make every implementation-word heuristic fire.
        reviewer = parent_is_reviewer or is_review_task(task)
        child_profile = tool_profile or (
            "review"
            if DEFAULT_CONFIG.tool_profiles and reviewer and not is_implementation_task(task)
            else "full"
        )
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
            # Inherited down the whole sub-tree so a reviewer cannot
            # launch reviewers through an intermediate helper child.
            "reviewer": reviewer,
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
                max_budget=(
                    child_budgets[idx] if child_budgets and child_budgets[idx] is not None
                    else max_budget
                ),
                model_config=model_config,
                base_system_prompt=base_system_prompt,
                system_prompt=system_prompt_suffix or None,
                web_tools=web_tools,
                use_memory=use_memory,
                tool_profile=child_profile,
                docker_image=docker_image,
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
                    _notify_subagent_done(
                        printer, _persisted_task_id(agent), sub_tab_id,
                        model_name or "",
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
        try:
            # Submission happens INSIDE the guarded region, appending
            # one future at a time: a stop injected while the tasks are
            # still being submitted must see every child submitted so
            # far, so the abandon path below runs for them instead of
            # the ``finally`` joining untracked running children with
            # ``shutdown(wait=True)``.
            for item in enumerate(tasks):
                futures.append(pool.submit(_run_single, item))
            results = _await_subagents(futures, parent_stop_event)
        except BaseException as exc:
            abandoned = any(not f.done() for f in futures)
            if abandoned and isinstance(exc, ToolCallInterrupted):
                fanout_stop_event.set()
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
                totals_out["budget_used_per_task"] = [u[0] for u in sub_usage]
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
