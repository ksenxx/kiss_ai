# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sorcar agent with both coding tools and browser automation."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar.cli_helpers import (
    _DEFAULT_TASK as _DEFAULT_TASK,
)
from kiss.agents.sorcar.cli_helpers import (
    _resolve_task as _resolve_task,
)
from kiss.agents.sorcar.cli_helpers import (
    cli_ask_user_question as cli_ask_user_question,
)
from kiss.agents.sorcar.persistence import _load_last_model
from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.agents.sorcar.skills import make_skill_tool
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.core.base import SYSTEM_PROMPT
from kiss.core.kiss_error import BudgetExceededError
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
) -> None:
    """Fill in the spend of children that never got to report it.

    A child fills its own ``sub_usage`` slot in its ``finally``, so a
    child the parent abandoned (see :func:`_await_subagents`) would leave
    a zero there and its cost, tokens and steps would silently vanish
    from the parent task's totals.  Reading the live figures off the
    child's agent recovers everything it had spent up to this instant —
    without waiting for it, which is the whole point of abandoning it.
    A live read of a still-running child can lag its true spend slightly
    (it may land mid-handoff between executor sessions), which is why a
    child that finishes in the meantime keeps its own final figure.

    Args:
        futures: One future per fanned-out sub-agent, in task order.
        sub_agents: The children's agents, in the same order; entries are
            ``None`` for children that never started.
        sub_usage: Per-child ``(cost, tokens, steps)`` slots, updated in
            place for unfinished children only.
    """
    for idx, future in enumerate(futures):
        agent = sub_agents[idx]
        if future.done() or agent is None:
            continue
        live = _live_agent_usage(agent)
        # A worker writes its own slot BEFORE its future completes, so a
        # future that finished while this live read was in flight has
        # already published a strictly better figure: keep it.
        if future.done():
            continue
        sub_usage[idx] = live


def _live_agent_usage(agent: Any) -> tuple[float, int, int]:
    """Return live ``(budget, tokens, steps)`` for *agent*, including its
    in-flight executor session.

    :class:`~kiss.agents.sorcar.relentless_agent.RelentlessAgent` folds a session
    executor's spend into the agent's totals only when the session ends,
    so mid-session the live spend is visible only on
    ``agent._current_executor``.
    """
    budget, tokens, steps = _agent_usage(agent)
    executor = getattr(agent, "_current_executor", None)
    if executor is not None:
        budget += float(getattr(executor, "budget_used", 0.0) or 0.0)
        tokens += int(getattr(executor, "total_tokens_used", 0) or 0)
        steps += int(getattr(executor, "step_count", 0) or 0)
    return budget, tokens, steps


class _LiveUsageMonitor:
    """Streams the parent task's live cumulative usage while parallel
    sub-agents run.

    Between the moment ``run_parallel`` blocks the parent's turn and the
    moment :func:`_attribute_sub_usage` folds the finished sub-agents'
    spend back into the parent, nothing else emits ``usage_info`` on the
    PARENT task — the cost/tokens header (chat webview top bar, sorcar
    CLI interactive) would otherwise show a stale figure that excludes
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

        Joining guarantees no further emission can race with the
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
        while not self._done.wait(self._interval):
            try:
                self._emit()
            except Exception:
                logger.debug("Live usage emission failed", exc_info=True)

    def _emit(self) -> None:
        """Broadcast a parent-task ``usage_info`` when the totals changed."""
        executor = getattr(self._parent, "_current_executor", None)
        if executor is not None:
            budget = float(getattr(executor, "budget_used", 0.0) or 0.0)
            tokens = int(getattr(executor, "total_tokens_used", 0) or 0)
            steps = int(getattr(executor, "step_count", 0) or 0)
        else:
            budget, tokens, steps = 0.0, 0, 0
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
    """
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
        self._is_parallel: bool = False

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

    def _run_tasks_parallel(
        self,
        tasks: list[str],
        max_workers: int | None = None,
    ) -> list[str]:
        """Execute multiple independent tasks concurrently using parallel agents.

        Each task gets its own ``ChatSorcarAgent`` instance.  Subclasses can
        override this method to change the agent type or pass extra context
        (e.g. ``ChatSorcarAgent`` propagates ``chat_id``).

        This method is a pure parallel executor.  It has no knowledge of
        backend task ids or any frontend concepts (tabs, ``new_tab``
        broadcasts, etc.).  Any sub-agent-specific frontend behaviour is
        owned by the sub-agent itself — see
        :meth:`ChatSorcarAgent.run`, which self-broadcasts a ``new_tab``
        message whenever it detects ``self._subagent_info`` is set.

        Args:
            tasks: List of self-contained task description strings.
            max_workers: Maximum concurrent threads (``None`` = auto).

        Returns:
            List of YAML result strings in the same order as *tasks*.
        """
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

            def Bash(command: str, description: str) -> str:  # noqa: N802
                """Run a command in the task's Docker container."""
                return self._docker_bash(command, description)

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

        Args:
            model_name: The model the agent just switched to.
        """
        show = getattr(self.printer, "broadcast_agent_model_pick", None)
        if not callable(show):
            return
        try:
            show(model_name, getattr(self, "_tab_id", "") or "")
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
        all_tools = self._get_tools() + tools
        if getattr(self, "_tab_id", None):
            self.pre_step_hook = self._drain_pending_user_messages
            self.tool_call_guard = self._block_finish_when_user_message_pending
        else:
            self.pre_step_hook = None
            self.tool_call_guard = None
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
        resolved_model = model_name or _load_last_model() or get_default_model()
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
        is_parallel: bool = False,
        verbose: bool | None = None,
        current_editor_file: str | None = None,
        attachments: list[Attachment] | None = None,
        ask_user_question_callback: Callable[[str], str] | None = None,
    ) -> str:
        """Run the assistant agent with coding tools and browser automation.

        Args:
            model_name: LLM model to use. Defaults to config value.
            prompt_template: Task prompt template with format placeholders.
            arguments: Dictionary of values to fill prompt_template placeholders.
            system_prompt: system prompt to be appended to the actual system prompt
            tools: List of tools to be added in addition to bash and web tools.
            max_steps: Maximum steps per sub-session. Defaults to 10000.
            max_budget: Maximum budget in USD. Defaults to config value.
            work_dir: Working directory for the agent. Defaults to artifact_dir/kiss_workdir.
            printer: Printer instance for output display.
            max_sub_sessions: Maximum continuation sub-sessions. Defaults to config value.
            docker_image: Docker image name to run tools inside a container.
            web_tools: Whether to include browser/web tools. Defaults to True.
                Set to False for terminal-only environments.
            is_parallel: Whether to include the run_parallel tool. Defaults to False.
                When True, the agent can spawn parallel sub-agents for independent tasks.
            verbose: Whether to print output to console. Defaults to config verbose setting.
            current_editor_file: Path to the currently active editor file, appended to prompt.
            attachments: Optional file attachments (images, PDFs) for the initial prompt.
            ask_user_question_callback: Optional callback used by the ask_user_question
                tool to collect a text response from the user.

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        self._ask_user_question_callback = ask_user_question_callback
        self._use_web_tools = web_tools
        self._is_parallel = is_parallel
        self.web_use_tool = None
        tl = getattr(printer, "_thread_local", None) if printer else None
        self._stop_event = getattr(tl, "stop_event", None) if tl else None
        try:
            system_instructions = (
                SYSTEM_PROMPT
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
        the owning :class:`_RunningAgentState`'s
        ``pending_user_messages`` list under
        :attr:`_RunningAgentState._registry_lock` (to keep the drain
        atomic against concurrent ``appendUserMessage`` commands from
        the frontend) and pushes each entry into *model*'s
        conversation as a ``user`` role message.  Each entry is
        wrapped as ``User says: <message>. Take the message into
        account and finish your task.`` so the model treats it as a
        mid-task steering instruction rather than a bare trajectory
        line.  The list is emptied on every drain so the same queued
        message is never injected twice.

        Messages whose ``prompt`` echo could not be attributed to a
        task id at queueing time (``unattributed_prompt_echoes`` —
        the narrow window between ``run()`` entry and ``_add_task``)
        get a durable copy HERE: a ``recordOnly`` broadcast from the
        agent thread, where the printer's thread-local task id names
        the task that actually consumed the message, so the echo is
        recorded and persisted into the correct trajectory instead of
        being lost on replay.  The copy is NOT re-sent live (the
        command handler already emitted a transient echo at queueing
        time — see ``_echo_injected_prompt`` — so a live re-send
        would render a duplicate prompt panel).

        Args:
            model: The live model whose conversation receives the
                queued user messages.
        """
        from kiss.agents.sorcar.running_agent_state import _RunningAgentState

        tab_id = getattr(self, "_tab_id", "") or ""
        if not tab_id:
            return
        with _RunningAgentState._registry_lock:
            tab = _RunningAgentState.running_agent_states.get(tab_id)
            if tab is None or (tab.agent is not None and tab.agent is not self):
                # Ownership check: register() explicitly allows a different
                # state to replace this key (tab reuse), so a stale agent
                # must never consume the replacement's queued input.
                return
            if not tab.pending_user_messages and not tab.unattributed_prompt_echoes:
                return
            queued = list(tab.pending_user_messages)
            tab.pending_user_messages.clear()
            deferred = list(tab.unattributed_prompt_echoes)
            tab.unattributed_prompt_echoes.clear()
        for msg in queued:
            model.add_message_to_conversation(
                "user",
                f"User says: {msg}. "
                "Take the message into account and finish your task.",
            )
        # The recordOnly echoes are emitted AFTER the queued messages
        # entered the model conversation, and each broadcast is guarded:
        # a broken printer must never lose the (already cleared) steering
        # input or abort the task from this best-effort persistence hook.
        if deferred:
            broadcast = getattr(
                getattr(self, "printer", None), "broadcast", None,
            )
            if broadcast is not None:
                for msg in deferred:
                    try:
                        broadcast({
                            "type": "prompt",
                            "text": msg,
                            "recordOnly": True,
                        })
                    except Exception:
                        # Requeue so the durable echo is retried on the
                        # next drain instead of being lost forever.
                        logger.debug(
                            "recordOnly prompt echo broadcast failed",
                            exc_info=True,
                        )
                        with _RunningAgentState._registry_lock:
                            owner = _RunningAgentState.running_agent_states.get(
                                tab_id
                            )
                            if owner is not None and (
                                owner.agent is None or owner.agent is self
                            ):
                                owner.unattributed_prompt_echoes.append(msg)

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
        tab_id = getattr(self, "_tab_id", "") or ""
        if not tab_id:
            return None
        from kiss.agents.sorcar.running_agent_state import _RunningAgentState

        with _RunningAgentState._registry_lock:
            tab = _RunningAgentState.running_agent_states.get(tab_id)
            pending = (
                tab is not None
                and (tab.agent is None or tab.agent is self)
                and bool(tab.pending_user_messages)
            )
        if not pending:
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
) -> list[str]:
    """Execute multiple SorcarAgent tasks concurrently using threads.

    Each task gets its own ``ChatSorcarAgent`` instance and runs in a
    separate thread via :class:`~concurrent.futures.ThreadPoolExecutor`.
    This is ideal for I/O-bound workloads (LLM API calls, network
    requests) where the GIL is released during I/O waits.

    This helper is a pure parallel executor: it has no knowledge of
    backend task ids or any frontend concepts.  It simply marks each
    spawned agent as a sub-agent (via ``_subagent_info``) and the
    sub-agent itself owns any sub-agent-specific behaviour (such as
    broadcasting ``new_tab`` to a browser-based frontend) inside its
    own ``run()`` method.

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
    # Published as soon as each child exists so an abandoned child's
    # spend can still be read (see _collect_unfinished_usage).
    sub_agents: list[Any] = [None] * len(tasks)

    parent_tl = getattr(printer, "_thread_local", None) if printer else None
    parent_key = getattr(parent_tl, "task_id", "") if parent_tl else ""
    parent_stop_event = getattr(parent_tl, "stop_event", None) if parent_tl else None

    def _run_single(args: tuple[int, str]) -> str:
        idx, task = args
        tl = getattr(printer, "_thread_local", None) if printer else None
        if tl is not None:
            tl.stop_event = parent_stop_event
        agent = ChatSorcarAgent(f"Parallel-{task[:40]}")
        sub_agents[idx] = agent
        agent._subagent_info = {"parent_task_id": "", "parent_tab_id": ""}
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
            )
            return result
        except Exception as exc:
            return _yaml_failure(exc)
        finally:
            # _live_agent_usage (not _agent_usage): an interrupted child
            # never folds its in-flight executor session's spend into the
            # agent totals, so the folded-only read would undercount it.
            sub_usage[idx] = _live_agent_usage(agent)
            if printer is not None and parent_key:
                _broadcast_subagent_done(
                    printer,
                    [f"task-{parent_key}__sub_{idx}"],
                    model_name or "",
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
        _collect_unfinished_usage(futures, sub_agents, sub_usage)
        if totals_out is not None:
            totals_out["budget_used"] = sum(u[0] for u in sub_usage)
            totals_out["total_tokens_used"] = sum(u[1] for u in sub_usage)
            totals_out["total_steps"] = sum(u[2] for u in sub_usage)
    return results
