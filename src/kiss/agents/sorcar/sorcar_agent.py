# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sorcar agent with both coding tools and browser automation."""

from __future__ import annotations

import functools
import json
import logging
import os
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

# CLI plumbing lives in cli_helpers; re-exported here for backwards
# compatibility (tests and callers import these from this module).
from kiss.agents.sorcar.cli_helpers import (
    _DEFAULT_TASK as _DEFAULT_TASK,
)
from kiss.agents.sorcar.cli_helpers import (
    _resolve_task as _resolve_task,
)
from kiss.agents.sorcar.cli_helpers import (
    cli_ask_user_question as cli_ask_user_question,
)
from kiss.agents.sorcar.code_graph import intercept_grep_hint
from kiss.agents.sorcar.persistence import _load_last_model
from kiss.agents.sorcar.skills import make_skill_tool
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.core.base import SYSTEM_PROMPT
from kiss.core.kiss_error import BudgetExceededError
from kiss.core.models.model import Attachment
from kiss.core.models.model_info import (
    MODEL_INFO,
    OPENAI_COMPATIBLE_PROVIDERS,
    get_default_model,
)
from kiss.core.models.model_info import model as _model_factory
from kiss.core.printer import Printer
from kiss.core.relentless_agent import RelentlessAgent
from kiss.core.useful_tools import UsefulTools

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
            message" toast without a follow-up.  The "committed"
            hook is also not invoked when ``commit_staged`` returns
            ``False`` after *message_fn* (e.g. pre-commit hook
            rejected the commit).

            All ``notify_fn`` exceptions are swallowed so a broken
            UI hook can never block the commit itself.

    Returns:
        True if a commit was created, False if nothing to commit.
    """
    from kiss.agents.sorcar.git_worktree import GitWorktreeOps

    GitWorktreeOps.stage_all(commit_dir)
    # Short-circuit cleanly when there is nothing to commit: no
    # "generating" toast (it would be misleading because no commit
    # will happen), no LLM call (saves tokens), no follow-up
    # "committed" toast.  Late-arriving files don't matter here
    # because there's no slow message_fn window to race against.
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
    # Re-stage immediately before committing to capture any files
    # that materialized during the (typically slow) *message_fn*
    # call — see the docstring above for the production race this
    # closes.  Cheap when nothing changed (``git add -A`` is a no-op
    # against an unchanged tree).
    GitWorktreeOps.stage_all(commit_dir)
    committed = GitWorktreeOps.commit_staged(commit_dir, msg)
    if committed:
        _safe_notify(notify_fn, "committed", _commit_subject(msg))
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


def _broadcast_subagent_done(printer: Any, tab_ids: list[str]) -> None:
    """Broadcast ``subagentDone`` for each tab id so the frontend can
    stop the running indicator on the sub-agent tab.  Errors are
    swallowed (the broadcast is best-effort UI signalling)."""
    broadcast = getattr(printer, "broadcast", None)
    if broadcast is None:
        return
    for vid in tab_ids:
        try:
            broadcast({"type": "subagentDone", "tab_id": vid, "tabId": ""})
        except Exception:
            pass


def _live_agent_usage(agent: Any) -> tuple[float, int, int]:
    """Return live ``(budget, tokens, steps)`` for *agent*, including its
    in-flight executor session.

    :class:`~kiss.core.relentless_agent.RelentlessAgent` folds a session
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
        self._last_emitted: tuple[str, int, int] | None = None
        # Capture the parent's thread-local task id HERE, in the calling
        # thread — the monitor thread must emit on the PARENT task's
        # stream (broadcast tags events and resolves usage offsets by
        # the emitting thread's task key).
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
                # One misbehaving sub-agent must not blind the whole
                # header to every other sub-agent's live spend.
                logger.debug("Live usage poll failed", exc_info=True)
                continue
            budget += sub_budget
            tokens += sub_tokens
            steps += sub_steps
        cost = f"${budget:.4f}"
        snapshot = (cost, tokens, steps)
        if snapshot == self._last_emitted:
            return
        self._last_emitted = snapshot
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


# Base URLs the ``model()`` factory itself assigns when routing a model
# name to its provider.  ``set_model`` must NOT carry one of these into the
# new model's ``model_config``: the factory bypasses provider routing
# whenever ``model_config`` contains ``base_url``, so restoring a default
# endpoint would e.g. point ``claude-*`` at ``api.openai.com`` after an
# OpenAI -> Anthropic switch.  Only *custom* endpoints (local gateways,
# proxies) are worth preserving across a switch.  Derived from the vendor
# registry so newly registered vendors are covered automatically.
_FACTORY_DEFAULT_BASE_URLS: frozenset[str] = frozenset(
    provider.base_url.rstrip("/") for provider in OPENAI_COMPATIBLE_PROVIDERS
)


# Attachment MIME-type prefixes and the human-readable labels used when
# describing attachments in the initial prompt, in display order.
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


def _make_plain_bash_tool(
    agent: SorcarAgent, useful_tools: UsefulTools,
) -> Callable[..., str]:
    """Build the non-Docker ``Bash`` tool with query-before-grep interception.

    The returned function answers grep-like commands from the built code
    graph (via :func:`intercept_grep_hint`, at most once per distinct
    hint) and otherwise delegates to :meth:`UsefulTools.Bash`.

    Args:
        agent: The owning agent (``agent.work_dir`` locates the graph).
        useful_tools: The tool collection executing real commands.

    Returns:
        The ``Bash`` tool function to register with the model.
    """
    hints_seen: set[str] = set()

    # ``wraps`` copies the delegate's name, docstring, and (crucially)
    # its concrete parameter annotations, so the model-visible tool
    # schema is byte-identical to registering ``useful_tools.Bash``
    # directly (this module's postponed annotations would otherwise
    # degrade ``timeout_seconds``/``max_output_chars`` to strings).
    @functools.wraps(useful_tools.Bash)
    def Bash(  # noqa: N802
        command: str,
        description: str,
        timeout_seconds: float = 300,
        max_output_chars: int = 50000,
    ) -> str:
        hint = intercept_grep_hint(command, agent.work_dir, hints_seen)
        if hint is not None:
            return hint
        return useful_tools.Bash(
            command, description, timeout_seconds, max_output_chars,
        )

    return Bash


class SorcarAgent(RelentlessAgent):
    """Agent with both coding tools and browser automation for web + code tasks."""

    # True only on subclasses that isolate every task in a git worktree
    # (see :class:`~kiss.agents.sorcar.worktree_sorcar_agent.WorktreeSorcarAgent`).
    uses_worktree: bool = False

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.web_use_tool: WebUseTool | None = None
        # NOT redundant: the base class only sets ``docker_manager`` in
        # ``_reset`` (called from ``run``); tools can be built earlier.
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
        # Keep one equal share for the parent to consume the sub-agent
        # results and issue its terminal ``finish`` call.  With no tasks
        # there is no fan-out and the full remainder stays with the parent.
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
        # Live-stream the parent task's cumulative usage (parent session
        # + all sub-agents) while the fan-out runs, so the cost/tokens
        # header stays accurate at every turn instead of freezing until
        # every sub-agent completes.  Stopped (joined) BEFORE
        # ``_attribute_sub_usage`` bumps the printer offsets, so a late
        # emission can never double-count the sub-agents' spend.
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
                # Cap every sub-agent to a fair share of THIS task's
                # remaining budget — without it each sub-agent would default
                # to the full configured budget and a single sub-agent could
                # spend the entire budget of the main task.  One equal share
                # remains reserved for the parent to process results and finish.
                max_budget=self._subagent_budget_share(len(tasks)),
                # Sub-agents must talk to the same provider endpoint as the
                # parent (custom ``base_url``/``api_key`` routing).
                model_config=getattr(self, "model_config", None),
            )
        finally:
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
            # ``talkId`` uniquely identifies this utterance.  The
            # printer fans out one tab-stamped copy per subscribed
            # viewer tab and delivers every copy to every connected
            # client; clients dedupe by talkId so each device speaks
            # the utterance exactly once (never twice).
            payload: dict[str, Any] = {
                "type": "talk",
                "language": language,
                "text": text,
                "emotion": emotion,
                "talkId": uuid.uuid4().hex,
            }
            # Synthesize a far more natural voice server-side with a
            # GPT audio model (gpt-audio-1.5) and ship the MP3 inside
            # the event; every interface (VS Code webview, browser tab,
            # iOS webapp) plays it directly and stays silent when
            # synthesis failed or audio playback is unavailable/blocked
            # on the device (the CLI terminal alone falls back to the
            # system TTS command).
            tts_usage: dict[str, Any] = {}
            try:
                from kiss.core.speech_synthesis import synthesize_talk_audio

                synthesized = synthesize_talk_audio(
                    text, language, emotion, usage_out=tts_usage,
                )
            except Exception:
                synthesized = None
            # Fold the synthesis agent's real spend into THIS task's
            # accounting — otherwise every talk() call's TTS cost is
            # missing from the reported per-task cost.
            _attribute_tts_usage(self, tts_usage)
            if synthesized:
                payload["audioB64"], payload["audioMime"] = synthesized
                # Attach the clip to the recorded/persisted ``talk``
                # ``tool_call`` event too: that event was emitted (and
                # persisted) BEFORE this tool ran, so it carries no
                # audio — and demo-mode replay plays exactly the
                # persisted ``extras.audioB64`` (the synthesis
                # fallback is gone).  Without this every replayed
                # narration would be silent.
                attach = getattr(self.printer, "attach_talk_audio", None)
                if callable(attach):
                    try:
                        attach(payload["audioB64"], payload["audioMime"])
                    except Exception:
                        logger.exception(
                            "failed to attach talk audio to the recorded"
                            " tool_call event"
                        )
            broadcast(payload)
            return f"Spoke to the user in language {language!r}."

        if self.docker_manager:
            from kiss.core.docker_tools import DockerTools

            docker_tools = DockerTools(self._docker_bash)
            code_graph_hints_seen: set[str] = set()

            def Bash(command: str, description: str) -> str:  # noqa: N802
                """Run a command in Docker, preferring a code-graph answer."""
                hint = intercept_grep_hint(
                    command, self.work_dir, code_graph_hints_seen,
                )
                if hint is not None:
                    return hint
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
                _make_plain_bash_tool(self, useful_tools),
                useful_tools.Read, useful_tools.Edit, useful_tools.Write,
            ]
        if self._use_web_tools and self.web_use_tool is None:
            if getattr(self, "_subagent_info", None) is not None:
                # Parallel sub-agents must never share (or escalate) the
                # user's persistent browser profile: each profile-lock
                # escalation opened one more visible Chromium window that
                # stayed open for the sub-agent's whole lifetime.  Give
                # sub-agents a headless browser in a throwaway profile
                # that close() (invoked in this agent's run() finally
                # block) deletes.
                self.web_use_tool = WebUseTool(
                    work_dir=self.work_dir, headless=True, ephemeral=True,
                )
            else:
                self.web_use_tool = WebUseTool(work_dir=self.work_dir)
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

            This does not persist ``last_model`` and therefore cannot
            change the user-facing model picker default.  Only an
            explicit picker selection should update that preference.

            Args:
                model_name: New LLM model name (for example
                    ``"gpt-5.5"``, ``"claude-sonnet-4-8"``,
                    ``"gemini-3.5-flash"``).

            Returns:
                A human-readable confirmation string describing the
                change (or a "no change" message when the requested
                model is already active).
            """
            # The model actually making LLM calls lives on the inner
            # per-session executor (``RelentlessAgent.perform_task``
            # creates a fresh ``KISSAgent`` per sub-session and stores
            # it in ``self._current_executor``); the relentless parent
            # (``self``) never instantiates a ``model`` of its own.
            # Target the executor so the swap lands on the instance the
            # ReAct loop reads on every step — otherwise the change
            # would silently defer to the NEXT sub-session, which most
            # runs never start.
            target = getattr(self, "_current_executor", None) or self
            old_model = getattr(target, "model", None)
            if old_model is None:
                self.model_name = model_name
                return (
                    f"Model deferred-changed to {model_name} "
                    "(no live model yet)."
                )
            if old_model.model_name == model_name:
                return f"Model is already {model_name}; no change."

            # Reconstruct ``model_config`` for the factory.  The
            # ``OpenAICompatibleModel`` factory path strips ``base_url``
            # and ``api_key`` from ``model_config`` before storing it,
            # so we restore them from the live model's attributes so the
            # new model lands on the same endpoint — but ONLY for custom
            # endpoints.  A factory-default ``base_url`` (e.g.
            # ``https://api.openai.com/v1`` on a routed ``gpt-*`` model)
            # must not be carried over: ``model()`` bypasses provider
            # routing whenever the config contains ``base_url``, which
            # would wrongly build an OpenAI-compatible client for a
            # ``claude-*``/``gemini-*`` name on a cross-provider switch.
            new_config: dict[str, Any] = dict(old_model.model_config or {})
            # Drop a factory-injected ``reasoning_effort`` default.  The
            # ``model()`` factory auto-defaults ``reasoning_effort`` into
            # ``model_config`` for models whose MODEL_INFO entry declares a
            # ``thinking`` level (e.g. ``"high"`` for gpt-5.5); such a value
            # describes the OLD model, not a user choice.  Carrying it over
            # breaks switches to non-reasoning models (OpenAI rejects
            # ``reasoning_effort`` for e.g. gpt-4o with a 400) and would
            # override the new model's own default (e.g. the ``-xhigh``
            # alias's level leaking onto the base model).  The new model
            # re-defaults its own level on construction.  A user-explicit
            # non-default effort is preserved.
            old_info = MODEL_INFO.get(old_model.model_name)
            if (
                old_info is not None
                and new_config.get("reasoning_effort") == old_info.thinking
            ):
                new_config.pop("reasoning_effort", None)
            old_base_url = getattr(old_model, "base_url", None)
            old_api_key = getattr(old_model, "api_key", None)
            if (
                old_base_url
                and old_base_url.rstrip("/") not in _FACTORY_DEFAULT_BASE_URLS
                and "base_url" not in new_config
            ):
                new_config["base_url"] = old_base_url
                if old_api_key is not None:
                    new_config["api_key"] = old_api_key
            new_model = _model_factory(
                model_name,
                model_config=new_config or None,
                token_callback=old_model.token_callback,
                thinking_callback=old_model.thinking_callback,
            )
            # The provider client (``self.client``) is created only
            # inside ``Model.initialize`` — a freshly-constructed
            # model has ``client = None`` and its first ``generate``
            # would crash with ``'NoneType' object has no attribute
            # 'chat'``.  Initialize with a throwaway prompt (building
            # the client), then hand off the live conversation below,
            # which discards that placeholder message.
            new_model.initialize("")
            # Carry over the live conversation state so the next LLM
            # call resumes from the same point.  Each provider stores a
            # different native format: OpenAI-schema models store
            # ``tool_calls`` with JSON-string arguments plus
            # ``role="tool"`` / ``role="system"`` messages;
            # AnthropicModel stores Anthropic Messages-format block
            # lists (``thinking`` / ``tool_use`` / ``tool_result``);
            # GeminiModel stores OpenAI-like messages but with dict
            # tool-call arguments and optional ``attachments`` keys.  A
            # direct hand-off is safe in every direction because each
            # model class converts the foreign format at request time:
            # ``_normalize_conversation_for_api`` in
            # ``OpenAICompatibleModel`` / ``AnthropicModel``,
            # ``_convert_conversation_to_gemini_contents`` in
            # ``GeminiModel`` (system text is hoisted into the
            # provider's top-level system parameter), and
            # ``flatten_content_to_text`` in the CLI-backed models.
            new_model.conversation = old_model.conversation
            new_model.usage_info_for_messages = old_model.usage_info_for_messages

            previous_name = old_model.model_name
            target.model = new_model  # type: ignore[attr-defined, union-attr]
            target.model_name = model_name
            # Keep the relentless parent's ``model_name`` in sync so
            # any subsequent sub-session (which re-reads it when
            # constructing its executor) stays on the new model.
            self.model_name = model_name
            # Rebuild the cached tools schema against the new model
            # (different providers can produce slightly different
            # schemas — e.g. Anthropic vs OpenAI).  The schema cache
            # lives on the same object as the swapped model (the
            # executor during a live run).
            if getattr(target, "function_map", None):
                target._cached_tools_schema = new_model._build_openai_tools_schema(  # type: ignore[attr-defined, union-attr]
                    target.function_map,
                )
            return f"Model changed from {previous_name} to {model_name}."

        # Agent Skills (https://agentskills.io): the tool's docstring
        # carries only each skill's name + description (token-efficient
        # progressive disclosure); full SKILL.md bodies load on demand.
        # No tool is registered when no skills are discovered.
        skill_tool = make_skill_tool(self.work_dir or ".")
        if skill_tool is not None:
            tools.append(skill_tool)
        # code_graph: local tree-sitter knowledge graph of the work dir
        # (query/path/explain instead of grep).  Registered only when the
        # optional tree-sitter grammars are importable; a failure here
        # must never break agent startup.
        try:
            from kiss.agents.sorcar.code_graph import make_code_graph_tool

            code_graph_tool = make_code_graph_tool(self.work_dir or ".")
            if code_graph_tool is not None:
                tools.append(code_graph_tool)
        except Exception:
            logger.warning("code_graph tool setup failed", exc_info=True)
        # MCP servers: every tool of every configured server becomes a
        # ``<server>_<tool>`` function (filtered by the
        # ``mcp_permissions`` wildcard rules).  A broken server is
        # logged and skipped so it can never break agent startup.
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
        # Wire up the pre-step hook so user prompts queued via the VS
        # Code frontend's ``appendUserMessage`` command while this task
        # is running get injected into the live model's conversation as
        # additional ``user`` messages immediately before the next model
        # call.  This MUST happen here (after ``RelentlessAgent.run`` has
        # already called ``_reset``, which clears ``pre_step_hook``) and
        # before ``super().perform_task`` runs the per-session executor
        # loop — that loop copies ``self.pre_step_hook`` onto each inner
        # executor.  Only meaningful when this agent has been bound to a
        # frontend tab (``_tab_id`` is set by
        # :meth:`_TaskRunnerMixin._run_task_inner`).
        if getattr(self, "_tab_id", None):
            self.pre_step_hook = self._drain_pending_user_messages
        else:
            self.pre_step_hook = None
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
        # Remember the model this task was LAUNCHED with.  The
        # ``set_model`` tool mutates ``self.model_name`` mid-task
        # (deliberately — subsequent sub-sessions of the SAME task must
        # stay on the switched model), so end-of-task persistence
        # (e.g. ``ChatSorcarAgent.run``'s final task-history save) must
        # read this attribute instead of ``self.model_name`` — the
        # recorded model of a task, like every other global model
        # preference, must never change because an agent switched its
        # own model while running (see INVARIANTS.md #217).
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
            max_steps: Maximum steps per sub-session. Defaults to config value.
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
        # NOTE: the pending-user-messages pre-step hook is wired up in
        # :meth:`perform_task` (which runs *after* ``RelentlessAgent.run``
        # calls ``_reset``).  Installing it here would be useless because
        # ``RelentlessAgent._reset`` — invoked by ``super().run`` below —
        # resets ``self.pre_step_hook`` back to ``None`` before the
        # per-session executor loop reads it.
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
            if tab is None or not tab.pending_user_messages:
                return
            queued = list(tab.pending_user_messages)
            tab.pending_user_messages.clear()
            deferred = list(tab.unattributed_prompt_echoes)
            tab.unattributed_prompt_echoes.clear()
        if deferred:
            broadcast = getattr(
                getattr(self, "printer", None), "broadcast", None,
            )
            if broadcast is not None:
                for msg in deferred:
                    broadcast({
                        "type": "prompt",
                        "text": msg,
                        "recordOnly": True,
                    })
        for msg in queued:
            model.add_message_to_conversation(
                "user",
                f"User says: {msg}. "
                "Take the message into account and finish your task.",
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
                # A JSON empty list means zero tasks (NOT one task whose
                # text is "[]"); non-string elements (e.g. '[1, 2]') are
                # coerced to one task string per element.
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

    # Local import: ``chat_sorcar_agent`` imports from this module, so a
    # top-level import would be circular.
    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

    # Per-sub-agent usage so the caller can aggregate it back into the
    # parent agent's accounting.  Each entry is a tuple of
    # ``(budget_used, total_tokens_used, total_steps)``.
    sub_usage: list[tuple[float, int, int]] = [(0.0, 0, 0)] * len(tasks)

    # Capture the parent's thread-local ``task_id`` HERE, in the calling
    # thread.  ``printer._thread_local`` is a real ``threading.local``,
    # so reading it inside a worker thread would never see the parent
    # thread's value (and ``ChatSorcarAgent.run`` clears the worker's
    # own id before ``_run_single``'s ``finally`` runs) — the
    # ``subagentDone`` broadcast would then never fire.
    parent_tl = getattr(printer, "_thread_local", None) if printer else None
    parent_key = getattr(parent_tl, "task_id", "") if parent_tl else ""
    # Capture the parent's stop_event HERE too (same thread-local
    # reasoning as ``task_id`` above): ``SorcarAgent.run`` resolves
    # ``self._stop_event`` from the *worker* thread's
    # ``printer._thread_local``, so unless the event is copied into
    # each worker thread-local below, sub-agents never see the parent
    # stop request and Stop cannot kill their Bash process groups.
    parent_stop_event = getattr(parent_tl, "stop_event", None) if parent_tl else None

    def _run_single(args: tuple[int, str]) -> str:
        idx, task = args
        tl = getattr(printer, "_thread_local", None) if printer else None
        if tl is not None:
            tl.stop_event = parent_stop_event
        agent = ChatSorcarAgent(f"Parallel-{task[:40]}")
        # Mark the spawned agent as a sub-agent.  ``ChatSorcarAgent.run``
        # reads this marker to drive its own sub-agent-specific
        # behaviour (e.g. broadcasting ``new_tab`` to a browser-based
        # frontend, persisting the ``subagent`` extra field).  This
        # keeps the parallel executor itself free of any task-id or
        # frontend knowledge.  The base ``SorcarAgent`` path has no
        # parent ``task_id`` to record, so ``parent_task_id`` is the
        # empty string here (the same "no persisted parent" sentinel
        # the chat-aware override uses — see r4-sorcar-H2 in
        # ``ChatSorcarAgent._run_tasks_parallel``; a ``None`` here
        # would persist ``subagent: {parent_task_id: null}`` while the
        # chat path persists ``""``, split-braining downstream
        # ``parent_task_id == ""`` checks).  ``parent_tab_id`` is
        # likewise ``""`` so the persisted payload shape matches the
        # chat path.
        agent._subagent_info = {"parent_task_id": "", "parent_tab_id": ""}
        if usage_monitor is not None:
            usage_monitor.track(agent)
        try:
            # ``is_parallel=True`` propagates the parallel capability so
            # sub-agents themselves get the ``run_parallel`` tool and
            # can invoke nested parallel execution.  Without this, nested
            # parallel (sub-agent calls run_parallel) is impossible.
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
            sub_usage[idx] = _agent_usage(agent)
            if printer is not None and parent_key:
                # Use the same ``task-{parent}__sub_{idx}`` tab-id
                # format that ``ChatSorcarAgent._run_tasks_parallel``
                # registers/broadcasts, so a frontend tab materialized
                # under the chat-style deterministic id can always be
                # matched and its running indicator stopped.
                _broadcast_subagent_done(
                    printer, [f"task-{parent_key}__sub_{idx}"],
                )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_run_single, enumerate(tasks)))

    if totals_out is not None:
        totals_out["budget_used"] = sum(u[0] for u in sub_usage)
        totals_out["total_tokens_used"] = sum(u[1] for u in sub_usage)
        totals_out["total_steps"] = sum(u[2] for u in sub_usage)
    return results





