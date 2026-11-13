# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Stateful Sorcar agent with chat-session persistence.

Subclasses :class:`SorcarAgent` to add multi-turn chat-session state
management — the same workflow that the daemon performs in
``AgentServer._run_task()``, but as a standalone reusable Python agent.
"""

from __future__ import annotations

import functools
import json
import logging
import platform
import re
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar.bare_path_task import with_open_directive
from kiss.agents.sorcar.git_worktree import strip_worktree_suffix
from kiss.agents.sorcar.persistence import (
    TASK_USAGE_LOCK,
    _add_task,
    _allocate_chat_id,
    _append_chat_event,
    _current_db_path,
    _load_chat_context,
    _load_task_chain_context,
    _save_task_extra,
    _save_task_result,
    _task_has_transcript_events,
    _write_event_batch,
    task_end_ms,
    usage_totals_event,
)
from kiss.agents.sorcar.relentless_agent import DEFAULT_MAX_BUDGET
from kiss.agents.sorcar.sea_commands import named_commands_help
from kiss.agents.sorcar.sorcar_agent import (
    SorcarAgent,
    _first_cleanup_error,
    canonical_tool_profile,
    classification_chooses_prompt,
    resolve_tool_profile,
)
from kiss.core._version import __version__
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.llm_reply import LLMReply
from kiss.core.locked import _race_delay, locked_call
from kiss.core.models.model import Attachment
from kiss.core.printer import Printer
from kiss.core.tool_verdict import Verdict
from kiss.core.utils import substitute_prompt_args

logger = logging.getLogger(__name__)

MAX_TASKS = 10
DIGEST_FULL_RESULTS = 2
"""The newest prior tasks whose result is quoted in full in the chat prefix."""
DIGEST_MAX_PREFIX_CHARS = 6_000
"""Older prior tasks are dropped (oldest first) until the prefix fits this."""
DIGEST_TASK_CHARS = 600
DIGEST_RESULT_CHARS = 300
_TAG_RE = re.compile(r"<[^<>]+>")


def _digest(text: str, limit: int) -> str:
    """Strip HTML tags, collapse whitespace and cut *text* to *limit* chars."""
    plain = " ".join(_TAG_RE.sub(" ", str(text)).split())
    return plain if len(plain) <= limit else plain[:limit].rstrip() + " …"


def _format_history(entries: list[dict[str, object]], full_from: int) -> str:
    """Render prior tasks; entries before index *full_from* are digested."""
    parts: list[str] = []
    for i, entry in enumerate(entries):
        task = str(entry["task"])
        result = str(entry.get("result") or "")
        if i < full_from:
            task = _digest(task, DIGEST_TASK_CHARS)
            result = _digest(result, DIGEST_RESULT_CHARS)
        parts.append(f"### Task {i + 1}\n{task}")
        if result:
            parts.append(f"### Result {i + 1}\n{result}")
    return "\n\n".join(parts)


def _digest_history(entries: list[dict[str, object]]) -> str:
    """Render the chat history with only the newest results in full.

    The prefix is re-sent on every step of the task, so older tasks are
    reduced to a tag-stripped digest and, when the prefix still exceeds
    :data:`DIGEST_MAX_PREFIX_CHARS`, dropped oldest-first.  The newest
    :data:`DIGEST_FULL_RESULTS` tasks are kept whole as long as they fit;
    the cap wins otherwise (they are digested too, then hard-truncated).

    Args:
        entries: Prior tasks, oldest first, each with ``task`` and ``result``.

    Returns:
        The rendered history section.
    """
    kept = list(entries)
    full = DIGEST_FULL_RESULTS
    while True:
        body = _format_history(kept, full_from=max(0, len(kept) - full))
        if len(body) <= DIGEST_MAX_PREFIX_CHARS:
            return body
        if len(kept) > DIGEST_FULL_RESULTS:
            del kept[0]
        elif full > 0:
            # Even the newest results are too long: digest them too,
            # oldest first, rather than exceed the cap.
            full -= 1
        else:
            return body[:DIGEST_MAX_PREFIX_CHARS].rstrip() + " …"


def _safe_broadcast(broadcast: Any, event: dict[str, Any]) -> None:
    """Send *event* through the printer's ``broadcast``, if it has one.

    A UI-only notification (``new_tab``, ``tasks_updated``,
    ``task_settings``): a failing transport is logged and never fails
    the run.

    Args:
        broadcast: The printer's ``broadcast`` callable, or ``None``.
        event: The event payload.
    """
    if broadcast is None:
        return
    try:
        broadcast(event)
    except Exception:
        logger.warning("%s broadcast raised", event.get("type"), exc_info=True)


def _retried_cleanup_step(
    first: BaseException | None, step: str, action: Callable[[], None],
) -> BaseException | None:
    """Run the idempotent run-cleanup *step*; retry it once when an exception cut it short.

    For a step whose completion the accounting depends on: the keyed
    classifier fold (exactly once however often it runs) and the row's
    final result and usage save (updates of the row by id, re-reading
    the ledger).  The runner's Stop injects ``KeyboardInterrupt`` into
    a live task thread at most twice (``_force_stop_thread``), so the
    retry of a step the second injection landed in runs to its end
    instead of leaving the row's cost at zero or the classifier's spend
    unbanked (review of 2026-11-02, findings 2 and 3).  Each exception is
    logged (:func:`~kiss.agents.sorcar.sorcar_agent._first_cleanup_error`).

    Args:
        first: The first exception an earlier step raised, or ``None``.
        step: The step's name, for the log.
        action: The step.

    Returns:
        The first exception any step raised so far, or ``None``.
    """
    for attempt in ("", " (retry)"):
        try:
            action()
            return first
        except BaseException as exc:  # noqa: BLE001 — held, see run()'s cleanup
            first = _first_cleanup_error(first, step + attempt, exc)
    return first


def _retire_printer_task(agent: Any, run_printer: Any, task_key: str) -> None:
    """Tell *run_printer* that *agent*'s run under *task_key* is over, then stop its recording.

    Both hooks are duck-typed, optional and idempotent.  Once entered,
    the finish hook (``JsonPrinter.agent_task_finished``) drops the
    state the printer bridge registered for the run whatever Stop
    lands in it.  A Stop raised earlier, as a call on the way to the
    hook is entered, would skip it: the state would stay registered,
    an active task of a thread that has left the run, and the Stop's
    watchdog would go on interrupting whatever the thread runs next.

    CPython raises an injected Stop only at a function's entry, after
    a call returns or at a loop's back-edge, and another thread
    can inject one only while this thread has given up the GIL: in a
    blocking call or at one of those points.  After a Stop is raised
    none is pending, so the next point raises none unless the thread
    blocked on the way there (one injected at that point is raised at
    the one after).  The hook is therefore the first call made here
    (its lookup is an attribute read, not a ``getattr`` call) and is
    made again at once when a ``BaseException`` cuts it short: the
    second call is past the hook's entry before a Stop can be raised
    again.  The caller repeats this function the same way, for a Stop
    raised at its entry.

    ``stop_recording`` runs whatever the hook did: it reads the
    thread's task binding, which the caller restores next, so it runs
    only while that binding is *task_key*'s.  A Stop that landed
    before the run bound the thread leaves the binding the calling
    task's (a sub-agent runs on its caller's thread), whose live
    recording an unkeyed stop would retire.

    Args:
        agent: The agent whose run is over.
        run_printer: The printer the run was given.
        task_key: The task id the run was registered under.
    """
    try:
        try:
            finished = run_printer.agent_task_finished
        except AttributeError:
            finished = None
        if finished is not None:
            try:
                finished(agent, task_key)
            except Exception:
                logger.warning("agent_task_finished(%r) raised", task_key, exc_info=True)
            except BaseException:
                # Inline, not through a helper: its entry would come first.
                try:
                    finished(agent, task_key)
                except Exception:
                    logger.warning("agent_task_finished(%r) raised", task_key, exc_info=True)
                raise
    finally:
        stop_rec = getattr(run_printer, "stop_recording", None)
        tl = getattr(run_printer, "_thread_local", None)
        if tl is not None and getattr(tl, "task_id", "") != task_key:
            stop_rec = None
        if stop_rec is not None:
            try:
                stop_rec()
            except Exception:
                # The recording then stays live for the daemon's lifetime
                # (every later broadcast under the task's id appends to it):
                # a failure worth a trace, like the finish hook's.
                logger.warning("stop_recording(%r) raised", task_key, exc_info=True)


def _parse_result(result: str) -> dict[str, Any] | None:
    """Return a finished run's *result* as a mapping, or ``None``.

    Args:
        result: The raw string returned by the agent run.

    Returns:
        The YAML mapping, or ``None`` when *result* is not valid YAML or
        not a mapping (a plain string, a list, a number).
    """
    try:
        data = yaml.safe_load(result)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _result_summary_text(result: str, parsed: dict[str, Any] | None) -> str:
    """Return the persistable summary text for a finished run's *result*.

    Handles every shape LLMs emit:

    * a mapping with a string ``summary`` — returned as-is;
    * a mapping with ``summary: null`` (or no ``summary`` key) — ``""``;
    * a mapping with a list/mapping/scalar ``summary`` — its YAML text
      form (passing the raw object to ``_save_task_result`` would raise
      ``sqlite3.ProgrammingError``, destroying the task's successful
      return value);
    * no mapping (valid YAML that is not a mapping, or unparseable
      text) — the raw text capped at 500 characters (otherwise the task
      history would record an empty result).

    Args:
        result: The raw string returned by the agent run.
        parsed: :func:`_parse_result` of *result*.

    Returns:
        The summary text to persist (possibly empty, never ``None``).
    """
    if parsed is None:
        return result[:500]
    summary_val = parsed.get("summary", "")
    if isinstance(summary_val, str):
        return summary_val
    if summary_val is None:
        return ""
    dumped: str = yaml.safe_dump(summary_val, sort_keys=False)
    return dumped.strip()


def _extract_result_summary(result: str) -> str:
    """Return the persistable summary text for a finished run's *result*.

    :func:`_result_summary_text` for a caller that holds only the raw
    result (the task-update side channel); :meth:`ChatSorcarAgent.run`
    parses once and shares the mapping with its persistence steps.

    Args:
        result: The raw string returned by the agent run.

    Returns:
        The summary text to persist (possibly empty, never ``None``).
    """
    return _result_summary_text(result, _parse_result(result))


class ChatSorcarAgent(SorcarAgent):
    """SorcarAgent with chat-session state management.

    Maintains a ``chat_id`` and automatically loads prior chat context,
    persists tasks and results to ``history.db``, and augments prompts
    with previous session context — replicating the daemon's stateful
    workflow as a standalone reusable agent.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._chat_id: str = ""
        self._context_task_id: str = ""
        # File stem of the SEA (SEA, wire field ``seaPath``)
        # this run executes on behalf of, e.g. ``write_paper_sea`` or
        # ``cron_agent``; set by the task runner and persisted in the
        # ``task_history.sea`` column.  Empty for a plain run.
        self.sea_name: str = ""
        # True when the cron scheduler launched this run, or a cron run
        # dispatched it as a sub-task (see cron_agent.is_cron_run); set
        # by the task runner and persisted in the ``task_history.is_cron``
        # column, which the CHATS list's Cron chip filters on.
        self.is_cron_job: bool = False
        # The task runner's verdict on whether the task runs unattended
        # (a cron prompt job or a sub-task of one, decided on the
        # submitted text): ``ask_user_question`` then refuses instead of
        # blocking.  None until stamped, when cron_agent.is_unattended
        # falls back to the task text (a library run).
        self.is_unattended: bool | None = None
        # The run's effective configuration beyond model / work dir /
        # budget (the keys kiss.agents.sorcar.run_config.run_config_line
        # prints): set per run by the task runner or the fan-out engine and
        # folded into the ``task_settings`` event.  Empty for a run
        # that is neither a daemon task nor a sub-agent.
        self.run_config: dict[str, Any] = {}
        self._last_task_id: str | None = None
        self._last_user_prompt: str = ""
        self._last_result_summary: str = ""
        self._task_id_lock: threading.RLock = threading.RLock()

    @property
    def chat_id(self) -> str:
        """Return the current chat session ID ("" means new session)."""
        return self._chat_id

    @property
    def last_task_id(self) -> str:
        """Return the ``task_history`` row id this agent last allocated.

        The readers live on other threads — the WebSocket command
        handler stamping a queued user message, the merge/discard
        flow, the printer's broadcast fan-out — so the read takes
        ``_task_id_lock``, the same lock the publishing assignment in
        :meth:`run` takes.  That pairing is what makes the lock mean
        anything: a lock only the writer holds excludes nobody.

        Returns:
            The row id, or ``""`` before this agent's first ``run``.
        """
        return locked_call(self._task_id_lock, getattr, self, "_last_task_id") or ""

    @property
    def last_user_prompt(self) -> str:
        """Return the task text of this agent's current (or last) :meth:`run`.

        The text as submitted — what the user typed, or the ``task`` a
        ``run_agent`` dispatch passed — before the chat history, the
        bare-path directive or ``AGENTS.md`` are added to the prompt the
        model sees.  A SEA's tool reads it through
        :func:`kiss.server.agent_state.current_agent` to parse options
        out of its own task (rsi7d's scope).

        Returns:
            The text, or ``""`` before this agent's first ``run``.
        """
        return self._last_user_prompt

    def new_chat(self) -> None:
        """Reset to a new chat session: the next :meth:`run` starts a fresh chat.

        Library API; the chat page never calls it (a new chat there is
        a new tab, whose agent is created for its first run).  Retires
        any pending worktree first (``SorcarAgent.new_chat``), and drops
        any pending one-shot :meth:`resume_from_task_id` seed: a
        brand-new chat must never have its first prompt augmented with
        the previous task's parent-chain context.
        """
        super().new_chat()
        self._chat_id = ""
        self._context_task_id = ""

    def _commit_context(self) -> tuple[str | None, str | None]:
        """The chat's last task text and result summary, for commit messages."""
        return self._last_user_prompt or None, self._last_result_summary or None

    def resume_chat_by_id(self, chat_id: str) -> None:
        """Resume a chat session using a stable chat identifier.

        Args:
            chat_id: String chat session identifier to resume.
        """
        if chat_id:
            self._chat_id = chat_id

    def resume_from_task_id(self, task_id: str) -> None:
        """Seed the next prompt's context from a task's parent chain.

        Called when the tab that owns this agent was opened by a
        specific task id (history click / ``resumeSession`` with
        ``taskId``) and no task has been run in the tab since: the
        first :meth:`build_chat_prompt` after this call traverses the
        ``parent_task_id`` links starting at *task_id* (via
        :func:`_load_task_chain_context`) instead of loading the whole
        chat.  The seed is one-shot — subsequent prompts fall back to
        the normal chat-context path.

        Args:
            task_id: The ``task_history.id`` the tab was opened with.
                Empty strings are ignored.
        """
        if task_id:
            self._context_task_id = task_id

    def build_chat_prompt(self, prompt: str, name_commands: bool = False) -> str:
        """Load chat context and augment prompt with previous tasks/results.

        Args:
            prompt: The original task prompt.
            name_commands: Whether to append the help of the slash
                commands that *prompt* and the chat's earlier tasks
                loaded here name (:func:`named_commands_help`; also one
                the history digest drops for size), so a task told to
                "use /discover" knows what it does.  Only for a plain
                task the user wrote: a SEA run or a sub-agent sees the
                same chat and must not be sent after itself.

        Returns:
            The augmented prompt with chat history prepended, or the
            original prompt if no prior context exists.
        """
        chat_context: list[dict[str, object]] = []
        if self._context_task_id:
            chat_context = _load_task_chain_context(self._context_task_id)
            self._context_task_id = ""
        if not chat_context:
            chat_context = _load_chat_context(self._chat_id)
        if not chat_context:
            augmented = "# Task\n" + prompt
        else:
            if len(chat_context) > MAX_TASKS:
                del chat_context[2:2 + len(chat_context) - MAX_TASKS]
            if DEFAULT_CONFIG.chat_history_digest:
                body = _digest_history(chat_context)
            else:
                body = _format_history(chat_context, full_from=0)
            augmented = (
                "## Previous tasks and results from the chat session for reference\n\n"
                + body + "\n\n---\n\n# Task (work on it now)\n\n" + prompt
            )
        named = (
            named_commands_help(prompt, (str(e["task"]) for e in chat_context))
            if name_commands
            else ""
        )
        return augmented + "\n\n" + named if named else augmented

    def _save_final_usage(
        self,
        extra_payload: dict[str, Any],
        task_id: str,
        prompt: str = "",
        parsed: dict[str, Any] | None = None,
        result_summary: str = "",
        *,
        unless_finished: bool = False,
    ) -> None:
        """Stamp the ledger totals and the end time on the row (``TASK_USAGE_LOCK`` held).

        A run that recorded no transcript events (one outside a chat
        page) gets its replay events synthesized first, with the same
        totals (:meth:`_persist_replay_events_if_missing`), so the one
        ``endTs`` save derives the row's tags over the complete
        trajectory (a secret in the augmented *prompt* is tagged).

        Args:
            extra_payload: The row's extra payload, completed in place.
            task_id: The task row.
            prompt: The prompt the agent ran with, for the synthesized
                ``prompt`` event.
            parsed: The run's result mapping (:func:`_parse_result`), or
                ``None``.
            result_summary: The extracted human-readable summary text.
            unless_finished: Write nothing when the row already carries
                an end time: the retry of a save a Stop interrupted
                (``run``'s cleanup) must not replace the totals a
                completed first attempt wrote, since a late charge may
                have been added to the finished row meanwhile
                (:func:`~kiss.server.task_update.charge_side_channel_usage`),
                which the ledger snapshot does not contain.  Read under
                the same lock, so no late add can fall between the check
                and the write.
        """
        while not (unless_finished and task_end_ms(task_id) > 0):
            owed_remote = self._bank_pending_remote_charges(task_id)
            # One coherent triple (see usage_snapshot): three property reads
            # could tear across a concurrent sub-task fold.
            final_cost, final_tokens, final_steps = self.usage_snapshot()
            _race_delay()  # test hook: widens the snapshot-to-save window
            extra_payload["tokens"] = final_tokens
            # Unrounded, like the live ``cost_usd`` and the late adds to the
            # row: a parent row sums its children's exact spend.
            extra_payload["cost"] = final_cost
            extra_payload["steps"] = final_steps
            # The run is over: stamp its end so sub-agents (which no server
            # runner finalises) get a duration too.
            extra_payload["endTs"] = int(time.time() * 1000)
            self._persist_replay_events_if_missing(
                task_id, prompt, parsed, result_summary,
                usage=(final_cost, final_tokens, final_steps),
            )
            if owed_remote:
                # The bank above may have grown the totals past the run's
                # last usage event: persist the final ones as the task's
                # latest, as a late charge of a finished row does, so its
                # replayed header shows what its row holds.  Before the row
                # is finished, so a save a Stop cuts persists it on retry (a
                # repeat of the same totals is harmless).
                totals = usage_totals_event(task_id, final_tokens, final_cost, final_steps)
                _append_chat_event({**totals, "ts": int(time.time() * 1000)}, task_id=task_id)
            if _save_task_extra(extra_payload, task_id=task_id, owed_through=owed_remote):
                return
            # Another process charged the row after the bank above read
            # its pending charges, and nothing was written: bank that
            # charge too, so that the ledger this run's caller folds
            # holds what the row holds, and save again.  A loop, as such
            # charges can outnumber the frames the stack has.

    def _bank_pending_remote_charges(self, task_id: str) -> int | None:
        """Bank on this run's ledger the remote charges its row got while no live agent held it.

        A remote sub-agent's late spend reported while the row runs is
        banked on the live agent the registry finds
        (``task_update._bank_remote_charge``); one reported when the
        registry has none (a bridge-owned run is unregistered before its
        final save, a run outside the daemon is never registered) is left
        on the row unbanked, and the row's finalization alone would add
        it, past the ledger whose totals the run's caller folds.  Banked
        here exactly as the live path does, with ``TASK_USAGE_LOCK`` held
        (so no charge this process records falls between this and the
        snapshot after it): the mark first, with the ledger totals it is
        banked over, then the record, under the offsets' publish lock
        (taken in slices: a Stop reaches a save another publisher blocks).
        A charge already marked is not marked again and its record is
        appended again: the ledger counts its key once, so one the live
        agent banked adds nothing, and one whose bank a Stop cut after
        its mark is banked by the retried save.  The snapshot then holds
        each and the final save marks them settled, so each counts once.

        Args:
            task_id: The finishing run's row.

        Returns:
            The id of the newest charge the row is owed and its totals
            do not hold yet (banked here or by the live agent), ``0``
            when it is owed none, ``None`` when they could not be read.
        """
        from kiss.agents.sorcar.persistence import _pending_remote_charges
        from kiss.agents.sorcar.relentless_agent import _OFFSET_PUBLISH_LOCK
        from kiss.agents.sorcar.sorcar_agent import _bank_remote_charge

        newest = None
        try:
            pending = _pending_remote_charges(task_id)
            newest = pending[-1][0] if pending else 0
            for charge, cost, tokens, steps, marked in pending:
                locked_call(
                    _OFFSET_PUBLISH_LOCK, _bank_remote_charge,
                    self, charge, cost, tokens, steps, marked,
                )
        except Exception:
            # Never fails the final save: an unbanked charge is still
            # settled into the row by its finalization.
            logger.warning("pending remote charges of %s not banked", task_id, exc_info=True)
        return newest

    def _build_extra_payload(
        self,
        model: str,
        work_dir: str,
        is_parallel: bool,
        max_budget: float | None = None,
    ) -> dict[str, object]:
        """Build the task-history "extra" payload for persistence.

        Shared by the early save (at task start, from the run kwargs)
        and the final save (at task end, from the live agent state).
        Includes the ``subagent`` marker when this agent is a parallel
        sub-agent.

        Args:
            model: Model name to record.
            work_dir: Working directory to record.
            is_parallel: Whether parallel sub-agents are enabled.
            max_budget: The run's resolved budget cap in USD, or None
                to omit it from the payload.

        Returns:
            The extra-payload dict.
        """
        payload: dict[str, object] = {
            "model": model,
            "work_dir": strip_worktree_suffix(work_dir),
            "version": __version__,
            "is_parallel": is_parallel,
        }
        if max_budget is not None:
            payload["max_budget"] = max_budget
        if self._subagent_info is not None:
            payload["subagent"] = self._subagent_info
        if self.sea_name:
            payload["sea"] = self.sea_name
        if self.is_cron_job:
            payload["is_cron"] = True
        return payload

    def _system_prompt_task_settings(self) -> dict[str, str]:
        """Extend the settings with this chat run's identity and modes.

        Adds the chat / task ids allocated at the top of :meth:`run`
        and the sub-agent parentage — everything
        this class knows that the base agents do not.

        Returns:
            The base label → value pairs plus the chat-level ones.
        """
        settings = super()._system_prompt_task_settings()
        settings["Chat id"] = self._chat_id
        settings["Task id"] = self.last_task_id
        sub = self._subagent_info
        settings["Is subagent"] = "yes" if sub is not None else "no"
        parent_id = str((sub or {}).get("parent_task_id") or "")
        if parent_id:
            settings["Parent task id"] = parent_id
        return settings

    def _tree_scratch_dir(self) -> str:
        """One scratch directory per task, ``./tmp/<task-id>/``.

        Concurrent siblings in the same work dir then never share a file
        name (P12 of the 2026-10-24 trajectory audit: two reviews read
        each other's tmp/paper.pdf), and each task's verifier is its own.
        """
        return f"./tmp/{self.last_task_id}/"

    def _task_settings_payload(
        self,
        model: str,
        work_dir: str,
        is_parallel: bool,
        max_budget: float | None,
        start_ts: int,
        task_id: str,
        tool_profile: str,
    ) -> dict[str, object]:
        """Build the ``task_settings`` display-event payload.

        The shape mirrors the task-history sidebar's session fields so
        the chat page can render the static task panel's info line
        exactly like a history row (see ``renderHistory`` and
        ``taskPanelInfoHTML`` in ``media/main.js``).

        Args:
            model: Model name the run was asked for ("" when the
                caller left it to the default).
            work_dir: Working directory of the run.
            is_parallel: Whether parallel sub-agents are enabled.
            max_budget: The run's budget cap in USD, or None when
                unstated.
            start_ts: The run's start timestamp (ms since epoch) — the
                same value persisted as the row's ``startTs``.
            task_id: The freshly allocated ``task_history`` row id.
            tool_profile: The effective tool profile (``"full"`` or a
                profile name, as :meth:`_tool_profile` returns it).

        Returns:
            The settings dict carried by the event: the keys above,
            ``chat_id``, ``is_subagent``, ``host`` (the machine the run
            is on, ``platform.node()``: ``sync-task-db.sh`` copies tasks
            between machines and the s10s hub sync sends only the tasks
            that ran on the sending one), ``parent_task_id`` when there
            is one, ``sea`` when the run executes a SEA and
            the :attr:`run_config` keys (the ones
            :func:`kiss.agents.sorcar.run_config.run_config_line` prints).
        """
        payload = self._build_extra_payload(model, work_dir, is_parallel, max_budget)
        del payload["version"]
        payload.pop("subagent", None)
        payload.update(
            start_ts=start_ts,
            chat_id=self._chat_id,
            task_id=task_id,
            is_subagent=self._subagent_info is not None,
            host=platform.node(),
        )
        parent_id = str((self._subagent_info or {}).get("parent_task_id") or "")
        if parent_id:
            payload["parent_task_id"] = parent_id
        payload.update(self.run_config)
        payload["tool_profile"] = tool_profile
        # Nobody named the profile and the reviewer rule chose
        # ``review``: say so (``tools=review(inferred)``), so the caller
        # can tell the inference from a profile it or a SEA asked for.
        if tool_profile == "review" and not self._tool_profile_name:
            payload["tool_profile_inferred"] = True
        return payload

    def _persist_replay_events_if_missing(
        self,
        task_id: str,
        prompt: str,
        parsed: dict[str, Any] | None,
        result_summary: str,
        usage: tuple[float, int, int],
    ) -> None:
        """Persist a minimal replayable event stream when none was recorded.

        Runs that happen from a chat page stream every agent event
        through a recording printer (the daemon's ``JsonPrinter`` /
        ``WebPrinter``), which persists them to the ``events`` table.
        Runs that happen OUTSIDE a chat page — the third-party channel
        agents, or an API invocation with a non-recording printer —
        leave the ``events`` table empty, so the chat page would load a
        blank session even though the task and its result are in
        ``task_history``.

        This synthesizes the events the chat page needs to render the
        exchange — a ``prompt`` event (the user's task), a ``result``
        event (the agent's summary / success / cost) and, when the agent
        proposed one via ``finish(suggested_next_task=...)``, a
        ``followup_suggestion`` event — but only when the task has no
        transcript events yet (the run's ``task_settings`` metadata
        event, persisted before any output, is ignored), so a recording
        printer's full event stream is never duplicated.  The events
        are written as one batch in one transaction: a Stop landing in
        the write leaves none of them or all of them, so the retry of
        the final save (``run``'s cleanup) either synthesizes them again
        or finds the complete transcript — never a prompt alone, which
        it would take for a recorded transcript.

        Args:
            task_id: Stable ``task_history`` row id for this run.
            prompt: The prompt the agent actually ran with (chat-context
                augmented), mirroring the ``prompt`` event a recording
                printer would have persisted.
            parsed: The run's result mapping (:func:`_parse_result`;
                used to recover ``success`` / ``is_continue``), or
                ``None``.
            result_summary: The extracted human-readable summary text.
            usage: One coherent ``(budget_used, total_tokens_used,
                total_steps)`` triple (:meth:`usage_snapshot`) to stamp
                on the ``result`` event.
        """
        if _task_has_transcript_events(task_id):
            return
        events: list[dict[str, object]] = []
        if prompt:
            events.append({"type": "prompt", "text": prompt})
        cost, tokens, steps = usage
        event: dict[str, object] = {
            "type": "result",
            "text": result_summary or "(no result)",
            "total_tokens": tokens,
            "cost": f"${cost:.4f}",
            "cost_usd": cost,
            "step_count": steps,
        }
        if parsed and "summary" in parsed:
            event["success"] = parsed.get("success")
            event["is_continue"] = bool(parsed.get("is_continue", False))
            event["summary"] = str(parsed["summary"])
        else:
            event["summary"] = result_summary or ""
        events.append(event)
        suggestion = str((parsed or {}).get("suggested_next_task") or "").strip()
        if suggestion:
            events.append({"type": "followup_suggestion", "text": suggestion})
        origin, now = _current_db_path(), time.time()
        _write_event_batch([(task_id, json.dumps(ev), now, origin) for ev in events])

    def _begin_run_usage_epoch(self) -> None:
        """Keep the usage epoch :meth:`run` started at row allocation.

        Swapping it again at ``_reset`` would discard spend a side
        channel banked into it while the task was still setting up
        (e.g. classifying).
        """

    def run(  # type: ignore[override]
        self,
        prompt_template: str = "",
        model_name: str | None = None,
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
        inherited_tools: list[Callable[..., Any]] | None = None,
        llm_call_hook: (
            Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None
        ) = None,
        tool_call_hook: Callable[[str, dict[str, Any]], Verdict] | None = None,
        llm_response_hook: Callable[[LLMReply], Verdict] | None = None,
        use_memory: bool | None = None,
        tool_profile: str = "",
        live_browser: Any = None,
        prompt_suffix: str = "",
        system_prompt_hook: Callable[[str], str] | None = None,
        tools_hook: Callable[[list[Callable[..., Any]]], list[Callable[..., Any]]] | None = None,
        _skip_persistence: bool = False,
        _on_task_id_allocated: Callable[[str, str], None] | None = None,
        _history_prompt: str | None = None,
        _open_bare_path: bool = True,
        _start_ms: int = 0,
        compaction: bool = True,
    ) -> str:
        """Run the agent with chat-session context management.

        Loads prior chat context, persists the new task, augments the
        prompt with previous tasks/results, runs the underlying agent,
        and saves the result back to history.

        Only the result summary is persisted here.  Callers that record
        chat events (e.g. the daemon) persist events incrementally
        via :func:`~kiss.agents.sorcar.persistence._append_chat_event`.

        Args:
            prompt_template: The task prompt.  First so the historical
                ``run("task", ...)`` call keeps working; every other
                parameter up to *tools_hook* but *arguments* is
                forwarded verbatim to :meth:`SorcarAgent.run`, in that
                method's order and with its defaults — see its
                docstring for what each one does.  *model_name* is
                resolved here once (:meth:`_resolve_model_name`) and
                the resolved name is what the base run receives, as
                is the profile decided for *tool_profile*.
            model_name: LLM model to use.  Defaults to the config value.
            arguments: Values filling the placeholders of
                *prompt_template*, and of nothing else: they are filled
                in here, before the chat history is put in front.
            system_prompt: Suffix appended to the actual system prompt.
            tools: Tools added on top of the built-in toolset.
            max_steps: Maximum steps per sub-session.
            max_budget: Maximum budget in USD; also recorded on the
                task-history row (``DEFAULT_MAX_BUDGET`` when omitted).
            model_config: Extra model configuration.
            work_dir: Working directory; its resolved absolute path is
                recorded on the task-history row.
            printer: Printer for output display; its ``broadcast`` (if
                any) carries the ``task_settings`` / ``new_tab`` /
                ``tasks_updated`` events of this run.
            max_sub_sessions: Maximum continuation sub-sessions.
            docker_image: Docker image to run tools inside.
            web_tools: Whether to include browser/web tools.
            is_parallel: Whether to include the ``run_parallel`` tool.
            verbose: Whether to print output to the console.
            current_editor_file: Active editor file appended to the prompt.
            attachments: File attachments for the initial prompt.
            ask_user_question_callback: Callback of the
                ``ask_user_question`` tool.
            base_system_prompt: Replacement for the default base system
                prompt.  A non-blank value also disables the bare-path
                "open it" directive (see *_open_bare_path*).
            append_basic_tools: Whether the built-in toolset is prepended.
            inherited_tools: Tools taken over from the dispatching task.
            llm_call_hook: Hook called before every LLM call.
            tool_call_hook: Hook called before every tool call.
            llm_response_hook: Hook called after every model reply.
            use_memory: Per-run persistent-memory toggle.
            tool_profile: Tool profile name the toolset is cut down to.
                The EFFECTIVE profile (``review`` for a reviewer
                sub-agent asked for none) is decided here, on the task
                without the chat history: the ``task_settings`` event
                reports it and the base run is given it.
            live_browser: The daemon's ``BrowserTabService``.
            prompt_suffix: Caller-supplied suffix already appended to
                *prompt_template*, recorded so a ``run_agent`` sub-task
                inherits it.
            system_prompt_hook: The SEA's ``system_prompt`` method.
            tools_hook: The SEA's ``tools`` method.
            _skip_persistence: When True the final result / usage /
                replay-event save is skipped (the daemon persists them
                itself); the task-history row is still allocated.
            _on_task_id_allocated: Called with ``(task_id, chat_id)`` as
                soon as the history row exists, before any event of the
                run is broadcast.
            _history_prompt: The raw user text to persist and display
                (history row, last-user-prompt cache) in place of
                *prompt_template*, which may carry an internal directive
                the LLM must see (a slash-command run).  ``None`` or
                blank: *prompt_template* with its *arguments* filled in
                is used.
            _open_bare_path: Whether a *prompt_template* that is only a
                filesystem path gets the "open it" directive.  The task
                runner passes False for a run it starts on behalf of a
                SEA or of another agent.
            _start_ms: The run's start as a ms epoch, when the caller
                stamped it before calling (the daemon's ``<task>``
                block); 0 (default) dates the run at row allocation.
            compaction: Whether the run compacts its context (see
                :meth:`SorcarAgent.run`): False never compacts, True (the
                default) compacts while the global Compaction setting is
                on.

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        # The arguments are filled in here, once: every reader below
        # (the history row, the profile and classifier verdicts, the
        # prompt) gets the task as the model sees it, and what
        # ``build_chat_prompt`` puts around it (an earlier task or
        # result may quote ``{task}``) is not a template, so the base
        # run is given no arguments.
        task_text = substitute_prompt_args(prompt_template, arguments)
        # A caller (currently the slash-command dispatch in
        # ``TaskRunner._run_task_inner``) can supply the raw user text
        # to use for every history / display write, while
        # ``prompt_template`` carries an internal directive the LLM
        # must actually see.  Absent, the two are the same and the
        # task is used everywhere.
        history_prompt = (
            _history_prompt
            if isinstance(_history_prompt, str) and _history_prompt
            else task_text
        )
        if self._chat_id == "":
            self._chat_id = _allocate_chat_id()
        # A worktree pending from the previous task is retired now,
        # while ``_commit_context`` still describes that task: the
        # commit message of its kept-for-review branch must not be
        # generated from the prompt below.
        self._begin_task(printer)
        # ``_last_task_id`` is deliberately NOT cleared here.  The next
        # two steps are a SQLite read (build_chat_prompt) and a SQLite
        # write (_add_task), and every server-thread reader of this
        # attribute — the queued-message stamper, the merge/discard
        # flow, the printer's fan-out — would resolve ``None`` for that
        # whole window and drop or misroute the user's action.  The
        # previous run's id is stale but valid, and it is replaced by
        # the single publish below.
        self._last_user_prompt = history_prompt
        self._last_result_summary = ""

        # A task that is only a filesystem path means "open it"; the
        # directive is added here, AFTER ``history_prompt`` was taken,
        # so history and the tab keep the raw path the user typed.
        # Only for a task the user typed into a chat box (or scheduled as
        # an in-chat cron job), though, which alone also gets the help of
        # the slash commands it names: the task runner passes
        # ``_open_bare_path=False`` for a run it
        # starts on behalf of a SEA (``seaPath``: an SEA
        # dispatched by ``/git_extract_knowledge /path/to/repo`` or
        # ``/review_paper /path/to/paper.pdf`` receives the path as its
        # whole task and defines what it means) or of another agent
        # (``parentTaskId``: a ``run_agent`` sub-task), and a custom
        # base system prompt (an SEA's ``system_prompt()``) suppresses
        # it as well, so a direct ``ChatSorcarAgent.run`` of an agent
        # script is not sent off to ``xdg-open`` the path either.
        open_bare_path = bool(_open_bare_path) and not str(
            base_system_prompt or "",
        ).strip()
        agent_prompt = self.build_chat_prompt(
            with_open_directive(task_text, work_dir or ".")
            if open_bare_path
            else task_text,
            name_commands=open_bare_path,
        )

        try:
            run_max_budget = None if max_budget is None else float(max_budget)
        except (TypeError, ValueError):
            run_max_budget = None
        # One authoritative per-run settings snapshot, resolved exactly
        # the way ``_reset`` will resolve them: the early history row,
        # the ``task_settings`` event, and the final save must all
        # agree with the run itself — not echo raw (possibly omitted)
        # kwargs that the run later resolves differently.  The model
        # is resolved HERE, once, and passed down: the fallback reads
        # the user's config, which the picker may change between two
        # reads (see ``SorcarAgent._resolve_model_name``).
        resolved_model = self._resolve_model_name(model_name)
        # The tool profile is the EFFECTIVE one, decided HERE, once,
        # and passed down like the model: a reviewer-marked sub-agent
        # asked for no profile runs ``review`` unless its task asks for
        # changes, and ``SorcarAgent.run`` could only judge
        # ``agent_prompt``, where an earlier task of the chat ("Fix the
        # parser bug") reads as such a request.  The verdict gates the
        # classifier and is recorded in the task settings below.
        self._tool_profile_name = canonical_tool_profile(tool_profile)
        effective_profile = self._tool_profile(task_text)
        # The verdict is about the task as submitted, not about the
        # conversation ``agent_prompt`` prepends to it: classify the
        # raw task here, ahead of ``SorcarAgent.run`` (which reuses the
        # verdict) — but only when the verdict can choose the prompt;
        # a run that cannot edit files gets SYSTEM_LITE.md regardless,
        # and one with its own base prompt that prompt.
        # The reset first drops a verdict a previous run of this agent
        # left behind by crashing before its own cleanup; a verdict the
        # task runner pre-seeded survives it.
        self._reset_task_classification()
        if classification_chooses_prompt(
            resolve_tool_profile(effective_profile), append_basic_tools, base_system_prompt,
        ):
            self._classify_task_once(resolved_model, task_text, model_config)
        resolved_budget = (
            run_max_budget if run_max_budget is not None else DEFAULT_MAX_BUDGET
        )
        resolved_work_dir = str(Path(work_dir or ".").resolve())
        run_is_parallel = bool(is_parallel)
        # The row's start is the run's start, not the row's: the daemon
        # (``TaskRunner._run_task_inner``) stamps a ``<task>`` block
        # when it begins and passes it as ``_start_ms``, so the row,
        # its ``task_settings`` event, the live ``status`` broadcast
        # and ``task_done`` all count from one instant — the worktree
        # setup and SEA import that precede this row would otherwise
        # make a client that attaches mid-run start its timer late.
        start_ts_ms = int(_start_ms or 0) or int(time.time() * 1000)

        early_extra = self._build_extra_payload(
            model=resolved_model,
            work_dir=resolved_work_dir,
            is_parallel=run_is_parallel,
            max_budget=resolved_budget,
        )
        early_extra["startTs"] = start_ts_ms

        # This run's usage epoch starts BEFORE its row exists: once the
        # row is visible a ``/update`` or ``/ask`` side channel may
        # capture the epoch and later bank its spend into it, so the
        # epoch must not be swapped again by ``_reset`` (see
        # ``_begin_run_usage_epoch``).  Classifier spend is held apart
        # until ``_reset`` banks it (the ``finally`` below, for a run
        # that ends earlier), so nothing is lost here.
        self.reset_usage()
        # ``super().run`` binds the printer from the *printer* argument
        # (``None`` gives a fresh console printer or none, never the one
        # an earlier run left in ``self.printer``), so the setup and
        # cleanup hooks below use the argument too.
        run_printer = printer
        result_summary = ""
        parsed: dict[str, Any] | None = None
        # Whether ``super().run`` was reached.  Until then every live
        # setting the final save would read — ``_launch_model_name``,
        # ``model_name``, ``_is_parallel`` — is either unset (fresh
        # agent) or the PREVIOUS run's (reused agent), so the final save
        # must fall back to this run's resolved settings.  (The usage
        # ledger is already this run's: see ``reset_usage`` above.)
        run_started = False
        # Whether the run returned normally (the cleanup re-raises a
        # Stop that landed in one of its own steps only then).
        completed = False
        # Every remaining setup step (the row's allocation, the id's
        # publication, state registration, printer wiring, ...) must run
        # inside the try below: an exception in any of them would
        # otherwise bypass the cleanup and leave a permanently "running"
        # task behind (F-14).  The locals the cleanup reads are bound
        # BEFORE the try.
        # The calling thread's previous task binding is restored on exit:
        # a side-channel task (``/update``, merge conflict resolution)
        # runs on a worker thread that then attributes the child's spend
        # to the PARENT agent, and that write lands under whatever
        # ``task_id`` the thread carries.  Resetting to ``""`` instead of
        # the previous value stored the parent's budget offset under an
        # orphan key, so the parent's tab under-counted until its own
        # thread rewrote the offset.
        previous_task_id = ""
        broadcast = getattr(run_printer, "broadcast", None)
        # The row's id is chosen here, not by the allocation: a Stop
        # landing as the allocation returns is raised before its result
        # is bound, and the cleanup would finish an id naming no row
        # while the committed one stayed "running".  When the allocation
        # fails, or a Stop lands in it before its commit, the id names
        # no row and every cleanup step keyed by it writes nothing.
        task_id = uuid.uuid4().hex
        try:
            _add_task(
                history_prompt, chat_id=self._chat_id, extra=early_extra, task_id=task_id,
            )
            # The publication's lock wait is interruptible, so it lives
            # inside the protected region: a Stop landing in it used to
            # skip the whole cleanup, and the runner (which finds the
            # row through ``last_task_id``) never learnt the id either.
            locked_call(self._task_id_lock, setattr, self, "_last_task_id", task_id)
            if run_printer is not None:
                tl = getattr(run_printer, "_thread_local", None)
                if tl is not None:
                    previous_task_id = getattr(tl, "task_id", "") or ""
                    tl.task_id = task_id
                allocated = getattr(run_printer, "agent_task_allocated", None)
                if allocated is not None:
                    allocated(self, task_id, self._chat_id)
                if self._subagent_info is not None:
                    _safe_broadcast(broadcast, {
                        "type": "new_tab",
                        "task_id": task_id,
                        "parent_tab_id": self._subagent_info.get("parent_tab_id", ""),
                        "taskId": "",
                    })
                start_rec = getattr(run_printer, "start_recording", None)
                if start_rec is not None:
                    start_rec()
                _safe_broadcast(broadcast, {"type": "tasks_updated", "taskId": ""})
            if _on_task_id_allocated is not None:
                try:
                    _on_task_id_allocated(task_id, self._chat_id)
                except Exception:
                    logger.warning(
                        "on_task_id_allocated(%r) raised", task_id, exc_info=True,
                    )
            # A fan-out parent reads the child's settings back from the
            # broadcast event for the ``ran:`` line of its result.
            task_settings = self._task_settings_payload(
                model=resolved_model,
                work_dir=resolved_work_dir,
                is_parallel=run_is_parallel,
                max_budget=resolved_budget,
                start_ts=start_ts_ms,
                task_id=task_id,
                tool_profile=effective_profile,
            )
            if run_printer is None:
                # Headless (a fan-out child without a printer): persist
                # the event ourselves, as :meth:`_persist_replay_events_if_missing`
                # does for the prompt and result.
                _append_chat_event(
                    {"type": "task_settings", "settings": task_settings},
                    task_id=task_id,
                )
            # Emitted AFTER on_task_id_allocated: the server's
            # WebPrinter only sends a task event's stamped copies
            # to tabs subscribed via ``subscribe_tab``, which that
            # callback performs.  The event is also recorded and
            # persisted (``task_settings`` is a display event), so
            # replays and shares repopulate the static task
            # panel's settings info.
            _safe_broadcast(
                broadcast, {"type": "task_settings", "settings": task_settings},
            )

            run_started = True
            result = super().run(
                model_name=resolved_model,
                prompt_template=agent_prompt,
                system_prompt=system_prompt,
                tools=tools,
                max_steps=max_steps,
                max_budget=max_budget,
                model_config=model_config,
                work_dir=work_dir,
                printer=printer,
                max_sub_sessions=max_sub_sessions,
                docker_image=docker_image,
                web_tools=web_tools,
                is_parallel=is_parallel,
                verbose=verbose,
                current_editor_file=current_editor_file,
                attachments=attachments,
                ask_user_question_callback=ask_user_question_callback,
                base_system_prompt=base_system_prompt,
                append_basic_tools=append_basic_tools,
                inherited_tools=inherited_tools,
                llm_call_hook=llm_call_hook,
                tool_call_hook=tool_call_hook,
                llm_response_hook=llm_response_hook,
                use_memory=use_memory,
                tool_profile=effective_profile,
                live_browser=live_browser,
                prompt_suffix=prompt_suffix,
                system_prompt_hook=system_prompt_hook,
                tools_hook=tools_hook,
                task_begun=True,
                compaction=compaction,
            )
            parsed = _parse_result(result)
            result_summary = _result_summary_text(result, parsed)
            completed = True
            return result
        except Exception:
            result_summary = "Task failed"
            raise
        except BaseException:
            result_summary = "Task interrupted"
            raise
        finally:
            # The cleanup steps are nested ``try``/``finally`` regions, as
            # in ``SorcarAgent._finish_run``: the runner's Stop injects
            # ``KeyboardInterrupt`` again while the thread is alive, and
            # one landing in a step (the printer's finish hook waiting
            # for ``STATE_LOCK``, say) must not skip the steps after it,
            # or the recording stays open, the calling thread keeps the
            # child's task binding and the row its running state.  The
            # first exception a step raised is re-raised after the last
            # step when the run was returning normally (the Stop must
            # still be honoured), or logged and dropped when the run is
            # already unwinding on an exception.
            first: BaseException | None = None
            try:
                # ``SorcarAgent.run``'s cleanup folds the pre-run classifier's
                # spend, but a run stopped or failing before ``super().run``
                # never gets there.  The fold is keyed and consumes the
                # record, so after a normal run it is a no-op; a Stop
                # landing in it gets it one retry (_retried_cleanup_step).
                # The row's id is passed: a Stop can land before it is
                # published as ``last_task_id``, which then still names the
                # previous row.
                first = _retried_cleanup_step(
                    first, "_fold_classifier_usage",
                    functools.partial(self._fold_classifier_usage, task_id),
                )
            finally:
                self._last_result_summary = result_summary
                try:
                    if run_printer is not None:
                        try:
                            _retire_printer_task(self, run_printer, task_id)
                        except BaseException:
                            # A Stop raised as the call was entered kept
                            # all of it from running.  The repeat, made at
                            # once, is past that entry before a Stop can
                            # be raised again (see _retire_printer_task).
                            _retire_printer_task(self, run_printer, task_id)
                            raise
                except BaseException as exc:  # noqa: BLE001
                    first = _first_cleanup_error(first, "_retire_printer_task", exc)
                finally:
                    try:
                        if run_printer is not None:
                            tl = getattr(run_printer, "_thread_local", None)
                            if tl is not None and getattr(tl, "task_id", "") == task_id:
                                tl.task_id = previous_task_id
                    except BaseException as exc:  # noqa: BLE001
                        first = _first_cleanup_error(first, "restore task binding", exc)
                    finally:
                        # A row whose id was never published (a Stop in
                        # the publication above) is unknown to the
                        # runner that would otherwise persist it, so it
                        # is finalised here even under ``_skip_persistence``.
                        if not _skip_persistence or self._last_task_id != task_id:
                            # One step, retried once when a Stop lands in
                            # it (_retried_cleanup_step): both writes are
                            # updates of the row by id, so the retry
                            # cannot double anything, and without it a
                            # row left unfinished would keep cost 0.  The
                            # retry leaves a row the first attempt did
                            # finish alone (a Stop right after its write).
                            attempts = 0

                            def final_save() -> None:
                                nonlocal attempts
                                attempts += 1
                                # The agent's own verdict (sub-agents and
                                # CLI runs, which no server runner
                                # persists) rides on the row with the
                                # summary, so the history sidebar shows a
                                # finish(success=False) run as failed.
                                verdict = parsed.get("success") if parsed else None
                                _save_task_result(
                                    task_id=task_id,
                                    result=result_summary,
                                    success=verdict if isinstance(verdict, bool) else None,
                                )
                                # Once ``super().run`` started, the live
                                # model settings are this run's (``_reset``
                                # resolved them); before that point they
                                # are unset or the previous run's, so the
                                # row keeps this run's resolved settings.
                                # Usage is always read from the ledger:
                                # its epoch began right before the row was
                                # allocated, so a task stopped during
                                # setup still records any side-channel
                                # spend banked into it.  The per-run
                                # resolved work_dir and budget are used
                                # either way: ``_reset`` sets the
                                # attributes to exactly these values on
                                # success, and ``self.work_dir`` /
                                # ``self.max_budget`` would otherwise carry
                                # a blank or the previous run's
                                # directory/budget into this row.
                                if run_started:
                                    final_model = self._launch_model_name or resolved_model
                                    final_is_parallel = self._is_parallel
                                else:
                                    final_model = resolved_model
                                    final_is_parallel = run_is_parallel
                                extra_payload = self._build_extra_payload(
                                    model=final_model,
                                    work_dir=resolved_work_dir,
                                    is_parallel=final_is_parallel,
                                    max_budget=resolved_budget,
                                )
                                # One coherent triple (see usage_snapshot):
                                # three property reads could tear across a
                                # concurrent sub-task fold.
                                # Read and saved under TASK_USAGE_LOCK, so
                                # a sub-task's late charge either lands on
                                # the ledger before this read or finds the
                                # finished row and adds to it
                                # (charge_side_channel_usage).
                                locked_call(
                                    TASK_USAGE_LOCK, self._save_final_usage, extra_payload,
                                    task_id, agent_prompt, parsed, result_summary,
                                    unless_finished=attempts > 1,
                                )

                            first = _retried_cleanup_step(first, "final save", final_save)
            if first is not None and completed:
                raise first
