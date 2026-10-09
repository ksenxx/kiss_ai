# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Stateful Sorcar agent with chat-session persistence.

Subclasses :class:`SorcarAgent` to add multi-turn chat-session state
management — the same workflow that the VS Code extension performs in
``VSCodeServer._run_task()``, but as a standalone reusable Python agent.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar._concurrency import _race_delay
from kiss.agents.sorcar.bare_path_task import with_open_directive
from kiss.agents.sorcar.git_worktree import strip_worktree_suffix
from kiss.agents.sorcar.persistence import (
    TASK_USAGE_LOCK,
    _add_task,
    _allocate_chat_id,
    _append_chat_event,
    _load_chat_context,
    _load_task_chain_context,
    _record_frequent_task,
    _save_task_extra,
    _save_task_result,
    _task_has_transcript_events,
)
from kiss.agents.sorcar.relentless_agent import DEFAULT_MAX_BUDGET
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core._version import __version__
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.printer import parse_result_yaml

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


def _dir_inside_worktree(work_dir: str, wt_dir: object) -> bool:
    """Return True when *work_dir* lies inside the agent's own worktree dir.

    Used by :meth:`ChatSorcarAgent.run` to decide the persisted
    ``is_worktree`` flag for worktree-capable subclasses whose ``run()``
    consumed the ``use_worktree`` kwarg before delegating:
    :class:`~kiss.agents.sorcar.worktree_sorcar_agent.WorktreeSorcarAgent`
    redirects ``work_dir`` into ``self._wt_dir`` only when a worktree
    was actually set up for the current run, so containment of the
    effective ``work_dir`` in ``wt_dir`` is the ground truth (a stale
    pending worktree from an earlier run does not match the current
    run's plain ``work_dir``, and an explicit ``use_worktree=False``
    fallback leaves ``work_dir`` untouched).

    Args:
        work_dir: The effective working directory of the current run.
        wt_dir: The agent's current worktree directory (``Path`` or
            ``None`` — typed loosely because plain ``ChatSorcarAgent``
            has no ``_wt_dir`` attribute).

    Returns:
        True only when both paths exist as strings and *work_dir*
        resolves to a path at or below *wt_dir*.
    """
    if not work_dir or wt_dir is None:
        return False
    try:
        return Path(work_dir).resolve().is_relative_to(Path(str(wt_dir)).resolve())
    except (OSError, ValueError):
        return False


def _extract_result_summary(result: str) -> str:
    """Return the persistable summary text for a finished run's *result*.

    Parses *result* as YAML and extracts its ``summary`` field for the
    task-history record.  Handles every shape LLMs emit:

    * dict with a string ``summary`` — returned as-is;
    * dict with ``summary: null`` (or no ``summary`` key) — ``""``;
    * dict with a list/mapping/scalar ``summary`` — its YAML text form
      (passing the raw object to ``_save_task_result`` would raise
      ``sqlite3.ProgrammingError``, destroying the task's successful
      return value);
    * valid YAML that is not a dict, or unparseable text — the raw
      text capped at 500 characters (otherwise the task history would
      record an empty result).

    Args:
        result: The raw string returned by the agent run.

    Returns:
        The summary text to persist (possibly empty, never ``None``).
    """
    try:
        result_yaml = yaml.safe_load(result)
        if isinstance(result_yaml, dict):
            summary_val = result_yaml.get("summary", "")
            if isinstance(summary_val, str):
                return summary_val
            if summary_val is None:
                return ""
            dumped: str = yaml.safe_dump(summary_val, sort_keys=False)
            return dumped.strip()
        return result[:500] if result else ""
    except Exception:
        return result[:500] if result else ""


class ChatSorcarAgent(SorcarAgent):
    """SorcarAgent with chat-session state management.

    Maintains a ``chat_id`` and automatically loads prior chat context,
    persists tasks and results to ``history.db``, and augments prompts
    with previous session context — replicating the stateful workflow
    from the VS Code extension as a standalone reusable agent.
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
        # The run's effective configuration beyond model / work dir /
        # budget (see kiss.agents.sorcar.run_config.RUN_CONFIG_KEYS):
        # set per run by the task runner or the fan-out engine and
        # folded into the ``task_settings`` event.  Empty for a run
        # that is neither a daemon task nor a sub-agent.
        self.run_config: dict[str, Any] = {}
        # The ``task_settings`` payload of the latest run (model, work
        # dir, budget and ``run_config``); empty before the first run.
        self.task_settings: dict[str, object] = {}
        self._last_task_id: str | None = None
        self._last_user_prompt: str = ""
        self._last_result_summary: str = ""
        self._task_id_lock: threading.RLock = threading.RLock()
        # Whether the CURRENT run executes inside a git worktree —
        # computed at the top of :meth:`run` and read by the
        # system-prompt settings hook and the ``task_settings``
        # broadcast.
        self._run_is_worktree: bool = False

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
        with self._task_id_lock:
            return self._last_task_id or ""

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
        """Reset to a new chat session (equivalent to VS Code 'Clear').

        Also drops any pending one-shot :meth:`resume_from_task_id`
        seed: a brand-new chat must never have its first prompt
        augmented with the previous task's parent-chain context.
        """
        self._chat_id = ""
        self._context_task_id = ""

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

    def build_chat_prompt(self, prompt: str) -> str:
        """Load chat context and augment prompt with previous tasks/results.

        Args:
            prompt: The original task prompt.

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
            return "# Task\n" + prompt
        if len(chat_context) > MAX_TASKS:
            del chat_context[2:2 + len(chat_context) - MAX_TASKS]
        if DEFAULT_CONFIG.chat_history_digest:
            body = _digest_history(chat_context)
        else:
            body = _format_history(chat_context, full_from=0)
        return (
            "## Previous tasks and results from the chat session for reference\n\n"
            + body + "\n\n---\n\n# Task (work on it now)\n\n" + prompt
        )

    def _build_extra_payload(
        self,
        model: str,
        work_dir: str,
        is_parallel: bool,
        is_worktree: bool,
        max_budget: float | None = None,
        auto_commit_mode: bool = True,
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
            is_worktree: Whether worktree isolation is in effect.
            max_budget: The run's resolved budget cap in USD, or None
                to omit it from the payload.
            auto_commit_mode: The run's effective auto-commit toggle.
                ``persistence._add_task`` maps an ABSENT toggle to
                manual-commit while the schema default is ON, so the
                early row must always carry the real value or a task
                viewed mid-run (killed, or running in another process)
                is labelled manual-commit.

        Returns:
            The extra-payload dict.
        """
        payload: dict[str, object] = {
            "model": model,
            "work_dir": strip_worktree_suffix(work_dir),
            "version": __version__,
            "is_parallel": is_parallel,
            "is_worktree": is_worktree,
            "auto_commit_mode": auto_commit_mode,
        }
        if max_budget is not None:
            payload["max_budget"] = max_budget
        if self._subagent_info is not None:
            payload["subagent"] = self._subagent_info
        if self.sea_name:
            payload["sea"] = self.sea_name
        return payload

    def _system_prompt_task_settings(self) -> dict[str, str]:
        """Extend the settings with this chat run's identity and modes.

        Adds the worktree mode, the chat / task ids allocated at the
        top of :meth:`run`, and the sub-agent parentage — everything
        this class knows that the base agents do not.

        Returns:
            The base label → value pairs plus the chat-level ones.
        """
        settings = super()._system_prompt_task_settings()
        settings["Worktree mode"] = (
            "worktree" if self._run_is_worktree else "no worktree"
        )
        settings["Chat id"] = self._chat_id
        settings["Task id"] = self.last_task_id
        sub = self._subagent_info
        settings["Is subagent"] = "yes" if sub is not None else "no"
        parent_id = str((sub or {}).get("parent_task_id") or "")
        if parent_id:
            settings["Parent task id"] = parent_id
        return settings

    def _task_settings_payload(
        self,
        model: str,
        work_dir: str,
        is_parallel: bool,
        is_worktree: bool,
        max_budget: float | None,
        start_ts: int,
        task_id: str,
        tool_profile: str = "",
    ) -> dict[str, object]:
        """Build the ``task_settings`` display-event payload.

        The shape mirrors the task-history sidebar's session fields so
        the chat webview can render the static task panel's info line
        exactly like a history row (see ``renderHistory`` and
        ``taskPanelInfoHTML`` in ``media/main.js``).

        Args:
            model: Model name the run was asked for ("" when the
                caller left it to the default).
            work_dir: Working directory of the run.
            is_parallel: Whether parallel sub-agents are enabled.
            is_worktree: Whether worktree isolation is in effect.
            max_budget: The run's budget cap in USD, or None when
                unstated.
            start_ts: The run's start timestamp (ms since epoch) — the
                same value persisted as the row's ``startTs``.
            task_id: The freshly allocated ``task_history`` row id.
            tool_profile: The effective tool profile (``"full"``, a
                profile name or ``""`` when unknown); recorded whenever
                it is known or the run carries a ``run_config``.

        Returns:
            The settings dict carried by the event: the keys above,
            ``chat_id``, ``is_subagent``, ``parent_task_id`` when there
            is one, ``sea`` when the run executes a SEA and
            the :attr:`run_config` keys (see
            :data:`kiss.agents.sorcar.run_config.RUN_CONFIG_KEYS`).
        """
        payload: dict[str, object] = {
            "model": model,
            "work_dir": strip_worktree_suffix(work_dir),
            "is_parallel": is_parallel,
            "is_worktree": is_worktree,
            "start_ts": start_ts,
            "chat_id": self._chat_id,
            "task_id": task_id,
            "is_subagent": self._subagent_info is not None,
        }
        if max_budget is not None:
            payload["max_budget"] = max_budget
        parent_id = str((self._subagent_info or {}).get("parent_task_id") or "")
        if parent_id:
            payload["parent_task_id"] = parent_id
        if self.sea_name:
            payload["sea"] = self.sea_name
        payload.update(self.run_config)
        if tool_profile != "" or "tool_profile" in payload:
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
        result_raw: str,
        result_summary: str,
        usage: tuple[float, int, int],
    ) -> None:
        """Persist a minimal replayable event stream when none was recorded.

        Runs that happen inside a chat webview stream every agent event
        through a recording printer (the VS Code server's ``JsonPrinter``
        / ``WebPrinter``), which persists them to the ``events`` table.
        Runs that happen OUTSIDE a chat webview — the third-party
        channel agents, or a remote webapp invocation with a
        non-recording printer — leave the ``events`` table empty, so the
        chat webview would load a blank session even though the task and
        its result are in ``task_history``.

        This synthesizes the events the webview needs to render the
        exchange — a ``prompt`` event (the user's task), a ``result``
        event (the agent's summary / success / cost) and, when the agent
        proposed one via ``finish(suggested_next_task=...)``, a
        ``followup_suggestion`` event — but only when the task has no
        transcript events yet (the run's ``task_settings`` metadata
        event, persisted before any output, is ignored), so a recording
        printer's full event stream is never duplicated.

        Args:
            task_id: Stable ``task_history`` row id for this run.
            prompt: The prompt the agent actually ran with (chat-context
                augmented), mirroring the ``prompt`` event a recording
                printer would have persisted.
            result_raw: The raw YAML result string returned by the run
                (used to recover ``success`` / ``is_continue``).
            result_summary: The extracted human-readable summary text.
            usage: One coherent ``(budget_used, total_tokens_used,
                total_steps)`` triple (:meth:`usage_snapshot`) to stamp
                on the ``result`` event.
        """
        if _task_has_transcript_events(task_id):
            return
        if prompt:
            _append_chat_event({"type": "prompt", "text": prompt}, task_id=task_id)
        cost, tokens, steps = usage
        event: dict[str, object] = {
            "type": "result",
            "text": result_summary or "(no result)",
            "total_tokens": tokens,
            "cost": f"${cost:.4f}",
            "cost_usd": cost,
            "step_count": steps,
        }
        parsed = parse_result_yaml(result_raw) if result_raw else None
        if parsed:
            event["success"] = parsed.get("success")
            event["is_continue"] = bool(parsed.get("is_continue", False))
            event["summary"] = str(parsed["summary"])
        else:
            event["summary"] = result_summary or ""
        _append_chat_event(event, task_id=task_id)
        suggestion = str((parsed or {}).get("suggested_next_task") or "").strip()
        if suggestion:
            _append_chat_event(
                {"type": "followup_suggestion", "text": suggestion}, task_id=task_id,
            )

    def _begin_run_usage_epoch(self) -> None:
        """Keep the usage epoch :meth:`run` started at row allocation.

        Swapping it again at ``_reset`` would discard spend a side
        channel banked into it while the task was still setting up
        (e.g. classifying).
        """

    def run(  # type: ignore[override]
        self,
        prompt_template: str = "",
        **kwargs: Any,
    ) -> str:
        """Run the agent with chat-session context management.

        Loads prior chat context, persists the new task, augments the
        prompt with previous tasks/results, runs the underlying agent,
        and saves the result back to history.

        Only the result summary is persisted here.  Callers that record
        chat events (e.g. the VS Code server) persist events incrementally
        via :func:`~kiss.agents.sorcar.persistence._append_chat_event`.

        Args:
            prompt_template: The task prompt.
            **kwargs: All other arguments forwarded to ``SorcarAgent.run()``.

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        skip_persistence = kwargs.pop("_skip_persistence", False)
        on_task_id_allocated = kwargs.pop("_on_task_id_allocated", None)
        # A caller (currently the slash-command dispatch in
        # ``TaskRunner._run_task_inner``) can supply the raw user text
        # to use for every history / display / frequent-task write,
        # while ``prompt_template`` carries an internal directive the
        # LLM must actually see.  Absent, the two are the same and
        # ``prompt_template`` is used everywhere — the original
        # behaviour.
        raw_history_prompt = kwargs.pop("_history_prompt", None)
        history_prompt = (
            raw_history_prompt
            if isinstance(raw_history_prompt, str) and raw_history_prompt
            else prompt_template
        )
        if self._chat_id == "":
            self._chat_id = _allocate_chat_id()
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
        # Only for a task the user typed into a chat box, though: the
        # task runner passes ``_open_bare_path=False`` for a run it
        # starts on behalf of a SEA (``seaPath``: an SEA
        # dispatched by ``/git_extract_knowledge /path/to/repo`` or
        # ``/review_paper /path/to/paper.pdf`` receives the path as its
        # whole task and defines what it means) or of another agent
        # (``parentTaskId``: a ``run_agent`` sub-task), and a custom
        # base system prompt (an SEA's ``system_prompt()``) suppresses
        # it as well, so a direct ``ChatSorcarAgent.run`` of an agent
        # script is not sent off to ``xdg-open`` the path either.
        open_bare_path = bool(kwargs.pop("_open_bare_path", True)) and not str(
            kwargs.get("base_system_prompt") or "",
        ).strip()
        agent_prompt = self.build_chat_prompt(
            with_open_directive(prompt_template, kwargs.get("work_dir") or ".")
            if open_bare_path
            else prompt_template,
        )

        # Consumed, never believed: ``SorcarAgent.run`` has no such
        # parameter, and whether a worktree EXISTS is the only honest
        # answer for the history badge — a caller asking for one does
        # not make one appear on this class.
        kwargs.pop("use_worktree", None)
        is_worktree = self.uses_worktree and _dir_inside_worktree(
            kwargs.get("work_dir", "") or "",
            getattr(self, "_wt_dir", None),
        )
        self._run_is_worktree = is_worktree
        raw_budget = kwargs.get("max_budget")
        try:
            run_max_budget = None if raw_budget is None else float(raw_budget)
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
        resolved_model = kwargs["model_name"] = self._resolve_model_name(
            kwargs.get("model_name"),
        )
        resolved_budget = (
            run_max_budget if run_max_budget is not None else DEFAULT_MAX_BUDGET
        )
        resolved_work_dir = str(Path(kwargs.get("work_dir") or ".").resolve())
        run_is_parallel = bool(kwargs.get("is_parallel", True))
        # The effective toggle: ``WorktreeSorcarAgent.run`` binds
        # ``auto_commit_enabled`` from its kwarg / config BEFORE
        # delegating here; this class has no toggle, and absent means
        # ON (schema default + legacy migration).
        auto_commit_mode = bool(getattr(self, "auto_commit_enabled", True))
        # The row's start is the run's start, not the row's: the daemon
        # (``TaskRunner._run_task_inner``) stamps a ``<task>`` block
        # when it begins and passes it as ``_start_ms``, so the row,
        # its ``task_settings`` event, the live ``status`` broadcast
        # and ``task_done`` all count from one instant — the worktree
        # setup and SEA import that precede this row would otherwise
        # make a client that attaches mid-run start its timer late.
        start_ts_ms = int(kwargs.pop("_start_ms", 0) or 0) or int(time.time() * 1000)

        early_extra = self._build_extra_payload(
            model=resolved_model,
            work_dir=resolved_work_dir,
            is_parallel=run_is_parallel,
            is_worktree=is_worktree,
            max_budget=resolved_budget,
            auto_commit_mode=auto_commit_mode,
        )
        early_extra["startTs"] = start_ts_ms

        # This run's usage epoch starts BEFORE its row exists: once the
        # row is visible a ``/update`` or ``/ask`` side channel may
        # capture the epoch and later bank its spend into it, so the
        # epoch must not be swapped again by ``_reset`` (see
        # ``_begin_run_usage_epoch``).  Classifier spend is held apart
        # until ``SorcarAgent.run`` folds it, so nothing is lost here.
        self.reset_usage()
        task_id, self._chat_id = _add_task(
            history_prompt, chat_id=self._chat_id, extra=early_extra,
        )
        with self._task_id_lock:
            self._last_task_id = task_id
        printer = kwargs.get("printer") or getattr(self, "printer", None)
        task_key = str(task_id)
        result_summary = ""
        result_raw = ""
        # Whether ``super().run`` was reached.  Until then every live
        # setting the final save would read — ``_launch_model_name``,
        # ``model_name``, ``_is_parallel`` — is either unset (fresh
        # agent) or the PREVIOUS run's (reused agent), so the final save
        # must fall back to this run's resolved settings.  (The usage
        # ledger is already this run's: see ``reset_usage`` above.)
        run_started = False
        # Every remaining setup step (state registration, printer
        # wiring, frequent-task recording, ...) must run inside the try
        # below: an exception in any of them would otherwise bypass the
        # cleanup and leave a permanently "running" task behind (F-14).
        # The calling thread's previous task binding is restored on exit:
        # a side-channel task (``/update``, merge conflict resolution)
        # runs on a worker thread that then attributes the child's spend
        # to the PARENT agent, and that write lands under whatever
        # ``task_id`` the thread carries.  Resetting to ``""`` instead of
        # the previous value stored the parent's budget offset under an
        # orphan key, so the parent's tab under-counted until its own
        # thread rewrote the offset.
        previous_task_id = ""
        broadcast = getattr(printer, "broadcast", None)
        try:
            if printer is not None:
                tl = getattr(printer, "_thread_local", None)
                if tl is not None:
                    previous_task_id = getattr(tl, "task_id", "") or ""
                    tl.task_id = task_key
                allocated = getattr(printer, "agent_task_allocated", None)
                if allocated is not None:
                    allocated(self, task_id, self._chat_id)
                if self._subagent_info is not None:
                    _safe_broadcast(broadcast, {
                        "type": "new_tab",
                        "task_id": task_id,
                        "parent_tab_id": self._subagent_info.get("parent_tab_id", ""),
                        "taskId": "",
                    })
                start_rec = getattr(printer, "start_recording", None)
                if start_rec is not None:
                    start_rec()
                _safe_broadcast(broadcast, {"type": "tasks_updated", "taskId": ""})
            if on_task_id_allocated is not None:
                try:
                    on_task_id_allocated(task_id, self._chat_id)
                except Exception:
                    logger.warning(
                        "on_task_id_allocated(%r) raised", task_id, exc_info=True,
                    )
            # Kept on the agent too: a fan-out parent reads the child's
            # settings back for the ``ran:`` line of its result.  The
            # tool profile is the EFFECTIVE one (``SorcarAgent.run``
            # resolves the same rule from the same name a moment later):
            # a reviewer-marked sub-agent asked for no profile runs
            # ``review``, not the full toolset.
            self._tool_profile_name = str(kwargs.get("tool_profile") or "")
            self.task_settings = self._task_settings_payload(
                model=resolved_model,
                work_dir=resolved_work_dir,
                is_parallel=run_is_parallel,
                is_worktree=is_worktree,
                max_budget=resolved_budget,
                start_ts=start_ts_ms,
                task_id=task_id,
                tool_profile=self._tool_profile(prompt_template),
            )
            if printer is None and task_id:
                # Headless (a fan-out child without a printer): persist
                # the event ourselves, as :meth:`_persist_replay_events_if_missing`
                # does for the prompt and result.
                _append_chat_event(
                    {"type": "task_settings", "settings": self.task_settings},
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
                broadcast, {"type": "task_settings", "settings": self.task_settings},
            )
            if self._subagent_info is None:
                _record_frequent_task(history_prompt)

            run_started = True
            result = super().run(prompt_template=agent_prompt, **kwargs)
            result_raw = result if isinstance(result, str) else ""
            result_summary = _extract_result_summary(result)
            return result
        except Exception:
            result_summary = "Task failed"
            raise
        except BaseException:
            result_summary = "Task interrupted"
            raise
        finally:
            self._last_result_summary = result_summary
            if printer is not None:
                finished = getattr(printer, "agent_task_finished", None)
                if finished is not None:
                    try:
                        finished(self, task_key)
                    except Exception:
                        logger.warning(
                            "agent_task_finished(%r) raised", task_key, exc_info=True,
                        )
                stop_rec = getattr(printer, "stop_recording", None)
                if stop_rec is not None:
                    try:
                        stop_rec()
                    except Exception:
                        pass
                tl = getattr(printer, "_thread_local", None)
                if tl is not None and getattr(tl, "task_id", "") == task_key:
                    tl.task_id = previous_task_id
            if not skip_persistence:
                _save_task_result(task_id=task_id, result=result_summary)
                # Once ``super().run`` started, the live model settings
                # are this run's (``_reset`` resolved them); before that
                # point they are unset or the previous run's, so the row
                # keeps this run's resolved settings.  Usage is always
                # read from the ledger: its epoch began right before the
                # row was allocated, so a task stopped during setup
                # still records any side-channel spend banked into it.
                # The per-run resolved work_dir and
                # budget are used either way: ``_reset`` sets the
                # attributes to exactly these values on success, and
                # ``self.work_dir`` / ``self.max_budget`` would otherwise
                # carry a blank or the previous run's directory/budget
                # into this row.
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
                    is_worktree=is_worktree,
                    max_budget=resolved_budget,
                    auto_commit_mode=auto_commit_mode,
                )
                # One coherent triple (see usage_snapshot): three
                # property reads could tear across a concurrent
                # abandoned-subagent reclaim.  Read and saved under
                # TASK_USAGE_LOCK, so a sub-task's late charge either
                # lands on the ledger before this read or finds the
                # finished row and adds to it (charge_side_channel_usage).
                with TASK_USAGE_LOCK:
                    final_cost, final_tokens, final_steps = self.usage_snapshot()
                    _race_delay()  # test hook: widens the snapshot-to-save window
                    extra_payload["tokens"] = final_tokens
                    extra_payload["cost"] = round(final_cost, 6)
                    extra_payload["steps"] = final_steps
                    # The run is over: stamp its end so sub-agents (which
                    # no server runner finalises) get a duration too.
                    extra_payload["endTs"] = int(time.time() * 1000)
                    _save_task_extra(extra_payload, task_id=task_id)
                self._persist_replay_events_if_missing(
                    task_id=task_id,
                    prompt=agent_prompt,
                    result_raw=result_raw,
                    result_summary=result_summary,
                    usage=(final_cost, final_tokens, final_steps),
                )
