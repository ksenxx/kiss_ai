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
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar.git_worktree import strip_worktree_suffix
from kiss.agents.sorcar.persistence import (
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
from kiss.core.printer import parse_result_yaml

MAX_TASKS = 10


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


def summary(description: str) -> str:
    """MANDATORY every 5 steps: summarize your last 6 steps of work.

    Your tool call on every step that is a multiple of 5 (step 5, 10,
    15, ...) MUST be this tool, BEFORE any other tool call (including
    finish).  This requirement applies to every task, no matter how
    simple, and is never overridden by the task prompt.

    The tool itself performs no action: the chat webview groups the
    preceding six event panels under this call's panel and collapses
    them, hiding the step-by-step detail while keeping the
    description visible as a running digest for the user.  The
    description is rendered as formatted Markdown in the panel.

    Args:
        description: Natural language summary in 5-10 sentences of
            what the agent did in the last 6 steps, written in
            Markdown format (use bullet lists for the steps, and
            ``**bold**`` / backtick code spans where helpful).

    Returns:
        A short confirmation string.
    """
    del description
    return "Summary recorded."


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
    persists tasks and results to ``sorcar.db``, and augments prompts
    with previous session context — replicating the stateful workflow
    from the VS Code extension as a standalone reusable agent.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._chat_id: str = ""
        self._context_task_id: str = ""
        self._subagent_info: dict[str, object] | None = None
        # Frontend tab this agent's events belong to.  The fan-out
        # engine assigns each sub-agent its own synthetic tab id, so
        # the attribute lives here rather than on the worktree
        # subclass that also sets it.
        self._tab_id: str = ""
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

    def _get_tools(self) -> list:
        """Extend the base toolset with the no-op ``summary`` tool.

        The ``summary`` tool lets the model periodically condense its
        recent activity; the chat webview reacts to the persisted
        ``tool_call`` event by nesting and collapsing the preceding
        event panels (see ``media/main.js``).  The every-5-steps
        cadence is requested by the SYSTEM.md instructions and this
        tool's docstring only — there is no mechanical enforcement.

        Returns:
            The base tools plus :func:`summary`.
        """
        return [*super()._get_tools(), summary]

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
        parts = ["## Previous tasks and results from the chat session for reference\n"]
        if len(chat_context) > MAX_TASKS:
            del chat_context[2:2 + len(chat_context) - MAX_TASKS]
        for i, entry in enumerate(chat_context, 1):
            parts.append(f"### Task {i}\n{entry['task']}")
            if entry.get("result"):
                parts.append(f"### Result {i}\n{entry['result']}")
        parts.append("---\n")
        return "\n\n".join(parts) + "# Task (work on it now)\n\n" + prompt

    def _build_extra_payload(
        self,
        model: str,
        work_dir: str,
        is_parallel: bool,
        is_worktree: bool,
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
            is_worktree: Whether worktree isolation is in effect.
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
            "is_worktree": is_worktree,
        }
        if max_budget is not None:
            payload["max_budget"] = max_budget
        if self._subagent_info is not None:
            payload["subagent"] = self._subagent_info
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

        Returns:
            The settings dict carried by the event.
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
        return payload

    def _persist_replay_events_if_missing(
        self,
        task_id: str,
        prompt: str,
        result_raw: str,
        result_summary: str,
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

        This synthesizes the two events the webview needs to render the
        exchange — a ``prompt`` event (the user's task) and a ``result``
        event (the agent's summary / success / cost) — but only when the
        task has no transcript events yet (the run's ``task_settings``
        metadata event, persisted before any output, is ignored), so a
        recording printer's full event stream is never duplicated.

        Args:
            task_id: Stable ``task_history`` row id for this run.
            prompt: The prompt the agent actually ran with (chat-context
                augmented), mirroring the ``prompt`` event a recording
                printer would have persisted.
            result_raw: The raw YAML result string returned by the run
                (used to recover ``success`` / ``is_continue``).
            result_summary: The extracted human-readable summary text.
        """
        if _task_has_transcript_events(task_id):
            return
        prompt_text = prompt or ""
        if prompt_text:
            _append_chat_event(
                {"type": "prompt", "text": prompt_text}, task_id=task_id,
            )
        event: dict[str, object] = {
            "type": "result",
            "text": result_summary or "(no result)",
            "total_tokens": int(getattr(self, "total_tokens_used", 0) or 0),
            "cost": f"${float(getattr(self, 'budget_used', 0.0) or 0.0):.4f}",
            "step_count": int(getattr(self, "total_steps", 0) or 0),
        }
        parsed = parse_result_yaml(result_raw) if result_raw else None
        if parsed:
            event["success"] = parsed.get("success")
            event["is_continue"] = bool(parsed.get("is_continue", False))
            event["summary"] = str(parsed["summary"])
        else:
            event["summary"] = result_summary or ""
        _append_chat_event(event, task_id=task_id)

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
        self._last_user_prompt = prompt_template
        self._last_result_summary = ""

        agent_prompt = self.build_chat_prompt(prompt_template)

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
        # kwargs that the run later resolves differently.
        resolved_model = self._resolve_model_name(kwargs.get("model_name"))
        resolved_budget = (
            run_max_budget if run_max_budget is not None else DEFAULT_MAX_BUDGET
        )
        resolved_work_dir = str(Path(kwargs.get("work_dir") or ".").resolve())
        run_is_parallel = bool(kwargs.get("is_parallel", True))
        start_ts_ms = int(time.time() * 1000)

        early_extra = self._build_extra_payload(
            model=resolved_model,
            work_dir=resolved_work_dir,
            is_parallel=run_is_parallel,
            is_worktree=is_worktree,
            max_budget=resolved_budget,
        )
        early_extra["startTs"] = start_ts_ms

        task_id, self._chat_id = _add_task(
            prompt_template, chat_id=self._chat_id, extra=early_extra,
        )
        with self._task_id_lock:
            self._last_task_id = task_id
        printer = kwargs.get("printer") or getattr(self, "printer", None)
        task_key = str(task_id)
        result_summary = ""
        result_raw = ""
        # Every remaining setup step (state registration, printer
        # wiring, frequent-task recording, ...) must run inside the try
        # below: an exception in any of them would otherwise bypass the
        # cleanup and leave a permanently "running" task behind (F-14).
        try:
            if printer is not None:
                tl = getattr(printer, "_thread_local", None)
                if tl is not None:
                    tl.task_id = task_key
                allocated = getattr(printer, "agent_task_allocated", None)
                if allocated is not None:
                    allocated(self, task_id, self._chat_id)
                if self._subagent_info is not None:
                    broadcast = getattr(printer, "broadcast", None)
                    if broadcast is not None:
                        try:
                            sub_info = self._subagent_info or {}
                            parent_tab_id_payload = sub_info.get(
                                "parent_tab_id", "",
                            )
                            broadcast({
                                "type": "new_tab",
                                "task_id": task_id,
                                "parent_tab_id": parent_tab_id_payload,
                                "taskId": "",
                            })
                        except Exception:
                            pass
                start_rec = getattr(printer, "start_recording", None)
                if start_rec is not None:
                    start_rec()
                broadcast = getattr(printer, "broadcast", None)
                if broadcast is not None:
                    try:
                        broadcast({"type": "tasks_updated", "taskId": ""})
                    except Exception:
                        pass
            if on_task_id_allocated is not None:
                try:
                    on_task_id_allocated(task_id, self._chat_id)
                except Exception:
                    logging.getLogger(__name__).warning(
                        "on_task_id_allocated(%r) raised",
                        task_id,
                        exc_info=True,
                    )
            if printer is not None:
                # Emitted AFTER on_task_id_allocated: the server's
                # WebPrinter only sends a task event's stamped copies
                # to tabs subscribed via register_task_ui, which that
                # callback performs.  The event is also recorded and
                # persisted (``task_settings`` is a display event), so
                # replays and shares repopulate the static task
                # panel's settings info.
                broadcast = getattr(printer, "broadcast", None)
                if broadcast is not None:
                    try:
                        broadcast({
                            "type": "task_settings",
                            "settings": self._task_settings_payload(
                                model=resolved_model,
                                work_dir=resolved_work_dir,
                                is_parallel=run_is_parallel,
                                is_worktree=is_worktree,
                                max_budget=resolved_budget,
                                start_ts=start_ts_ms,
                                task_id=task_id,
                            ),
                        })
                    except Exception:
                        logging.getLogger(__name__).warning(
                            "task_settings broadcast raised", exc_info=True,
                        )
            if self._subagent_info is None:
                _record_frequent_task(prompt_template)

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
                        logging.getLogger(__name__).warning(
                            "agent_task_finished(%r) raised",
                            task_key,
                            exc_info=True,
                        )
                stop_rec = getattr(printer, "stop_recording", None)
                if stop_rec is not None:
                    try:
                        stop_rec()
                    except Exception:
                        pass
                tl = getattr(printer, "_thread_local", None)
                if tl is not None and getattr(tl, "task_id", "") == task_key:
                    tl.task_id = ""
            if not skip_persistence:
                _save_task_result(task_id=task_id, result=result_summary)
                # getattr defaults: when setup failed BEFORE super().run
                # ran _reset (e.g. a broken printer hook), the usage
                # fields do not exist yet; the persistence path must not
                # raise from this finally and mask the original error.
                extra_payload = self._build_extra_payload(
                    model=(
                        getattr(self, "_launch_model_name", "")
                        or getattr(self, "model_name", "")
                        or resolved_model
                    ),
                    work_dir=self.work_dir,
                    is_parallel=self._is_parallel,
                    is_worktree=is_worktree,
                    # The per-run resolved value, NOT ``self.max_budget``:
                    # on a reused agent whose setup failed before
                    # ``_reset``, the attribute still holds the PREVIOUS
                    # run's budget.  ``_reset`` sets the attribute to
                    # exactly this value, so nothing is lost on success.
                    max_budget=resolved_budget,
                )
                extra_payload["tokens"] = int(
                    getattr(self, "total_tokens_used", 0) or 0
                )
                extra_payload["cost"] = round(
                    float(getattr(self, "budget_used", 0.0) or 0.0), 6
                )
                _save_task_extra(extra_payload, task_id=task_id)
                self._persist_replay_events_if_missing(
                    task_id=task_id,
                    prompt=agent_prompt,
                    result_raw=result_raw,
                    result_summary=result_summary,
                )
