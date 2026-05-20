"""Stateful Sorcar agent with chat-session persistence.

Subclasses :class:`SorcarAgent` to add multi-turn chat-session state
management — the same workflow that the VS Code extension performs in
``VSCodeServer._run_task()``, but as a standalone reusable Python agent.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import yaml

from kiss.agents.sorcar.persistence import (
    _add_task,
    _load_chat_context,
    _load_task_chat_id,
    _record_frequent_task,
    _save_task_extra,
    _save_task_result,
)
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.sorcar_agent import SorcarAgent, _coerce_tasks

MAX_TASKS = 10


class ChatSorcarAgent(SorcarAgent):
    """SorcarAgent with chat-session state management.

    Maintains a ``chat_id`` and automatically loads prior chat context,
    persists tasks and results to ``sorcar.db``, and augments prompts
    with previous session context — replicating the stateful workflow
    from the VS Code extension as a standalone reusable agent.
    """

    # Process-global map of ``task_history.id`` (the database primary
    # key minted by :func:`_add_task`) → the :class:`ChatSorcarAgent`
    # instance currently executing that task.  Every ``run()`` call —
    # top-level chat tasks as well as parallel sub-agent tasks
    # spawned by :meth:`_run_tasks_parallel` — inserts itself here
    # immediately after the task row is written to ``sorcar.db`` and
    # removes itself in the ``finally`` block once ``run()`` returns
    # (or raises).  External observers can use this map to ask
    # "which agent (if any) is currently driving task X?" without
    # having to scan :attr:`_RunningAgentState.running_agent_states`
    # (which is keyed by frontend tab id and tracks per-tab UI
    # state, not per-task agents).  Insertions and removals on
    # distinct ``task_id`` keys never collide so no lock is required
    # — CPython dict ``__setitem__`` / ``pop`` are atomic under the
    # GIL.
    running_agents: dict[int, ChatSorcarAgent] = {}

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._chat_id: str = ""
        self._last_task_id: int | None = None
        # The most recent user task prompt seen by ``run()``.  Used
        # by auto-commit code paths to include the user's intent in
        # the generated commit message (see
        # :func:`~kiss.agents.vscode.helpers.generate_commit_message_from_diff`).
        # Empty string when the agent has not yet run any task.
        self._last_user_prompt: str = ""
        # Populated by ``_run_tasks_parallel`` on each sub-agent
        # before it runs.  The single ``parent_task_id`` field links
        # the sub-agent's ``task_history`` row back to the row of
        # the parent task that spawned it.  The presence of this
        # field on a row's ``extra.subagent`` payload is itself the
        # signal that the row is a sub-agent task; no separate
        # boolean is stored.  ``None`` on the top-level agent.
        self._subagent_info: dict[str, object] | None = None

    @property
    def chat_id(self) -> str:
        """Return the current chat session ID ("" means new session)."""
        return self._chat_id

    def new_chat(self) -> None:
        """Reset to a new chat session (equivalent to VS Code 'Clear')."""
        self._chat_id = ""

    def resume_chat(self, task: str) -> None:
        """Resume a previous chat session by looking up the task's chat_id.

        If the task has an associated ``chat_id`` in history, subsequent
        ``run()`` calls will continue that session.

        Args:
            task: The task description string to look up.
        """
        chat_id = _load_task_chat_id(task)
        if chat_id:
            self.resume_chat_by_id(chat_id)

    def resume_chat_by_id(self, chat_id: str) -> None:
        """Resume a chat session using a stable chat identifier.

        Args:
            chat_id: String chat session identifier to resume.
        """
        if chat_id:
            self._chat_id = chat_id

    def build_chat_prompt(self, prompt: str) -> str:
        """Load chat context and augment prompt with previous tasks/results.

        Args:
            prompt: The original task prompt.

        Returns:
            The augmented prompt with chat history prepended, or the
            original prompt if no prior context exists.
        """
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

    def _run_tasks_parallel(
        self,
        tasks: list[str],
        max_workers: int | None = None,
    ) -> list[str]:
        """Execute parallel tasks using ChatSorcarAgent sub-agents.

        Mirrors the structure of :func:`run_tasks_parallel` in
        :mod:`sorcar_agent` but adds three pieces of chat-specific
        behaviour the base helper cannot supply:

        1. Each sub-agent ``resume_chat_by_id`` s the parent's
           ``chat_id`` so all parallel tasks contribute to the same
           chat session history.
        2. The parent's ``task_history.id`` is recorded on every
           sub-agent via ``_subagent_info`` so persisted rows link
           back to the parent task.
        3. The parent task thread's cooperative ``stop_event`` is
           propagated onto each worker's ``printer._thread_local`` so
           a Stop click on the parent terminates the whole task tree
           (including nested ``_run_tasks_parallel`` calls).

        This method itself has no knowledge of backend task ids or any
        frontend concepts.  Each spawned sub-agent receives its
        ``_subagent_info`` marker here, and :meth:`ChatSorcarAgent.run`
        in the sub-agent thread reads that marker to drive any
        sub-agent-specific behaviour (e.g. broadcasting ``new_tab`` to
        a browser-based frontend so a fresh tab opens and subscribes
        to the live event stream).

        Args:
            tasks: List of self-contained task description strings.
            max_workers: Maximum concurrent threads (``None`` = auto).

        Returns:
            List of YAML result strings in the same order as *tasks*.

        Raises:
            TypeError: If *tasks* is neither a ``str`` nor a ``list[str]``.
        """
        # Coerce ``str`` → ``[str]``; otherwise ``enumerate(tasks)`` would
        # iterate a bare-string ``tasks`` character-by-character and
        # create one sub-agent per character (LLM tool-call bug).
        tasks = _coerce_tasks(tasks)
        model = getattr(self, "model_name", None)
        work_dir = getattr(self, "work_dir", None)
        chat_id = self._chat_id
        # The parent's ``task_history.id`` — set when the parent's
        # ``run()`` calls ``_add_task`` — is the only piece of
        # provenance a sub-agent row persists.  Captured into a local
        # so each worker sees the parent id at spawn time.
        parent_task_id = self._last_task_id
        printer = self.printer
        thread_local = getattr(printer, "_thread_local", None) if printer else None
        # Cooperative-stop event of the parent task thread.  Captured
        # here so each worker can copy it onto its own
        # ``printer._thread_local.stop_event`` slot before invoking
        # the sub-agent.  Without this propagation a Stop click on
        # the parent tab sets only the parent thread's event; the
        # sub-agent worker's ``printer._check_stop()`` reads a fresh
        # (empty) thread-local and becomes a silent no-op.
        parent_stop_event = (
            getattr(thread_local, "stop_event", None) if thread_local else None
        )

        # Per-sub-agent usage so the parent can aggregate cost /
        # tokens / steps back into its own running totals after the
        # pool drains.  Indexed by task position so order matches
        # *tasks*.  Each tuple is ``(budget_used, total_tokens_used,
        # total_steps)``.
        sub_usage: list[tuple[float, int, int]] = [(0.0, 0, 0)] * len(tasks)

        def _run_single(args: tuple[int, str]) -> str:
            idx, task = args
            tl = getattr(printer, "_thread_local", None) if printer else None
            if tl is not None:
                # Propagate the parent's stop_event into this worker
                # thread so ``printer._check_stop()`` and
                # ``SorcarAgent.run``'s ``self._stop_event`` snapshot
                # both honour a Stop click on the parent tab.  Nested
                # ``_run_tasks_parallel`` calls re-read this same slot,
                # so the propagation is transitive across every level
                # of nesting.
                tl.stop_event = parent_stop_event
            agent = ChatSorcarAgent(f"Parallel-{task[:40]}")
            if chat_id:
                agent.resume_chat_by_id(chat_id)
            # Persist the parent's ``task_history.id`` on the
            # sub-agent's row so the history sidebar can identify
            # the row as a sub-agent task (presence of the
            # ``subagent`` key implies the row is a sub-agent).
            agent._subagent_info = {"parent_task_id": parent_task_id}
            try:
                # ``is_parallel=True`` propagates the parallel
                # capability so sub-agents themselves get the
                # ``run_parallel`` tool and can invoke nested parallel
                # execution.  The ``new_tab`` broadcast (so the
                # frontend opens a tab and resumes into the live
                # stream) is owned by :meth:`ChatSorcarAgent.run` in
                # the sub-agent thread; it triggers automatically
                # because ``_subagent_info`` was set above.
                result: str = agent.run(
                    prompt_template=task,
                    model_name=model,
                    work_dir=work_dir,
                    printer=printer,
                    is_parallel=True,
                )
                return result
            except Exception as exc:
                error_result: str = yaml.dump(
                    {"success": False, "summary": f"Unhandled exception: {exc}"},
                    sort_keys=False,
                )
                return error_result
            finally:
                sub_usage[idx] = (
                    float(getattr(agent, "budget_used", 0.0) or 0.0),
                    int(getattr(agent, "total_tokens_used", 0) or 0),
                    int(getattr(agent, "total_steps", 0) or 0),
                )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_run_single, enumerate(tasks)))

        # Roll the sub-agents' cost / tokens / steps into this
        # (parent) agent's running totals so global accounting and UI
        # reflect the full work done.  Without this aggregation,
        # nested parallel sub-agent budgets stay invisible to the
        # parent.  ``getattr`` with float/int defaults because tests
        # that drive ``_run_tasks_parallel`` directly (without going
        # through :meth:`run`) may not have initialised the running-
        # totals attributes on the parent.
        self.budget_used = (
            float(getattr(self, "budget_used", 0.0) or 0.0)
            + sum(u[0] for u in sub_usage)
        )
        self.total_tokens_used = (
            int(getattr(self, "total_tokens_used", 0) or 0)
            + sum(u[1] for u in sub_usage)
        )
        self.total_steps = (
            int(getattr(self, "total_steps", 0) or 0)
            + sum(u[2] for u in sub_usage)
        )
        if self.printer is not None:
            try:
                self.printer.budget_offset = self.budget_used  # type: ignore[attr-defined]
                self.printer.tokens_offset = self.total_tokens_used  # type: ignore[attr-defined]
                self.printer.steps_offset = self.total_steps  # type: ignore[attr-defined]
            except Exception:
                pass
        return results

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
        # Frontend tab id that should be subscribed to this task's
        # event stream.  When provided, the printer's
        # ``subscribe_tab`` is called after ``task_id`` is allocated
        # so live events fan out to that tab.  The caller (e.g.
        # :class:`_TaskRunnerMixin`) supplies the initial tab id this
        # way because the printer no longer carries per-thread tab
        # state — tabs are pure subscribers indexed by ``task_id``.
        subscribe_tab_id = kwargs.pop("_subscribe_tab_id", "")
        # Mint a fresh chat id at run-start when one is not already
        # set (e.g. a brand-new chat tab that has never resumed a
        # history entry).  Establishing the chat id BEFORE _add_task
        # ensures every persisted event is tagged with the same
        # canonical chat id.  ``tab_id`` (frontend routing key) and
        # ``chat_id`` (persistence key) are orthogonal; the extension
        # layer maintains its own chat_id ↔ tab_id index for routing.
        if self._chat_id == "":
            self._chat_id = uuid.uuid4().hex

        # Stash the raw user prompt BEFORE any chat-context
        # augmentation so auto-commit (worktree finalize, the
        # ``update_settings(auto_commit=True)`` tool, etc.) can include
        # the user's own words — not the synthetic
        # ``## Previous tasks...`` augmented prompt — in the generated
        # commit message.
        self._last_user_prompt = prompt_template

        agent_prompt = self.build_chat_prompt(prompt_template)

        # Build an initial ``extra`` payload with values known at task
        # creation time so the history sidebar can display them
        # immediately — even while the task is still running.  Post-
        # completion values (tokens, cost) are merged by the final
        # ``_save_task_extra`` call in the ``finally`` block below (or
        # by ``task_runner.py`` for the VS Code path).
        from kiss._version import __version__

        early_extra: dict[str, object] = {
            "model": kwargs.get("model_name", "") or "",
            "work_dir": kwargs.get("work_dir", "") or "",
            "version": __version__,
            "is_parallel": bool(kwargs.get("is_parallel", False)),
            "is_worktree": (
                bool(kwargs.get("use_worktree", False))
                or type(self).__name__ == "WorktreeSorcarAgent"
            ),
        }
        if self._subagent_info is not None:
            early_extra["subagent"] = self._subagent_info

        task_id, self._chat_id = _add_task(
            prompt_template, chat_id=self._chat_id, extra=early_extra,
        )
        self._last_task_id = task_id
        # When this run is a sub-agent (its ``_subagent_info`` marker
        # was set by a parallel executor before calling ``run``),
        # broadcast a ``new_tab`` message to the frontend.  This is
        # the single hook that informs a browser-based frontend that
        # a fresh sub-agent task exists: the frontend's ``new_tab``
        # handler allocates a tab and posts ``resumeSession`` with
        # the same task id to subscribe to the live event stream.
        # Keeping this self-broadcast inside ``run`` (rather than
        # inside the parallel executor) means the executors carry no
        # task-id or frontend knowledge at all.
        if self._subagent_info is not None:
            broadcast_for_subagent = (
                kwargs.get("printer") or getattr(self, "printer", None)
            )
            if broadcast_for_subagent is not None:
                broadcast = getattr(broadcast_for_subagent, "broadcast", None)
                if broadcast is not None:
                    try:
                        broadcast({"type": "new_tab", "task_id": int(task_id)})
                    except Exception:
                        pass
        # Publish ``self`` as the agent currently driving ``task_id``.
        # The entry is added the moment the row exists in
        # ``task_history`` and removed in the matching ``finally``
        # block below so the lifetime mirrors exactly "task is in
        # flight inside this ``run()`` call".
        ChatSorcarAgent.running_agents[task_id] = self
        # Bind the printer's per-task state to ``task_id``: set the
        # agent thread's thread-local ``task_id`` so subsequent
        # ``broadcast`` calls are tagged with this id, register
        # ``self`` in ``_persist_agents[str(task_id)]`` so the
        # background DB writer can be located, subscribe the caller-
        # supplied initial tab to the stream, and start recording.
        # All four are mirror-image cleanups in the ``finally`` block.
        # ``self.printer`` is not yet assigned at this point (the
        # base ``KissAgent.run`` sets it from the ``printer`` kwarg);
        # so consult the kwarg directly with a fall-back to
        # ``self.printer`` for callers that pre-assign it.
        printer_for_mirror = kwargs.get("printer") or getattr(self, "printer", None)
        task_key = str(task_id)
        if printer_for_mirror is not None:
            tl = getattr(printer_for_mirror, "_thread_local", None)
            if tl is not None:
                tl.task_id = task_key
            persist_map = getattr(printer_for_mirror, "_persist_agents", None)
            if persist_map is not None:
                persist_map[task_key] = self
            subscribe = getattr(printer_for_mirror, "subscribe_tab", None)
            if subscribe is not None and subscribe_tab_id:
                subscribe(task_id, subscribe_tab_id)
            start_rec = getattr(printer_for_mirror, "start_recording", None)
            if start_rec is not None:
                start_rec()
        # Mirror ``task_id`` onto the :class:`_RunningAgentState` for
        # the tab the caller passed in (when one exists).  This makes
        # ``task_history_id`` available DURING the run so that
        # :meth:`VSCodeServer._reattach_running_chat` can disambiguate
        # by task id when multiple live states share the same
        # ``chat_id`` (parent + parallel sub-agents).
        mirrored_state: _RunningAgentState | None = None
        if subscribe_tab_id:
            mirrored_state = _RunningAgentState.running_agent_states.get(
                subscribe_tab_id,
            )
            if mirrored_state is not None:
                mirrored_state.task_history_id = task_id
        _record_frequent_task(prompt_template)

        result_summary = ""
        try:
            result = super().run(prompt_template=agent_prompt, **kwargs)
            try:
                result_yaml = yaml.safe_load(result)
                if isinstance(result_yaml, dict):
                    result_summary = result_yaml.get("summary", "")
            except Exception:
                result_summary = result[:500] if result else ""
            return result
        except Exception:
            result_summary = "Task failed"
            raise
        finally:
            ChatSorcarAgent.running_agents.pop(task_id, None)
            if mirrored_state is not None and mirrored_state.task_history_id == task_id:
                mirrored_state.task_history_id = None
            # Tear down recording immediately so its memory is freed
            # while the agent finishes its post-task bookkeeping.
            # ``_persist_agents`` and the thread-local ``task_id``
            # are intentionally LEFT in place so any post-task
            # broadcasts emitted by the calling task-runner (e.g.
            # ``task_done`` events, ``tasks_updated``) still get
            # tagged and fanned out under this task; the caller
            # tears those down when it finally calls
            # :meth:`BaseBrowserPrinter.cleanup_task`.
            if printer_for_mirror is not None:
                stop_rec = getattr(printer_for_mirror, "stop_recording", None)
                if stop_rec is not None:
                    try:
                        stop_rec()
                    except Exception:
                        pass
            if not skip_persistence:
                _save_task_result(task_id=task_id, result=result_summary)
                from kiss._version import __version__

                extra_payload: dict[str, object] = {
                    "model": getattr(self, "model_name", ""),
                    "work_dir": getattr(self, "work_dir", ""),
                    "version": __version__,
                    "tokens": self.total_tokens_used,
                    "cost": round(self.budget_used, 6),
                    "is_parallel": self._is_parallel,
                    "is_worktree": type(self).__name__ == "WorktreeSorcarAgent",
                }
                if self._subagent_info is not None:
                    extra_payload["subagent"] = self._subagent_info
                _save_task_extra(extra_payload, task_id=task_id)


