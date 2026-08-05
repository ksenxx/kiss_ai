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
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
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
    _task_has_events,
)
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.sorcar_agent import (
    SorcarAgent,
    _attribute_sub_usage,
    _await_subagents,
    _broadcast_subagent_done,
    _coerce_tasks,
    _collect_unfinished_usage,
    _live_agent_usage,
    _LiveUsageMonitor,
    _yaml_failure,
)
from kiss.core._version import __version__
from kiss.core.printer import parse_result_yaml

MAX_TASKS = 10


class _SubagentStopEvent(threading.Event):
    """Per-sub-agent stop event chained to the parent task's stop event.

    Each parallel sub-agent worker gets its own instance so the user
    can stop ONLY that sub-agent's task (``VSCodeServer._stop_task``
    resolves the sub-agent's ``_RunningAgentState.stop_event`` and
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


_SUMMARY_GATE_REJECTION = (
    "Error: tool call rejected — this step is a multiple of 5, so your "
    "next tool call MUST be summary(description=\"natural language "
    "summary in 5-10 sentences of what you did in the last 6 steps\"). "
    "Call summary first, then retry this tool call."
)


def summary(description: str) -> str:
    """MANDATORY every 5 steps: summarize your last 6 steps of work.

    Your tool call on every step that is a multiple of 5 (step 5, 10,
    15, ...) MUST be this tool, BEFORE any other tool call (including
    finish).  Any other tool call made on such a step is rejected
    until summary has been called.  This requirement applies to every
    task, no matter how simple, and is never overridden by the task
    prompt.

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

    running_agents: dict[str, ChatSorcarAgent] = {}
    _running_agents_lock: threading.RLock = threading.RLock()

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._chat_id: str = ""
        self._context_task_id: str = ""
        self._subagent_info: dict[str, object] | None = None
        self._last_task_id: str | None = None
        self._last_user_prompt: str = ""
        self._last_result_summary: str = ""
        self._task_id_lock: threading.RLock = threading.RLock()

    @property
    def chat_id(self) -> str:
        """Return the current chat session ID ("" means new session)."""
        return self._chat_id

    def _get_tools(self) -> list:
        """Extend the base toolset with the no-op ``summary`` tool.

        The ``summary`` tool lets the model periodically condense its
        recent activity; the chat webview reacts to the persisted
        ``tool_call`` event by nesting and collapsing the preceding
        event panels (see ``media/main.js``).  Enforcement — on steps
        divisible by 5 the ONLY accepted tool call is ``summary`` — is
        implemented by :meth:`_summary_tool_guard`, the executor-level
        :attr:`tool_call_guard` that also covers ``finish`` and any
        caller-supplied extra tools.

        Returns:
            The base tools plus :func:`summary`.
        """
        return [*super()._get_tools(), summary]

    @property
    def tool_call_guard(self) -> Any:
        """The per-tool-call guard copied onto each session executor.

        Always returns :meth:`_summary_tool_guard`, which delegates to
        any guard installed by parent classes (stored by the setter
        below) and then enforces the every-5-steps ``summary`` gate.
        A property for the same reason as :attr:`pre_step_hook`:
        ``RelentlessAgent`` reads this attribute when wiring each
        per-session executor, after ``_reset`` has assigned ``None``
        through the setter.
        """
        return self._summary_tool_guard

    @tool_call_guard.setter
    def tool_call_guard(  # pyright: ignore[reportIncompatibleVariableOverride]
        self, guard: Any
    ) -> None:
        """Store the parent-installed guard to delegate to.

        Args:
            guard: The guard installed by parent classes (or ``None``).
        """
        self._inner_tool_call_guard = guard

    def _summary_tool_guard(self, name: str, args: dict[str, Any]) -> str | None:
        """Reject every tool call except ``summary`` while one is due.

        Consulted by ``KISSAgent._execute_step`` before EVERY tool
        call — including ``finish`` (a blocked ``finish`` is not
        terminal) and caller-supplied extra tools.  While the current
        executor's ``_summary_due`` flag is set (armed by
        :meth:`_summary_reminder_hook` at the top of every step
        divisible by 5), any tool call other than ``summary`` is
        blocked with an instructive error until the model calls
        ``summary``; the executor prints blocked calls with
        ``is_error=True``.  There is deliberately no escape hatch —
        an ignored gate keeps rejecting (bounded by the task's
        ``max_steps``/budget), so the summary reliably lands on the
        boundary step.

        Args:
            name: The tool name the model is calling.
            args: The tool call arguments (forwarded to the delegated
                parent guard).

        Returns:
            ``None`` to allow the call, or the rejection message to
            block it.
        """
        inner = getattr(self, "_inner_tool_call_guard", None)
        if inner is not None:
            blocked = inner(name, args)
            if blocked is not None:
                return str(blocked)
        executor = getattr(self, "_current_executor", None)
        if executor is None or not getattr(executor, "_summary_due", False):
            return None
        if name == "summary":
            executor._summary_due = False
            return None
        return _SUMMARY_GATE_REJECTION

    @property
    def pre_step_hook(self) -> Any:
        """The per-step hook copied onto each session executor.

        Always returns :meth:`_summary_reminder_hook`, which first
        delegates to whatever hook the parent classes installed (the
        pending-user-messages drain from ``SorcarAgent.perform_task``,
        stored by the setter below) and then enforces the
        every-5-steps ``summary`` tool reminder.  Exposed as a
        property because ``SorcarAgent.perform_task`` assigns
        ``self.pre_step_hook`` immediately before
        ``RelentlessAgent.perform_task`` copies it onto the inner
        executor — wrapping at read time is the only seam that
        composes with that assignment.
        """
        return self._summary_reminder_hook

    @pre_step_hook.setter
    def pre_step_hook(self, hook: Any) -> None:
        """Store the parent-installed hook to delegate to.

        Args:
            hook: The hook installed by parent classes (or ``None``).
        """
        self._inner_pre_step_hook = hook

    def _summary_reminder_hook(self, model: Any) -> None:
        """Arm the summary gate on every step divisible by 5.

        Runs at the top of every executor step.  Delegates to the
        parent-installed hook first, then — whenever the step ABOUT to
        run is a multiple of 5 — sets the executor's ``_summary_due``
        flag (armed exactly once per boundary) and appends a user
        message instructing the model to call
        ``summary(description=...)`` recapping its last 6 steps.  The
        SYSTEM.md instruction and reminder messages alone are not
        reliably followed by every model (verified live: summaries
        landed on steps 6/11/16 or were skipped entirely), so the
        armed flag makes :meth:`_summary_tool_guard` reject every
        other tool call (including ``finish``) until ``summary`` runs
        — the summary tool call lands exactly on step 5, 10, 15, ....
        An armed flag survives into later steps until the model
        complies, so an ignored reminder can no longer skip a whole
        5-step window.  The step number is GLOBAL (prior sub-sessions'
        steps included), matching the step counter the UI displays.

        Args:
            model: The live model whose conversation receives the
                reminder message.
        """
        inner = getattr(self, "_inner_pre_step_hook", None)
        if inner is not None:
            inner(model)
        executor = getattr(self, "_current_executor", None)
        if executor is None:
            return
        step = int(getattr(executor, "step_count", 0) or 0) + int(
            getattr(self, "total_steps", 0) or 0
        )
        if step < 5 or step % 5:
            return
        if getattr(executor, "_summary_reminder_step", 0) == step:
            return
        executor._summary_reminder_step = step
        executor._summary_due = True
        model.add_message_to_conversation(
            "user",
            f"You are now on step {step}, a multiple of 5. Call the "
            'summary tool NOW — summary(description="natural language '
            'summary in 5-10 sentences of what you did in the last 6 '
            'steps") — before any other tool call. Every other tool '
            "call (including finish) will be rejected until you do. "
            "After the summary, continue the task.",
        )

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

    def _register_running_state(self) -> bool:
        """Publish ``self`` in :attr:`_RunningAgentState.running_agent_states` for this chat.

        Maintains the *registered-with-the-server* invariant: every
        live :class:`ChatSorcarAgent` instance must be discoverable
        through some entry of
        :attr:`_RunningAgentState.running_agent_states` whose
        ``state.agent is self``.  Consumers that rely on this
        invariant include
        :meth:`VSCodeServer._reattach_running_chat`,
        :meth:`VSCodeServer._get_running_task_ids` (the History-
        sidebar running indicator), and the parent-tab-id scan inside
        :meth:`ChatSorcarAgent._run_tasks_parallel`.

        Skips registration when an entry whose ``chat_id`` matches
        ``self._chat_id`` is already present: the VS Code server
        pre-populates a ``_RunningAgentState`` keyed by the frontend
        tab id ahead of run-start (with ``chat_id`` set on the
        state); :class:`WorktreeSorcarAgent.run` registers its own
        entry before delegating to :meth:`ChatSorcarAgent.run`; and
        :meth:`ChatSorcarAgent._run_tasks_parallel` registers each
        sub-agent's per-thread state before invoking its ``run()``.
        Re-registering on top of any of those would either clobber
        lifecycle flags (server flow) or shadow the per-tab routing
        key (worktree / sub-agent flow).  In CLI / third-party
        invocations of plain :class:`ChatSorcarAgent` (no
        pre-population), this method adds the missing entry keyed by
        ``self._chat_id``.

        Returns:
            ``True`` when a fresh entry was added (and the caller
            must remove it in its own ``finally``); ``False`` when an
            entry was already present (the existing owner is
            responsible for cleanup).
        """
        with _RunningAgentState._registry_lock:
            for state in _RunningAgentState.running_agent_states.values():
                if state.chat_id == self._chat_id and (
                    state.agent is None or state.agent is self
                ):
                    return False
            state = _RunningAgentState(
                self._chat_id,
                getattr(self, "model_name", "") or "",
                agent=self,  # type: ignore[arg-type]
            )
            state.chat_id = self._chat_id
            state.is_task_active = True
            _RunningAgentState.register(self._chat_id, state)
            return True

    def _unregister_running_state(self) -> None:
        """Remove ``self``'s entry from :attr:`_RunningAgentState.running_agent_states`.

        Only removes the entry we ourselves added (matched by both
        ``state.agent is self`` and ``state.chat_id == self._chat_id``).
        A different code path (e.g. the VS Code server) may have
        replaced it mid-run; in that case the new owner is
        responsible for its own cleanup.
        """
        with _RunningAgentState._registry_lock:
            target_key: str | None = None
            for key, state in _RunningAgentState.running_agent_states.items():
                if state.agent is self and state.chat_id == self._chat_id:
                    target_key = key
                    break
            if target_key is not None:
                current = _RunningAgentState.running_agent_states[target_key]
                current.is_task_active = False
                _RunningAgentState.running_agent_states.pop(target_key, None)

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
        if self._subagent_info is not None:
            payload["subagent"] = self._subagent_info
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
        Runs that happen OUTSIDE a chat webview — the CLI, the
        third-party channel agents, or a remote webapp invocation with a
        non-recording printer — leave the ``events`` table empty, so the
        chat webview would load a blank session even though the task and
        its result are in ``task_history``.

        This synthesizes the two events the webview needs to render the
        exchange — a ``prompt`` event (the user's task) and a ``result``
        event (the agent's summary / success / cost) — but only when the
        task has no events yet, so a recording printer's full event
        stream is never duplicated.

        Args:
            task_id: Stable ``task_history`` row id for this run.
            prompt: The prompt the agent actually ran with (chat-context
                augmented), mirroring the ``prompt`` event a recording
                printer would have persisted.
            result_raw: The raw YAML result string returned by the run
                (used to recover ``success`` / ``is_continue``).
            result_summary: The extracted human-readable summary text.
        """
        if _task_has_events(task_id):
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

    def _run_tasks_parallel(
        self,
        tasks: list[str],
        max_workers: int | None = None,
    ) -> list[str]:
        """Execute parallel tasks using ChatSorcarAgent sub-agents.

        """
        tasks = _coerce_tasks(tasks)
        model = self.model_name
        work_dir = self.work_dir
        chat_id = self._chat_id
        budget_share = self._subagent_budget_share(len(tasks))
        model_config = getattr(self, "model_config", None)
        persisted_parent_task_id = self._last_task_id
        if (
            not isinstance(persisted_parent_task_id, str)
            or not persisted_parent_task_id
        ):
            persisted_parent_task_id = ""
        if persisted_parent_task_id:
            routing_parent_key = persisted_parent_task_id
        else:
            routing_parent_key = uuid.uuid4().hex
        parent_task_id = routing_parent_key
        parent_tab_id = ""
        with _RunningAgentState._registry_lock:
            for tid, state in _RunningAgentState.running_agent_states.items():
                if state.agent is self:
                    parent_tab_id = tid
                    break
        printer = self.printer
        if self._subagent_info is not None and printer is not None:
            fanout = getattr(printer, "_fanout_targets", None)
            own_task_id = self._last_task_id
            if fanout is not None and own_task_id:
                viewer_ids = fanout(own_task_id)
                if viewer_ids:
                    parent_tab_id = sorted(viewer_ids)[0]
        thread_local = getattr(printer, "_thread_local", None) if printer else None
        parent_stop_event = (
            getattr(thread_local, "stop_event", None) if thread_local else None
        )

        sub_usage: list[tuple[float, int, int]] = [(0.0, 0, 0)] * len(tasks)
        # Each child's agent, published as soon as it exists so the
        # parent can still read the spend of a child it had to abandon.
        sub_agents: list[Any] = [None] * len(tasks)
        usage_monitor = _LiveUsageMonitor(self, printer)

        def _run_single(args: tuple[int, str]) -> str:
            idx, task = args
            sub_stop_event = _SubagentStopEvent(parent_stop_event)
            tl = getattr(printer, "_thread_local", None) if printer else None
            if tl is not None:
                tl.stop_event = sub_stop_event
            agent = ChatSorcarAgent(f"Parallel-{task[:40]}")
            sub_agents[idx] = agent
            usage_monitor.track(agent)
            if chat_id:
                agent.resume_chat_by_id(chat_id)
            sub_tab_id = f"task-{parent_task_id}__sub_{idx}"
            agent._tab_id = sub_tab_id  # type: ignore[attr-defined]
            sub_persisted_parent = self._last_task_id
            if (
                not isinstance(sub_persisted_parent, str)
                or not sub_persisted_parent
            ):
                sub_persisted_parent = persisted_parent_task_id
            agent._subagent_info = {
                "parent_task_id": sub_persisted_parent,
                "parent_tab_id": parent_tab_id,
            }
            sub_state = _RunningAgentState(
                sub_tab_id,
                model or "",
                agent=agent,  # type: ignore[arg-type]
                chat_id=chat_id,
                is_subagent=True,
                parent_task_id=sub_persisted_parent,
                is_task_active=True,
                stop_event=sub_stop_event,
            )
            sub_state.task_thread = threading.current_thread()
            _RunningAgentState.register(sub_tab_id, sub_state)
            try:
                result: str = agent.run(
                    prompt_template=task,
                    model_name=model,
                    work_dir=work_dir,
                    printer=printer,
                    is_parallel=True,
                    max_budget=budget_share,
                    model_config=model_config,
                )
                return result
            except KeyboardInterrupt:
                if parent_stop_event is not None and parent_stop_event.is_set():
                    raise
                stopped: str = yaml.dump(
                    {
                        "success": False,
                        "summary": "Sub-agent task stopped by user.",
                    },
                    sort_keys=False,
                )
                return stopped
            except Exception as exc:
                return _yaml_failure(exc)
            finally:
                with _RunningAgentState._registry_lock:
                    sub_state.task_thread = None
                # _live_agent_usage (not _agent_usage): an interrupted
                # child never folds its in-flight executor session's
                # spend into its totals, so the folded-only read would
                # undercount that child.
                sub_usage[idx] = _live_agent_usage(agent)
                if printer is not None:
                    try:
                        sub_task_id = getattr(agent, "_last_task_id", None)
                        fanout = getattr(printer, "_fanout_targets", None)
                        viewer_ids: list[str] = []
                        if fanout and sub_task_id is not None:
                            viewer_ids = fanout(sub_task_id)
                        if sub_tab_id not in viewer_ids:
                            viewer_ids.append(sub_tab_id)
                        _broadcast_subagent_done(
                            printer, viewer_ids, model or "",
                        )
                    except Exception:
                        pass
                _RunningAgentState.unregister(sub_tab_id, sub_state)
                # Pool workers are reused across fan-outs, and the
                # binding is per THREAD (it is what lets a model stream
                # see a stop), so leaving it behind would let an
                # unrelated sibling inherit a stop meant for this task.
                if tl is not None:
                    tl.stop_event = None

        usage_monitor.start()
        pool: ThreadPoolExecutor | None = None
        futures: list[Future[str]] = []
        abandoned = False
        try:
            pool = ThreadPoolExecutor(max_workers=max_workers)
            futures = [
                pool.submit(_run_single, item) for item in enumerate(tasks)
            ]
            try:
                results = _await_subagents(futures, parent_stop_event)
            except BaseException:
                # Includes the KeyboardInterrupt that _stop_task injects
                # into this thread, which lands as soon as a wait slice
                # ends — i.e. well before the grace period above.
                abandoned = any(not f.done() for f in futures)
                raise
        finally:
            # Only a deliberately abandoned child skips the join: it is
            # ignoring its stop event, and waiting for it would put the
            # parent straight back into the uninterruptible wait this
            # fix removes.  Every other path joins exactly as the old
            # `with ThreadPoolExecutor(...)` block did, which also
            # RECLAIMS each level's worker thread — nested fan-outs rely
            # on that to bound how many threads exist at once.
            if pool is not None:
                pool.shutdown(wait=not abandoned, cancel_futures=abandoned)
            # stop() joins the monitor BEFORE the offsets bump so a late
            # emission can never double-count.  The attribution runs in
            # this finally so a parent stop that unwinds the fan-out
            # cannot make completed siblings' (and interrupted children's
            # live) spend disappear from the parent task's totals.
            usage_monitor.stop()
            _collect_unfinished_usage(futures, sub_agents, sub_usage)
            _attribute_sub_usage(
                self,
                sum(u[0] for u in sub_usage),
                sum(u[1] for u in sub_usage),
                sum(u[2] for u in sub_usage),
            )
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
        subscribe_tab_id = kwargs.pop("_subscribe_tab_id", "")
        on_task_id_allocated = kwargs.pop("_on_task_id_allocated", None)
        if self._chat_id == "":
            self._chat_id = _allocate_chat_id()
        with self._task_id_lock:
            self._last_task_id = None
            registered_here = self._register_running_state()

        try:
            self._last_user_prompt = prompt_template
            self._last_result_summary = ""

            agent_prompt = self.build_chat_prompt(prompt_template)

            explicit_worktree = kwargs.pop("use_worktree", None)
            if explicit_worktree is not None:
                is_worktree = bool(explicit_worktree)
            else:
                is_worktree = self.uses_worktree and _dir_inside_worktree(
                    kwargs.get("work_dir", "") or "",
                    getattr(self, "_wt_dir", None),
                )

            early_extra = self._build_extra_payload(
                model=kwargs.get("model_name", "") or "",
                work_dir=kwargs.get("work_dir", "") or "",
                is_parallel=bool(kwargs.get("is_parallel", False)),
                is_worktree=is_worktree,
            )

            task_id, self._chat_id = _add_task(
                prompt_template, chat_id=self._chat_id, extra=early_extra,
            )
        except BaseException:
            if registered_here:
                self._unregister_running_state()
            raise
        with self._task_id_lock:
            self._last_task_id = task_id
        printer = kwargs.get("printer") or getattr(self, "printer", None)
        task_key = str(task_id)
        result_summary = ""
        result_raw = ""
        run_started = False
        with ChatSorcarAgent._running_agents_lock:
            ChatSorcarAgent.running_agents[task_id] = self
        # From this point on, BOTH registries hold entries for this run,
        # so every remaining setup step (printer wiring, subscription,
        # frequent-task recording, ...) must run inside the try below:
        # an exception in any of them would otherwise bypass the cleanup
        # and leave a permanently "running" task behind (F-14).
        try:
            if self._subagent_info is not None:
                with _RunningAgentState._registry_lock:
                    for state in (
                        _RunningAgentState.running_agent_states.values()
                    ):
                        if state.agent is self:
                            state.task_history_id = task_id
                            break
            if printer is not None:
                tl = getattr(printer, "_thread_local", None)
                if tl is not None:
                    tl.task_id = task_key
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
                persist_map = getattr(printer, "_persist_agents", None)
                if persist_map is not None:
                    printer_lock = getattr(printer, "_lock", None)
                    if printer_lock is not None:
                        with printer_lock:
                            persist_map[task_key] = self
                    else:
                        persist_map[task_key] = self
                subscribe = getattr(printer, "subscribe_tab", None)
                if subscribe is not None and subscribe_tab_id:
                    subscribe(task_id, subscribe_tab_id)
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
            if self._subagent_info is None:
                _record_frequent_task(prompt_template)

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
            with ChatSorcarAgent._running_agents_lock:
                ChatSorcarAgent.running_agents.pop(task_id, None)
            if registered_here:
                self._unregister_running_state()
            if printer is not None:
                if not run_started:
                    # Setup failed before the run started: nothing else
                    # will ever remove the persist-agent registration we
                    # installed above, so drop it here (identity-checked)
                    # to avoid a stale strong reference in the printer.
                    persist_map = getattr(printer, "_persist_agents", None)
                    if (
                        persist_map is not None
                        and persist_map.get(task_key) is self
                    ):
                        printer_lock = getattr(printer, "_lock", None)
                        if printer_lock is not None:
                            with printer_lock:
                                if persist_map.get(task_key) is self:
                                    persist_map.pop(task_key, None)
                        else:
                            persist_map.pop(task_key, None)
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
                    ),
                    work_dir=self.work_dir,
                    is_parallel=self._is_parallel,
                    is_worktree=is_worktree,
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
