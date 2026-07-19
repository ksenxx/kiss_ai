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
from concurrent.futures import ThreadPoolExecutor
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
    _agent_usage,
    _attribute_sub_usage,
    _broadcast_subagent_done,
    _coerce_tasks,
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
    description visible as a running digest for the user.

    Args:
        description: Natural language summary in 5-10 sentences of
            what the agent did in the last 6 steps.

    Returns:
        A short confirmation string.
    """
    del description  # Displayed by the chat UI; nothing to do here.
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
    # Guards every mutation of :attr:`running_agents` so concurrent
    # ``run()`` invocations across threads (CLI multi-task, sub-agent
    # parallel dispatch, VS Code task-runner worker pool) cannot corrupt
    # the dict's internal hash table.
    _running_agents_lock: threading.RLock = threading.RLock()

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._chat_id: str = ""
        # When non-empty, the NEXT ``build_chat_prompt`` call builds
        # its "previous tasks" context by traversing this task's
        # ``parent_task_id`` chain instead of loading the whole chat
        # (see :meth:`resume_from_task_id`).  One-shot: consumed (and
        # cleared) by the first ``build_chat_prompt`` after it is set.
        self._context_task_id: str = ""
        self._subagent_info: dict[str, object] | None = None
        self._last_task_id: str | None = None
        self._last_user_prompt: str = ""
        # Result summary of the most recently completed run(); appended
        # to auto-commit messages under a "Result:" heading so the
        # commit records both the task description and its outcome.
        self._last_result_summary: str = ""
        # r4-sorcar-H3 — guards the paired ``self._last_task_id = ...``
        # assignment and ``_register_running_state()`` /
        # ``_unregister_running_state()`` calls so a concurrent reader
        # cannot observe a half-applied clear+register pair.
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
    def tool_call_guard(self, guard: Any) -> None:
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
            # The summary call satisfies the gate; clear it so the
            # remaining calls in this response (and later steps) run.
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
        # ``step_count`` was already incremented for the step ABOUT to
        # run.  Add the steps accumulated by prior sub-sessions
        # (``RelentlessAgent.total_steps`` — the same offset the UI
        # adds when displaying "Steps: N") so the cadence tracks the
        # GLOBAL step number: a summary is due when THIS step's global
        # number is a multiple of 5, landing exactly on step 5, 10, ....
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
        # Acquire the shared ``_registry_lock`` for the whole
        # scan-then-modify so a concurrent sub-agent thread cannot
        # resize ``running_agent_states`` while we iterate, and so
        # the insertion is atomic w.r.t. the VS Code server's
        # iteration loops (which hold the very same lock under the
        # ``_state_lock`` alias).
        with _RunningAgentState._registry_lock:
            for state in _RunningAgentState.running_agent_states.values():
                # r4-sorcar-H1: do NOT treat an existing entry for the
                # same ``chat_id`` as the existing owner unless its
                # ``agent`` is either ``None`` (a server-side
                # pre-allocated entry waiting for a real agent) or
                # ``self`` (idempotent re-register).  Two distinct
                # agents sharing a ``chat_id`` (e.g. CLI + remote
                # webapp picked the same chat) must each be
                # discoverable through their own entry.
                # r5-sorcar-H3 REJECTED: the round-5 review proposed
                # binding ``state.agent = self`` here when
                # ``state.agent is None``.  ``test_running_agent_state_on_run::
                # test_run_does_not_clobber_preexisting_state``
                # asserts the OPPOSITE contract: a pre-allocated
                # entry with ``agent=None`` belongs to another owner
                # (the server frame or parent worktree agent) and
                # must NOT be hijacked by a standalone child agent
                # that happens to share the chat_id.  Keep the
                # original "skip without binding" semantics.
                if state.chat_id == self._chat_id and (
                    state.agent is None or state.agent is self
                ):
                    return False
            state = _RunningAgentState(
                self._chat_id,
                getattr(self, "model_name", "") or "",
                agent=self,  # type: ignore[arg-type]
            )
            # Tag the state with the canonical chat id so subsequent
            # lookups (e.g. multi-viewer subscribe,
            # ``_unregister_running_state``) can route by chat id
            # without depending on the dict key.
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
        # Scan-then-pop must be atomic w.r.t. concurrent producers
        # (parallel sub-agents in :meth:`_run_tasks_parallel`, the
        # VS Code server's tab lifecycle handlers) so the dict is
        # never resized between the lookup and the pop.
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
            # Tab opened by a task id and no task run since: build the
            # context from the opened task's parent chain (oldest
            # ancestor first).  One-shot — clear the seed so follow-up
            # tasks in the same chat use the normal chat context.
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
        # Persist the user-visible workspace folder, not the ephemeral
        # ``<repo>/.kiss-worktrees/kiss_wt-<slug>`` directory that gets
        # removed when the worktree is merged or discarded.
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
            "total_tokens": int(self.total_tokens_used),
            "cost": f"${self.budget_used:.4f}",
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
        # Cap every sub-agent to a fair share of THIS task's remaining
        # budget (see :meth:`SorcarAgent._subagent_budget_share`) —
        # without it each sub-agent would default to the full configured
        # budget and a single sub-agent could spend the entire budget of
        # the main task.  The share calculation also reserves one equal
        # share for the parent to process results and finish.  Also forward
        # the parent's ``model_config`` so
        # sub-agents talk to the same provider endpoint (custom
        # ``base_url``/``api_key`` routing).
        budget_share = self._subagent_budget_share(len(tasks))
        model_config = getattr(self, "model_config", None)
        # IMPORTANT: keep two ids strictly separate.
        #
        # ``persisted_parent_task_id`` is the REAL ``task_history.id``
        # of the parent task.  It (and only it) is allowed to land
        # in the sub-agent's ``task_history.parent_task_id`` column.
        # If we don't yet have one (e.g. CLI flow that called
        # ``_run_tasks_parallel`` before its own ``_add_task``
        # returned), pass empty so the sub-agent row appears as a
        # normal top-level row rather than as an orphan pointing at
        # a synthetic UUID that no real row carries.
        #
        # ``routing_parent_key`` is a process-local identifier used
        # ONLY for building the in-memory tab key
        # (``f"task-{routing_parent_key}__sub_<N>"``) and matching
        # against ``_RunningAgentState.running_agent_states``.
        # When no real parent id exists we still need a unique
        # routing key so the bogus literal ``"task-None__sub_*"``
        # cannot leak into the registry or the global ``new_tab``
        # broadcast.
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
        # Resolve the parent's frontend tab id from the running-agent
        # registry so we can thread it through ``_subagent_info`` to
        # each sub-agent.  The sub-agent's ``new_tab`` broadcast (in
        # :meth:`run` below) stamps the payload with
        # ``parent_tab_id``; the frontend's ``case 'new_tab':`` /
        # ``case 'openSubagentTab':`` handlers then ignore the event
        # when no local tab has that id — which is the case for
        # webviews bound to a different chat.  Without this routing
        # hint, the global ``new_tab`` broadcast (``taskId=""``)
        # reaches every connected WS / UDS client and every webview
        # materialises phantom sub-agent tabs.
        parent_tab_id = ""
        with _RunningAgentState._registry_lock:
            for tid, state in _RunningAgentState.running_agent_states.items():
                if state.agent is self:
                    parent_tab_id = tid
                    break
        printer = self.printer
        # NESTED run_parallel: when this agent is itself a sub-agent
        # (``_subagent_info`` set), the registry key found above is the
        # BACKEND synthetic id ``task-{grandparent_task_id}__sub_{idx}``
        # — but the frontend tab that displays this sub-agent was
        # created by ``createBackgroundSubagentTab`` with a RANDOM
        # frontend id.  Stamping the children's ``new_tab`` broadcasts
        # with the synthetic key makes every webview's
        # ``case 'new_tab':`` guard (``tabs.find(t => t.id ===
        # ev.parent_tab_id)``) drop them, so nested sub-agents never
        # open any tabs.  Resolve the real frontend viewer tab id from
        # the printer's subscriber map instead: the frontend posted
        # ``resumeSession`` (→ ``subscribe_tab``) for this sub-agent's
        # own task id when it materialised our tab.  Fall back to the
        # registry key for the ``_open_persisted_subagent_tabs`` path,
        # where the frontend tab id IS the deterministic ``sub_tab_id``
        # and no subscriber may be recorded yet.
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
        # Created HERE, in the parent task's thread, so the monitor
        # captures the parent's thread-local ``task_id`` and its
        # emissions land on the parent's event stream / usage offsets.
        usage_monitor = _LiveUsageMonitor(self, printer)

        def _run_single(args: tuple[int, str]) -> str:
            idx, task = args
            # Give THIS sub-agent its own stop event, chained to the
            # parent's: the frontend's Stop button on the sub-agent tab
            # resolves to this event (via the ``_RunningAgentState``
            # registered below) and stops ONLY this sub-agent, while a
            # parent-task stop still propagates to every worker through
            # the chained ``is_set()``.  Installed on the worker
            # thread-local so ``JsonPrinter._check_stop`` (raises
            # ``KeyboardInterrupt`` on the next print) and
            # ``SorcarAgent.run`` (snapshots it into ``self._stop_event``
            # for bash process-group kills) both observe it.
            sub_stop_event = _SubagentStopEvent(parent_stop_event)
            tl = getattr(printer, "_thread_local", None) if printer else None
            if tl is not None:
                tl.stop_event = sub_stop_event
            agent = ChatSorcarAgent(f"Parallel-{task[:40]}")
            usage_monitor.track(agent)
            if chat_id:
                agent.resume_chat_by_id(chat_id)
            sub_tab_id = f"task-{parent_task_id}__sub_{idx}"
            # Route mid-task prompt injection (``appendUserMessage``)
            # to THIS sub-agent: ``_drain_pending_user_messages`` (the
            # pre-step hook installed by ``SorcarAgent.perform_task``)
            # drains ``pending_user_messages`` from the registry entry
            # keyed by ``self._tab_id`` — without this the hook is
            # never even installed and prompts injected on a running
            # sub-agent tab would silently vanish.
            agent._tab_id = sub_tab_id  # type: ignore[attr-defined]
            # Only persist the REAL parent_task_id; the synthetic
            # routing key (used for tab routing only) must NEVER
            # land in the task_history.parent_task_id column, where
            # it would orphan the sub-agent row.
            #
            # Re-snapshot the parent's persisted ``task_history.id`` at
            # the moment this worker starts.  Defeats the TOCTOU window
            # where the outer ``_run_tasks_parallel`` captured
            # ``self._last_task_id`` BEFORE the parent's ``_add_task``
            # had assigned a row (e.g. when a concurrent
            # ``_run_tasks_parallel`` batch on the same parent agent
            # had not yet returned).  When the outer snapshot was
            # already real, this re-read is a no-op.
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
            # Populate all sub-agent state fields via the constructor
            # so peer threads holding :attr:`_registry_lock` never
            # observe a half-built state object.  The post-construct
            # attribute writes that previously sat between the
            # constructor and ``register`` could be re-ordered relative
            # to ``register``'s lock acquisition, which made the
            # documented "never observe the dict mid-resize" invariant
            # underdocument the equally important "never observe a
            # half-built state" invariant.
            sub_state = _RunningAgentState(
                sub_tab_id,
                model or "",
                agent=agent,  # type: ignore[arg-type]
                chat_id=chat_id,
                is_subagent=True,
                # r4-sorcar-H2: pass ``sub_persisted_parent`` directly
                # (always a ``str``, possibly ``""``) so the sentinel
                # matches ``_subagent_info["parent_task_id"]`` which is
                # also ``""`` when no persisted parent exists.  Mapping
                # ``""`` to ``None`` here produced split-brain sentinels
                # (``parent_task_id is None`` vs ``parent_task_id == ""``).
                parent_task_id=sub_persisted_parent,
                is_task_active=True,
                # Publish the per-sub-agent stop event so
                # ``VSCodeServer._stop_task`` (directly by
                # ``sub_tab_id``, or via the viewer-tab fallback
                # ``_find_source_tab_for_viewer``) can stop ONLY this
                # sub-agent's task.
                stop_event=sub_stop_event,
            )
            # Publish the pool worker thread so ``_stop_task``'s
            # force-stop watchdog can inject a ``KeyboardInterrupt``
            # into a sub-agent wedged in an uninterruptible LLM/API
            # call (the cooperative event only fires on the next
            # printer poll).  The watchdog's ownership guard (see
            # ``_force_stop_thread``) re-checks — under
            # ``_registry_lock`` — that this state still maps this
            # thread before injecting, so ``ThreadPoolExecutor``
            # thread reuse can never route the interrupt into a
            # SIBLING task that later runs on the same worker thread.
            sub_state.task_thread = threading.current_thread()
            # Route the insert through the locked helper so peer
            # parallel sub-agents and VS Code server iteration loops
            # never observe the dict mid-resize and never raise
            # ``RuntimeError: dictionary changed size during iteration``.
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
                # A cooperative stop reached this worker (raised by
                # ``JsonPrinter._check_stop`` on the sub-agent's next
                # print).  When the PARENT task is being stopped the
                # interrupt must keep propagating so
                # ``ThreadPoolExecutor.map`` re-raises it in the parent
                # task thread and ``_TaskRunnerMixin._run_task`` can
                # surface ``task_stopped`` — the pre-existing whole-tree
                # stop path.  When only THIS sub-agent's own event was
                # set (Stop clicked on the sub-agent's tab) the parent
                # and the sibling sub-agents must keep running, so the
                # interrupt is absorbed here and reported as this one
                # task's failure result.
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
                # Disown the pool worker thread FIRST (under the
                # registry lock, which the force-stop watchdog's
                # ownership guard also holds across its check+inject):
                # once cleared, a pending ``_stop_task`` watchdog can
                # no longer inject a ``KeyboardInterrupt`` into this
                # thread — which is about to return to the pool and
                # may pick up a SIBLING task.
                with _RunningAgentState._registry_lock:
                    sub_state.task_thread = None
                sub_usage[idx] = _agent_usage(agent)
                # Broadcast ``subagentDone`` so the frontend can stop
                # the running indicator on the sub-agent tab.
                #
                # The frontend tab that displays this sub-agent was
                # created by ``new_tab`` → ``createNewTab()`` with a
                # randomly-generated frontend tab id, NOT the
                # backend's ``sub_tab_id``.  The printer's subscriber
                # map (``_subscribers[task_id]``) records that
                # frontend tab id when ``_reattach_running_chat``
                # subscribes it to this sub-agent's event stream.
                # Resolve the actual viewer tab ids from the
                # subscriber map so ``subagentDone`` reaches the
                # correct frontend tab; fall back to ``sub_tab_id``
                # for the ``_open_persisted_subagent_tabs`` path
                # where the tab id is deterministic.
                if printer is not None:
                    try:
                        sub_task_id = getattr(agent, "_last_task_id", None)
                        fanout = getattr(printer, "_fanout_targets", None)
                        viewer_ids: list[str] = []
                        if fanout and sub_task_id is not None:
                            viewer_ids = fanout(sub_task_id)
                        if sub_tab_id not in viewer_ids:
                            viewer_ids.append(sub_tab_id)
                        _broadcast_subagent_done(printer, viewer_ids)
                    except Exception:
                        pass
                _RunningAgentState.unregister(sub_tab_id)

        # Live-stream the parent task's cumulative usage (parent session
        # + all sub-agents) while the fan-out runs, so the cost/tokens
        # header stays accurate at every turn instead of freezing until
        # every sub-agent completes.  Stopped (joined) BEFORE
        # ``_attribute_sub_usage`` bumps the printer offsets, so a late
        # emission can never double-count the sub-agents' spend.
        usage_monitor.start()
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                results = list(pool.map(_run_single, enumerate(tasks)))
        finally:
            usage_monitor.stop()

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
        # Mint a fresh chat id only if no caller (or prior ``run()``)
        # already established one.  Resetting unconditionally here would
        # discard the ``chat_id`` that ``_run_tasks_parallel`` propagates
        # from the parent via ``resume_chat_by_id``.  Mint through the
        # SAME :func:`_allocate_chat_id` helper that
        # :meth:`WorktreeSorcarAgent.run` uses so the two paths can
        # never drift (e.g. if the helper ever gains reservation /
        # collision-check side effects).
        if self._chat_id == "":
            self._chat_id = _allocate_chat_id()
        # Self-register in the per-tab state registry so the
        # *registered-with-the-server* invariant holds for CLI /
        # third-party / remote-webapp invocations that never go through
        # :meth:`VSCodeServer._TaskRunnerMixin._run_task_inner`.  UI
        # launches, sub-agent runs, and
        # :class:`WorktreeSorcarAgent.run` already pre-populate an
        # entry for ``self._chat_id`` (or an equivalent tab id with
        # ``state.agent is self``); ``_register_running_state``
        # detects the existing entry and returns ``False`` so we do
        # not double-register and the existing owner remains
        # responsible for cleanup.
        # Clear ``_last_task_id`` BEFORE registering so a concurrent
        # consumer that wakes up between ``_register_running_state``
        # and the upcoming ``_add_task`` cannot read a stale
        # task_history_id from a previous run of this same agent
        # instance (chat-resume / multi-task CLI use case).
        # r4-sorcar-H3 — paired clear+register guarded by the
        # per-instance lock so a concurrent reader cannot observe
        # ``_last_task_id`` cleared *before* ``running_agent_states``
        # has been re-published.
        with self._task_id_lock:
            self._last_task_id = None
            registered_here = self._register_running_state()

        # r3-sorcar-H1 — if anything between
        # ``_register_running_state`` and the successful return of
        # ``_add_task`` raises, the agent would otherwise be wedged in
        # ``running_agent_states`` forever (the surrounding ``finally``
        # only runs after the row exists).  Wrap the early section in
        # a defensive try/except that unregisters on failure.
        try:
            self._last_user_prompt = prompt_template
            # Reset the previous run's result so a failure before
            # this run's summary is computed can never leak a stale
            # result into this run's auto-commit message.
            self._last_result_summary = ""

            agent_prompt = self.build_chat_prompt(prompt_template)

            # Resolve whether THIS run actually executes inside a git
            # worktree — used by BOTH the early extra save below and
            # the final extra save in the ``finally`` block, so the
            # two can never disagree.
            #
            # ``pop`` (not ``get``): ``SorcarAgent.run()`` has no
            # ``use_worktree`` parameter, so forwarding it via
            # ``**kwargs`` would raise ``TypeError``.  An explicit
            # kwarg wins.  When absent (``WorktreeSorcarAgent.run``
            # pops it before delegating here), the class-level
            # ``uses_worktree`` flag alone is WRONG — a
            # ``WorktreeSorcarAgent`` invoked with
            # ``use_worktree=False``, or falling back to direct
            # execution (work_dir not a git repo, detached HEAD,
            # setup failure), runs on the main working tree.  Probe
            # whether the effective ``work_dir`` was actually
            # redirected into this agent's own worktree directory.
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
        with ChatSorcarAgent._running_agents_lock:
            ChatSorcarAgent.running_agents[task_id] = self
        # Mirror this run's task_history_id onto the per-thread
        # sub-agent state so ``VSCodeServer._reattach_running_chat``
        # can disambiguate the sub-agent from its parent by task id.
        if self._subagent_info is not None:
            # Hold the shared registry lock while scanning so a peer
            # sub-agent registering / unregistering in
            # :meth:`_run_tasks_parallel` cannot resize the dict
            # underneath us.
            with _RunningAgentState._registry_lock:
                for state in _RunningAgentState.running_agent_states.values():
                    if state.agent is self:
                        state.task_history_id = task_id
                        break
        printer = kwargs.get("printer") or getattr(self, "printer", None)
        task_key = str(task_id)
        if printer is not None:
            # IMPORTANT: set the thread-local ``task_id`` BEFORE
            # emitting the ``new_tab`` broadcast.  ``_run_tasks_parallel``
            # reuses ``ThreadPoolExecutor`` worker threads across
            # multiple sub-agents (e.g. when ``max_workers`` is less
            # than the number of tasks); without setting this first,
            # the ``new_tab`` event — and any other early broadcast —
            # would be ``_inject_task_id``-stamped with the PREVIOUS
            # sub-agent's task id that the worker thread still
            # carries, mis-routing it through the wrong tab's stream.
            tl = getattr(printer, "_thread_local", None)
            if tl is not None:
                tl.task_id = task_key
            if self._subagent_info is not None:
                broadcast = getattr(printer, "broadcast", None)
                if broadcast is not None:
                    try:
                        # ``taskId=""`` keeps this a global system
                        # event so it reaches every connected client —
                        # the frontend needs the broadcast to allocate
                        # the new tab; only after allocation does it
                        # subscribe to ``task_id``'s stream.  Without
                        # the explicit empty ``taskId``,
                        # ``_inject_task_id`` would stamp the event
                        # with the just-set ``task_key`` and
                        # ``WebPrinter.broadcast`` would fan it out
                        # only to subscribers of that task — of which
                        # there are none until the frontend has
                        # received the new_tab and subscribed.
                        # Include ``parent_tab_id`` so the frontend
                        # can drop the event in webviews that don't
                        # own the parent tab (e.g. webviews bound to
                        # a different chat).  See the symmetric
                        # ``openSubagentTab`` guard in main.js.
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
                # Mutate ``_persist_agents`` under the printer's
                # ``_lock`` so the registration is serialised against
                # ``cleanup_task`` (pop under ``_lock``) and the
                # ``_persist_event`` lookup (get under ``_lock``).
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
            # Notify the frontend that ``task_history`` has gained a
            # new row so the History sidebar refreshes IMMEDIATELY
            # at task start — not only when the task ends.  Without
            # this, the only refresh trigger sent at start is
            # ``status running=True`` (in ``web_server._run_task_inner``)
            # which fires BEFORE ``_add_task`` has inserted the row;
            # ``refreshHistory`` then fetches a history list that
            # does not yet include the running task and the new
            # task panel never appears in the History sidebar until
            # the task finishes (where ``task_runner`` emits
            # ``tasks_updated`` in its post-task block).  Broadcasting
            # here, immediately after ``_add_task`` has committed
            # the row, makes the running task appear in History
            # right away across all launch paths (VS Code UI, CLI,
            # remote browser, sub-agents).
            broadcast = getattr(printer, "broadcast", None)
            if broadcast is not None:
                try:
                    broadcast({"type": "tasks_updated", "taskId": ""})
                except Exception:
                    pass
        if on_task_id_allocated is not None:
            # Tell the caller (the VS Code task runner) which
            # ``task_history`` row id this run owns, BEFORE any agent
            # event is broadcast.  The server uses the hook to
            # subscribe every other tab that currently has this
            # ``chat_id`` open (in any VS Code window / browser
            # window) to the new task's event stream, so live events
            # reach all viewers of the chat — not only the tab that
            # launched the run.
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

        result_summary = ""
        # Captured for the synthesized result event persisted in the
        # ``finally`` block when this run produced no live event stream
        # (i.e. ran outside a chat webview, with no recording printer).
        result_raw = ""
        try:
            result = super().run(prompt_template=agent_prompt, **kwargs)
            result_raw = result if isinstance(result, str) else ""
            result_summary = _extract_result_summary(result)
            return result
        except Exception:
            result_summary = "Task failed"
            raise
        except BaseException:
            # KeyboardInterrupt (user Stop / graceful daemon shutdown),
            # SystemExit, etc. are NOT matched by ``except Exception``.
            # Persisting the initial ``""`` here would overwrite the
            # "Agent Failed Abruptly" sentinel with an empty string —
            # which the startup orphan sweep (matching the sentinel
            # exactly) can never repair, leaving the row permanently
            # blank (incident: task_history row 3624, killed by the
            # 2026-06-11 00:37:45 daemon restart mid-run).  Persist an
            # explicit marker instead.  Top-level VS Code runs pass
            # ``_skip_persistence=True`` and are unaffected (the task
            # runner's ``_cancel_outcome`` owns their result); this
            # covers sub-agents and CLI/standalone runs.
            result_summary = "Task interrupted"
            raise
        finally:
            # Record the run's outcome so a later auto-commit (e.g.
            # from ``_finalize_worktree`` or merge/teardown paths)
            # can append it to the commit message under "Result:".
            self._last_result_summary = result_summary
            with ChatSorcarAgent._running_agents_lock:
                ChatSorcarAgent.running_agents.pop(task_id, None)
            if registered_here:
                # Mirror the registration above — only the frame that
                # added the entry removes it.  Frames that observed an
                # existing owner (server / worktree / parent
                # sub-agent register) leave cleanup to that owner.
                self._unregister_running_state()
            if printer is not None:
                stop_rec = getattr(printer, "stop_recording", None)
                if stop_rec is not None:
                    try:
                        stop_rec()
                    except Exception:
                        pass
                # Clear the thread-local ``task_id`` so the next
                # sub-agent that runs on this same (reused)
                # ``ThreadPoolExecutor`` worker thread does NOT
                # inherit our task id and mis-route its broadcasts.
                # See the symmetric note above where we set this
                # before emitting ``new_tab``.
                tl = getattr(printer, "_thread_local", None)
                if tl is not None and getattr(tl, "task_id", "") == task_key:
                    tl.task_id = ""
            if not skip_persistence:
                _save_task_result(task_id=task_id, result=result_summary)
                extra_payload = self._build_extra_payload(
                    # The LAUNCH model, not ``self.model_name``: the
                    # ``set_model`` tool mutates the latter mid-task,
                    # and a task's recorded model (surfaced in the chat
                    # webview's History sidebar and other global
                    # consumers) must always be the model the task was
                    # started with — an agent switching its own model
                    # must never change any globally-visible model
                    # preference (INVARIANTS.md #217).
                    model=(
                        getattr(self, "_launch_model_name", "")
                        or self.model_name
                    ),
                    work_dir=self.work_dir,
                    is_parallel=self._is_parallel,
                    # Reuse the value resolved at task start so the
                    # final save can never contradict the early save
                    # (``self.uses_worktree`` is a class-level
                    # capability flag, not a statement about whether
                    # THIS run actually used a worktree).
                    is_worktree=is_worktree,
                )
                extra_payload["tokens"] = self.total_tokens_used
                extra_payload["cost"] = round(self.budget_used, 6)
                _save_task_extra(extra_payload, task_id=task_id)
                # When this run produced NO live event stream — i.e. it
                # ran outside a chat webview (CLI, third-party channel
                # agent, remote webapp) with a printer that does not
                # record/persist events — the ``events`` table is empty
                # for this task and the chat webview would load a blank
                # session.  Synthesize a minimal replayable event stream
                # (the user prompt followed by the result) so the run can
                # still be opened and replayed in the chat webview.  A
                # recording printer (VS Code server's JsonPrinter /
                # WebPrinter) already persisted the full event stream, so
                # ``_task_has_events`` returns True and we skip — no
                # duplication.
                self._persist_replay_events_if_missing(
                    task_id=task_id,
                    prompt=agent_prompt,
                    result_raw=result_raw,
                    result_summary=result_summary,
                )


