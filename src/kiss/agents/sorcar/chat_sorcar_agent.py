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

import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import yaml

from kiss._version import __version__
from kiss.agents.sorcar.persistence import (
    _add_task,
    _load_chat_context,
    _load_task_chat_id,
    _record_frequent_task,
    _save_task_extra,
    _save_task_result,
)
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.sorcar_agent import (
    SorcarAgent,
    _agent_usage,
    _attribute_sub_usage,
    _broadcast_subagent_done,
    _coerce_tasks,
    _yaml_failure,
)

MAX_TASKS = 10


class ChatSorcarAgent(SorcarAgent):
    """SorcarAgent with chat-session state management.

    Maintains a ``chat_id`` and automatically loads prior chat context,
    persists tasks and results to ``sorcar.db``, and augments prompts
    with previous session context — replicating the stateful workflow
    from the VS Code extension as a standalone reusable agent.
    """

    running_agents: dict[int, ChatSorcarAgent] = {}

    def __init__(self, name: str) -> None:
        super().__init__(name)
        # These four attributes are initialized once at construction and
        # must NOT be reset at the start of ``run()``.  The parallel
        # executor (``_run_tasks_parallel``) constructs a fresh
        # sub-agent and pre-sets ``_subagent_info`` (and ``_chat_id``
        # via ``resume_chat_by_id``) BEFORE calling ``run()``; a
        # per-run reset would clobber them and silently disable the
        # ``new_tab`` broadcast that wires the sub-agent's event stream
        # to a fresh frontend tab.
        self._chat_id: str = ""
        self._subagent_info: dict[str, object] | None = None
        self._last_task_id: int | None = None
        self._last_user_prompt: str = ""

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
            "work_dir": work_dir,
            "version": __version__,
            "is_parallel": is_parallel,
            "is_worktree": is_worktree,
        }
        if self._subagent_info is not None:
            payload["subagent"] = self._subagent_info
        return payload

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
        parent_task_id = self._last_task_id
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
        thread_local = getattr(printer, "_thread_local", None) if printer else None
        parent_stop_event = (
            getattr(thread_local, "stop_event", None) if thread_local else None
        )

        sub_usage: list[tuple[float, int, int]] = [(0.0, 0, 0)] * len(tasks)

        def _run_single(args: tuple[int, str]) -> str:
            idx, task = args
            tl = getattr(printer, "_thread_local", None) if printer else None
            if tl is not None:
                tl.stop_event = parent_stop_event
            agent = ChatSorcarAgent(f"Parallel-{task[:40]}")
            if chat_id:
                agent.resume_chat_by_id(chat_id)
            agent._subagent_info = {
                "parent_task_id": parent_task_id,
                "parent_tab_id": parent_tab_id,
            }
            sub_tab_id = f"task-{parent_task_id}__sub_{idx}"
            sub_state = _RunningAgentState(sub_tab_id, model or "")
            sub_state.chat_id = chat_id
            sub_state.is_subagent = True
            sub_state.parent_task_id = parent_task_id
            sub_state.is_task_active = True
            sub_state.agent = agent  # type: ignore[assignment]
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
                )
                return result
            except Exception as exc:
                return _yaml_failure(exc)
            finally:
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

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_run_single, enumerate(tasks)))

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
        # from the parent via ``resume_chat_by_id``.
        if self._chat_id == "":
            self._chat_id = uuid.uuid4().hex
        self._last_task_id = None

        self._last_user_prompt = prompt_template

        agent_prompt = self.build_chat_prompt(prompt_template)

        early_extra = self._build_extra_payload(
            model=kwargs.get("model_name", "") or "",
            work_dir=kwargs.get("work_dir", "") or "",
            is_parallel=bool(kwargs.get("is_parallel", False)),
            # ``pop`` (not ``get``): ``SorcarAgent.run()`` has no
            # ``use_worktree`` parameter, so forwarding it via
            # ``**kwargs`` would raise ``TypeError``.  Only
            # ``WorktreeSorcarAgent`` consumes this kwarg (and pops it
            # before delegating here).
            is_worktree=(
                bool(kwargs.pop("use_worktree", False)) or self.uses_worktree
            ),
        )

        task_id, self._chat_id = _add_task(
            prompt_template, chat_id=self._chat_id, extra=early_extra,
        )
        self._last_task_id = task_id
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
                            "task_id": int(task_id),
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
                on_task_id_allocated(int(task_id), self._chat_id)
            except Exception:
                pass
        if self._subagent_info is None:
            _record_frequent_task(prompt_template)

        result_summary = ""
        try:
            result = super().run(prompt_template=agent_prompt, **kwargs)
            try:
                result_yaml = yaml.safe_load(result)
                if isinstance(result_yaml, dict):
                    summary_val = result_yaml.get("summary", "")
                    if isinstance(summary_val, str):
                        result_summary = summary_val
                    elif summary_val is None:
                        result_summary = ""
                    else:
                        # LLMs sometimes emit a YAML list/mapping under
                        # ``summary``.  Persist its text form — passing
                        # the raw object to ``_save_task_result`` would
                        # raise sqlite3.ProgrammingError from the
                        # ``finally`` block below, destroying the task's
                        # successful return value.
                        result_summary = yaml.safe_dump(
                            summary_val, sort_keys=False,
                        ).strip()
                else:
                    # Valid YAML but not a dict (plain string, list,
                    # number): persist the raw text, consistent with
                    # the parse-failure fallback below — otherwise the
                    # task history records an empty result.
                    result_summary = result[:500] if result else ""
            except Exception:
                result_summary = result[:500] if result else ""
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
            ChatSorcarAgent.running_agents.pop(task_id, None)
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
                    model=self.model_name,
                    work_dir=self.work_dir,
                    is_parallel=self._is_parallel,
                    is_worktree=self.uses_worktree,
                )
                extra_payload["tokens"] = self.total_tokens_used
                extra_payload["cost"] = round(self.budget_used, 6)
                _save_task_extra(extra_payload, task_id=task_id)


