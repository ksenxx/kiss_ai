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

    def _run_tasks_parallel(
        self,
        tasks: list[str],
        max_workers: int | None = None,
    ) -> list[str]:
        """Execute parallel tasks using ChatSorcarAgent sub-agents.

        """
        tasks = _coerce_tasks(tasks)
        model = getattr(self, "model_name", None)
        work_dir = getattr(self, "work_dir", None)
        chat_id = self._chat_id
        parent_task_id = self._last_task_id
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
            agent._subagent_info = {"parent_task_id": parent_task_id}
            sub_tab_id = f"task-{parent_task_id}__sub_{idx}"
            sub_state = _RunningAgentState(sub_tab_id, model or "")
            sub_state.chat_id = chat_id
            sub_state.is_subagent = True
            sub_state.parent_task_id = parent_task_id
            sub_state.is_task_active = True
            sub_state.agent = agent  # type: ignore[assignment]
            _RunningAgentState.running_agent_states[sub_tab_id] = sub_state
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
                _RunningAgentState.running_agent_states.pop(sub_tab_id, None)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_run_single, enumerate(tasks)))

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
        subscribe_tab_id = kwargs.pop("_subscribe_tab_id", "")
        # Mint a fresh chat id only if no caller (or prior ``run()``)
        # already established one.  Resetting unconditionally here would
        # discard the ``chat_id`` that ``_run_tasks_parallel`` propagates
        # from the parent via ``resume_chat_by_id``.
        if self._chat_id == "":
            self._chat_id = uuid.uuid4().hex
        self._last_task_id = None

        self._last_user_prompt = prompt_template

        agent_prompt = self.build_chat_prompt(prompt_template)

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
        ChatSorcarAgent.running_agents[task_id] = self
        # Mirror this run's task_history_id onto the per-thread
        # sub-agent state so ``VSCodeServer._reattach_running_chat``
        # can disambiguate the sub-agent from its parent by task id.
        if self._subagent_info is not None:
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
                        broadcast({
                            "type": "new_tab",
                            "task_id": int(task_id),
                            "taskId": "",
                        })
                    except Exception:
                        pass
            persist_map = getattr(printer, "_persist_agents", None)
            if persist_map is not None:
                persist_map[task_key] = self
            subscribe = getattr(printer, "subscribe_tab", None)
            if subscribe is not None and subscribe_tab_id:
                subscribe(task_id, subscribe_tab_id)
            start_rec = getattr(printer, "start_recording", None)
            if start_rec is not None:
                start_rec()
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


