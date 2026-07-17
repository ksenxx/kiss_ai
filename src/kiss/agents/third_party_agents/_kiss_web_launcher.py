# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Launch third-party agents as kiss-web registered agents.

Instead of calling ``SorcarAgent.run`` directly, every agent in
``kiss/agents/third_party_agents/`` is launched through
:func:`run_agent_via_kiss_web`, which drives
:meth:`kiss.server.commands._CommandsMixin._cmd_run` on a
:class:`~kiss.server.server.VSCodeServer`.  The launcher
pre-registers the agent instance in
:attr:`_RunningAgentState.running_agent_states` (the *kiss-web
registered agent* registry), so while the task runs it is
discoverable, openable and interactable from any connected remote
webview — exactly like a task started from the chat UI.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import yaml

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

if TYPE_CHECKING:
    from kiss.server.server import VSCodeServer

logger = logging.getLogger(__name__)

# Process-global lazily-created server used when the caller does not
# provide one (CLI channel agents, cron pollers).  Guarded by
# ``_DEFAULT_SERVER_LOCK`` so concurrent first launches build exactly
# one server.
_DEFAULT_SERVER: VSCodeServer | None = None
_DEFAULT_SERVER_LOCK = threading.Lock()


def _failure_yaml(exc: BaseException) -> str:
    """Return a YAML failure envelope for a task exception."""
    summary = "Task interrupted" if isinstance(exc, KeyboardInterrupt) else f"Task failed: {exc}"
    return str(yaml.safe_dump({"success": False, "summary": summary}, sort_keys=False))


def default_server() -> VSCodeServer:
    """Return the process-global :class:`VSCodeServer`, creating it lazily.

    The server hosts ``_cmd_run`` / the running-agent registry for
    third-party launches in processes that do not already run a
    ``kiss-web`` daemon (channel CLIs, cron pollers).

    Returns:
        The cached process-global ``VSCodeServer`` instance.
    """
    global _DEFAULT_SERVER
    with _DEFAULT_SERVER_LOCK:
        if _DEFAULT_SERVER is None:
            from kiss.server.server import VSCodeServer

            _DEFAULT_SERVER = VSCodeServer()
        return _DEFAULT_SERVER


class KissWebChatSorcarAgent(ChatSorcarAgent):
    """Chat-session agent that records its run result for the launcher.

    ``_cmd_run``'s task runner discards the YAML string returned by
    ``agent.run`` (it only broadcasts / persists the summary), so the
    launcher reads :attr:`last_run_result` off the agent instance
    after the task thread finishes.
    """

    last_run_result: str = ""

    def run(self, prompt_template: str = "", **kwargs: Any) -> str:  # type: ignore[override]
        """Run the chat agent and record the returned YAML result.

        Args:
            prompt_template: The task prompt.
            **kwargs: Forwarded to :meth:`ChatSorcarAgent.run`.

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        try:
            result = super().run(prompt_template=prompt_template, **kwargs)
        except BaseException as exc:
            self.last_run_result = _failure_yaml(exc)
            raise
        self.last_run_result = result
        return result


class KissWebWorktreeSorcarAgent(WorktreeSorcarAgent):
    """Worktree agent that records its run result for the launcher."""

    last_run_result: str = ""

    def run(self, prompt_template: str = "", **kwargs: Any) -> str:  # type: ignore[override]
        """Run the worktree agent and record the returned YAML result.

        Args:
            prompt_template: The task prompt.
            **kwargs: Forwarded to :meth:`WorktreeSorcarAgent.run`.

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        try:
            result = super().run(prompt_template=prompt_template, **kwargs)
        except BaseException as exc:
            self.last_run_result = _failure_yaml(exc)
            raise
        self.last_run_result = result
        return result


def run_agent_via_kiss_web(
    agent: Any,
    prompt_template: str,
    *,
    model_name: str = "",
    work_dir: str = "",
    max_budget: float | None = None,
    tools: list[Callable[..., Any]] | None = None,
    use_worktree: bool = False,
    model_config: dict[str, Any] | None = None,
    web_tools: bool | None = None,
    is_parallel: bool = False,
    server: VSCodeServer | None = None,
    timeout: float | None = None,
) -> str:
    """Launch *agent* as a kiss-web registered agent via ``_cmd_run``.

    Pre-registers a :class:`_RunningAgentState` whose ``agent`` slot
    holds *agent* (the documented injection point honoured by
    ``_TaskRunnerMixin._run_task_inner``), then submits a ``run``
    command through :meth:`VSCodeServer._cmd_run`.  The task executes
    on ``_cmd_run``'s background worker thread with full kiss-web
    lifecycle: live event broadcasts to every connected webview,
    ``pending_user_messages`` follow-up injection, stop support, and
    registry-based discovery.  Blocks until the task thread finishes
    (or *timeout* elapses) and returns the agent's YAML result.

    Args:
        agent: The third-party agent instance to run.  Extra run-time
            kwargs injected by the task runner are absorbed by
            ``BaseChannelAgent.run`` (channel agents) or natively by
            ``ChatSorcarAgent.run`` (poller agents).
        prompt_template: The task prompt.
        model_name: LLM model name; empty selects the server default.
        work_dir: Working directory for the run.
        max_budget: Per-task budget override in USD; ``None`` uses the
            kiss-web config default.
        tools: Extra tool callables merged into the agent's run tools
            (stashed on ``agent._extra_run_tools``; consumed by
            ``BaseChannelAgent.run``).
        use_worktree: Run inside an isolated git worktree (only
            meaningful for :class:`WorktreeSorcarAgent`-derived
            agents).
        model_config: Per-task model configuration override (custom
            endpoint / headers), matching ``SorcarAgent.run``.
        web_tools: Per-task browser-tool enablement override. ``None``
            uses the kiss-web config default.
        is_parallel: Whether the agent may spawn parallel sub-agents.
        server: The :class:`VSCodeServer` to launch on.  ``None`` uses
            the process-global :func:`default_server`.
        timeout: Max seconds to wait for the task thread; ``None``
            waits indefinitely.  On timeout the task keeps running in
            the background and ``""`` is returned.

    Returns:
        The YAML result string recorded by the agent
        (``last_run_result``), or ``""`` when unavailable (timeout, or
        a failure before the agent's run recorded a result).
    """
    srv = server if server is not None else default_server()

    agent._extra_run_tools = list(tools) if tools else []
    # Always overwrite per-run overrides so reusing an agent instance
    # cannot leak a prior launch's budget / endpoint / web-tools
    # settings into the next launch.
    agent._max_budget_override = max_budget
    agent._model_config_override = dict(model_config) if model_config is not None else None
    agent._web_tools_override = web_tools

    tab_id = f"tp-{uuid.uuid4().hex}"
    with srv._state_lock:
        state = _RunningAgentState(tab_id, srv._default_model)
        state.agent = agent
        state.auto_commit_mode = False
        # Preserve a chat id the agent already carries (a poller
        # resuming a chat via ``resume_chat_by_id``).  ``_cmd_run``
        # mints a fresh uuid only when ``state.chat_id`` is empty, and
        # the task runner then syncs ``tab.chat_id`` onto
        # ``agent._chat_id`` — so without this the resumed chat id
        # would be clobbered by the minted one.
        state.chat_id = str(
            getattr(agent, "chat_id", "") or getattr(agent, "_chat_id", "") or "",
        )
        _RunningAgentState.register(tab_id, state)

    try:
        srv._cmd_run(
            {
                "type": "run",
                "tabId": tab_id,
                "prompt": prompt_template,
                "model": model_name,
                "workDir": work_dir,
                "useWorktree": use_worktree,
                "autoCommit": False,
                "useParallel": is_parallel,
            }
        )
    except BaseException:
        with srv._state_lock:
            state.frontend_closed = True
        srv._dispose_if_closed(tab_id)
        raise

    with srv._state_lock:
        thread = state.task_thread
    # ``_cmd_run`` always sets ``task_thread`` before returning (it
    # rolls the assignment back only on the raise path handled above),
    # so the thread is guaranteed to exist here.
    assert thread is not None
    thread.join(timeout=timeout)
    if thread.is_alive():
        logger.warning(
            "kiss-web launch timed out after %.1fs (tab %s); the "
            "task keeps running in the background",
            timeout or 0.0,
            tab_id,
        )
        # Mark for deferred disposal once the task finishes.
        with srv._state_lock:
            state.frontend_closed = True
        return ""

    result = str(getattr(agent, "last_run_result", "") or "")

    # Dispose the launcher-owned registry entry now that the task is
    # over — mirrors the frontend closing its tab.
    with srv._state_lock:
        state.frontend_closed = True
    srv._dispose_if_closed(tab_id)
    return result
