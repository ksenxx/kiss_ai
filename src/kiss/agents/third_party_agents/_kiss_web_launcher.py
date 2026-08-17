# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Launch third-party agents through the ``kiss.server.sorcar.run`` API.

Every agent in ``kiss/agents/third_party_agents/`` is launched through
:func:`run_agent_via_kiss_web`, which is implemented on top of the
public synchronous client API :func:`kiss.server.sorcar.run`: the
launcher connects to a kiss-web daemon's Unix-domain socket, submits
the documented ``run`` command, and blocks until the daemon reports
the task finished.  The task therefore executes with the full kiss-web
lifecycle — live event broadcasts to every connected webview,
follow-up message injection, stop support, chat persistence — exactly
like a task started from the chat UI.

The agent's channel tools are supplied through the API's ``tools=``
*file path* contract directly: each agent module defines a top-level
``get_tools()`` that builds a fresh agent from the credentials
persisted under ``~/.kiss`` and returns its authentication and backend
tools, so the agent's OWN module file (``agent.tools_file``) is passed
as the ``tools=`` argument and the daemon imports it and calls its
``get_tools()``.  No bridge, registry, wrapper, or generated file is
involved.  The active workspace travels to the daemon-side
``get_tools()`` through the ``KISS_CHANNEL_WORKSPACE`` environment
variable.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from kiss.agents.sorcar.channel_workspace import (
    enter_workspace as _enter_workspace,
)
from kiss.agents.sorcar.channel_workspace import (
    exit_workspace as _exit_workspace,
)
from kiss.agents.third_party_agents._channel_agent_utils import BaseChannelAgent

if TYPE_CHECKING:
    from kiss.server.web_server import RemoteAccessServer

logger = logging.getLogger(__name__)

_NO_TIMEOUT_SECONDS = 10 * 365 * 24 * 3600.0

# The workspace publication registry (reference-counted, shared with
# the sorcar ``run_agent`` dispatch tool) lives in
# ``kiss.agents.sorcar.channel_workspace``; the private aliases above
# keep this module's public surface unchanged.

_API_SERVER: RemoteAccessServer | None = None
_API_SERVER_SOCK: str = ""
_API_SERVER_LOCK = threading.Lock()

_SOCK_PATH_OVERRIDE: str | None = None


def _ensure_api_server() -> str:
    """Start the process-global in-process daemon; return its UDS path.

    Creates one :class:`~kiss.server.web_server.RemoteAccessServer` —
    the production daemon class — serving only a private Unix-domain
    socket (mode 0o600 in a private temp directory) on a dedicated
    asyncio loop thread.  The launcher's ``sorcar.run`` calls connect
    to this socket, so channel agents work without any externally
    started kiss-web daemon.

    Returns:
        The Unix-domain socket path of the in-process daemon.
    """
    global _API_SERVER, _API_SERVER_SOCK
    with _API_SERVER_LOCK:
        if _API_SERVER is None:
            from kiss.server.web_server import RemoteAccessServer

            sock_dir = tempfile.mkdtemp(prefix="kiss-tp-api-")
            sock_path = str(Path(sock_dir) / "sorcar.sock")
            loop = asyncio.new_event_loop()
            threading.Thread(
                target=loop.run_forever,
                name="kiss-tp-api-server",
                daemon=True,
            ).start()
            server = RemoteAccessServer(uds_path=sock_path)
            server._printer._loop = loop
            server._loop = loop
            asyncio.run_coroutine_threadsafe(
                asyncio.start_unix_server(
                    server._uds_handler, path=sock_path,
                ),
                loop,
            ).result(timeout=30)
            _API_SERVER = server
            _API_SERVER_SOCK = sock_path
        return _API_SERVER_SOCK


class KissWebChatAgent(BaseChannelAgent):
    """Chat-session carrier for API launches.

    A plain :class:`BaseChannelAgent` (no auth tools, no backend) that
    additionally carries the daemon chat id across launches: the
    channel runner calls :meth:`resume_chat_by_id` before launching and
    reads :attr:`chat_id` after, so each conversation thread maps to a
    persistent daemon chat.  Like every channel agent it never runs
    anything itself — the inherited ``run()`` submits the task through
    :func:`kiss.server.sorcar.run` via :func:`run_agent_via_kiss_web`,
    which records the YAML result in ``last_run_result`` plus the
    cost / token / step totals.
    """

    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        self._chat_id: str = ""

    @property
    def chat_id(self) -> str:
        """The daemon chat-session identifier carried across launches."""
        return self._chat_id

    def new_chat(self) -> None:
        """Start a fresh chat: the next launch gets a new daemon chat id."""
        self._chat_id = ""

    def resume_chat_by_id(self, chat_id: str) -> None:
        """Resume an existing chat session on the next launch.

        Args:
            chat_id: String chat session identifier to resume.
        """
        if chat_id:
            self._chat_id = chat_id


def run_agent_via_kiss_web(
    agent: BaseChannelAgent,
    prompt_template: str,
    *,
    model_name: str = "",
    work_dir: str = "",
    max_budget: float | None = None,
    tools: str | Path | None = None,
    use_worktree: bool = True,
    model_config: dict[str, Any] | None = None,
    web_tools: bool | None = None,
    is_parallel: bool = True,
    timeout: float | None = None,
    sock_path: str | None = None,
) -> str:
    """Launch *agent*'s task through :func:`kiss.server.sorcar.run`.

    Supplies the agent's channel tools through the API's ``tools=``
    file-path contract (by default ``agent.tools_file`` — the agent's
    own module, whose top-level ``get_tools()`` the daemon calls to
    build a fresh agent from the credentials persisted under
    ``~/.kiss``), appends the agent's ``channel_system_prompt``
    guidance to the prompt (the API carries no system prompt), and
    submits the task to the in-process kiss-web daemon over its
    Unix-domain socket.  Blocks until the daemon reports the task
    finished (or *timeout* elapses) and returns the task's YAML result.

    While the task runs, the ``KISS_CHANNEL_WORKSPACE`` environment
    variable holds ``agent.workspace`` so the daemon-side
    ``get_tools()`` authenticates under the same workspace (concurrent
    launches from one process should therefore use the same
    workspace).

    The passed *agent* instance is never executed — the daemon builds
    its own chat agent.  The instance serves as the carrier of channel
    identity: the launcher propagates the daemon chat id onto it (so
    the channel runner can resume the conversation), records the YAML
    result in
    ``agent.last_run_result``, and copies the task's cost / token /
    step totals onto the instance for CLI run stats.

    Args:
        agent: The third-party agent instance supplying the channel
            tools file (``agent.tools_file``), the workspace,
            ``channel_system_prompt`` guidance, and the chat id to
            continue (``agent.chat_id`` on :class:`KissWebChatAgent`
            carriers).
        prompt_template: The task prompt.
        model_name: LLM model name; empty selects the daemon default.
        work_dir: Working directory for the run.
        max_budget: Per-task budget override in USD; ``None`` uses the
            kiss-web config default.
        tools: Path of a Python file whose top-level ``get_tools()``
            supplies the task's extra tools (the API's tools-file
            contract).  ``None`` uses ``agent.tools_file`` — the
            agent's own module.
        use_worktree: Run the task in an isolated git worktree.
        model_config: Per-task model configuration override (custom
            endpoint / headers).
        web_tools: Per-task browser-tool enablement override. ``None``
            uses the kiss-web config default.
        is_parallel: Whether the agent may spawn parallel sub-agents.
        timeout: Max seconds to wait for the task; ``None`` waits
            indefinitely.  On timeout the task keeps running in the
            daemon and ``""`` is returned.
        sock_path: Daemon UDS path override.  ``None`` uses the
            process-global in-process daemon
            (:func:`_ensure_api_server`).

    Returns:
        YAML string with 'success' and 'summary' keys, or ``""`` when
        the task did not finish within *timeout*.

    Raises:
        ValueError: When *tools* (or ``agent.tools_file``) is not the
            path of an existing Python file.
        ConnectionError: When the daemon socket cannot be reached.
    """
    from kiss.server import sorcar

    prompt = prompt_template + agent.channel_system_prompt
    if not prompt.strip():
        result_yaml = str(yaml.safe_dump(
            {"success": False, "summary": "Task failed: empty prompt"},
            sort_keys=False,
        ))
        agent.last_run_result = result_yaml
        return result_yaml
    chat_id = agent.chat_id if isinstance(agent, KissWebChatAgent) else ""
    # The agent's channel tools (auth tools + authenticated backend
    # methods) are built inside the daemon: it imports the tools file
    # and calls its get_tools(); the daemon-built agent supplies the
    # standard tools itself.
    tools_path = str(tools) if tools else agent.tools_file
    sock = sock_path or _SOCK_PATH_OVERRIDE or _ensure_api_server()
    _enter_workspace(agent.workspace)
    try:
        try:
            result = sorcar.run(
                prompt,
                work_dir=work_dir,
                model=model_name,
                chat_id=chat_id,
                tools=tools_path or None,
                use_worktree=use_worktree,
                max_budget=max_budget,
                model_config=model_config,
                web_tools=web_tools,
                is_parallel=is_parallel,
                timeout=timeout if timeout is not None else _NO_TIMEOUT_SECONDS,
                sock_path=sock,
            )
        except TimeoutError:
            logger.warning(
                "kiss-web API launch timed out after %.1fs; the task "
                "keeps running in the daemon",
                timeout or 0.0,
            )
            return ""
    finally:
        _exit_workspace(agent.workspace)

    summary = result.text or ("" if result.success else "Task failed")
    result_yaml = str(yaml.safe_dump(
        {"success": result.success, "summary": summary},
        sort_keys=False,
    ))
    if result.chat_id and isinstance(agent, KissWebChatAgent):
        agent._chat_id = result.chat_id
    agent.last_run_result = result_yaml
    agent.budget_used = result.cost
    agent.total_tokens_used = result.tokens
    agent.total_steps = result.steps
    return result_yaml
