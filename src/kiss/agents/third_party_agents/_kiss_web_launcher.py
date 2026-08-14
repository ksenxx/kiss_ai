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

The agent's live channel tools (authentication closures, authenticated
backend bound methods, per-message ``reply`` closures) are supplied
through the API's ``tools=`` *file path* contract via
:mod:`._api_tools_bridge`: the launcher generates a real tools file
whose top-level wrappers dispatch back to the live callables, which
works because the daemon the launcher talks to runs in this same
process (see :func:`_ensure_api_server`).
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from kiss.agents.third_party_agents._api_tools_bridge import (
    register_tools,
    release_tools,
)
from kiss.agents.third_party_agents._channel_agent_utils import BaseChannelAgent

if TYPE_CHECKING:
    from kiss.server.web_server import RemoteAccessServer

logger = logging.getLogger(__name__)

_NO_TIMEOUT_SECONDS = 10 * 365 * 24 * 3600.0

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
    to this socket.  The daemon MUST live in this process: the live
    tool callables bridged by :mod:`._api_tools_bridge` are reachable
    only through this process's registry.

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
    pollers call :meth:`resume_chat_by_id` before launching and read
    :attr:`chat_id` after, so each conversation thread maps to a
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
    tools: list[Callable[..., Any]] | None = None,
    use_worktree: bool = True,
    model_config: dict[str, Any] | None = None,
    web_tools: bool | None = None,
    is_parallel: bool = True,
    timeout: float | None = None,
    sock_path: str | None = None,
) -> str:
    """Launch *agent*'s task through :func:`kiss.server.sorcar.run`.

    Bridges the agent's live channel tools into a generated tools file
    (:func:`~._api_tools_bridge.register_tools`), appends the agent's
    ``channel_system_prompt`` guidance to the prompt (the API carries
    no system prompt), and submits the task to the in-process kiss-web
    daemon over its Unix-domain socket.  Blocks until the daemon
    reports the task finished (or *timeout* elapses) and returns the
    task's YAML result.

    The passed *agent* instance is never executed — the daemon builds
    its own chat agent.  The instance serves as the carrier of channel
    tools and chat identity: the launcher propagates the daemon chat
    id onto it (so pollers can resume the conversation), records the
    YAML result in ``agent.last_run_result``, and copies the task's
    cost / token / step totals onto the instance for CLI run stats.

    Args:
        agent: The third-party agent instance supplying channel tools,
            ``channel_system_prompt`` guidance, and the chat id to
            continue (``agent.chat_id`` on :class:`KissWebChatAgent`
            carriers).
        prompt_template: The task prompt.
        model_name: LLM model name; empty selects the daemon default.
        work_dir: Working directory for the run.
        max_budget: Per-task budget override in USD; ``None`` uses the
            kiss-web config default.
        tools: Extra live tool callables bridged into the task's tools
            (e.g. ``ChannelRunner``'s per-message ``reply`` closure).
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
        ValueError: When a live tool cannot be expressed through the
            API's tools-file contract (see
            :func:`~._api_tools_bridge.register_tools`).
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
    # methods) plus any launcher-supplied extras (e.g. ChannelRunner's
    # per-message ``reply`` closure); the daemon-built agent supplies
    # the standard tools itself.
    live_tools = [*agent._get_tools(), *(tools or [])]
    token = ""
    tools_path: str | None = None
    if live_tools:
        token, tools_path = register_tools(live_tools)
    sock = sock_path or _SOCK_PATH_OVERRIDE or _ensure_api_server()
    try:
        try:
            result = sorcar.run(
                prompt,
                work_dir=work_dir,
                model=model_name,
                chat_id=chat_id,
                tools=tools_path,
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
        if token:
            release_tools(token)

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
