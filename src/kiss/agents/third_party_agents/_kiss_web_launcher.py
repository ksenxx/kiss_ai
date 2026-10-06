# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Launch third-party agents through the ``kiss.server.sorcar.run`` API.

Every agent in ``kiss/agents/third_party_agents/`` is launched through
:func:`run_agent_via_kiss_web`, which is implemented on top of the
public synchronous client API :func:`kiss.server.sorcar.run`: the
launcher connects to a kiss-web daemon's local WSS endpoint, submits
the documented ``run`` command, and blocks until the daemon reports
the task finished.  The task therefore executes with the full kiss-web
lifecycle — live event broadcasts to every connected webview,
follow-up message injection, stop support, chat persistence — exactly
like a task started from the chat UI.

The agent's channel tools are supplied through the API's
``extension_agent_path`` SEA contract directly: each agent module
defines a SEA class whose ``tools()`` builds a fresh agent from the
credentials persisted under the KISS home and adds its authentication
and backend tools, so the agent's OWN module file (``agent.sea_path``)
is passed as ``extension_agent_path`` and the daemon loads it and runs
its ``tools()``.  No bridge, registry, wrapper, or generated file is
involved.  The agent's workspace travels on the ``run`` command's
``workspace`` field; the daemon holds it for the run's lifetime and
publishes it to the daemon-side ``tools()`` through
``KISS_CHANNEL_WORKSPACE``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import shutil
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from kiss.agents.third_party_agents._channel_agent_utils import BaseChannelAgent

if TYPE_CHECKING:
    from kiss.server.web_server import RemoteAccessServer

_API_SERVER: RemoteAccessServer | None = None
_API_SERVER_ENDPOINT: str = ""
_API_SERVER_THREAD: threading.Thread | None = None
_API_SERVER_LOCK = threading.Lock()

_ENDPOINT_FILE_OVERRIDE: str | None = None
"""Endpoint file every launch uses instead of the process-global daemon.

Set by tests that run their own daemon; ``None`` (the default) selects
:func:`_ensure_api_server`.
"""


def _ensure_api_server() -> str:
    """Start the process-global in-process daemon; return its endpoint file.

    Creates one :class:`~kiss.server.web_server.RemoteAccessServer` —
    the production daemon class — serving only a loopback WSS listener
    on an ephemeral port, whose endpoint file (mode 0600, in a private
    temp directory) carries a private local token, on a dedicated
    asyncio loop thread.  The launcher's ``sorcar.run`` calls connect
    through that file, so channel agents work without any externally
    started kiss-web daemon.

    Returns:
        The endpoint file path of the in-process daemon.
    """
    global _API_SERVER, _API_SERVER_ENDPOINT, _API_SERVER_THREAD
    with _API_SERVER_LOCK:
        if _API_SERVER is None:
            from kiss.server.web_server import RemoteAccessServer

            private_dir = tempfile.mkdtemp(prefix="kiss-tp-api-")
            endpoint_file = str(Path(private_dir) / "sorcar-local.json")
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever,
                name="kiss-tp-api-server",
                daemon=True,
            )
            thread.start()
            server: RemoteAccessServer | None = None
            startup: concurrent.futures.Future[None] | None = None
            try:
                server = RemoteAccessServer(
                    host="127.0.0.1", port=0, local_endpoint_file=endpoint_file,
                )
                # This daemon shares the KISS home (database, chats) with
                # the canonical kiss-web daemon but must not share its tab
                # registry: ``tabs.json`` tolerates exactly one owner, and
                # a second one erased the canonical daemon's tabs with its
                # stale snapshot on every channel run.  The channel tabs
                # are transient, so they live next to the private
                # endpoint file.
                server._vscode_server.use_private_tab_registry(
                    Path(private_dir) / "tabs.json",
                )
                startup = asyncio.run_coroutine_threadsafe(
                    server.start_private_async(), loop,
                )
                startup.result(timeout=60)
            except BaseException:
                # Without this the ``run_forever`` daemon thread (and any
                # listener the timed-out coroutine went on to create)
                # would outlive the failed attempt, one per retry.
                _abort_api_server_startup(loop, thread, server, startup, private_dir)
                raise
            _API_SERVER = server
            _API_SERVER_ENDPOINT = endpoint_file
            _API_SERVER_THREAD = thread
        return _API_SERVER_ENDPOINT


def _stop_api_server() -> None:
    """Stop the process-global daemon started by :func:`_ensure_api_server`.

    Closes its listeners and endpoint file (``stop_async``), stops and
    joins its loop thread, removes the private directory and resets the
    globals so the next launch starts a fresh daemon.  A no-op when no
    daemon is running.  The whole teardown runs under ``_API_SERVER_LOCK``
    (like the startup in :func:`_ensure_api_server`), so a concurrent
    launch cannot publish a replacement daemon while the old one is still
    shutting down.  Tests that start the global daemon against a
    temporary environment call this so later tests never reuse a daemon
    wired to a deleted tmpdir.
    """
    global _API_SERVER, _API_SERVER_ENDPOINT, _API_SERVER_THREAD
    with _API_SERVER_LOCK:
        server, thread, endpoint = _API_SERVER, _API_SERVER_THREAD, _API_SERVER_ENDPOINT
        _API_SERVER, _API_SERVER_THREAD, _API_SERVER_ENDPOINT = None, None, ""
        if server is None or thread is None or server._loop is None:
            return
        _abort_api_server_startup(
            server._loop, thread, server, None, str(Path(endpoint).parent),
        )


async def _stop_after_startup(server: RemoteAccessServer) -> None:
    """Stop *server* once its (possibly cancelled) startup has settled.

    Then cancels every other task still on the loop so that stopping the
    loop afterwards destroys nothing while pending.
    """
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await server.stop_async()
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def _stop_loop_after_pending_cancels() -> None:
    # A task cancelled from another thread receives its CancelledError
    # one iteration later and reports completion the iteration after
    # that; yielding twice lets it finish instead of being destroyed
    # while pending when the loop stops.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    asyncio.get_running_loop().stop()


def _abort_api_server_startup(
    loop: asyncio.AbstractEventLoop,
    thread: threading.Thread,
    server: RemoteAccessServer | None,
    startup: concurrent.futures.Future[None] | None,
    private_dir: str,
) -> None:
    """Tear down a half-started private daemon after a startup failure.

    Cancels the pending startup coroutine (or stops the server it
    already bound), stops the loop thread and joins it with a finite
    timeout, closes the loop once it has stopped, and removes the
    private directory.  Every step is best-effort: this runs on an
    error path and must never mask the original exception.

    Args:
        loop: The dedicated event loop the thread is running.
        thread: The ``run_forever`` thread.
        server: The server being started, or ``None`` when the failure
            happened before it was constructed.
        startup: The ``start_private_async`` future, or ``None`` when
            the failure happened before it was submitted.
        private_dir: The private temp directory holding the endpoint
            file and tab registry.
    """
    if startup is not None:
        startup.cancel()
    if server is not None:
        # Whether the cancel landed (the coroutine rolled its listener
        # back) or lost the race with completion (a bound listener and a
        # published endpoint exist), ``stop_async`` closes what is left;
        # it waits for the startup's lifecycle lock, so it runs after
        # the startup has finished either way.
        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(
                _stop_after_startup(server), loop,
            ).result(timeout=10)
    # Not awaited: the loop stops before this future's completion
    # callback could run, so ``thread.join`` is the wait.
    asyncio.run_coroutine_threadsafe(_stop_loop_after_pending_cancels(), loop)
    thread.join(timeout=2)
    if not thread.is_alive():
        loop.close()
    shutil.rmtree(private_dir, ignore_errors=True)


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
    cost / token / step totals.  This module defines no SEA class, so
    the carrier adds no channel tools of its own;
    the channel runner passes the channel module's path as *sea_path*
    so the launch still gets that channel's tools.
    """

    def __init__(self, name: str = "", sea_path: str = "") -> None:
        super().__init__(name)
        self._chat_id: str = ""
        self._sea_path = sea_path

    @property
    def sea_path(self) -> str:
        """Path of the SEA file whose ``tools()`` supplies the launch's tools.

        The *sea_path* given at construction (a channel module, for the
        channel runner's launches), or ``""`` for no extra tools.
        """
        return self._sea_path

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
    model_config: dict[str, Any] | None = None,
    web_tools: bool | None = None,
    is_parallel: bool = True,
) -> str:
    """Launch *agent*'s task through :func:`kiss.server.sorcar.run`.

    Supplies the agent's channel tools through the API's
    ``extension_agent_path`` SEA contract (``agent.sea_path`` — the
    agent's own module, whose SEA class's ``tools()`` the daemon runs
    to build a fresh agent from the credentials persisted under the
    KISS home and whose ``settings()`` / ``system_prompt()``
    make the run a ``channel``-kind session with the channel's
    guidance in its system prompt), and submits the task to the in-process kiss-web daemon over its
    Unix-domain socket.  Blocks until the daemon reports the task
    finished and returns the task's YAML result.

    The keyword surface is exactly what the channel CLI and the
    channel runner pass; everything else about the run (worktree,
    auto-commit, browser, memory, work directory, extra prompt text)
    comes from the channel SEA's ``settings()`` / ``system_prompt()``
    / ``prompt()`` methods, which the ``channel`` kind locks.  The
    daemon is the process-global in-process one
    (:func:`_ensure_api_server`) unless a test has pointed the module
    at another endpoint file.

    ``agent.workspace`` is sent as the run's ``workspace``; the daemon
    holds it while the task runs (a launch whose workspace differs from
    a running channel task's waits for that task to finish) and exports
    it as ``KISS_CHANNEL_WORKSPACE`` so the daemon-side SEA ``tools()``
    authenticates under the same workspace.

    The passed *agent* instance is never executed — the daemon builds
    its own chat agent.  The instance serves as the carrier of channel
    identity: the launcher propagates the daemon chat id onto it (so
    the channel runner can resume the conversation), records the YAML
    result in
    ``agent.last_run_result``, and copies the task's cost / token /
    step totals onto the instance for CLI run stats.

    Args:
        agent: The third-party agent instance supplying the channel
            agent script (``agent.sea_path``), the workspace, and the chat id to
            continue (``agent.chat_id`` on :class:`KissWebChatAgent`
            carriers).
        prompt_template: The task prompt.
        model_name: LLM model name; empty selects the daemon default.
        work_dir: Working directory for the run.
        max_budget: Per-task budget override in USD; ``None`` uses the
            kiss-web config default.
        model_config: Per-task model configuration override (custom
            endpoint / headers).
        web_tools: Per-task browser-tool enablement override. ``None``
            uses the kiss-web config default.
        is_parallel: Whether the agent may spawn parallel sub-agents.

    Returns:
        YAML string with 'success' and 'summary' keys.

    Raises:
        ValueError: When ``agent.sea_path`` is not the path of an
            existing Python file.
        ConnectionError: When the daemon cannot be reached.
    """
    from kiss.server import sorcar

    prompt = prompt_template
    if not prompt.strip():
        result_yaml = str(yaml.safe_dump(
            {"success": False, "summary": "Task failed: empty prompt"},
            sort_keys=False,
        ))
        agent.last_run_result = result_yaml
        return result_yaml
    chat_id = agent.chat_id if isinstance(agent, KissWebChatAgent) else ""
    # The agent's channel tools (auth tools + authenticated backend
    # methods) are built inside the daemon: it imports the agent's
    # module as the run's SEA and runs its tools() method;
    # the daemon-built agent supplies the standard tools itself.
    endpoint = _ENDPOINT_FILE_OVERRIDE or _ensure_api_server()
    # The workspace travels on the wire; the daemon holds it (and
    # publishes it to ``KISS_CHANNEL_WORKSPACE``) for the run's lifetime,
    # exactly as it does for ``/slack ...`` and ``run_agent("slack")``.
    # The values a caller of this launcher can be seen to have passed
    # (the others have defaults indistinguishable from "not given"):
    # they win over the channel script's settings, or clash with a
    # locked one (kiss.agents.sorcar.sea_settings.locked_conflicts).
    given = {
        "model": model_name, "max_budget": max_budget,
        "model_config": model_config, "use_web_tools": web_tools,
    }
    provenance = {key: "explicit" for key, value in given.items() if value not in (None, "")}
    result = sorcar.run(
        prompt,
        work_dir=work_dir,
        model=model_name,
        chat_id=chat_id,
        extension_agent_path=agent.sea_path,
        max_budget=max_budget,
        model_config=model_config,
        use_web_tools=web_tools,
        allow_fan_out=is_parallel,
        workspace=agent.workspace,
        provenance=provenance,
        timeout=None,
        endpoint_file=endpoint,
    )
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
