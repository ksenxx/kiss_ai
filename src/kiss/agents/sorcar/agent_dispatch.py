# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Immediate dispatch of tasks to agent scripts and channel agents.

Gives the Sorcar agent a single tool, ``run_agent`` (built per task by
:func:`make_run_agent_tool`), that runs any agent on a task right away
— an installed third-party channel agent named by channel (Slack,
Telegram, Discord, email, WhatsApp, Home Assistant, ...), the built-in
``cron`` agent (the scheduled-automations agent script
``kiss.agents.sorcar.cron_agent``, which supplies the ``cron_job``
tool), or an arbitrary *agent script* named by its ``.py`` file path —
so a request like "Send 'hello' to the #sorcar Slack channel", "every
morning at 9 summarize my inbox", or "run my_agent.py on this task" is
executed in one tool call instead of the agent first rediscovering
what those agents are and how they work.

The channel agents are looked up dynamically, the same soft-plugin
style the cron deliverer uses: any module named
``kiss.agents.third_party_agents.<channel>_agent`` that defines a
``BaseChannelAgent`` subclass is dispatchable.  This module never
imports ``kiss.agents.third_party_agents`` statically — only the
requested channel module is imported, dynamically, at dispatch time
(the layering invariant in
``kiss.tests.agents.sorcar.test_layering_invariants`` forbids more) —
so it works (with an empty channel list) when those optional modules
are absent.

Each dispatch is a plain call of the daemon client
:func:`kiss.agents.sorcar.daemon_client.run` (re-exported as the
public API ``kiss.server.sorcar.run``) passing the prompt and the agent file's
path as ``extension_agent_path``: the daemon imports the file as an
agent script
and applies its ``get_X()`` parameter overrides.  For a channel, the
module's ``get_tools()`` returns the channel's tool callables, so the
script serves as its own tools file — the daemon-built agent gets the
channel's authenticated API tools (credentials persisted under
``~/.kiss``) on top of the standard tools (bash, files, browser) — and
the sub-task runs in the channel agents' shared work directory
(``~/.kiss/channel_work``).  For a path-named agent script, whatever
getters the file defines (``get_tools``, ``get_model``,
``get_system_prompt``, ...) configure the session the same way; a
relative path is resolved against the CALLING task's work directory
(captured by :func:`make_run_agent_tool` — the tool runs in the daemon
process, whose own working directory is unrelated to the user's
project), and the sub-task runs in that same directory, so the
dispatched agent operates on the calling project through the standard
task lifecycle (worktree, auto-commit) unless its getters say
otherwise.  Inside the kiss-web daemon the sub-task is submitted back
through the daemon's own UDS socket (recorded at boot by the cron
scheduler); standalone runs use the standard socket resolution
(``KISS_SORCAR_SOCK``, then ``$KISS_HOME/sorcar.sock``) and need a
reachable daemon.
"""

import importlib
import importlib.util
import inspect
import logging
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from kiss.core.config import kiss_home

logger = logging.getLogger(__name__)

WORKSPACE_WAIT_TIMEOUT_SECONDS = 900.0
"""Bound on the wait for a conflicting channel workspace to free up.

Bounds ONLY the pre-dispatch :func:`~kiss.agents.sorcar.channel_workspace.enter_workspace`
wait for a concurrent channel dispatch that holds a DIFFERENT
workspace, and is further capped by the call's ``timeout`` when that
is smaller.  The wait for the dispatched sub-task itself is bounded
separately, by the ``run_agent`` tool's ``timeout`` parameter
(default :data:`DEFAULT_DISPATCH_TIMEOUT_SECONDS`), so a channel call
that hits both waits can take up to twice the ``timeout`` (capped at
``timeout`` + this constant) — plus, on timeout, the client's bounded
stop-confirmation grace (``daemon_client._STOP_CONFIRM_GRACE_SECONDS``,
20 s) — before returning.
"""

DEFAULT_DISPATCH_TIMEOUT_SECONDS = 300.0
"""Default bound on the wait for a dispatched sub-task's result.

Used when the ``run_agent`` tool's ``timeout`` argument is empty; a
per-call value overrides it.  When the wait times out, the tool
returns an error string and the sub-task is STOPPED
(:func:`kiss.agents.sorcar.daemon_client.run` is called with
``stop_on_timeout=True``, which also awaits the stop's
terminal-status confirmation before returning; if a wedged daemon
never confirms it within the bounded grace, the error string says the
task may still be running instead of claiming it was stopped): a
channel sub-task
must not outlive its workspace reservation — the process-global
workspace is released the moment the dispatch returns, so a surviving
sub-task could bind another account's credentials when its channel
tools load — and a surviving path/cron sub-task would keep spending
invisibly.  Work the sub-task completed before the stop (side
effects, spend) is not reported back to the calling task.
"""

_NON_CHANNEL_MODULES = frozenset({"a2a_agent", "openai_compat_agent"})
"""Modules matching ``*_agent.py`` that are not user-facing channels.

``a2a_agent`` (agent-to-agent protocol plumbing) and
``openai_compat_agent`` (an OpenAI-compatible HTTP server) subclass
``BaseChannelAgent`` for infrastructure reasons but are not services a
user asks Sorcar to act on, so they are hidden from the tool.
"""


def _package_dir() -> Path | None:
    """Return the directory of the third-party agents package.

    Located through the import system without importing the package's
    (heavy, optional) modules.

    Returns:
        The package directory, or ``None`` when the package is absent.
    """
    try:
        spec = importlib.util.find_spec("kiss.agents.third_party_agents")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def available_channels() -> list[str]:
    """Return the names of the installed third-party channel agents.

    A channel is any ``<channel>_agent.py`` module in the third-party
    agents package (private ``_``-prefixed helpers and the known
    non-channel infrastructure modules excluded).  The scan reads the
    directory listing only — no channel module is imported.

    Returns:
        Sorted channel names, e.g. ``["discord", ..., "slack", ...]``;
        empty when the package is absent.
    """
    package_dir = _package_dir()
    if package_dir is None:
        return []
    return sorted(
        path.stem[: -len("_agent")]
        for path in package_dir.glob("*_agent.py")
        if not path.name.startswith("_")
        and path.stem not in _NON_CHANNEL_MODULES
    )


def _squash(name: str) -> str:
    """Normalize a channel name for forgiving lookup.

    Case, spaces, hyphens, and underscores are ignored, so
    ``"Home Assistant"`` matches the ``homeassistant`` channel and
    ``"phone control"`` matches ``phone_control``.

    Args:
        name: A user- or model-supplied channel name.

    Returns:
        The lowercase name with separator characters removed.
    """
    return re.sub(r"[\s\-_]+", "", name.strip().lower())


def _agent_class(module: Any) -> type | None:
    """Return the channel agent class defined in *module*.

    The channel-agent contract (see ``BaseChannelAgent``): each channel
    module defines exactly one ``BaseChannelAgent`` subclass of its
    own.  Classes merely imported into the module are ignored.

    Args:
        module: An imported ``<channel>_agent`` module.

    Returns:
        The agent class, or ``None`` when the module defines none.
    """
    # Structural check (a base named ``BaseChannelAgent`` anywhere in
    # the MRO) instead of an ``issubclass`` against the class imported
    # from ``_channel_agent_utils``: the sorcar layer must not import
    # ``kiss.agents.third_party_agents`` (see the layering invariant in
    # ``kiss.tests.agents.sorcar.test_layering_invariants``), and the
    # channel modules are soft plugins reached only through the dynamic
    # per-channel import above — matching the contract by shape keeps
    # the lookup import-free.
    for value in vars(module).values():
        if (
            inspect.isclass(value)
            and value.__module__ == module.__name__
            and any(
                base.__name__ == "BaseChannelAgent"
                for base in inspect.getmro(value)[1:]
            )
        ):
            return value
    return None


def _daemon_sock_path() -> str | None:
    """Return the UDS path of the daemon hosting this process, if any.

    Inside the kiss-web daemon the cron scheduler records the daemon's
    own socket at boot; dispatched sub-tasks must go back through it.
    Standalone (no scheduler running in this process) returns ``None``
    and :func:`kiss.server.sorcar.run` applies its standard socket
    resolution.

    Returns:
        The daemon socket path, or ``None`` when not inside a daemon.
    """
    from kiss.agents.sorcar import cron_agent

    return cron_agent._daemon_sock_path


def _attribute_dispatch_usage(parent_agent: Any, result: Any) -> None:
    """Fold a dispatched sub-task's spend into the calling agent.

    The daemon's terminal ``result`` event carries the sub-task's
    cost, tokens, and steps (parsed into the
    :class:`~kiss.agents.sorcar.daemon_client.TaskResult`).  Without
    this fold, that spend would vanish from the calling task's
    accounting — the parent's end-of-task cost, its live usage
    header, and its persisted per-task cost would all lie low —
    exactly the gap :func:`~kiss.agents.sorcar.sorcar_agent._attribute_sub_usage`
    already closes for ``run_parallel`` sub-agents and ``talk`` TTS
    calls.

    Args:
        parent_agent: The agent that called ``run_agent``; ``None``
            (standalone tools-file use, where no calling agent
            exists) attributes nothing.
        result: The dispatched sub-task's ``TaskResult``.
    """
    if parent_agent is None:
        return
    try:
        from kiss.agents.sorcar.sorcar_agent import _attribute_sub_usage

        _attribute_sub_usage(
            parent_agent,
            float(getattr(result, "cost", 0.0) or 0.0),
            int(getattr(result, "tokens", 0) or 0),
            int(getattr(result, "steps", 0) or 0),
        )
    except Exception:  # pragma: no cover — attribution must never break dispatch
        logger.warning("dispatched sub-task usage attribution failed", exc_info=True)


def _dispatch(
    name: str,
    prompt: str,
    agent_path: str,
    work_dir: str,
    model_name: str,
    budget: float | None,
    timeout: float,
    parent_agent: Any = None,
    scope_work_dir: str = "",
) -> str:
    """Submit an agent-script task to the kiss-web daemon and wait.

    The shared tail of :func:`_run_agent`'s channel and path modes:
    calls :func:`kiss.server.sorcar.run` with *agent_path* as its
    *extension_agent_path* and returns
    the result — or a clean error string — never raising.

    Args:
        name: Display name of the agent for error messages (the
            channel name, or the script's file stem).
        prompt: The full prompt for the sub-task.
        agent_path: Absolute path of the agent script.
        work_dir: Working directory for the sub-task; created when
            absent (an agent script's ``get_work_dir()`` still wins).
        model_name: LLM model for the sub-task; empty for the daemon
            default (an agent script's ``get_model()`` still wins).
        budget: Per-task USD budget override; ``None`` for the daemon
            default.
        timeout: Maximum seconds to wait for the sub-task's result.
            On timeout the sub-task is stopped and an error string is
            returned (see :data:`DEFAULT_DISPATCH_TIMEOUT_SECONDS`).
        parent_agent: The agent calling ``run_agent``, when there is
            one: the sub-task's cost/tokens/steps are folded into its
            task accounting (see :func:`_attribute_dispatch_usage`).
        scope_work_dir: The CALLING task's work directory, used as the
            sub-task's tab workspace-scope so its tab shows in the
            caller's tab bar even though the sub-task executes in
            *work_dir* (a channel/cron scratch directory).  Empty
            (standalone tools-file use) leaves the scope falling back
            to *work_dir*.

    Returns:
        The sub-task's YAML result ("success" and "summary" keys), or
        an error message.
    """
    from kiss.agents.sorcar import daemon_client

    Path(work_dir).mkdir(parents=True, exist_ok=True)
    try:
        result = daemon_client.run(
            prompt,
            extension_agent_path=agent_path,
            work_dir=work_dir,
            scope_work_dir=scope_work_dir,
            model=model_name,
            max_budget=budget,
            timeout=timeout,
            stop_on_timeout=True,
            sock_path=_daemon_sock_path(),
        )
    except daemon_client.StopUnconfirmedTimeoutError:
        return (
            f"Error: the {name} agent task did not finish within "
            f"{timeout:g}s; a stop was requested but the daemon never "
            f"confirmed it, so the task MAY STILL BE RUNNING (and "
            f"spending) on the daemon. Check what it already did "
            f"before retrying with a larger `timeout` argument."
        )
    except TimeoutError:
        return (
            f"Error: the {name} agent task did not finish within "
            f"{timeout:g}s and was stopped; work it completed before "
            f"the stop (side effects, spend) is not reported here. "
            f"Check what it already did before retrying with a larger "
            f"`timeout` argument."
        )
    except Exception as e:
        logger.warning("agent dispatch failed", exc_info=True)
        return f"Error: the {name} agent task could not run: {e}"
    _attribute_dispatch_usage(parent_agent, result)
    summary = result.text or ("" if result.success else "Task failed")
    return str(yaml.safe_dump(
        {"success": result.success, "summary": summary}, sort_keys=False,
    ))


def _run_agent(
    parent_work_dir: str,
    agent: str,
    task: str,
    workspace: str,
    model_name: str,
    max_budget: str,
    timeout: str,
    parent_agent: Any = None,
) -> str:
    """Run a channel agent or an agent script on a task immediately.

    The implementation behind the per-task ``run_agent`` tool built by
    :func:`make_run_agent_tool`, which captures *parent_work_dir*; the
    remaining arguments are the tool's.

    Args:
        parent_work_dir: Work directory of the calling task.  In path
            mode, a relative agent path is resolved against it and the
            sub-task runs in it.  Empty (standalone use, or the module
            loaded directly as a tools file) resolves relative paths
            against the process working directory and runs path-mode
            sub-tasks in ``~/.kiss/agent_work``.
        agent: Channel name or agent-script path (see the tool doc).
        task: The task for the agent.
        workspace: Workspace/account identifier for multi-account
            channels; ignored in path mode.
        model_name: LLM model for the sub-task; empty for the daemon
            default.
        max_budget: Per-task USD budget override as a number string;
            empty for the daemon default.
        timeout: Maximum seconds to wait for the sub-task's result, as
            a number string; empty for the default
            :data:`DEFAULT_DISPATCH_TIMEOUT_SECONDS` (300).  On
            timeout the sub-task is stopped and an error string is
            returned.  It also caps the channel-mode workspace wait
            (see :data:`WORKSPACE_WAIT_TIMEOUT_SECONDS`).
        parent_agent: The agent calling ``run_agent``, when there is
            one; the sub-task's spend is folded into its task
            accounting (see :func:`_attribute_dispatch_usage`).

    Returns:
        The sub-task's YAML result ("success" and "summary" keys), or
        an error message.
    """
    if not task.strip():
        return "Error: task must be a non-empty string."
    try:
        budget = float(max_budget) if max_budget.strip() else None
    except ValueError:
        return f"Error: max_budget must be a number, got {max_budget!r}."
    if budget is not None and (not math.isfinite(budget) or budget <= 0):
        return (
            f"Error: max_budget must be a positive finite number, "
            f"got {max_budget!r}."
        )
    try:
        wait = (
            float(timeout) if timeout.strip()
            else DEFAULT_DISPATCH_TIMEOUT_SECONDS
        )
    except ValueError:
        return f"Error: timeout must be a number of seconds, got {timeout!r}."
    if not math.isfinite(wait) or wait <= 0:
        return (
            f"Error: timeout must be a positive finite number of seconds, "
            f"got {timeout!r}."
        )
    requested = agent.strip()
    if requested.endswith(".py") or "/" in requested or "\\" in requested:
        # Path mode: any agent-script file.  The task is passed through
        # unchanged — no channel preamble or workspace handling; the
        # script itself configures the session via its getters.
        from kiss.agents.sorcar.daemon_client import resolve_agent_path

        candidate = Path(requested).expanduser()
        if not candidate.is_absolute() and parent_work_dir:
            candidate = Path(parent_work_dir) / candidate
        try:
            agent_path = resolve_agent_path(str(candidate))
        except ValueError as e:
            return f"Error: {e}"
        work_dir = parent_work_dir or str(kiss_home() / "agent_work")
        return _dispatch(Path(agent_path).stem, task, agent_path,
                         work_dir, model_name, budget, wait, parent_agent,
                         scope_work_dir=parent_work_dir)
    # Forgiving lookup: "Home Assistant", "phone control", and
    # "nextcloud-talk" all resolve — spelling variants differ only in
    # case, spaces, hyphens, and underscores.
    squashed = _squash(requested)
    if squashed == "cron":
        # The scheduled-automations agent: an agent script in the
        # sorcar package (not a third-party channel), dispatched the
        # same way — its get_tools() supplies the cron_job tool and
        # its get_work_dir()/get_use_worktree()/get_auto_commit()
        # getters keep the session in ~/.kiss/cron/work, out of the
        # calling project's git lifecycle.
        from kiss.agents.sorcar import cron_agent

        return _dispatch(
            "cron", cron_agent.CRON_DISPATCH_PREAMBLE + task,
            str(cron_agent.__file__), cron_agent.get_work_dir(),
            model_name, budget, wait, parent_agent,
            scope_work_dir=parent_work_dir,
        )
    channels = available_channels()
    matches = [name for name in channels if _squash(name) == squashed]
    if not matches:
        return (
            f"Error: unknown agent {agent!r} — not the built-in cron "
            f"agent, not an installed channel, and not a path to a .py "
            f"agent script. Available channels: "
            f"{', '.join(channels) or 'none installed'}."
        )
    channel = matches[0]
    try:
        module = importlib.import_module(
            f"kiss.agents.third_party_agents.{channel}_agent"
        )
    except Exception as e:
        logger.warning("channel module import failed", exc_info=True)
        return f"Error: the {channel} agent module failed to import: {e}"
    agent_cls = _agent_class(module)
    if agent_cls is None or not module.__file__:  # pragma: no cover — contract violation only
        return f"Error: {channel!r} defines no channel agent class."
    from kiss.agents.sorcar.channel_workspace import (
        enter_workspace,
        exit_workspace,
    )

    preamble = (
        f"You are the {channel} channel agent: this session already has "
        f"the authenticated {channel} API tools — use them directly and "
        "immediately, without exploring any source code.  Never call "
        "run_agent here: it would just recurse into another session "
        "like this one.\n\n"
    )
    workspace = workspace.strip() or "default"
    guidance = str(getattr(agent_cls, "channel_system_prompt", "")).strip()
    prompt = preamble + task + (f"\n\n{guidance}" if guidance else "")
    # The channel agents' shared work directory — the same default
    # their poll-mode runner uses — so dispatched channel sessions keep
    # seeing the files of earlier channel sessions.
    work_dir = str(kiss_home() / "channel_work")
    # The workspace env var is process-global and managed by the
    # shared reference-counting registry (used by the channel CLIs'
    # launcher too), not by save/restore snapshots: snapshots taken by
    # overlapping dispatches would restore each other's values out of
    # order and leave a stale workspace exported.  A dispatch whose
    # workspace DIFFERS from a running one's blocks here instead of
    # overwriting the exported value mid-flight (which would hand the
    # running session the wrong account's credentials); the wait is
    # bounded — by the call's own timeout, and by
    # WORKSPACE_WAIT_TIMEOUT_SECONDS for very large timeouts — so a
    # conflicting dispatch cannot hang this one forever.
    workspace_wait = min(wait, WORKSPACE_WAIT_TIMEOUT_SECONDS)
    if not enter_workspace(workspace, timeout=workspace_wait):
        return (
            f"Error: workspace {workspace!r} could not be activated for "
            f"the {channel} agent within {workspace_wait:g}s "
            f"because a concurrent channel task is still using a "
            f"different workspace; retry when it finishes."
        )
    try:
        return _dispatch(channel, prompt, str(module.__file__),
                         work_dir, model_name, budget, wait, parent_agent,
                         scope_work_dir=parent_work_dir)
    finally:
        exit_workspace(workspace)


def make_run_agent_tool(
    work_dir: str, parent_agent: Any = None,
) -> Callable[..., str]:
    """Build the ``run_agent`` tool for the agent running in *work_dir*.

    The tool executes in the daemon process, whose own working
    directory is unrelated to the user's project, so the calling
    task's work directory must be captured here (exactly like
    ``make_skill_tool``): it anchors relative agent-script paths and
    is the work directory the dispatched path-mode sub-task runs in.

    Args:
        work_dir: The calling task's work directory.  Empty applies
            the standalone defaults (see :func:`_run_agent`).
        parent_agent: The agent this tool is built for, when there is
            one.  Each dispatched sub-task's cost/tokens/steps are
            folded into its task accounting so the calling task's
            end-of-task cost includes ``run_agent`` spend.  ``None``
            (standalone tools-file use) disables attribution.

    Returns:
        The ``run_agent`` tool callable.
    """

    def run_agent(
        agent: str,
        task: str,
        workspace: str = "default",
        model_name: str = "",
        max_budget: str = "",
        timeout: str = "",
    ) -> str:
        """Run an agent — a channel agent or any agent script — on a task now.

        DO NOT CREATE AN AGENT JUST TO RUN THE `run_agent` tool.  Use
        `run_parallel` tool instead.
        Use this tool RIGHT AWAY — as the first action, without
        exploring any source code — whenever the task is to act on an
        external messaging service, mailbox, or device channel:
        sending or reading messages, posting, authenticating a
        channel, managing chats, and so on.  Pass the user's request
        through as the task; the channel agent has its own
        authenticated API tools and resolves channel or user names
        itself.  Use it the same way for scheduled automations (cron
        jobs) — creating, listing, removing, pausing, resuming, or
        immediately running a scheduled job: pass ``"cron"`` as the
        agent and the scheduling request as the task (the cron agent
        translates natural-language schedules itself).  Also use it
        whenever the user names an agent file (an *agent script*) to
        run a task with: pass the file's path as the agent.

        Available channels: {channels}.  The built-in ``"cron"``
        agent (scheduled automations) is always available.

        The task runs as a fresh session on the kiss-web daemon — the
        agent file's path is passed as the ``extension_agent_path`` of
        :func:`kiss.server.sorcar.run`, so the session is configured
        by the file's ``get_X()`` getters (a channel module's
        ``get_tools()`` supplies that channel's authenticated tools,
        credentials persisted under ``~/.kiss``) on top of the
        standard tools.  A path-named agent's session runs in THIS
        task's work directory (so it operates on the same project,
        with the standard worktree/auto-commit lifecycle) unless the
        script's ``get_work_dir()`` says otherwise; a channel agent's
        session runs in the channels' shared ``~/.kiss/channel_work``.
        This call blocks until the task finishes or the ``timeout``
        (default 300 seconds) expires, whichever comes first; a
        timed-out call spends up to 20 further seconds confirming the
        stop, and a channel dispatch queued behind a concurrent
        channel task holding a different workspace may additionally
        wait up to ``timeout`` (capped at 900 s) for that workspace
        before the sub-task starts.

        Args:
            agent: WHICH agent to run — an installed channel name,
                e.g. ``"slack"``, ``"telegram"``, ``"discord"``,
                ``"email"``, ``"whatsapp"`` (case, spaces, hyphens,
                and underscores are ignored: "Home Assistant" resolves
                to ``homeassistant``); or ``"cron"`` for the
                scheduled-automations agent; or the path of a Python
                agent-script file, e.g. ``"agents/researcher.py"``
                (recognized by its ``.py`` suffix or a path separator;
                must exist; a relative path is resolved against this
                task's work directory).
            task: The task for the agent, e.g. "Send 'hello' to the
                #sorcar channel".  A path-named agent script's
                ``get_prompt()``, if defined, replaces it.
            workspace: Workspace/account identifier for multi-account
                channels (default ``"default"``).  Ignored for
                path-named agent scripts.
            model_name: LLM model for the sub-task; empty uses the
                daemon default.  An agent script's ``get_model()``
                still wins.
            max_budget: Per-task USD budget override as a number
                string; empty uses the daemon default.
            timeout: Maximum seconds to wait for the task to finish,
                as a number string; empty uses the default of 300
                seconds.  Pass a larger value for tasks expected to
                run long.  On timeout this call STOPS the task and
                returns an error string (which says the task may
                still be running in the rare case the daemon never
                confirms the stop); work the task completed before
                the stop (side effects, spend) is not reported back
                here, so check what it already did before retrying
                with a larger timeout.

        Returns:
            The sub-task's YAML result ("success" and "summary" keys),
            an error message naming the available channels, or a
            timeout error message when the task outlived ``timeout``.
        """
        return _run_agent(
            work_dir, agent, task, workspace, model_name, max_budget,
            timeout, parent_agent,
        )

    run_agent.__doc__ = (run_agent.__doc__ or "").replace(
        "{channels}", ", ".join(available_channels()) or "none installed"
    )
    return run_agent


def get_tools() -> list:
    """Return the dispatch tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument.  No calling-task work directory
    exists in that setting, so the tool applies the standalone
    defaults (see :func:`_run_agent`).

    Returns:
        The list containing the ``run_agent`` tool.
    """
    return [make_run_agent_tool("")]
