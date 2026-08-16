# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Immediate dispatch of channel tasks to the third-party channel agents.

Gives the Sorcar agent a single tool, :func:`run_channel_agent`, that
runs any installed third-party channel agent (Slack, Telegram,
Discord, email, WhatsApp, Home Assistant, ...) on a task right away —
so a request like "Send 'hello' to the #sorcar Slack channel" is
executed in one tool call instead of the agent first rediscovering
what the channel agents are and how they work.

The channel agents are looked up dynamically, the same soft-plugin
style the cron deliverer uses: any module named
``kiss.agents.third_party_agents.<channel>_agent`` that defines a
``BaseChannelAgent`` subclass is dispatchable.  This module never
imports from ``kiss.agents.third_party_agents`` at module scope, so it
works (with an empty channel list) when those optional modules are
absent.

Each dispatch is a plain call of the public API
:func:`kiss.server.sorcar.run` passing the prompt and the channel
agent module's file path as ``agent_path``: the daemon imports the
file as an agent script, and because the module's ``get_tools()``
returns the channel's tool callables, the script serves as its own
tools file — the daemon-built agent gets the channel's authenticated
API tools (credentials persisted under ``~/.kiss``) on top of the
standard tools (bash, files, browser).  Inside the kiss-web daemon the
sub-task is submitted back through the daemon's own UDS socket
(recorded at boot by the cron scheduler); standalone runs use the
standard socket resolution (``KISS_SORCAR_SOCK``, then
``$KISS_HOME/sorcar.sock``) and need a reachable daemon.
"""

import importlib
import importlib.util
import inspect
import logging
import math
import re
from pathlib import Path
from typing import Any

import yaml

from kiss.core.config import kiss_home

logger = logging.getLogger(__name__)

DISPATCH_TIMEOUT_SECONDS = 900.0

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
    from kiss.agents.third_party_agents._channel_agent_utils import BaseChannelAgent

    for value in vars(module).values():
        if (
            inspect.isclass(value)
            and issubclass(value, BaseChannelAgent)
            and value is not BaseChannelAgent
            and value.__module__ == module.__name__
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


def run_channel_agent(
    channel: str,
    task: str,
    workspace: str = "default",
    model_name: str = "",
    max_budget: str = "",
) -> str:
    """Run a third-party channel agent on a task immediately.

    Use this tool RIGHT AWAY — as the first action, without exploring
    any source code — whenever the task is to act on an external
    messaging service, mailbox, or device channel: sending or reading
    messages, posting, authenticating a channel, managing chats, and so
    on.  Pass the user's request through as the task; the channel
    agent has its own authenticated API tools and resolves channel or
    user names itself.

    Available channels: {channels}.

    The task runs as a fresh session on the kiss-web daemon — the
    channel agent module's path is passed as the ``agent_path`` of
    :func:`kiss.server.sorcar.run`, so the session gets that channel's
    authenticated tools (credentials persisted under ``~/.kiss``) plus
    the standard tools.  This call blocks until the task finishes (up
    to 15 minutes).

    Args:
        channel: The channel name, e.g. ``"slack"``, ``"telegram"``,
            ``"discord"``, ``"email"``, ``"whatsapp"``.  Case, spaces,
            hyphens, and underscores are ignored ("Home Assistant"
            resolves to ``homeassistant``).
        task: The task for the channel agent, e.g. "Send 'hello' to
            the #sorcar channel".
        workspace: Workspace/account identifier for multi-account
            channels (default ``"default"``).
        model_name: LLM model for the sub-task; empty uses the daemon
            default.
        max_budget: Per-task USD budget override as a number string;
            empty uses the daemon default.

    Returns:
        The sub-task's YAML result ("success" and "summary" keys), or
        an error message naming the available channels.
    """
    channels = available_channels()
    # Forgiving lookup: "Home Assistant", "phone control", and
    # "nextcloud-talk" all resolve — spelling variants differ only in
    # case, spaces, hyphens, and underscores.
    requested = _squash(channel)
    matches = [name for name in channels if _squash(name) == requested]
    if not matches:
        return (
            f"Error: unknown channel {channel!r}. "
            f"Available channels: {', '.join(channels) or 'none installed'}."
        )
    channel = matches[0]
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
        module = importlib.import_module(
            f"kiss.agents.third_party_agents.{channel}_agent"
        )
    except Exception as e:
        logger.warning("channel module import failed", exc_info=True)
        return f"Error: the {channel} agent module failed to import: {e}"
    agent_cls = _agent_class(module)
    if agent_cls is None or not module.__file__:  # pragma: no cover — contract violation only
        return f"Error: {channel!r} defines no channel agent class."
    from kiss.agents.third_party_agents._kiss_web_launcher import (
        _enter_workspace,
        _exit_workspace,
    )
    from kiss.server import sorcar

    preamble = (
        f"You are the {channel} channel agent: this session already has "
        f"the authenticated {channel} API tools — use them directly and "
        "immediately, without exploring any source code.  Never call "
        "run_channel_agent here: it would just recurse into another "
        "session like this one.\n\n"
    )
    work_dir = kiss_home() / "channel_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    workspace = workspace.strip() or "default"
    guidance = str(getattr(agent_cls, "channel_system_prompt", "")).strip()
    prompt = preamble + task + (f"\n\n{guidance}" if guidance else "")
    # The workspace env var is process-global and managed by the
    # launcher's reference-counting registry (shared with the channel
    # CLIs), not by save/restore snapshots: snapshots taken by
    # overlapping dispatches would restore each other's values out of
    # order and leave a stale workspace exported.
    _enter_workspace(workspace)
    try:
        result = sorcar.run(
            prompt,
            agent_path=str(module.__file__),
            work_dir=str(work_dir),
            model=model_name,
            max_budget=budget,
            timeout=DISPATCH_TIMEOUT_SECONDS,
            sock_path=_daemon_sock_path(),
        )
    except TimeoutError:
        return (
            f"The {channel} agent task did not finish within "
            f"{DISPATCH_TIMEOUT_SECONDS:.0f}s; it keeps running on the daemon."
        )
    except Exception as e:
        logger.warning("channel agent dispatch failed", exc_info=True)
        return f"Error: the {channel} agent task could not run: {e}"
    finally:
        _exit_workspace(workspace)
    summary = result.text or ("" if result.success else "Task failed")
    return str(yaml.safe_dump(
        {"success": result.success, "summary": summary}, sort_keys=False,
    ))


run_channel_agent.__doc__ = (run_channel_agent.__doc__ or "").replace(
    "{channels}", ", ".join(available_channels()) or "none installed"
)


def get_tools() -> list:
    """Return the channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument.

    Returns:
        The list containing the :func:`run_channel_agent` tool.
    """
    return [run_channel_agent]
