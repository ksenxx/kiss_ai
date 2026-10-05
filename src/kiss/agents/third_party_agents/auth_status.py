# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Report the authentication status of every third-party channel agent.

The right sidebar's "Apps" panel lists each installed channel agent
with a connected / not-connected badge.  Deciding that means importing
the channel module, building its agent (which loads the credentials
persisted under ``$KISS_HOME/third_party_agents/``) and asking
``_is_authenticated()`` -- heavy imports and client construction that
must not run inside the long-lived kiss-web daemon.  The daemon
therefore runs this module as a short-lived subprocess::

    python -m kiss.agents.third_party_agents.auth_status

which prints one JSON array on stdout, one object per channel::

    {"name": "slack", "label": "Slack", "authenticated": true, "error": ""}

Channels are probed concurrently, each with a deadline: a channel whose
constructor hangs (a network call, a stuck bridge) is reported with
``authenticated: null`` instead of delaying every other row.

A status check must not change anything.  In Muse-auth mode (the
default, see :mod:`kiss.agents.third_party_agents.muse_auth`) building
an agent mints a vault surrogate -- an entry the auth daemon keeps --
and migrates legacy plaintext credentials into the vault.  So the probe
asks the vault once which services are enrolled, then builds the agents
with Muse-auth switched off (``KISS_MUSE_AUTH=0``, which only reads the
legacy credential files); a channel counts as connected when either
source has its credential.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any

from kiss.agents.sorcar.agent_dispatch import available_channels
from kiss.agents.third_party_agents._channel_agent_utils import BaseChannelAgent
from kiss.agents.third_party_agents._composio_google import TOOLKITS

# Per-run deadline for the whole probe; channels still running when it
# expires are reported as unknown.
PROBE_TIMEOUT_SECONDS = 20.0


def _agent_class(module: Any) -> type[BaseChannelAgent] | None:
    """Return the channel agent class *module* defines, or ``None``.

    The channel-agent contract: each channel module defines exactly
    one :class:`BaseChannelAgent` subclass of its own.  Classes merely
    imported into the module are ignored.

    Args:
        module: An imported ``<channel>.<channel>_sea`` module.
    """
    for value in vars(module).values():
        if (
            inspect.isclass(value)
            and value.__module__ == module.__name__
            and issubclass(value, BaseChannelAgent)
            and value is not BaseChannelAgent
        ):
            return value
    return None
_MAX_WORKERS = 8

# Muse vault service names that differ from the channel name and are
# not declared as the module's ``_SERVICE`` (the Google Workspace
# channels declare theirs, e.g. ``gcal`` -> ``google_calendar``).
_MUSE_SERVICES = {"brave": "brave_search"}

# Brand spellings that splitting the class name at case changes gets
# wrong (``GitHubAgent`` -> "Git Hub").
_BRAND_LABELS = {
    "bluebubbles": "BlueBubbles",
    "dingtalk": "DingTalk",
    "github": "GitHub",
    "imessage": "iMessage",
    "line": "LINE",
    "msteams": "Microsoft Teams",
    "ntfy": "ntfy",
    "simplex": "SimpleX Chat",
    "wecom": "WeCom",
    "whatsapp": "WhatsApp",
}


def channel_label(name: str, agent_cls: type | None = None) -> str:
    """Return a human-readable name for a channel.

    Known brand spellings win; otherwise the label is derived from the
    agent class name (``HomeAssistantAgent`` -> ``"Home Assistant"``),
    falling back to the capitalized module name.

    Args:
        name: The channel name (``<name>/<name>_sea.py``).
        agent_cls: The channel's agent class, when it could be loaded.

    Returns:
        The display label.
    """
    if name in _BRAND_LABELS:
        return _BRAND_LABELS[name]
    base = agent_cls.__name__ if agent_cls is not None else ""
    if base.endswith("Agent"):
        base = base[: -len("Agent")]
    if not base:
        return name.capitalize()
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", base)


def vault_services() -> set[str] | None:
    """Return the services enrolled in the Muse-auth vault.

    Returns:
        The enrolled service names, or ``None`` when Muse-auth is off or
        its daemon cannot answer.
    """
    from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled
    from kiss.agents.third_party_agents.muse_auth.client import enrolled_services

    if not muse_auth_enabled():
        return None
    try:
        return set(enrolled_services())
    except Exception:  # noqa: BLE001 - no vault answer: fall back to the files
        return None


def channel_status(name: str, enrolled: set[str] | None = None) -> dict[str, Any]:
    """Probe one channel agent's authentication state.

    Args:
        name: The channel name, e.g. ``"slack"``.
        enrolled: Services enrolled in the Muse-auth vault (see
            :func:`vault_services`); a channel whose service is in it is
            connected whatever its credential files say.

    Returns:
        ``{"name", "label", "authenticated", "error"}`` where
        ``authenticated`` is ``True``/``False``, or ``None`` when the
        agent could not be built (``error`` then says why).
    """
    agent_cls: type | None = None
    try:
        module = importlib.import_module(f"kiss.agents.third_party_agents.{name}.{name}_sea")
        agent_cls = _agent_class(module)
        if agent_cls is None:
            raise RuntimeError("module defines no channel agent")
        service = getattr(module, "_SERVICE", None) or _MUSE_SERVICES.get(name, name)
        # Google services left the vault for Composio: an old enrollment
        # there says nothing about the current connection.
        in_vault = enrolled is not None and service in enrolled and service not in TOOLKITS
        authenticated: bool | None = in_vault or bool(agent_cls()._is_authenticated())
        error = ""
    except Exception as exc:  # noqa: BLE001 - one broken channel must not hide the rest
        authenticated = None
        error = f"{type(exc).__name__}: {exc}"[:300]
    return {
        "name": name,
        "label": channel_label(name, agent_cls),
        "authenticated": authenticated,
        "error": error,
    }


def all_channel_statuses(
    timeout: float = PROBE_TIMEOUT_SECONDS, enrolled: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Probe every installed channel agent concurrently.

    Args:
        timeout: Seconds to wait for all probes; channels still running
            afterwards are reported with ``authenticated: None``.
        enrolled: Services enrolled in the Muse-auth vault, passed to
            :func:`channel_status`.

    Returns:
        One status dict per channel (see :func:`channel_status`), in
        channel-name order.
    """
    names = available_channels()
    pool = ThreadPoolExecutor(max_workers=_MAX_WORKERS)
    futures = {name: pool.submit(channel_status, name, enrolled) for name in names}
    wait(futures.values(), timeout=timeout)
    pool.shutdown(wait=False, cancel_futures=True)
    results = []
    for name, future in futures.items():
        if future.done() and not future.cancelled():
            results.append(future.result())
        else:
            results.append({
                "name": name,
                "label": channel_label(name),
                "authenticated": None,
                "error": "status check timed out",
            })
    return results


def main() -> None:
    """Print every channel's status as one JSON array and exit.

    Reads the vault's enrollments first, then switches Muse-auth off
    for this process so building the agents mints and migrates nothing.
    Exits through ``os._exit`` so a probe thread stuck in a hung
    constructor cannot keep the process alive after the answer is out.
    """
    enrolled = vault_services()
    os.environ["KISS_MUSE_AUTH"] = "0"
    statuses = all_channel_statuses(enrolled=enrolled)
    sys.stdout.write(json.dumps(statuses) + "\n")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
