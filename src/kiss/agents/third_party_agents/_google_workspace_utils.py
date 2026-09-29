# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared Composio sign-in tools and prompt for the Google Workspace agents.

Every Google agent (Gmail, Calendar, Drive, Docs, Sheets, Chat) signs in
the same way: ``authenticate_<service>()`` hands the user a Composio
Connect Link, the user signs in to Google and clicks Allow in their own
browser, and ``finish_<service>_auth()`` records the connection.  The
Google token stays at Composio (see :mod:`._composio_google`).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from kiss.agents.third_party_agents._composio_google import (
    clear_connection,
    composio_api_key,
    finish_connect,
    save_api_key,
    start_connect,
)


def google_auth_prompt(service: str, label: str) -> str:
    """Build the Authentication section of a Google agent's system prompt.

    Args:
        service: KISS service name (used in the tool names).
        label: Human-readable service label.

    Returns:
        The prompt section: check, authenticate, the user opens the
        Connect Link in their own browser, finish.
    """
    return (
        f"\n\n## {label} Authentication\n"
        f"{label} access is brokered by Composio. Always call "
        f"check_{service}_auth() first; if it returns ok, report that {label} "
        "is connected and stop. Otherwise call "
        f"authenticate_{service}() (pass api_key='...' only when it reports "
        "that no Composio API key is configured; the user creates one at "
        "https://dashboard.composio.dev under Settings > API Keys). It returns "
        "a Composio Connect Link: give that exact URL to the user with "
        "ask_user_question() to open in their OWN browser, sign in to Google "
        "and click Allow. Never open the link in your built-in browser and "
        "never ask for the user's Google password or 2FA code. Then call "
        f"finish_{service}_auth(); if it returns 'pending', wait a few seconds "
        "and call it again."
    )


def make_google_auth_tools(
    agent: Any,
    service: str,
    label: str,
    on_connected: Callable[[], bool],
    on_cleared: Callable[[], None] | None = None,
) -> list:
    """Build the Composio sign-in tool set for a Google Workspace agent.

    Args:
        agent: The channel agent; its ``_is_authenticated()`` reports
            whether *service* has a Composio connection.
        service: KISS service name (e.g. ``"google_calendar"``), also
            used in the generated tool names.
        label: Human-readable service label (e.g. ``"Google Calendar"``).
        on_connected: Called without arguments after a connection is
            completed to wire the agent's backend (its ``connect()``);
            a False result turns the finish answer into an error.
        on_cleared: Called after the Composio connection is cleared so
            the agent can drop other credentials it holds. Optional.

    Returns:
        ``check_<service>_auth``, ``authenticate_<service>``,
        ``clear_<service>_auth`` and ``finish_<service>_auth``.
    """

    def check_auth() -> str:
        if agent._is_authenticated():
            return json.dumps({"ok": True, "message": f"{label} is connected."})
        missing_key = "" if composio_api_key() else (
            " No Composio API key is configured: ask the user for a project "
            "API key (ak_...) from "
            "https://dashboard.composio.dev/~/project/settings/api-keys "
            "(Settings → Project Settings → API Keys) and pass it "
            f"as authenticate_{service}(api_key='...'). Note: old consumer "
            "keys (ck_...) no longer work."
        )
        return (
            f"Not authenticated with {label}. Call authenticate_{service}() to get "
            "a Composio Connect Link for the user to open in their own browser, "
            f"then finish_{service}_auth().{missing_key}"
        )

    def authenticate(api_key: str = "") -> str:
        if api_key.strip():
            save_api_key(api_key)
        return json.dumps(start_connect(service, label))

    def finish_auth() -> str:
        result = finish_connect(service, label)
        if result.get("ok") and not on_connected():
            info = str(getattr(getattr(agent, "_backend", None), "_connection_info", "") or "")
            result = {
                "ok": False,
                "error": f"{label} is connected at Composio but its first API call failed"
                + (f": {info}" if info else "."),
            }
        return json.dumps(result)

    def clear_auth() -> str:
        clear_connection(service)
        if on_cleared is not None:
            on_cleared()
        return f"{label} connection cleared."

    check_auth.__name__ = f"check_{service}_auth"
    check_auth.__doc__ = (
        f"Check whether {label} is connected through Composio.\n\n"
        "Returns:\n"
        "    Connection status, or instructions for how to connect."
    )
    authenticate.__name__ = f"authenticate_{service}"
    authenticate.__doc__ = (
        f"Start connecting {label} through a Composio Connect Link.\n\n"
        "The user opens the returned link in their own browser, signs in to\n"
        f"Google and clicks Allow; then call finish_{service}_auth().\n\n"
        "Args:\n"
        "    api_key: Composio project API key (ak_...) to save first, shared\n"
        "        by all Google agents. Optional; only needed when none is\n"
        "        configured.\n\n"
        "Returns:\n"
        "    JSON with status 'consent_required', the verification_uri and\n"
        "    instructions, or {\"ok\": false, \"error\": ...}."
    )
    finish_auth.__name__ = f"finish_{service}_auth"
    finish_auth.__doc__ = (
        f"Complete the {label} connection started by authenticate_{service}().\n\n"
        "Returns:\n"
        "    JSON with ok true once connected, status 'pending' while the\n"
        "    user has not approved yet, or an error."
    )
    clear_auth.__name__ = f"clear_{service}_auth"
    clear_auth.__doc__ = (
        f"Disconnect {label}: forget the Composio connection and delete it.\n\n"
        "Returns:\n"
        "    Status message."
    )
    return [check_auth, authenticate, clear_auth, finish_auth]
