# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Public client IDs of the KISS-owned OAuth apps.

KISS ships one registered OAuth app per provider so a user connects by
signing in and clicking Allow, without registering an app of their own.
Every app is a PUBLIC client: sign-in uses PKCE or the device flow, so
no client secret exists and the IDs below are safe to publish.

Registration settings per app (maintainers, in the provider portal):

* ``github`` — OAuth app, "Enable Device Flow" ticked.
* ``msteams`` — Microsoft Entra app registration, "Accounts in any
  organizational directory", "Allow public client flows" = Yes,
  delegated Microsoft Graph permissions ``ChannelMessage.Send``,
  ``ChannelMessage.Read.All``, ``Chat.ReadWrite``, ``Team.ReadBasic.All``,
  ``Channel.ReadBasic.All``, ``User.Read``, ``offline_access``.
* ``slack`` — Slack app with PKCE enabled
  (``oauth_config.pkce_enabled: true``) and redirect URL
  ``http://localhost:53682/callback``; user token scopes only.
* ``discord`` — Discord application with "Public Client" on and
  redirect ``http://localhost:53682/callback``.

An environment variable ``KISS_<PROVIDER>_CLIENT_ID`` overrides the
embedded ID (a fork, a self-registered app, or a test server).
"""

from __future__ import annotations

import os

# Fixed loopback redirect registered with every PKCE app above.  Slack
# and Discord match redirect URIs exactly, so the port cannot vary.
LOOPBACK_PORT = 53682
LOOPBACK_REDIRECT_URI = f"http://localhost:{LOOPBACK_PORT}/callback"

# Embedded public client IDs.  An empty value means the KISS app for
# that provider is not registered yet; the env override still works.
KISS_OAUTH_CLIENT_IDS: dict[str, str] = {
    "github": "",
    "msteams": "",
    "slack": "",
    "discord": "",
}


def oauth_client_id(provider: str) -> str:
    """Return the public OAuth client ID KISS uses for *provider*.

    Args:
        provider: Provider key (``github``, ``msteams``, ``slack`` or
            ``discord``).

    Returns:
        ``$KISS_<PROVIDER>_CLIENT_ID`` when set, else the embedded ID
        (``""`` when none is registered).
    """
    override = os.environ.get(f"KISS_{provider.upper()}_CLIENT_ID", "").strip()
    return override or KISS_OAUTH_CLIENT_IDS.get(provider, "")


def missing_client_id_error(provider: str, label: str) -> str:
    """Build the error shown when no client ID is available for *provider*.

    Args:
        provider: Provider key.
        label: Human-readable provider name.

    Returns:
        A one-sentence error naming the environment variable to set.
    """
    return (
        f"No KISS OAuth app client ID is configured for {label}. Set "
        f"KISS_{provider.upper()}_CLIENT_ID to the public client ID of a "
        f"{label} OAuth app registered as described in "
        "kiss/agents/third_party_agents/_oauth_apps.py."
    )
