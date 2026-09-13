# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared OAuth2 credential helpers for Google Workspace channel agents.

The Google Workspace adapters (Calendar, Drive, Docs, Sheets) all
authenticate the same way Gmail does: an OAuth2 installed-app flow
whose client secret lives in a ``credentials.json`` (downloaded from
Google Cloud Console) and whose resulting user token is persisted as
``token.json`` under the service's directory in
``$KISS_HOME/third_party_agents/<service>/``.

This module centralizes that flow so each adapter only declares its
service name and scopes.  A service without its own
``credentials.json`` falls back to the shared
``$KISS_HOME/third_party_agents/google/credentials.json`` and then to
Gmail's ``$KISS_HOME/third_party_agents/gmail/credentials.json``, so
one Google Cloud OAuth client can serve every Google adapter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from kiss.agents.third_party_agents._backend_utils import is_headless_environment
from kiss.agents.third_party_agents._channel_agent_utils import write_private_file
from kiss.core.config import kiss_home


def google_service_dir(service: str) -> Path:
    """Return a Google service's credential directory, honoring ``KISS_HOME``.

    Args:
        service: Service directory name (e.g. ``"google_calendar"``).

    Returns:
        Path to ``$KISS_HOME/third_party_agents/<service>``.
    """
    return kiss_home() / "third_party_agents" / service


def token_path(service: str) -> Path:
    """Return the path of a service's stored OAuth2 token file.

    Args:
        service: Service directory name (e.g. ``"google_drive"``).

    Returns:
        Path to ``token.json`` inside :func:`google_service_dir`.
    """
    return google_service_dir(service) / "token.json"


def credentials_path(service: str) -> Path:
    """Return the OAuth2 client-secret file used for a service's flow.

    Looks for ``credentials.json`` in the service's own directory
    first, then in the shared ``google`` directory, then in Gmail's
    directory.  When none exists, the service's own (not yet created)
    path is returned so callers can name the expected location in
    error messages.

    Args:
        service: Service directory name (e.g. ``"google_docs"``).

    Returns:
        Path of the first existing ``credentials.json`` candidate, or
        the service's own candidate when none exists.
    """
    own = google_service_dir(service) / "credentials.json"
    for candidate in (own, google_service_dir("google") / "credentials.json",
                      google_service_dir("gmail") / "credentials.json"):
        if candidate.exists():
            return candidate
    return own


def load_google_credentials(service: str, scopes: list[str]) -> Credentials | None:
    """Load a service's stored OAuth2 credentials from disk.

    Expired credentials with a refresh token are refreshed and
    re-persisted transparently.

    Args:
        service: Service directory name (e.g. ``"google_sheets"``).
        scopes: OAuth scopes the credentials must carry.

    Returns:
        Valid :class:`Credentials`, or ``None`` when missing, invalid,
        or unrefreshable.
    """
    path = token_path(service)
    if not path.exists():
        return None
    try:
        creds: Credentials = Credentials.from_authorized_user_file(str(path), scopes)
    except Exception:
        # A loader must never raise: token.json may hold malformed JSON,
        # valid JSON of the wrong shape ([] / null / a bare string, which
        # crashes from_authorized_user_file with AttributeError), or be
        # unreadable — all mean "no usable credentials".
        return None
    if creds.valid:
        return creds
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            save_google_credentials(service, creds)
            return creds
        except Exception:
            return None
    return None


def save_google_credentials(service: str, creds: Credentials) -> None:
    """Persist a service's OAuth2 credentials atomically with 0600 permissions.

    Args:
        service: Service directory name.
        creds: Google OAuth2 credentials to persist.
    """
    write_private_file(token_path(service), creds.to_json())


def clear_google_credentials(service: str) -> None:
    """Delete a service's stored OAuth2 token, if any.

    Args:
        service: Service directory name.
    """
    path = token_path(service)
    if path.exists():
        path.unlink()


def run_google_oauth_flow(service: str, scopes: list[str]) -> Credentials | None:
    """Run the OAuth2 installed-app flow and persist the new token.

    In headless environments the local-server flow runs with
    ``open_browser=False`` so the auth URL is printed for manual
    visiting instead of a browser window being opened.

    Args:
        service: Service directory name the token is stored under.
        scopes: OAuth scopes to request.

    Returns:
        New :class:`Credentials`, or ``None`` when no
        ``credentials.json`` is available (see :func:`credentials_path`).
    """
    creds_path = credentials_path(service)
    if not creds_path.exists():
        return None
    flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), scopes)
    if is_headless_environment():
        creds = cast(Credentials, flow.run_local_server(port=0, open_browser=False))
    else:
        creds = cast(Credentials, flow.run_local_server(port=0))
    save_google_credentials(service, creds)
    return creds


def fresh_access_token(creds: Any) -> str:
    """Return a currently valid access token from *creds*, refreshing if needed.

    Args:
        creds: Google OAuth2 credentials, or ``None``.

    Returns:
        The bearer access token, or ``""`` when *creds* is ``None`` or
        cannot be refreshed.
    """
    if creds is None:
        return ""
    try:
        if not creds.valid and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        return str(creds.token or "")
    except Exception:
        return ""


def make_google_auth_tools(
    agent: Any,
    service: str,
    label: str,
    scopes: list[str],
    on_credentials: Any,
) -> list:
    """Build the standard auth tool quartet for a Google Workspace agent.

    Produces ``check_<service>_auth``, ``authenticate_<service>``,
    ``clear_<service>_auth``, and ``start_<service>_browser_setup``
    closures whose behaviour mirrors the Gmail agent's auth tools.

    Args:
        agent: The channel agent instance (used to reach its backend).
        service: Service directory name (e.g. ``"google_calendar"``),
            also used in the generated tool names.
        label: Human-readable service label (e.g. ``"Google Calendar"``)
            used in messages and docstrings.
        scopes: OAuth scopes to request.
        on_credentials: Callback ``(creds | None) -> None`` invoked with
            new credentials after authentication (or ``None`` after
            clearing) so the agent can wire its backend.

    Returns:
        The four auth tool callables, named for *service*.
    """

    def check_auth() -> str:
        if not agent._is_authenticated():
            creds_file = credentials_path(service)
            if creds_file.exists():
                return (
                    f"Not authenticated with {label}. A credentials.json exists at "
                    f"{creds_file}. Call authenticate_{service}() to run the OAuth2 flow."
                )
            return (
                f"Not authenticated with {label}. Call start_{service}_browser_setup() "
                "to create OAuth credentials in Google Cloud Console, then "
                f"authenticate_{service}() to complete the OAuth2 flow."
            )
        return json.dumps({"ok": True, "message": f"{label} credentials are configured."})

    def authenticate() -> str:
        try:
            creds = run_google_oauth_flow(service, scopes)
        except Exception as e:
            return json.dumps(
                {
                    "ok": False,
                    "error": (
                        f"OAuth flow failed for {label}: {e}. The credentials.json at "
                        f"{credentials_path(service)} may be malformed; re-download it "
                        "from Google Cloud Console."
                    ),
                }
            )
        if creds is None:
            expected = google_service_dir(service) / "credentials.json"
            return (
                f"credentials.json not found for {label}. Download it from Google "
                "Cloud Console > APIs & Services > Credentials > OAuth 2.0 Client "
                f"IDs > Download JSON, then save it to {expected} (a copy at "
                f"{google_service_dir('google') / 'credentials.json'} is shared by "
                "all Google agents)."
            )
        on_credentials(creds)
        return json.dumps({"ok": True, "message": f"{label} authentication successful."})

    def clear_auth() -> str:
        clear_google_credentials(service)
        on_credentials(None)
        return f"{label} authentication cleared."

    def start_browser_setup() -> str:
        return (
            "Open https://console.cloud.google.com/apis/credentials with your "
            "go_to_url browser tool and complete these steps autonomously: "
            f"1. Create or select a project. 2. Enable the {label} API "
            "(APIs & Services > Enable APIs). 3. Credentials > Create Credentials "
            "> OAuth client ID > Desktop app. 4. Download the JSON and save it to "
            f"{google_service_dir(service) / 'credentials.json'}. "
            f"5. Call authenticate_{service}() to finish the OAuth consent flow. "
            "Use ask_user_question() only if stuck on a Google login screen."
        )

    check_auth.__name__ = f"check_{service}_auth"
    check_auth.__doc__ = (
        f"Check whether {label} OAuth2 credentials are configured.\n\n"
        "Returns:\n"
        "    Authentication status, or instructions for how to authenticate."
    )
    authenticate.__name__ = f"authenticate_{service}"
    authenticate.__doc__ = (
        f"Run the {label} OAuth2 installed-app flow and store the token.\n\n"
        "Opens a browser window (or prints an auth URL when headless) for the\n"
        "user to authorize access. Requires a credentials.json from Google\n"
        "Cloud Console.\n\n"
        "Returns:\n"
        "    Authentication result, instructions when credentials.json is missing,\n"
        "    or an {\"ok\": false, \"error\": ...} JSON string when the OAuth flow\n"
        "    fails (e.g. a malformed credentials.json)."
    )
    clear_auth.__name__ = f"clear_{service}_auth"
    clear_auth.__doc__ = (
        f"Clear the stored {label} OAuth2 token.\n\nReturns:\n    Status message."
    )
    start_browser_setup.__name__ = f"start_{service}_browser_setup"
    start_browser_setup.__doc__ = (
        f"Begin automated {label} API credential setup via the browser.\n\n"
        "Returns:\n"
        "    Step-by-step instructions for navigating Google Cloud Console."
    )
    return [check_auth, authenticate, clear_auth, start_browser_setup]
