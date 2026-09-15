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
import threading
from pathlib import Path
from typing import Any, cast

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from kiss.agents.third_party_agents._backend_utils import is_headless_environment
from kiss.agents.third_party_agents._channel_agent_utils import write_private_file
from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled
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

    In Muse-auth mode (the default) the real token stays in
    the daemon vault and a surrogate-bearing handle is returned
    instead; the agent process never reads ``token.json``.

    Returns:
        Valid :class:`Credentials`, a
        :class:`~kiss.agents.third_party_agents.muse_auth.client.SurrogateCredentials`
        in Muse-auth mode, or ``None`` when missing, invalid, or
        unrefreshable.
    """
    if muse_auth_enabled():
        # A leftover legacy token.json (working install upgraded to the
        # Muse-auth default) is migrated into the vault and removed.
        from kiss.agents.third_party_agents.muse_auth.client import mint_surrogate_migrating

        return cast(
            "Credentials | None", mint_surrogate_migrating(service, token_path(service), scopes)
        )
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


def save_google_credentials(service: str, creds: Any) -> None:
    """Persist a service's OAuth2 credentials atomically with 0600 permissions.

    In Muse-auth mode the credential goes into the daemon vault instead
    of an agent-readable ``token.json``; surrogate handles are skipped
    (there is nothing real to persist).

    Args:
        service: Service directory name.
        creds: Google OAuth2 credentials (or a surrogate handle) to persist.
    """
    if muse_auth_enabled():
        from kiss.agents.third_party_agents.muse_auth.client import (
            SurrogateCredentials,
            store_credentials,
        )

        if not isinstance(creds, SurrogateCredentials):
            store_credentials(service, creds, list(getattr(creds, "scopes", None) or []))
        return
    write_private_file(token_path(service), creds.to_json())


def clear_google_credentials(service: str) -> None:
    """Delete a service's stored OAuth2 token, if any.

    Clears both the legacy ``token.json`` and, in Muse-auth mode, the
    daemon vault entry (invalidating outstanding surrogates).

    Args:
        service: Service directory name.
    """
    path = token_path(service)
    if path.exists():
        path.unlink()
    if muse_auth_enabled():
        from kiss.agents.third_party_agents.muse_auth.client import clear_credentials

        clear_credentials(service)


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
    if muse_auth_enabled():
        # The real credential now lives in the daemon vault; hand the
        # caller a surrogate so no real token stays in agent memory.
        return load_google_credentials(service, scopes)
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
    """Build the standard auth tool set for a Google Workspace agent.

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
        The five auth tool callables, named for *service*.
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

    missing_credentials_message = (
        f"credentials.json not found for {label}. Download it from Google "
        "Cloud Console > APIs & Services > Credentials > OAuth 2.0 Client "
        f"IDs > Download JSON, then save it to "
        f"{google_service_dir(service) / 'credentials.json'} (a copy at "
        f"{google_service_dir('google') / 'credentials.json'} is shared by "
        "all Google agents)."
    )

    def flow_failed(e: Exception) -> str:
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

    def authenticate() -> str:
        if is_headless_environment():
            # Remote machine: hand back the consent URL for the user to
            # approve in their own browser; the pasted redirect URL is
            # replayed against the local consent server, then
            # finish_<service>_auth() completes the exchange.
            try:
                session = RemoteOAuthSession.start(service, scopes)
            except Exception as e:
                return flow_failed(e)
            if session is None:
                return missing_credentials_message
            return json.dumps(
                {
                    "ok": True,
                    "status": "consent_required",
                    "auth_url": session.auth_url,
                    "instructions": remote_oauth_instructions(service, label, session.auth_url),
                }
            )
        try:
            creds = run_google_oauth_flow(service, scopes)
        except Exception as e:
            return flow_failed(e)
        if creds is None:
            return missing_credentials_message
        on_credentials(creds)
        return json.dumps({"ok": True, "message": f"{label} authentication successful."})

    def finish_auth() -> str:
        creds, status = RemoteOAuthSession.finish(service, scopes)
        if status == "pending":
            return json.dumps(
                {
                    "ok": False,
                    "status": "pending",
                    "error": "Consent is not completed yet; finish the flow in the "
                             "browser, then call this tool again.",
                }
            )
        if creds is None:
            return json.dumps({"ok": False, "error": f"OAuth flow failed for {label}: {status}"})
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
    finish_auth.__name__ = f"finish_{service}_auth"
    finish_auth.__doc__ = (
        f"Complete a remote {label} OAuth consent started by "
        f"authenticate_{service}().\n\n"
        "Call after the user has approved consent in their own browser and\n"
        "the pasted redirect URL has been delivered to the local consent\n"
        "server (``curl -s '<pasted redirect URL>'``).\n\n"
        "Returns:\n"
        "    Authentication result, a pending status when consent is not\n"
        "    finished, or an error message."
    )
    return [check_auth, authenticate, clear_auth, start_browser_setup, finish_auth]


def google_api_session(service: str) -> Any:
    """Return the HTTP executor a Google REST backend should use.

    Legacy mode returns the ``requests`` module (direct calls, real
    token in the Authorization header).  In Muse-auth mode
    (the default) it returns a
    :class:`~kiss.agents.third_party_agents.muse_auth.client.MuseBoundarySession`
    that ships every request to the Muse-auth daemon, where Sentinel
    authorizes it and the surrogate bearer token is swapped for the
    real credential at the network boundary.

    Args:
        service: Connector service name (e.g. ``"google_drive"``).

    Returns:
        An object exposing ``request/get/post/put/patch/delete`` with
        the ``requests`` API.
    """
    if muse_auth_enabled():
        from kiss.agents.third_party_agents.muse_auth.client import MuseBoundarySession

        return MuseBoundarySession(service)
    return requests


class _OAuthCallbackApp:
    """Tiny WSGI app that records the OAuth redirect request URI."""

    def __init__(self) -> None:
        self.request_uri = ""

    def __call__(self, environ: Any, start_response: Any) -> list[bytes]:
        """Record the redirect URI and show a completion page.

        Args:
            environ: WSGI environment of the redirect request.
            start_response: WSGI start-response callable.

        Returns:
            The completion page body.
        """
        import wsgiref.util

        self.request_uri = wsgiref.util.request_uri(environ)
        start_response("200 OK", [("Content-Type", "text/plain; charset=utf-8")])
        return [b"Authentication complete. You can close this tab and return to the chat."]


class RemoteOAuthSession:
    """OAuth consent flow split for remote/headless machines.

    ``InstalledAppFlow.run_local_server`` blocks until a browser
    completes consent — useless on a remote machine where the user
    cannot see a local browser window.  This session starts the
    loopback redirect server in a background thread and hands back the
    authorization URL for the USER to open in their own browser (agent
    browsers frequently cannot reach ``accounts.google.com``, and the
    sign-in belongs to the user).  The user's browser then lands on a
    ``http://localhost:PORT/?state=...&code=...`` URL that fails to
    load on their machine; the user pastes that URL back and the agent
    replays it against the loopback server here (``curl <url>``) so
    the exchange completes locally.  ``finish`` collects the resulting
    credentials once the redirect has been delivered.
    """

    _active: dict[str, RemoteOAuthSession] = {}

    def __init__(self, service: str, scopes: list[str]) -> None:
        import wsgiref.simple_server

        class _QuietHandler(wsgiref.simple_server.WSGIRequestHandler):
            """Redirect-server handler with request logging silenced."""

            def log_message(self, *_args: Any) -> None:  # type: ignore[override]
                """Silence per-request logging."""

        self.service = service
        self.scopes = scopes
        flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path(service)), scopes)
        self._app = _OAuthCallbackApp()
        self._server = wsgiref.simple_server.make_server(
            "localhost", 0, self._app, handler_class=_QuietHandler
        )
        flow.redirect_uri = f"http://localhost:{self._server.server_port}/"
        self.auth_url, _ = flow.authorization_url()
        self._flow = flow
        self.credentials: Credentials | None = None
        self.error = ""
        self._cancelled = False
        self._thread = threading.Thread(target=self._wait_for_consent, daemon=True)
        self._thread.start()

    def _wait_for_consent(self) -> None:
        """Serve redirect requests until consent completes or cancelled.

        A short poll timeout lets an abandoned/replaced session's thread
        exit promptly instead of blocking forever on a single request.
        """
        self._server.timeout = 1.0
        try:
            while not self._cancelled and not self._app.request_uri:
                self._server.handle_request()
            if self._cancelled:
                return
            # oauthlib insists on https URLs; the loopback redirect is
            # local, so upgrading the scheme string is safe (this is
            # exactly what run_local_server does).
            response = self._app.request_uri.replace("http://", "https://", 1)
            self._flow.fetch_token(authorization_response=response)
            self.credentials = cast(Credentials, self._flow.credentials)
        except Exception as e:
            self.error = str(e)
        finally:
            self._server.server_close()

    def cancel(self) -> None:
        """Stop the consent server so its background thread can exit."""
        self._cancelled = True

    @classmethod
    def start(cls, service: str, scopes: list[str]) -> RemoteOAuthSession | None:
        """Begin (or restart) a remote consent session for *service*.

        A previously started, still-pending session for the same service
        is cancelled and its server closed before the new one starts, so
        abandoned consent servers/threads never accumulate.

        Args:
            service: Service directory name.
            scopes: OAuth scopes to request.

        Returns:
            The running session, or ``None`` when no ``credentials.json``
            exists for the service.
        """
        if not credentials_path(service).exists():
            return None
        previous = cls._active.pop(service, None)
        if previous is not None:
            previous.cancel()
        session = cls(service, scopes)
        cls._active[service] = session
        return session

    @classmethod
    def finish(cls, service: str, scopes: list[str]) -> tuple[Any, str]:
        """Collect the credentials of a completed consent session.

        On success the credentials are persisted through
        :func:`save_google_credentials` (vault in Muse-auth mode) and
        the caller receives the mode-appropriate handle.

        Args:
            service: Service directory name.
            scopes: OAuth scopes the session requested.

        Returns:
            ``(credentials, "ok")`` on success (a surrogate handle in
            Muse-auth mode), ``(None, "pending")`` while consent is
            still incomplete, or ``(None, error_message)`` when the
            flow failed or no session was started.
        """
        session = cls._active.get(service)
        if session is None:
            return None, f"no OAuth session in progress; call authenticate_{service}() first"
        session._thread.join(timeout=2.0)
        if session._thread.is_alive():
            return None, "pending"
        del cls._active[service]
        if session.credentials is None:
            return None, session.error or "OAuth flow failed"
        save_google_credentials(service, session.credentials)
        if muse_auth_enabled():
            return load_google_credentials(service, scopes), "ok"
        return session.credentials, "ok"


def remote_oauth_instructions(service: str, label: str, auth_url: str) -> str:
    """Build the agent-facing instructions for a remote consent session.

    Args:
        service: Service directory name (used in the finish tool name).
        label: Human-readable service label.
        auth_url: The authorization URL to hand to the user.

    Returns:
        Step-by-step instructions for the user-driven consent hand-off:
        the user authorizes in their own browser and pastes back the
        loopback redirect URL, which the agent replays locally.
    """
    return (
        f"Complete the {label} consent WITHOUT driving Google sign-in pages "
        "yourself: do NOT open accounts.google.com or this auth URL in your "
        "built-in browser (it is often blocked with net::ERR_FAILED), and "
        "never ask for or type the user's Google password or 2FA code. "
        "Steps: 1) Call ask_user_question() giving the user this exact URL "
        f"to open in their OWN browser: {auth_url} — tell them to approve "
        "access and paste back the complete redirect URL from the address "
        "bar (it looks like http://localhost:PORT/?state=...&code=... and "
        "shows a connection error page, which is expected). 2) The loopback "
        "consent server runs on THIS machine: deliver the pasted URL to it "
        "with Bash: curl -s '<pasted redirect URL>' (quote it; it contains "
        f"& characters). 3) Call finish_{service}_auth() to store the "
        "token; if it returns 'pending', wait 2 seconds and call it once "
        "more. If any Google page fails to load in the browser, do not "
        "retry — use this hand-off."
    )
