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

from kiss.agents.third_party_agents._browser_handoff import (
    open_in_default_browser,
    portal_handoff,
)
from kiss.agents.third_party_agents._channel_agent_utils import write_private_file
from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled
from kiss.core.config import kiss_home

# Where the user creates the OAuth "Desktop app" client (credentials.json).
CLOUD_CONSOLE_URL = "https://console.cloud.google.com/apis/credentials"


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


def start_google_consent(service: str, label: str, scopes: list[str]) -> str | None:
    """Start the OAuth consent for *service* and hand it to the user.

    The loopback consent server starts in the background
    (:class:`RemoteOAuthSession`), the Google consent page is opened in
    the user's default browser when this machine has one, and the
    ``authenticate_<service>()`` answer carries the URL for the user to
    open by hand otherwise.  ``finish_<service>_auth()`` completes it.

    Args:
        service: Service directory name the token is stored under.
        label: Human-readable service label.
        scopes: OAuth scopes to request.

    Returns:
        The JSON answer of ``authenticate_<service>()``: ``status:
        consent_required`` with ``auth_url``, ``browser_opened`` and
        ``instructions``, or ``ok: False`` with an ``error`` when the
        session could not start; ``None`` when no ``credentials.json``
        exists for the service.
    """
    try:
        session = RemoteOAuthSession.start(service, scopes)
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
    if session is None:
        return None
    browser_opened = open_in_default_browser(session.auth_url)
    return json.dumps(
        {
            "ok": True,
            "status": "consent_required",
            "auth_url": session.auth_url,
            "browser_opened": browser_opened,
            "instructions": remote_oauth_instructions(
                service, label, session.auth_url, browser_opened
            ),
        }
    )


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
    ``clear_<service>_auth``, ``start_<service>_browser_setup``, and
    ``finish_<service>_auth`` closures whose behaviour mirrors the
    Gmail agent's auth tools.

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
                    f"{creds_file}. Call authenticate_{service}() to start the OAuth2 "
                    "consent; it opens the consent page in the user's default browser "
                    "when it can and returns the auth_url to show the user."
                )
            return (
                f"Not authenticated with {label}. Call start_{service}_browser_setup() "
                "to open Google Cloud Console in the user's default browser so they "
                f"can create OAuth credentials, then authenticate_{service}() to "
                "start the OAuth2 consent."
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

    def authenticate() -> str:
        answer = start_google_consent(service, label, scopes)
        if answer is None:
            return missing_credentials_message
        return answer

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
            f"The user creates the OAuth client themselves. {portal_handoff(CLOUD_CONSOLE_URL)} "
            "Ask them to: 1. Create or select a project. 2. Enable the "
            f"{label} API (APIs & Services > Enable APIs). 3. Credentials > Create "
            "Credentials > OAuth client ID > Desktop app. 4. Download the JSON and "
            "either paste its content back or save it to "
            f"{google_service_dir(service) / 'credentials.json'}. Write pasted "
            "content to that path yourself, then call "
            f"authenticate_{service}() to start the OAuth consent. Do not drive "
            "Google Cloud Console or any Google sign-in page with your built-in "
            "browser, and never ask for the user's Google password or 2FA code."
        )

    check_auth.__name__ = f"check_{service}_auth"
    check_auth.__doc__ = (
        f"Check whether {label} OAuth2 credentials are configured.\n\n"
        "Returns:\n"
        "    Authentication status, or instructions for how to authenticate."
    )
    authenticate.__name__ = f"authenticate_{service}"
    authenticate.__doc__ = (
        f"Start the {label} OAuth2 consent flow.\n\n"
        "Starts the loopback consent server, opens the Google consent page in\n"
        "the user's default browser when this machine has one, and returns the\n"
        "auth_url for the user to open by hand otherwise. Requires a\n"
        "credentials.json from Google Cloud Console. Complete the flow with\n"
        f"finish_{service}_auth().\n\n"
        "Returns:\n"
        "    status 'consent_required' with auth_url, browser_opened and\n"
        "    instructions; instructions when credentials.json is missing; or an\n"
        "    {\"ok\": false, \"error\": ...} JSON string when the OAuth flow\n"
        "    fails (e.g. a malformed credentials.json)."
    )
    clear_auth.__name__ = f"clear_{service}_auth"
    clear_auth.__doc__ = (
        f"Clear the stored {label} OAuth2 token.\n\nReturns:\n    Status message."
    )
    start_browser_setup.__name__ = f"start_{service}_browser_setup"
    start_browser_setup.__doc__ = (
        f"Open Google Cloud Console for the user to create {label} OAuth credentials.\n\n"
        "Opens the Credentials page in the user's default browser when this\n"
        "machine has one and returns the steps to relay to the user.\n\n"
        "Returns:\n"
        "    The console URL and step-by-step instructions for the user."
    )
    finish_auth.__name__ = f"finish_{service}_auth"
    finish_auth.__doc__ = (
        f"Complete the {label} OAuth consent started by "
        f"authenticate_{service}().\n\n"
        "Call after the user has approved consent in their own browser (and,\n"
        "when they did so on another machine, after the pasted redirect URL\n"
        "has been delivered to the local consent server with\n"
        "``curl -s '<pasted redirect URL>'``).\n\n"
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
    """Non-blocking OAuth consent: loopback server now, credentials later.

    ``InstalledAppFlow.run_local_server`` blocks until a browser
    completes consent, which stalls the agent's tool call and is
    useless on a remote machine where the user cannot see a local
    browser window.  This session starts the loopback redirect server
    in a background thread and hands back the authorization URL; the
    caller opens it in the user's default browser when it can and shows
    it to the USER in any case (agent browsers frequently cannot reach
    ``accounts.google.com``, and the sign-in belongs to the user).  An
    approval in a browser on this machine lands on the loopback server
    directly.  From another device the user's browser lands on a
    ``http://localhost:PORT/?state=...&code=...`` URL that fails to
    load there; the user pastes that URL back and the agent replays it
    against the loopback server here (``curl <url>``) so the exchange
    completes locally.  ``finish`` collects the resulting credentials
    once the redirect has been delivered.
    """

    _active: dict[str, RemoteOAuthSession] = {}
    _registry_lock = threading.Lock()

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
        with cls._registry_lock:
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
        with cls._registry_lock:
            session = cls._active.get(service)
        if session is None:
            return None, f"no OAuth session in progress; call authenticate_{service}() first"
        session._thread.join(timeout=2.0)
        if session._thread.is_alive():
            return None, "pending"
        with cls._registry_lock:
            # Compare-and-pop: a newer session registered meanwhile
            # must stay active, and this stale one must not be handed out.
            if cls._active.get(service) is not session:
                return None, "pending"
            del cls._active[service]
        if session.credentials is None:
            return None, session.error or "OAuth flow failed"
        save_google_credentials(service, session.credentials)
        if muse_auth_enabled():
            return load_google_credentials(service, scopes), "ok"
        return session.credentials, "ok"


def remote_oauth_instructions(
    service: str, label: str, auth_url: str, browser_opened: bool = False
) -> str:
    """Build the agent-facing instructions for a started consent session.

    Args:
        service: Service directory name (used in the finish tool name).
        label: Human-readable service label.
        auth_url: The authorization URL to hand to the user.
        browser_opened: Whether the consent page was already opened in
            the user's default browser on this machine.

    Returns:
        Step-by-step instructions for the user-driven consent hand-off:
        the user authorizes in their own browser (already open when
        *browser_opened*; the loopback redirect then completes by itself)
        and otherwise pastes back the loopback redirect URL, which the
        agent replays locally.
    """
    if browser_opened:
        opened = (
            "The consent page has just been opened in the user's default browser "
            "on this machine, where the loopback consent server also runs, so an "
            "approval in that window completes by itself. Still show the URL: "
        )
    else:
        opened = "No browser could be opened from this machine (headless or remote). "
    return (
        f"Complete the {label} consent WITHOUT driving Google sign-in pages "
        "yourself: do NOT open accounts.google.com or this auth URL in your "
        "built-in browser (it is often blocked with net::ERR_FAILED), and "
        f"never ask for or type the user's Google password or 2FA code. {opened}"
        "Steps: 1) Call ask_user_question() giving the user this exact URL "
        f"to open in their OWN browser if no window appeared: {auth_url} — tell "
        "them to approve access and reply when done; if their browser ends on a "
        "connection error page at http://localhost:PORT/?state=...&code=... "
        "(it does when they used another device), ask them to paste back that "
        "complete redirect URL. 2) Only if a URL was pasted back: the loopback "
        "consent server runs on THIS machine, so deliver it with Bash: "
        "curl -s '<pasted redirect URL>' (quote it; it contains & characters). "
        f"3) Call finish_{service}_auth() to store the token; if it returns "
        "'pending', wait 2 seconds and call it once more. If any Google page "
        "fails to load in the browser, do not retry — use this hand-off."
    )


def google_consent_steps(service: str) -> str:
    """Build the consent hand-off paragraph shared by the Google agent prompts.

    Args:
        service: Service directory name (used in the tool names).

    Returns:
        The prompt text that follows ``authenticate_<service>()``: the tool
        opens the consent page in the user's default browser when it can,
        the agent always shows the auth_url, never drives Google pages,
        replays a pasted redirect URL when there is one, and finishes with
        ``finish_<service>_auth()``.
    """
    return (
        f"When authenticate_{service}() returns status 'consent_required' with an "
        "auth_url, it has already tried to open that URL in the user's default "
        "browser on this machine ('browser_opened' says whether it could); do NOT "
        "open the auth_url or any accounts.google.com page in your own browser, "
        "and never ask for or type the user's Google password or 2FA code: Google "
        "sign-in pages are often blocked in the built-in browser (net::ERR_FAILED), "
        "and the sign-in belongs to the user. Hand off consent instead:\n"
        "1. ALWAYS call ask_user_question() with the full auth_url, asking the user "
        "to open it in their OWN browser if no window appeared, approve access, and "
        "reply when done; if their browser ends on a connection error page at "
        "http://localhost:PORT/?state=...&code=... (it does when they approved on "
        "another device), ask them to paste back the complete redirect URL from the "
        "address bar.\n"
        "2. Only if a redirect URL was pasted back: the loopback consent server runs "
        "on THIS machine, so deliver the pasted URL to it with Bash: curl -s "
        "'<pasted redirect URL>' (quote the URL; it contains & characters).\n"
        f"3. Call finish_{service}_auth(); if it returns 'pending', wait 2 seconds "
        "and call it once more.\n"
        "If any browser navigation to a Google page fails, do not retry it or "
        "relaunch the browser — switch to this hand-off immediately."
    )
