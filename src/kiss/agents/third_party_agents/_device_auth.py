# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Connect-style consent sessions: the user signs in, the agent polls.

Meta's Muse app connects a third-party service with one click: a browser
page opens, the user signs in with the service and approves, and the app
is connected.  Nothing is copied by hand.  Two poll-based consent
protocols give KISS connectors the same experience even when the agent
runs on a remote or headless machine, because no redirect ever has to
reach the agent's host:

* the OAuth 2.0 Device Authorization Grant (RFC 8628) — GitHub, Twitch
  and Microsoft Entra ID (Teams) support it for public clients, i.e.
  with a client ID only and no client secret;
* Nextcloud's Login Flow v2 — needs no client registration at all.

Both work the same way: ``authenticate_<service>()`` starts a session,
opens the sign-in URL for the USER (in the Browser tab streamed to every
KISS surface when the kiss-web daemon runs this agent, else in the
user's default browser on this machine; see
:mod:`kiss.core.browser_handoff`), and hands back that URL (plus a
short code where the provider does not pre-fill it) so the agent can
relay it when no page could be shown; a background thread polls the
provider until the approval lands; and
``finish_<service>_auth()`` collects the result and enrolls the
credential.  The sign-in page itself is never driven by the agent, and
the user's password or second factor is never requested.
"""

from __future__ import annotations

import base64
import hashlib
import math
import secrets
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit

import requests

from kiss.core.brand import PRODUCT_NAME
from kiss.core.browser_handoff import BROWSER_TAB, DEFAULT_BROWSER, open_for_user

DEVICE_CODE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
USER_AGENT = PRODUCT_NAME

# Poll-loop bounds.  Providers announce their own ``expires_in`` and
# ``interval``; these caps keep a hostile or buggy answer from creating
# a thread that polls forever or hammers the endpoint.
_MAX_LIFETIME = 30 * 60.0
_MIN_INTERVAL = 1.0
_MAX_INTERVAL = 60.0
_SLOW_DOWN_STEP = 5.0
_TIMEOUT = 30.0


@dataclass(frozen=True)
class DeviceFlowProvider:
    """Endpoints and dialect of one RFC 8628 authorization server.

    Attributes:
        device_url: Device-authorization endpoint (``POST`` form).
        token_url: Token endpoint polled with the device code.
        scope_param: Name of the scope form field (Twitch spells it
            ``scopes``).
        token_scope_param: When set, the scope is repeated in every
            token poll under this field name (Twitch requires it).
    """

    device_url: str
    token_url: str
    scope_param: str = "scope"
    token_scope_param: str = ""


def _post_form(url: str, form: dict[str, str] | None = None) -> tuple[int, dict[str, Any]]:
    """POST a URL-encoded form and decode the JSON answer.

    Redirects are never followed: the endpoint was fixed by the
    provider spec (or by the user's own server URL), and a 3xx must not
    forward the form to another origin.  Like the Muse boundary, the
    request ignores ambient proxy and netrc configuration (the form
    carries the device code and, on success, the answer carries the
    credential).

    Args:
        url: Endpoint URL.
        form: Form fields (``None`` for an empty body).

    Returns:
        ``(status_code, body)`` where *body* is the decoded JSON object,
        or ``{}`` when the answer carries no JSON object.

    Raises:
        requests.RequestException: On a transport failure.
    """
    session = requests.Session()
    session.trust_env = False
    with session:
        resp = session.post(
            url,
            data=form,
            headers={"Accept": "application/json", "User-Agent": USER_AGENT},
            timeout=_TIMEOUT,
            allow_redirects=False,
        )
    try:
        body = resp.json() if resp.content else {}
    except ValueError:
        body = {}
    return resp.status_code, body if isinstance(body, dict) else {}


def _bounded(value: Any, default: float, low: float, high: float) -> float:
    """Coerce a provider-announced number of seconds into ``[low, high]``.

    Args:
        value: The raw JSON value (any type).
        default: Used when *value* is not a finite number.
        low: Lower bound.
        high: Upper bound.

    Returns:
        The clamped number of seconds.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    return min(max(number, low), high)


class ConsentSession:
    """One pending user consent, polled by a background thread.

    Subclasses implement :meth:`_poll_once` (one provider round trip
    returning the credential result or ``None`` while pending) and set
    ``verification_uri``/``user_code`` in their constructor.  At most
    one session per service is active: starting a new one cancels the
    previous one, so abandoned sessions never accumulate threads.
    """

    _active: dict[str, ConsentSession] = {}
    _registry_lock = threading.Lock()

    def __init__(self, service: str, lifetime: float, interval: float) -> None:
        self.service = service
        self.verification_uri = ""
        self.user_code = ""
        # True when ``verification_uri`` already carries the code.
        self.code_prefilled = False
        self.expires_in = int(lifetime)
        self.result: dict[str, Any] | None = None
        # Wall-clock time the approval landed; token lifetimes count
        # from here, not from when finish_<service>_auth() is called.
        self.result_at = 0.0
        self.error = ""
        # Non-secret choices made when the session was started (e.g. a
        # read-only flag) that the finish step applies; kept here so an
        # unfinished sign-in never alters the stored configuration.
        self.options: dict[str, Any] = {}
        self._interval = interval
        self._deadline = time.monotonic() + lifetime
        self._cancelled = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _poll_once(self) -> dict[str, Any] | None:
        """Ask the provider once whether the user has approved.

        Returns:
            The credential result when approval landed, else ``None``.

        Raises:
            RuntimeError: When the provider reports a terminal failure.
        """
        raise NotImplementedError

    def _run(self) -> None:
        """Poll until approval, terminal failure, expiry, or cancel."""
        try:
            while not self._cancelled and time.monotonic() < self._deadline:
                self._sleep(self._interval)
                if self._cancelled:
                    return
                try:
                    result = self._poll_once()
                except (requests.ConnectionError, requests.Timeout):
                    # RFC 8628 section 3.5: after a connection timeout
                    # keep polling at a reduced frequency.
                    self.slow_down()
                    continue
                if result is not None:
                    self.result_at = time.time()
                    self.result = result
                    return
            if not self._cancelled:
                self.error = "the sign-in request expired before it was approved"
        except Exception as e:
            self.error = str(e) or type(e).__name__

    def _sleep(self, seconds: float) -> None:
        """Sleep *seconds* in short steps so a cancel is honored promptly."""
        end = time.monotonic() + seconds
        while not self._cancelled and time.monotonic() < end:
            time.sleep(min(0.25, max(0.0, end - time.monotonic())))

    def slow_down(self) -> None:
        """Back off after a ``slow_down`` answer (RFC 8628 section 3.5)."""
        self._interval = min(self._interval + _SLOW_DOWN_STEP, _MAX_INTERVAL)

    def cancel(self) -> None:
        """Stop polling; the thread exits at its next wake-up."""
        self._cancelled = True

    def register(self) -> None:
        """Make this the active session of its service and start polling.

        A previously started, still-pending session for the same
        service is cancelled first.
        """
        with ConsentSession._registry_lock:
            previous = ConsentSession._active.pop(self.service, None)
            ConsentSession._active[self.service] = self
        if previous is not None:
            previous.cancel()
        self._thread.start()

    @classmethod
    def finish(cls, service: str, wait_seconds: float = 5.0) -> tuple[ConsentSession | None, str]:
        """Collect *service*'s active session once the user approved.

        Args:
            service: Connector service name.
            wait_seconds: How long to wait for a still-running poll
                thread before reporting ``pending``.

        Returns:
            ``(session, "ok")`` once the user approved (``session.result``
            holds the provider's answer), ``(None, "pending")`` while
            the approval has not landed yet, or ``(None, error_message)``
            when no session is active or the session failed (the
            session is dropped in that case).
        """
        with cls._registry_lock:
            session = cls._active.get(service)
        if session is None:
            return None, f"no sign-in in progress; call authenticate_{service}() first"
        session._thread.join(timeout=wait_seconds)
        if session._thread.is_alive():
            return None, "pending"
        with cls._registry_lock:
            # Compare-and-pop: a newer session registered meanwhile
            # must stay active, and this stale one must not be handed out.
            if cls._active.get(service) is not session:
                return None, "pending"
            del cls._active[service]
        if session.result is None:
            return None, session.error or "sign-in failed"
        return session, "ok"

    @classmethod
    def cancel_active(cls, service: str) -> None:
        """Cancel and forget *service*'s active session, if any."""
        with cls._registry_lock:
            session = cls._active.pop(service, None)
        if session is not None:
            session.cancel()


class DeviceFlowSession(ConsentSession):
    """RFC 8628 device authorization grant for a public OAuth client."""

    def __init__(
        self, service: str, provider: DeviceFlowProvider, client_id: str, scope: str
    ) -> None:
        """Request the device and user codes and start polling.

        Args:
            service: Connector service name.
            provider: The authorization server's endpoints and dialect.
            client_id: Public OAuth client ID (never a secret).
            scope: Space-separated scopes to request.

        Raises:
            RuntimeError: When the device-authorization request is refused.
        """
        form = {"client_id": client_id, provider.scope_param: scope}
        status, data = _post_form(provider.device_url, form)
        if status != 200 or not data.get("device_code") or not data.get("verification_uri"):
            raise RuntimeError(_describe_failure(status, data, "device authorization"))
        super().__init__(
            service,
            _bounded(data.get("expires_in"), 900.0, 30.0, _MAX_LIFETIME),
            _bounded(data.get("interval"), 5.0, _MIN_INTERVAL, _MAX_INTERVAL),
        )
        self.provider = provider
        self.client_id = client_id
        self.scope = scope
        self.device_code = str(data["device_code"])
        # A provider that pre-fills the code (``verification_uri_complete``,
        # RFC 8628 section 3.3.1) spares the user from typing it; the code
        # is still shown so they can confirm it matches (GitHub and
        # Microsoft only offer the plain URI).
        complete = data.get("verification_uri_complete")
        self.verification_uri = str(complete or data["verification_uri"])
        self.user_code = str(data.get("user_code") or "")
        self.code_prefilled = bool(complete)

    def _poll_once(self) -> dict[str, Any] | None:
        """Poll the token endpoint once with the device code."""
        form = {
            "client_id": self.client_id,
            "device_code": self.device_code,
            "grant_type": DEVICE_CODE_GRANT,
        }
        if self.provider.token_scope_param:
            form[self.provider.token_scope_param] = self.scope
        status, data = _post_form(self.provider.token_url, form)
        if data.get("access_token"):
            return data
        # Twitch reports the RFC error code under ``message``.
        code = str(data.get("error") or data.get("message") or "")
        if code == "authorization_pending":
            return None
        if code == "slow_down":
            self.slow_down()
            return None
        raise RuntimeError(_describe_failure(status, data, "sign-in"))


def _describe_failure(status: int, data: dict[str, Any], what: str) -> str:
    """Turn a token-endpoint refusal into a short, credential-free message."""
    code = str(data.get("error") or data.get("message") or "").strip()
    description = str(data.get("error_description") or "").strip()
    detail = code or f"HTTP {status}"
    if description and description.lower() != code.lower():
        detail = f"{detail}: {description[:200]}"
    return f"{what} refused ({detail})"


class NextcloudLoginSession(ConsentSession):
    """Nextcloud Login Flow v2: sign in and grant access in the browser."""

    def __init__(self, service: str, base_url: str) -> None:
        """Open a login-flow session on the Nextcloud server.

        Args:
            service: Connector service name (``"nextcloud"``).
            base_url: The server's base URL (no trailing slash).

        Raises:
            RuntimeError: When the server does not offer Login Flow v2
                or points the poll endpoint at a different origin.
        """
        status_code, data = _post_form(f"{base_url}/index.php/login/v2")
        poll = data.get("poll")
        login = data.get("login")
        if (
            status_code != 200
            or not isinstance(poll, dict)
            or not poll.get("token")
            or not poll.get("endpoint")
            or not login
        ):
            raise RuntimeError(
                f"{base_url} did not offer Nextcloud Login Flow v2 (HTTP {status_code})"
            )
        endpoint = str(poll["endpoint"])
        if _origin(endpoint) != _origin(base_url):
            # The poll answer carries the new app password; it must go
            # to the server the user configured, nowhere else.
            raise RuntimeError(
                f"{base_url} returned a login poll endpoint on another origin ({endpoint})"
            )
        # Nextcloud keeps the poll token for 20 minutes.
        super().__init__(service, 20 * 60.0, 1.0)
        self.base_url = base_url
        self.verification_uri = str(login)
        self._poll_endpoint = endpoint
        self._poll_token = str(poll["token"])

    def _poll_once(self) -> dict[str, Any] | None:
        """Poll once; 404 means the user has not granted access yet."""
        status, data = _post_form(self._poll_endpoint, {"token": self._poll_token})
        if status == 404:
            return None
        if status == 200 and data.get("appPassword") and data.get("loginName"):
            return data
        raise RuntimeError(f"Nextcloud login poll failed (HTTP {status})")

    def server_url(self) -> str:
        """Return the server base URL the granted app password belongs to.

        Login Flow v2 answers with the server's canonical ``server``
        URL, which clients are meant to use from then on (it may differ
        from what the user typed by scheme, trailing path or host
        canonicalization).  It is honored when it is an http(s) URL;
        otherwise the configured base URL is kept.
        """
        server = str((self.result or {}).get("server") or "").strip().rstrip("/")
        parts = urlsplit(server)
        if parts.scheme.lower() in ("http", "https") and parts.hostname:
            return server
        return self.base_url


@dataclass(frozen=True)
class PkceProvider:
    """Endpoints of an OAuth 2.0 authorization server used with PKCE.

    Attributes:
        authorize_url: Authorization endpoint the user's browser opens.
        token_url: Token endpoint the code is exchanged at.
    """

    authorize_url: str
    token_url: str


class _CallbackHandler(BaseHTTPRequestHandler):
    """Records the OAuth redirect on the loopback server's session."""

    server: _LoopbackServer  # type: ignore[assignment]
    # Socket timeout of an accepted connection: a half-open browser
    # request must not pin the serving thread (and the port) forever.
    timeout = 5

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        """Record the redirect query and show a completion page."""
        parts = urlsplit(self.path)
        if parts.path != self.server.callback_path:
            self.send_error(404)
            return
        self.server.session._callback = dict(parse_qsl(parts.query))
        body = b"Sign-in received. You can close this tab and return to the chat."
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: Any) -> None:  # type: ignore[override]
        """Silence per-request logging."""


class _LoopbackServer(HTTPServer):
    """Loopback redirect server bound to one :class:`LoopbackPkceSession`."""

    allow_reuse_address = True

    def __init__(self, port: int, callback_path: str, session: LoopbackPkceSession) -> None:
        self.callback_path = callback_path
        self.session = session
        super().__init__(("127.0.0.1", port), _CallbackHandler)


class LoopbackPkceSession(ConsentSession):
    """OAuth 2.0 authorization-code grant with PKCE for a public client.

    Slack and Discord offer no device flow, but both accept a public
    client that proves possession with PKCE (RFC 7636) instead of a
    client secret.  The session binds a loopback redirect server, hands
    back the authorization URL for the user's browser, and exchanges the
    returned code for tokens once the redirect lands.  When the user
    approves on another device, their browser stops at an unreachable
    ``http://localhost:PORT/callback?...`` page; the agent replays that
    pasted URL here with ``curl`` and the exchange completes locally.

    At most one loopback session runs at a time because every KISS app
    registers the same fixed redirect port.
    """

    loopback = True
    _port_lock = threading.RLock()
    _port_owner: LoopbackPkceSession | None = None

    def __init__(
        self,
        service: str,
        provider: PkceProvider,
        client_id: str,
        redirect_uri: str,
        params: dict[str, str],
        lifetime: float = 600.0,
    ) -> None:
        """Bind the redirect server and build the authorization URL.

        Args:
            service: Connector service name.
            provider: The authorization server's endpoints.
            client_id: Public OAuth client ID (never a secret).
            redirect_uri: Registered ``http://localhost:PORT/path`` URI.
            params: Extra authorization parameters (scopes and any
                provider-specific fields).
            lifetime: Seconds the user has to approve.

        Raises:
            OSError: When the redirect port cannot be bound.
        """
        super().__init__(service, lifetime, 0.5)
        self.provider = provider
        self.client_id = client_id
        self.redirect_uri = redirect_uri
        self._callback: dict[str, str] | None = None
        self._verifier = secrets.token_urlsafe(64)
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(self._verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        self._state = secrets.token_urlsafe(24)
        redirect = urlsplit(redirect_uri)
        with LoopbackPkceSession._port_lock:
            owner = LoopbackPkceSession._port_owner
            if owner is not None:
                owner.cancel()
            self._server = _LoopbackServer(redirect.port or 80, redirect.path or "/", self)
            self._server.timeout = 0.25
            self._server_thread = threading.Thread(target=self._serve, daemon=True)
            self._server_thread.start()
            # Published only once fully set up: a concurrent constructor
            # cancels the owner, which joins its serving thread.
            LoopbackPkceSession._port_owner = self
        query = {
            **params,
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": self._state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        self.verification_uri = f"{provider.authorize_url}?{urlencode(query)}"

    def _serve(self) -> None:
        """Answer redirect requests until the session stops, then close the socket.

        The socket is closed here, on the serving thread: closing it from
        another thread while ``handle_request`` is blocked in ``select``
        keeps the kernel socket (and the port) alive until that call
        returns, so a replacing session would fail to bind.
        """
        try:
            while not self._cancelled and self.result is None and not self.error:
                self._server.handle_request()
        finally:
            self._server.server_close()

    def cancel(self) -> None:
        """Stop polling and release the redirect port."""
        super().cancel()
        self._close_server()

    def _close_server(self) -> None:
        """Wait for the redirect server to close and give up port ownership.

        Every caller has already set a stop condition (cancel, result or
        error), so the serving loop exits within one ``handle_request``
        timeout.
        """
        with LoopbackPkceSession._port_lock:
            if LoopbackPkceSession._port_owner is self:
                LoopbackPkceSession._port_owner = None
            self._server_thread.join()

    def _run(self) -> None:
        """Poll for the redirect, then release the port."""
        try:
            super()._run()
        finally:
            self._close_server()

    def _poll_once(self) -> dict[str, Any] | None:
        """Exchange the code once the redirect has arrived."""
        callback = self._callback
        if callback is None:
            return None
        if not secrets.compare_digest(callback.get("state", ""), self._state):
            raise RuntimeError("sign-in refused (state mismatch)")
        if callback.get("error"):
            raise RuntimeError(_describe_failure(400, callback, "sign-in"))
        form = {
            "grant_type": "authorization_code",
            "code": callback.get("code", ""),
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "code_verifier": self._verifier,
        }
        status, data = _post_form(self.provider.token_url, form)
        # Slack answers HTTP 200 with ``ok: false`` on failure.
        if status == 200 and not data.get("error") and data.get("ok", True) is not False:
            return data
        raise RuntimeError(_describe_failure(status, data, "code exchange"))


def _origin(url: str) -> tuple[str, str, int | None]:
    """Return ``(scheme, host, port)`` of *url* for same-origin checks.

    The scheme's default port is made explicit so ``https://h`` and
    ``https://h:443`` compare equal.
    """
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    port = parts.port
    if port is None:
        port = {"http": 80, "https": 443}.get(scheme)
    return scheme, (parts.hostname or "").lower(), port


def consent_instructions(
    service: str, label: str, session: ConsentSession, opened_in: str = ""
) -> str:
    """Build the agent-facing hand-off text for a started session.

    Args:
        service: Connector service name (used in the finish tool name).
        label: Human-readable service label.
        session: The session whose URL and code the user needs.
        opened_in: Where :func:`~kiss.core.browser_handoff.open_for_user`
            put the sign-in page: ``BROWSER_TAB`` (the streamed Browser
            tab every KISS surface switched to), ``DEFAULT_BROWSER``
            (the user's browser on this machine) or ``""`` (nowhere).

    Returns:
        Step-by-step instructions: the USER signs in and approves; in the
        Browser tab the agent must not hand out the URL at all, otherwise
        it shows the URL and code, then it calls the finish tool.
    """
    code_step = ""
    if session.user_code and session.code_prefilled:
        code_step = f" and confirm the code shown is {session.user_code}"
    elif session.user_code:
        code_step = f" and enter the code {session.user_code} when the page asks for it"
    minutes = max(session.expires_in // 60, 1)
    if opened_in == BROWSER_TAB:
        return (
            f"Connect {label} the way the Muse app does: the USER signs in and "
            "approves; you only tell them what to do. The sign-in page is already "
            "open in the Browser tab that every KISS surface (web app and VS Code) "
            "has just switched to, so the user is looking at it now: do NOT ask "
            "them to open a URL and do NOT give them the URL to open; do NOT open "
            f"this URL or any {label} sign-in page in your built-in browser, and "
            "never ask for or type the user's password or 2FA code. Steps: 1) Call "
            f"ask_user_question() telling the user to sign in to {label} in the "
            f"Browser tab that just opened{code_step}, approve the access request, "
            f"and reply here when done (the page is valid for about {minutes} "
            "minutes); only if they answer that they cannot see the page, give "
            f"them {session.verification_uri} to open themselves. 2) Call "
            f"finish_{service}_auth(); if it returns 'pending', wait a few seconds "
            "and call it again. Nothing has to be pasted back."
            + _loopback_step(session)
        )
    if opened_in == DEFAULT_BROWSER:
        opened = (
            "The sign-in page has just been opened in the user's default browser "
            "on this machine; "
        )
    else:
        opened = "No browser could be opened from this machine (headless or remote); "
    return (
        f"Connect {label} the way the Muse app does: the USER signs in and "
        f"approves; you only relay the link and code. {opened}do NOT open this "
        f"URL or any {label} sign-in page in your built-in browser, and never "
        "ask for or type the user's password or 2FA code. Steps: 1) Call "
        "ask_user_question() giving the user this exact URL to open in their "
        f"OWN browser if no window appeared: {session.verification_uri} — tell "
        f"them to sign in to {label}{code_step}, approve the access request, and "
        f"reply here when done (the link is valid for about {minutes} minutes). "
        f"2) Call finish_{service}_auth(); if it returns 'pending', wait a few "
        "seconds and call it again. Nothing has to be pasted back."
        + _loopback_step(session)
    )


def _loopback_step(session: ConsentSession) -> str:
    """Return the paste-back hand-off for a loopback-redirect session.

    Args:
        session: The started session.

    Returns:
        Extra instructions for a :class:`LoopbackPkceSession` (whose
        redirect only reaches this machine), else ``""``.
    """
    if not getattr(session, "loopback", False):
        return ""
    return (
        " Exception: if the user approved on ANOTHER device, their browser ends "
        "on an unreachable http://localhost:PORT/callback?... page; ask them to "
        "paste that complete URL back and deliver it to this machine with Bash: "
        "curl -s '<pasted URL>' (quote it), then call the finish tool."
    )


def consent_required(service: str, label: str, session: ConsentSession) -> dict[str, Any]:
    """Build the ``authenticate_<service>()`` answer for a started session.

    The verification page is opened for the user by
    :func:`~kiss.core.browser_handoff.open_for_user`: in the streamed
    Browser tab when the kiss-web daemon runs this agent, else in the
    user's default browser when this machine has one.  The URL and code
    are returned in every case so the agent can relay them when the
    user cannot see the page.

    Args:
        service: Connector service name.
        label: Human-readable service label.
        session: The freshly started session.

    Returns:
        A JSON-ready dict with ``status: consent_required``, the URL, the
        code (empty when pre-filled), the expiry, where the page was
        opened (``opened_in``: ``browser_tab``, ``default_browser`` or
        ``""``), ``browser_opened`` (whether it was opened anywhere), and
        the instructions.
    """
    opened_in = open_for_user(session.verification_uri)
    return {
        "ok": True,
        "status": "consent_required",
        "verification_uri": session.verification_uri,
        "user_code": session.user_code,
        "expires_in": session.expires_in,
        "opened_in": opened_in,
        "browser_opened": bool(opened_in),
        "instructions": consent_instructions(service, label, session, opened_in),
    }


@dataclass
class TokenGrant:
    """A token response mapped onto a Muse vault credential.

    Attributes:
        access_token: The bearer access token.
        refresh_token: The refresh token, or ``""`` when the provider
            issued a non-expiring token.
        expires_in: Access-token lifetime in seconds (0 when unknown).
        scope: Granted scope as reported by the provider (string form).
        raw: The full token response.
    """

    access_token: str
    refresh_token: str = ""
    expires_in: float = 0.0
    scope: str = ""
    acquired_at: float = field(default_factory=time.time)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_session(cls, session: ConsentSession) -> TokenGrant:
        """Normalize the token response a finished device-flow session holds.

        Args:
            session: A session whose ``result`` is set.

        Returns:
            The grant, with ``acquired_at`` set to the moment the poll
            thread received the token so the lifetime is not extended by
            a late ``finish_<service>_auth()`` call.
        """
        return cls.from_response(session.result or {}, acquired_at=session.result_at)

    @classmethod
    def from_response(cls, data: dict[str, Any], acquired_at: float = 0.0) -> TokenGrant:
        """Normalize a token-endpoint success body.

        Args:
            data: The decoded token response.
            acquired_at: Wall-clock time the response was received
                (0 means now).

        Returns:
            The grant; ``scope`` joins list-valued scopes (Twitch) with
            spaces.
        """
        scope = data.get("scope") or ""
        if isinstance(scope, list):
            scope = " ".join(str(s) for s in scope)
        return cls(
            access_token=str(data.get("access_token") or ""),
            refresh_token=str(data.get("refresh_token") or ""),
            expires_in=_bounded(data.get("expires_in"), 0.0, 0.0, 30 * 86400.0),
            scope=str(scope),
            acquired_at=acquired_at or time.time(),
            raw=data,
        )

    def vault_credential(
        self, token_url: str, client_id: str, refresh_scope: str = ""
    ) -> dict[str, Any]:
        """Return the vault payload for this grant.

        A grant with a refresh token becomes an ``oauth2_refresh_token``
        credential that the Muse daemon refreshes itself (public client:
        no secret involved); a non-expiring token is a plain ``bearer``.

        Args:
            token_url: Token endpoint the daemon refreshes against.
            client_id: Public client ID sent with the refresh grant.
            refresh_scope: Scope to repeat on refresh (Microsoft), or
                ``""`` to omit the field.

        Returns:
            The ``authorized_user_info`` dict for ``store_credentials``.
        """
        if not self.refresh_token:
            return {"kind": "bearer", "token": self.access_token}
        lifetime = self.expires_in if self.expires_in > 0 else 3600.0
        info: dict[str, Any] = {
            "kind": "oauth2_refresh_token",
            "token_url": token_url,
            "client_id": client_id,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.acquired_at + lifetime,
        }
        if refresh_scope:
            info["token_scope"] = refresh_scope
        return info


def connect_prompt(service: str, label: str, start_call: str, prerequisite: str) -> str:
    """Build the "## <label> Authentication" channel-prompt section.

    Args:
        service: Connector service name (used in tool names).
        label: Human-readable service label.
        start_call: The ``authenticate_<service>(...)`` call that starts
            the browser sign-in, as shown to the agent.
        prerequisite: One sentence on what the user must have (e.g. a
            public client ID) before the sign-in can start.

    Returns:
        The prompt section describing the Connect-style hand-off.
    """
    return (
        f"\n\n## {label} Authentication\n"
        f"1. Call check_{service}_auth() first; if it reports ok, use the tools and "
        "never re-run authentication over a valid credential.\n"
        f"2. To connect, call {start_call}. {prerequisite} It returns "
        "status 'consent_required' with a verification URL (and a short code when "
        "the provider does not pre-fill it) and opens that page for the user by "
        "itself: 'opened_in' is 'browser_tab' when it is already showing in the "
        "Browser tab on every KISS surface, 'default_browser' when it opened in the "
        "user's browser on this machine, or empty when nothing could be opened.\n"
        "3. The USER completes the sign-in, exactly like clicking Connect in the "
        "Muse app, and you follow the tool's 'instructions': when the page is in "
        "the Browser tab, call ask_user_question() telling the user to sign in to "
        f"{label} there (and the code, if any) and reply when done — never ask them "
        "to open a URL; otherwise call ask_user_question() with the URL (and code) "
        "so they can open it in their OWN browser. Do NOT open the URL or any "
        "sign-in page in your built-in browser, do not retry or relaunch the "
        "browser, and never ask for or type the user's password or 2FA code. "
        "Nothing is pasted back.\n"
        f"4. Then call finish_{service}_auth(); if it returns 'pending', wait a few "
        "seconds and call it again. Confirm the result with "
        f"check_{service}_auth()."
    )
