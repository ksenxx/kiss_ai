# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Matrix Agent — channel agent with Matrix protocol tools.

Provides authenticated access to Matrix via matrix-nio. Stores credentials
in ``~/.kiss/third_party_agents/matrix/config.json``.

Sign-in works the way the Muse app connects a service when the homeserver
offers the Matrix OAuth 2.0 API (matrix.org and every homeserver backed
by Matrix Authentication Service): ``authenticate_matrix(homeserver_url)``
discovers the authorization server, registers this client dynamically,
and starts the device authorisation grant; the user approves in their own
browser and ``finish_matrix_auth()`` collects the short-lived access token
plus its refresh token, which the backend renews by itself.  Homeservers
without the OAuth API still take a hand-supplied access token.

Usage::

    agent = MatrixAgent()
    agent.run(prompt_template="Send 'Hello!' to #general:matrix.org")
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import requests

from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
    config_file_lock,
    write_private_file,
)
from kiss.agents.third_party_agents._device_auth import (
    DEVICE_CODE_GRANT,
    ConsentSession,
    DeviceFlowProvider,
    DeviceFlowSession,
    TokenGrant,
    connect_prompt,
    consent_required,
)

logger = logging.getLogger(__name__)

_MATRIX_DIR = Path.home() / ".kiss" / "third_party_agents" / "matrix"
_config = ChannelConfig(
    _MATRIX_DIR,
    (
        "homeserver_url",
        "access_token",
    ),
)


# Client metadata registered with the homeserver's authorization server
# (Matrix spec, "Client registration").  ``client_uri`` must be https and
# is the common base of every other URI; the client is public
# (``token_endpoint_auth_method: none``) and uses only the device and
# refresh grants, so no redirect URI is registered.
_CLIENT_METADATA: dict[str, Any] = {
    "client_name": "KISS Sorcar",
    "client_uri": "https://kisssorcar.github.io/",
    "application_type": "native",
    "token_endpoint_auth_method": "none",
    "grant_types": [DEVICE_CODE_GRANT, "refresh_token"],
    "response_types": [],
}
_DEVICE_ID_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_HTTP_TIMEOUT = 30.0
# Refresh this long before the announced expiry so a token about to
# expire mid-request is never used.
_REFRESH_SKEW = 60.0


def _http_session() -> requests.Session:
    """Return a session immune to ambient proxy/netrc configuration."""
    session = requests.Session()
    session.trust_env = False
    return session


def _discover_oauth(homeserver_url: str) -> dict[str, Any] | None:
    """Fetch the homeserver's OAuth 2.0 authorization server metadata.

    ``GET /_matrix/client/v1/auth_metadata`` (RFC 8414 metadata) tells
    whether the homeserver supports the OAuth 2.0 API and, if so, where
    the registration, device authorization, token and revocation
    endpoints live.

    Args:
        homeserver_url: The homeserver base URL (no trailing slash).

    Returns:
        The metadata dict when the homeserver supports the device
        authorisation grant, or ``None`` when it does not offer the
        OAuth 2.0 API at all (HTTP 404).

    Raises:
        RuntimeError: When the metadata cannot be fetched, is malformed,
            or lacks the device grant / dynamic registration.
    """
    with _http_session() as session:
        resp = session.get(
            f"{homeserver_url}/_matrix/client/v1/auth_metadata",
            headers={"Accept": "application/json"},
            timeout=_HTTP_TIMEOUT,
            allow_redirects=False,
        )
    if resp.status_code == 404:
        return None
    try:
        data = resp.json() if resp.content else {}
    except ValueError:
        data = {}
    if resp.status_code != 200 or not isinstance(data, dict):
        raise RuntimeError(f"auth metadata request failed (HTTP {resp.status_code})")
    required = ("token_endpoint", "device_authorization_endpoint", "registration_endpoint")
    missing = [key for key in required if not _secure_endpoint(str(data.get(key) or ""))]
    if missing:
        raise RuntimeError(
            "the homeserver's OAuth 2.0 metadata lacks an https "
            + ", ".join(missing)
            + "; pass an access_token instead"
        )
    if DEVICE_CODE_GRANT not in (data.get("grant_types_supported") or []):
        raise RuntimeError(
            "the homeserver's authorization server does not offer the device "
            "authorisation grant; pass an access_token instead"
        )
    # Tokens are only ever POSTed to an https revocation endpoint; an
    # insecure or missing one simply disables revocation on clear.
    if not _secure_endpoint(str(data.get("revocation_endpoint") or "")):
        data["revocation_endpoint"] = ""
    return data


def _secure_endpoint(url: str) -> bool:
    """Return whether *url* may receive tokens: https, or loopback (tests)."""
    from kiss.agents.third_party_agents.muse_auth._common import is_loopback_host

    parts = urllib.parse.urlsplit(url)
    if not parts.hostname:
        return False
    if parts.scheme == "https":
        return True
    return parts.scheme == "http" and is_loopback_host(parts.hostname)


def _register_client(registration_endpoint: str) -> str:
    """Register this client dynamically and return its ``client_id``.

    Args:
        registration_endpoint: The authorization server's registration
            endpoint from the metadata.

    Returns:
        The allocated public client ID.

    Raises:
        RuntimeError: When the server refuses the registration.
    """
    with _http_session() as session:
        resp = session.post(
            registration_endpoint,
            json=_CLIENT_METADATA,
            headers={"Accept": "application/json"},
            timeout=_HTTP_TIMEOUT,
            allow_redirects=False,
        )
    try:
        data = resp.json() if resp.content else {}
    except ValueError:
        data = {}
    client_id = data.get("client_id") if isinstance(data, dict) else None
    if resp.status_code not in (200, 201) or not client_id:
        code = data.get("error") if isinstance(data, dict) else None
        detail = code if isinstance(code, str) else f"HTTP {resp.status_code}"
        raise RuntimeError(f"client registration refused ({detail})")
    return str(client_id)


def _new_device_id() -> str:
    """Return a fresh 10-character device ID from RFC 3986 unreserved chars."""
    return "".join(secrets.choice(_DEVICE_ID_ALPHABET) for _ in range(10))


def _whoami(homeserver_url: str, access_token: str) -> tuple[dict[str, Any] | None, str]:
    """Validate an access token with a direct ``/account/whoami`` read.

    Args:
        homeserver_url: The homeserver base URL.
        access_token: The token to validate.

    Returns:
        ``(answer, "")`` with the decoded whoami object on success, or
        ``(None, error)`` where *error* never contains the token.
    """
    try:
        with _http_session() as session:
            resp = session.get(
                f"{homeserver_url}/_matrix/client/v3/account/whoami",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=_HTTP_TIMEOUT,
                allow_redirects=False,
            )
    except Exception as e:
        return None, f"{type(e).__name__} while validating the token"
    try:
        data = resp.json() if resp.content else {}
    except ValueError:
        data = {}
    if resp.status_code != 200 or not isinstance(data, dict) or not data.get("user_id"):
        code = data.get("errcode") if isinstance(data, dict) else None
        return None, str(code) if isinstance(code, str) else f"HTTP {resp.status_code}"
    return data, ""


def _refresh_grant(cfg: dict[str, str]) -> dict[str, str]:
    """Run the refresh token grant for an OAuth-issued Matrix session.

    Matrix spec, "Refresh token grant": network failures and 5xx answers
    leave the session intact (retry later with the old refresh token);
    a 4xx answer means the session was logged out.

    Args:
        cfg: The stored config with ``token_url``, ``oauth_client_id``
            and ``refresh_token``.

    Returns:
        The config updates (``access_token``, ``refresh_token`` when
        rotated, ``expires_at``), or ``{}`` when the refresh must be
        retried later.

    Raises:
        RuntimeError: When the authorization server rejected the
            refresh token (the session is logged out).
    """
    form = {
        "grant_type": "refresh_token",
        "refresh_token": cfg["refresh_token"],
        "client_id": cfg["oauth_client_id"],
    }
    try:
        with _http_session() as session:
            resp = session.post(
                cfg["token_url"],
                data=form,
                headers={"Accept": "application/json"},
                timeout=_HTTP_TIMEOUT,
                allow_redirects=False,
            )
    except requests.RequestException:
        return {}
    if resp.status_code >= 500:
        return {}
    try:
        data = resp.json() if resp.content else {}
    except ValueError:
        data = {}
    if not isinstance(data, dict) or not data.get("access_token") or resp.status_code != 200:
        code = data.get("error") if isinstance(data, dict) else None
        detail = code if isinstance(code, str) else f"HTTP {resp.status_code}"
        raise RuntimeError(
            f"the Matrix session was logged out ({detail}); run authenticate_matrix() again"
        )
    grant = TokenGrant.from_response(data)
    updates = {
        "access_token": grant.access_token,
        "expires_at": str(grant.acquired_at + (grant.expires_in or 3600.0)),
    }
    if grant.refresh_token:
        updates["refresh_token"] = grant.refresh_token
    return updates


def _revoke(cfg: dict[str, str]) -> bool:
    """Revoke an OAuth-issued session (RFC 7009), best effort.

    Args:
        cfg: The stored config with ``revocation_url``, ``access_token``
            and ``oauth_client_id``.

    Returns:
        True when the server confirmed the revocation.
    """
    try:
        with _http_session() as session:
            resp = session.post(
                cfg["revocation_url"],
                data={
                    "token": cfg["access_token"],
                    "token_type_hint": "access_token",
                    "client_id": cfg.get("oauth_client_id", ""),
                },
                timeout=_HTTP_TIMEOUT,
                allow_redirects=False,
            )
    except Exception:
        return False
    return resp.status_code == 200


def _client_from_config(cfg: dict[str, str]) -> Any:
    """Build a matrix-nio ``AsyncClient`` from a stored config."""
    from nio import AsyncClient

    client = AsyncClient(cfg["homeserver_url"])
    client.access_token = cfg["access_token"]
    if cfg.get("device_id"):
        client.device_id = cfg["device_id"]
    if cfg.get("user_id"):
        client.user_id = cfg["user_id"]
    return client


def _raise_on_send_error(resp: Any, room_id: str) -> Any:
    """Raise ``RuntimeError`` when *resp* is a matrix-nio error response.

    The shared ``ChannelRunner`` treats a ``send_message`` that returns
    without raising as a successful delivery and deletes the reply from its
    at-least-once ledger, but nio's ``room_send`` reports failures (rate
    limits, auth failures) by *returning* an error response rather than
    raising — so those responses must be converted into exceptions here or
    the reply is silently lost.

    When matrix-nio is importable, the check is
    ``isinstance(resp, nio.ErrorResponse)``, which covers
    ``nio.responses.RoomSendError`` and every other error subclass.  nio is
    an optional dependency (imported lazily throughout this module), so when
    it is missing the check falls back to the same structural contract: nio
    error responses carry both ``message`` and ``status_code`` attributes,
    while success responses such as ``RoomSendResponse`` carry neither.

    Args:
        resp: Response object returned by ``AsyncClient.room_send``.
        room_id: Target room, included in the error message for context.

    Returns:
        *resp* unchanged when it is not an error response.

    Raises:
        RuntimeError: If *resp* is an error response; the message includes
            the response's ``status_code`` and ``message`` details.
    """
    try:
        from nio import ErrorResponse
    except ImportError:
        error_type: type | None = None
    else:  # pragma: no cover - nio is not installed in the test environment
        error_type = ErrorResponse
    if error_type is not None:  # pragma: no cover - nio-installed path
        is_error = isinstance(resp, error_type)
    else:
        is_error = hasattr(resp, "message") and hasattr(resp, "status_code")
    if is_error:
        status = getattr(resp, "status_code", None) or "unknown"
        message = getattr(resp, "message", "")
        raise RuntimeError(f"Matrix send to {room_id} failed: {status} {message}".rstrip())
    return resp


class MatrixChannelBackend(ToolMethodBackend):
    """Channel backend for Matrix via matrix-nio."""

    def __init__(self) -> None:
        self._client: Any = None
        self._next_batch: str = ""
        self._connection_info: str = ""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._refresh_lock = threading.Lock()

    def refresh_if_needed(self) -> None:
        """Renew an OAuth-issued access token shortly before it expires.

        No-op for hand-supplied (legacy) tokens, which carry no refresh
        token.  The rotated refresh token is persisted at once so a
        lost answer cannot strand the session.

        Raises:
            RuntimeError: When the authorization server logged the
                session out (a 4xx refusal).
        """
        with self._refresh_lock:
            cfg = _config.load() or {}
            if not cfg.get("refresh_token") or self._client is None:
                return
            try:
                expires_at = float(cfg.get("expires_at") or 0.0)
            except ValueError:
                expires_at = 0.0
            if time.time() < expires_at - _REFRESH_SKEW:
                return
            updates = _refresh_grant(cfg)
            if not updates:
                return
            # Compare-and-swap under the cross-process config lock: a
            # clear or a direct re-authentication that landed while the
            # grant was in flight wins, and the tokens minted for the
            # abandoned session are revoked instead of resurrecting it.
            with config_file_lock(_config.path):
                current = _config.load() or {}
                same_session = current.get("refresh_token") == cfg["refresh_token"]
                if same_session:
                    current.update(updates)
                    write_private_file(_config.path, json.dumps(current, indent=2))
            if not same_session:
                if cfg.get("revocation_url"):
                    _revoke({**cfg, "access_token": updates["access_token"]})
                return
            if self._client is not None:
                self._client.access_token = updates["access_token"]

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Lazily start (and reuse) the persistent background event loop.

        nio's ``AsyncClient`` caches its aiohttp session on the event loop
        of the first request, so every coroutine must run on ONE loop that
        stays alive for the backend's lifetime.
        """
        if self._loop is None:
            loop = asyncio.new_event_loop()
            thread = threading.Thread(target=loop.run_forever, name="matrix-loop", daemon=True)
            thread.start()
            self._loop = loop
            self._loop_thread = thread
        return self._loop

    def _run(self, coro: Coroutine[Any, Any, Any], timeout: float = 120.0) -> Any:
        """Run ``coro`` on the persistent background loop and return its result."""
        try:
            self.refresh_if_needed()
        except RuntimeError:
            coro.close()
            raise
        future = asyncio.run_coroutine_threadsafe(coro, self._ensure_loop())
        return future.result(timeout=timeout)

    def connect(self) -> bool:
        """Authenticate with Matrix using stored config and validate the token."""
        cfg = _config.load()
        if not cfg:  # pragma: no branch
            self._connection_info = "No Matrix config found."
            return False
        try:
            self._client = _client_from_config(cfg)
            resp = self._run(self._client.whoami())
            user_id = getattr(resp, "user_id", "")
            if not user_id:
                self._connection_info = f"Matrix auth failed: {resp}"
                return False
            if not self._client.user_id:  # pragma: no branch
                self._client.user_id = user_id
            self._connection_info = f"Connected to {cfg['homeserver_url']} as {user_id}"
            return True
        except Exception as e:
            self._connection_info = f"Matrix connection failed: {e}"
            return False

    def disconnect(self) -> None:
        """Close the Matrix client session and stop the background loop."""
        if self._client is not None and self._loop is not None:
            try:
                self._run(self._client.close())
            except Exception:
                pass
        loop = self._loop
        thread = self._loop_thread
        self._loop = None
        self._loop_thread = None
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
            if thread is not None:  # pragma: no branch
                thread.join(timeout=10)
            loop.close()

    def find_channel(self, name: str) -> str | None:
        """Resolve a room alias (#room:server) to its room ID.

        Args:
            name: Room ID (!room:server) or alias (#room:server).

        Returns:
            The room ID, or ``None`` if *name* is empty or unresolvable.
        """
        if not name:
            return None
        if not name.startswith("#") or not self._client:
            return name
        try:
            resp = self._run(self._client.room_resolve_alias(name))
            room_id = getattr(resp, "room_id", "")
            return str(room_id) if room_id else None
        except Exception:
            return None

    def join_channel(self, channel_id: str) -> None:
        """Join a Matrix room."""
        if self._client:  # pragma: no branch
            self._run(self._client.join(channel_id))

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll for new Matrix messages via sync.

        The runner's persisted cursor (*oldest*, a ``next_batch`` sync
        token) takes precedence over the in-memory ``_next_batch``,
        which is empty in each fresh cron-tick process — syncing with
        ``since=None`` there would re-deliver recent room timelines on
        every tick.

        When the homeserver rejects a supplied ``since`` token (nio
        returns an error response without ``next_batch`` — e.g. a
        stale or invalidated token), the sync is retried ONCE with
        ``since=None``: the runner's nonempty-cursor guard would
        otherwise retain the rejected token forever, permanently
        bricking the poll.  If the full-sync retry also fails, the
        tick is a transient no-op (``([], oldest)``).
        """
        if not self._client:  # pragma: no branch
            return [], oldest
        try:
            from nio import RoomMessageText

            since = oldest if oldest not in ("", "0") else (self._next_batch or None)

            async def _sync(since_token: str | None) -> Any:
                return await self._client.sync(since=since_token, timeout=0)

            resp = self._run(_sync(since))
            if since is not None and not hasattr(resp, "next_batch"):
                logger.warning(
                    "Matrix sync rejected since-token %r; retrying with a full sync",
                    since,
                )
                resp = self._run(_sync(None))
                if not hasattr(resp, "next_batch"):
                    return [], oldest
            if hasattr(resp, "next_batch"):  # pragma: no branch
                self._next_batch = resp.next_batch
            messages: list[dict[str, Any]] = []
            if channel_id and hasattr(resp, "rooms"):  # pragma: no branch
                room = resp.rooms.join.get(channel_id)
                if room:  # pragma: no branch
                    for event in room.timeline.events:  # pragma: no branch
                        if isinstance(event, RoomMessageText):  # pragma: no branch
                            messages.append(
                                {
                                    "ts": str(event.server_timestamp),
                                    "user": event.sender,
                                    "text": event.body,
                                    "event_id": event.event_id,
                                }
                            )
            return messages, self._next_batch
        except Exception:
            return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a Matrix text message.

        Raises:
            RuntimeError: If the homeserver rejects the send (nio returns an
                error response such as ``RoomSendError`` instead of raising),
                so the channel runner's at-least-once delivery ledger does
                not count the reply as delivered and can redeliver it.
        """
        if not self._client:  # pragma: no branch
            return

        async def _send() -> Any:
            return await self._client.room_send(
                channel_id,
                message_type="m.room.message",
                content={"msgtype": "m.text", "body": text},
            )

        _raise_on_send_error(self._run(_send()), channel_id)

    def send_typing(self, channel_id: str, thread_ts: str = "") -> None:
        """Send a Hermes-style typing indicator to a Matrix room.

        Issues ``PUT /_matrix/client/v3/rooms/{roomId}/typing/{userId}``
        with body ``{"typing": true, "timeout": 15000}`` against the
        backend's configured homeserver, authenticating with the stored
        access token. Best-effort: any transport or server error is
        logged and swallowed, never raised. If the backend has no client
        or no stored user id, this returns without doing anything.

        Args:
            channel_id: Matrix room ID (!room:server) to show typing in.
            thread_ts: Unused; present for channel-backend signature parity.
        """
        del thread_ts
        client = self._client
        if client is None:
            return
        user_id = str(getattr(client, "user_id", "") or "")
        if not user_id:
            return
        try:
            homeserver = str(getattr(client, "homeserver", "") or "").rstrip("/")
            access_token = str(getattr(client, "access_token", "") or "")
            url = (
                f"{homeserver}/_matrix/client/v3/rooms/"
                f"{urllib.parse.quote(channel_id, safe='')}/typing/"
                f"{urllib.parse.quote(user_id, safe='')}"
            )
            body = json.dumps({"typing": True, "timeout": 15000}).encode("utf-8")
            request = urllib.request.Request(
                url,
                data=body,
                method="PUT",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=10):
                pass
        except Exception as e:
            logging.getLogger(__name__).debug("Matrix typing indicator failed: %s", e)

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Check if message is from the bot."""
        if self._client and hasattr(self._client, "user_id"):  # pragma: no branch
            return bool(msg.get("user", "") == self._client.user_id)
        return False

    def list_rooms(self) -> str:
        """List joined Matrix rooms.

        Returns:
            JSON string with room list (id, name, topic).
        """
        if not self._client:  # pragma: no branch
            return json.dumps({"ok": False, "error": "Not connected"})
        try:

            async def _get() -> Any:
                return await self._client.joined_rooms()

            resp = self._run(_get())
            rooms = [{"id": r} for r in getattr(resp, "rooms", [])]
            return json.dumps({"ok": True, "rooms": rooms}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def join_room(self, room_id_or_alias: str) -> str:
        """Join a Matrix room.

        Args:
            room_id_or_alias: Room ID (!room:server.org) or alias (#room:server.org).

        Returns:
            JSON string with ok status and room id.
        """
        if not self._client:  # pragma: no branch
            return json.dumps({"ok": False, "error": "Not connected"})
        try:

            async def _join() -> Any:
                return await self._client.join(room_id_or_alias)

            resp = self._run(_join())
            return json.dumps({"ok": True, "room_id": getattr(resp, "room_id", room_id_or_alias)})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def leave_room(self, room_id: str) -> str:
        """Leave a Matrix room.

        Args:
            room_id: Room ID to leave.

        Returns:
            JSON string with ok status.
        """
        if not self._client:  # pragma: no branch
            return json.dumps({"ok": False, "error": "Not connected"})
        try:

            async def _leave() -> None:
                await self._client.room_leave(room_id)

            self._run(_leave())
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_text_message(self, room_id: str, text: str) -> str:
        """Send a text message to a Matrix room.

        Args:
            room_id: Room ID.
            text: Message text.

        Returns:
            JSON string with ok status and event id.
        """
        if not self._client:  # pragma: no branch
            return json.dumps({"ok": False, "error": "Not connected"})
        try:

            async def _send() -> Any:
                return await self._client.room_send(
                    room_id,
                    message_type="m.room.message",
                    content={"msgtype": "m.text", "body": text},
                )

            resp = self._run(_send())
            return json.dumps({"ok": True, "event_id": getattr(resp, "event_id", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_notice(self, room_id: str, text: str) -> str:
        """Send a notice (bot message) to a Matrix room.

        Args:
            room_id: Room ID.
            text: Notice text.

        Returns:
            JSON string with ok status and event id.
        """
        if not self._client:  # pragma: no branch
            return json.dumps({"ok": False, "error": "Not connected"})
        try:

            async def _send() -> Any:
                return await self._client.room_send(
                    room_id,
                    message_type="m.room.message",
                    content={"msgtype": "m.notice", "body": text},
                )

            resp = self._run(_send())
            return json.dumps({"ok": True, "event_id": getattr(resp, "event_id", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_room_members(self, room_id: str) -> str:
        """Get members of a Matrix room.

        Args:
            room_id: Room ID.

        Returns:
            JSON string with member list.
        """
        if not self._client:  # pragma: no branch
            return json.dumps({"ok": False, "error": "Not connected"})
        try:

            async def _get() -> Any:
                return await self._client.joined_members(room_id)

            resp = self._run(_get())
            members = [
                {"user_id": m.user_id, "display_name": m.display_name or ""}
                for m in getattr(resp, "members", [])
            ]
            return json.dumps({"ok": True, "members": members}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def invite_user(self, room_id: str, user_id: str) -> str:
        """Invite a user to a Matrix room.

        Args:
            room_id: Room ID.
            user_id: User ID to invite (@user:server.org).

        Returns:
            JSON string with ok status.
        """
        if not self._client:  # pragma: no branch
            return json.dumps({"ok": False, "error": "Not connected"})
        try:

            async def _invite() -> None:
                await self._client.room_invite(room_id, user_id)

            self._run(_invite())
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def kick_user(self, room_id: str, user_id: str, reason: str = "") -> str:
        """Kick a user from a Matrix room.

        Args:
            room_id: Room ID.
            user_id: User ID to kick.
            reason: Optional reason for kick.

        Returns:
            JSON string with ok status.
        """
        if not self._client:  # pragma: no branch
            return json.dumps({"ok": False, "error": "Not connected"})
        try:

            async def _kick() -> None:
                await self._client.room_kick(room_id, user_id, reason=reason)

            self._run(_kick())
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def create_room(
        self,
        name: str = "",
        topic: str = "",
        is_public: bool = False,
        alias: str = "",
    ) -> str:
        """Create a new Matrix room.

        Args:
            name: Room display name.
            topic: Room topic.
            is_public: Whether the room is publicly joinable. Default: False.
            alias: Optional local alias (without server part).

        Returns:
            JSON string with room id.
        """
        if not self._client:  # pragma: no branch
            return json.dumps({"ok": False, "error": "Not connected"})
        try:
            from nio import RoomVisibility

            async def _create() -> Any:
                return await self._client.room_create(
                    name=name,
                    topic=topic,
                    is_direct=False,
                    visibility=RoomVisibility.public if is_public else RoomVisibility.private,
                    alias=alias or None,
                )

            resp = self._run(_create())
            return json.dumps({"ok": True, "room_id": getattr(resp, "room_id", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_profile(self, user_id: str) -> str:
        """Get a Matrix user's profile.

        Args:
            user_id: User ID (@user:server.org).

        Returns:
            JSON string with display name and avatar.
        """
        if not self._client:  # pragma: no branch
            return json.dumps({"ok": False, "error": "Not connected"})
        try:

            async def _get() -> Any:
                return await self._client.get_profile(user_id)

            resp = self._run(_get())
            return json.dumps(
                {
                    "ok": True,
                    "display_name": getattr(resp, "displayname", ""),
                    "avatar_url": getattr(resp, "avatar_url", ""),
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class MatrixAgent(BaseChannelAgent):
    """Channel agent with Matrix protocol tools."""

    channel_system_prompt = connect_prompt(
        "matrix",
        "Matrix",
        "authenticate_matrix(homeserver_url=...) without an access_token",
        "It works on homeservers with the Matrix OAuth 2.0 API (matrix.org and any "
        "server backed by Matrix Authentication Service); the tool says so when the "
        "homeserver lacks it, in which case the user may hand you an access token for "
        "authenticate_matrix(homeserver_url=..., access_token=...).",
    ).lstrip()

    def __init__(self) -> None:
        super().__init__("Matrix Agent")
        self._backend = MatrixChannelBackend()
        cfg = _config.load()
        if cfg:  # pragma: no branch
            try:
                self._backend._client = _client_from_config(cfg)
            except Exception:
                pass

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return self._backend._client is not None

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_matrix_auth() -> str:
            """Check if Matrix credentials are configured and valid.

            Returns:
                Authentication status or instructions.
            """
            if agent._backend._client is None:  # pragma: no branch
                return (
                    "Not authenticated with Matrix. Call "
                    "authenticate_matrix(homeserver_url=...) (e.g. https://matrix.org) "
                    "to sign in the way the Muse app connects: it returns a link for the "
                    "user to open in their OWN browser, sign in and approve; then call "
                    "finish_matrix_auth(). Never ask for the user's Matrix password or "
                    "2FA code. Only when the homeserver lacks the OAuth 2.0 API (the "
                    "tool says so) may the user hand you an access token (Element > All "
                    "Settings > Help & About > 'Access Token') for "
                    "authenticate_matrix(homeserver_url=..., access_token=...)."
                )
            try:
                resp = agent._backend.list_rooms()
                data = json.loads(resp)
                if data.get("ok"):  # pragma: no branch
                    return json.dumps({"ok": True, "room_count": len(data.get("rooms", []))})
                return str(resp)
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_matrix(
            homeserver_url: str,
            access_token: str = "",
            device_id: str = "",
            user_id: str = "",
        ) -> str:
            """Connect Matrix by browser sign-in or with an access token.

            Without ``access_token`` this uses the homeserver's OAuth 2.0
            API: the authorization server is discovered, this client is
            registered as a public client, and the device authorisation
            grant starts.  The answer is ``consent_required`` with a
            verification URL (and code): give it to the user
            (ask_user_question) to complete in their OWN browser, then
            call finish_matrix_auth().  With ``access_token`` the token is
            stored directly (needed for homeservers without the OAuth
            2.0 API).

            Args:
                homeserver_url: Matrix homeserver URL (e.g. "https://matrix.org").
                access_token: Optional access token from Element or the login API.
                device_id: Optional device ID (browser sign-in allocates one).
                user_id: Optional user ID (@user:server.org).

            Returns:
                A consent_required JSON answer, an authentication result,
                or an error message.
            """
            from kiss.agents.third_party_agents.muse_auth._common import valid_http_url

            homeserver_url = homeserver_url.strip().rstrip("/")
            if not homeserver_url:  # pragma: no branch
                return "homeserver_url cannot be empty."
            if not valid_http_url(homeserver_url):
                return json.dumps(
                    {"ok": False, "error": f"{homeserver_url!r} is not a valid http(s):// URL."}
                )
            if access_token.strip():
                # A hand-supplied token supersedes any browser sign-in
                # still pending.
                ConsentSession.cancel_active("matrix")
                try:
                    cfg = {
                        "homeserver_url": homeserver_url,
                        "access_token": access_token.strip(),
                        "device_id": device_id.strip(),
                        "user_id": user_id.strip(),
                    }
                    agent._backend._client = _client_from_config(cfg)
                    _config.save(cfg)
                    return json.dumps(
                        {
                            "ok": True,
                            "message": "Matrix credentials saved.",
                            "homeserver": homeserver_url,
                        }
                    )
                except Exception as e:
                    return json.dumps({"ok": False, "error": str(e)})
            try:
                metadata = _discover_oauth(homeserver_url)
                if metadata is None:
                    return json.dumps(
                        {
                            "ok": False,
                            "error": f"{homeserver_url} does not offer the Matrix OAuth 2.0 "
                            "API, so there is no browser sign-in. Ask the user for an "
                            "access token (Element > All Settings > Help & About > "
                            "'Access Token') and call authenticate_matrix(homeserver_url"
                            "=..., access_token=...).",
                        }
                    )
                # Matrix spec: a new client registration at the start
                # of each authorisation flow (servers de-duplicate
                # identical metadata).
                client_id = _register_client(str(metadata["registration_endpoint"]))
                new_device_id = device_id.strip() or _new_device_id()
                session = DeviceFlowSession(
                    "matrix",
                    DeviceFlowProvider(
                        device_url=str(metadata["device_authorization_endpoint"]),
                        token_url=str(metadata["token_endpoint"]),
                    ),
                    client_id,
                    f"urn:matrix:client:api:* urn:matrix:client:device:{new_device_id}",
                )
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})
            session.options = {
                "homeserver_url": homeserver_url,
                "device_id": new_device_id,
                "revocation_url": str(metadata.get("revocation_endpoint") or ""),
            }
            session.register()
            return json.dumps(consent_required("matrix", "Matrix", session))

        def finish_matrix_auth() -> str:
            """Complete a browser sign-in started by authenticate_matrix().

            Call after the user reports that they approved the sign-in;
            the access token is validated with ``/account/whoami`` and
            stored together with its refresh token, which the agent uses
            to renew the short-lived access token by itself.

            Returns:
                The authentication result, a pending status while the
                user has not approved yet, or an error message.
            """
            session, status = ConsentSession.finish("matrix")
            if status == "pending":
                return json.dumps(
                    {
                        "ok": False,
                        "status": "pending",
                        "error": "The user has not approved yet; ask them to finish "
                        "the sign-in, then call this tool again.",
                    }
                )
            if not isinstance(session, DeviceFlowSession) or session.result is None:
                return json.dumps({"ok": False, "error": f"Matrix sign-in failed: {status}"})
            grant = TokenGrant.from_session(session)
            if not grant.refresh_token:
                return json.dumps(
                    {
                        "ok": False,
                        "error": "The homeserver issued no refresh token; its short-lived "
                        "access token could not be kept alive. Sign in again.",
                    }
                )
            homeserver_url = str(session.options["homeserver_url"])
            who, error = _whoami(homeserver_url, grant.access_token)
            if who is None:
                return json.dumps(
                    {"ok": False, "error": f"the homeserver rejected the new token: {error}"}
                )
            cfg = {
                "homeserver_url": homeserver_url,
                "access_token": grant.access_token,
                "device_id": str(who.get("device_id") or session.options["device_id"]),
                "user_id": str(who["user_id"]),
                "refresh_token": grant.refresh_token,
                "expires_at": str(grant.acquired_at + (grant.expires_in or 3600.0)),
                "token_url": session.provider.token_url,
                "oauth_client_id": session.client_id,
                "revocation_url": str(session.options["revocation_url"]),
            }
            try:
                _config.save(cfg)
            except Exception as e:
                return json.dumps({"ok": False, "error": f"failed to store the session: {e}"})
            answer: dict[str, Any] = {
                "ok": True,
                "message": "Matrix connected.",
                "homeserver": homeserver_url,
                "user_id": cfg["user_id"],
                "device_id": cfg["device_id"],
            }
            try:
                agent._backend._client = _client_from_config(cfg)
            except ImportError:
                # matrix-nio is an optional dependency; the session is
                # stored and picked up once it is installed.
                answer["warning"] = (
                    "matrix-nio is not installed (pip install matrix-nio); the "
                    "session is stored and will be used once it is available."
                )
            return json.dumps(answer)

        def clear_matrix_auth() -> str:
            """Clear the stored Matrix credentials.

            An OAuth-issued session is revoked on the authorization
            server first (RFC 7009), so the device disappears from the
            user's session list.

            Returns:
                Status message.
            """
            ConsentSession.cancel_active("matrix")
            cfg = _config.load() or {}
            revoked = bool(cfg.get("revocation_url") and cfg.get("refresh_token")) and _revoke(cfg)
            _config.clear()
            agent._backend._client = None
            if revoked:
                return "Matrix authentication cleared; the session was revoked."
            return "Matrix authentication cleared."

        return [check_matrix_auth, authenticate_matrix, finish_matrix_auth, clear_matrix_auth]


def _make_backend() -> MatrixChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = MatrixChannelBackend()
    cfg = _config.load()
    if not cfg:  # pragma: no branch
        print("Not authenticated. Run: kiss-matrix -t 'authenticate'")
        sys.exit(1)
    backend._client = _client_from_config(cfg)
    return backend


def main() -> None:
    """Run the MatrixAgent from the command line with chat persistence."""
    channel_main(
        MatrixAgent,
        "kiss-matrix",
        channel_name="Matrix",
        make_backend=_make_backend,
    )


def tools() -> list:
    """Return the Matrix channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return MatrixAgent()._get_tools()


if __name__ == "__main__":
    main()
