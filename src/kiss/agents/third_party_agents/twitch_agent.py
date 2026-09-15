# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Twitch Agent — channel agent with Twitch Helix API + Chat tools.

Provides authenticated access to Twitch via OAuth2 user tokens.  Connects
like the Muse app: ``authenticate_twitch(client_id=...)`` starts Twitch's
device code grant for a public client and hands back a
``twitch.tv/activate`` link with the code pre-filled; the user signs in
and authorizes in their own browser and ``finish_twitch_auth()`` stores
the token pair (the Muse daemon refreshes it, no client secret needed).
An access token can still be supplied directly.  Uses requests for Helix
API and twitchio for chat.  Stores config in
``~/.kiss/third_party_agents/twitch/config.json``.

Usage::

    agent = TwitchAgent()
    agent.run(prompt_template="Get stream info for channel 'shroud'")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
    save_json_config,
)
from kiss.agents.third_party_agents._device_auth import (
    ConsentSession,
    DeviceFlowProvider,
    DeviceFlowSession,
    TokenGrant,
    connect_prompt,
    consent_required,
)

_TWITCH_DIR = Path.home() / ".kiss" / "third_party_agents" / "twitch"
_HELIX_BASE = "https://api.twitch.tv/helix"
_DEFAULT_OAUTH_BASE = "https://id.twitch.tv"
# Scopes the chat, moderation and clip tools need; public data needs none.
_DEFAULT_SCOPES = (
    "user:read:chat user:write:chat user:bot channel:bot "
    "moderator:read:chatters moderator:manage:banned_users clips:edit"
)


def _device_provider() -> DeviceFlowProvider:
    """Return Twitch's device-code-grant endpoints, resolved per call.

    ``TWITCH_OAUTH_BASE`` lets tests point the flow (and the daemon-side
    token refresh, which only accepts the pinned host or loopback) at a
    loopback authorization server.  Twitch spells the scope field
    ``scopes`` and requires it in the token poll as well.

    Returns:
        The provider with ``/oauth2/device`` and ``/oauth2/token``.
    """
    base = os.environ.get("TWITCH_OAUTH_BASE", "") or _DEFAULT_OAUTH_BASE
    return DeviceFlowProvider(
        device_url=f"{base}/oauth2/device",
        token_url=f"{base}/oauth2/token",
        scope_param="scopes",
        token_scope_param="scopes",
    )


_config = ChannelConfig(
    _TWITCH_DIR,
    (
        "client_id",
        "access_token",
    ),
)


def _scrub_config_secrets() -> None:
    """Remove vault-migrated secrets from config.json.

    Finishes the Muse migration automatically: the ``access_token``
    lives in the vault and the unused ``client_secret`` must not linger
    in plaintext either; the non-secret ``client_id``/``channel_name``
    metadata is kept and the file is deleted when nothing else was
    stored.
    """
    try:
        cfg = json.loads(_config.path.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(cfg, dict) or not ("access_token" in cfg or "client_secret" in cfg):
        return
    kept = {k: str(v) for k, v in cfg.items() if k not in ("access_token", "client_secret") and v}
    if kept:
        save_json_config(_config.path, kept)
    else:
        _config.clear()


class TwitchChannelBackend(ToolMethodBackend):
    """Channel backend for Twitch Helix API."""

    def __init__(self, helix_base: str = _HELIX_BASE) -> None:
        self._helix_base: str = helix_base
        self._client_id: str = ""
        self._access_token: str = ""
        self._user_id: str = ""
        self._http: Any = requests
        self._muse: bool = False
        self._connection_info: str = ""

    def _headers(self) -> dict[str, str]:
        # In Muse mode ``_access_token`` holds a surrogate: the daemon
        # swaps this bearer for the real OAuth token at the network
        # boundary.  The Client-ID is not a secret and travels as-is.
        return {
            "Client-ID": self._client_id,
            "Authorization": f"Bearer {self._access_token}",
        }

    def _wire_muse(self) -> bool:
        """Acquire a Twitch surrogate and wire the boundary session.

        An ``access_token`` still in the legacy config is the newest
        user intent (initial migration, or a rotation done while Muse
        was off): it is enrolled as a bearer credential replacing any
        vault entry, and scrubbed from ``config.json`` (together with
        the unused ``client_secret``) only after the vault holds it.
        No network round trip happens here.

        Returns:
            True when the backend holds a surrogate and boundary session.
        """
        from kiss.agents.third_party_agents.muse_auth.client import (
            MuseBoundarySession,
            mint_surrogate,
            store_credentials,
        )

        cfg = _config.load_metadata() or {}
        token = cfg.get("access_token", "")
        if token:
            store_credentials("twitch", {"kind": "bearer", "token": token}, [])
        handle = mint_surrogate("twitch")
        if handle is None:
            self._connection_info = "No Twitch credential in the Muse vault or config."
            return False
        _scrub_config_secrets()
        self._client_id = cfg.get("client_id", "")
        self._access_token = handle.token
        self._http = MuseBoundarySession("twitch")
        self._muse = True
        return True

    def _get(self, path: str, params: dict | None = None) -> dict[str, Any]:  # type: ignore[type-arg]
        resp = self._http.get(
            f"{self._helix_base}{path}", headers=self._headers(), params=params, timeout=30
        )
        result: dict[str, Any] = resp.json() if resp.content else {}
        if resp.status_code >= 400:  # pragma: no branch
            result["ok"] = False
        return result

    def _post(self, path: str, json_body: dict | None = None) -> dict[str, Any]:  # type: ignore[type-arg]
        resp = self._http.post(
            f"{self._helix_base}{path}", headers=self._headers(), json=json_body, timeout=30
        )
        result: dict[str, Any] = resp.json() if resp.content else {"ok": True}
        if resp.status_code >= 400:  # pragma: no branch
            result["ok"] = False
        return result

    def connect(self) -> bool:
        """Authenticate with Twitch using stored config."""
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        if muse_auth_enabled():
            # Vault-first surrogate wiring; validation below runs the
            # /users read through the daemon boundary (audited).
            if not self._wire_muse():
                return False
        else:
            cfg = _config.load()
            if not cfg:  # pragma: no branch
                self._connection_info = "No Twitch config found."
                return False
            self._client_id = cfg["client_id"]
            self._access_token = cfg["access_token"]
        try:
            result = self._get("/users")
            if "data" in result:  # pragma: no branch
                users = result["data"]
                name = users[0].get("login", "") if users else ""
                self._user_id = users[0].get("id", "") if users else ""
                self._connection_info = f"Authenticated as {name}"
                return True
            self._connection_info = f"Twitch auth failed: {result}"
            return False
        except Exception as e:
            self._connection_info = f"Twitch connection failed: {e}"
            return False

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll for Twitch events (basic REST polling)."""
        return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a Twitch chat message."""
        if not self._user_id:
            users = self._get("/users").get("data", [])
            self._user_id = users[0].get("id", "") if users else ""
        result = self._post(
            "/chat/messages",
            {"broadcaster_id": channel_id, "sender_id": self._user_id, "message": text},
        )
        if result.get("ok") is False:  # pragma: no branch
            raise RuntimeError(f"Twitch send_message failed: {result}")

    def get_stream_info(self, broadcaster_login: str) -> str:
        """Get live stream information for a Twitch channel.

        Args:
            broadcaster_login: Twitch channel username.

        Returns:
            JSON string with stream info (game, title, viewer count, etc).
        """
        try:
            result = self._get("/streams", params={"user_login": broadcaster_login})
            streams = result.get("data", [])
            if not streams:  # pragma: no branch
                return json.dumps({"ok": True, "live": False, "channel": broadcaster_login})
            stream = streams[0]
            return json.dumps(
                {
                    "ok": True,
                    "live": True,
                    "title": stream.get("title", ""),
                    "game_name": stream.get("game_name", ""),
                    "viewer_count": stream.get("viewer_count", 0),
                    "started_at": stream.get("started_at", ""),
                    "language": stream.get("language", ""),
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_channel_info(self, broadcaster_id: str) -> str:
        """Get channel information for a Twitch broadcaster.

        Args:
            broadcaster_id: Twitch broadcaster ID.

        Returns:
            JSON string with channel info.
        """
        try:
            result = self._get("/channels", params={"broadcaster_id": broadcaster_id})
            channels = result.get("data", [])
            if not channels:  # pragma: no branch
                return json.dumps({"ok": False, "error": "Channel not found"})
            return json.dumps({"ok": True, "channel": channels[0]}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_user_info(self, login_or_id: str) -> str:
        """Get Twitch user information.

        Args:
            login_or_id: Twitch username (login) or user ID.

        Returns:
            JSON string with user info.
        """
        try:
            if login_or_id.isdigit():  # pragma: no branch
                result = self._get("/users", params={"id": login_or_id})
            else:
                result = self._get("/users", params={"login": login_or_id})
            users = result.get("data", [])
            if not users:  # pragma: no branch
                return json.dumps({"ok": False, "error": "User not found"})
            return json.dumps({"ok": True, "user": users[0]}, indent=2)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_chatters(self, broadcaster_id: str, moderator_id: str = "") -> str:
        """Get current chatters in a Twitch channel.

        Args:
            broadcaster_id: Broadcaster user ID.
            moderator_id: Moderator user ID (optional, defaults to broadcaster).

        Returns:
            JSON string with chatters list.
        """
        try:
            params: dict[str, str] = {"broadcaster_id": broadcaster_id}
            if moderator_id:  # pragma: no branch
                params["moderator_id"] = moderator_id
            else:
                params["moderator_id"] = broadcaster_id
            result = self._get("/chat/chatters", params=params)
            return json.dumps({"ok": True, **result}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_chat_message(self, broadcaster_id: str, sender_id: str, message: str) -> str:
        """Send a message to a Twitch chat.

        Args:
            broadcaster_id: Broadcaster channel ID.
            sender_id: Sender user ID.
            message: Message text.

        Returns:
            JSON string with ok status.
        """
        try:
            result = self._post(
                "/chat/messages",
                {
                    "broadcaster_id": broadcaster_id,
                    "sender_id": sender_id,
                    "message": message,
                },
            )
            return json.dumps({"ok": True, **result})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def ban_user(
        self,
        broadcaster_id: str,
        moderator_id: str,
        user_id: str,
        duration: int = 0,
        reason: str = "",
    ) -> str:
        """Ban or timeout a Twitch user.

        Args:
            broadcaster_id: Broadcaster channel ID.
            moderator_id: Moderator user ID.
            user_id: User ID to ban.
            duration: Timeout duration in seconds (0 = permanent ban).
            reason: Optional ban reason.

        Returns:
            JSON string with ok status.
        """
        try:
            body: dict[str, Any] = {"user_id": user_id}
            if duration:  # pragma: no branch
                body["duration"] = duration
            if reason:  # pragma: no branch
                body["reason"] = reason
            result = self._post(
                f"/moderation/bans?broadcaster_id={broadcaster_id}&moderator_id={moderator_id}",
                {"data": body},
            )
            return json.dumps({"ok": True, **result})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def search_third_party_agents(self, query: str, limit: int = 10) -> str:
        """Search for Twitch channels by name.

        Args:
            query: Search query.
            limit: Maximum channels to return. Default: 10.

        Returns:
            JSON string with matching channels.
        """
        try:
            params = {"query": query, "first": limit}
            result = self._get("/search/channels", params=params)
            data = {"ok": True, "third_party_agents": result.get("data", [])}
            return json.dumps(data, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_clips(self, broadcaster_id: str, limit: int = 20) -> str:
        """Get clips from a Twitch channel.

        Args:
            broadcaster_id: Broadcaster ID.
            limit: Maximum clips to return. Default: 20.

        Returns:
            JSON string with clip list.
        """
        try:
            result = self._get("/clips", params={"broadcaster_id": broadcaster_id, "first": limit})
            return json.dumps({"ok": True, "clips": result.get("data", [])}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def create_clip(self, broadcaster_id: str, has_delay: bool = False) -> str:
        """Create a clip from a live stream.

        Args:
            broadcaster_id: Broadcaster ID.
            has_delay: Whether to add a 5-second delay. Default: False.

        Returns:
            JSON string with clip edit URL.
        """
        try:
            result = self._post(
                f"/clips?broadcaster_id={broadcaster_id}&has_delay={str(has_delay).lower()}"
            )
            return json.dumps({"ok": True, **result})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


def _muse_authenticate(
    backend: TwitchChannelBackend,
    client_id: str,
    credential: dict[str, Any],
    channel_name: str,
    login: str,
) -> str:
    """Enroll an already-validated Twitch credential into the Muse vault.

    The credential — a plain ``bearer`` access token the user supplied,
    or the ``oauth2_refresh_token`` pair a device-code sign-in produced
    — was checked against ``/users`` by the caller (:func:`_probe_token`)
    BEFORE this call, so nothing here can leave an invalid token
    enrolled and a rejected candidate never touches the previous
    enrollment.  The store atomically replaces any previous vault entry
    (the vault never holds a half-written one); only the non-secret
    ``client_id``/``channel_name`` metadata is written to
    ``config.json``.  The ``client_secret`` is never persisted in Muse
    mode: the connector does not use it, and an unused plaintext secret
    must not linger.  A daemon failure is reported without clearing the
    vault: whichever credential it holds at that point (the untouched
    old one or the just-validated new one) is worth keeping.

    Args:
        backend: The agent's Twitch backend to (re)wire.
        client_id: Twitch app client ID (not a secret).
        credential: The vault ``authorized_user_info`` payload.
        channel_name: Optional default channel metadata.
        login: The account login the probe reported.

    Returns:
        JSON string with the result.
    """
    from kiss.agents.third_party_agents.muse_auth.client import store_credentials

    try:
        meta = {k: v for k, v in (("client_id", client_id), ("channel_name", channel_name)) if v}
        save_json_config(_config.path, meta)
        store_credentials("twitch", credential, [])
        if not backend._wire_muse():  # pragma: no cover - credential was just stored
            return json.dumps({"ok": False, "error": backend._connection_info})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})
    return json.dumps(
        {"ok": True, "message": "Twitch credentials saved (Muse-auth).", "login": login}
    )


def _probe_token(helix_base: str, client_id: str, access_token: str) -> tuple[str | None, str]:
    """Validate a freshly issued token with a direct ``GET /users``.

    Runs BEFORE the token replaces any stored credential, so a token
    Twitch rejects leaves the previous configuration untouched.  The
    request bypasses ambient proxy/netrc settings and never follows a
    redirect (the header carries the token).

    Args:
        helix_base: The Helix API base URL.
        client_id: The app's public client ID.
        access_token: The access token to validate.

    Returns:
        ``(login, "")`` on success (``login`` may be ``""`` for an app
        token), or ``(None, error)`` where *error* never contains the
        token.
    """
    session = requests.Session()
    session.trust_env = False
    try:
        with session:
            resp = session.get(
                f"{helix_base}/users",
                headers={"Client-ID": client_id, "Authorization": f"Bearer {access_token}"},
                timeout=30,
                allow_redirects=False,
            )
    except Exception as e:
        return None, f"{type(e).__name__} while validating the token"
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}"
    try:
        data = resp.json()
    except ValueError:
        return None, "non-JSON answer from /users"
    users = data.get("data") if isinstance(data, dict) else None
    if not isinstance(users, list):
        return None, "unexpected /users answer"
    return (str(users[0].get("login", "")) if users else ""), ""


def _legacy_authenticate(
    backend: TwitchChannelBackend,
    client_id: str,
    access_token: str,
    channel_name: str,
    client_secret: str = "",
) -> str:
    """Validate a Twitch access token directly and write ``config.json``.

    Used when Muse-auth is switched off.  Device-code tokens are stored
    as plain access tokens here (no refresh); they last about four
    hours, after which ``authenticate_twitch`` must be run again.

    Args:
        backend: The agent's Twitch backend to configure.
        client_id: Twitch app client ID.
        access_token: OAuth2 access token.
        channel_name: Optional default channel metadata.
        client_secret: Optional app secret kept for the legacy config.

    Returns:
        JSON string with the validation result.
    """
    backend._client_id = client_id
    backend._access_token = access_token
    try:
        result = backend._get("/users")
        if "data" in result:  # pragma: no branch
            _config.save(
                {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "access_token": access_token,
                    "channel_name": channel_name,
                }
            )
            return json.dumps(
                {
                    "ok": True,
                    "message": "Twitch credentials saved.",
                    "login": result["data"][0].get("login", "") if result["data"] else "",
                }
            )
        return json.dumps({"ok": False, "error": str(result)})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


class TwitchAgent(BaseChannelAgent):
    """Channel agent with Twitch Helix API tools."""

    channel_system_prompt = connect_prompt(
        "twitch",
        "Twitch",
        "authenticate_twitch(client_id=...) without an access_token",
        "The client_id is the public Client ID of an app registered at "
        "https://dev.twitch.tv/console/apps (client type Public; no secret); the "
        "returned twitch.tv/activate link already carries the code.",
    ).lstrip()

    def __init__(self) -> None:
        super().__init__("Twitch Agent")
        self._backend = TwitchChannelBackend()
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        if muse_auth_enabled():
            from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError

            # Muse-auth mode: wire a vault surrogate and the boundary
            # session (no network round trip); the real token never
            # enters this process once migrated.  A daemon failure
            # leaves the agent constructible (fail closed, tokenless)
            # so its authenticate/clear tools stay available.
            try:
                self._backend._wire_muse()
            except MuseAuthError as e:
                self._backend._access_token = ""
                self._backend._client_id = ""
                self._backend._connection_info = f"Muse-auth wiring failed: {e}"
            return
        cfg = _config.load()
        if cfg:  # pragma: no branch
            self._backend._client_id = cfg["client_id"]
            self._backend._access_token = cfg["access_token"]

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        if muse_auth_enabled():
            # In Muse mode only a wired boundary session counts: after a
            # failed enrollment rollback the backend holds no credential
            # and must not fall back to direct legacy requests.
            return self._backend._muse and bool(self._backend._access_token)
        return bool(self._backend._client_id)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_twitch_auth() -> str:
            """Check if Twitch credentials are configured and valid.

            Returns:
                Authentication status or instructions.
            """
            if not agent._is_authenticated():  # pragma: no branch
                return (
                    "Not authenticated with Twitch. Call "
                    "authenticate_twitch(client_id=...) to sign in the way the Muse "
                    "app connects: it returns a twitch.tv/activate link (code "
                    "pre-filled) for the user to open in their OWN browser, sign in "
                    "and authorize; then call finish_twitch_auth(). The client_id "
                    "is the public Client ID of an app registered at "
                    "https://dev.twitch.tv/console/apps (client type Public; no "
                    "secret is needed). Never ask for the user's Twitch password or "
                    "2FA code. Alternatively the user may hand you an access token "
                    "for authenticate_twitch(client_id=..., access_token=...)."
                )
            try:
                result = agent._backend._get("/users")
                if "data" in result:  # pragma: no branch
                    users = result["data"]
                    return json.dumps(
                        {
                            "ok": True,
                            "login": users[0].get("login", "") if users else "",
                        }
                    )
                return json.dumps({"ok": False, "error": str(result)})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_twitch(
            client_id: str,
            client_secret: str = "",
            access_token: str = "",
            channel_name: str = "",
            scopes: str = "",
        ) -> str:
            """Connect Twitch by browser sign-in (device code) or with a token.

            Without ``access_token`` this starts Twitch's device code grant
            for the public app ``client_id`` and returns a
            ``consent_required`` answer: give the user the activation URL
            (ask_user_question) to open in their OWN browser, where they
            sign in and authorize; then call finish_twitch_auth().  With
            ``access_token`` the token is validated and stored directly.

            Args:
                client_id: Twitch app client ID from the dev console (public).
                client_secret: Optional app secret (legacy config only; the
                    device flow and Muse-auth never use it).
                access_token: Optional OAuth2 user/app access token to store
                    directly instead of signing in.
                channel_name: Default channel to monitor. Optional.
                scopes: Space-separated scopes for the device flow (default
                    covers chat, moderation and clips).

            Returns:
                A consent_required JSON answer, a validation result, or an
                error message.
            """
            if not client_id.strip():  # pragma: no branch
                return "client_id cannot be empty."
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if access_token.strip():
                # A hand-supplied token supersedes any browser sign-in
                # still pending; drop it so a late approval cannot
                # overwrite this credential.
                ConsentSession.cancel_active("twitch")
                if muse_auth_enabled():
                    login, error = _probe_token(
                        agent._backend._helix_base, client_id.strip(), access_token.strip()
                    )
                    if error:
                        return json.dumps(
                            {"ok": False, "error": f"Twitch rejected the token: {error}"}
                        )
                    return _muse_authenticate(
                        agent._backend,
                        client_id.strip(),
                        {"kind": "bearer", "token": access_token.strip()},
                        channel_name.strip(),
                        login or "",
                    )
                return _legacy_authenticate(
                    agent._backend,
                    client_id.strip(),
                    access_token.strip(),
                    channel_name.strip(),
                    client_secret.strip(),
                )
            try:
                session = DeviceFlowSession(
                    "twitch",
                    _device_provider(),
                    client_id.strip(),
                    scopes.strip() or _DEFAULT_SCOPES,
                )
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})
            # The metadata the finish step needs rides on the session; the
            # stored configuration is untouched until the sign-in lands.
            session.options["channel_name"] = channel_name.strip()
            session.register()
            return json.dumps(consent_required("twitch", "Twitch", session))

        def finish_twitch_auth() -> str:
            """Complete a browser sign-in started by authenticate_twitch().

            Call after the user reports that they authorized the app; the
            token pair Twitch issued is validated with a `/users` read and
            stored (Muse vault when enabled, where the daemon refreshes it
            with the public client ID; no secret involved).

            Returns:
                The validation result, a pending status while the user has
                not authorized yet, or an error message.
            """
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            session, status = ConsentSession.finish("twitch")
            if status == "pending":
                return json.dumps(
                    {
                        "ok": False,
                        "status": "pending",
                        "error": "The user has not authorized yet; ask them to finish "
                        "the sign-in, then call this tool again.",
                    }
                )
            if not isinstance(session, DeviceFlowSession) or session.result is None:
                return json.dumps({"ok": False, "error": f"Twitch sign-in failed: {status}"})
            grant = TokenGrant.from_session(session)
            channel_name = str(session.options.get("channel_name", ""))
            # Validate first: a rejected token must not disturb the
            # credential that is currently in use.
            login, error = _probe_token(
                agent._backend._helix_base, session.client_id, grant.access_token
            )
            if error:
                return json.dumps({"ok": False, "error": f"Twitch rejected the new token: {error}"})
            if muse_auth_enabled():
                return _muse_authenticate(
                    agent._backend,
                    session.client_id,
                    grant.vault_credential(session.provider.token_url, session.client_id),
                    channel_name,
                    login or "",
                )
            return _legacy_authenticate(
                agent._backend, session.client_id, grant.access_token, channel_name
            )

        def clear_twitch_auth() -> str:
            """Clear the stored Twitch credentials.

            Returns:
                Status message.
            """
            ConsentSession.cancel_active("twitch")
            _config.clear()
            agent._backend._client_id = ""
            agent._backend._access_token = ""
            agent._backend._http = requests
            agent._backend._muse = False
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if muse_auth_enabled():
                from kiss.agents.third_party_agents.muse_auth.client import clear_credentials

                clear_credentials("twitch")
            return "Twitch authentication cleared."

        return [check_twitch_auth, authenticate_twitch, finish_twitch_auth, clear_twitch_auth]


def main() -> None:
    """Run the TwitchAgent from the command line with chat persistence."""
    channel_main(TwitchAgent, "kiss-twitch")


def tools() -> list:
    """Return the Twitch channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return TwitchAgent()._get_tools()


if __name__ == "__main__":
    main()
