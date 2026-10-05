# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Discord Agent — channel agent with Discord REST API tools.

Uses the Discord REST API v10 directly via requests (no discord.py).

Primary sign-in is click-Allow: ``authenticate_discord()`` runs the
OAuth authorization-code grant with PKCE against the KISS-owned public
Discord app (no client secret) with scopes ``identify guilds
webhook.incoming``.  The user signs in, picks a server and a channel,
and clicks Authorize; ``finish_discord_auth()`` stores the resulting
user token and the incoming webhook Discord created for that channel.
A user token can read the user's profile and server list but, by
Discord's rules, cannot read or send channel messages, so messages go
to the authorized channel through the webhook.  Bot-only features
(reading messages, channel poll mode, managing messages) still need a
bot token passed to ``authenticate_discord(bot_token=...)``, because
Discord offers no OAuth equivalent for them.

Storage.  Exactly one Discord credential is active at a time:

* Muse-auth mode (the default): the credential lives in the Muse vault
  under service ``discord`` (the only name the daemon lets reach
  ``discord.com`` and refresh against ``discord.com/api/oauth2/token``).
  A user sign-in is an ``oauth2_refresh_token`` entry the daemon
  refreshes itself; a bot token is a header-kind entry occupying the
  ``Authorization`` header (``Bot <token>``).  Either way this process
  holds only a surrogate bearer.  ``config.json`` keeps non-secret
  metadata only; ``auth_mode: user`` marks a user sign-in.
* Legacy mode: ``~/.kiss/third_party_agents/discord/config.json`` holds
  ``bot_token``, or ``access_token`` plus ``auth_mode: user``.

The webhook URL embeds its own secret token, so it is kept like a
credential in ``~/.kiss/third_party_agents/discord/webhook/config.json``
(mode 0600) together with its channel and server IDs, and is never
printed.

Usage::

    agent = DiscordAgent()
    agent.run(prompt_template="List all channels in my server")
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
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
    LoopbackPkceSession,
    PkceProvider,
    TokenGrant,
    connect_prompt,
    consent_required,
)
from kiss.agents.third_party_agents._oauth_apps import (
    LOOPBACK_REDIRECT_URI,
    missing_client_id_error,
    oauth_client_id,
)
from kiss.core.config import kiss_home

_DISCORD_DIR = kiss_home() / "third_party_agents" / "discord"
_API_BASE = "https://discord.com/api/v10"
_OAUTH_SCOPES = "identify guilds webhook.incoming"
_config = ChannelConfig(_DISCORD_DIR, ("bot_token",))
_webhook_config = ChannelConfig(_DISCORD_DIR / "webhook", ("url",))
# Config keys holding real secrets (scrubbed once the vault has them).
_SECRET_KEYS = ("bot_token", "access_token", "refresh_token")

_BOT_ONLY_ERROR = json.dumps(
    {
        "ok": False,
        "error": "This needs a Discord bot token: the signed-in user token cannot "
        "read channels or manage messages (Discord bans self-bots). Ask the user "
        "for a bot token and call authenticate_discord(bot_token=...).",
    }
)


def description() -> str:
    """Return the one-sentence help text shown by ``/discord help``."""
    return (
        "Channel agent for Discord that signs in with click-Allow OAuth (or a bot token for "
        "reading, polling and managing messages), posts to the authorized channel through its "
        "webhook and lists servers and channels via the REST API v10; use "
        "`run_agent(agent=\"discord\", task=...)` or the `kiss-discord` CLI."
    )


def _pkce_provider() -> PkceProvider:
    """Return Discord's OAuth endpoints (``$DISCORD_OAUTH_BASE`` overrides the host)."""
    base = os.environ.get("DISCORD_OAUTH_BASE", "https://discord.com").rstrip("/")
    return PkceProvider(f"{base}/oauth2/authorize", f"{base}/api/oauth2/token")


_REFRESH_MARGIN = 300.0


def _legacy_vault_credential(cfg: dict[str, str]) -> dict[str, Any]:
    """Map a legacy user sign-in config onto its Muse vault credential.

    Args:
        cfg: The stored config dict (``access_token`` and, for a
            refreshable sign-in, ``refresh_token``/``expires_at``/``client_id``).

    Returns:
        An ``oauth2_refresh_token`` credential when a refresh token is
        stored (the daemon keeps refreshing it), else a plain ``bearer``.
    """
    if not cfg.get("refresh_token"):
        return {"kind": "bearer", "token": cfg["access_token"]}
    return {
        "kind": "oauth2_refresh_token",
        "token_url": _pkce_provider().token_url,
        "client_id": cfg.get("client_id", ""),
        "access_token": cfg["access_token"],
        "refresh_token": cfg["refresh_token"],
        "expires_at": float(cfg.get("expires_at") or 0),
    }


def _refresh_legacy_user_token(cfg: dict[str, str]) -> dict[str, str]:
    """Rotate a legacy-mode user token that is about to expire and save it.

    Args:
        cfg: The stored config dict.

    Returns:
        *cfg* itself when no refresh is due or possible, else the
        updated dict (unchanged when Discord refuses the refresh; the
        old token is then reported as expired by the next API call).
    """
    expires_at = float(cfg.get("expires_at") or 0)
    if not cfg.get("refresh_token") or expires_at - _REFRESH_MARGIN > time.time():
        return cfg
    form = {
        "grant_type": "refresh_token",
        "refresh_token": cfg["refresh_token"],
        "client_id": cfg.get("client_id", ""),
    }
    try:
        data = requests.post(_pkce_provider().token_url, data=form, timeout=30).json()
    except (requests.RequestException, ValueError):
        return cfg
    if not isinstance(data, dict) or not data.get("access_token"):
        return cfg
    grant = TokenGrant.from_response(data)
    new_cfg = {
        **cfg,
        "access_token": grant.access_token,
        "refresh_token": grant.refresh_token or cfg["refresh_token"],
        "expires_at": str(grant.acquired_at + (grant.expires_in or 3600.0)),
    }
    _config.save(new_cfg)
    return new_cfg


def _scrub_config_token() -> None:
    """Remove vault-migrated secrets (``bot_token``/``access_token``) from config.json.

    Finishes the Muse migration automatically: the non-secret metadata
    (``application_id``, ``guild_ids``, ``auth_mode``) is kept and the
    file is deleted when nothing but the token was stored.
    """
    try:
        cfg = json.loads(_config.path.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(cfg, dict) or not any(k in cfg for k in _SECRET_KEYS):
        return
    # A migrated bot token supersedes an older user sign-in marker.
    drop: tuple[str, ...] = _SECRET_KEYS + ("expires_at", "client_id")
    if "bot_token" in cfg:
        drop += ("auth_mode",)
    kept = {k: str(v) for k, v in cfg.items() if k not in drop and v}
    if kept:
        save_json_config(_config.path, kept)
    else:
        _config.clear()


def _snowflake_key(msg: dict) -> int:  # type: ignore[type-arg]
    """Return the numeric value of a message's snowflake id for sorting."""
    try:
        return int(msg.get("id", "0"))
    except ValueError:
        return 0


def _probe_user(api_base: str, access_token: str) -> dict[str, Any]:
    """Read ``/users/@me`` with a freshly issued user token.

    Runs before the token is stored so a rejected sign-in never
    replaces the credential in use.  Ambient proxy settings are ignored
    because the request carries the new token.

    Args:
        api_base: Discord API base URL.
        access_token: The OAuth user access token.

    Returns:
        The decoded user object, or the error body when ``id`` is missing.

    Raises:
        requests.RequestException: On a transport failure.
    """
    with requests.Session() as session:
        session.trust_env = False
        resp = session.get(
            f"{api_base}/users/@me",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30,
        )
    try:
        body = resp.json()
    except ValueError:
        body = {"status": resp.status_code}
    return body if isinstance(body, dict) else {"status": resp.status_code}


class DiscordChannelBackend(ToolMethodBackend):
    """Channel backend for Discord REST API v10."""

    def __init__(self, api_base: str = "") -> None:
        self._api_base = api_base or os.environ.get("DISCORD_API_BASE", _API_BASE)
        # The credential (or its Muse surrogate): a bot token, or a
        # user OAuth token when ``_user_auth`` is set.
        self._token: str = ""
        self._user_auth: bool = False
        self._bot_user_id: str = ""
        self._http: Any = requests
        self._muse: bool = False
        self._connection_info: str = ""
        self._last_message_id: str = ""

    def _headers(self) -> dict[str, str]:
        if self._muse or self._user_auth:
            # In Muse mode ``_token`` is a surrogate the daemon swaps at
            # the network boundary for the real ``Bearer`` user token or
            # (header-kind entry) the real ``Authorization: Bot ...``.
            return {"Authorization": f"Bearer {self._token}"}
        return {"Authorization": f"Bot {self._token}"}

    def _load_legacy_config(self) -> bool:
        """Load the legacy ``config.json`` credential into the backend.

        Returns:
            True when a bot token or a user access token was found.
        """
        cfg = _config.load_metadata() or {}
        if cfg.get("bot_token"):
            self._token, self._user_auth = cfg["bot_token"], False
            return True
        if cfg.get("access_token"):
            cfg = _refresh_legacy_user_token(cfg)
            self._token, self._user_auth = cfg["access_token"], True
            return True
        return False

    def _wire_muse(self) -> bool:
        """Acquire a Discord surrogate and wire the boundary session.

        A secret still in the legacy config is the newest user intent
        (initial migration, or a rotation done while Muse was off): a
        ``bot_token`` is enrolled as a header-kind credential
        (``Authorization: Bot <token>``), an ``access_token`` from a
        legacy user sign-in as a bearer; either replaces any vault
        entry and is scrubbed from ``config.json`` only after the vault
        holds it.  No network round trip happens here.

        Returns:
            True when the backend holds a surrogate and boundary session.
        """
        from kiss.agents.third_party_agents.muse_auth.client import (
            MuseBoundarySession,
            mint_surrogate,
            store_credentials,
        )

        cfg = _config.load_metadata() or {}
        if cfg.get("bot_token"):
            store_credentials(
                "discord",
                {"kind": "header", "header": "Authorization", "token": f"Bot {cfg['bot_token']}"},
                [],
            )
        elif cfg.get("access_token"):
            store_credentials("discord", _legacy_vault_credential(cfg), [])
        handle = mint_surrogate("discord")
        if handle is None:
            self._connection_info = "No Discord credential in the Muse vault or config."
            return False
        _scrub_config_token()
        self._token = handle.token
        self._user_auth = not cfg.get("bot_token") and cfg.get("auth_mode") == "user"
        self._http = MuseBoundarySession("discord")
        self._muse = True
        return True

    def _get(self, path: str, params: dict | None = None) -> Any:  # type: ignore[type-arg]
        resp = self._http.get(
            f"{self._api_base}{path}", headers=self._headers(), params=params, timeout=30
        )
        return resp.json()

    def _post(  # type: ignore[type-arg]
        self, path: str, json_body: dict | None = None, raise_on_error: bool = False
    ) -> Any:
        """POST *json_body* to the Discord API and return the parsed JSON.

        Args:
            path: API path appended to the api base (e.g. ``/channels/1/messages``).
            json_body: Optional JSON payload for the request body.
            raise_on_error: When True, raise ``requests.HTTPError`` on any
                non-2xx response instead of returning the error body. Transport
                failures (e.g. unreachable server) always raise. Default False
                preserves the tool-facing callers' return-the-error-body
                contract.

        Returns:
            The decoded JSON response body.
        """
        resp = self._http.post(
            f"{self._api_base}{path}", headers=self._headers(), json=json_body, timeout=30
        )
        if raise_on_error:
            resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str) -> Any:  # type: ignore[type-arg]
        resp = self._http.delete(f"{self._api_base}{path}", headers=self._headers(), timeout=30)
        if resp.status_code == 204:  # pragma: no branch
            return {"ok": True}
        return resp.json()

    def _patch(self, path: str, json_body: dict | None = None) -> Any:  # type: ignore[type-arg]
        resp = self._http.patch(
            f"{self._api_base}{path}", headers=self._headers(), json=json_body, timeout=30
        )
        return resp.json()

    def connect(self) -> bool:
        """Authenticate with Discord using the stored bot or user token."""
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        if muse_auth_enabled():
            # Vault-first surrogate wiring; validation below runs the
            # /users/@me read through the daemon boundary (audited).
            if not self._wire_muse():
                return False
        elif not self._load_legacy_config():
            self._connection_info = "No Discord token found."
            return False
        try:
            result = self._get("/users/@me")
            if "id" in result:  # pragma: no branch
                self._bot_user_id = str(result["id"])
                username = result.get("username", "")
                discriminator = result.get("discriminator", "")
                kind = "user sign-in" if self._user_auth else "bot"
                self._connection_info = f"Authenticated as {username}#{discriminator} ({kind})"
                return True
            self._connection_info = f"Discord auth failed: {result}"
            return False
        except Exception as e:
            self._connection_info = f"Discord auth failed: {e}"
            return False

    def find_channel(self, name: str) -> str | None:
        """Find a channel by name or numeric ID.

        If *name* is already a numeric snowflake ID, returns it as-is.
        Otherwise queries all guilds for a channel matching the name.

        Args:
            name: Channel name or numeric ID.

        Returns:
            The channel snowflake ID string, or None if not found.
        """
        if not name:
            return None
        if name.isdigit():
            return name
        if self._user_auth:
            return None
        try:
            guilds = self._get("/users/@me/guilds", params={"limit": 100})
            if not isinstance(guilds, list):
                return None
            for guild in guilds:
                channels = self._get(f"/guilds/{guild['id']}/channels")
                if not isinstance(channels, list):
                    continue
                for ch in channels:
                    if ch.get("name") == name:
                        return str(ch["id"])
        except Exception:
            pass
        return None

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll for new Discord messages using REST API."""
        if not channel_id:  # pragma: no branch
            return [], oldest
        try:
            params: dict[str, Any] = {"limit": limit}
            if oldest and oldest != "0":
                params["after"] = oldest
            else:
                params["after"] = str((int((time.time() - 1) * 1000) - 1420070400000) << 22)
            result = self._get(f"/channels/{channel_id}/messages", params=params)
            if not isinstance(result, list):  # pragma: no branch
                return [], oldest
            msgs: list[dict[str, Any]] = sorted(result, key=_snowflake_key)
            new_oldest = oldest
            messages = []
            for m in msgs:  # pragma: no branch
                new_oldest = m["id"]
                messages.append(
                    {
                        "ts": m.get("id", ""),
                        "timestamp": m.get("timestamp", ""),
                        "user": m.get("author", {}).get("id", ""),
                        "text": m.get("content", ""),
                        "id": m.get("id", ""),
                    }
                )
            return messages, new_oldest
        except Exception:
            return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a Discord message, optionally as a reply to *thread_ts*.

        Raises on transport failure and on non-2xx API responses so the
        ChannelRunner's delivery ledger sees failed sends and can retry
        them instead of silently marking them delivered.

        Args:
            channel_id: Channel snowflake ID to post into.
            text: Message content.
            thread_ts: Optional message ID to reply to.

        Raises:
            requests.HTTPError: On a non-2xx Discord API response.
            requests.RequestException: On transport failure (e.g. the
                server is unreachable).
        """
        body: dict[str, Any] = {"content": text}
        if thread_ts:
            body["message_reference"] = {"message_id": thread_ts}
        self._post(f"/channels/{channel_id}/messages", body, raise_on_error=True)

    def send_typing(self, channel_id: str, thread_ts: str = "") -> None:
        """Show a typing indicator in a Discord channel (best-effort).

        POSTs the Discord ``/channels/{channel_id}/typing`` REST endpoint,
        which displays "bot is typing…" for a few seconds. Any network or
        API error (including non-2xx responses and unreachable servers) is
        swallowed so a failed indicator can never break message handling.

        Args:
            channel_id: Channel snowflake ID to show the indicator in.
            thread_ts: Reply-target message ID, accepted for interface
                parity with :meth:`send_message`. Ignored here because
                ``send_message`` treats thread ids as message references
                inside *channel_id* (not as separate channels), so the
                typing indicator belongs to *channel_id* as well.
        """
        del thread_ts
        try:
            self._http.post(
                f"{self._api_base}/channels/{channel_id}/typing",
                headers=self._headers(),
                timeout=30,
            )
        except Exception:
            pass

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Return True if *msg* was sent by this bot's own user.

        Args:
            msg: Message dict from :meth:`poll_messages`.

        Returns:
            Whether the message author is the bot itself.
        """
        return bool(self._bot_user_id) and msg.get("user") == self._bot_user_id

    def disconnect(self) -> None:
        """Release Discord backend state before stop or reconnect."""
        self._last_message_id = ""

    def list_guilds(self, limit: int = 100) -> str:
        """List guilds (servers) the bot or signed-in user is a member of.

        Args:
            limit: Maximum guilds to return (1-200). Default: 100.

        Returns:
            JSON string with guild list (id, name, icon).
        """
        try:
            result = self._get("/users/@me/guilds", params={"limit": min(limit, 200)})
            if isinstance(result, list):  # pragma: no branch
                guilds = [{"id": g["id"], "name": g.get("name", "")} for g in result]
                return json.dumps({"ok": True, "guilds": guilds}, indent=2)[:8000]
            return json.dumps({"ok": False, "error": str(result)})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_third_party_agents(self, guild_id: str, channel_type: str = "") -> str:
        """List channels in a guild.

        Args:
            guild_id: Guild (server) ID.
            channel_type: Optional filter by type (0=text, 2=voice, 4=category).

        Returns:
            JSON string with channel list (id, name, type, topic).
        """
        if self._user_auth:
            return _BOT_ONLY_ERROR
        try:
            result = self._get(f"/guilds/{guild_id}/channels")
            if not isinstance(result, list):  # pragma: no branch
                return json.dumps({"ok": False, "error": str(result)})
            third_party_agents = [
                {
                    "id": c["id"],
                    "name": c.get("name", ""),
                    "type": c.get("type", 0),
                    "topic": c.get("topic", ""),
                    "position": c.get("position", 0),
                }
                for c in result
                if not channel_type or str(c.get("type", "")) == channel_type
            ]
            payload = {"ok": True, "third_party_agents": third_party_agents}
            return json.dumps(payload, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_channel(self, channel_id: str) -> str:
        """Get information about a channel.

        Args:
            channel_id: Channel ID.

        Returns:
            JSON string with channel details.
        """
        if self._user_auth:
            return _BOT_ONLY_ERROR
        try:
            result = self._get(f"/channels/{channel_id}")
            if "id" not in result:  # pragma: no branch
                return json.dumps({"ok": False, "error": str(result)})
            return json.dumps({"ok": True, **result}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_channel_messages(
        self,
        channel_id: str,
        limit: int = 50,
        before: str = "",
        after: str = "",
    ) -> str:
        """Get messages from a channel.

        Args:
            channel_id: Channel ID.
            limit: Number of messages (1-100). Default: 50.
            before: Get messages before this message ID.
            after: Get messages after this message ID.

        Returns:
            JSON string with message list.
        """
        if self._user_auth:
            return _BOT_ONLY_ERROR
        try:
            params: dict[str, Any] = {"limit": min(limit, 100)}
            if before:  # pragma: no branch
                params["before"] = before
            if after:  # pragma: no branch
                params["after"] = after
            result = self._get(f"/channels/{channel_id}/messages", params=params)
            if not isinstance(result, list):  # pragma: no branch
                return json.dumps({"ok": False, "error": str(result)})
            messages = [
                {
                    "id": m["id"],
                    "author": m.get("author", {}).get("username", ""),
                    "content": m.get("content", ""),
                    "timestamp": m.get("timestamp", ""),
                }
                for m in result
            ]
            return json.dumps({"ok": True, "messages": messages}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def post_message(
        self,
        channel_id: str,
        content: str,
        tts: bool = False,
        reply_to: str = "",
    ) -> str:
        """Send a message to a Discord channel.

        With a bot token the message is posted through the API to any
        channel the bot can see.  With a user sign-in it goes through
        the incoming webhook created at sign-in, so only the channel the
        user authorized is reachable and replies are not possible.

        Args:
            channel_id: Channel ID.
            content: Message text (up to 2000 chars).
            tts: Text-to-speech flag. Default: False.
            reply_to: Optional message ID to reply to (bot token only).

        Returns:
            JSON string with ok status and message id.
        """
        if self._user_auth:
            return _post_webhook(channel_id, content, tts, reply_to)
        try:
            body: dict[str, Any] = {"content": content, "tts": tts}
            if reply_to:  # pragma: no branch
                body["message_reference"] = {"message_id": reply_to}
            result = self._post(f"/channels/{channel_id}/messages", body)
            if "id" not in result:  # pragma: no branch
                return json.dumps({"ok": False, "error": str(result)})
            return json.dumps({"ok": True, "id": result["id"]})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def edit_message(self, channel_id: str, message_id: str, content: str) -> str:
        """Edit an existing Discord message.

        Args:
            channel_id: Channel ID.
            message_id: Message ID.
            content: New content.

        Returns:
            JSON string with ok status.
        """
        if self._user_auth:
            return _BOT_ONLY_ERROR
        try:
            result = self._patch(
                f"/channels/{channel_id}/messages/{message_id}", {"content": content}
            )
            if "id" not in result:  # pragma: no branch
                return json.dumps({"ok": False, "error": str(result)})
            return json.dumps({"ok": True, "id": result["id"]})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def delete_message(self, channel_id: str, message_id: str) -> str:
        """Delete a Discord message.

        Args:
            channel_id: Channel ID.
            message_id: Message ID to delete.

        Returns:
            JSON string with ok status.
        """
        if self._user_auth:
            return _BOT_ONLY_ERROR
        try:
            result = self._delete(f"/channels/{channel_id}/messages/{message_id}")
            if isinstance(result, dict) and result.get("ok") is True:
                return json.dumps({"ok": True})
            return json.dumps({"ok": False, "error": str(result)})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def add_reaction(self, channel_id: str, message_id: str, emoji: str) -> str:
        """Add a reaction to a message.

        Args:
            channel_id: Channel ID.
            message_id: Message ID.
            emoji: Emoji (e.g. "👍" or "name:id" for custom emojis).

        Returns:
            JSON string with ok status.
        """
        if self._user_auth:
            return _BOT_ONLY_ERROR
        try:
            from urllib.parse import quote

            emoji_url = f"{self._api_base}/channels/{channel_id}/messages/{message_id}"
            emoji_url += f"/reactions/{quote(emoji)}/@me"
            resp = self._http.put(emoji_url, headers=self._headers(), timeout=30)
            return json.dumps({"ok": resp.status_code == 204})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def create_thread(
        self,
        channel_id: str,
        message_id: str,
        name: str,
        auto_archive_duration: int = 1440,
    ) -> str:
        """Create a thread from a message.

        Args:
            channel_id: Channel ID.
            message_id: Message ID to create thread from.
            name: Thread name.
            auto_archive_duration: Minutes before auto-archive (60/1440/4320/10080).

        Returns:
            JSON string with thread id and name.
        """
        if self._user_auth:
            return _BOT_ONLY_ERROR
        try:
            result = self._post(
                f"/channels/{channel_id}/messages/{message_id}/threads",
                {"name": name, "auto_archive_duration": auto_archive_duration},
            )
            if "id" not in result:  # pragma: no branch
                return json.dumps({"ok": False, "error": str(result)})
            return json.dumps({"ok": True, "id": result["id"], "name": result.get("name", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_guild_members(self, guild_id: str, limit: int = 100, after: str = "") -> str:
        """List members of a guild.

        Args:
            guild_id: Guild ID.
            limit: Max members to return (1-1000). Default: 100.
            after: User ID to start after (for pagination).

        Returns:
            JSON string with member list.
        """
        if self._user_auth:
            return _BOT_ONLY_ERROR
        try:
            params: dict[str, Any] = {"limit": min(limit, 1000)}
            if after:  # pragma: no branch
                params["after"] = after
            result = self._get(f"/guilds/{guild_id}/members", params=params)
            if not isinstance(result, list):  # pragma: no branch
                return json.dumps({"ok": False, "error": str(result)})
            members = [
                {
                    "id": m.get("user", {}).get("id", ""),
                    "username": m.get("user", {}).get("username", ""),
                    "nick": m.get("nick", ""),
                    "roles": m.get("roles", []),
                }
                for m in result
            ]
            return json.dumps({"ok": True, "members": members}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def create_invite(self, channel_id: str, max_age: int = 86400, max_uses: int = 0) -> str:
        """Create an invite link for a channel.

        Args:
            channel_id: Channel ID.
            max_age: Invite expiry in seconds (0 = never). Default: 86400 (1 day).
            max_uses: Maximum uses (0 = unlimited). Default: 0.

        Returns:
            JSON string with invite code and URL.
        """
        if self._user_auth:
            return _BOT_ONLY_ERROR
        try:
            result = self._post(
                f"/channels/{channel_id}/invites",
                {"max_age": max_age, "max_uses": max_uses},
            )
            if "code" not in result:  # pragma: no branch
                return json.dumps({"ok": False, "error": str(result)})
            return json.dumps(
                {
                    "ok": True,
                    "code": result["code"],
                    "url": f"https://discord.gg/{result['code']}",
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


def _post_webhook(channel_id: str, content: str, tts: bool, reply_to: str) -> str:
    """Post *content* through the webhook authorized at user sign-in.

    Args:
        channel_id: Target channel; must be the webhook's channel.
        content: Message text.
        tts: Text-to-speech flag.
        reply_to: Must be empty (webhooks cannot reply).

    Returns:
        JSON string with ok status and message id, or an error that
        never contains the webhook URL.
    """
    webhook = _webhook_config.load()
    if not webhook:
        return json.dumps(
            {
                "ok": False,
                "error": "No Discord webhook is stored; sign in again with "
                "authenticate_discord() and pick a channel.",
            }
        )
    if reply_to or channel_id != webhook.get("channel_id"):
        error = (
            f"The signed-in user can post only to channel {webhook.get('channel_id')} "
            "(the one authorized at sign-in) and cannot reply. Anything else needs a "
            "bot token: ask the user for one and call authenticate_discord(bot_token=...)."
        )
        return json.dumps({"ok": False, "error": error})
    try:
        resp = requests.post(
            webhook["url"],
            params={"wait": "true"},
            json={"content": content, "tts": tts},
            timeout=30,
        )
        result = resp.json() if resp.content else {}
    except (requests.RequestException, ValueError) as e:
        return json.dumps({"ok": False, "error": f"webhook post failed: {type(e).__name__}"})
    if resp.ok and isinstance(result, dict) and "id" in result:
        return json.dumps({"ok": True, "id": result["id"]})
    return json.dumps({"ok": False, "error": f"webhook post failed (HTTP {resp.status_code})"})


def _muse_authenticate(
    backend: DiscordChannelBackend, bot_token: str, application_id: str, guild_ids: str
) -> str:
    """Enroll a bot token into the Muse vault and validate it at the boundary.

    The plaintext token goes straight into the vault as a header-kind
    credential (``Authorization: Bot <token>``), atomically replacing
    any previous enrollment, and is never written to ``config.json``
    (only the non-secret metadata is).  Validation runs ``/users/@me``
    through the daemon boundary, so it is audited; an invalid token
    leaves the vault empty (the previous credential was already
    replaced by the user's explicit rotation).

    Args:
        backend: The agent's Discord backend to (re)wire.
        bot_token: Discord bot token (advanced path for bot-only features).
        application_id: Optional application ID metadata.
        guild_ids: Optional comma-separated guild ID metadata.

    Returns:
        JSON string with the validation result.
    """
    from kiss.agents.third_party_agents.muse_auth.client import (
        MuseBoundarySession,
        clear_credentials,
        mint_surrogate,
        store_credentials,
    )

    try:
        # Overwrite-store: replaces the old vault entry in one step (no
        # window where the vault is empty) and invalidates its surrogates.
        store_credentials(
            "discord",
            {"kind": "header", "header": "Authorization", "token": f"Bot {bot_token}"},
            [],
        )
        handle = mint_surrogate("discord")
        backend._token = handle.token if handle else ""
        backend._http = MuseBoundarySession("discord")
        backend._muse = True
        backend._user_auth = False
        result = backend._get("/users/@me")
        if "id" in result:
            meta = {
                k: v for k, v in (("application_id", application_id), ("guild_ids", guild_ids)) if v
            }
            # Never persist the token; also drop any stale plaintext
            # copy a pre-Muse config may still hold.
            if meta:
                save_json_config(_config.path, meta)
            else:
                _config.clear()
            return json.dumps(
                {
                    "ok": True,
                    "message": "Discord token saved and validated (Muse-auth).",
                    "username": result.get("username", ""),
                    "id": result.get("id", ""),
                }
            )
        error = json.dumps({"ok": False, "error": str(result)})
    except Exception as e:
        error = json.dumps({"ok": False, "error": str(e)})
    # Roll the vault back so a bad token is not left enrolled.
    with contextlib.suppress(Exception):
        clear_credentials("discord")
    backend._token = ""
    backend._http = requests
    backend._muse = False
    return error


def _store_user_grant(
    backend: DiscordChannelBackend, session: LoopbackPkceSession, grant: TokenGrant
) -> None:
    """Persist a validated user sign-in and wire *backend* to use it.

    Muse mode enrolls the grant in the vault under ``discord`` (the
    daemon refreshes it with the public client ID; no secret exists);
    legacy mode writes the access token to ``config.json``.  Either way
    this replaces a stored bot token.  The webhook Discord created for
    the chosen channel is saved to its own 0600 file.

    Args:
        backend: The agent's backend to (re)wire.
        session: The finished PKCE session (token URL and client ID).
        grant: The validated token grant.
    """
    from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

    if muse_auth_enabled():
        from kiss.agents.third_party_agents.muse_auth.client import (
            MuseBoundarySession,
            mint_surrogate,
            store_credentials,
        )

        store_credentials(
            "discord", grant.vault_credential(session.provider.token_url, session.client_id), []
        )
        _config.save({"auth_mode": "user"})
        handle = mint_surrogate("discord")
        backend._token = handle.token if handle else ""
        backend._http = MuseBoundarySession("discord")
        backend._muse = True
    else:
        cfg = {"auth_mode": "user", "access_token": grant.access_token}
        if grant.refresh_token:
            cfg.update(
                refresh_token=grant.refresh_token,
                expires_at=str(grant.acquired_at + (grant.expires_in or 3600.0)),
                client_id=session.client_id,
            )
        _config.save(cfg)
        backend._token = grant.access_token
    backend._user_auth = True
    webhook = grant.raw.get("webhook")
    if isinstance(webhook, dict) and webhook.get("url") and webhook.get("channel_id"):
        _webhook_config.save(
            {
                "url": str(webhook["url"]),
                "channel_id": str(webhook["channel_id"]),
                "guild_id": str(webhook.get("guild_id") or ""),
            }
        )
    else:
        # A new sign-in without a channel choice must not keep posting
        # through the previous account's webhook.
        _webhook_config.clear()


class DiscordAgent(BaseChannelAgent):
    """Channel agent with Discord REST API tools.

    Example::

        agent = DiscordAgent()
        result = agent.run(prompt_template="List all channels in my server")
    """

    channel_system_prompt = connect_prompt(
        "discord",
        "Discord",
        "authenticate_discord() with no arguments",
        "It asks Discord for the scopes identify, guilds and webhook.incoming, so "
        "on the approval page the user also picks the server and channel KISS may "
        "post into.",
    ).lstrip() + (
        "\nReading messages, channel poll mode and message management need a bot "
        "token because Discord offers no OAuth sign-in for them: only for those, ask "
        "the user for a bot token and call authenticate_discord(bot_token=...)."
    )

    def __init__(self) -> None:
        super().__init__("Discord Agent")
        self._backend = DiscordChannelBackend()
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        if muse_auth_enabled():
            from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError

            # Muse-auth mode: wire a vault surrogate and the boundary
            # session (no network round trip); the real token never
            # enters this process once migrated.  A daemon failure
            # leaves the agent constructible (fail closed, tokenless) so
            # its authenticate/clear tools stay available.
            try:
                self._backend._wire_muse()
            except MuseAuthError as e:
                self._backend._token = ""
                self._backend._connection_info = f"Muse-auth wiring failed: {e}"
            return
        self._backend._load_legacy_config()

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._token)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_discord_auth() -> str:
            """Check whether a Discord credential is configured and valid.

            Returns:
                JSON with the account, whether it is a user sign-in or a
                bot, and the authorized webhook channel; or instructions
                for how to authenticate.
            """
            if not agent._backend._token:  # pragma: no branch
                return (
                    "Not authenticated with Discord. Call authenticate_discord() to "
                    "start the browser sign-in and follow its 'instructions': the user "
                    "signs in to Discord on the page it opened for them, picks a server "
                    "and channel, and clicks Authorize; "
                    "then call finish_discord_auth(). Never ask for the user's Discord "
                    "password or 2FA code."
                )
            try:
                result = agent._backend._get("/users/@me")
                if "id" not in result:  # pragma: no branch
                    return json.dumps({"ok": False, "error": str(result)})
                answer: dict[str, Any] = {
                    "ok": True,
                    "auth": "user" if agent._backend._user_auth else "bot",
                    "username": result.get("username", ""),
                    "id": result.get("id", ""),
                }
                webhook = _webhook_config.load()
                if webhook:
                    answer["webhook_channel_id"] = webhook.get("channel_id", "")
                if agent._backend._user_auth:
                    answer["note"] = (
                        "User sign-in: list_guilds works and post_message can post to "
                        "the webhook channel; other tools need a bot token."
                    )
                return json.dumps(answer)
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_discord(
            bot_token: str = "",
            application_id: str = "",
            guild_ids: str = "",
        ) -> str:
            """Connect Discord by browser sign-in, or store a bot token.

            Without ``bot_token`` this starts the OAuth sign-in (PKCE, KISS's
            public Discord app, scopes ``identify guilds webhook.incoming``)
            and returns a ``consent_required`` answer whose ``instructions``
            say how to hand the page to the user (ask_user_question); they
            sign in, pick a server and channel, and click Authorize; then
            call finish_discord_auth().  A ``bot_token`` is the advanced
            path, only for bot-only features (reading messages, channel
            poll mode, managing messages); it is validated and stored.

            Args:
                bot_token: Optional Discord bot token (bot-only features).
                application_id: Optional application ID (bot token only).
                guild_ids: Optional comma-separated guild IDs (bot token only).

            Returns:
                A consent_required JSON answer, a validation result, or an
                error message.
            """
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            bot_token = bot_token.strip()
            if not bot_token:
                client_id = oauth_client_id("discord")
                if not client_id:
                    return json.dumps(
                        {"ok": False, "error": missing_client_id_error("discord", "Discord")}
                    )
                try:
                    session = LoopbackPkceSession(
                        "discord",
                        _pkce_provider(),
                        client_id,
                        LOOPBACK_REDIRECT_URI,
                        {"scope": _OAUTH_SCOPES},
                    )
                except OSError as e:
                    return json.dumps(
                        {"ok": False, "error": f"cannot listen on {LOOPBACK_REDIRECT_URI}: {e}"}
                    )
                session.register()
                return json.dumps(consent_required("discord", "Discord", session))
            # A bot token supersedes any browser sign-in still pending.
            ConsentSession.cancel_active("discord")
            if muse_auth_enabled():
                return _muse_authenticate(
                    agent._backend, bot_token, application_id.strip(), guild_ids.strip()
                )
            agent._backend._token = bot_token
            agent._backend._user_auth = False
            try:
                result = agent._backend._get("/users/@me")
                if "id" in result:  # pragma: no branch
                    _config.save(
                        {
                            "bot_token": bot_token,
                            "application_id": application_id.strip(),
                            "guild_ids": guild_ids.strip(),
                        }
                    )
                    return json.dumps(
                        {
                            "ok": True,
                            "message": "Discord token saved and validated.",
                            "username": result.get("username", ""),
                            "id": result.get("id", ""),
                        }
                    )
                agent._backend._token = ""
                return json.dumps({"ok": False, "error": str(result)})
            except Exception as e:
                agent._backend._token = ""
                return json.dumps({"ok": False, "error": str(e)})

        def finish_discord_auth() -> str:
            """Complete a browser sign-in started by authenticate_discord().

            Call after the user reports that they clicked Authorize.  The
            user token Discord issued is validated with a ``/users/@me``
            read and stored (Muse vault when enabled, where the daemon
            refreshes it with the public client ID; no secret involved),
            replacing any stored bot token.  The incoming webhook for the
            channel the user picked is stored as a secret and never shown.

            Returns:
                The validation result (user, webhook channel and server
                IDs), a pending status while the user has not approved
                yet, or an error message.
            """
            session, status = ConsentSession.finish("discord")
            if status == "pending":
                return json.dumps(
                    {
                        "ok": False,
                        "status": "pending",
                        "error": "The user has not authorized yet; ask them to finish "
                        "the sign-in, then call this tool again.",
                    }
                )
            if not isinstance(session, LoopbackPkceSession) or session.result is None:
                return json.dumps({"ok": False, "error": f"Discord sign-in failed: {status}"})
            grant = TokenGrant.from_session(session)
            # Validate first: a rejected token must not disturb the
            # credential that is currently in use.
            try:
                user = _probe_user(agent._backend._api_base, grant.access_token)
            except requests.RequestException as e:
                return json.dumps({"ok": False, "error": f"Discord unreachable: {e}"})
            if "id" not in user:
                return json.dumps({"ok": False, "error": f"Discord rejected the new token: {user}"})
            try:
                _store_user_grant(agent._backend, session, grant)
            except Exception as e:
                return json.dumps({"ok": False, "error": f"storing the sign-in failed: {e}"})
            webhook = _webhook_config.load() or {}
            return json.dumps(
                {
                    "ok": True,
                    "message": "Discord sign-in saved (user token).",
                    "username": user.get("username", ""),
                    "id": user.get("id", ""),
                    "webhook_channel_id": webhook.get("channel_id", ""),
                    "webhook_guild_id": webhook.get("guild_id", ""),
                }
            )

        def clear_discord_auth() -> str:
            """Clear the stored Discord credential and webhook.

            Returns:
                Status message.
            """
            ConsentSession.cancel_active("discord")
            _config.clear()
            _webhook_config.clear()
            agent._backend._token = ""
            agent._backend._user_auth = False
            agent._backend._http = requests
            agent._backend._muse = False
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if muse_auth_enabled():
                from kiss.agents.third_party_agents.muse_auth.client import clear_credentials

                clear_credentials("discord")
            return "Discord authentication cleared."

        return [
            check_discord_auth,
            authenticate_discord,
            finish_discord_auth,
            clear_discord_auth,
        ]


def _make_backend() -> DiscordChannelBackend:
    """Create a configured backend for channel poll mode (needs a bot token)."""
    backend = DiscordChannelBackend()
    from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

    wired = backend._wire_muse() if muse_auth_enabled() else backend._load_legacy_config()
    if not wired:
        print("Not authenticated. Run: kiss-discord -t 'authenticate'")
        sys.exit(1)
    if backend._user_auth:
        print(
            "Channel poll mode needs a Discord bot token (a user sign-in cannot read "
            "messages). Run: kiss-discord -t 'authenticate with my bot token'"
        )
        sys.exit(1)
    return backend


def main() -> None:
    """Run the DiscordAgent from the command line with chat persistence."""
    channel_main(
        DiscordAgent,
        "kiss-discord",
        channel_name="Discord",
        make_backend=_make_backend,
    )


def add_to_tools() -> list:
    """Return the Discord channel tools (``kiss.server.sorcar.run`` agent-script contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``extension_agent_path``: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return DiscordAgent()._get_tools()


def settings() -> dict:
    """Run as a ``channel`` worker (``kiss.server.sorcar.run`` agent-script contract).

    No git lifecycle, nothing inherited from the calling task, the
    channel preamble in the system prompt (see
    :mod:`kiss.agents.sorcar.sea_settings`).
    """
    return {"kind": "channel"}


def add_to_system_prompt() -> str:
    """Return the channel guidance appended to the run's system prompt."""
    return DiscordAgent.channel_system_prompt


if __name__ == "__main__":
    main()
