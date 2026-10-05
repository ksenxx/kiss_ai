# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Slack Agent — channel agent with Slack API tools.

Provides authenticated access to a Slack workspace with multi-turn
chat-session persistence.  The user connects by signing in to Slack in
their own browser and clicking Allow on the KISS-owned Slack app
(OAuth 2.0 authorization code + PKCE with a loopback redirect, no
client secret).  Slack only grants USER scopes to a localhost redirect,
so the agent acts with a rotating user token (``xoxp-``, about 12
hours) plus a refresh token.  In Muse-auth mode (the default) the pair
lives in the Muse vault and the daemon refreshes it; in legacy mode it
is stored in ``~/.kiss/third_party_agents/slack/<workspace>/token.json``
and refreshed in-process.  The agent exposes a focused set of Slack
Web API tools for messaging, channels, users, reactions, and search.

Usage::

    agent = SlackAgent()
    agent.run(prompt_template="List all public channels in my workspace")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any

import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ToolMethodBackend,
    channel_main,
    clear_json_config,
    load_json_config,
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
from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError
from kiss.core.config import kiss_home

logger = logging.getLogger(__name__)


def description() -> str:
    """Return the one-sentence help text shown by ``/slack help``."""
    return (
        "Messages, channels, users, reactions and search in a Slack workspace the user "
        "authorizes once in the browser (OAuth user token, refreshed automatically); use it "
        "with `run_agent(agent=\"slack\", task=...)` or the `kiss-slack -t \"<task>\"` CLI "
        "(`kiss-slack --channel <channel>` polls a channel and answers new messages, "
        "`--workspace <name>` selects among several signed-in workspaces)."
    )


def _call_with_retry(fn: Any, what: str) -> Any:
    """Call a Slack API function, retrying transient network errors.

    Retries up to 3 times on ``OSError`` (e.g. SSL handshake timeouts,
    connection resets) or ``MuseAuthError`` (Muse-mode boundary
    transport failures, which replace urllib's ``OSError`` family)
    with exponential backoff.  Sentinel policy denials are ordinary
    Slack API error responses, not transport errors, so they are never
    retried here.

    Args:
        fn: Zero-argument callable performing the API request.
        what: Human-readable label for the warning log (e.g.
            ``"messages"`` or ``"thread replies"``).

    Returns:
        The API response.

    Raises:
        OSError: When all 3 attempts fail.
        MuseAuthError: When all 3 attempts fail at the Muse boundary.
    """
    last_err: OSError | MuseAuthError | None = None
    for attempt in range(3):  # pragma: no branch
        try:
            return fn()
        except (OSError, MuseAuthError) as e:
            last_err = e
            if attempt < 2:  # pragma: no branch
                logger.warning(
                    "Network error polling %s (attempt %d/3): %s", what, attempt + 1, e
                )
                time.sleep(2**attempt)
    raise last_err  # type: ignore[misc]


def _advance_cursor(messages: list[dict[str, Any]], oldest: str) -> str:
    """Return the poll cursor advanced past the newest message.

    Args:
        messages: Polled messages (each with a ``ts`` field).
        oldest: The current cursor.

    Returns:
        One microsecond past the newest message timestamp at or after
        *oldest*, or *oldest* unchanged when no message is newer.
    """
    new_oldest = oldest
    for msg in messages:  # pragma: no branch
        ts = float(msg.get("ts", "0"))
        if ts >= float(new_oldest):  # pragma: no branch
            new_oldest = f"{ts + 0.000001:.6f}"
    return new_oldest


def _slack_dir() -> Path:
    """Return the Slack token directory, ``$KISS_HOME/third_party_agents/slack``.

    Resolved lazily on every call (like ``ChannelConfig.path``) so a
    ``KISS_HOME`` set after import, or a process that re-executes this
    module as an agent script, never reads the developer's real tokens.
    """
    return kiss_home() / "third_party_agents" / "slack"


# User-token scopes the Slack tools need.  Slack grants only user
# scopes to a localhost redirect, so the KISS app requests these as
# ``user_scope`` and the agent acts as the signed-in user.
_USER_SCOPES = ",".join(
    [
        "chat:write",  # chat.postMessage / update / delete
        "channels:read",  # conversations.list / info
        "channels:history",  # conversations.history / replies
        "channels:write",  # conversations.join / create / invite / setTopic
        "groups:read",
        "groups:history",
        "groups:write",
        "im:read",
        "im:history",
        "mpim:read",
        "mpim:history",
        "users:read",  # users.list / info
        "users:read.email",  # email field of users.info
        "reactions:write",  # reactions.add
        "search:read",  # search.messages
        "files:write",  # files_upload_v2
    ]
)

# Refresh a legacy-mode access token this many seconds before it expires.
_REFRESH_MARGIN = 300.0


def _slack_base() -> str:
    """Return the Slack origin used for sign-in and API calls.

    Returns:
        ``$KISS_SLACK_BASE_URL`` without a trailing slash (tests point it
        at a local server), else ``https://slack.com``.
    """
    return (os.environ.get("KISS_SLACK_BASE_URL", "").strip() or "https://slack.com").rstrip("/")


def _token_url() -> str:
    """Return Slack's ``oauth.v2.access`` endpoint (code exchange and refresh)."""
    return f"{_slack_base()}/api/oauth.v2.access"


def _token_path(workspace: str = "default") -> Path:
    """Return the path to the stored Slack token file for a workspace.

    Args:
        workspace: Workspace identifier used to key the token storage.
            Defaults to ``"default"``.

    Returns:
        Path to ``~/.kiss/third_party_agents/slack/{workspace}/token.json``.
    """
    return _slack_dir() / workspace / "token.json"


def _migrate_legacy_token() -> None:
    """Migrate a legacy token file to the workspace-keyed path.

    Moves ``~/.kiss/third_party_agents/slack/token.json`` to
    ``~/.kiss/third_party_agents/slack/default/token.json`` if the legacy file
    exists and the new location does not.
    """
    legacy = _slack_dir() / "token.json"
    dest = _token_path("default")
    if legacy.is_file() and not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        legacy.rename(dest)


def _load_config(workspace: str = "default") -> dict[str, str] | None:
    """Load a workspace's stored Slack token file.

    Args:
        workspace: Workspace identifier. Defaults to ``"default"``.

    Returns:
        The stored dict (``access_token`` plus, for a rotating token,
        ``refresh_token``, ``expires_at`` and ``client_id``), or None
        when missing or invalid.
    """
    if workspace == "default":
        _migrate_legacy_token()
    return load_json_config(_token_path(workspace), ("access_token",))


def _load_token(workspace: str = "default") -> str | None:
    """Load a stored Slack access token from disk.

    Args:
        workspace: Workspace identifier. Defaults to ``"default"``.

    Returns:
        The access token string, or None if not found or invalid.
    """
    cfg = _load_config(workspace)
    return cfg["access_token"] if cfg else None


def _save_token(
    token: str,
    workspace: str = "default",
    refresh_token: str = "",
    expires_at: float = 0.0,
    client_id: str = "",
) -> None:
    """Save a Slack token to disk with restricted permissions.

    Args:
        token: The access token string (e.g. ``xoxp-...``).
        workspace: Workspace identifier. Defaults to ``"default"``.
        refresh_token: Refresh token of a rotating token, or ``""``.
        expires_at: Wall-clock expiry of *token* (0 when it never expires).
        client_id: Public client ID the refresh grant is sent with.
    """
    cfg = {"access_token": token.strip()}
    if refresh_token:
        cfg.update(refresh_token=refresh_token, expires_at=str(expires_at), client_id=client_id)
    save_json_config(_token_path(workspace), cfg)


def _refresh_legacy_token(workspace: str, cfg: dict[str, str]) -> dict[str, str]:
    """Rotate a legacy-mode token with the refresh-token grant and save it.

    Args:
        workspace: Workspace identifier.
        cfg: The stored token dict (must carry ``refresh_token``).

    Returns:
        The updated token dict.

    Raises:
        SlackApiError: When Slack refuses the refresh or is unreachable.
    """
    form = {
        "grant_type": "refresh_token",
        "refresh_token": cfg["refresh_token"],
        "client_id": cfg.get("client_id", ""),
    }
    try:
        data = requests.post(_token_url(), data=form, timeout=30).json()
    except (requests.RequestException, ValueError) as e:
        raise SlackApiError(f"Slack token refresh failed: {type(e).__name__}", None) from None
    if not isinstance(data, dict) or not data.get("ok") or not data.get("access_token"):
        error = data.get("error", "") if isinstance(data, dict) else ""
        raise SlackApiError(f"Slack token refresh failed: {error or 'no token'}", None)
    grant = TokenGrant.from_response(data)
    new_cfg = {
        **cfg,
        "access_token": grant.access_token,
        "refresh_token": grant.refresh_token or cfg["refresh_token"],
        "expires_at": str(grant.acquired_at + (grant.expires_in or 3600.0)),
    }
    save_json_config(_token_path(workspace), new_cfg)
    return new_cfg


class _RefreshingWebClient(WebClient):
    """Legacy-mode WebClient that rotates its user token before it expires."""

    def __init__(self, workspace: str, cfg: dict[str, str], base_url: str) -> None:
        super().__init__(token=cfg["access_token"], base_url=base_url, retry_handlers=[])
        self._workspace = workspace
        self._cfg = cfg
        self._refresh_lock = threading.Lock()

    def api_call(self, *args: Any, **kwargs: Any) -> Any:
        """Refresh the token when it is about to expire, then call Slack.

        Args:
            *args: Positional arguments of :meth:`WebClient.api_call`.
            **kwargs: Keyword arguments of :meth:`WebClient.api_call`.

        Returns:
            The Slack response.
        """
        with self._refresh_lock:
            expires_at = float(self._cfg.get("expires_at") or 0)
            if self._cfg.get("refresh_token") and expires_at - _REFRESH_MARGIN < time.time():
                # Another client (or process) may have rotated the token
                # already: a consumed refresh token is refused, so start
                # from what is on disk.
                stored = _load_config(self._workspace) or self._cfg
                if float(stored.get("expires_at") or 0) - _REFRESH_MARGIN < time.time():
                    stored = _refresh_legacy_token(self._workspace, stored)
                self._cfg = stored
                self.token = self._cfg["access_token"]
        return super().api_call(*args, **kwargs)


def _api_base() -> str:
    """Return the Slack Web API base URL (``<origin>/api/``)."""
    return f"{_slack_base()}/api/"


def _legacy_client(workspace: str, base_url: str = "") -> WebClient | None:
    """Build a legacy-mode client from the workspace's token file.

    Args:
        workspace: Workspace identifier.
        base_url: Slack Web API base URL; ``""`` means :func:`_api_base`.

    Returns:
        A client that refreshes a rotating token in-process, or None
        when no token is stored.
    """
    cfg = _load_config(workspace)
    if cfg is None:
        return None
    return _RefreshingWebClient(workspace, cfg, base_url or _api_base())


def _muse_service(workspace: str = "default") -> str:
    """Return the Muse vault service name for a Slack workspace.

    The default workspace enrolls as ``slack``; other workspaces get a
    deterministic ``slack-<slug>-<hash>`` name.  The 64-bit digest of
    the exact workspace label disambiguates workspaces whose names
    collide after lowercasing/sanitizing (a 32-bit digest admits
    practical birthday collisions that would hand one workspace the
    other's token).

    Args:
        workspace: Workspace identifier used to key the token storage.

    Returns:
        A valid Muse service name.
    """
    if workspace == "default":
        return "slack"
    slug = re.sub(r"[^a-z0-9_-]", "-", workspace.lower())[:40]
    digest = hashlib.sha256(workspace.encode()).hexdigest()[:16]
    return f"slack-{slug}-{digest}"


def _legacy_vault_credential(cfg: dict[str, str]) -> dict[str, Any]:
    """Map a token file onto the Muse vault credential it migrates to.

    Args:
        cfg: The stored token dict.

    Returns:
        An ``oauth2_refresh_token`` credential for a rotating token
        (the daemon keeps refreshing it), else a plain ``bearer``.
    """
    if not cfg.get("refresh_token"):
        return {"kind": "bearer", "token": cfg["access_token"]}
    return {
        "kind": "oauth2_refresh_token",
        "token_url": _token_url(),
        "client_id": cfg.get("client_id", ""),
        "access_token": cfg["access_token"],
        "refresh_token": cfg["refresh_token"],
        "expires_at": float(cfg.get("expires_at") or 0),
    }


def _muse_web_client(workspace: str, base_url: str = "") -> WebClient | None:
    """Build a Muse-boundary WebClient for *workspace*.

    Vault-first: mints a surrogate for the workspace's enrolled
    credential, migrating a legacy on-disk token into the vault when no
    enrollment exists yet.  The returned client only ever holds the
    surrogate; the daemon swaps in (and refreshes) the real token at
    the network boundary.

    Args:
        workspace: Workspace identifier used to key the token storage.
        base_url: Slack Web API base URL; ``""`` means :func:`_api_base`
            (tests point this at an emulated API).

    Returns:
        A connected-capable client, or None when neither the vault nor
        the legacy token file has a credential.
    """
    from kiss.agents.third_party_agents.muse_auth.client import (
        mint_surrogate,
        store_credentials,
        vault_has_credentials,
    )
    from kiss.agents.third_party_agents.muse_auth.slack_transport import MuseWebClient

    service = _muse_service(workspace)
    handle = mint_surrogate(service)
    if handle is None:
        # One-time migration: the token file is only read when the
        # vault has no enrollment yet, and deleted once enrolled.
        cfg = _load_config(workspace)
        if cfg is None or vault_has_credentials(service):
            return None
        store_credentials(service, _legacy_vault_credential(cfg), [])
        handle = mint_surrogate(service)
        if handle is None:
            return None
        _clear_token(workspace)
    return MuseWebClient(service, handle.token, base_url=base_url or _api_base())


def _clear_token(workspace: str = "default") -> None:
    """Delete the stored Slack token for a workspace.

    Args:
        workspace: Workspace identifier. Defaults to ``"default"``.
    """
    clear_json_config(_token_path(workspace))


def _make_client(workspace: str, base_url: str = "") -> WebClient | None:
    """Build the Slack client for *workspace* in the active auth mode.

    Args:
        workspace: Workspace identifier.
        base_url: Slack Web API base URL; ``""`` means :func:`_api_base`.

    Returns:
        A Muse-boundary client (Muse-auth mode) or an in-process
        refreshing client (legacy mode), or None when no credential is
        stored.
    """
    from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

    if muse_auth_enabled():
        return _muse_web_client(workspace, base_url)
    return _legacy_client(workspace, base_url)


class SlackChannelBackend(ToolMethodBackend):
    """Slack channel backend.

    Provides channel monitoring and message sending for the channel
    poller and interactive agent.
    """

    def __init__(self, workspace: str = "default") -> None:
        self._client: WebClient | None = None
        self._bot_user_id: str = ""
        self._connection_info: str = ""
        self._workspace = workspace
        self._api_base_url: str = _api_base()

    def connect(self) -> bool:
        """Authenticate with Slack using the stored user token.

        Uses the workspace set at construction time to load the
        appropriate credential (see :func:`_make_client`).  In Muse-auth
        mode (the default) the real token lives in the Muse vault; this
        process only holds a surrogate and every Web API call is
        executed at the daemon boundary.

        Returns:
            True on success, False on failure.
        """
        client = _make_client(self._workspace, self._api_base_url)
        if client is None:
            self._connection_info = (
                f"No Slack credential for workspace {self._workspace!r}. Sign in first: "
                f"kiss-slack --workspace {self._workspace} -t 'authenticate'"
            )
            return False
        self._client = client
        try:
            auth = self._client.auth_test()
            self._bot_user_id = auth.get("user_id", "")
            self._connection_info = (
                f"Authenticated as {auth.get('user', '')} in {auth.get('team', '')}"
            )
            return True
        except SlackApiError as e:
            self._connection_info = f"Slack auth failed: {e}"
            return False

    def find_channel(self, name: str) -> str | None:
        """Find a Slack channel ID by name or ID.

        A *name* that is already a Slack conversation ID (``C…``, ``G…``
        or ``D…``, e.g. ``C0AKYSNLB7W``) is verified via
        ``conversations.info`` and returned directly, so private channels
        and DMs — which name lookup cannot list without extra scopes — can
        be addressed by ID.  If verification fails the value is treated as
        a plain name and the public and private channel lists visible to
        the token are searched.

        Args:
            name: Channel name without '#', or a conversation ID.

        Returns:
            Channel ID string, or None if not found.
        """
        assert self._client is not None
        if re.fullmatch(r"[CGD][A-Z0-9]{8,}", name):
            try:
                self._client.conversations_info(channel=name)
                return name
            except SlackApiError:
                pass
        cursor = ""
        while True:
            kwargs: dict[str, Any] = {
                "types": "public_channel,private_channel",
                "limit": 200,
            }
            if cursor:  # pragma: no branch
                kwargs["cursor"] = cursor
            resp = self._client.conversations_list(**kwargs)
            channels: list[dict[str, Any]] = resp.get("channels", [])
            for ch in channels:  # pragma: no branch
                if ch.get("name") == name:  # pragma: no branch
                    return str(ch["id"])
            cursor = (resp.get("response_metadata") or {}).get("next_cursor", "")
            if not cursor:  # pragma: no branch
                return None

    def find_user(self, username: str) -> str | None:
        """Find a Slack user ID by display name or username.

        A *username* that is already a Slack user ID (``U…`` or ``W…``)
        is returned unchanged, avoiding a full ``users.list``
        pagination (rate-limited on large workspaces).

        Args:
            username: Slack username (without @), or a user ID.

        Returns:
            User ID string, or None if not found.
        """
        assert self._client is not None
        if re.fullmatch(r"[UW][A-Z0-9]{7,}", username):
            return username
        cursor = ""
        while True:
            kwargs: dict[str, Any] = {"limit": 200}
            if cursor:  # pragma: no branch
                kwargs["cursor"] = cursor
            resp = self._client.users_list(**kwargs)
            members: list[dict[str, Any]] = resp.get("members", [])
            for u in members:  # pragma: no branch
                name_match = u.get("name") == username
                real_match = str(u.get("real_name", "")).lower() == username.lower()
                if name_match or real_match:  # pragma: no branch
                    return str(u["id"])
            cursor = (resp.get("response_metadata") or {}).get("next_cursor", "")
            if not cursor:  # pragma: no branch
                return None

    def join_channel(self, channel_id: str) -> None:
        """Join a Slack channel (bot needs to be a member to read/post).

        Args:
            channel_id: Channel ID to join.
        """
        assert self._client is not None
        try:
            self._client.conversations_join(channel=channel_id)
        except SlackApiError:
            pass

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll a Slack channel for new messages.

        Retries up to 3 times on transient network errors (e.g. SSL
        handshake timeouts, connection resets) with exponential backoff.

        Args:
            channel_id: Channel ID to poll.
            oldest: Only return messages newer than this timestamp.
            limit: Maximum number of messages to return.

        Returns:
            Tuple of (messages sorted oldest-first, updated oldest timestamp).
        """
        client = self._client
        assert client is not None
        resp = _call_with_retry(
            partial(client.conversations_history, channel=channel_id, oldest=oldest, limit=limit),
            "messages",
        )
        messages: list[dict[str, Any]] = resp.get("messages", [])
        messages.sort(key=lambda m: float(m.get("ts", "0")))
        return messages, _advance_cursor(messages, oldest)

    def poll_thread_messages(
        self, channel_id: str, thread_ts: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll a Slack thread for new replies since *oldest*.

        Used by the poller to detect user replies within active threads.
        The parent message itself is excluded from the results.

        Retries up to 3 times on transient network errors with
        exponential backoff (same strategy as ``poll_messages``).

        Args:
            channel_id: Channel ID containing the thread.
            thread_ts: Timestamp of the parent message (thread root).
            oldest: Only return messages newer than this timestamp.
            limit: Maximum number of messages to return.

        Returns:
            Tuple of (reply messages sorted oldest-first, updated oldest
            timestamp).
        """
        client = self._client
        assert client is not None
        resp = _call_with_retry(
            partial(
                client.conversations_replies,
                channel=channel_id,
                ts=thread_ts,
                oldest=oldest,
                limit=limit,
            ),
            "thread replies",
        )
        messages: list[dict[str, Any]] = resp.get("messages", [])
        messages = [m for m in messages if m.get("ts") != thread_ts]
        messages.sort(key=lambda m: float(m.get("ts", "0")))
        return messages, _advance_cursor(messages, oldest)

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a message to a Slack channel, optionally in a thread.

        Args:
            channel_id: Channel ID to post to.
            text: Message text (supports Slack mrkdwn formatting).
            thread_ts: If non-empty, reply in this thread.
        """
        assert self._client is not None
        kwargs: dict[str, Any] = {"channel": channel_id, "text": text}
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        self._client.chat_postMessage(**kwargs)

    def disconnect(self) -> None:
        """Release Slack backend state before stop or reconnect."""
        self._client = None
        self._bot_user_id = ""

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Check if a message was sent by the bot itself.

        Args:
            msg: Message dict from poll_messages.

        Returns:
            True if the message is from the bot.
        """
        return bool(msg.get("bot_id")) or msg.get("user", "") == self._bot_user_id

    def strip_bot_mention(self, text: str) -> str:
        """Remove bot mention markers from message text.

        Args:
            text: Raw message text.

        Returns:
            Cleaned text with bot mentions removed.
        """
        if self._bot_user_id:
            return text.replace(f"<@{self._bot_user_id}>", "").strip()
        return text

    def list_third_party_agents(
        self, types: str = "public_channel", limit: int = 200, cursor: str = ""
    ) -> str:
        """List channels in the Slack workspace.

        Args:
            types: Comma-separated channel types. Options:
                public_channel, private_channel, mpim, im.
                Default: "public_channel".
            limit: Maximum number of channels to return (1-1000).
                Default: 200.
            cursor: Pagination cursor for next page of results.
                Pass the value from the previous response's
                response_metadata.next_cursor.

        Returns:
            JSON string with channel list (id, name, purpose, num_members)
            and pagination cursor.
        """
        assert self._client is not None
        try:
            kwargs: dict[str, Any] = {"types": types, "limit": min(limit, 1000)}
            if cursor:  # pragma: no branch
                kwargs["cursor"] = cursor
            resp = self._client.conversations_list(**kwargs)
            raw_channels: list[dict[str, Any]] = resp.get("channels", [])
            third_party_agents = [
                {
                    "id": ch["id"],
                    "name": ch.get("name", ""),
                    "is_private": ch.get("is_private", False),
                    "purpose": ch.get("purpose", {}).get("value", ""),
                    "num_members": ch.get("num_members", 0),
                }
                for ch in raw_channels
            ]
            result: dict[str, Any] = {"ok": True, "third_party_agents": third_party_agents}
            next_cursor = (resp.get("response_metadata") or {}).get("next_cursor", "")
            if next_cursor:  # pragma: no branch
                result["next_cursor"] = next_cursor
            return json.dumps(result, indent=2)[:8000]
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def read_messages(
        self,
        channel: str,
        limit: int = 20,
        cursor: str = "",
        oldest: str = "",
        newest: str = "",
    ) -> str:
        """Read messages from a Slack channel.

        Args:
            channel: Channel ID (e.g. "C01234567").
            limit: Number of messages to return (1-1000). Default: 20.
            cursor: Pagination cursor for next page.
            oldest: Only messages after this Unix timestamp.
            newest: Only messages before this Unix timestamp.

        Returns:
            JSON string with messages (user, text, ts, thread_ts)
            and pagination cursor.
        """
        assert self._client is not None
        try:
            kwargs: dict[str, Any] = {"channel": channel, "limit": min(limit, 1000)}
            if cursor:  # pragma: no branch
                kwargs["cursor"] = cursor
            if oldest:  # pragma: no branch
                kwargs["oldest"] = oldest
            if newest:  # pragma: no branch
                kwargs["newest"] = newest
            resp = self._client.conversations_history(**kwargs)
            raw_msgs: list[dict[str, Any]] = resp.get("messages", [])
            messages = [
                {
                    "user": msg.get("user", ""),
                    "text": msg.get("text", ""),
                    "ts": msg.get("ts", ""),
                    "thread_ts": msg.get("thread_ts", ""),
                    "reply_count": msg.get("reply_count", 0),
                }
                for msg in raw_msgs
            ]
            result: dict[str, Any] = {"ok": True, "messages": messages}
            next_cursor = (resp.get("response_metadata") or {}).get("next_cursor", "")
            if next_cursor:  # pragma: no branch
                result["next_cursor"] = next_cursor
            return json.dumps(result, indent=2)[:8000]
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def read_thread(self, channel: str, thread_ts: str, limit: int = 50, cursor: str = "") -> str:
        """Read replies in a message thread.

        Args:
            channel: Channel ID where the thread lives.
            thread_ts: Timestamp of the parent message.
            limit: Number of replies to return (1-1000). Default: 50.
            cursor: Pagination cursor for next page.

        Returns:
            JSON string with thread messages and pagination cursor.
        """
        assert self._client is not None
        try:
            kwargs: dict[str, Any] = {
                "channel": channel,
                "ts": thread_ts,
                "limit": min(limit, 1000),
            }
            if cursor:  # pragma: no branch
                kwargs["cursor"] = cursor
            resp = self._client.conversations_replies(**kwargs)
            raw_msgs: list[dict[str, Any]] = resp.get("messages", [])
            messages = [
                {
                    "user": msg.get("user", ""),
                    "text": msg.get("text", ""),
                    "ts": msg.get("ts", ""),
                }
                for msg in raw_msgs
            ]
            result: dict[str, Any] = {"ok": True, "messages": messages}
            next_cursor = (resp.get("response_metadata") or {}).get("next_cursor", "")
            if next_cursor:  # pragma: no branch
                result["next_cursor"] = next_cursor
            return json.dumps(result, indent=2)[:8000]
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def post_message(self, channel: str, text: str, thread_ts: str = "", blocks: str = "") -> str:
        """Send a message to a Slack channel.

        Args:
            channel: Channel ID or name (e.g. "C01234567" or "#general").
            text: Message text (supports Slack mrkdwn formatting).
            thread_ts: Optional parent message timestamp to reply in a thread.
            blocks: Optional JSON string of Block Kit blocks for rich
                formatting. If provided, text becomes the fallback.

        Returns:
            JSON string with ok status and the message timestamp (ts).
        """
        assert self._client is not None
        try:
            kwargs: dict[str, Any] = {"channel": channel, "text": text}
            if thread_ts:  # pragma: no branch
                kwargs["thread_ts"] = thread_ts
            if blocks:  # pragma: no branch
                kwargs["blocks"] = json.loads(blocks)
            resp = self._client.chat_postMessage(**kwargs)
            return json.dumps(
                {"ok": True, "ts": resp.get("ts", ""), "channel": resp.get("channel", "")}
            )
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def update_message(self, channel: str, ts: str, text: str, blocks: str = "") -> str:
        """Update an existing message in a Slack channel.

        Args:
            channel: Channel ID where the message is.
            ts: Timestamp of the message to update.
            text: New message text.
            blocks: Optional JSON string of Block Kit blocks.

        Returns:
            JSON string with ok status and updated timestamp.
        """
        assert self._client is not None
        try:
            kwargs: dict[str, Any] = {"channel": channel, "ts": ts, "text": text}
            if blocks:  # pragma: no branch
                kwargs["blocks"] = json.loads(blocks)
            resp = self._client.chat_update(**kwargs)
            return json.dumps({"ok": True, "ts": resp.get("ts", "")})
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def delete_message(self, channel: str, ts: str) -> str:
        """Delete a message from a Slack channel.

        Args:
            channel: Channel ID where the message is.
            ts: Timestamp of the message to delete.

        Returns:
            JSON string with ok status.
        """
        assert self._client is not None
        try:
            self._client.chat_delete(channel=channel, ts=ts)
            return json.dumps({"ok": True})
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_users(self, limit: int = 200, cursor: str = "") -> str:
        """List users in the Slack workspace.

        Args:
            limit: Maximum number of users to return (1-1000). Default: 200.
            cursor: Pagination cursor for next page.

        Returns:
            JSON string with user list (id, name, real_name, is_bot)
            and pagination cursor.
        """
        assert self._client is not None
        try:
            kwargs: dict[str, Any] = {"limit": min(limit, 1000)}
            if cursor:  # pragma: no branch
                kwargs["cursor"] = cursor
            resp = self._client.users_list(**kwargs)
            raw_members: list[dict[str, Any]] = resp.get("members", [])
            users = [
                {
                    "id": u["id"],
                    "name": u.get("name", ""),
                    "real_name": u.get("real_name", ""),
                    "is_bot": u.get("is_bot", False),
                    "is_admin": u.get("is_admin", False),
                }
                for u in raw_members
            ]
            result: dict[str, Any] = {"ok": True, "users": users}
            next_cursor = (resp.get("response_metadata") or {}).get("next_cursor", "")
            if next_cursor:  # pragma: no branch
                result["next_cursor"] = next_cursor
            return json.dumps(result, indent=2)[:8000]
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_user_info(self, user: str) -> str:
        """Get detailed information about a Slack user.

        Args:
            user: User ID (e.g. "U01234567").

        Returns:
            JSON string with user profile details.
        """
        assert self._client is not None
        try:
            resp = self._client.users_info(user=user)
            u: dict[str, Any] = resp.get("user", {})
            profile: dict[str, Any] = u.get("profile", {})
            return json.dumps(
                {
                    "ok": True,
                    "user": {
                        "id": u.get("id", ""),
                        "name": u.get("name", ""),
                        "real_name": u.get("real_name", ""),
                        "display_name": profile.get("display_name", ""),
                        "email": profile.get("email", ""),
                        "title": profile.get("title", ""),
                        "is_bot": u.get("is_bot", False),
                        "is_admin": u.get("is_admin", False),
                        "tz": u.get("tz", ""),
                    },
                },
                indent=2,
            )
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def create_channel(self, name: str, is_private: bool = False) -> str:
        """Create a new Slack channel.

        Args:
            name: Channel name (lowercase, no spaces, max 80 chars).
                Use hyphens instead of spaces.
            is_private: If True, create a private channel. Default: False.

        Returns:
            JSON string with the new channel's id and name.
        """
        assert self._client is not None
        try:
            resp = self._client.conversations_create(name=name, is_private=is_private)
            ch: dict[str, Any] = resp.get("channel", {})
            return json.dumps(
                {
                    "ok": True,
                    "channel": {"id": ch.get("id", ""), "name": ch.get("name", "")},
                }
            )
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def invite_to_channel(self, channel: str, users: str) -> str:
        """Invite users to a Slack channel.

        Args:
            channel: Channel ID to invite users to.
            users: Comma-separated list of user IDs to invite.

        Returns:
            JSON string with ok status.
        """
        assert self._client is not None
        try:
            self._client.conversations_invite(channel=channel, users=users)
            return json.dumps({"ok": True})
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def add_reaction(self, channel: str, timestamp: str, name: str) -> str:
        """Add an emoji reaction to a message.

        Args:
            channel: Channel ID where the message is.
            timestamp: Timestamp of the message to react to.
            name: Emoji name without colons (e.g. "thumbsup", "heart").

        Returns:
            JSON string with ok status.
        """
        assert self._client is not None
        try:
            self._client.reactions_add(channel=channel, timestamp=timestamp, name=name)
            return json.dumps({"ok": True})
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def search_messages(self, query: str, count: int = 20, sort: str = "timestamp") -> str:
        """Search for messages across the workspace.

        Note: Requires a user token with search:read scope
        (granted by the KISS Slack app sign-in).

        Args:
            query: Search query string (supports Slack search modifiers
                like "in:#channel", "from:@user", "has:link").
            count: Number of results to return (1-100). Default: 20.
            sort: Sort order — "timestamp" (default) or "score".

        Returns:
            JSON string with matching messages.
        """
        assert self._client is not None
        try:
            resp = self._client.search_messages(query=query, count=min(count, 100), sort=sort)
            msg_data: dict[str, Any] = resp.get("messages", {})
            matches: list[dict[str, Any]] = msg_data.get("matches", [])
            results = [
                {
                    "text": m.get("text", ""),
                    "user": m.get("user", ""),
                    "ts": m.get("ts", ""),
                    "channel": m.get("channel", {}).get("name", ""),
                    "permalink": m.get("permalink", ""),
                }
                for m in matches
            ]
            return json.dumps({"ok": True, "messages": results}, indent=2)[:8000]
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def set_channel_topic(self, channel: str, topic: str) -> str:
        """Set the topic for a Slack channel.

        Args:
            channel: Channel ID.
            topic: New topic text.

        Returns:
            JSON string with ok status.
        """
        assert self._client is not None
        try:
            self._client.conversations_setTopic(channel=channel, topic=topic)
            return json.dumps({"ok": True})
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def upload_file(
        self, third_party_agents: str, content: str, filename: str, title: str = "",
    ) -> str:
        """Upload text content as a file to Slack channels.

        Args:
            third_party_agents: Comma-separated channel IDs to share the file in.
            content: Text content of the file.
            filename: Name for the file (e.g. "report.txt").
            title: Optional title for the file.

        Returns:
            JSON string with ok status and file id.
        """
        assert self._client is not None
        try:
            channel_list = [c.strip() for c in third_party_agents.split(",") if c.strip()]
            resp = self._client.files_upload_v2(
                channels=channel_list,
                content=content,
                filename=filename,
                title=title or filename,
            )
            file_data: dict[str, Any] = resp.get("file", {})
            return json.dumps({"ok": True, "file_id": file_data.get("id", "")})
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_channel_info(self, channel: str) -> str:
        """Get detailed information about a Slack channel.

        Args:
            channel: Channel ID (e.g. "C01234567").

        Returns:
            JSON string with channel details (name, topic, purpose,
            num_members, created, creator).
        """
        assert self._client is not None
        try:
            resp = self._client.conversations_info(channel=channel)
            ch: dict[str, Any] = resp.get("channel", {})
            return json.dumps(
                {
                    "ok": True,
                    "channel": {
                        "id": ch.get("id", ""),
                        "name": ch.get("name", ""),
                        "topic": ch.get("topic", {}).get("value", ""),
                        "purpose": ch.get("purpose", {}).get("value", ""),
                        "num_members": ch.get("num_members", 0),
                        "is_private": ch.get("is_private", False),
                        "created": ch.get("created", 0),
                        "creator": ch.get("creator", ""),
                    },
                },
                indent=2,
            )
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})


class SlackAgent(BaseChannelAgent):
    """Channel agent with Slack workspace tools.

    Tasks run on the kiss-web daemon's agent (which supplies bash,
    file editing, and browser automation) with authenticated Slack API tools for
    messaging, channel management, user lookup, reactions, search,
    and file uploads.

    The agent picks up a stored credential on initialization.  If there
    is none, ``authenticate_slack()`` starts a browser sign-in to the
    KISS-owned Slack app and ``finish_slack_auth()`` stores the result.

    Example::

        agent = SlackAgent()
        result = agent.run(
            prompt_template="Send 'Hello!' to #general",
        )
    """

    channel_system_prompt = connect_prompt(
        "slack",
        "Slack",
        "authenticate_slack()",
        "It takes no arguments and needs no Slack app of the user's own: the "
        "user approves KISS's Slack app, and the agent then acts as that user "
        "(messages are posted under their name).",
    )

    def __init__(self, workspace: str = "default") -> None:
        super().__init__("Slack Agent", workspace=workspace)
        self._backend = SlackChannelBackend(workspace=workspace)
        # Muse-auth mode: the client holds only a vault surrogate; the
        # real token never enters this process once migrated.
        self._backend._client = _make_client(workspace, self._backend._api_base_url)

    def _is_authenticated(self) -> bool:
        """Return True if a Slack client is configured."""
        return self._backend._client is not None

    def _get_auth_tools(self) -> list:
        """Return Slack authentication tool functions."""
        agent = self

        def check_slack_auth() -> str:
            """Check if a Slack credential is configured and valid.

            Tests the stored token against the Slack API (auth.test).

            Returns:
                Authentication status with workspace and user info, or
                instructions for how to authenticate.
            """
            if agent._backend._client is None:
                return (
                    "Not authenticated with Slack. Call authenticate_slack() to "
                    "start the browser sign-in to KISS's Slack app and follow its "
                    "'instructions' (ask_user_question) so the user signs in and "
                    "clicks Allow on the page it opened for them, then call "
                    "finish_slack_auth(). Never ask for the user's Slack password "
                    "or 2FA code."
                )
            try:
                resp = agent._backend._client.auth_test()
                return json.dumps(
                    {
                        "ok": True,
                        "team": resp.get("team", ""),
                        "user": resp.get("user", ""),
                        "user_id": resp.get("user_id", ""),
                        "url": resp.get("url", ""),
                    }
                )
            except (SlackApiError, MuseAuthError) as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_slack() -> str:
            """Start the browser sign-in to the KISS-owned Slack app.

            Opens Slack's authorization page (user scopes only, PKCE, no
            client secret) for the user (the Browser tab on every KISS
            surface, else their default browser) and waits for Slack's
            redirect on http://localhost:53682/callback.  Follow the
            returned ``instructions`` with ask_user_question(); after
            they click Allow,
            call finish_slack_auth().  The credential is stored under the
            current workspace (set via ``--workspace``).

            Returns:
                JSON with ``status: consent_required``, the sign-in URL
                and instructions, or an error.
            """
            client_id = oauth_client_id("slack")
            if not client_id:
                return json.dumps(
                    {"ok": False, "error": missing_client_id_error("slack", "Slack")}
                )
            provider = PkceProvider(f"{_slack_base()}/oauth/v2/authorize", _token_url())
            try:
                session = LoopbackPkceSession(
                    "slack",
                    provider,
                    client_id,
                    LOOPBACK_REDIRECT_URI,
                    {"user_scope": _USER_SCOPES},
                )
            except OSError as e:
                return json.dumps(
                    {"ok": False, "error": f"Could not start the Slack sign-in: {e}"}
                )
            session.register()
            return json.dumps(consent_required("slack", "Slack", session))

        def finish_slack_auth() -> str:
            """Complete the sign-in started by authenticate_slack().

            Call after the user reports that they clicked Allow.  The
            user token Slack issued is validated with auth.test and
            stored: in the Muse vault when Muse-auth is enabled (the
            daemon refreshes the rotating token with the public client
            ID), otherwise in the workspace's token file (refreshed
            in-process before it expires).

            Returns:
                The validation result, a pending status while the user
                has not approved yet, or an error message.
            """
            session, status = ConsentSession.finish("slack")
            if status == "pending":
                return json.dumps(
                    {
                        "ok": False,
                        "status": "pending",
                        "error": "The user has not approved yet; ask them to finish "
                        "the sign-in, then call this tool again.",
                    }
                )
            if not isinstance(session, LoopbackPkceSession):
                return json.dumps({"ok": False, "error": f"Slack sign-in failed: {status}"})
            # Slack nests the user token under ``authed_user``.
            authed_user = (session.result or {}).get("authed_user") or {}
            grant = TokenGrant.from_response(authed_user, acquired_at=session.result_at)
            if not grant.access_token:
                return json.dumps(
                    {"ok": False, "error": "Slack returned no user token for this sign-in."}
                )
            return _store_grant(
                agent, grant, session.client_id, session.provider.token_url
            )

        def clear_slack_auth() -> str:
            """Clear the stored Slack credential for the current workspace.

            Returns:
                Status message.
            """
            ConsentSession.cancel_active("slack")
            _clear_token(workspace=agent.workspace)
            agent._backend._client = None
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if muse_auth_enabled():
                from kiss.agents.third_party_agents.muse_auth.client import clear_credentials

                clear_credentials(_muse_service(agent.workspace))
            return "Slack authentication cleared."

        return [check_slack_auth, authenticate_slack, finish_slack_auth, clear_slack_auth]


def _store_grant(agent: SlackAgent, grant: TokenGrant, client_id: str, token_url: str) -> str:
    """Validate a signed-in user token and store it for the agent's workspace.

    The token is checked with auth.test before anything is stored, so a
    rejected sign-in leaves the credential in use untouched.

    Args:
        agent: The Slack agent whose workspace receives the credential.
        grant: The user token (and refresh token) Slack issued.
        client_id: Public client ID the refresh grant is sent with.
        token_url: Slack's ``oauth.v2.access`` endpoint.

    Returns:
        JSON with the team and user on success, else the error.
    """
    from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

    base_url = agent._backend._api_base_url
    try:
        resp = WebClient(
            token=grant.access_token, base_url=base_url, retry_handlers=[]
        ).auth_test()
    except (SlackApiError, OSError) as e:
        return json.dumps({"ok": False, "error": f"Token validation failed: {e}"})
    info = grant.vault_credential(token_url, client_id)
    workspace = agent.workspace
    if muse_auth_enabled():
        from kiss.agents.third_party_agents.muse_auth.client import (
            clear_credentials,
            store_credentials,
        )

        service = _muse_service(workspace)
        clear_credentials(service)
        store_credentials(service, info, [])
        # Non-secret marker so --list-workspaces can discover a
        # workspace whose only credential lives in the vault.
        _token_path(workspace).parent.mkdir(parents=True, exist_ok=True)
        where = "the Muse vault"
    else:
        _save_token(
            grant.access_token,
            workspace,
            refresh_token=grant.refresh_token,
            expires_at=float(info.get("expires_at", 0.0)),
            client_id=client_id,
        )
        where = str(_token_path(workspace))
    agent._backend._client = _make_client(workspace, base_url)
    return json.dumps(
        {
            "ok": True,
            "message": f"Signed in to Slack; the user token is stored in {where}.",
            "team": resp.get("team", ""),
            "user": resp.get("user", ""),
            "workspace": workspace,
        }
    )


def _workspace_vault_file(workspace: str) -> Path:
    """Return the Muse vault file backing a workspace's credential.

    Checked directly on disk so vault-awareness does not spin up the
    daemon when nothing is enrolled.

    Args:
        workspace: Workspace identifier.

    Returns:
        Path to ``$KISS_HOME/muse_auth/vault/<service>.json``.
    """
    from kiss.agents.third_party_agents.muse_auth._common import muse_auth_dir

    return muse_auth_dir() / "vault" / f"{_muse_service(workspace)}.json"


def _delete_workspace(workspace: str) -> None:
    """Delete a workspace's token directory and Muse vault credential.

    Removes the entire ``~/.kiss/third_party_agents/slack/{workspace}/``
    directory (token file plus any other workspace-specific files) and,
    when the workspace is enrolled in the Muse vault, clears that
    credential too so deletion leaves nothing mintable behind.

    Args:
        workspace: Workspace identifier to delete.
    """
    ws_dir = _slack_dir() / workspace
    vault_file = _workspace_vault_file(workspace)
    if not ws_dir.is_dir() and not vault_file.exists():
        print(f"Workspace {workspace!r} not found.")
        sys.exit(1)
    if ws_dir.is_dir():
        shutil.rmtree(ws_dir)
    if vault_file.exists():
        from kiss.agents.third_party_agents.muse_auth.client import clear_credentials

        clear_credentials(_muse_service(workspace))
    print(f"Workspace {workspace!r} deleted.")


def _list_workspaces() -> None:
    """Display all authenticated Slack workspaces and their token status.

    Scans ``~/.kiss/third_party_agents/slack/`` for workspace
    subdirectories containing ``token.json`` files and includes
    workspaces whose credential lives only in the Muse vault.
    Vault-backed workspaces print a ``✓ vault`` status (the plaintext
    is gone, so no direct validation happens here); plaintext tokens
    are validated against the Slack API with the workspace name,
    status, team name, and signed-in user.
    """
    workspaces: list[str] = []
    slack_dir = _slack_dir()
    if slack_dir.is_dir():
        for entry in sorted(slack_dir.iterdir()):
            has_token = (entry / "token.json").is_file()
            if entry.is_dir() and (has_token or _workspace_vault_file(entry.name).exists()):
                workspaces.append(entry.name)
    # The default workspace may exist only as a Muse vault enrollment
    # (its plaintext token file is deleted after migration).
    if "default" not in workspaces and _workspace_vault_file("default").exists():
        workspaces.insert(0, "default")
    if not workspaces:
        print("No workspaces found.")
        return
    print(f"{'Workspace':<20} {'Status':<12} {'Team':<20} {'User'}")
    print("-" * 72)
    for ws in workspaces:
        if _workspace_vault_file(ws).exists():
            print(f"{ws:<20} {'✓ vault':<12} {'-':<20} -")
            continue
        client = _legacy_client(ws)
        if client is None:
            print(f"{ws:<20} {'no token':<12} {'-':<20} -")
            continue
        try:
            resp = client.auth_test()
            team = str(resp.get("team", ""))
            user = str(resp.get("user", ""))
            print(f"{ws:<20} {'✓ valid':<12} {team:<20} {user}")
        except SlackApiError:
            print(f"{ws:<20} {'✗ invalid':<12} {'-':<20} -")
        except Exception:
            print(f"{ws:<20} {'✗ error':<12} {'-':<20} -")


def _make_backend(workspace: str = "default") -> SlackChannelBackend:
    """Create a configured backend for channel poll mode.

    Args:
        workspace: Workspace identifier for token lookup.
    """
    backend = SlackChannelBackend(workspace=workspace)
    backend._client = _make_client(workspace, backend._api_base_url)
    if backend._client is None:
        print(
            f"Not authenticated for workspace {workspace!r}. "
            f"Run: kiss-slack --workspace {workspace} -t 'authenticate'"
        )
        sys.exit(1)
    return backend

def main() -> None:
    """Run the SlackAgent from the command line with chat persistence."""
    if "--list-workspaces" in sys.argv:
        _list_workspaces()
        return
    if "--delete-workspace" in sys.argv:
        idx = sys.argv.index("--delete-workspace")
        if idx + 1 >= len(sys.argv):
            print("Usage: kiss-slack --delete-workspace <workspace>")
            sys.exit(1)
        _delete_workspace(sys.argv[idx + 1])
        return
    channel_main(
        SlackAgent,
        "kiss-slack",
        channel_name="Slack",
        make_backend=_make_backend,
        extra_usage="[--list-workspaces] [--delete-workspace WS]",
    )


def add_to_tools() -> list:
    """Return the Slack channel tools (``kiss.server.sorcar.run`` agent-script contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``extension_agent_path``: builds a fresh agent from the token
    persisted under ``~/.kiss`` and returns its authentication and
    backend tools.  The workspace comes from the
    ``KISS_CHANNEL_WORKSPACE`` environment variable (set by the
    launcher while the task runs), defaulting to ``"default"``.
    """
    workspace = os.environ.get("KISS_CHANNEL_WORKSPACE", "default") or "default"
    return SlackAgent(workspace=workspace)._get_tools()


def settings() -> dict:
    """Run as a ``channel`` worker (``kiss.server.sorcar.run`` agent-script contract).

    No git lifecycle, nothing inherited from the calling task, the
    channel preamble in the system prompt (see
    :mod:`kiss.agents.sorcar.sea_settings`).
    """
    return {"kind": "channel"}


def add_to_system_prompt() -> str:
    """Return the channel guidance appended to the run's system prompt."""
    return SlackAgent.channel_system_prompt


if __name__ == "__main__":
    main()
