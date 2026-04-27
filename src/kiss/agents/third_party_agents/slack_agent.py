"""Slack Agent — ChatSorcarAgent extension with Slack API tools.

Provides authenticated access to a Slack workspace via a bot token
with multi-turn chat-session persistence.  Handles authentication
(reading token from disk or prompting the user via the browser),
stores the token securely in ``~/.kiss/third_party_agents/slack/token.json``,
and exposes a focused set of Slack Web API tools that give the agent
full control over messaging, third_party_agents, users, reactions, and search.

Usage::

    agent = SlackAgent()
    agent.run(prompt_template="List all public third_party_agents in my workspace")
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, cast

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.third_party_agents._backend_utils import wait_for_matching_message
from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ToolMethodBackend,
    channel_main,
    clear_json_config,
    load_json_config,
    save_json_config,
)

logger = logging.getLogger(__name__)

_SLACK_DIR = Path.home() / ".kiss" / "third_party_agents" / "slack"


def _token_path(workspace: str = "default") -> Path:
    """Return the path to the stored Slack bot token file for a workspace.

    Args:
        workspace: Workspace identifier used to key the token storage.
            Defaults to ``"default"``.

    Returns:
        Path to ``~/.kiss/third_party_agents/slack/{workspace}/token.json``.
    """
    return _SLACK_DIR / workspace / "token.json"


def _migrate_legacy_token() -> None:
    """Migrate a legacy token file to the workspace-keyed path.

    Moves ``~/.kiss/third_party_agents/slack/token.json`` to
    ``~/.kiss/third_party_agents/slack/default/token.json`` if the legacy file
    exists and the new location does not.
    """
    legacy = _SLACK_DIR / "token.json"
    dest = _token_path("default")
    if legacy.is_file() and not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        legacy.rename(dest)


def _load_token(workspace: str = "default") -> str | None:
    """Load a stored Slack bot token from disk.

    Args:
        workspace: Workspace identifier. Defaults to ``"default"``.

    Returns:
        The bot token string, or None if not found or invalid.
    """
    if workspace == "default":
        _migrate_legacy_token()
    cfg = load_json_config(_token_path(workspace), ("access_token",))
    return cfg["access_token"] if cfg else None


def _save_token(token: str, workspace: str = "default") -> None:
    """Save a Slack bot token to disk with restricted permissions.

    Args:
        token: The bot token string (e.g. ``xoxb-...``).
        workspace: Workspace identifier. Defaults to ``"default"``.
    """
    save_json_config(_token_path(workspace), {"access_token": token.strip()})


def _clear_token(workspace: str = "default") -> None:
    """Delete the stored Slack bot token for a workspace.

    Args:
        workspace: Workspace identifier. Defaults to ``"default"``.
    """
    clear_json_config(_token_path(workspace))


_REPLY_POLL_INTERVAL = 2.0


class SlackChannelBackend(ToolMethodBackend):
    """Slack channel backend.

    Provides channel monitoring, message sending, and reply waiting for
    the channel poller and interactive agent.
    """

    def __init__(self, workspace: str = "default") -> None:
        self._client: WebClient | None = None
        self._bot_user_id: str = ""
        self._connection_info: str = ""
        self._workspace = workspace

    def connect(self) -> bool:
        """Authenticate with Slack using the stored bot token.

        Uses the workspace set at construction time to load the
        appropriate token.

        Returns:
            True on success, False on failure.
        """
        token = _load_token(self._workspace)
        if not token:
            self._connection_info = (
                "No Slack token found. Please store a bot token first.\n"
                "Run: uv run python -m kiss.agents.third_party_agents"
                ".slack_agent --task 'check auth'\n"
                "Or manually save token to ~/.kiss/third_party_agents/slack/token.json"
            )
            return False
        self._client = WebClient(token=token, retry_handlers=[])
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
        """Find a Slack channel ID by name.

        Args:
            name: Channel name without '#'.

        Returns:
            Channel ID string, or None if not found.
        """
        assert self._client is not None
        cursor = ""
        while True:
            kwargs: dict[str, Any] = {"types": "public_channel", "limit": 200}
            if cursor:  # pragma: no branch
                kwargs["cursor"] = cursor
            resp = self._client.conversations_list(**kwargs)
            third_party_agents: list[dict[str, Any]] = resp.get("third_party_agents", [])
            for ch in third_party_agents:  # pragma: no branch
                if ch.get("name") == name:  # pragma: no branch
                    return str(ch["id"])
            cursor = (resp.get("response_metadata") or {}).get("next_cursor", "")
            if not cursor:  # pragma: no branch
                return None

    def find_user(self, username: str) -> str | None:
        """Find a Slack user ID by display name or username.

        Args:
            username: Slack username (without @).

        Returns:
            User ID string, or None if not found.
        """
        assert self._client is not None
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
        assert self._client is not None
        last_err: OSError | None = None
        for attempt in range(3):  # pragma: no branch
            try:
                resp = self._client.conversations_history(
                    channel=channel_id, oldest=oldest, limit=limit
                )
                break
            except OSError as e:
                last_err = e
                if attempt < 2:  # pragma: no branch
                    logger.warning(
                        "Network error polling messages (attempt %d/3): %s",
                        attempt + 1,
                        e,
                    )
                    time.sleep(2**attempt)
        else:
            raise last_err  # type: ignore[misc]
        messages: list[dict[str, Any]] = resp.get("messages", [])
        messages.sort(key=lambda m: float(m.get("ts", "0")))
        new_oldest = oldest
        for msg in messages:  # pragma: no branch
            ts = float(msg.get("ts", "0"))
            if ts >= float(new_oldest):  # pragma: no branch
                new_oldest = f"{ts + 0.000001:.6f}"
        return messages, new_oldest

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
        assert self._client is not None
        last_err: OSError | None = None
        for attempt in range(3):  # pragma: no branch
            try:
                resp = self._client.conversations_replies(
                    channel=channel_id, ts=thread_ts, oldest=oldest, limit=limit
                )
                break
            except OSError as e:
                last_err = e
                if attempt < 2:  # pragma: no branch
                    logger.warning(
                        "Network error polling thread replies (attempt %d/3): %s",
                        attempt + 1,
                        e,
                    )
                    time.sleep(2**attempt)
        else:
            raise last_err  # type: ignore[misc]
        messages: list[dict[str, Any]] = resp.get("messages", [])
        messages = [m for m in messages if m.get("ts") != thread_ts]
        messages.sort(key=lambda m: float(m.get("ts", "0")))
        new_oldest = oldest
        for msg in messages:  # pragma: no branch
            ts = float(msg.get("ts", "0"))
            if ts >= float(new_oldest):  # pragma: no branch
                new_oldest = f"{ts + 0.000001:.6f}"
        return messages, new_oldest

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

    def wait_for_reply(
        self,
        channel_id: str,
        thread_ts: str,
        user_id: str,
        timeout_seconds: float = 300.0,
    ) -> str | None:
        """Poll a Slack thread for a reply from a specific user.

        Args:
            channel_id: Channel ID containing the thread.
            thread_ts: Timestamp of the parent message (thread root).
            user_id: User ID to wait for a reply from.

        Returns:
            The text of the user's reply message, or ``None`` on timeout.
        """
        client = self._client
        assert client is not None
        seen_ts: set[str] = set()
        try:
            resp = client.conversations_replies(channel=channel_id, ts=thread_ts, limit=100)
            for msg in cast(list[dict[str, Any]], resp.get("messages", [])):
                seen_ts.add(str(msg["ts"]))
        except SlackApiError:
            pass

        def poll() -> list[dict[str, Any]]:
            try:
                resp = client.conversations_replies(channel=channel_id, ts=thread_ts, limit=100)
            except (SlackApiError, OSError):
                logger.debug("Error polling thread replies", exc_info=True)
                return []
            replies: list[dict[str, Any]] = []
            for reply in cast(list[dict[str, Any]], resp.get("messages", [])):
                ts = str(reply["ts"])
                if ts in seen_ts:
                    continue
                seen_ts.add(ts)
                replies.append(reply)
            return replies

        return wait_for_matching_message(
            poll=poll,
            matches=lambda reply: reply.get("user") == user_id,
            extract_text=lambda reply: str(reply.get("text", "")),
            timeout_seconds=timeout_seconds,
            poll_interval=_REPLY_POLL_INTERVAL,
        )

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
        """List third_party_agents in the Slack workspace.

        Args:
            types: Comma-separated channel types. Options:
                public_channel, private_channel, mpim, im.
                Default: "public_channel".
            limit: Maximum number of third_party_agents to return (1-1000).
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
            raw_third_party_agents: list[dict[str, Any]] = resp.get("third_party_agents", [])
            third_party_agents = [
                {
                    "id": ch["id"],
                    "name": ch.get("name", ""),
                    "is_private": ch.get("is_private", False),
                    "purpose": ch.get("purpose", {}).get("value", ""),
                    "num_members": ch.get("num_members", 0),
                }
                for ch in raw_third_party_agents
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

        Note: Requires a user token with search:read scope.
        Bot tokens cannot use this method.

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
        """Upload text content as a file to Slack third_party_agents.

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
                third_party_agents=channel_list,
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


class SlackAgent(BaseChannelAgent, ChatSorcarAgent):
    """ChatSorcarAgent extended with Slack workspace tools.

    Inherits all standard SorcarAgent capabilities (bash, file editing,
    browser automation) and adds authenticated Slack API tools for
    messaging, channel management, user lookup, reactions, search,
    and file uploads.

    The agent checks for a stored bot token on initialization. If no
    token is found, authentication tools guide the user through
    obtaining and storing one.

    Example::

        agent = SlackAgent()
        result = agent.run(
            prompt_template="Send 'Hello!' to #general",
        )
    """

    def __init__(self, workspace: str = "default") -> None:
        super().__init__("Slack Agent")
        self._workspace = workspace
        self._backend = SlackChannelBackend(workspace=workspace)
        token = _load_token(workspace)
        if token:
            self._backend._client = WebClient(token=token, retry_handlers=[])

    def run(self, **kwargs: Any) -> str:  # type: ignore[override]
        """Run with Slack-specific system prompt encouraging browser-based auth."""
        channel_prompt = (
            "\n\n## Slack Authentication\n"
            "If authentication is needed, call start_slack_browser_auth() to open "
            "the Slack API portal, then use browser tools to create an app, configure "
            "OAuth scopes, install it to a workspace, and copy the xoxb- bot token. "
            "Complete these steps autonomously — do NOT instruct the user to do them manually."
        )
        kwargs["system_prompt"] = (kwargs.get("system_prompt") or "") + channel_prompt
        return super().run(**kwargs)

    def _is_authenticated(self) -> bool:
        """Return True if a Slack client is configured."""
        return self._backend._client is not None

    def _get_auth_tools(self) -> list:
        """Return Slack authentication tool functions."""
        agent = self

        def check_slack_auth() -> str:
            """Check if the Slack bot token is configured and valid.

            Tests the stored token against the Slack API (auth.test).

            Returns:
                Authentication status with workspace and bot user info,
                or instructions for how to authenticate.
            """
            if agent._backend._client is None:
                return (
                    "Not authenticated with Slack. Call start_slack_browser_auth() "
                    "to open the Slack API portal in the browser and create an app "
                    "autonomously, then call authenticate_slack(token=...) with the "
                    "xoxb- bot token you retrieve."
                )
            try:
                resp = agent._backend._client.auth_test()
                return json.dumps(
                    {
                        "ok": True,
                        "team": resp.get("team", ""),
                        "user": resp.get("user", ""),
                        "bot_id": resp.get("bot_id", ""),
                        "url": resp.get("url", ""),
                    }
                )
            except SlackApiError as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_slack(token: str) -> str:
            """Store and validate a Slack bot token.

            Saves the token under the current workspace (set via
            ``--workspace``).  Validates it with auth.test.

            Args:
                token: Slack bot token (starts with 'xoxb-' for bot tokens
                    or 'xoxp-' for user tokens).

            Returns:
                Validation result with workspace info, or error message.
            """
            token = token.strip()
            if not token:
                return "Token cannot be empty."
            agent._backend._client = WebClient(token=token, retry_handlers=[])
            try:
                resp = agent._backend._client.auth_test()
                _save_token(token, workspace=agent._workspace)
                return json.dumps(
                    {
                        "ok": True,
                        "message": "Slack token saved and validated.",
                        "team": resp.get("team", ""),
                        "user": resp.get("user", ""),
                        "workspace": agent._workspace,
                    }
                )
            except SlackApiError as e:
                agent._backend._client = None
                return json.dumps({"ok": False, "error": f"Token validation failed: {e}"})

        def clear_slack_auth() -> str:
            """Clear the stored Slack authentication token for the current workspace.

            Returns:
                Status message.
            """
            _clear_token(workspace=agent._workspace)
            agent._backend._client = None
            return "Slack authentication cleared."

        def start_slack_browser_auth() -> str:
            """Begin automated Slack app creation and token retrieval via browser.

            Navigates to the Slack API portal. Use your browser tools (go_to_url,
            click, type_text) to complete the following steps autonomously:
            1. Create a new app ("From scratch"), give it a name, select a workspace.
            2. Go to "OAuth & Permissions", add bot scopes
               (third_party_agents:read, chat:write, etc.).
            3. Click "Install to Workspace" and approve the installation.
            4. Copy the "Bot User OAuth Token" (starts with xoxb-).
            5. Call authenticate_slack(token=<the token>).
            Use ask_user_question() if you need user help with login or workspace-selection screens.

            Returns:
                Page content of the Slack API portal to begin navigation.
            """
            if agent.web_use_tool is None:  # pragma: no branch
                return (
                    "Browser not available. Use authenticate_slack(token=...) "
                    "with an xoxb- token from https://api.slack.com/apps."
                )
            return agent.web_use_tool.go_to_url("https://api.slack.com/apps")

        return [
            check_slack_auth,
            authenticate_slack,
            clear_slack_auth,
            start_slack_browser_auth,
        ]


def _delete_workspace(workspace: str) -> None:
    """Delete a workspace's token directory from disk.

    Removes the entire ``~/.kiss/third_party_agents/slack/{workspace}/`` directory,
    including the token file and any other workspace-specific files.

    Args:
        workspace: Workspace identifier to delete.
    """
    ws_dir = _SLACK_DIR / workspace
    if not ws_dir.is_dir():
        print(f"Workspace {workspace!r} not found.")
        sys.exit(1)
    shutil.rmtree(ws_dir)
    print(f"Workspace {workspace!r} deleted.")


def _list_workspaces() -> None:
    """Display all authenticated Slack workspaces and their token status.

    Scans ``~/.kiss/third_party_agents/slack/`` for workspace subdirectories
    containing ``token.json`` files.  For each workspace, validates the
    token against the Slack API and prints the workspace name, status,
    team name, and bot user.
    """
    if not _SLACK_DIR.is_dir():
        print("No workspaces found.")
        return
    workspaces: list[str] = []
    for entry in sorted(_SLACK_DIR.iterdir()):
        if entry.is_dir() and (entry / "token.json").is_file():
            workspaces.append(entry.name)
    if not workspaces:
        print("No workspaces found.")
        return
    print(f"{'Workspace':<20} {'Status':<12} {'Team':<20} {'Bot User'}")
    print("-" * 72)
    for ws in workspaces:
        token = _load_token(ws)
        if not token:
            print(f"{ws:<20} {'no token':<12} {'-':<20} -")
            continue
        try:
            client = WebClient(token=token, retry_handlers=[])
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
    token = _load_token(workspace)
    if not token:  # pragma: no branch
        print(
            f"Not authenticated for workspace {workspace!r}. "
            f"Run: kiss-slack --workspace {workspace} -t 'authenticate'"
        )
        sys.exit(1)
    backend._client = WebClient(token=token, retry_handlers=[])
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


if __name__ == "__main__":
    main()
