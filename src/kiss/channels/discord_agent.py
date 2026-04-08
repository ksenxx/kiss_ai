"""Discord Agent — StatefulSorcarAgent extension with Discord REST API tools.

Provides authenticated access to Discord via a bot token. Uses the Discord
REST API v10 directly via requests (no discord.py needed). Stores the token
in ``~/.kiss/channels/discord/config.json``.

Usage::

    agent = DiscordAgent()
    agent.run(prompt_template="List all channels in my server")
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.channels._backend_utils import wait_for_matching_message
from kiss.channels._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

_DISCORD_DIR = Path.home() / ".kiss" / "channels" / "discord"
_API_BASE = "https://discord.com/api/v10"
_config = ChannelConfig(_DISCORD_DIR, ("bot_token",))


# ---------------------------------------------------------------------------
# DiscordChannelBackend
# ---------------------------------------------------------------------------


class DiscordChannelBackend(ToolMethodBackend):
    """Channel backend for Discord REST API v10."""

    def __init__(self) -> None:
        self._bot_token: str = ""
        self._connection_info: str = ""
        self._last_message_id: str = ""

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bot {self._bot_token}"}

    def _get(self, path: str, params: dict | None = None) -> Any:  # type: ignore[type-arg]
        resp = requests.get(
            f"{_API_BASE}{path}", headers=self._headers(), params=params, timeout=30
        )
        return resp.json()

    def _post(self, path: str, json_body: dict | None = None) -> Any:  # type: ignore[type-arg]
        resp = requests.post(
            f"{_API_BASE}{path}", headers=self._headers(), json=json_body, timeout=30
        )
        return resp.json()

    def _delete(self, path: str) -> Any:  # type: ignore[type-arg]
        resp = requests.delete(f"{_API_BASE}{path}", headers=self._headers(), timeout=30)
        if resp.status_code == 204:  # pragma: no branch
            return {"ok": True}
        return resp.json()

    def _patch(self, path: str, json_body: dict | None = None) -> Any:  # type: ignore[type-arg]
        resp = requests.patch(
            f"{_API_BASE}{path}", headers=self._headers(), json=json_body, timeout=30
        )
        return resp.json()

    def connect(self) -> bool:
        """Authenticate with Discord using the stored bot token."""
        cfg = _config.load()
        if not cfg:  # pragma: no branch
            self._connection_info = "No Discord token found."
            return False
        self._bot_token = cfg["bot_token"]
        try:
            result = self._get("/users/@me")
            if "id" in result:  # pragma: no branch
                username = result.get("username", "")
                discriminator = result.get("discriminator", "")
                self._connection_info = f"Authenticated as {username}#{discriminator}"
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
        # If already a numeric ID, return as-is
        if name.isdigit():
            return name
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
            if oldest:  # pragma: no branch
                params["after"] = oldest
            else:
                # Snowflake for "1 second ago"
                params["after"] = str((int((time.time() - 1) * 1000) - 1420070400000) << 22)
            result = self._get(f"/channels/{channel_id}/messages", params=params)
            if not isinstance(result, list):  # pragma: no branch
                return [], oldest
            msgs: list[dict[str, Any]] = sorted(result, key=lambda m: m.get("id", ""))
            new_oldest = oldest
            messages = []
            for m in msgs:  # pragma: no branch
                new_oldest = m["id"]
                messages.append(
                    {
                        "ts": m.get("timestamp", ""),
                        "user": m.get("author", {}).get("id", ""),
                        "text": m.get("content", ""),
                        "id": m.get("id", ""),
                    }
                )
            return messages, new_oldest
        except Exception:
            return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a Discord message."""
        target = thread_ts if thread_ts else channel_id
        self._post(f"/channels/{target}/messages", {"content": text})

    def wait_for_reply(
        self,
        channel_id: str,
        thread_ts: str,
        user_id: str,
        timeout_seconds: float = 300.0,
    ) -> str | None:
        """Poll for a reply from a specific user."""
        oldest = self._last_message_id

        def poll() -> list[dict[str, Any]]:
            nonlocal oldest
            msgs, oldest = self.poll_messages(channel_id, oldest)
            return msgs

        return wait_for_matching_message(
            poll=poll,
            matches=lambda msg: msg.get("user") == user_id,
            extract_text=lambda msg: str(msg.get("text", "")),
            timeout_seconds=timeout_seconds,
            poll_interval=2.0,
        )

    def disconnect(self) -> None:
        """Release Discord backend state before stop or reconnect."""
        self._last_message_id = ""

    # -------------------------------------------------------------------
    # Discord API tool methods
    # -------------------------------------------------------------------

    def list_guilds(self, limit: int = 100) -> str:
        """List guilds (servers) the bot is a member of.

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

    def list_channels(self, guild_id: str, channel_type: str = "") -> str:
        """List channels in a guild.

        Args:
            guild_id: Guild (server) ID.
            channel_type: Optional filter by type (0=text, 2=voice, 4=category).

        Returns:
            JSON string with channel list (id, name, type, topic).
        """
        try:
            result = self._get(f"/guilds/{guild_id}/channels")
            if not isinstance(result, list):  # pragma: no branch
                return json.dumps({"ok": False, "error": str(result)})
            channels = [
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
            return json.dumps({"ok": True, "channels": channels}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_channel(self, channel_id: str) -> str:
        """Get information about a channel.

        Args:
            channel_id: Channel ID.

        Returns:
            JSON string with channel details.
        """
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

        Args:
            channel_id: Channel ID.
            content: Message text (up to 2000 chars).
            tts: Text-to-speech flag. Default: False.
            reply_to: Optional message ID to reply to.

        Returns:
            JSON string with ok status and message id.
        """
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
        try:
            self._delete(f"/channels/{channel_id}/messages/{message_id}")
            return json.dumps({"ok": True})
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
        try:
            from urllib.parse import quote

            emoji_url = f"{_API_BASE}/channels/{channel_id}/messages/{message_id}"
            emoji_url += f"/reactions/{quote(emoji)}/@me"
            resp = requests.put(emoji_url, headers=self._headers(), timeout=30)
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


# ---------------------------------------------------------------------------
# DiscordAgent
# ---------------------------------------------------------------------------


class DiscordAgent(BaseChannelAgent, StatefulSorcarAgent):
    """StatefulSorcarAgent extended with Discord REST API tools.

    Example::

        agent = DiscordAgent()
        result = agent.run(prompt_template="List all channels in my server")
    """

    def __init__(self) -> None:
        super().__init__("Discord Agent")
        self._backend = DiscordChannelBackend()
        cfg = _config.load()
        if cfg:  # pragma: no branch
            self._backend._bot_token = cfg["bot_token"]

    def run(self, **kwargs: Any) -> str:  # type: ignore[override]
        """Run with Discord-specific system prompt encouraging browser-based auth."""
        channel_prompt = (
            "\n\n## Discord Authentication\n"
            "If authentication is needed, call start_discord_browser_auth() to open "
            "the Discord Developer Portal, then use browser tools to create an app, "
            "add a bot, and copy the bot token. Complete these steps autonomously — "
            "do NOT instruct the user to do them manually."
        )
        kwargs["system_prompt"] = (kwargs.get("system_prompt") or "") + channel_prompt
        return super().run(**kwargs)

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._bot_token)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_discord_auth() -> str:
            """Check if the Discord bot token is configured and valid.

            Returns:
                Authentication status or instructions for how to authenticate.
            """
            if not agent._backend._bot_token:  # pragma: no branch
                return (
                    "Not authenticated with Discord. Call start_discord_browser_auth() "
                    "to open the Discord Developer Portal in the browser and create a "
                    "bot autonomously, then call authenticate_discord(bot_token=...) "
                    "with the token you retrieve."
                )
            try:
                result = agent._backend._get("/users/@me")
                if "id" in result:  # pragma: no branch
                    return json.dumps(
                        {
                            "ok": True,
                            "username": result.get("username", ""),
                            "id": result.get("id", ""),
                        }
                    )
                return json.dumps({"ok": False, "error": str(result)})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_discord(
            bot_token: str,
            application_id: str = "",
            guild_ids: str = "",
        ) -> str:
            """Store and validate a Discord bot token.

            Args:
                bot_token: Discord bot token from the Developer Portal.
                application_id: Optional application ID.
                guild_ids: Optional comma-separated guild IDs.

            Returns:
                Validation result with bot info, or error message.
            """
            bot_token = bot_token.strip()
            if not bot_token:  # pragma: no branch
                return "bot_token cannot be empty."
            agent._backend._bot_token = bot_token
            try:
                result = agent._backend._get("/users/@me")
                if "id" in result:  # pragma: no branch
                    _config.save(
                        {
                            "bot_token": bot_token.strip(),
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
                agent._backend._bot_token = ""
                return json.dumps({"ok": False, "error": str(result)})
            except Exception as e:
                agent._backend._bot_token = ""
                return json.dumps({"ok": False, "error": str(e)})

        def clear_discord_auth() -> str:
            """Clear the stored Discord bot token.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._bot_token = ""
            return "Discord authentication cleared."

        def start_discord_browser_auth() -> str:
            """Begin automated Discord bot creation and token retrieval via browser.

            Navigates to the Discord Developer Portal. Use your browser tools
            (go_to_url, click, type_text) to complete the following steps autonomously:
            1. Click "New Application", give it a name, and create it.
            2. Go to the "Bot" section, click "Add Bot" (or "Reset Token").
            3. Copy the bot token shown.
            4. Enable any required Privileged Gateway Intents (Message Content, etc.).
            5. Call authenticate_discord(bot_token=<the token>).
            Use ask_user_browser_action() for any login screens.

            Returns:
                Page content of the Discord Developer Portal to begin navigation.
            """
            if agent.web_use_tool is None:  # pragma: no branch
                return (
                    "Browser not available. Use authenticate_discord(bot_token=...) "
                    "with a token from https://discord.com/developers/applications."
                )
            return agent.web_use_tool.go_to_url("https://discord.com/developers/applications")

        return [
            check_discord_auth,
            authenticate_discord,
            clear_discord_auth,
            start_discord_browser_auth,
        ]


def _make_backend() -> DiscordChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = DiscordChannelBackend()
    cfg = _config.load()
    if not cfg:  # pragma: no branch
        print("Not authenticated. Run: kiss-discord -t 'authenticate'")
        sys.exit(1)
    backend._bot_token = cfg["bot_token"]
    return backend


def main() -> None:
    """Run the DiscordAgent from the command line with chat persistence."""
    channel_main(
        DiscordAgent,
        "kiss-discord",
        channel_name="Discord",
        make_backend=_make_backend,
    )


if __name__ == "__main__":
    main()
