"""Mattermost Agent — StatefulSorcarAgent extension with Mattermost REST API tools.

Provides authenticated access to Mattermost via a personal access token.
Stores config in ``~/.kiss/channels/mattermost/config.json``.

Usage::

    agent = MattermostAgent()
    agent.run(prompt_template="List all channels in the team")
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.channels._backend_utils import wait_for_matching_message
from kiss.channels._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

_MATTERMOST_DIR = Path.home() / ".kiss" / "channels" / "mattermost"
_config = ChannelConfig(
    _MATTERMOST_DIR,
    (
        "url",
        "token",
    ),
)


class MattermostChannelBackend(ToolMethodBackend):
    """Channel backend for Mattermost REST API."""

    def __init__(self) -> None:
        self._driver: Any = None
        self._last_post_time: int = 0
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Authenticate with Mattermost using stored config."""
        cfg = _config.load()
        if not cfg:  # pragma: no branch
            self._connection_info = "No Mattermost config found."
            return False
        try:
            from mattermostdriver import Driver

            self._driver = Driver(
                {
                    "url": cfg["url"],
                    "token": cfg["token"],
                    "port": int(cfg.get("port", 443)),
                    "scheme": cfg.get("scheme", "https"),
                }
            )
            self._driver.login()
            me = self._driver.users.get_user("me")
            self._connection_info = f"Authenticated as {me.get('username', '')}"
            self._last_post_time = int(time.time() * 1000)
            return True
        except Exception as e:
            self._connection_info = f"Mattermost connection failed: {e}"
            return False

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll Mattermost channel for new posts."""
        if not self._driver or not channel_id:  # pragma: no branch
            return [], oldest
        try:
            since = int(oldest) if oldest else self._last_post_time
            posts = self._driver.posts.get_posts_for_channel(channel_id, params={"since": since})
            order = posts.get("order", [])
            posts_data = posts.get("posts", {})
            messages: list[dict[str, Any]] = []
            new_oldest = oldest
            for post_id in reversed(order):  # pragma: no branch
                post = posts_data.get(post_id, {})
                ts = str(post.get("create_at", ""))
                new_oldest = ts
                messages.append(
                    {
                        "ts": ts,
                        "user": post.get("user_id", ""),
                        "text": post.get("message", ""),
                        "id": post.get("id", ""),
                    }
                )
            if messages:  # pragma: no branch
                self._last_post_time = int(new_oldest) + 1
            return messages, new_oldest
        except Exception:
            return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a Mattermost post."""
        if not self._driver:  # pragma: no branch
            return
        post: dict[str, Any] = {"channel_id": channel_id, "message": text}
        if thread_ts:  # pragma: no branch
            post["root_id"] = thread_ts
        self._driver.posts.create_post(options=post)

    def wait_for_reply(
        self,
        channel_id: str,
        thread_ts: str,
        user_id: str,
        timeout_seconds: float = 300.0,
        stop_event: threading.Event | None = None,
    ) -> str | None:
        """Poll for a reply from a specific user."""
        oldest = str(self._last_post_time)

        def poll() -> list[dict[str, Any]]:
            nonlocal oldest
            msgs, oldest = self.poll_messages(channel_id, oldest)
            return msgs

        return wait_for_matching_message(
            poll=poll,
            matches=lambda msg: msg.get("user") == user_id,
            extract_text=lambda msg: str(msg.get("text", "")),
            timeout_seconds=timeout_seconds,
            stop_event=stop_event,
            poll_interval=2.0,
        )

    def list_teams(self) -> str:
        """List Mattermost teams.

        Returns:
            JSON string with team list (id, name, display_name).
        """
        assert self._driver is not None
        try:
            teams = self._driver.teams.get_teams()
            result = [
                {
                    "id": t.get("id", ""),
                    "name": t.get("name", ""),
                    "display_name": t.get("display_name", ""),
                }
                for t in teams
            ]
            return json.dumps({"ok": True, "teams": result}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_channels(self, team_id: str, page: int = 0, per_page: int = 60) -> str:
        """List channels in a Mattermost team.

        Args:
            team_id: Team ID.
            page: Page number for pagination. Default: 0.
            per_page: Channels per page. Default: 60.

        Returns:
            JSON string with channel list.
        """
        assert self._driver is not None
        try:
            channels = self._driver.channels.get_channels_for_user(
                "me", team_id, params={"page": page, "per_page": per_page}
            )
            result = [
                {
                    "id": c.get("id", ""),
                    "name": c.get("name", ""),
                    "display_name": c.get("display_name", ""),
                    "type": c.get("type", ""),
                }
                for c in channels
            ]
            return json.dumps({"ok": True, "channels": result}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_channel(self, channel_id: str) -> str:
        """Get information about a Mattermost channel.

        Args:
            channel_id: Channel ID.

        Returns:
            JSON string with channel details.
        """
        assert self._driver is not None
        try:
            channel = self._driver.channels.get_channel(channel_id)
            return json.dumps({"ok": True, **channel}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_channel_posts(self, channel_id: str, page: int = 0, per_page: int = 30) -> str:
        """List posts in a Mattermost channel.

        Args:
            channel_id: Channel ID.
            page: Page number. Default: 0.
            per_page: Posts per page. Default: 30.

        Returns:
            JSON string with post list.
        """
        assert self._driver is not None
        try:
            posts = self._driver.posts.get_posts_for_channel(
                channel_id, params={"page": page, "per_page": per_page}
            )
            order = posts.get("order", [])
            posts_data = posts.get("posts", {})
            result = [
                {
                    "id": post_id,
                    "message": posts_data[post_id].get("message", ""),
                    "user_id": posts_data[post_id].get("user_id", ""),
                    "create_at": posts_data[post_id].get("create_at", 0),
                }
                for post_id in order
                if post_id in posts_data
            ]
            return json.dumps({"ok": True, "posts": result}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def create_post(
        self, channel_id: str, message: str, root_id: str = "", file_ids: str = ""
    ) -> str:
        """Create a post in a Mattermost channel.

        Args:
            channel_id: Channel ID.
            message: Post message text.
            root_id: Root post ID if this is a reply.
            file_ids: Comma-separated file IDs to attach.

        Returns:
            JSON string with ok status and post id.
        """
        assert self._driver is not None
        try:
            post: dict[str, Any] = {"channel_id": channel_id, "message": message}
            if root_id:  # pragma: no branch
                post["root_id"] = root_id
            if file_ids:  # pragma: no branch
                post["file_ids"] = [f.strip() for f in file_ids.split(",") if f.strip()]
            result = self._driver.posts.create_post(options=post)
            return json.dumps({"ok": True, "id": result.get("id", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def delete_post(self, post_id: str) -> str:
        """Delete a Mattermost post.

        Args:
            post_id: Post ID to delete.

        Returns:
            JSON string with ok status.
        """
        assert self._driver is not None
        try:
            self._driver.posts.delete_post(post_id)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_user(self, user_id_or_username: str) -> str:
        """Get a Mattermost user's information.

        Args:
            user_id_or_username: User ID or username. Use "me" for current user.

        Returns:
            JSON string with user details.
        """
        assert self._driver is not None
        try:
            user = self._driver.users.get_user(user_id_or_username)
            return json.dumps(
                {
                    "ok": True,
                    "id": user.get("id", ""),
                    "username": user.get("username", ""),
                    "email": user.get("email", ""),
                    "first_name": user.get("first_name", ""),
                    "last_name": user.get("last_name", ""),
                    "roles": user.get("roles", ""),
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_users(
        self, page: int = 0, per_page: int = 60, in_team: str = "", in_channel: str = ""
    ) -> str:
        """List Mattermost users.

        Args:
            page: Page number. Default: 0.
            per_page: Users per page. Default: 60.
            in_team: Optional team ID to filter by.
            in_channel: Optional channel ID to filter by.

        Returns:
            JSON string with user list.
        """
        assert self._driver is not None
        try:
            params: dict[str, Any] = {"page": page, "per_page": per_page}
            if in_team:  # pragma: no branch
                params["in_team"] = in_team
            if in_channel:  # pragma: no branch
                params["in_channel"] = in_channel
            users = self._driver.users.get_users(params=params)
            result = [
                {
                    "id": u.get("id", ""),
                    "username": u.get("username", ""),
                    "email": u.get("email", ""),
                }
                for u in users
            ]
            return json.dumps({"ok": True, "users": result}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def create_direct_message_channel(self, user1_id: str, user2_id: str) -> str:
        """Create a direct message channel between two users.

        Args:
            user1_id: First user ID.
            user2_id: Second user ID.

        Returns:
            JSON string with channel id.
        """
        assert self._driver is not None
        try:
            channel = self._driver.channels.create_direct_message_channel(
                options=[user1_id, user2_id]
            )
            return json.dumps({"ok": True, "channel_id": channel.get("id", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def add_reaction(self, user_id: str, post_id: str, emoji_name: str) -> str:
        """Add a reaction to a post.

        Args:
            user_id: User ID adding the reaction.
            post_id: Post ID.
            emoji_name: Emoji name (without colons, e.g. "thumbsup").

        Returns:
            JSON string with ok status.
        """
        assert self._driver is not None
        try:
            self._driver.reactions.create_reaction(
                options={"user_id": user_id, "post_id": post_id, "emoji_name": emoji_name}
            )
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class MattermostAgent(BaseChannelAgent, StatefulSorcarAgent):
    """StatefulSorcarAgent extended with Mattermost REST API tools."""

    def __init__(self) -> None:
        super().__init__("Mattermost Agent")
        self._backend = MattermostChannelBackend()
        cfg = _config.load()
        if cfg:  # pragma: no branch
            try:
                from mattermostdriver import Driver

                self._backend._driver = Driver(
                    {
                        "url": cfg["url"],
                        "token": cfg["token"],
                        "port": int(cfg.get("port", 443)),
                        "scheme": cfg.get("scheme", "https"),
                    }
                )
                self._backend._driver.login()
            except Exception:
                pass

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return self._backend._driver is not None

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_mattermost_auth() -> str:
            """Check if Mattermost credentials are configured and valid.

            Returns:
                Authentication status or instructions.
            """
            if agent._backend._driver is None:  # pragma: no branch
                return (
                    "Not authenticated with Mattermost. "
                    "Use authenticate_mattermost() to configure.\n"
                    "You need: server URL and a personal access token."
                )
            try:
                result = json.loads(agent._backend.get_user("me"))
                if result.get("ok"):  # pragma: no branch
                    return json.dumps({"ok": True, "username": result.get("username", "")})
                return json.dumps({"ok": False, "error": "Could not verify authentication."})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_mattermost(
            url: str,
            token: str,
            port: int = 443,
            scheme: str = "https",
        ) -> str:
            """Store and validate Mattermost credentials.

            Args:
                url: Mattermost server URL (e.g. "mattermost.example.com").
                token: Personal access token from Account Settings > Security.
                port: Server port. Default: 443.
                scheme: "https" or "http". Default: "https".

            Returns:
                Validation result or error message.
            """
            for val, name in [(url, "url"), (token, "token")]:  # pragma: no branch
                if not val.strip():  # pragma: no branch
                    return f"{name} cannot be empty."
            try:
                from mattermostdriver import Driver

                driver = Driver(
                    {
                        "url": url.strip(),
                        "token": token.strip(),
                        "port": port,
                        "scheme": scheme,
                    }
                )
                driver.login()
                me = driver.users.get_user("me")
                _config.save(
                    {
                        "url": url.strip(),
                        "token": token.strip(),
                        "port": str(port),
                        "scheme": scheme.strip(),
                    }
                )
                agent._backend._driver = driver
                return json.dumps(
                    {
                        "ok": True,
                        "message": "Mattermost credentials saved.",
                        "username": me.get("username", ""),
                    }
                )
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def clear_mattermost_auth() -> str:
            """Clear the stored Mattermost credentials.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._driver = None
            return "Mattermost authentication cleared."

        return [check_mattermost_auth, authenticate_mattermost, clear_mattermost_auth]


def _make_backend() -> MattermostChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = MattermostChannelBackend()
    cfg = _config.load()
    if not cfg:  # pragma: no branch
        print("Not authenticated. Run: kiss-mattermost -t 'authenticate'")
        sys.exit(1)
    from mattermostdriver import Driver

    backend._driver = Driver(
        {
            "url": cfg["url"],
            "token": cfg["token"],
            "port": int(cfg.get("port", 443)),
            "scheme": cfg.get("scheme", "https"),
        }
    )
    backend._driver.login()
    return backend


def main() -> None:
    """Run the MattermostAgent from the command line with chat persistence."""
    channel_main(
        MattermostAgent,
        "kiss-mattermost",
        channel_name="Mattermost",
        make_backend=_make_backend,
    )


if __name__ == "__main__":
    main()
