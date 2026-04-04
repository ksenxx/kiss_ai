"""Mattermost Agent — StatefulSorcarAgent extension with Mattermost REST API tools.

Provides authenticated access to Mattermost via a personal access token.
Stores config in ``~/.kiss/channels/mattermost/config.json``.

Usage::

    agent = MattermostAgent()
    agent.run(prompt_template="List all channels in the team")
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.sorcar_agent import (
    _build_arg_parser,
    _resolve_task,
    cli_ask_user_question,
    cli_wait_for_user,
)
from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.channels._backend_utils import wait_for_matching_message

logger = logging.getLogger(__name__)

_MATTERMOST_DIR = Path.home() / ".kiss" / "channels" / "mattermost"


def _config_path() -> Path:
    """Return the path to the stored Mattermost config file."""
    return _MATTERMOST_DIR / "config.json"


def _load_config() -> dict[str, str] | None:
    """Load stored Mattermost config from disk."""
    path = _config_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and data.get("url") and data.get("token"):  # pragma: no branch
            return {
                "url": data["url"],
                "token": data["token"],
                "port": str(data.get("port", 443)),
                "scheme": data.get("scheme", "https"),
            }
        return None
    except (json.JSONDecodeError, OSError):
        return None


def _save_config(
    url: str, token: str, port: int = 443, scheme: str = "https"
) -> None:
    """Save Mattermost config to disk with restricted permissions."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "url": url.strip(),
        "token": token.strip(),
        "port": port,
        "scheme": scheme,
    }, indent=2))
    if sys.platform != "win32":  # pragma: no branch
        path.chmod(0o600)


def _clear_config() -> None:
    """Delete the stored Mattermost config."""
    path = _config_path()
    if path.exists():  # pragma: no branch
        path.unlink()


class MattermostChannelBackend:
    """ChannelBackend implementation for Mattermost REST API."""

    def __init__(self) -> None:
        self._driver: Any = None
        self._last_post_time: int = 0
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Authenticate with Mattermost using stored config."""
        cfg = _load_config()
        if not cfg:  # pragma: no branch
            self._connection_info = "No Mattermost config found."
            return False
        try:
            from mattermostdriver import Driver

            self._driver = Driver({
                "url": cfg["url"],
                "token": cfg["token"],
                "port": int(cfg.get("port", 443)),
                "scheme": cfg.get("scheme", "https"),
            })
            self._driver.login()
            me = self._driver.users.get_user("me")
            self._connection_info = f"Authenticated as {me.get('username', '')}"
            self._last_post_time = int(time.time() * 1000)
            return True
        except Exception as e:
            self._connection_info = f"Mattermost connection failed: {e}"
            return False

    @property
    def connection_info(self) -> str:
        """Human-readable connection status string."""
        return self._connection_info

    def find_channel(self, name: str) -> str | None:
        """Return channel name as channel ID."""
        return name if name else None

    def find_user(self, username: str) -> str | None:
        """Return username as user ID."""
        return username if username else None

    def join_channel(self, channel_id: str) -> None:
        """No-op for Mattermost (bots are added to channels by admins)."""

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll Mattermost channel for new posts."""
        if not self._driver or not channel_id:  # pragma: no branch
            return [], oldest
        try:
            since = int(oldest) if oldest else self._last_post_time
            posts = self._driver.posts.get_posts_for_channel(
                channel_id, params={"since": since}
            )
            order = posts.get("order", [])
            posts_data = posts.get("posts", {})
            messages: list[dict[str, Any]] = []
            new_oldest = oldest
            for post_id in reversed(order):  # pragma: no branch
                post = posts_data.get(post_id, {})
                ts = str(post.get("create_at", ""))
                new_oldest = ts
                messages.append({
                    "ts": ts,
                    "user": post.get("user_id", ""),
                    "text": post.get("message", ""),
                    "id": post.get("id", ""),
                })
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

    def disconnect(self) -> None:
        """Release backend resources before stop or reconnect."""

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Check if message is from the bot."""
        return False

    def strip_bot_mention(self, text: str) -> str:
        """Remove bot mentions from text."""
        return text

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

    def list_channels(
        self, team_id: str, page: int = 0, per_page: int = 60
    ) -> str:
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

    def list_channel_posts(
        self, channel_id: str, page: int = 0, per_page: int = 30
    ) -> str:
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
                for post_id in order if post_id in posts_data
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
            return json.dumps({
                "ok": True,
                "id": user.get("id", ""),
                "username": user.get("username", ""),
                "email": user.get("email", ""),
                "first_name": user.get("first_name", ""),
                "last_name": user.get("last_name", ""),
                "roles": user.get("roles", ""),
            })
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
            self._driver.reactions.create_reaction(options={
                "user_id": user_id, "post_id": post_id, "emoji_name": emoji_name
            })
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_tool_methods(self) -> list:
        """Return list of bound tool methods for use by the LLM agent."""
        non_tool = frozenset({
            "connect", "find_channel", "find_user", "join_channel",
            "poll_messages", "send_message", "wait_for_reply",
            "is_from_bot", "strip_bot_mention", "get_tool_methods",
        })
        return [
            getattr(self, name)
            for name in sorted(dir(self))
            if not name.startswith("_")
            and name not in non_tool
            and callable(getattr(self, name))
        ]


class MattermostAgent(StatefulSorcarAgent):
    """StatefulSorcarAgent extended with Mattermost REST API tools."""

    def __init__(self) -> None:
        super().__init__("Mattermost Agent")
        self._backend = MattermostChannelBackend()
        cfg = _load_config()
        if cfg:  # pragma: no branch
            try:
                from mattermostdriver import Driver

                self._backend._driver = Driver({
                    "url": cfg["url"],
                    "token": cfg["token"],
                    "port": int(cfg.get("port", 443)),
                    "scheme": cfg.get("scheme", "https"),
                })
                self._backend._driver.login()
            except Exception:
                pass

    def _get_tools(self) -> list:
        """Return SorcarAgent tools + Mattermost auth tools + API tools."""
        tools = super()._get_tools()
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

                driver = Driver({
                    "url": url.strip(),
                    "token": token.strip(),
                    "port": port,
                    "scheme": scheme,
                })
                driver.login()
                me = driver.users.get_user("me")
                _save_config(url, token, port, scheme)
                agent._backend._driver = driver
                return json.dumps({
                    "ok": True,
                    "message": "Mattermost credentials saved.",
                    "username": me.get("username", ""),
                })
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def clear_mattermost_auth() -> str:
            """Clear the stored Mattermost credentials.

            Returns:
                Status message.
            """
            _clear_config()
            agent._backend._driver = None
            return "Mattermost authentication cleared."

        tools.extend([check_mattermost_auth, authenticate_mattermost, clear_mattermost_auth])

        if agent._backend._driver is not None:  # pragma: no branch
            tools.extend(agent._backend.get_tool_methods())

        return tools


def main() -> None:
    """Run the MattermostAgent from the command line with chat persistence."""
    import sys
    import time as time_mod

    if len(sys.argv) <= 1:  # pragma: no branch
        print("Usage: kiss-mattermost [-m MODEL] [-t TASK] [-n] [--daemon]")
        sys.exit(1)

    parser = _build_arg_parser()
    parser.add_argument("-n", "--new", action="store_true", help="Start a new chat session")
    parser.add_argument("--daemon", action="store_true", help="Run as background daemon")
    parser.add_argument("--daemon-channel", default="", help="Channel ID to monitor")
    parser.add_argument("--allow-users", default="", help="Comma-separated user IDs to allow")
    args = parser.parse_args()

    if args.daemon:  # pragma: no branch
        from kiss.channels.background_agent import ChannelDaemon

        backend = MattermostChannelBackend()
        cfg = _load_config()
        if not cfg:  # pragma: no branch
            print("Not authenticated. Run: kiss-mattermost -t 'authenticate'")
            sys.exit(1)
        from mattermostdriver import Driver
        backend._driver = Driver({
            "url": cfg["url"], "token": cfg["token"],
            "port": int(cfg.get("port", 443)), "scheme": cfg.get("scheme", "https")
        })
        backend._driver.login()
        allow_users = [u.strip() for u in args.allow_users.split(",") if u.strip()] or None
        daemon = ChannelDaemon(
            backend=backend,
            channel_name=args.daemon_channel,
            agent_name="Mattermost Background Agent",
            extra_tools=backend.get_tool_methods(),
            model_name=args.model_name,
            max_budget=args.max_budget,
            work_dir=args.work_dir or str(Path.home() / ".kiss" / "daemon_work"),
            allow_users=allow_users,
        )
        print("Starting Mattermost daemon... (Ctrl+C to stop)")
        try:
            daemon.run()
        except KeyboardInterrupt:
            print("Daemon stopped.")
        return

    agent = MattermostAgent()
    task_description = _resolve_task(args)
    work_dir = args.work_dir or str(Path(".").resolve())
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    if args.new:  # pragma: no branch
        agent.new_chat()
    else:
        agent.resume_chat(task_description)

    model_config: dict[str, Any] = {}
    if args.endpoint:  # pragma: no branch
        model_config["base_url"] = args.endpoint

    run_kwargs: dict[str, Any] = {
        "prompt_template": task_description,
        "model_name": args.model_name,
        "max_budget": args.max_budget,
        "model_config": model_config,
        "work_dir": work_dir,
        "headless": args.headless,
        "verbose": args.verbose,
        "wait_for_user_callback": cli_wait_for_user,
        "ask_user_question_callback": cli_ask_user_question,
    }

    start_time = time_mod.time()
    agent.run(**run_kwargs)
    elapsed = time_mod.time() - start_time

    print(f"Time: {elapsed:.1f}s")
    print(f"Cost: ${agent.budget_used:.4f}")
    print(f"Total tokens: {agent.total_tokens_used}")


if __name__ == "__main__":
    main()
