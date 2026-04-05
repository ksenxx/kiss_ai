"""Twitch Agent — StatefulSorcarAgent extension with Twitch Helix API + Chat tools.

Provides authenticated access to Twitch via OAuth2 tokens. Uses requests
for Helix API and twitchio for chat. Stores config in
``~/.kiss/channels/twitch/config.json``.

Usage::

    agent = TwitchAgent()
    agent.run(prompt_template="Get stream info for channel 'shroud'")
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

import requests

from kiss.agents.sorcar.sorcar_agent import (
    _build_arg_parser,
    _resolve_task,
    cli_ask_user_question,
    cli_wait_for_user,
)
from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent

_TWITCH_DIR = Path.home() / ".kiss" / "channels" / "twitch"
_HELIX_BASE = "https://api.twitch.tv/helix"


def _config_path() -> Path:
    """Return the path to the stored Twitch config file."""
    return _TWITCH_DIR / "config.json"


def _load_config() -> dict[str, str] | None:
    """Load stored Twitch config from disk."""
    path = _config_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if (  # pragma: no branch
            isinstance(data, dict) and data.get("client_id") and data.get("access_token")
        ):
            return {
                "client_id": data["client_id"],
                "client_secret": data.get("client_secret", ""),
                "access_token": data["access_token"],
                "channel_name": data.get("channel_name", ""),
            }
        return None
    except (json.JSONDecodeError, OSError):
        return None


def _save_config(
    client_id: str,
    client_secret: str,
    access_token: str,
    channel_name: str = "",
) -> None:
    """Save Twitch config to disk with restricted permissions."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "client_id": client_id.strip(),
        "client_secret": client_secret.strip(),
        "access_token": access_token.strip(),
        "channel_name": channel_name.strip(),
    }, indent=2))
    if sys.platform != "win32":  # pragma: no branch
        path.chmod(0o600)


def _clear_config() -> None:
    """Delete the stored Twitch config."""
    path = _config_path()
    if path.exists():  # pragma: no branch
        path.unlink()


class TwitchChannelBackend:
    """ChannelBackend implementation for Twitch Helix API."""

    def __init__(self) -> None:
        self._client_id: str = ""
        self._access_token: str = ""
        self._connection_info: str = ""

    def _headers(self) -> dict[str, str]:
        return {
            "Client-ID": self._client_id,
            "Authorization": f"Bearer {self._access_token}",
        }

    def _get(self, path: str, params: dict | None = None) -> dict[str, Any]:  # type: ignore[type-arg]
        resp = requests.get(
            f"{_HELIX_BASE}{path}", headers=self._headers(), params=params, timeout=30
        )
        return resp.json()  # type: ignore[no-any-return]

    def _post(self, path: str, json_body: dict | None = None) -> dict[str, Any]:  # type: ignore[type-arg]
        resp = requests.post(
            f"{_HELIX_BASE}{path}", headers=self._headers(), json=json_body, timeout=30
        )
        return resp.json() if resp.content else {"ok": True}  # type: ignore[no-any-return]

    def connect(self) -> bool:
        """Authenticate with Twitch using stored config."""
        cfg = _load_config()
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
                self._connection_info = f"Authenticated as {name}"
                return True
            self._connection_info = f"Twitch auth failed: {result}"
            return False
        except Exception as e:
            self._connection_info = f"Twitch connection failed: {e}"
            return False

    @property
    def connection_info(self) -> str:
        """Human-readable connection status string."""
        return self._connection_info

    def find_channel(self, name: str) -> str | None:
        """Return channel name."""
        return name if name else None

    def find_user(self, username: str) -> str | None:
        """Return username."""
        return username if username else None

    def join_channel(self, channel_id: str) -> None:
        """No-op for Twitch."""

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll for Twitch events (basic REST polling)."""
        return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a Twitch chat message."""
        # Requires broadcaster_id and sender_id
        self._post("/chat/messages", {"broadcaster_id": channel_id, "message": text})

    def wait_for_reply(
        self,
        channel_id: str,
        thread_ts: str,
        user_id: str,
        timeout_seconds: float = 300.0,
        stop_event: threading.Event | None = None,
    ) -> str | None:
        """Reply waiting is not currently supported for Twitch."""
        return None

    def disconnect(self) -> None:
        """Release backend resources before stop or reconnect."""

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Check if message is from the bot."""
        return False

    def strip_bot_mention(self, text: str) -> str:
        """Remove bot mentions from text."""
        return text

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
            return json.dumps({
                "ok": True,
                "live": True,
                "title": stream.get("title", ""),
                "game_name": stream.get("game_name", ""),
                "viewer_count": stream.get("viewer_count", 0),
                "started_at": stream.get("started_at", ""),
                "language": stream.get("language", ""),
            })
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

    def send_chat_message(
        self, broadcaster_id: str, sender_id: str, message: str
    ) -> str:
        """Send a message to a Twitch chat.

        Args:
            broadcaster_id: Broadcaster channel ID.
            sender_id: Sender user ID.
            message: Message text.

        Returns:
            JSON string with ok status.
        """
        try:
            result = self._post("/chat/messages", {
                "broadcaster_id": broadcaster_id,
                "sender_id": sender_id,
                "message": message,
            })
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

    def search_channels(self, query: str, limit: int = 10) -> str:
        """Search for Twitch channels by name.

        Args:
            query: Search query.
            limit: Maximum channels to return. Default: 10.

        Returns:
            JSON string with matching channels.
        """
        try:
            result = self._get("/search/channels", params={"query": query, "first": limit})
            return json.dumps({"ok": True, "channels": result.get("data", [])}, indent=2)[:8000]
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

    def get_tool_methods(self) -> list:
        """Return list of bound tool methods for use by the LLM agent."""
        non_tool = frozenset({
            "connect", "find_channel", "find_user", "join_channel",
            "poll_messages", "send_message", "wait_for_reply",
            "is_from_bot", "strip_bot_mention", "disconnect", "get_tool_methods",
        })
        return [
            getattr(self, name)
            for name in sorted(dir(self))
            if not name.startswith("_")
            and name not in non_tool
            and callable(getattr(self, name))
        ]


class TwitchAgent(StatefulSorcarAgent):
    """StatefulSorcarAgent extended with Twitch Helix API tools."""

    def __init__(self) -> None:
        super().__init__("Twitch Agent")
        self._backend = TwitchChannelBackend()
        cfg = _load_config()
        if cfg:  # pragma: no branch
            self._backend._client_id = cfg["client_id"]
            self._backend._access_token = cfg["access_token"]

    def _get_tools(self) -> list:
        """Return SorcarAgent tools + Twitch auth tools + Twitch API tools."""
        tools = super()._get_tools()
        agent = self

        def check_twitch_auth() -> str:
            """Check if Twitch credentials are configured and valid.

            Returns:
                Authentication status or instructions.
            """
            if not agent._backend._client_id:  # pragma: no branch
                return (
                    "Not authenticated with Twitch. Use authenticate_twitch() to configure.\n"
                    "You need client_id and access_token from https://dev.twitch.tv/console."
                )
            try:
                result = agent._backend._get("/users")
                if "data" in result:  # pragma: no branch
                    users = result["data"]
                    return json.dumps({
                        "ok": True,
                        "login": users[0].get("login", "") if users else "",
                    })
                return json.dumps({"ok": False, "error": str(result)})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_twitch(
            client_id: str,
            client_secret: str,
            access_token: str,
            channel_name: str = "",
        ) -> str:
            """Store and validate Twitch API credentials.

            Args:
                client_id: Twitch app client ID from dev console.
                client_secret: Twitch app client secret.
                access_token: OAuth2 access token (user or app token).
                channel_name: Default channel to monitor. Optional.

            Returns:
                Validation result or error message.
            """
            for val, name in [(client_id, "client_id"), (access_token, "access_token")]:
                if not val.strip():  # pragma: no branch
                    return f"{name} cannot be empty."
            agent._backend._client_id = client_id.strip()
            agent._backend._access_token = access_token.strip()
            try:
                result = agent._backend._get("/users")
                if "data" in result:  # pragma: no branch
                    _save_config(client_id, client_secret, access_token, channel_name)
                    return json.dumps({
                        "ok": True,
                        "message": "Twitch credentials saved.",
                        "login": result["data"][0].get("login", "") if result["data"] else "",
                    })
                return json.dumps({"ok": False, "error": str(result)})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def clear_twitch_auth() -> str:
            """Clear the stored Twitch credentials.

            Returns:
                Status message.
            """
            _clear_config()
            agent._backend._client_id = ""
            agent._backend._access_token = ""
            return "Twitch authentication cleared."

        tools.extend([check_twitch_auth, authenticate_twitch, clear_twitch_auth])

        if agent._backend._client_id:  # pragma: no branch
            tools.extend(agent._backend.get_tool_methods())

        return tools


def main() -> None:
    """Run the TwitchAgent from the command line with chat persistence."""
    import sys
    import time as time_mod

    if len(sys.argv) <= 1:  # pragma: no branch
        print("Usage: kiss-twitch [-m MODEL] [-t TASK] [-n]")
        sys.exit(1)

    parser = _build_arg_parser()
    parser.add_argument("-n", "--new", action="store_true", help="Start a new chat session")
    args = parser.parse_args()

    agent = TwitchAgent()
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
