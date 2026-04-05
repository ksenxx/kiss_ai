"""Matrix Agent — StatefulSorcarAgent extension with Matrix protocol tools.

Provides authenticated access to Matrix via matrix-nio. Stores credentials
in ``~/.kiss/channels/matrix/config.json``.

Usage::

    agent = MatrixAgent()
    agent.run(prompt_template="Send 'Hello!' to #general:matrix.org")
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
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

_MATRIX_DIR = Path.home() / ".kiss" / "channels" / "matrix"


def _config_path() -> Path:
    """Return the path to the stored Matrix config file."""
    return _MATRIX_DIR / "config.json"


def _load_config() -> dict[str, str] | None:
    """Load stored Matrix config from disk."""
    path = _config_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if (  # pragma: no branch
            isinstance(data, dict)
            and data.get("homeserver_url")
            and data.get("access_token")
        ):
            return {
                "homeserver_url": data["homeserver_url"],
                "access_token": data["access_token"],
                "device_id": data.get("device_id", ""),
                "user_id": data.get("user_id", ""),
            }
        return None
    except (json.JSONDecodeError, OSError):
        return None


def _save_config(
    homeserver_url: str,
    access_token: str,
    device_id: str = "",
    user_id: str = "",
) -> None:
    """Save Matrix config to disk with restricted permissions."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "homeserver_url": homeserver_url.strip(),
        "access_token": access_token.strip(),
        "device_id": device_id.strip(),
        "user_id": user_id.strip(),
    }, indent=2))
    if sys.platform != "win32":  # pragma: no branch
        path.chmod(0o600)


def _clear_config() -> None:
    """Delete the stored Matrix config."""
    path = _config_path()
    if path.exists():  # pragma: no branch
        path.unlink()


class MatrixChannelBackend:
    """ChannelBackend implementation for Matrix via matrix-nio."""

    def __init__(self) -> None:
        self._client: Any = None
        self._next_batch: str = ""
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Authenticate with Matrix using stored config."""
        cfg = _load_config()
        if not cfg:  # pragma: no branch
            self._connection_info = "No Matrix config found."
            return False
        try:
            from nio import AsyncClient

            self._client = AsyncClient(cfg["homeserver_url"])
            self._client.access_token = cfg["access_token"]
            if cfg.get("device_id"):  # pragma: no branch
                self._client.device_id = cfg["device_id"]
            if cfg.get("user_id"):  # pragma: no branch
                self._client.user_id = cfg["user_id"]
            self._connection_info = f"Connected to {cfg['homeserver_url']}"
            return True
        except Exception as e:
            self._connection_info = f"Matrix connection failed: {e}"
            return False

    @property
    def connection_info(self) -> str:
        """Human-readable connection status string."""
        return self._connection_info

    def find_channel(self, name: str) -> str | None:
        """Return room alias or ID."""
        return name if name else None

    def find_user(self, username: str) -> str | None:
        """Return username as user ID."""
        return username if username else None

    def join_channel(self, channel_id: str) -> None:
        """Join a Matrix room."""
        if self._client:  # pragma: no branch
            asyncio.run(self._client.join(channel_id))

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll for new Matrix messages via sync."""
        if not self._client:  # pragma: no branch
            return [], oldest
        try:
            from nio import RoomMessageText

            async def _sync() -> Any:
                return await self._client.sync(since=self._next_batch or None, timeout=0)

            resp = asyncio.run(_sync())
            if hasattr(resp, "next_batch"):  # pragma: no branch
                self._next_batch = resp.next_batch
            messages: list[dict[str, Any]] = []
            if channel_id and hasattr(resp, "rooms"):  # pragma: no branch
                room = resp.rooms.join.get(channel_id)
                if room:  # pragma: no branch
                    for event in room.timeline.events:  # pragma: no branch
                        if isinstance(event, RoomMessageText):  # pragma: no branch
                            messages.append({
                                "ts": str(event.server_timestamp),
                                "user": event.sender,
                                "text": event.body,
                                "event_id": event.event_id,
                            })
            return messages, self._next_batch
        except Exception:
            return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a Matrix text message."""
        if not self._client:  # pragma: no branch
            return

        async def _send() -> None:
            await self._client.room_send(
                channel_id,
                message_type="m.room.message",
                content={"msgtype": "m.text", "body": text},
            )

        asyncio.run(_send())

    def wait_for_reply(
        self,
        channel_id: str,
        thread_ts: str,
        user_id: str,
        timeout_seconds: float = 300.0,
        stop_event: threading.Event | None = None,
    ) -> str | None:
        """Poll for a reply from a specific user."""
        return wait_for_matching_message(
            poll=lambda: self.poll_messages(channel_id, "")[0],
            matches=lambda msg: msg.get("user") == user_id,
            extract_text=lambda msg: str(msg.get("text", "")),
            timeout_seconds=timeout_seconds,
            stop_event=stop_event,
            poll_interval=3.0,
        )

    def disconnect(self) -> None:
        """Release backend resources before stop or reconnect."""

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Check if message is from the bot."""
        if self._client and hasattr(self._client, "user_id"):  # pragma: no branch
            return bool(msg.get("user", "") == self._client.user_id)
        return False

    def strip_bot_mention(self, text: str) -> str:
        """Remove bot mentions from text."""
        return text

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

            resp = asyncio.run(_get())
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

            resp = asyncio.run(_join())
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

            asyncio.run(_leave())
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

            resp = asyncio.run(_send())
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

            resp = asyncio.run(_send())
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

            resp = asyncio.run(_get())
            members = [
                {"user_id": uid, "display_name": m.display_name or ""}
                for uid, m in getattr(resp, "members", {}).items()
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

            asyncio.run(_invite())
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

            asyncio.run(_kick())
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
            async def _create() -> Any:
                return await self._client.room_create(
                    name=name,
                    topic=topic,
                    is_direct=False,
                    visibility="public" if is_public else "private",
                    alias=alias or None,
                )

            resp = asyncio.run(_create())
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

            resp = asyncio.run(_get())
            return json.dumps({
                "ok": True,
                "display_name": getattr(resp, "displayname", ""),
                "avatar_url": getattr(resp, "avatar_url", ""),
            })
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


class MatrixAgent(StatefulSorcarAgent):
    """StatefulSorcarAgent extended with Matrix protocol tools."""

    def __init__(self) -> None:
        super().__init__("Matrix Agent")
        self._backend = MatrixChannelBackend()
        cfg = _load_config()
        if cfg:  # pragma: no branch
            try:
                from nio import AsyncClient

                self._backend._client = AsyncClient(cfg["homeserver_url"])
                self._backend._client.access_token = cfg["access_token"]
                if cfg.get("device_id"):  # pragma: no branch
                    self._backend._client.device_id = cfg["device_id"]
                if cfg.get("user_id"):  # pragma: no branch
                    self._backend._client.user_id = cfg["user_id"]
            except Exception:
                pass

    def _get_tools(self) -> list:
        """Return SorcarAgent tools + Matrix auth tools + Matrix API tools."""
        tools = super()._get_tools()
        agent = self

        def check_matrix_auth() -> str:
            """Check if Matrix credentials are configured and valid.

            Returns:
                Authentication status or instructions.
            """
            if agent._backend._client is None:  # pragma: no branch
                return (
                    "Not authenticated with Matrix. Use authenticate_matrix() to configure.\n"
                    "You need: homeserver_url, access_token (from Element > Settings > Security)."
                )
            try:
                resp = agent._backend.list_rooms()
                data = json.loads(resp)
                if data.get("ok"):  # pragma: no branch
                    return json.dumps({"ok": True, "room_count": len(data.get("rooms", []))})
                return resp
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_matrix(
            homeserver_url: str,
            access_token: str,
            device_id: str = "",
            user_id: str = "",
        ) -> str:
            """Store Matrix credentials.

            Args:
                homeserver_url: Matrix homeserver URL (e.g. "https://matrix.org").
                access_token: Matrix access token from Element or login API.
                device_id: Optional device ID.
                user_id: Optional user ID (@user:server.org).

            Returns:
                Authentication result or error message.
            """
            for val, name in [(homeserver_url, "homeserver_url"), (access_token, "access_token")]:
                if not val.strip():  # pragma: no branch
                    return f"{name} cannot be empty."
            try:
                from nio import AsyncClient

                client = AsyncClient(homeserver_url.strip())
                client.access_token = access_token.strip()
                if device_id:  # pragma: no branch
                    client.device_id = device_id.strip()
                if user_id:  # pragma: no branch
                    client.user_id = user_id.strip()
                agent._backend._client = client
                _save_config(homeserver_url, access_token, device_id, user_id)
                return json.dumps({
                    "ok": True,
                    "message": "Matrix credentials saved.",
                    "homeserver": homeserver_url,
                })
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def clear_matrix_auth() -> str:
            """Clear the stored Matrix credentials.

            Returns:
                Status message.
            """
            _clear_config()
            agent._backend._client = None
            return "Matrix authentication cleared."

        tools.extend([check_matrix_auth, authenticate_matrix, clear_matrix_auth])

        if agent._backend._client is not None:  # pragma: no branch
            tools.extend(agent._backend.get_tool_methods())

        return tools


def main() -> None:
    """Run the MatrixAgent from the command line with chat persistence."""
    import sys
    import time as time_mod

    if len(sys.argv) <= 1:  # pragma: no branch
        print("Usage: kiss-matrix [-m MODEL] [-t TASK] [-n] [--daemon]")
        sys.exit(1)

    parser = _build_arg_parser()
    parser.add_argument("-n", "--new", action="store_true", help="Start a new chat session")
    parser.add_argument("--daemon", action="store_true", help="Run as background daemon")
    parser.add_argument("--daemon-channel", default="", help="Room ID to monitor")
    parser.add_argument("--allow-users", default="", help="Comma-separated user IDs to allow")
    args = parser.parse_args()

    if args.daemon:  # pragma: no branch
        from kiss.channels.background_agent import ChannelDaemon

        backend = MatrixChannelBackend()
        cfg = _load_config()
        if not cfg:  # pragma: no branch
            print("Not authenticated. Run: kiss-matrix -t 'authenticate'")
            sys.exit(1)
        from nio import AsyncClient
        backend._client = AsyncClient(cfg["homeserver_url"])
        backend._client.access_token = cfg["access_token"]
        allow_users = [u.strip() for u in args.allow_users.split(",") if u.strip()] or None
        daemon = ChannelDaemon(
            backend=backend,
            channel_name=args.daemon_channel,
            agent_name="Matrix Background Agent",
            extra_tools=backend.get_tool_methods(),
            model_name=args.model_name,
            max_budget=args.max_budget,
            work_dir=args.work_dir or str(Path.home() / ".kiss" / "daemon_work"),
            allow_users=allow_users,
        )
        print("Starting Matrix daemon... (Ctrl+C to stop)")
        try:
            daemon.run()
        except KeyboardInterrupt:
            print("Daemon stopped.")
        return

    agent = MatrixAgent()
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
