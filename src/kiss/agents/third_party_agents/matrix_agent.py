# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Matrix Agent — SorcarAgent extension with Matrix protocol tools.

Provides authenticated access to Matrix via matrix-nio. Stores credentials
in ``~/.kiss/third_party_agents/matrix/config.json``.

Usage::

    agent = MatrixAgent()
    agent.run(prompt_template="Send 'Hello!' to #general:matrix.org")
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.third_party_agents._backend_utils import wait_for_matching_message
from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

_MATRIX_DIR = Path.home() / ".kiss" / "third_party_agents" / "matrix"
_config = ChannelConfig(
    _MATRIX_DIR,
    (
        "homeserver_url",
        "access_token",
    ),
)


class MatrixChannelBackend(ToolMethodBackend):
    """Channel backend for Matrix via matrix-nio."""

    def __init__(self) -> None:
        self._client: Any = None
        self._next_batch: str = ""
        self._connection_info: str = ""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None

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
        future = asyncio.run_coroutine_threadsafe(coro, self._ensure_loop())
        return future.result(timeout=timeout)

    def connect(self) -> bool:
        """Authenticate with Matrix using stored config and validate the token."""
        cfg = _config.load()
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
        """Poll for new Matrix messages via sync."""
        if not self._client:  # pragma: no branch
            return [], oldest
        try:
            from nio import RoomMessageText

            async def _sync() -> Any:
                return await self._client.sync(since=self._next_batch or None, timeout=0)

            resp = self._run(_sync())
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
        """Send a Matrix text message."""
        if not self._client:  # pragma: no branch
            return

        async def _send() -> None:
            await self._client.room_send(
                channel_id,
                message_type="m.room.message",
                content={"msgtype": "m.text", "body": text},
            )

        self._run(_send())

    def wait_for_reply(
        self,
        channel_id: str,
        thread_ts: str,
        user_id: str,
        timeout_seconds: float = 300.0,
    ) -> str | None:
        """Poll for a reply from a specific user."""
        return wait_for_matching_message(
            poll=lambda: self.poll_messages(channel_id, "")[0],
            matches=lambda msg: msg.get("user") == user_id,
            extract_text=lambda msg: str(msg.get("text", "")),
            timeout_seconds=timeout_seconds,
            poll_interval=3.0,
        )

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


class MatrixAgent(BaseChannelAgent, SorcarAgent):
    """SorcarAgent extended with Matrix protocol tools."""

    def __init__(self) -> None:
        super().__init__("Matrix Agent")
        self._backend = MatrixChannelBackend()
        cfg = _config.load()
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
                    "Not authenticated with Matrix. Use authenticate_matrix() to configure.\n"
                    "You need: homeserver_url (e.g. https://matrix.org) and access_token.\n"
                    "To get an access token: Element > All Settings > Help & About > "
                    "scroll to 'Access Token', or use the POST /_matrix/client/v3/login API."
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
                _config.save(
                    {
                        "homeserver_url": homeserver_url.strip(),
                        "access_token": access_token.strip(),
                        "device_id": device_id.strip(),
                        "user_id": user_id.strip(),
                    }
                )
                return json.dumps(
                    {
                        "ok": True,
                        "message": "Matrix credentials saved.",
                        "homeserver": homeserver_url,
                    }
                )
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def clear_matrix_auth() -> str:
            """Clear the stored Matrix credentials.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._client = None
            return "Matrix authentication cleared."

        return [check_matrix_auth, authenticate_matrix, clear_matrix_auth]


def _make_backend() -> MatrixChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = MatrixChannelBackend()
    cfg = _config.load()
    if not cfg:  # pragma: no branch
        print("Not authenticated. Run: kiss-matrix -t 'authenticate'")
        sys.exit(1)
    from nio import AsyncClient

    backend._client = AsyncClient(cfg["homeserver_url"])
    backend._client.access_token = cfg["access_token"]
    if cfg.get("device_id"):  # pragma: no branch
        backend._client.device_id = cfg["device_id"]
    if cfg.get("user_id"):  # pragma: no branch
        backend._client.user_id = cfg["user_id"]
    return backend


def main() -> None:
    """Run the MatrixAgent from the command line with chat persistence."""
    channel_main(
        MatrixAgent,
        "kiss-matrix",
        channel_name="Matrix",
        make_backend=_make_backend,
    )


if __name__ == "__main__":
    main()
