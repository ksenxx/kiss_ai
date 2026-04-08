"""Nextcloud Talk Agent — StatefulSorcarAgent extension with Nextcloud Talk API tools.

Provides authenticated access to Nextcloud Talk via username/password.
Stores config in ``~/.kiss/channels/nextcloud/config.json``.

Usage::

    agent = NextcloudTalkAgent()
    agent.run(prompt_template="List all rooms")
"""

from __future__ import annotations

import json
import sys
import threading
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

_NEXTCLOUD_DIR = Path.home() / ".kiss" / "channels" / "nextcloud"
_config = ChannelConfig(
    _NEXTCLOUD_DIR,
    (
        "url",
        "username",
    ),
)


class NextcloudTalkChannelBackend(ToolMethodBackend):
    """Channel backend for Nextcloud Talk REST API."""

    def __init__(self) -> None:
        self._url: str = ""
        self._auth: tuple[str, str] = ("", "")
        self._last_message_id: int = 0
        self._connection_info: str = ""

    def _base(self) -> str:
        return f"{self._url}/ocs/v2.php/apps/spreed/api/v4"

    def _headers(self) -> dict[str, str]:
        return {"OCS-APIRequest": "true", "Accept": "application/json"}

    def _get(self, path: str, params: dict | None = None) -> dict[str, Any]:  # type: ignore[type-arg]
        resp = requests.get(
            f"{self._base()}{path}",
            auth=self._auth,
            headers=self._headers(),
            params=params,
            timeout=30,
        )
        return resp.json()  # type: ignore[no-any-return]

    def _post(self, path: str, data: dict | None = None) -> dict[str, Any]:  # type: ignore[type-arg]
        resp = requests.post(
            f"{self._base()}{path}",
            auth=self._auth,
            headers=self._headers(),
            json=data,
            timeout=30,
        )
        return resp.json()  # type: ignore[no-any-return]

    def connect(self) -> bool:
        """Authenticate with Nextcloud Talk."""
        cfg = _config.load()
        if not cfg:  # pragma: no branch
            self._connection_info = "No Nextcloud config found."
            return False
        self._url = cfg["url"]
        self._auth = (cfg["username"], cfg["password"])
        try:
            result = self._get("/room")
            if "ocs" in result:  # pragma: no branch
                self._connection_info = f"Connected to {self._url} as {cfg['username']}"
                return True
            self._connection_info = f"Nextcloud auth failed: {result}"
            return False
        except Exception as e:
            self._connection_info = f"Nextcloud connection failed: {e}"
            return False

    def join_channel(self, channel_id: str) -> None:
        """Join a Nextcloud Talk room."""
        try:
            self._post(f"/room/{channel_id}/participants")
        except Exception:
            pass

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll a Nextcloud Talk room for new messages."""
        if not channel_id:  # pragma: no branch
            return [], oldest
        try:
            params: dict[str, Any] = {
                "lookIntoFuture": 0,
                "limit": limit,
                "lastKnownMessageId": self._last_message_id,
            }
            result = self._get(f"/chat/{channel_id}", params=params)
            msgs = result.get("ocs", {}).get("data", [])
            messages: list[dict[str, Any]] = []
            for msg in msgs:  # pragma: no branch
                msg_id = msg.get("id", 0)
                if msg_id > self._last_message_id:  # pragma: no branch
                    self._last_message_id = msg_id
                messages.append(
                    {
                        "ts": str(msg.get("timestamp", "")),
                        "user": msg.get("actorId", ""),
                        "text": msg.get("message", ""),
                        "id": str(msg_id),
                    }
                )
            return messages, str(self._last_message_id)
        except Exception:
            return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a Nextcloud Talk message."""
        kwargs: dict[str, Any] = {"message": text, "replyTo": 0}
        if thread_ts:  # pragma: no branch
            kwargs["replyTo"] = int(thread_ts)
        self._post(f"/chat/{channel_id}", kwargs)

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

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Check if message is from the bot."""
        return bool(msg.get("user", "") == self._auth[0])

    def list_rooms(self) -> str:
        """List Nextcloud Talk rooms.

        Returns:
            JSON string with room list (token, displayName, type).
        """
        try:
            result = self._get("/room")
            rooms = [
                {
                    "token": r.get("token", ""),
                    "display_name": r.get("displayName", ""),
                    "type": r.get("type", 0),
                    "participants": r.get("participantCount", 0),
                }
                for r in result.get("ocs", {}).get("data", [])
            ]
            return json.dumps({"ok": True, "rooms": rooms}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_room(self, token: str) -> str:
        """Get information about a Nextcloud Talk room.

        Args:
            token: Room token.

        Returns:
            JSON string with room details.
        """
        try:
            result = self._get(f"/room/{token}")
            room_data = result.get("ocs", {}).get("data", {})
            return json.dumps({"ok": True, "room": room_data}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def create_room(self, room_type: int = 3, invite: str = "", room_name: str = "") -> str:
        """Create a Nextcloud Talk room.

        Args:
            room_type: 1=one-to-one, 2=group, 3=public. Default: 3.
            invite: User ID, group ID, or circle ID to invite.
            room_name: Room display name.

        Returns:
            JSON string with room token.
        """
        try:
            data: dict[str, Any] = {"roomType": room_type}
            if invite:  # pragma: no branch
                data["invite"] = invite
            if room_name:  # pragma: no branch
                data["roomName"] = room_name
            result = self._post("/room", data)
            room = result.get("ocs", {}).get("data", {})
            return json.dumps(
                {
                    "ok": True,
                    "token": room.get("token", ""),
                    "name": room.get("displayName", ""),
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_participants(self, token: str) -> str:
        """List participants in a room.

        Args:
            token: Room token.

        Returns:
            JSON string with participant list.
        """
        try:
            result = self._get(f"/room/{token}/participants")
            participants = result.get("ocs", {}).get("data", [])
            return json.dumps({"ok": True, "participants": participants}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_messages(
        self,
        token: str,
        look_into_future: int = 0,
        limit: int = 100,
        last_known_message_id: int = 0,
    ) -> str:
        """List messages in a Nextcloud Talk room.

        Args:
            token: Room token.
            look_into_future: 0 for history, 1 for new messages. Default: 0.
            limit: Maximum messages. Default: 100.
            last_known_message_id: Last message ID seen (for pagination).

        Returns:
            JSON string with message list.
        """
        try:
            params: dict[str, Any] = {
                "lookIntoFuture": look_into_future,
                "limit": limit,
            }
            if last_known_message_id:  # pragma: no branch
                params["lastKnownMessageId"] = last_known_message_id
            result = self._get(f"/chat/{token}", params=params)
            messages = result.get("ocs", {}).get("data", [])
            return json.dumps({"ok": True, "messages": messages}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def post_message(self, token: str, message: str, reply_to: int = 0) -> str:
        """Post a message to a Nextcloud Talk room.

        Args:
            token: Room token.
            message: Message text.
            reply_to: Message ID to reply to. Default: 0 (no reply).

        Returns:
            JSON string with ok status and message id.
        """
        try:
            result = self._post(f"/chat/{token}", {"message": message, "replyTo": reply_to})
            msg_data = result.get("ocs", {}).get("data", {})
            return json.dumps({"ok": True, "id": msg_data.get("id", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def set_room_name(self, token: str, name: str) -> str:
        """Set the name of a Nextcloud Talk room.

        Args:
            token: Room token.
            name: New room name.

        Returns:
            JSON string with ok status.
        """
        try:
            resp = requests.put(
                f"{self._base()}/room/{token}/name",
                auth=self._auth,
                headers=self._headers(),
                json={"roomName": name},
                timeout=30,
            )
            return json.dumps({"ok": resp.status_code == 200})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def delete_message(self, token: str, message_id: int) -> str:
        """Delete a message from a room.

        Args:
            token: Room token.
            message_id: Message ID to delete.

        Returns:
            JSON string with ok status.
        """
        try:
            resp = requests.delete(
                f"{self._base()}/chat/{token}/{message_id}",
                auth=self._auth,
                headers=self._headers(),
                timeout=30,
            )
            return json.dumps({"ok": resp.status_code == 200})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class NextcloudTalkAgent(BaseChannelAgent, StatefulSorcarAgent):
    """StatefulSorcarAgent extended with Nextcloud Talk API tools."""

    def __init__(self) -> None:
        super().__init__("Nextcloud Talk Agent")
        self._backend = NextcloudTalkChannelBackend()
        cfg = _config.load()
        if cfg:  # pragma: no branch
            self._backend._url = cfg["url"]
            self._backend._auth = (cfg["username"], cfg["password"])

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._url)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_nextcloud_auth() -> str:
            """Check if Nextcloud Talk credentials are configured and valid.

            Returns:
                Authentication status or instructions.
            """
            if not agent._backend._url:  # pragma: no branch
                return (
                    "Not authenticated with Nextcloud Talk. "
                    "Use authenticate_nextcloud() to configure.\n"
                    "You need: server URL, username, and password."
                )
            try:
                result = json.loads(agent._backend.list_rooms())
                if result.get("ok"):  # pragma: no branch
                    return json.dumps({"ok": True, "room_count": len(result.get("rooms", []))})
                return json.dumps({"ok": False, "error": "Authentication failed."})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_nextcloud(url: str, username: str, password: str) -> str:
            """Store and validate Nextcloud Talk credentials.

            Args:
                url: Nextcloud server URL (e.g. "https://nextcloud.example.com").
                username: Nextcloud username.
                password: Nextcloud password or app password.

            Returns:
                Validation result or error message.
            """
            for val, name in [(url, "url"), (username, "username"), (password, "password")]:
                if not val.strip():  # pragma: no branch
                    return f"{name} cannot be empty."
            agent._backend._url = url.strip().rstrip("/")
            agent._backend._auth = (username.strip(), password)
            try:
                result = self._backend.list_rooms()
                data = json.loads(result)
                if data.get("ok"):  # pragma: no branch
                    _config.save(
                        {
                            "url": url.strip(),
                            "username": username.strip(),
                            "password": password.strip(),
                        }
                    )
                    return json.dumps({"ok": True, "message": "Nextcloud credentials saved."})
                return json.dumps({"ok": False, "error": "Authentication failed."})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def clear_nextcloud_auth() -> str:
            """Clear the stored Nextcloud credentials.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._url = ""
            agent._backend._auth = ("", "")
            return "Nextcloud authentication cleared."

        return [check_nextcloud_auth, authenticate_nextcloud, clear_nextcloud_auth]


def _make_backend() -> NextcloudTalkChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = NextcloudTalkChannelBackend()
    cfg = _config.load()
    if not cfg:  # pragma: no branch
        print("Not authenticated. Run: kiss-nextcloud -t 'authenticate'")
        sys.exit(1)
    backend._url = cfg["url"]
    backend._auth = (cfg["username"], cfg["password"])
    return backend


def main() -> None:
    """Run the NextcloudTalkAgent from the command line with chat persistence."""
    channel_main(
        NextcloudTalkAgent,
        "kiss-nextcloud",
        channel_name="Nextcloud Talk",
        make_backend=_make_backend,
    )


if __name__ == "__main__":
    main()
