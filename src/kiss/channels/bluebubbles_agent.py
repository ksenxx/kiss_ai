"""BlueBubbles Agent — StatefulSorcarAgent extension with BlueBubbles REST API tools.

Provides access to iMessage via the BlueBubbles server running on a local Mac.
macOS only. Stores config in ``~/.kiss/channels/bluebubbles/config.json``.

Usage::

    agent = BlueBubblesAgent()
    agent.run(prompt_template="List recent iMessage conversations")
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests

from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.channels._backend_utils import wait_for_matching_message
from kiss.channels._channel_agent_utils import (
    BaseChannelAgent,
    ToolMethodBackend,
    channel_main,
    clear_json_config,
    load_json_config,
    save_json_config,
)

_BB_DIR = Path.home() / ".kiss" / "channels" / "bluebubbles"

_PLATFORM_ERROR = json.dumps({
    "ok": False,
    "error": "BlueBubbles requires macOS with a running BlueBubbles server.",
})


def _config_path() -> Path:
    """Return the path to the stored BlueBubbles config file."""
    return _BB_DIR / "config.json"


def _load_config() -> dict[str, str] | None:
    """Load stored Bluebubbles config from disk."""
    return load_json_config(_config_path(), ("server_url", "password",))


def _save_config(server_url: str, password: str) -> None:
    """Save Bluebubbles config to disk with restricted permissions."""
    save_json_config(
        _config_path(),
        {
            "server_url": server_url.strip(),
            "password": password.strip(),
        },
    )


def _clear_config() -> None:
    """Delete the stored Bluebubbles config."""
    clear_json_config(_config_path())


class BlueBubblesChannelBackend(ToolMethodBackend):
    """ChannelBackend implementation for BlueBubbles REST API."""

    def __init__(self) -> None:
        self._server_url: str = ""
        self._password: str = ""
        self._last_ts: float = 0.0
        self._connection_info: str = ""

    def _url(self, path: str) -> str:
        return f"{self._server_url}{path}"

    def _params(self) -> dict[str, str]:
        return {"password": self._password}

    def connect(self) -> bool:
        """Connect to BlueBubbles server."""
        if sys.platform != "darwin":  # pragma: no branch
            self._connection_info = "BlueBubbles requires macOS."
            return False
        cfg = _load_config()
        if not cfg:  # pragma: no branch
            self._connection_info = "No BlueBubbles config found."
            return False
        self._server_url = cfg["server_url"].rstrip("/")
        self._password = cfg["password"]
        try:
            resp = requests.get(self._url("/api/v1/server/info"), params=self._params(), timeout=10)
            data = resp.json()
            if data.get("status") == 200:  # pragma: no branch
                self._connection_info = f"Connected to BlueBubbles at {self._server_url}"
                self._last_ts = time.time() * 1000
                return True
            self._connection_info = f"BlueBubbles auth failed: {data}"
            return False
        except Exception as e:
            self._connection_info = f"BlueBubbles connection failed: {e}"
            return False

    @property
    def connection_info(self) -> str:
        """Human-readable connection status string."""
        return self._connection_info

    def find_channel(self, name: str) -> str | None:
        """Return chat GUID."""
        return name if name else None

    def find_user(self, username: str) -> str | None:
        """Return username as user ID."""
        return username if username else None

    def join_channel(self, channel_id: str) -> None:
        """No-op for BlueBubbles."""

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll BlueBubbles for new messages."""
        try:
            params = {**self._params(), "after": str(int(self._last_ts)), "limit": str(limit)}
            resp = requests.get(self._url("/api/v1/message"), params=params, timeout=10)
            data = resp.json()
            messages: list[dict[str, Any]] = []
            for msg in data.get("data", []):  # pragma: no branch
                ts = float(msg.get("dateCreated", 0))
                if ts > self._last_ts:  # pragma: no branch
                    self._last_ts = ts
                messages.append({
                    "ts": str(ts),
                    "user": msg.get("sender", {}).get("address", ""),
                    "text": msg.get("text", "") or "",
                    "guid": msg.get("guid", ""),
                    "chat_guid": channel_id,
                })
            return messages, oldest
        except Exception:
            return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a BlueBubbles message."""
        requests.post(
            self._url("/api/v1/message/text"),
            params=self._params(),
            json={"chatGuid": channel_id, "message": text, "method": "private-api"},
            timeout=30,
        )

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
        return False

    def strip_bot_mention(self, text: str) -> str:
        """Remove bot mentions from text."""
        return text

    def list_chats(self, limit: int = 25, offset: int = 0) -> str:
        """List recent iMessage conversations.

        Args:
            limit: Maximum chats to return. Default: 25.
            offset: Pagination offset. Default: 0.

        Returns:
            JSON string with chat list.
        """
        if sys.platform != "darwin":  # pragma: no branch
            return _PLATFORM_ERROR
        try:
            resp = requests.get(
                self._url("/api/v1/chat"),
                params={**self._params(), "limit": str(limit), "offset": str(offset)},
                timeout=10,
            )
            data = resp.json()
            chats = [
                {
                    "guid": c.get("guid", ""),
                    "display_name": c.get("displayName", ""),
                    "participants": [p.get("address", "") for p in c.get("participants", [])],
                }
                for c in data.get("data", [])
            ]
            return json.dumps({"ok": True, "chats": chats}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_chat(self, chat_guid: str) -> str:
        """Get a specific iMessage conversation.

        Args:
            chat_guid: Chat GUID (from list_chats).

        Returns:
            JSON string with chat details.
        """
        if sys.platform != "darwin":  # pragma: no branch
            return _PLATFORM_ERROR
        try:
            resp = requests.get(
                self._url(f"/api/v1/chat/{chat_guid}"),
                params=self._params(), timeout=10
            )
            return json.dumps({"ok": True, "chat": resp.json().get("data", {})}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_chat_messages(
        self, chat_guid: str, limit: int = 25, before: str = "", after: str = ""
    ) -> str:
        """Get messages from a specific conversation.

        Args:
            chat_guid: Chat GUID.
            limit: Maximum messages to return. Default: 25.
            before: Return messages before this timestamp (ms).
            after: Return messages after this timestamp (ms).

        Returns:
            JSON string with message list.
        """
        if sys.platform != "darwin":  # pragma: no branch
            return _PLATFORM_ERROR
        try:
            params: dict[str, Any] = {**self._params(), "limit": limit}
            if before:  # pragma: no branch
                params["before"] = before
            if after:  # pragma: no branch
                params["after"] = after
            resp = requests.get(
                self._url(f"/api/v1/chat/{chat_guid}/message"),
                params=params, timeout=10
            )
            messages = [
                {
                    "guid": m.get("guid", ""),
                    "text": m.get("text", "") or "",
                    "sender": m.get("sender", {}).get("address", ""),
                    "date_created": m.get("dateCreated", 0),
                    "is_from_me": m.get("isFromMe", False),
                }
                for m in resp.json().get("data", [])
            ]
            return json.dumps({"ok": True, "messages": messages}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def post_message(self, chat_guid: str, text: str) -> str:
        """Send a message to an iMessage conversation.

        Args:
            chat_guid: Chat GUID to send to.
            text: Message text.

        Returns:
            JSON string with ok status.
        """
        if sys.platform != "darwin":  # pragma: no branch
            return _PLATFORM_ERROR
        try:
            resp = requests.post(
                self._url("/api/v1/message/text"),
                params=self._params(),
                json={"chatGuid": chat_guid, "message": text, "method": "private-api"},
                timeout=30,
            )
            data = resp.json()
            if data.get("status") == 200:  # pragma: no branch
                return json.dumps({"ok": True})
            return json.dumps({"ok": False, "error": str(data)})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_server_info(self) -> str:
        """Get BlueBubbles server information.

        Returns:
            JSON string with server info.
        """
        if sys.platform != "darwin":  # pragma: no branch
            return _PLATFORM_ERROR
        try:
            resp = requests.get(self._url("/api/v1/server/info"), params=self._params(), timeout=10)
            return json.dumps({"ok": True, "info": resp.json().get("data", {})}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def mark_chat_read(self, chat_guid: str) -> str:
        """Mark a chat as read.

        Args:
            chat_guid: Chat GUID to mark as read.

        Returns:
            JSON string with ok status.
        """
        if sys.platform != "darwin":  # pragma: no branch
            return _PLATFORM_ERROR
        try:
            resp = requests.post(
                self._url(f"/api/v1/chat/{chat_guid}/read"),
                params=self._params(), timeout=10
            )
            return json.dumps({"ok": resp.json().get("status") == 200})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})



class BlueBubblesAgent(BaseChannelAgent, StatefulSorcarAgent):
    """StatefulSorcarAgent extended with BlueBubbles REST API tools (macOS only)."""

    def __init__(self) -> None:
        super().__init__("BlueBubbles Agent")
        self._backend = BlueBubblesChannelBackend()
        cfg = _load_config()
        if cfg:  # pragma: no branch
            self._backend._server_url = cfg["server_url"].rstrip("/")
            self._backend._password = cfg["password"]

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._server_url)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self


        def check_bluebubbles_auth() -> str:
            """Check if BlueBubbles is configured and reachable.

            Returns:
                Connection status or instructions.
            """
            if not agent._backend._server_url:  # pragma: no branch
                return (
                    "Not configured for BlueBubbles. Use authenticate_bluebubbles() to configure.\n"
                    "Requires BlueBubbles server running on a Mac."
                )
            return json.loads(agent._backend.get_server_info()).get("ok") and json.dumps({
                "ok": True,
                "server_url": agent._backend._server_url,
            }) or agent._backend.get_server_info()

        def authenticate_bluebubbles(server_url: str, password: str) -> str:
            """Configure BlueBubbles connection.

            Args:
                server_url: BlueBubbles server URL (e.g. "http://localhost:1234").
                password: BlueBubbles server password.

            Returns:
                Connection result or error message.
            """
            if sys.platform != "darwin":  # pragma: no branch
                return _PLATFORM_ERROR
            for val, name in [(server_url, "server_url"), (password, "password")]:
                if not val.strip():  # pragma: no branch
                    return f"{name} cannot be empty."
            agent._backend._server_url = server_url.strip().rstrip("/")
            agent._backend._password = password.strip()
            result = json.loads(agent._backend.get_server_info())
            if result.get("ok"):  # pragma: no branch
                _save_config(server_url, password)
                return json.dumps({"ok": True, "message": "BlueBubbles configured."})
            return json.dumps({"ok": False, "error": "Could not connect to BlueBubbles server."})

        def clear_bluebubbles_auth() -> str:
            """Clear the stored BlueBubbles configuration.

            Returns:
                Status message.
            """
            _clear_config()
            agent._backend._server_url = ""
            agent._backend._password = ""
            return "BlueBubbles configuration cleared."

        return [check_bluebubbles_auth, authenticate_bluebubbles, clear_bluebubbles_auth]


def _make_daemon_backend() -> BlueBubblesChannelBackend:
    """Create a configured BlueBubblesChannelBackend for daemon mode."""
    backend = BlueBubblesChannelBackend()
    cfg = _load_config()
    if not cfg:  # pragma: no branch
        print("Not configured. Run: kiss-bluebubbles -t 'authenticate'")
        sys.exit(1)
    backend._server_url = cfg["server_url"].rstrip("/")
    backend._password = cfg["password"]
    return backend


def main() -> None:
    """Run the BlueBubblesAgent from the command line with chat persistence."""
    channel_main(
        BlueBubblesAgent, "kiss-bluebubbles",
        channel_name="BlueBubbles",
        make_daemon_backend=_make_daemon_backend,
    )

if __name__ == "__main__":
    main()
