"""Tlon/Urbit Agent — StatefulSorcarAgent extension with Tlon/Urbit Eyre HTTP tools.

Provides access to Urbit/Tlon via the Eyre HTTP server. Stores config
in ``~/.kiss/channels/tlon/config.json``.

Usage::

    agent = TlonAgent()
    agent.run(prompt_template="List my Urbit groups")
"""

from __future__ import annotations

import json
import queue
import threading
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

_TLON_DIR = Path.home() / ".kiss" / "channels" / "tlon"
_config = ChannelConfig(_TLON_DIR, ("ship_url", "code"))


class TlonChannelBackend(ToolMethodBackend):
    """Channel backend for Tlon/Urbit Eyre HTTP."""

    def __init__(self) -> None:
        self._ship_url: str = ""
        self._session: requests.Session = requests.Session()
        self._event_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._sse_thread: threading.Thread | None = None
        self._channel_uid: str = ""
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Authenticate with Urbit ship."""
        cfg = _config.load()
        if not cfg:  # pragma: no branch
            self._connection_info = "No Tlon config found."
            return False
        self._ship_url = cfg["ship_url"]
        try:
            resp = self._session.post(
                f"{self._ship_url}/~/login",
                data={"password": cfg["code"]},
                timeout=10,
            )
            if resp.status_code in (200, 204):  # pragma: no branch
                self._connection_info = f"Connected to {self._ship_url}"
                return True
            self._connection_info = f"Tlon login failed: {resp.status_code}"
            return False
        except Exception as e:
            self._connection_info = f"Tlon connection failed: {e}"
            return False

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll event queue for messages."""
        messages: list[dict[str, Any]] = []
        while not self._event_queue.empty() and len(messages) < limit:  # pragma: no branch
            messages.append(self._event_queue.get_nowait())
        return messages, oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a Tlon/Urbit poke."""
        parts = channel_id.split("/", 2)
        if len(parts) >= 3:  # pragma: no branch
            group_path, channel_name = "/".join(parts[:2]), parts[2]
            self.post_message(group_path, channel_name, text)

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

    def list_groups(self) -> str:
        """List Urbit groups.

        Returns:
            JSON string with group list.
        """
        try:
            result = self.scry("groups", "/groups/light")
            return result
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_channels(self, group_path: str) -> str:
        """List channels in an Urbit group.

        Args:
            group_path: Group path (e.g. "~sampel/my-group").

        Returns:
            JSON string with channel list.
        """
        try:
            result = self.scry("channels", f"/channels/{group_path}/light")
            return result
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_messages(self, group_path: str, channel_name: str, count: int = 20) -> str:
        """Get recent messages from a Tlon channel.

        Args:
            group_path: Group path.
            channel_name: Channel name within the group.
            count: Number of messages to retrieve. Default: 20.

        Returns:
            JSON string with messages.
        """
        try:
            path = f"/channel/{group_path}/{channel_name}/posts/newest/{count}/15"
            result = self.scry("channels", path)
            return result
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def post_message(self, group_path: str, channel_name: str, content: str) -> str:
        """Post a message to a Tlon channel.

        Args:
            group_path: Group path (e.g. "~sampel/my-group").
            channel_name: Channel name within the group.
            content: Message content text.

        Returns:
            JSON string with ok status.
        """
        try:
            result = self.poke(
                "channels",
                "channel-action",
                json.dumps(
                    {
                        "channel-action": {
                            "post": {
                                "group": group_path,
                                "channel": channel_name,
                                "action": {
                                    "add": {
                                        "memo": {"content": [{"inline": [content]}], "author": "~"}
                                    }
                                },
                            }
                        }
                    }
                ),
            )
            return result
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_profile(self) -> str:
        """Get the current ship's profile.

        Returns:
            JSON string with profile info.
        """
        try:
            return self.scry("contacts", "/profile")
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def poke(self, app: str, mark: str, json_body: str) -> str:
        """Send a poke to an Urbit app.

        Args:
            app: Gall agent name (e.g. "groups").
            mark: Mark name (e.g. "groups-action").
            json_body: JSON string of the poke body.

        Returns:
            JSON string with ok status.
        """
        try:
            resp = self._session.post(
                f"{self._ship_url}/~/channel",
                json={
                    "id": int(time.time() * 1000),
                    "action": "poke",
                    "ship": "",
                    "app": app,
                    "mark": mark,
                    "json": json.loads(json_body),
                },
                timeout=30,
            )
            return json.dumps({"ok": resp.status_code in (200, 204)})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def scry(self, app: str, path: str) -> str:
        """Perform a scry request on an Urbit app.

        Args:
            app: Gall agent name.
            path: Scry path (starting with /).

        Returns:
            JSON string with scry result.
        """
        try:
            resp = self._session.get(
                f"{self._ship_url}/~/scry/{app}{path}.json",
                timeout=30,
            )
            if resp.status_code == 200:  # pragma: no branch
                return json.dumps({"ok": True, "data": resp.json()}, indent=2)[:8000]
            return json.dumps({"ok": False, "error": f"HTTP {resp.status_code}"})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class TlonAgent(BaseChannelAgent, StatefulSorcarAgent):
    """StatefulSorcarAgent extended with Tlon/Urbit Eyre HTTP tools."""

    def __init__(self) -> None:
        super().__init__("Tlon Agent")
        self._backend = TlonChannelBackend()
        cfg = _config.load()
        if cfg:  # pragma: no branch
            self._backend._ship_url = cfg["ship_url"]

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._ship_url)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_tlon_auth() -> str:
            """Check if Tlon/Urbit is configured.

            Returns:
                Configuration status or instructions.
            """
            if not agent._backend._ship_url:  # pragma: no branch
                return (
                    "Not configured for Tlon. Use authenticate_tlon(ship_url=..., code=...) "
                    "to configure. You need your ship URL and +code from the Urbit terminal."
                )
            return json.dumps({"ok": True, "ship_url": agent._backend._ship_url})

        def authenticate_tlon(ship_url: str, code: str) -> str:
            """Configure Tlon/Urbit connection.

            Args:
                ship_url: URL of your Urbit ship (e.g. "http://localhost:8080").
                code: Urbit access code from running +code in the terminal.

            Returns:
                Authentication result or error message.
            """
            for val, name in [(ship_url, "ship_url"), (code, "code")]:  # pragma: no branch
                if not val.strip():  # pragma: no branch
                    return f"{name} cannot be empty."
            agent._backend._ship_url = ship_url.strip().rstrip("/")
            try:
                resp = agent._backend._session.post(
                    f"{agent._backend._ship_url}/~/login",
                    data={"password": code.strip()},
                    timeout=10,
                )
                if resp.status_code in (200, 204):  # pragma: no branch
                    _config.save({"ship_url": ship_url.strip(), "code": code.strip()})
                    return json.dumps({"ok": True, "message": "Tlon configured."})
                return json.dumps({"ok": False, "error": f"Login failed: {resp.status_code}"})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def clear_tlon_auth() -> str:
            """Clear the stored Tlon configuration.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._ship_url = ""
            return "Tlon configuration cleared."

        return [check_tlon_auth, authenticate_tlon, clear_tlon_auth]


def main() -> None:
    """Run the TlonAgent from the command line with chat persistence."""
    channel_main(TlonAgent, "kiss-tlon")


if __name__ == "__main__":
    main()
