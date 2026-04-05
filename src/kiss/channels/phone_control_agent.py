"""Phone Control Agent — StatefulSorcarAgent extension with Android phone control tools.

Provides access to Android SMS, calls, and notifications via a companion
REST app. Stores config in ``~/.kiss/channels/phone/config.json``.

Usage::

    agent = PhoneControlAgent()
    agent.run(prompt_template="List recent SMS conversations")
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

_PHONE_DIR = Path.home() / ".kiss" / "channels" / "phone"


def _config_path() -> Path:
    """Return the path to the stored phone config file."""
    return _PHONE_DIR / "config.json"


def _load_config() -> dict[str, str] | None:
    """Load stored Phone Control config from disk."""
    return load_json_config(_config_path(), ("device_ip",))


def _save_config(device_ip: str, device_port: str, api_key: str) -> None:
    """Save Phone Control config to disk with restricted permissions."""
    save_json_config(
        _config_path(),
        {
            "device_ip": device_ip.strip(),
            "device_port": device_port.strip(),
            "api_key": api_key.strip(),
        },
    )


def _clear_config() -> None:
    """Delete the stored Phone Control config."""
    clear_json_config(_config_path())


class PhoneControlChannelBackend(ToolMethodBackend):
    """ChannelBackend implementation for Android phone control via REST API."""

    def __init__(self) -> None:
        self._device_url: str = ""
        self._api_key: str = ""
        self._last_msg_id: str = ""
        self._connection_info: str = ""

    def _url(self, path: str) -> str:
        return f"{self._device_url}{path}"

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._api_key:  # pragma: no branch
            headers["X-API-Key"] = self._api_key
        return headers

    def connect(self) -> bool:
        """Connect to phone companion app."""
        cfg = _load_config()
        if not cfg:  # pragma: no branch
            self._connection_info = "No phone config found."
            return False
        self._device_url = f"http://{cfg['device_ip']}:{cfg['device_port']}"
        self._api_key = cfg.get("api_key", "")
        try:
            resp = requests.get(self._url("/api/device/info"), headers=self._headers(), timeout=5)
            if resp.status_code == 200:  # pragma: no branch
                data = resp.json()
                self._connection_info = f"Connected to {data.get('device_name', 'phone')}"
                return True
            self._connection_info = f"Phone connection failed: {resp.status_code}"
            return False
        except Exception as e:
            self._connection_info = f"Phone connection failed: {e}"
            return False

    @property
    def connection_info(self) -> str:
        """Human-readable connection status string."""
        return self._connection_info

    def find_channel(self, name: str) -> str | None:
        """Return phone number as channel ID."""
        return name if name else None

    def find_user(self, username: str) -> str | None:
        """Return username as user ID."""
        return username if username else None

    def join_channel(self, channel_id: str) -> None:
        """No-op for phone control."""

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll for new SMS messages."""
        try:
            params: dict[str, Any] = {"since": oldest or self._last_msg_id, "limit": limit}
            resp = requests.get(
                self._url("/api/sms/messages"), headers=self._headers(), params=params, timeout=10
            )
            data = resp.json()
            messages: list[dict[str, Any]] = []
            new_oldest = oldest
            for msg in data.get("messages", []):  # pragma: no branch
                ts = str(msg.get("timestamp", ""))
                new_oldest = ts
                messages.append({
                    "ts": ts,
                    "user": msg.get("from", ""),
                    "text": msg.get("body", ""),
                    "id": str(msg.get("id", "")),
                })
            return messages, new_oldest
        except Exception:
            return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send an SMS."""
        requests.post(
            self._url("/api/sms/send"),
            headers=self._headers(),
            json={"to": channel_id, "body": text},
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
        """Poll for a reply SMS from a specific number."""
        oldest = str(int(time.time() * 1000))

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
            poll_interval=3.0,
        )

    def disconnect(self) -> None:
        """Release backend resources before stop or reconnect."""

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Check if message is from the phone itself (sent)."""
        return False

    def strip_bot_mention(self, text: str) -> str:
        """Remove bot mentions from text."""
        return text

    def send_sms(self, to: str, text: str) -> str:
        """Send an SMS message.

        Args:
            to: Recipient phone number.
            text: Message text.

        Returns:
            JSON string with ok status.
        """
        try:
            resp = requests.post(
                self._url("/api/sms/send"),
                headers=self._headers(),
                json={"to": to, "body": text},
                timeout=30,
            )
            return json.dumps({"ok": resp.status_code == 200})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def make_call(self, to: str) -> str:
        """Make a phone call.

        Args:
            to: Phone number to call.

        Returns:
            JSON string with ok status.
        """
        try:
            resp = requests.post(
                self._url("/api/call/make"),
                headers=self._headers(),
                json={"to": to},
                timeout=30,
            )
            return json.dumps({"ok": resp.status_code == 200})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def end_call(self) -> str:
        """End the current active call.

        Returns:
            JSON string with ok status.
        """
        try:
            resp = requests.post(self._url("/api/call/end"), headers=self._headers(), timeout=10)
            return json.dumps({"ok": resp.status_code == 200})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_sms_conversations(self, limit: int = 20) -> str:
        """List recent SMS conversations.

        Args:
            limit: Maximum conversations to return. Default: 20.

        Returns:
            JSON string with conversation list.
        """
        try:
            resp = requests.get(
                self._url("/api/sms/conversations"),
                headers=self._headers(),
                params={"limit": limit},
                timeout=10,
            )
            data = resp.json()
            convos = data.get("conversations", [])
            return json.dumps({"ok": True, "conversations": convos}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_sms_messages(self, thread_id: str, limit: int = 50) -> str:
        """Get messages in an SMS thread.

        Args:
            thread_id: Thread ID from list_sms_conversations.
            limit: Maximum messages to return. Default: 50.

        Returns:
            JSON string with message list.
        """
        try:
            resp = requests.get(
                self._url(f"/api/sms/thread/{thread_id}"),
                headers=self._headers(),
                params={"limit": limit},
                timeout=10,
            )
            data = resp.json()
            return json.dumps({"ok": True, "messages": data.get("messages", [])}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_call_log(self, limit: int = 20) -> str:
        """Get recent call log.

        Args:
            limit: Maximum calls to return. Default: 20.

        Returns:
            JSON string with call list.
        """
        try:
            resp = requests.get(
                self._url("/api/call/log"),
                headers=self._headers(),
                params={"limit": limit},
                timeout=10,
            )
            data = resp.json()
            return json.dumps({"ok": True, "calls": data.get("calls", [])}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_device_info(self) -> str:
        """Get phone device information.

        Returns:
            JSON string with device info (model, battery, etc).
        """
        try:
            resp = requests.get(self._url("/api/device/info"), headers=self._headers(), timeout=10)
            return json.dumps({"ok": True, "device": resp.json()}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_notifications(self) -> str:
        """List current phone notifications.

        Returns:
            JSON string with notification list.
        """
        try:
            resp = requests.get(
                self._url("/api/notifications"), headers=self._headers(), timeout=10
            )
            data = resp.json()
            notifs = data.get("notifications", [])
            return json.dumps({"ok": True, "notifications": notifs}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def dismiss_notification(self, notification_id: str) -> str:
        """Dismiss a phone notification.

        Args:
            notification_id: Notification ID to dismiss.

        Returns:
            JSON string with ok status.
        """
        try:
            resp = requests.delete(
                self._url(f"/api/notifications/{notification_id}"),
                headers=self._headers(),
                timeout=10,
            )
            return json.dumps({"ok": resp.status_code == 200})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_notification_reply(self, notification_id: str, text: str) -> str:
        """Reply to a phone notification (e.g. WhatsApp, Signal).

        Args:
            notification_id: Notification ID to reply to.
            text: Reply text.

        Returns:
            JSON string with ok status.
        """
        try:
            resp = requests.post(
                self._url(f"/api/notifications/{notification_id}/reply"),
                headers=self._headers(),
                json={"text": text},
                timeout=30,
            )
            return json.dumps({"ok": resp.status_code == 200})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})



class PhoneControlAgent(BaseChannelAgent, StatefulSorcarAgent):
    """StatefulSorcarAgent extended with Android phone control tools."""

    def __init__(self) -> None:
        super().__init__("Phone Control Agent")
        self._backend = PhoneControlChannelBackend()
        cfg = _load_config()
        if cfg:  # pragma: no branch
            self._backend._device_url = f"http://{cfg['device_ip']}:{cfg.get('device_port', 8080)}"
            self._backend._api_key = cfg.get("api_key", "")

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._device_url)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self


        def check_phone_auth() -> str:
            """Check if phone control is configured and device is reachable.

            Returns:
                Connection status or instructions.
            """
            if not agent._backend._device_url:  # pragma: no branch
                return (
                    "Not configured for phone control. Use authenticate_phone() to configure.\n"
                    "Requires a companion app running on your Android device."
                )
            try:
                result = json.loads(agent._backend.get_device_info())
                if result.get("ok"):  # pragma: no branch
                    return json.dumps({"ok": True, "device": result.get("device", {})})
                return json.dumps({"ok": False, "error": "Device unreachable."})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_phone(
            device_ip: str, device_port: int = 8080, api_key: str = ""
        ) -> str:
            """Configure phone control connection.

            Args:
                device_ip: IP address of the Android device on your network.
                device_port: Port the companion app listens on. Default: 8080.
                api_key: Optional API key for authentication.

            Returns:
                Connection result or error message.
            """
            if not device_ip.strip():  # pragma: no branch
                return "device_ip cannot be empty."
            agent._backend._device_url = f"http://{device_ip.strip()}:{device_port}"
            agent._backend._api_key = api_key.strip()
            try:
                result = json.loads(agent._backend.get_device_info())
                if result.get("ok"):  # pragma: no branch
                    _save_config(device_ip, str(device_port), api_key)
                    return json.dumps({"ok": True, "message": "Phone control configured."})
                return json.dumps({"ok": False, "error": "Could not connect to device."})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def clear_phone_auth() -> str:
            """Clear the stored phone configuration.

            Returns:
                Status message.
            """
            _clear_config()
            agent._backend._device_url = ""
            agent._backend._api_key = ""
            return "Phone configuration cleared."

        return [check_phone_auth, authenticate_phone, clear_phone_auth]


def _make_daemon_backend() -> PhoneControlChannelBackend:
    """Create a configured PhoneControlChannelBackend for daemon mode."""
    backend = PhoneControlChannelBackend()
    cfg = _load_config()
    if not cfg:  # pragma: no branch
        print("Not configured. Run: kiss-phone -t 'authenticate'")
        sys.exit(1)
    backend._device_url = f"http://{cfg['device_ip']}:{cfg.get('device_port', 8080)}"
    backend._api_key = cfg.get("api_key", "")
    return backend


def main() -> None:
    """Run the PhoneControlAgent from the command line with chat persistence."""
    channel_main(
        PhoneControlAgent, "kiss-phone",
        channel_name="Phone Control",
        make_daemon_backend=_make_daemon_backend,
    )

if __name__ == "__main__":
    main()
