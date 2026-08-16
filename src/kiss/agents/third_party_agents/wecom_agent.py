# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""WeCom Agent — channel agent for WeCom (WeChat Work) group robots.

Sends messages through a WeCom group-robot incoming webhook.  This
adapter is **outbound-only**: WeCom inbound callbacks require the
enterprise AES message envelope (encrypted XML with an app-level
EncodingAESKey), which is out of scope here, so ``poll_messages``
always returns no messages and poll mode is disabled.
Stores config in ``~/.kiss/third_party_agents/wecom/config.json``.

Usage::

    agent = WeComAgent()
    agent.run(prompt_template="Send a message to the team")
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import requests

from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

logger = logging.getLogger(__name__)

_WECOM_DIR = Path.home() / ".kiss" / "third_party_agents" / "wecom"
_config = ChannelConfig(_WECOM_DIR, ("webhook_url",))


class WeComChannelBackend(ToolMethodBackend):
    """Channel backend for WeCom group robots.

    Sends messages via the group-robot incoming webhook URL.  Inbound
    messages are not supported (see the module docstring).
    """

    def __init__(self) -> None:
        self._webhook_url: str = ""
        self._send_lock = threading.Lock()
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Load WeCom config."""
        cfg = _config.load()
        if not cfg:
            self._connection_info = "No WeCom config found."
            return False
        self._webhook_url = cfg["webhook_url"]
        self._connection_info = "WeCom configured"
        return True

    def _post_payload(self, payload: dict[str, Any]) -> None:
        """POST a message payload to the group-robot webhook.

        Raises:
            RuntimeError: If the HTTP request fails or WeCom returns a
                non-zero ``errcode``.
        """
        with self._send_lock:
            resp = requests.post(self._webhook_url, json=payload, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"WeCom send failed: HTTP {resp.status_code}")
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if isinstance(data, dict) and data.get("errcode", 0) != 0:
            raise RuntimeError(f"WeCom send failed: {data}")

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Return no messages: WeCom inbound callbacks are unsupported.

        Receiving requires the enterprise AES message envelope, which
        this outbound-only adapter does not implement.
        """
        return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a WeCom text message via the group-robot webhook.

        Group-robot webhooks are bound to a fixed group chat at
        creation, so ``channel_id`` and ``thread_ts`` are ignored.

        Raises:
            RuntimeError: If the webhook request fails or WeCom returns
                a non-zero ``errcode``.
        """
        self._post_payload({"msgtype": "text", "text": {"content": text}})

    def post_message(self, text: str, mentioned_list: str = "") -> str:
        """Send a text message to the WeCom group via the robot webhook.

        Args:
            text: Message text.
            mentioned_list: Comma-separated user IDs to @-mention
                (optional; use ``@all`` to mention everyone).

        Returns:
            JSON string with ok status.
        """
        try:
            payload = {
                "msgtype": "text",
                "text": {
                    "content": text,
                    "mentioned_list": [m.strip() for m in mentioned_list.split(",") if m.strip()],
                },
            }
            self._post_payload(payload)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def post_markdown(self, markdown: str) -> str:
        """Send a markdown message to the WeCom group via the robot webhook.

        Args:
            markdown: WeCom-flavored markdown message body.

        Returns:
            JSON string with ok status.
        """
        try:
            payload = {"msgtype": "markdown", "markdown": {"content": markdown}}
            self._post_payload(payload)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class WeComAgent(BaseChannelAgent):
    """Channel agent with WeCom group-robot tools."""

    channel_system_prompt = (
        "You are posting via a WeCom (WeChat Work) group robot. The "
        "channel is outbound-only: use post_message for text (with "
        "optional @-mentions by user ID) and post_markdown for "
        "WeCom-flavored markdown."
    )

    def __init__(self) -> None:
        super().__init__("WeCom Agent")
        self._backend = WeComChannelBackend()
        cfg = _config.load()
        if cfg:
            self._backend._webhook_url = cfg["webhook_url"]

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._webhook_url)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_wecom_auth() -> str:
            """Check if WeCom is configured.

            Returns:
                Configuration status or instructions.
            """
            if not agent._backend._webhook_url:
                return (
                    "Not configured for WeCom. Use authenticate_wecom() to configure.\n"
                    "You need the group robot webhook URL from WeCom group settings > "
                    "Group Robot > Add Robot."
                )
            return json.dumps(
                {
                    "ok": True,
                    "webhook_url": agent._backend._webhook_url[:50] + "...",
                }
            )

        def authenticate_wecom(webhook_url: str) -> str:
            """Configure the WeCom group robot.

            Args:
                webhook_url: WeCom group-robot incoming webhook URL.

            Returns:
                Configuration result or error message.
            """
            if not webhook_url.strip():
                return "webhook_url cannot be empty."
            agent._backend._webhook_url = webhook_url.strip()
            _config.save({"webhook_url": webhook_url.strip()})
            return json.dumps({"ok": True, "message": "WeCom configured."})

        def clear_wecom_auth() -> str:
            """Clear the stored WeCom configuration.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._webhook_url = ""
            return "WeCom configuration cleared."

        return [check_wecom_auth, authenticate_wecom, clear_wecom_auth]


def main() -> None:
    """Run the WeComAgent from the command line with chat persistence.

    Poll mode is disabled: WeCom inbound callbacks require the
    enterprise AES envelope, which this outbound-only adapter omits.
    """
    channel_main(WeComAgent, "kiss-wecom", channel_name="WeCom")


def get_tools() -> list:
    """Return the WeCom channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return WeComAgent()._get_tools()


if __name__ == "__main__":
    main()
