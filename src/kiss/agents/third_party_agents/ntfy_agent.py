# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""ntfy Agent — channel agent for the ntfy.sh pub-sub notification service.

Provides publish/subscribe access to an ntfy topic over plain HTTP:
messages are published by POSTing to ``{server}/{topic}`` and read back
by polling ``{server}/{topic}/json?poll=1``.  Loop prevention uses an
echo tag: every message the agent publishes is tagged (default
``kiss-sorcar``) and tagged messages are treated as bot messages.

Stores config in ``~/.kiss/third_party_agents/ntfy/config.json`` with a
required ``topic`` and optional ``server`` (default ``https://ntfy.sh``),
``token`` (sent as ``Authorization: Bearer``) and ``echo_tag``.

Usage::

    agent = NtfyAgent()
    agent.run(prompt_template="Notify me that the build finished")
"""

from __future__ import annotations

import json
import logging
import sys
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

_DEFAULT_SERVER = "https://ntfy.sh"
_DEFAULT_ECHO_TAG = "kiss-sorcar"

_NTFY_DIR = Path.home() / ".kiss" / "third_party_agents" / "ntfy"
_config = ChannelConfig(_NTFY_DIR, ("topic",))


class NtfyChannelBackend(ToolMethodBackend):
    """Channel backend for the ntfy HTTP pub-sub API.

    Publishes by POSTing plain text to ``{server}/{topic}`` and polls
    with ``GET {server}/{topic}/json?poll=1&since=...``.  Messages the
    backend publishes carry the echo tag so :meth:`is_from_bot` can
    filter them out of subsequent polls.
    """

    def __init__(self) -> None:
        self._server: str = _DEFAULT_SERVER
        self._topic: str = ""
        self._token: str = ""
        self._echo_tag: str = _DEFAULT_ECHO_TAG
        self._send_lock = threading.Lock()
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Load the ntfy config and mark the backend as connected."""
        cfg = _config.load()
        if not cfg:  # pragma: no branch
            self._connection_info = "No ntfy config found."
            return False
        self._apply_config(cfg)
        self._connection_info = f"ntfy configured: {self._server}/{self._topic}"
        return True

    def _apply_config(self, cfg: dict[str, str]) -> None:
        """Copy persisted config values onto the backend, applying defaults."""
        self._topic = cfg["topic"]
        self._server = (cfg.get("server") or _DEFAULT_SERVER).rstrip("/")
        self._token = cfg.get("token", "")
        self._echo_tag = cfg.get("echo_tag") or _DEFAULT_ECHO_TAG

    def _auth_headers(self) -> dict[str, str]:
        """Return HTTP headers with Bearer authorization when a token is set."""
        if self._token:
            return {"Authorization": f"Bearer {self._token}"}
        return {}

    def _echo_tags(self) -> list[str]:
        """Return the echo tag as a list of stripped comma-separated sub-tags."""
        return [t.strip() for t in self._echo_tag.split(",") if t.strip()]

    def _fetch_messages(
        self, topic: str, oldest: str, limit: int
    ) -> tuple[list[dict[str, Any]], str]:
        """Fetch and normalize messages from an ntfy topic, raising on failure.

        Fetches ``{server}/{topic}/json?poll=1&since={oldest or 'all'}``,
        parses the newline-delimited JSON stream and keeps only
        ``message`` events.  When a message carries a title, the title is
        prepended to the normalized ``text`` so downstream consumers that
        only read ``text`` still see it.

        Args:
            topic: Topic to poll.
            oldest: ``since`` cursor (unix time as string, or empty for all).
            limit: Maximum number of messages to return.

        Returns:
            Tuple of (normalized message dicts, newest seen time as string).

        Raises:
            RuntimeError: If the poll request returns a non-200 status.
            requests.RequestException: If the HTTP request itself fails.
        """
        resp = requests.get(
            f"{self._server}/{topic}/json",
            params={"poll": "1", "since": oldest or "all"},
            headers=self._auth_headers(),
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"ntfy poll failed: HTTP {resp.status_code}")
        messages: list[dict[str, Any]] = []
        newest = oldest
        for line in resp.text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("event") != "message":
                continue
            event_time = int(event.get("time", 0) or 0)
            event_topic = str(event.get("topic", "") or topic)
            title = str(event.get("title", "") or "")
            text = str(event.get("message", "") or "")
            if title:
                text = f"{title}\n\n{text}"
            messages.append(
                {
                    "ts": str(event_time),
                    "user": event_topic,
                    "text": text,
                    "title": title,
                    "channel_id": event_topic,
                    "tags": list(event.get("tags") or []),
                }
            )
            if not newest or event_time > int(newest):
                newest = str(event_time)
            if len(messages) >= limit:
                break
        return messages, newest

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll an ntfy topic for new messages, swallowing failures.

        Args:
            channel_id: Topic to poll when no topic is configured.
            oldest: ``since`` cursor (unix time as string, or empty for all).
            limit: Maximum number of messages to return.

        Returns:
            Tuple of (normalized message dicts, newest seen time as string);
            ``([], oldest)`` on any failure.
        """
        topic = self._topic or channel_id
        try:
            return self._fetch_messages(topic, oldest, limit)
        except Exception:
            logger.debug("ntfy poll failed for topic %s", topic, exc_info=True)
            return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Publish a plain-text message to the ntfy topic.

        ntfy has no threads, so ``thread_ts`` is ignored.  The message
        is tagged with the echo tag for loop prevention.

        Args:
            channel_id: Topic to publish to when no topic is configured.
            text: Message body.
            thread_ts: Ignored (ntfy has no threading).

        Raises:
            RuntimeError: If the publish request returns a non-2xx status.
        """
        topic = self._topic or channel_id
        headers = self._auth_headers()
        headers["X-Tags"] = ",".join(self._echo_tags())
        with self._send_lock:
            resp = requests.post(
                f"{self._server}/{topic}", data=text.encode("utf-8"), headers=headers, timeout=30
            )
        if not 200 <= resp.status_code < 300:
            raise RuntimeError(f"ntfy publish failed: HTTP {resp.status_code}")

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Return True when a polled message carries the echo tag.

        Args:
            msg: Message dict from :meth:`poll_messages`.

        Returns:
            Whether the message was published by this agent.
        """
        tags = msg.get("tags") or []
        return any(sub_tag in tags for sub_tag in self._echo_tags())

    def publish_notification(
        self,
        message: str,
        title: str = "",
        priority: str = "",
        tags: str = "",
        click_url: str = "",
    ) -> str:
        """Publish a notification to the configured ntfy topic.

        Args:
            message: Notification body text.
            title: Optional notification title.
            priority: Optional priority (``min``, ``low``, ``default``,
                ``high`` or ``urgent``, or 1-5).
            tags: Optional comma-separated tags/emoji shortcodes.
            click_url: Optional URL opened when the notification is tapped.

        Returns:
            JSON string with ok status and the published message id.
        """
        try:
            all_tags = [t.strip() for t in tags.split(",") if t.strip()]
            all_tags.extend(self._echo_tags())
            headers = self._auth_headers()
            headers["X-Tags"] = ",".join(all_tags)
            if title:
                headers["X-Title"] = title
            if priority:
                headers["X-Priority"] = priority
            if click_url:
                headers["X-Click"] = click_url
            with self._send_lock:
                resp = requests.post(
                    f"{self._server}/{self._topic}",
                    data=message.encode("utf-8"),
                    headers=headers,
                    timeout=30,
                )
            if not 200 <= resp.status_code < 300:
                return json.dumps(
                    {"ok": False, "error": f"ntfy publish failed: HTTP {resp.status_code}"}
                )
            try:
                message_id = str(resp.json().get("id", ""))
            except ValueError:
                message_id = ""
            return json.dumps({"ok": True, "id": message_id})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def poll_topic(self, since: str = "all", limit: int = 10) -> str:
        """Read recent messages from the configured ntfy topic.

        Args:
            since: ``since`` cursor — ``all`` for the full cache, or a
                unix timestamp string returned by a previous poll.
            limit: Maximum number of messages to return.

        Returns:
            JSON object with ``ok`` status: on success ``messages`` holds
            message dicts with ``ts``, ``user``, ``text``, ``title``,
            ``channel_id`` and ``tags`` keys; on failure ``error`` holds
            the failure reason.
        """
        try:
            messages, _ = self._fetch_messages(self._topic, "" if since == "all" else since, limit)
            return json.dumps({"ok": True, "messages": messages})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class NtfyAgent(BaseChannelAgent):
    """Channel agent with ntfy pub-sub notification tools."""

    channel_system_prompt = (
        "You are chatting via ntfy (https://ntfy.sh), a topic-based HTTP "
        "pub-sub notification service. Messages are plain text published "
        "to a topic; there are no threads, users or rich formatting. Use "
        "publish_notification to send notifications (optionally with a "
        "title, priority, tags and click URL) and poll_topic to read "
        "recent messages from the topic."
    )

    def __init__(self) -> None:
        super().__init__("Ntfy Agent")
        self._backend = NtfyChannelBackend()
        cfg = _config.load()
        if cfg:  # pragma: no branch
            self._backend._apply_config(cfg)

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._topic)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_ntfy_auth() -> str:
            """Check if ntfy is configured.

            Returns:
                Configuration status or instructions.
            """
            if not agent._backend._topic:  # pragma: no branch
                return (
                    "Not configured for ntfy. Use authenticate_ntfy() to configure.\n"
                    "You need a topic name; the public https://ntfy.sh server "
                    "works without a token, self-hosted servers may need an "
                    "access token."
                )
            return json.dumps(
                {
                    "ok": True,
                    "server": agent._backend._server,
                    "topic": agent._backend._topic,
                    "echo_tag": agent._backend._echo_tag,
                }
            )

        def authenticate_ntfy(
            topic: str, server: str = "", token: str = "", echo_tag: str = ""
        ) -> str:
            """Configure the ntfy topic and server.

            Args:
                topic: ntfy topic name to publish to and poll.
                server: ntfy server base URL (default ``https://ntfy.sh``).
                token: Optional access token sent as ``Authorization: Bearer``.
                echo_tag: Tag marking the agent's own messages for loop
                    prevention (default ``kiss-sorcar``).

            Returns:
                Configuration result or error message.
            """
            if not topic.strip():  # pragma: no branch
                return "topic cannot be empty."
            cfg = {
                "topic": topic.strip(),
                "server": (server.strip() or _DEFAULT_SERVER).rstrip("/"),
                "token": token.strip(),
                "echo_tag": echo_tag.strip() or _DEFAULT_ECHO_TAG,
            }
            _config.save(cfg)
            agent._backend._apply_config(cfg)
            return json.dumps({"ok": True, "message": "ntfy configured."})

        def clear_ntfy_auth() -> str:
            """Clear the stored ntfy configuration.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._topic = ""
            agent._backend._server = _DEFAULT_SERVER
            agent._backend._token = ""
            agent._backend._echo_tag = _DEFAULT_ECHO_TAG
            return "ntfy configuration cleared."

        return [check_ntfy_auth, authenticate_ntfy, clear_ntfy_auth]


def _make_backend() -> NtfyChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = NtfyChannelBackend()
    cfg = _config.load()
    if not cfg:  # pragma: no branch
        print("Not configured. Run: kiss-ntfy -t 'authenticate'")
        sys.exit(1)
    backend._apply_config(cfg)
    return backend


def main() -> None:
    """Run the NtfyAgent from the command line with chat persistence."""
    channel_main(NtfyAgent, "kiss-ntfy", channel_name="ntfy", make_backend=_make_backend)


def get_tools() -> list:
    """Return the ntfy channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return NtfyAgent()._get_tools()


if __name__ == "__main__":
    main()
