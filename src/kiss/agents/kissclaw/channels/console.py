"""Console/in-memory channel for KISSClaw testing and simple usage."""

from __future__ import annotations

from collections.abc import Callable

from kiss.agents.kissclaw.channels.base import Channel
from kiss.agents.kissclaw.types import Message


class ConsoleChannel(Channel):
    """In-memory channel for testing. Messages can be injected programmatically."""

    def __init__(self, jid_prefix: str = "console") -> None:
        self._jid_prefix = jid_prefix
        self._connected = False
        self._sent_messages: list[tuple[str, str]] = []
        self._on_message: Callable[[str, Message], None] | None = None
        self._on_chat_metadata: Callable[[str, str], None] | None = None

    @property
    def name(self) -> str:
        return "console"

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def send_message(self, jid: str, text: str) -> None:
        self._sent_messages.append((jid, text))

    def is_connected(self) -> bool:
        return self._connected

    def owns_jid(self, jid: str) -> bool:
        return jid.startswith(self._jid_prefix)

    def get_sent_messages(self) -> list[tuple[str, str]]:
        return list(self._sent_messages)

    def clear_sent(self) -> None:
        self._sent_messages.clear()

    def inject_message(self, msg: Message) -> None:
        """Inject a message as if it was received from the channel."""
        if self._on_message:
            self._on_message(msg.chat_jid, msg)
        if self._on_chat_metadata:
            self._on_chat_metadata(msg.chat_jid, msg.timestamp)
