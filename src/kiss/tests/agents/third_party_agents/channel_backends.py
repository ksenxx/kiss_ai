"""In-memory channel backends shared by the runner and launcher tests.

Real classes implementing the ``ToolMethodBackend`` protocol with
deterministic in-memory data (not mocks of KISS code): every connect,
poll, send and typing call is recorded for the test to inspect.
"""

from __future__ import annotations

from typing import Any

from kiss.agents.third_party_agents._channel_agent_utils import ToolMethodBackend


class RecordingBackend(ToolMethodBackend):
    """In-memory channel backend that records every interaction."""

    def __init__(
        self,
        messages: list[dict[str, Any]] | None = None,
        connect_ok: bool = True,
        send_fail_times: int = 0,
        cursor: str = "",
    ) -> None:
        self.messages = messages or []
        self.connect_ok = connect_ok
        self.send_fail_times = send_fail_times
        self.cursor = cursor
        self.sent: list[tuple[str, str, str]] = []
        self.typing: list[tuple[str, str]] = []
        self.connect_calls = 0
        self.poll_calls = 0
        self.polled_oldest: list[str] = []
        self._connection_info = "recording backend"

    def connect(self) -> bool:
        """Record the connect attempt and return the configured result."""
        self.connect_calls += 1
        return self.connect_ok

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 50
    ) -> tuple[list[dict[str, Any]], str]:
        """Return the configured message list and cursor, recording *oldest*."""
        self.poll_calls += 1
        self.polled_oldest.append(oldest)
        return list(self.messages), self.cursor

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Record the send, raising for the first *send_fail_times* calls."""
        if self.send_fail_times > 0:
            self.send_fail_times -= 1
            raise ConnectionError("simulated send failure")
        self.sent.append((channel_id, text, thread_ts))

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """A message is from the bot when its ``bot`` flag is set."""
        return bool(msg.get("bot"))

    def send_typing(self, channel_id: str, thread_ts: str = "") -> None:
        """Record the typing indicator call."""
        self.typing.append((channel_id, thread_ts))


class ThreadBackend(RecordingBackend):
    """Recording backend that also supports thread polling."""

    def __init__(
        self,
        thread_replies: dict[str, list[dict[str, Any]]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.thread_replies = thread_replies or {}
        self.thread_polls: list[str] = []

    def poll_thread_messages(
        self, channel_id: str, thread_ts: str, oldest: str, limit: int = 100
    ) -> tuple[list[dict[str, Any]], str]:
        """Return the configured replies for *thread_ts*."""
        self.thread_polls.append(thread_ts)
        return list(self.thread_replies.get(thread_ts, [])), ""
