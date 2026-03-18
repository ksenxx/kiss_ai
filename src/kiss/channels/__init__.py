"""Channel integrations for KISS agents."""

from __future__ import annotations

from typing import Any, Protocol


class ChannelBackend(Protocol):
    """Protocol for messaging channel backends used by the background agent.

    Each backend (Slack, Gmail, WhatsApp, etc.) implements this interface
    to provide channel monitoring, message sending, and reply waiting.
    """

    def connect(self) -> bool:
        """Authenticate and connect to the channel service.

        Returns:
            True on success, False on failure (check connection_info for details).
        """
        ...

    @property
    def connection_info(self) -> str:
        """Human-readable connection status string."""
        ...

    def find_channel(self, name: str) -> str | None:
        """Find a channel/conversation by name.

        Args:
            name: Channel name to search for.

        Returns:
            Channel ID string, or None if not found.
        """
        ...

    def find_user(self, username: str) -> str | None:
        """Find a user by username or display name.

        Args:
            username: Username to search for.

        Returns:
            User ID string, or None if not found.
        """
        ...

    def join_channel(self, channel_id: str) -> None:
        """Join/subscribe to a channel so the bot can read and post messages.

        Args:
            channel_id: Channel ID to join.
        """
        ...

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll for new messages since a given timestamp.

        Args:
            channel_id: Channel ID to poll.
            oldest: Only return messages newer than this timestamp.
            limit: Maximum number of messages to return.

        Returns:
            Tuple of (messages, updated_oldest). Each message dict has at
            minimum: ``ts``, ``user``, ``text``.
        """
        ...

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a message to a channel, optionally in a thread.

        Args:
            channel_id: Channel ID to post to.
            text: Message text.
            thread_ts: If non-empty, reply in this thread.
        """
        ...

    def wait_for_reply(self, channel_id: str, thread_ts: str, user_id: str) -> str:
        """Block until a specific user replies in a thread.

        Args:
            channel_id: Channel ID containing the thread.
            thread_ts: Timestamp of the parent message (thread root).
            user_id: User ID to wait for a reply from.

        Returns:
            The text of the user's reply.
        """
        ...

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Check if a message was sent by the bot itself.

        Args:
            msg: Message dict from poll_messages.

        Returns:
            True if the message is from the bot.
        """
        ...

    def strip_bot_mention(self, text: str) -> str:
        """Remove bot mention markers from message text.

        Args:
            text: Raw message text.

        Returns:
            Cleaned text with bot mentions removed.
        """
        ...
