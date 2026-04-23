"""Telegram Agent — ChatSorcarAgent extension with Telegram Bot API tools.

Provides authenticated access to Telegram via a bot token from @BotFather.
Stores the token securely in ``~/.kiss/channels/telegram/config.json`` and
exposes a focused set of Telegram Bot API tools.

Usage::

    agent = TelegramAgent()
    agent.run(prompt_template="Send 'Hello!' to chat_id 123456789")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.channels._backend_utils import wait_for_matching_message
from kiss.channels._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

_TELEGRAM_DIR = Path.home() / ".kiss" / "channels" / "telegram"
_config = ChannelConfig(_TELEGRAM_DIR, ("bot_token",))


class TelegramChannelBackend(ToolMethodBackend):
    """Channel backend for Telegram Bot API.

    Uses python-telegram-bot sync Bot for API calls and long-polling
    getUpdates for message polling.
    """

    def __init__(self) -> None:
        self._bot: Any = None
        self._last_update_id: int = -1
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Authenticate with Telegram using the stored bot token."""
        cfg = _config.load()
        if not cfg:  # pragma: no branch
            self._connection_info = "No Telegram token found."
            return False
        try:
            from telegram import Bot

            self._bot = Bot(token=cfg["bot_token"])
            me = self._bot.get_me()
            self._connection_info = f"Authenticated as @{me.username}"
            return True
        except Exception as e:
            self._connection_info = f"Telegram auth failed: {e}"
            return False

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll for new Telegram updates via getUpdates."""
        assert self._bot is not None
        try:
            offset = self._last_update_id + 1 if self._last_update_id >= 0 else None
            updates = self._bot.get_updates(offset=offset, limit=limit, timeout=0)
            messages: list[dict[str, Any]] = []
            for update in updates:  # pragma: no branch
                if update.update_id > self._last_update_id:  # pragma: no branch
                    self._last_update_id = update.update_id
                msg = update.message or update.channel_post
                if msg and msg.text:  # pragma: no branch
                    chat_id = str(msg.chat.id)
                    if not channel_id or chat_id == channel_id:  # pragma: no branch
                        messages.append(
                            {
                                "ts": str(msg.date.timestamp()) if msg.date else "",
                                "user": str(msg.from_user.id) if msg.from_user else "",
                                "text": msg.text,
                                "message_id": str(msg.message_id),
                                "chat_id": chat_id,
                            }
                        )
            return messages, oldest
        except Exception:
            return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a Telegram message."""
        assert self._bot is not None
        kwargs: dict[str, Any] = {"chat_id": int(channel_id), "text": text}
        if thread_ts:  # pragma: no branch
            kwargs["reply_to_message_id"] = int(thread_ts)
        self._bot.send_message(**kwargs)

    def wait_for_reply(
        self,
        channel_id: str,
        thread_ts: str,
        user_id: str,
        timeout_seconds: float = 300.0,
    ) -> str | None:
        """Poll for a reply from a specific user."""
        assert self._bot is not None
        return wait_for_matching_message(
            poll=lambda: self.poll_messages(channel_id, "")[0],
            matches=lambda msg: msg.get("user") == user_id,
            extract_text=lambda msg: str(msg.get("text", "")),
            timeout_seconds=timeout_seconds,
            poll_interval=2.0,
        )


    def send_text(self, chat_id: str, text: str, reply_to_message_id: str = "") -> str:
        """Send a text message to a Telegram chat.

        Args:
            chat_id: Chat ID (integer as string) or @username.
            text: Message text (supports Markdown).
            reply_to_message_id: Optional message ID to reply to.

        Returns:
            JSON string with ok status and message_id.
        """
        assert self._bot is not None
        try:
            cid: Any = int(chat_id) if chat_id.lstrip("-").isdigit() else chat_id
            kwargs: dict[str, Any] = {"chat_id": cid, "text": text}
            if reply_to_message_id:  # pragma: no branch
                kwargs["reply_to_message_id"] = int(reply_to_message_id)
            msg = self._bot.send_message(**kwargs)
            return json.dumps({"ok": True, "message_id": msg.message_id})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_photo(self, chat_id: str, photo_url_or_path: str, caption: str = "") -> str:
        """Send a photo to a Telegram chat.

        Args:
            chat_id: Chat ID or @username.
            photo_url_or_path: URL or local file path of the photo.
            caption: Optional caption text.

        Returns:
            JSON string with ok status and message_id.
        """
        assert self._bot is not None
        try:
            kwargs: dict[str, Any] = {
                "chat_id": int(chat_id) if chat_id.lstrip("-").isdigit() else chat_id,
            }
            if photo_url_or_path.startswith("http"):  # pragma: no branch
                kwargs["photo"] = photo_url_or_path
                if caption:  # pragma: no branch
                    kwargs["caption"] = caption
                msg = self._bot.send_photo(**kwargs)
            else:
                with open(photo_url_or_path, "rb") as f:
                    kwargs["photo"] = f
                    if caption:  # pragma: no branch
                        kwargs["caption"] = caption
                    msg = self._bot.send_photo(**kwargs)
            return json.dumps({"ok": True, "message_id": msg.message_id})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_document(self, chat_id: str, document_path: str, caption: str = "") -> str:
        """Send a document/file to a Telegram chat.

        Args:
            chat_id: Chat ID or @username.
            document_path: Local file path to send.
            caption: Optional caption text.

        Returns:
            JSON string with ok status and message_id.
        """
        assert self._bot is not None
        try:
            cid: Any = int(chat_id) if chat_id.lstrip("-").isdigit() else chat_id
            with open(document_path, "rb") as f:
                msg = self._bot.send_document(chat_id=cid, document=f, caption=caption)
            return json.dumps({"ok": True, "message_id": msg.message_id})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def edit_message_text(self, chat_id: str, message_id: str, text: str) -> str:
        """Edit an existing message text.

        Args:
            chat_id: Chat ID where the message is.
            message_id: ID of the message to edit.
            text: New message text.

        Returns:
            JSON string with ok status.
        """
        assert self._bot is not None
        try:
            cid: Any = int(chat_id) if chat_id.lstrip("-").isdigit() else chat_id
            self._bot.edit_message_text(chat_id=cid, message_id=int(message_id), text=text)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def delete_message(self, chat_id: str, message_id: str) -> str:
        """Delete a message.

        Args:
            chat_id: Chat ID where the message is.
            message_id: ID of the message to delete.

        Returns:
            JSON string with ok status.
        """
        assert self._bot is not None
        try:
            cid: Any = int(chat_id) if chat_id.lstrip("-").isdigit() else chat_id
            self._bot.delete_message(chat_id=cid, message_id=int(message_id))
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def pin_message(self, chat_id: str, message_id: str) -> str:
        """Pin a message in a chat.

        Args:
            chat_id: Chat ID.
            message_id: ID of the message to pin.

        Returns:
            JSON string with ok status.
        """
        assert self._bot is not None
        try:
            cid: Any = int(chat_id) if chat_id.lstrip("-").isdigit() else chat_id
            self._bot.pin_chat_message(chat_id=cid, message_id=int(message_id))
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def unpin_message(self, chat_id: str, message_id: str = "") -> str:
        """Unpin a message (or all messages) in a chat.

        Args:
            chat_id: Chat ID.
            message_id: ID of specific message to unpin. If empty, unpins all.

        Returns:
            JSON string with ok status.
        """
        assert self._bot is not None
        try:
            cid: Any = int(chat_id) if chat_id.lstrip("-").isdigit() else chat_id
            if message_id:  # pragma: no branch
                self._bot.unpin_chat_message(chat_id=cid, message_id=int(message_id))
            else:
                self._bot.unpin_all_chat_messages(chat_id=cid)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_chat(self, chat_id: str) -> str:
        """Get information about a chat.

        Args:
            chat_id: Chat ID or @username.

        Returns:
            JSON string with chat info (id, title, type, members_count).
        """
        assert self._bot is not None
        try:
            cid: Any = int(chat_id) if chat_id.lstrip("-").isdigit() else chat_id
            chat = self._bot.get_chat(chat_id=cid)
            return json.dumps(
                {
                    "ok": True,
                    "id": chat.id,
                    "title": chat.title or "",
                    "type": chat.type,
                    "username": chat.username or "",
                    "description": chat.description or "",
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_chat_members_count(self, chat_id: str) -> str:
        """Get the number of members in a chat.

        Args:
            chat_id: Chat ID or @username.

        Returns:
            JSON string with member count.
        """
        assert self._bot is not None
        try:
            cid: Any = int(chat_id) if chat_id.lstrip("-").isdigit() else chat_id
            count = self._bot.get_chat_member_count(chat_id=cid)
            return json.dumps({"ok": True, "count": count})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_chat_member(self, chat_id: str, user_id: str) -> str:
        """Get information about a chat member.

        Args:
            chat_id: Chat ID.
            user_id: User ID.

        Returns:
            JSON string with member info (user, status).
        """
        assert self._bot is not None
        try:
            cid: Any = int(chat_id) if chat_id.lstrip("-").isdigit() else chat_id
            member = self._bot.get_chat_member(chat_id=cid, user_id=int(user_id))
            user = member.user
            return json.dumps(
                {
                    "ok": True,
                    "user_id": user.id,
                    "username": user.username or "",
                    "first_name": user.first_name or "",
                    "status": member.status,
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def ban_chat_member(self, chat_id: str, user_id: str) -> str:
        """Ban a user from a chat.

        Args:
            chat_id: Chat ID.
            user_id: User ID to ban.

        Returns:
            JSON string with ok status.
        """
        assert self._bot is not None
        try:
            cid: Any = int(chat_id) if chat_id.lstrip("-").isdigit() else chat_id
            self._bot.ban_chat_member(chat_id=cid, user_id=int(user_id))
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def unban_chat_member(self, chat_id: str, user_id: str) -> str:
        """Unban a user from a chat.

        Args:
            chat_id: Chat ID.
            user_id: User ID to unban.

        Returns:
            JSON string with ok status.
        """
        assert self._bot is not None
        try:
            cid: Any = int(chat_id) if chat_id.lstrip("-").isdigit() else chat_id
            self._bot.unban_chat_member(chat_id=cid, user_id=int(user_id))
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_updates(self, offset: str = "", limit: int = 10) -> str:
        """Get recent updates (messages) from the bot.

        Args:
            offset: Update ID offset for pagination.
            limit: Maximum number of updates to return (1-100).

        Returns:
            JSON string with list of update objects.
        """
        assert self._bot is not None
        try:
            kwargs: dict[str, Any] = {"limit": min(limit, 100), "timeout": 0}
            if offset:  # pragma: no branch
                kwargs["offset"] = int(offset)
            updates = self._bot.get_updates(**kwargs)
            results = []
            for u in updates:  # pragma: no branch
                msg = u.message or u.channel_post
                results.append(
                    {
                        "update_id": u.update_id,
                        "chat_id": str(msg.chat.id) if msg else "",
                        "user_id": str(msg.from_user.id) if msg and msg.from_user else "",
                        "text": msg.text or "" if msg else "",
                        "message_id": str(msg.message_id) if msg else "",
                    }
                )
            return json.dumps({"ok": True, "updates": results}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_poll(
        self,
        chat_id: str,
        question: str,
        options_json: str,
        is_anonymous: bool = True,
    ) -> str:
        """Send a poll to a chat.

        Args:
            chat_id: Chat ID.
            question: Poll question.
            options_json: JSON array of option strings (2-10 options).
            is_anonymous: Whether the poll is anonymous. Default: True.

        Returns:
            JSON string with ok status and message_id.
        """
        assert self._bot is not None
        try:
            cid: Any = int(chat_id) if chat_id.lstrip("-").isdigit() else chat_id
            options = json.loads(options_json)
            msg = self._bot.send_poll(
                chat_id=cid, question=question, options=options, is_anonymous=is_anonymous
            )
            return json.dumps({"ok": True, "message_id": msg.message_id})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def forward_message(self, chat_id: str, from_chat_id: str, message_id: str) -> str:
        """Forward a message to another chat.

        Args:
            chat_id: Target chat ID.
            from_chat_id: Source chat ID.
            message_id: ID of the message to forward.

        Returns:
            JSON string with ok status and message_id.
        """
        assert self._bot is not None
        try:
            to_cid: Any = int(chat_id) if chat_id.lstrip("-").isdigit() else chat_id
            from_cid: Any = (
                int(from_chat_id) if from_chat_id.lstrip("-").isdigit() else from_chat_id
            )
            msg = self._bot.forward_message(
                chat_id=to_cid, from_chat_id=from_cid, message_id=int(message_id)
            )
            return json.dumps({"ok": True, "message_id": msg.message_id})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class TelegramAgent(BaseChannelAgent, ChatSorcarAgent):
    """ChatSorcarAgent extended with Telegram Bot API tools.

    Example::

        agent = TelegramAgent()
        result = agent.run(prompt_template="Send 'Hello!' to chat 123456789")
    """

    def __init__(self) -> None:
        super().__init__("Telegram Agent")
        self._backend = TelegramChannelBackend()
        cfg = _config.load()
        if cfg:  # pragma: no branch
            try:
                from telegram import Bot

                self._backend._bot = Bot(token=cfg["bot_token"])
            except Exception:
                pass

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return self._backend._bot is not None

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_telegram_auth() -> str:
            """Check if the Telegram bot token is configured and valid.

            Returns:
                Authentication status or instructions for how to authenticate.
            """
            if agent._backend._bot is None:  # pragma: no branch
                return (
                    "Not authenticated with Telegram. Use authenticate_telegram(bot_token=...) "
                    "to configure. Get a token by messaging @BotFather on Telegram: "
                    "send /newbot, follow the prompts, and copy the HTTP API token."
                )
            try:
                me = agent._backend._bot.get_me()
                return json.dumps(
                    {
                        "ok": True,
                        "username": me.username,
                        "first_name": me.first_name,
                        "id": me.id,
                    }
                )
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_telegram(bot_token: str) -> str:
            """Store and validate a Telegram bot token.

            Args:
                bot_token: Bot token from @BotFather (e.g. "123456:ABC-DEF...").

            Returns:
                Validation result with bot info, or error message.
            """
            bot_token = bot_token.strip()
            if not bot_token:  # pragma: no branch
                return "bot_token cannot be empty."
            try:
                from telegram import Bot

                bot = Bot(token=bot_token)
                me = bot.get_me()
                _config.save({"bot_token": bot_token.strip()})
                agent._backend._bot = bot
                return json.dumps(
                    {
                        "ok": True,
                        "message": "Telegram token saved and validated.",
                        "username": me.username,
                        "id": me.id,
                    }
                )
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def clear_telegram_auth() -> str:
            """Clear the stored Telegram bot token.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._bot = None
            return "Telegram authentication cleared."

        return [check_telegram_auth, authenticate_telegram, clear_telegram_auth]


def _make_backend() -> TelegramChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = TelegramChannelBackend()
    cfg = _config.load()
    if not cfg:  # pragma: no branch
        print("Not authenticated. Run: kiss-telegram -t 'authenticate'")
        sys.exit(1)
    from telegram import Bot

    backend._bot = Bot(token=cfg["bot_token"])
    return backend


def main() -> None:
    """Run the TelegramAgent from the command line with chat persistence."""
    channel_main(
        TelegramAgent,
        "kiss-telegram",
        channel_name="Telegram",
        make_backend=_make_backend,
    )


if __name__ == "__main__":
    main()
