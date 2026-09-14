# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Telegram Agent — channel agent with Telegram Bot API tools.

Provides authenticated access to Telegram via a bot token from @BotFather.
Stores the token securely in ``~/.kiss/third_party_agents/telegram/config.json`` and
exposes a focused set of Telegram Bot API tools.

Usage::

    agent = TelegramAgent()
    agent.run(prompt_template="Send 'Hello!' to chat_id 123456789")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import requests

from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
    config_file_lock,
    write_private_file,
)

_TELEGRAM_DIR = Path.home() / ".kiss" / "third_party_agents" / "telegram"
_config = ChannelConfig(_TELEGRAM_DIR, ("bot_token",))

_DEFAULT_API_BASE = "https://api.telegram.org"


def _raw_config() -> dict[str, Any]:
    """Return the legacy config parsed WITHOUT type coercion.

    ``ChannelConfig.load_metadata`` stringifies every value, which would
    turn a JSON boolean ``true`` into the credential string ``"True"``;
    migration must see the real JSON types so a malformed config is
    rejected, not coerced.

    Returns:
        The parsed config dict, or ``{}`` when missing or not an object.
    """
    try:
        data = json.loads(_config.path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _scrub_config_token(expected: str | None = None) -> None:
    """Remove a vault-migrated ``bot_token`` from config.json.

    Finishes the Muse migration automatically: any non-secret settings
    survive and the file is deleted when nothing but the token was
    stored (the usual case — Telegram's config holds only the token).

    The whole read-compare-replace cycle runs under
    :func:`config_file_lock`, which every config writer shares, so it
    is a true compare-and-swap: a newer token a concurrent writer
    lands either arrives before the read (the comparison sees it and
    the scrub backs off) or after the replacement (it survives), never
    in between.

    Args:
        expected: When given, the exact token that was migrated; the
            key is scrubbed only if the config still holds that value.
            A concurrent writer that replaced it with a NEWER token
            since the migration is left untouched (its value must not
            be deleted — it never made it into the vault).
    """
    with config_file_lock(_config.path):
        try:
            cfg = json.loads(_config.path.read_text())
        except (OSError, ValueError):
            return
        if not isinstance(cfg, dict) or "bot_token" not in cfg:
            return
        if expected is not None and cfg.get("bot_token") != expected:
            return
        kept = {k: str(v) for k, v in cfg.items() if k != "bot_token" and v}
        # Raw primitives: config_file_lock is not reentrant, so the
        # locked save_json_config/clear_json_config must not be used.
        if kept:
            write_private_file(_config.path, json.dumps(kept, indent=2))
        elif _config.path.exists():  # pragma: no branch - read above proved it exists
            _config.path.unlink()


def _ns_message(msg: dict[str, Any] | None) -> Any:
    """Convert a Bot API message dict into an SDK-shaped namespace.

    Args:
        msg: The ``message``/``channel_post`` dict, or None.

    Returns:
        A namespace with ``message_id``/``text``/``chat``/``from_user``
        attributes, or None when *msg* is empty.
    """
    if not msg:
        return None
    sender = msg.get("from")
    return SimpleNamespace(
        message_id=msg.get("message_id"),
        text=msg.get("text"),
        chat=SimpleNamespace(id=(msg.get("chat") or {}).get("id")),
        from_user=SimpleNamespace(id=sender.get("id")) if sender else None,
    )


class _TelegramBot:
    """Synchronous Telegram Bot API adapter speaking raw Bot API JSON.

    Provides the slice of a sync ``Bot`` surface the backend uses
    (``get_me``, ``send_message``, ...) on top of a ``requests``-style
    session, so the agent has no dependency on ``python-telegram-bot``.

    Two transports share this class:

    * Muse mode (no *session* given): the request URL embeds the
      SURROGATE in the token path segment (``/bot<surrogate>/<Method>``)
      and a ``MuseBoundarySession`` carries it to the daemon, which
      splices in the real bot token — a path-kind vault credential —
      just before the send, so this process never holds it.
    * Legacy mode (a plain ``requests.Session``): *token* is the real
      bot token and requests go straight to the Bot API.

    Attributes:
        token: The surrogate (Muse) or real (legacy) bot token;
            ``_bot_token`` reads it for the backend's direct Bot API
            calls (poll/typing), which flow through the same transport.
    """

    def __init__(
        self, backend: TelegramChannelBackend, token: str, session: Any | None = None
    ) -> None:
        self._backend = backend
        self.token = token
        self._muse = session is None
        if session is None:
            from kiss.agents.third_party_agents.muse_auth.client import MuseBoundarySession

            session = MuseBoundarySession("telegram")
        self._session = session

    def _call(
        self,
        api_method: str,
        payload: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
    ) -> Any:
        """Execute one Bot API method over the configured transport.

        The API base is read from the backend per call, so tests can
        re-point an already-wired backend at an emulator.  The bearer
        header identifying the surrogate is only sent in Muse mode; in
        legacy mode the real token already sits in the URL.

        Args:
            api_method: Bot API method name (e.g. ``"getMe"``).
            payload: JSON payload (or multipart form fields with
                *files*).
            files: Optional ``requests``-style file mapping for uploads.

        Returns:
            The response envelope's ``result`` value.

        Raises:
            RuntimeError: On an error envelope or HTTP error (mirroring
                the SDK, which raises ``TelegramError``); Sentinel
                denials surface here with their grant instructions.
        """
        url = f"{self._backend._api_base}/bot{self.token}/{api_method}"
        headers = {"Authorization": f"Bearer {self.token}"} if self._muse else {}
        if files:
            resp = self._session.request(
                "POST", url, headers=headers, data=payload, files=files, timeout=120
            )
        else:
            resp = self._session.request(
                "POST", url, headers=headers, json=payload or {}, timeout=30
            )
        try:
            data = resp.json() if resp.content else {}
        except ValueError:
            data = {}
        if not (isinstance(data, dict) and data.get("ok")):
            detail = ""
            if isinstance(data, dict):
                error = data.get("error")
                detail = str(
                    data.get("description")
                    or (error.get("message", "") if isinstance(error, dict) else "")
                )
            raise RuntimeError(
                f"Telegram API {api_method} failed: HTTP {resp.status_code} {detail[:300]}"
            )
        return data.get("result")

    def get_me(self) -> Any:
        """Return the bot's own user namespace."""
        user = self._call("getMe") or {}
        return SimpleNamespace(
            id=user.get("id"),
            username=user.get("username"),
            first_name=user.get("first_name"),
        )

    def send_message(
        self, chat_id: Any, text: str, reply_to_message_id: int | None = None
    ) -> Any:
        """Send a text message; returns a namespace with ``message_id``."""
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = reply_to_message_id
        result = self._call("sendMessage", payload) or {}
        return SimpleNamespace(message_id=result.get("message_id"))

    def send_photo(self, chat_id: Any, photo: Any, caption: str | None = None) -> Any:
        """Send a photo by URL or open file object."""
        payload: dict[str, Any] = {"chat_id": chat_id}
        if caption:
            payload["caption"] = caption
        if isinstance(photo, str):
            payload["photo"] = photo
            result = self._call("sendPhoto", payload)
        else:
            result = self._call("sendPhoto", payload, files={"photo": photo})
        return SimpleNamespace(message_id=(result or {}).get("message_id"))

    def send_document(self, chat_id: Any, document: Any, caption: str = "") -> Any:
        """Send a document from an open file object."""
        payload: dict[str, Any] = {"chat_id": chat_id}
        if caption:
            payload["caption"] = caption
        result = self._call("sendDocument", payload, files={"document": document})
        return SimpleNamespace(message_id=(result or {}).get("message_id"))

    def edit_message_text(self, chat_id: Any, message_id: int, text: str) -> None:
        """Edit a message's text."""
        self._call(
            "editMessageText", {"chat_id": chat_id, "message_id": message_id, "text": text}
        )

    def delete_message(self, chat_id: Any, message_id: int) -> None:
        """Delete a message."""
        self._call("deleteMessage", {"chat_id": chat_id, "message_id": message_id})

    def pin_chat_message(self, chat_id: Any, message_id: int) -> None:
        """Pin a message."""
        self._call("pinChatMessage", {"chat_id": chat_id, "message_id": message_id})

    def unpin_chat_message(self, chat_id: Any, message_id: int) -> None:
        """Unpin one message."""
        self._call("unpinChatMessage", {"chat_id": chat_id, "message_id": message_id})

    def unpin_all_chat_messages(self, chat_id: Any) -> None:
        """Unpin every message in a chat."""
        self._call("unpinAllChatMessages", {"chat_id": chat_id})

    def get_chat(self, chat_id: Any) -> Any:
        """Return a chat namespace (id, title, type, username, description)."""
        chat = self._call("getChat", {"chat_id": chat_id}) or {}
        return SimpleNamespace(
            id=chat.get("id"),
            title=chat.get("title"),
            type=chat.get("type"),
            username=chat.get("username"),
            description=chat.get("description"),
        )

    def get_chat_member_count(self, chat_id: Any) -> int:
        """Return the number of members in a chat."""
        return int(self._call("getChatMemberCount", {"chat_id": chat_id}) or 0)

    def get_chat_member(self, chat_id: Any, user_id: int) -> Any:
        """Return a chat-member namespace (user, status)."""
        member = self._call("getChatMember", {"chat_id": chat_id, "user_id": user_id}) or {}
        user = member.get("user") or {}
        return SimpleNamespace(
            user=SimpleNamespace(
                id=user.get("id"),
                username=user.get("username"),
                first_name=user.get("first_name"),
            ),
            status=member.get("status"),
        )

    def ban_chat_member(self, chat_id: Any, user_id: int) -> None:
        """Ban a user from a chat."""
        self._call("banChatMember", {"chat_id": chat_id, "user_id": user_id})

    def unban_chat_member(self, chat_id: Any, user_id: int) -> None:
        """Unban a user from a chat."""
        self._call("unbanChatMember", {"chat_id": chat_id, "user_id": user_id})

    def get_updates(
        self, offset: int | None = None, limit: int = 10, timeout: int = 0
    ) -> list[Any]:
        """Return recent updates as SDK-shaped namespaces."""
        payload: dict[str, Any] = {"limit": limit, "timeout": timeout}
        if offset is not None:
            payload["offset"] = offset
        updates = self._call("getUpdates", payload) or []
        return [
            SimpleNamespace(
                update_id=u.get("update_id"),
                message=_ns_message(u.get("message")),
                channel_post=_ns_message(u.get("channel_post")),
            )
            for u in updates
        ]

    def send_poll(
        self, chat_id: Any, question: str, options: list[str], is_anonymous: bool = True
    ) -> Any:
        """Send a poll; returns a namespace with ``message_id``."""
        result = self._call(
            "sendPoll",
            {
                "chat_id": chat_id,
                "question": question,
                "options": options,
                "is_anonymous": is_anonymous,
            },
        ) or {}
        return SimpleNamespace(message_id=result.get("message_id"))

    def forward_message(self, chat_id: Any, from_chat_id: Any, message_id: int) -> Any:
        """Forward a message; returns a namespace with ``message_id``."""
        result = self._call(
            "forwardMessage",
            {"chat_id": chat_id, "from_chat_id": from_chat_id, "message_id": message_id},
        ) or {}
        return SimpleNamespace(message_id=result.get("message_id"))


class TelegramChannelBackend(ToolMethodBackend):
    """Channel backend for Telegram Bot API.

    Uses the sync :class:`_TelegramBot` adapter for most API calls;
    message polling (``poll_messages``) and typing indicators speak the
    Bot API directly over HTTP against ``_api_base`` so they honor
    persisted cursors and remain testable against a local server.
    """

    def __init__(self) -> None:
        self._bot: Any = None
        self._last_update_id: int = -1
        self._connection_info: str = ""
        self._api_base: str = _DEFAULT_API_BASE
        self._http: Any = requests
        self._muse: bool = False

    def _request_headers(self) -> dict[str, str]:
        """Return headers for direct Bot API calls (poll/typing).

        In Muse mode the surrogate bearer identifies the request at the
        daemon, which swaps the URL's ``/bot<surrogate>/`` path segment
        for the real token at the boundary; legacy mode needs no
        headers because the real token is already in the URL.

        Returns:
            Header dict for :attr:`_http` requests.
        """
        if self._muse:
            return {"Authorization": f"Bearer {self._bot_token()}"}
        return {}

    def _wire_muse(self) -> bool:
        """Acquire a Telegram surrogate and wire the boundary transport.

        On the FIRST migration (the vault holds no ``telegram``
        credential yet) a ``bot_token`` in the legacy config seeds the
        vault as a path-kind credential and is scrubbed from
        ``config.json`` afterwards.  Once the vault holds a credential
        it is authoritative: a bare config token is NOT auto-applied,
        because an unvalidated config value (a typo, a bad rotation
        done while Muse was off) must never clobber a working vault
        credential — rotations go through ``authenticate_telegram``,
        which validates the candidate before replacing anything.  No
        network round trip happens here.

        Returns:
            True when the backend holds a surrogate-backed adapter.
        """
        from kiss.agents.third_party_agents.muse_auth.client import (
            MuseBoundarySession,
            mint_surrogate,
            store_credentials,
        )

        migrated = False
        token = _raw_config().get("bot_token")
        if isinstance(token, str) and token:
            # Store-if-absent is ATOMIC in the daemon: a config token
            # seeds the vault only when it holds no credential yet, and
            # a concurrent authoritative writer can never be clobbered
            # by this stale config candidate.  A non-string/malformed
            # token is refused by store_credentials (path-splice
            # validation) before anything is enrolled or scrubbed.
            migrated = store_credentials(
                "telegram", {"kind": "path", "token": token}, [], only_if_absent=True
            )
        handle = mint_surrogate("telegram")
        if handle is None:
            self._connection_info = "No Telegram credential in the Muse vault or config."
            return False
        if migrated and isinstance(token, str):
            # Compare-and-scrub: only remove the token we actually
            # migrated, so a newer token a concurrent writer placed in
            # config between the store and here is not deleted.
            _scrub_config_token(expected=token)
        self._bot = _TelegramBot(self, handle.token)
        self._http = MuseBoundarySession("telegram")
        self._muse = True
        return True

    def _bot_token(self) -> str:
        """Return the bot token from the live Bot or the stored config.

        Returns:
            The bot token string, or ``""`` when neither the connected
            ``Bot`` instance nor the persisted config provides one.
        """
        token = str(getattr(self._bot, "token", "") or "")
        if token:
            return token
        cfg = _config.load()
        return cfg["bot_token"] if cfg else ""

    def connect(self) -> bool:
        """Authenticate with Telegram using the stored bot token."""
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled
        from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError

        if muse_auth_enabled():
            # Vault-first surrogate wiring; the getMe validation below
            # runs through the daemon boundary (audited).  A malformed
            # legacy-config token makes the daemon refuse enrollment
            # (MuseAuthError); fail closed with a bool like the legacy
            # path rather than letting it escape this documented -> bool
            # method.
            try:
                if not self._wire_muse():
                    return False
            except MuseAuthError as e:
                self._connection_info = f"Telegram auth failed: {e}"
                return False
            try:
                me = self._bot.get_me()
                self._connection_info = f"Authenticated as @{me.username}"
                return True
            except Exception as e:
                self._connection_info = f"Telegram auth failed: {e}"
                return False
        cfg = _config.load()
        if not cfg:  # pragma: no branch
            self._connection_info = "No Telegram token found."
            return False
        try:
            self._bot = _TelegramBot(self, cfg["bot_token"], requests.Session())
            me = self._bot.get_me()
            self._connection_info = f"Authenticated as @{me.username}"
            return True
        except Exception as e:
            self._connection_info = f"Telegram auth failed: {e}"
            return False

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll for new Telegram updates via the Bot API ``getUpdates`` method.

        Cursor contract (Telegram offset-confirmation convention):

        - ``oldest`` is the persisted cursor from the previous poll.  When it
          is a digit-string greater than ``0`` it is sent as the
          ``getUpdates`` ``offset``, telling Telegram to confirm (drop) every
          update with ``update_id < offset`` and return only newer ones.
          When ``oldest`` is ``"0"``, empty, or non-numeric, the legacy
          process-local behavior applies: the offset is derived from
          ``_last_update_id + 1`` when a previous in-process poll saw
          updates, or omitted entirely on a fresh backend.
        - The process-local ``_last_update_id`` keeps working for the
          interactive tools path; when both a numeric ``oldest`` and a
          process-local offset are available, the **maximum** of the two is
          used so polling is monotonic and never re-fetches updates either
          source has already confirmed.
        - The returned ``new_cursor`` is ``str(highest update_id processed
          + 1)`` — suitable for persisting verbatim and passing back as
          ``oldest`` on the next poll — or the passed-in ``oldest``
          unchanged when no updates arrived.

        Never raises: any transport, HTTP, or parse failure returns
        ``([], oldest)`` so a failed poll can never break a channel tick.

        Args:
            channel_id: Chat ID to filter messages to; empty for all chats.
            oldest: Cursor from the previous poll (see contract above).
            limit: Maximum number of updates to fetch (Telegram cap: 100).

        Returns:
            Tuple of (list of normalized message dicts with ``ts``,
            ``date``, ``user``, ``text``, ``message_id``, and ``chat_id``
            keys, new cursor string).
        """
        try:
            legacy = self._last_update_id + 1 if self._last_update_id >= 0 else None
            requested = int(oldest) if oldest.isdigit() and int(oldest) > 0 else None
            candidates = [c for c in (legacy, requested) if c is not None]
            payload: dict[str, Any] = {"timeout": 0, "limit": min(limit, 100)}
            if candidates:
                payload["offset"] = max(candidates)
            response = self._http.post(
                f"{self._api_base}/bot{self._bot_token()}/getUpdates",
                headers=self._request_headers(),
                json=payload,
                timeout=30,
            )
            data = response.json()
            if not data.get("ok"):
                return [], oldest
            messages: list[dict[str, Any]] = []
            highest = -1
            for update in data.get("result", []):
                update_id = int(update["update_id"])
                highest = max(highest, update_id)
                if update_id > self._last_update_id:
                    self._last_update_id = update_id
                msg = update.get("message") or update.get("channel_post")
                if msg and msg.get("text"):
                    chat_id = str(msg["chat"]["id"])
                    if not channel_id or chat_id == channel_id:
                        messages.append(
                            {
                                "ts": str(msg["message_id"]),
                                "date": str(float(msg["date"])) if msg.get("date") else "",
                                "user": str(msg["from"]["id"]) if msg.get("from") else "",
                                "text": msg["text"],
                                "message_id": str(msg["message_id"]),
                                "chat_id": chat_id,
                            }
                        )
            new_cursor = str(highest + 1) if highest >= 0 else oldest
            return messages, new_cursor
        except Exception:
            return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a Telegram message."""
        assert self._bot is not None
        kwargs: dict[str, Any] = {"chat_id": int(channel_id), "text": text}
        if thread_ts:  # pragma: no branch
            kwargs["reply_to_message_id"] = int(thread_ts)
        self._bot.send_message(**kwargs)

    def send_typing(self, channel_id: str, thread_ts: str = "") -> None:
        """Show a "typing…" indicator in a Telegram chat (best-effort).

        POSTs the Bot API ``sendChatAction`` method with
        ``action="typing"``, which displays "bot is typing…" for a few
        seconds or until the next message is sent.  Any failure — a
        missing token, a non-2xx response, or an unreachable server —
        is swallowed so a failed indicator can never break message
        handling.

        Args:
            channel_id: Chat ID (integer as string) or @username.
            thread_ts: Reply-target message ID, accepted for interface
                parity with :meth:`send_message`.  Ignored here because
                :meth:`send_message` threads via ``reply_to_message_id``
                (a plain reply inside *channel_id*, not a forum topic),
                and ``sendChatAction`` has no reply-target parameter,
                so the indicator belongs to the whole chat.
        """
        del thread_ts
        try:
            token = self._bot_token()
            if not token:
                return
            cid: Any = int(channel_id) if channel_id.lstrip("-").isdigit() else channel_id
            self._http.post(
                f"{self._api_base}/bot{token}/sendChatAction",
                headers=self._request_headers(),
                json={"chat_id": cid, "action": "typing"},
                timeout=30,
            )
        except Exception:
            pass

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


def _probe_candidate_token(backend: TelegramChannelBackend, bot_token: str) -> tuple[str, Any]:
    """Validate a candidate bot token without touching the live enrollment.

    The candidate is enrolled under the scratch service
    ``telegram-pending`` (which inherits Telegram's host pinning and
    action classification), a ``getMe`` probe runs through the audited
    daemon boundary, and the scratch entry is removed again — so a
    rejected rotation can never destroy an existing working
    ``telegram`` vault credential.

    Args:
        backend: The agent's Telegram backend (supplies the API base).
        bot_token: Candidate bot token (already path-splice-validated).

    Returns:
        ``("", user_dict)`` on success (the ``getMe`` result), or
        ``(error_message, {})`` when the API rejects the token.
    """
    import contextlib
    import secrets

    from kiss.agents.third_party_agents.muse_auth.client import (
        MuseBoundarySession,
        clear_credentials,
        mint_surrogate,
        store_credentials,
    )

    # A per-attempt unique scratch service (``telegram-pending-<hex>``)
    # so two concurrent validations can never probe or clear each
    # other's candidate; it still resolves to ``telegram`` for host and
    # policy inheritance via service_root.
    scratch = f"telegram-pending-{secrets.token_hex(8)}"
    store_credentials(scratch, {"kind": "path", "token": bot_token}, [])
    try:
        handle = mint_surrogate(scratch)
        if handle is None:  # pragma: no cover - defense in depth
            return "could not mint a scratch validation surrogate", {}
        resp = MuseBoundarySession(scratch).request(
            "POST",
            f"{backend._api_base}/bot{handle.token}/getMe",
            headers={"Authorization": f"Bearer {handle.token}"},
            json={},
            timeout=30,
        )
        try:
            data = resp.json() if resp.content else {}
        except ValueError:
            data = {}
        if isinstance(data, dict) and data.get("ok"):
            return "", dict(data.get("result") or {})
        detail = ""
        if isinstance(data, dict):
            error = data.get("error")
            detail = str(
                data.get("description")
                or (error.get("message", "") if isinstance(error, dict) else "")
            )
        return f"Telegram API getMe failed: HTTP {resp.status_code} {detail[:300]}", {}
    finally:
        with contextlib.suppress(Exception):
            clear_credentials(scratch)


def _muse_authenticate(backend: TelegramChannelBackend, bot_token: str) -> str:
    """Enroll a Telegram bot token into the Muse vault and validate it.

    The candidate is validated FIRST, against a scratch ``-pending``
    enrollment, so a rejected token mutates nothing — neither the
    config nor an existing working vault credential.  Only a proven
    token replaces the live enrollment; the plaintext is never written
    to ``config.json`` (any legacy copy there is removed).  If the
    swap itself fails midway, the pre-call config bytes are restored —
    the vault is never cleared, because whichever credential it holds
    at that point (the untouched old one or the just-validated new
    one) is worth keeping.

    Args:
        backend: The agent's Telegram backend to (re)wire.
        bot_token: Bot token from @BotFather.

    Returns:
        JSON string with the validation result.
    """
    import contextlib

    from kiss.agents.third_party_agents.muse_auth._common import valid_credential_path_value
    from kiss.agents.third_party_agents.muse_auth.client import store_credentials

    if not valid_credential_path_value(bot_token):
        # Pre-validate with the vault's own path-splice rule so a
        # doomed enrollment never mutates any stored state.
        return json.dumps(
            {
                "ok": False,
                "error": "bot_token contains characters that are not URL-path-safe; "
                "copy it exactly from @BotFather.",
            }
        )
    try:
        failure, user = _probe_candidate_token(backend, bot_token)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})
    if failure:
        return json.dumps({"ok": False, "error": failure})
    try:
        prev_raw: str | None = _config.path.read_text()
    except OSError:
        prev_raw = None
    try:
        # The Telegram config exists only to hold the token, which now
        # lives in the vault; remove any plaintext copy first.
        _config.clear()
        store_credentials("telegram", {"kind": "path", "token": bot_token}, [])
        if backend._wire_muse():  # pragma: no branch - credential was just stored
            return json.dumps(
                {
                    "ok": True,
                    "message": "Telegram token saved and validated (Muse-auth).",
                    "username": user.get("username"),
                    "id": user.get("id"),
                }
            )
        error = json.dumps(  # pragma: no cover - defense in depth
            {"ok": False, "error": backend._connection_info}
        )
    except Exception as e:
        error = json.dumps({"ok": False, "error": str(e)})
    # Restore the pre-call config bytes; a failed swap must not leave
    # half-migrated state (the restored bytes are exactly what was on
    # disk before, so no new secret lands in the file).
    with contextlib.suppress(Exception):
        if prev_raw is None:
            _config.clear()
        else:
            # Atomic 0600 restore, serialized against every other
            # config writer via the shared config lock.
            with config_file_lock(_config.path):
                write_private_file(_config.path, prev_raw)
    backend._bot = None
    backend._muse = False
    backend._http = requests
    return error


class TelegramAgent(BaseChannelAgent):
    """Channel agent with Telegram Bot API tools.

    Example::

        agent = TelegramAgent()
        result = agent.run(prompt_template="Send 'Hello!' to chat 123456789")
    """

    def __init__(self) -> None:
        super().__init__("Telegram Agent")
        self._backend = TelegramChannelBackend()
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        if muse_auth_enabled():
            from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError

            # Muse-auth mode: wire a vault surrogate and the boundary
            # adapter (no network round trip); the real token never
            # enters this process once migrated.  A daemon failure
            # leaves the agent constructible (fail closed, tokenless)
            # so its authenticate/clear tools stay available.
            try:
                self._backend._wire_muse()
            except MuseAuthError as e:
                self._backend._bot = None
                self._backend._connection_info = f"Muse-auth wiring failed: {e}"
            return
        cfg = _config.load()
        if cfg:  # pragma: no branch
            self._backend._bot = _TelegramBot(
                self._backend, cfg["bot_token"], requests.Session()
            )

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
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if muse_auth_enabled():
                return _muse_authenticate(agent._backend, bot_token)
            try:
                bot = _TelegramBot(agent._backend, bot_token, requests.Session())
                me = bot.get_me()
                _config.save({"bot_token": bot_token})
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
            agent._backend._muse = False
            agent._backend._http = requests
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if muse_auth_enabled():
                from kiss.agents.third_party_agents.muse_auth.client import clear_credentials

                clear_credentials("telegram")
            return "Telegram authentication cleared."

        return [check_telegram_auth, authenticate_telegram, clear_telegram_auth]


def _make_backend() -> TelegramChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = TelegramChannelBackend()
    from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled
    from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError

    if muse_auth_enabled():
        try:
            wired = backend._wire_muse()
        except MuseAuthError:
            wired = False
        if wired:
            return backend
        print("Not authenticated. Run: kiss-telegram -t 'authenticate'")
        sys.exit(1)
    cfg = _config.load()
    if not cfg:  # pragma: no branch
        print("Not authenticated. Run: kiss-telegram -t 'authenticate'")
        sys.exit(1)
    backend._bot = _TelegramBot(backend, cfg["bot_token"], requests.Session())
    return backend


def main() -> None:
    """Run the TelegramAgent from the command line with chat persistence."""
    channel_main(
        TelegramAgent,
        "kiss-telegram",
        channel_name="Telegram",
        make_backend=_make_backend,
    )


def tools() -> list:
    """Return the Telegram channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return TelegramAgent()._get_tools()


if __name__ == "__main__":
    main()
