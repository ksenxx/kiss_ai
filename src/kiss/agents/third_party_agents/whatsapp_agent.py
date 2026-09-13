# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""WhatsApp Agent — channel agent for a personal WhatsApp account, QR-paired.

Uses the same approach as the ``whatsapp`` entry in ``connectors/``: the
`lharries/whatsapp-mcp <https://github.com/lharries/whatsapp-mcp>`_ Go
bridge speaks the WhatsApp Web multidevice protocol (whatsmeow), pairs
once via a QR code scanned from the phone, and mirrors all message
history into a local SQLite database (``whatsapp-bridge/store/``).
Nothing new sees the traffic — it is the normal end-to-end-encrypted
WhatsApp Web protocol, and all data stays on this machine.

This module manages the bridge itself (clone, build, run, QR pairing)
and exposes messaging tools that read the bridge's SQLite database and
call its localhost REST API — no Meta Business account, access token,
or webhook is involved.

Pairing renders the bridge's QR code into a local HTML page
(``~/.kiss/third_party_agents/whatsapp/qr.html``) so the agent can show
it in the browser for the user to scan with their phone (WhatsApp →
Settings → Linked devices → Link a device).

Usage::

    agent = WhatsAppAgent()
    agent.run(prompt_template="Send 'Hello!' to +1234567890")
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)
from kiss.core.config import kiss_home

logger = logging.getLogger(__name__)

_WHATSAPP_DIR = Path.home() / ".kiss" / "third_party_agents" / "whatsapp"
_BRIDGE_REPO_URL = "https://github.com/lharries/whatsapp-mcp"
_BRIDGE_BINARY_NAME = "kiss-whatsapp-bridge"
# The upstream bridge hardcodes its REST port (startRESTServer(..., 8080)).
_DEFAULT_BRIDGE_PORT = 8080
_PAIRED_MARKER = "Successfully connected and authenticated!"
_CONNECTED_MARKER = "Connected to WhatsApp!"
_QR_TIMEOUT_MARKER = "Timeout waiting for QR code scan"
# qrterminal.GenerateHalfBlock output: QR-dark modules are SPACES, QR-light
# modules are full/half blocks, so a QR line contains only these 4 chars.
_QR_LINE_CHARS = frozenset("█▀▄ ")
_config = ChannelConfig(_WHATSAPP_DIR, ())


def _channel_dir() -> Path:
    """Return the WhatsApp channel data directory, honouring ``KISS_HOME``."""
    return kiss_home() / "third_party_agents" / "whatsapp"


def _default_repo_dir() -> Path:
    """Return the default whatsapp-mcp clone location.

    Reuses an existing ``~/.kiss/connectors/whatsapp-mcp`` clone (made by
    ``connectors/enable.py enable whatsapp``) so the device is paired only
    once; otherwise a clone inside the channel directory is used.
    """
    connectors_clone = kiss_home() / "connectors" / "whatsapp-mcp"
    if (connectors_clone / "whatsapp-bridge" / "main.go").exists():
        return connectors_clone
    return _channel_dir() / "whatsapp-mcp"


def _apply_config(backend: WhatsAppChannelBackend) -> None:
    """Apply the persisted config (repo_dir, bridge_port) to *backend*.

    Args:
        backend: The backend to configure.
    """
    cfg = _config.load() or {}
    if cfg.get("repo_dir"):
        backend._repo_dir = cfg["repo_dir"]
    try:
        if cfg.get("bridge_port"):
            backend._bridge_port = int(cfg["bridge_port"])
    except ValueError:
        logger.warning("Ignoring invalid bridge_port in %s", _config.path)


def _dump_clipped(payload: dict[str, Any], *list_keys: str) -> str:
    """Serialize *payload* to JSON, bounded to roughly 8000 characters.

    Unlike slicing the encoded text (which cuts inside JSON tokens), this
    drops trailing items from the named list values until the result fits,
    marking the payload with ``"truncated": true``.

    Args:
        payload: The response dict to serialize.
        *list_keys: Keys of list values that may be shortened.

    Returns:
        Valid JSON text.
    """
    text = json.dumps(payload, indent=2)
    lists = [payload[k] for k in list_keys if isinstance(payload.get(k), list)]
    while len(text) > 7900 and any(lists):
        max(lists, key=len).pop()
        payload["truncated"] = True
        text = json.dumps(payload, indent=2)
    return text


def _parse_time_bound(value: str) -> tuple[str, str]:
    """Validate an ISO-8601 time bound for message filters.

    Args:
        value: User-supplied timestamp text (``T`` or space separator).

    Returns:
        Tuple of (normalized bound, "") on success, or ("", error message).
    """
    from datetime import datetime

    try:
        return datetime.fromisoformat(value).isoformat(sep=" "), ""
    except ValueError:
        return "", (
            f"Invalid date format: {value!r}. Use ISO-8601 (UTC), "
            "e.g. '2026-09-10 10:00:00'."
        )


def _sender_forms(contact: str) -> tuple[str, str]:
    """Return both stored sender representations for a contact identifier.

    The bridge stores live-event senders as the bare user (digits) but
    history-synced group senders as the full JID, so matching must accept
    both exact forms.

    Args:
        contact: A JID or phone number in any common format.

    Returns:
        Tuple of (bare user, full JID).
    """
    jid = _to_jid(contact)
    return jid.split("@")[0], jid


def _to_jid(recipient: str) -> str:
    """Normalize a recipient to a WhatsApp JID.

    Args:
        recipient: A JID (``123@s.whatsapp.net``, group ``123@g.us``) or a
            phone number in any common format (``+1 415-555-2671``).

    Returns:
        The JID unchanged, or ``<digits>@s.whatsapp.net`` for a phone number.
    """
    recipient = recipient.strip()
    if "@" in recipient:
        return recipient
    return re.sub(r"\D", "", recipient) + "@s.whatsapp.net"


def _rest_recipient(recipient: str) -> str:
    """Normalize a recipient for the bridge REST API.

    The bridge accepts either a bare number (country code, digits only)
    or a full JID.

    Args:
        recipient: JID or phone number in any common format.

    Returns:
        The JID unchanged, or the digits of the phone number.
    """
    recipient = recipient.strip()
    if "@" in recipient:
        return recipient
    return re.sub(r"\D", "", recipient)


class WhatsAppChannelBackend(ToolMethodBackend):
    """Channel backend for personal WhatsApp via the whatsapp-mcp Go bridge.

    Reads message history from the bridge's SQLite database
    (``whatsapp-bridge/store/messages.db``) and sends messages / downloads
    media through the bridge's localhost REST API (``/api/send``,
    ``/api/download``).
    """

    def __init__(self, repo_dir: str = "", bridge_port: int = _DEFAULT_BRIDGE_PORT) -> None:
        """Initialize the backend.

        Args:
            repo_dir: Path of the whatsapp-mcp clone. Empty selects the
                default location (the connectors clone when present).
            bridge_port: Port of the bridge REST API. The upstream bridge
                always listens on 8080; override only for a patched bridge
                or a test stand-in.
        """
        self._repo_dir = repo_dir
        self._bridge_port = bridge_port
        self._connection_info: str = ""

    @property
    def repo_dir(self) -> Path:
        """Absolute path of the whatsapp-mcp clone."""
        if self._repo_dir:
            return Path(self._repo_dir).expanduser().resolve()
        return _default_repo_dir()

    @property
    def bridge_dir(self) -> Path:
        """Path of the Go bridge directory inside the clone."""
        return self.repo_dir / "whatsapp-bridge"

    @property
    def messages_db(self) -> Path:
        """Path of the bridge's message-history SQLite database."""
        return self.bridge_dir / "store" / "messages.db"

    @property
    def session_db(self) -> Path:
        """Path of the bridge's whatsmeow session SQLite database."""
        return self.bridge_dir / "store" / "whatsapp.db"

    def _is_paired(self) -> bool:
        """Return True if a paired WhatsApp device session exists on disk.

        The bridge creates ``store/whatsapp.db`` on first launch even
        before the QR code is scanned, so mere file existence is not
        pairing: a paired session has a row in whatsmeow's device table.
        """
        if not self.session_db.exists():
            return False
        try:
            conn = sqlite3.connect(f"file:{self.session_db}?mode=ro", uri=True, timeout=10)
            try:
                rows = conn.execute("SELECT count(*) FROM whatsmeow_device").fetchone()
            finally:
                conn.close()
            return bool(rows and rows[0])
        except sqlite3.Error:
            # Unreadable or pre-device-table database: treat as unpaired.
            return False

    def _api_url(self, endpoint: str) -> str:
        """Return the bridge REST API URL for *endpoint* (``send``/``download``)."""
        return f"http://127.0.0.1:{self._bridge_port}/api/{endpoint}"

    def _bridge_running(self) -> bool:
        """Return True if the bridge REST API answers on its port.

        The bridge's ``/api/send`` handler only allows POST and answers a
        GET with exactly 405; requiring that status keeps an unrelated
        service on the same port from being mistaken for the bridge.
        """
        try:
            resp = requests.get(self._api_url("send"), timeout=3)
            return resp.status_code == 405
        except requests.RequestException:
            return False

    def _api_send(self, payload: dict[str, str]) -> tuple[bool, str]:
        """POST *payload* to the bridge ``/api/send`` endpoint.

        Args:
            payload: JSON body with ``recipient`` and ``message`` and/or
                ``media_path``.

        Returns:
            Tuple of (success, status message).
        """
        try:
            resp = requests.post(self._api_url("send"), json=payload, timeout=60)
        except requests.RequestException as e:
            return False, (
                f"Bridge not reachable: {e}. Start it with start_whatsapp_bridge()."
            )
        try:
            result = resp.json()
        except ValueError:
            return False, f"HTTP {resp.status_code}: {resp.text[:500]}"
        return bool(result.get("success", False)), str(result.get("message", ""))

    def _query(self, sql: str, params: tuple[Any, ...]) -> list[tuple[Any, ...]]:
        """Run a read-only SQL query against the message database.

        Args:
            sql: SQL SELECT statement.
            params: Bound query parameters.

        Returns:
            All result rows.

        Raises:
            FileNotFoundError: If the message database does not exist yet.
            sqlite3.Error: On SQL errors.
        """
        if not self.messages_db.exists():
            raise FileNotFoundError(
                f"{self.messages_db} not found. Pair WhatsApp first "
                "(check_whatsapp_auth() explains the steps) and give the "
                "bridge a few minutes to sync history."
            )
        conn = sqlite3.connect(f"file:{self.messages_db}?mode=ro", uri=True, timeout=10)
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Channel protocol
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Verify the device is paired and the bridge is reachable.

        Returns:
            True on success, False (with ``connection_info``) otherwise.
        """
        _apply_config(self)
        if not self._is_paired():
            self._connection_info = (
                "WhatsApp is not paired. Run: kiss-whatsapp -t 'authenticate whatsapp'"
            )
            return False
        if not self._bridge_running():
            self._connection_info = (
                "WhatsApp bridge is not running. "
                "Run: kiss-whatsapp -t 'start the whatsapp bridge'"
            )
            return False
        self._connection_info = f"WhatsApp bridge connected (repo: {self.repo_dir})"
        return True

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Return new messages from the bridge's SQLite database.

        Args:
            channel_id: Chat to monitor — a JID or a phone number
                (normalized to ``<digits>@s.whatsapp.net``). Empty
                monitors all chats.
            oldest: Timestamp cursor — the raw ``messages.timestamp``
                string of the newest message already seen. ``""``/``"0"``
                fetches the most recent messages without an after-filter.
            limit: Maximum messages to return.

        Returns:
            Tuple of (messages oldest-first, new cursor). Each message
            dict has ts, user (sender), text, id, chat_jid, is_from_me.
        """
        where = []
        params: list[Any] = []
        has_cursor = oldest not in ("", "0")
        if channel_id:
            where.append("chat_jid = ?")
            params.append(_to_jid(channel_id))
        if has_cursor:
            where.append("timestamp > ?")
            params.append(oldest)
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        # With a cursor, take the OLDEST unseen rows so a burst larger
        # than *limit* is delivered across ticks instead of being skipped;
        # without one, seed from the most recent rows.
        order = "ASC" if has_cursor else "DESC"
        try:
            rows = self._query(
                "SELECT id, chat_jid, sender, content, timestamp, is_from_me, media_type "
                f"FROM messages {where_sql} ORDER BY timestamp {order}, id LIMIT ?",
                (*params, limit),
            )
        except (FileNotFoundError, sqlite3.Error) as e:
            logger.warning("WhatsApp poll failed: %s", e)
            return [], oldest
        if not has_cursor:
            rows = list(reversed(rows))
        new_cursor = oldest
        messages: list[dict[str, Any]] = []
        for msg_id, chat_jid, sender, content, ts, is_from_me, media_type in rows:
            ts = str(ts or "")
            if ts > new_cursor:
                new_cursor = ts
            messages.append(
                {
                    "ts": ts,
                    "user": str(sender or ""),
                    "text": str(content or "") or f"[{media_type} message]",
                    "id": str(msg_id or ""),
                    "chat_jid": str(chat_jid or ""),
                    "is_from_me": bool(is_from_me),
                }
            )
        return messages, new_cursor

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a text message through the bridge REST API.

        Args:
            channel_id: Recipient JID or phone number.
            text: Message text.
            thread_ts: Unused for WhatsApp.

        Raises:
            RuntimeError: If the bridge reports failure or is unreachable.
        """
        ok, message = self._api_send(
            {"recipient": _rest_recipient(channel_id), "message": text}
        )
        if not ok:
            raise RuntimeError(f"WhatsApp send failed: {message}")

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Return True for messages sent from this paired account."""
        return bool(msg.get("is_from_me"))

    # ------------------------------------------------------------------
    # Agent tools (public methods are exposed automatically)
    # ------------------------------------------------------------------

    def search_whatsapp_contacts(self, query: str) -> str:
        """Search WhatsApp contacts by name or phone number.

        Args:
            query: Search term matched (case-insensitively) against
                contact names and JIDs.

        Returns:
            JSON string with a list of contacts (jid, name, phone_number).
        """
        try:
            rows = self._query(
                "SELECT DISTINCT jid, name FROM chats "
                "WHERE (LOWER(name) LIKE LOWER(?) OR LOWER(jid) LIKE LOWER(?)) "
                "AND jid NOT LIKE '%@g.us' ORDER BY name, jid LIMIT 50",
                (f"%{query}%", f"%{query}%"),
            )
            contacts = [
                {"jid": jid, "name": name, "phone_number": str(jid).split("@")[0]}
                for jid, name in rows
            ]
            return _dump_clipped({"ok": True, "contacts": contacts}, "contacts")
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_whatsapp_chats(
        self,
        query: str = "",
        limit: int = 20,
        page: int = 0,
        include_last_message: bool = True,
        sort_by: str = "last_active",
    ) -> str:
        """List WhatsApp chats (direct and group) with metadata.

        Args:
            query: Optional term to filter chats by name or JID.
            limit: Maximum chats to return. Default: 20.
            page: Page number for pagination. Default: 0.
            include_last_message: Include each chat's last message.
            sort_by: "last_active" (default) or "name".

        Returns:
            JSON string with a list of chats (jid, name,
            last_message_time, and optionally last_message details).
        """
        try:
            if include_last_message:
                select = (
                    "SELECT chats.jid, chats.name, chats.last_message_time, "
                    "messages.content, messages.sender, messages.is_from_me FROM chats "
                    "LEFT JOIN messages ON chats.jid = messages.chat_jid "
                    "AND chats.last_message_time = messages.timestamp"
                )
                # Same-second messages can tie last_message_time; keep one
                # row per chat so LIMIT/OFFSET stay correct.
                group_by = "GROUP BY chats.jid"
            else:
                select = (
                    "SELECT chats.jid, chats.name, chats.last_message_time, "
                    "NULL, NULL, NULL FROM chats"
                )
                group_by = ""
            where = ""
            params: list[Any] = []
            if query:
                where = "WHERE (LOWER(chats.name) LIKE LOWER(?) OR chats.jid LIKE ?)"
                params += [f"%{query}%", f"%{query}%"]
            order = "chats.last_message_time DESC" if sort_by == "last_active" else "chats.name"
            rows = self._query(
                f"{select} {where} {group_by} ORDER BY {order} LIMIT ? OFFSET ?",
                (*params, limit, page * limit),
            )
            chats = [self._chat_dict(row, include_last_message) for row in rows]
            return _dump_clipped({"ok": True, "chats": chats}, "chats")
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def _chat_dict(self, row: tuple[Any, ...], include_last_message: bool) -> dict[str, Any]:
        """Convert a chats+last-message row into a result dict.

        Args:
            row: (jid, name, last_message_time, content, sender, is_from_me).
            include_last_message: Whether the message columns are meaningful.

        Returns:
            Chat dict for JSON serialization.
        """
        chat: dict[str, Any] = {
            "jid": row[0],
            "name": row[1],
            "last_message_time": row[2],
        }
        if include_last_message:
            chat["last_message"] = row[3]
            chat["last_sender"] = row[4]
            chat["last_is_from_me"] = bool(row[5]) if row[5] is not None else None
        return chat

    def get_whatsapp_chat(self, chat_jid: str, include_last_message: bool = True) -> str:
        """Get WhatsApp chat metadata by JID.

        Args:
            chat_jid: The chat JID (``...@s.whatsapp.net`` or ``...@g.us``).
            include_last_message: Include the chat's last message.

        Returns:
            JSON string with the chat metadata, or an error.
        """
        try:
            rows = self._query(
                "SELECT c.jid, c.name, c.last_message_time, m.content, m.sender, "
                "m.is_from_me FROM chats c LEFT JOIN messages m ON c.jid = m.chat_jid "
                "AND c.last_message_time = m.timestamp WHERE c.jid = ? LIMIT 1",
                (chat_jid,),
            )
            if not rows:
                return json.dumps({"ok": False, "error": f"Chat not found: {chat_jid}"})
            return json.dumps(
                {"ok": True, "chat": self._chat_dict(rows[0], include_last_message)}, indent=2
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_whatsapp_direct_chat_by_contact(self, sender_phone_number: str) -> str:
        """Find the direct (non-group) chat with a phone number.

        Args:
            sender_phone_number: Phone number to search for (digits;
                country code included).

        Returns:
            JSON string with the chat metadata, or an error.
        """
        try:
            digits = re.sub(r"\D", "", sender_phone_number)
            rows = self._query(
                "SELECT c.jid, c.name, c.last_message_time, m.content, m.sender, "
                "m.is_from_me FROM chats c LEFT JOIN messages m ON c.jid = m.chat_jid "
                "AND c.last_message_time = m.timestamp "
                "WHERE c.jid LIKE ? AND c.jid NOT LIKE '%@g.us' LIMIT 1",
                (f"%{digits}%",),
            )
            if not rows:
                return json.dumps(
                    {"ok": False, "error": f"No direct chat found with {sender_phone_number}"}
                )
            return json.dumps({"ok": True, "chat": self._chat_dict(rows[0], True)}, indent=2)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_whatsapp_contact_chats(self, jid: str, limit: int = 20, page: int = 0) -> str:
        """List all chats (direct and group) involving a contact.

        Args:
            jid: The contact's JID (``...@s.whatsapp.net``).
            limit: Maximum chats to return. Default: 20.
            page: Page number for pagination. Default: 0.

        Returns:
            JSON string with the list of chats.
        """
        try:
            rows = self._query(
                "SELECT DISTINCT c.jid, c.name, c.last_message_time, NULL, NULL, NULL "
                "FROM chats c JOIN messages m ON c.jid = m.chat_jid "
                "WHERE m.sender IN (?, ?) OR c.jid = ? "
                "ORDER BY c.last_message_time DESC LIMIT ? OFFSET ?",
                (*_sender_forms(jid), jid, limit, page * limit),
            )
            chats = [self._chat_dict(row, False) for row in rows]
            return _dump_clipped({"ok": True, "chats": chats}, "chats")
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_whatsapp_last_interaction(self, jid: str) -> str:
        """Get the most recent message involving a contact.

        Args:
            jid: The contact's JID (``...@s.whatsapp.net``).

        Returns:
            JSON string with the most recent message, or an error.
        """
        try:
            rows = self._query(
                "SELECT m.id, m.chat_jid, c.name, m.sender, m.content, m.timestamp, "
                "m.is_from_me, m.media_type FROM messages m "
                "JOIN chats c ON m.chat_jid = c.jid "
                "WHERE m.sender IN (?, ?) OR c.jid = ? "
                "ORDER BY m.timestamp DESC, m.id DESC LIMIT 1",
                (*_sender_forms(jid), jid),
            )
            if not rows:
                return json.dumps({"ok": False, "error": f"No messages found for {jid}"})
            return json.dumps({"ok": True, "message": self._message_dict(rows[0])}, indent=2)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def _message_dict(self, row: tuple[Any, ...]) -> dict[str, Any]:
        """Convert a message row into a result dict.

        Args:
            row: (id, chat_jid, chat_name, sender, content, timestamp,
                is_from_me, media_type).

        Returns:
            Message dict for JSON serialization.
        """
        content = row[4]
        if isinstance(content, str) and len(content) > 2000:
            content = content[:2000] + "…[truncated]"
        return {
            "id": row[0],
            "chat_jid": row[1],
            "chat_name": row[2],
            "sender": row[3],
            "content": content,
            "timestamp": row[5],
            "is_from_me": bool(row[6]),
            "media_type": row[7],
        }

    _MESSAGE_SELECT = (
        "SELECT m.id, m.chat_jid, c.name, m.sender, m.content, m.timestamp, "
        "m.is_from_me, m.media_type FROM messages m JOIN chats c ON m.chat_jid = c.jid"
    )

    def list_whatsapp_messages(
        self,
        chat_jid: str = "",
        sender_phone_number: str = "",
        query: str = "",
        after: str = "",
        before: str = "",
        limit: int = 20,
        page: int = 0,
    ) -> str:
        """List WhatsApp messages matching the given filters, newest first.

        Args:
            chat_jid: Only messages in this chat JID.
            sender_phone_number: Only messages from this sender (digits or JID).
            query: Only messages whose text contains this term
                (case-insensitive).
            after: Only messages after this timestamp
                (``YYYY-MM-DD HH:MM:SS``; compared against the stored
                timestamp string).
            before: Only messages before this timestamp.
            limit: Maximum messages to return. Default: 20.
            page: Page number for pagination. Default: 0.

        Returns:
            JSON string with the matching messages. Media messages have
            empty content and a media_type; use
            download_whatsapp_media(message_id, chat_jid) to fetch the file.
        """
        try:
            where = []
            params: list[Any] = []
            if chat_jid:
                where.append("m.chat_jid = ?")
                params.append(chat_jid)
            if sender_phone_number:
                where.append("m.sender IN (?, ?)")
                params.extend(_sender_forms(sender_phone_number))
            if query:
                where.append("LOWER(m.content) LIKE LOWER(?)")
                params.append(f"%{query}%")
            for bound, op in ((after, ">"), (before, "<")):
                if not bound:
                    continue
                normalized, error = _parse_time_bound(bound)
                if error:
                    return json.dumps({"ok": False, "error": error})
                where.append(f"datetime(m.timestamp) {op} datetime(?)")
                params.append(normalized)
            where_sql = ("WHERE " + " AND ".join(where)) if where else ""
            rows = self._query(
                f"{self._MESSAGE_SELECT} {where_sql} "
                "ORDER BY m.timestamp DESC LIMIT ? OFFSET ?",
                (*params, limit, page * limit),
            )
            messages = [self._message_dict(row) for row in rows]
            return _dump_clipped({"ok": True, "messages": messages}, "messages")
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_whatsapp_message_context(
        self, message_id: str, before: int = 5, after: int = 5
    ) -> str:
        """Get the messages surrounding a specific message in its chat.

        Args:
            message_id: ID of the target message.
            before: Number of earlier messages to include. Default: 5.
            after: Number of later messages to include. Default: 5.

        Returns:
            JSON string with before/message/after message lists.
        """
        try:
            target = self._query(
                f"{self._MESSAGE_SELECT} WHERE m.id = ? LIMIT 1", (message_id,)
            )
            if not target:
                return json.dumps({"ok": False, "error": f"Message not found: {message_id}"})
            msg = self._message_dict(target[0])
            # Compound (timestamp, id) comparisons keep same-second
            # neighbours (whole-second history timestamps tie often).
            before_rows = self._query(
                f"{self._MESSAGE_SELECT} WHERE m.chat_jid = ? "
                "AND (m.timestamp < ? OR (m.timestamp = ? AND m.id < ?)) "
                "ORDER BY m.timestamp DESC, m.id DESC LIMIT ?",
                (msg["chat_jid"], msg["timestamp"], msg["timestamp"], msg["id"], before),
            )
            after_rows = self._query(
                f"{self._MESSAGE_SELECT} WHERE m.chat_jid = ? "
                "AND (m.timestamp > ? OR (m.timestamp = ? AND m.id > ?)) "
                "ORDER BY m.timestamp ASC, m.id ASC LIMIT ?",
                (msg["chat_jid"], msg["timestamp"], msg["timestamp"], msg["id"], after),
            )
            return _dump_clipped(
                {
                    "ok": True,
                    "before": [self._message_dict(r) for r in reversed(before_rows)],
                    "message": msg,
                    "after": [self._message_dict(r) for r in after_rows],
                },
                "before",
                "after",
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_whatsapp_message(self, recipient: str, message: str) -> str:
        """Send a WhatsApp text message to a person or group.

        Args:
            recipient: Phone number with country code (e.g. "+14155238886"
                or "14155238886") or a JID ("123456789@s.whatsapp.net";
                groups use "123456789@g.us").
            message: The message text to send.

        Returns:
            JSON string with ok status and the bridge's status message.
        """
        if not recipient.strip():
            return json.dumps({"ok": False, "error": "recipient is required"})
        ok, status = self._api_send(
            {"recipient": _rest_recipient(recipient), "message": message}
        )
        return json.dumps({"ok": ok, "message": status})

    def send_whatsapp_file(self, recipient: str, media_path: str) -> str:
        """Send a file (image, video, raw audio, document) via WhatsApp.

        Args:
            recipient: Phone number with country code or a JID
                (groups use "...@g.us").
            media_path: Absolute path of the file to send.

        Returns:
            JSON string with ok status and the bridge's status message.
        """
        if not recipient.strip():
            return json.dumps({"ok": False, "error": "recipient is required"})
        if not Path(media_path).is_file():
            return json.dumps({"ok": False, "error": f"File not found: {media_path}"})
        ok, status = self._api_send(
            {"recipient": _rest_recipient(recipient), "media_path": str(media_path)}
        )
        return json.dumps({"ok": ok, "message": status})

    def send_whatsapp_audio_message(self, recipient: str, media_path: str) -> str:
        """Send an audio file as a playable WhatsApp voice message.

        Non-``.ogg`` files are converted to Opus with ffmpeg first; if
        ffmpeg is unavailable, use send_whatsapp_file() instead (the audio
        then arrives as a plain file, not a voice note).

        Args:
            recipient: Phone number with country code or a JID.
            media_path: Absolute path of the audio file.

        Returns:
            JSON string with ok status and the bridge's status message.
        """
        if not recipient.strip():
            return json.dumps({"ok": False, "error": "recipient is required"})
        path = Path(media_path)
        if not path.is_file():
            return json.dumps({"ok": False, "error": f"File not found: {media_path}"})
        if path.suffix != ".ogg":
            fd, converted = tempfile.mkstemp(suffix=".ogg", prefix="kiss-whatsapp-voice.")
            os.close(fd)
            cmd = [
                "ffmpeg", "-i", str(path), "-c:a", "libopus", "-b:a", "32k",
                "-ar", "24000", "-application", "voip", "-vbr", "on",
                "-compression_level", "10", "-frame_duration", "60", "-y",
                converted,
            ]
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
            except (OSError, subprocess.SubprocessError) as e:
                Path(converted).unlink(missing_ok=True)
                return json.dumps(
                    {
                        "ok": False,
                        "error": f"ffmpeg conversion failed ({e}); "
                        "use send_whatsapp_file() to send the raw audio instead.",
                    }
                )
            path = Path(converted)
        try:
            ok, status = self._api_send(
                {"recipient": _rest_recipient(recipient), "media_path": str(path)}
            )
        finally:
            if path != Path(media_path):
                path.unlink(missing_ok=True)
        return json.dumps({"ok": ok, "message": status})

    def download_whatsapp_media(self, message_id: str, chat_jid: str) -> str:
        """Download the media of a WhatsApp message to a local file.

        Args:
            message_id: ID of the message containing media (from
                list_whatsapp_messages).
            chat_jid: JID of the chat containing the message.

        Returns:
            JSON string with ok status and the local file path.
        """
        try:
            resp = requests.post(
                self._api_url("download"),
                json={"message_id": message_id, "chat_jid": chat_jid},
                timeout=120,
            )
            result = resp.json()
            return json.dumps(
                {
                    "ok": bool(result.get("success", False)),
                    "message": result.get("message", ""),
                    "file_path": result.get("path", ""),
                    "filename": result.get("filename", ""),
                }
            )
        except (requests.RequestException, ValueError) as e:
            return json.dumps({"ok": False, "error": str(e)})


# ----------------------------------------------------------------------
# Bridge process management and QR pairing
# ----------------------------------------------------------------------


# The whatsmeow version pinned by upstream whatsapp-mcp (Mar 2025) is now
# rejected by WhatsApp servers ("Client outdated (405)"), so the bridge is
# built against the latest whatsmeow.  Newer whatsmeow added a
# context.Context first argument to these calls in the bridge's main.go;
# the plain-string rewrites below adapt it, plus one security fix (each
# rewrite is a no-op once applied).
_BRIDGE_SOURCE_FIXES = (
    (
        "client.Download(downloader)",
        "client.Download(context.Background(), downloader)",
    ),
    (
        'sqlstore.New("sqlite3"',
        'sqlstore.New(context.Background(), "sqlite3"',
    ),
    (
        "container.GetFirstDevice()",
        "container.GetFirstDevice(context.Background())",
    ),
    (
        "client.GetGroupInfo(jid)",
        "client.GetGroupInfo(context.Background(), jid)",
    ),
    (
        "client.Store.Contacts.GetContact(jid)",
        "client.Store.Contacts.GetContact(context.Background(), jid)",
    ),
    # Security hardening: upstream binds its unauthenticated REST API to
    # every interface (":8080"); restrict it to loopback.
    (
        'fmt.Sprintf(":%d", port)',
        'fmt.Sprintf("127.0.0.1:%d", port)',
    ),
)


def _modernize_bridge_source(bridge_dir: Path) -> str:
    """Upgrade the bridge's whatsmeow dependency and adapt its source.

    Runs ``go get go.mau.fi/whatsmeow@latest`` and ``go mod tidy``, then
    applies the mechanical context-argument rewrites the newer whatsmeow
    API requires (see ``_BRIDGE_SOURCE_FIXES``).

    Args:
        bridge_dir: The ``whatsapp-bridge`` directory of the clone.

    Returns:
        "" on success, or an error message.
    """
    for cmd in (["go", "get", "go.mau.fi/whatsmeow@latest"], ["go", "mod", "tidy"]):
        result = subprocess.run(
            cmd, cwd=str(bridge_dir), capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            return f"{' '.join(cmd)} failed: {result.stderr[-1000:]}"
    main_go = bridge_dir / "main.go"
    src = main_go.read_text(encoding="utf-8")
    for old, new in _BRIDGE_SOURCE_FIXES:
        src = src.replace(old, new)
    main_go.write_text(src, encoding="utf-8")
    return ""


def _bridge_log_path() -> Path:
    """Return the bridge stdout/stderr log path."""
    return _channel_dir() / "bridge.log"


def _bridge_pid_path() -> Path:
    """Return the bridge PID file path."""
    return _channel_dir() / "bridge.pid"


def _qr_html_path() -> Path:
    """Return the QR pairing page path."""
    return _channel_dir() / "qr.html"


def _bridge_pid() -> int:
    """Return the recorded bridge PID, or 0 if none/invalid."""
    try:
        return int(_bridge_pid_path().read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0


def _pid_alive(pid: int) -> bool:
    """Return True if *pid* refers to a live process."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _is_qr_line(line: str) -> bool:
    """Return True if *line* looks like a qrterminal half-block QR row.

    qrterminal's ``GenerateHalfBlock`` renders QR-dark modules as spaces
    and QR-light modules as ``█``/``▀``/``▄``, so a QR row contains only
    those four characters.
    """
    line = line.rstrip("\r\n")
    return len(line) >= 20 and bool(set(line) & set("█▀▄")) and set(line) <= _QR_LINE_CHARS


def _extract_last_qr(log_text: str) -> str:
    """Extract the most recent QR code block from bridge log text.

    Args:
        log_text: Full text of the bridge log.

    Returns:
        The QR block (newline-joined half-block rows), or "" if none.
    """
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in log_text.splitlines():
        if _is_qr_line(line):
            current.append(line.rstrip("\r\n"))
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    for block in reversed(blocks):
        if len(block) >= 10:
            return "\n".join(block)
    return ""


def _write_qr_html(qr_text: str) -> Path:
    """Write the QR pairing page and return its path.

    The half-block QR maps light modules to block glyphs and dark modules
    to spaces, so the page uses white glyphs on a black background — the
    correct polarity for phone cameras.

    Args:
        qr_text: The captured half-block QR block.

    Returns:
        Path of the written HTML file.
    """
    from html import escape

    path = _qr_html_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta http-equiv='refresh' content='4'>"
        "<title>Link WhatsApp</title></head>"
        "<body style='background:#000;color:#fff;font-family:sans-serif;"
        "text-align:center;padding-top:24px'>"
        "<h2>Link this computer to WhatsApp</h2>"
        "<p>On your phone: WhatsApp &rarr; Settings &rarr; Linked devices "
        "&rarr; Link a device &mdash; then scan this code:</p>"
        "<pre style=\"font-family:'DejaVu Sans Mono','Menlo','Consolas',"
        "monospace;font-size:12px;line-height:1;letter-spacing:0;"
        "display:inline-block;background:#000;color:#fff\">"
        f"{escape(qr_text)}</pre>"
        "<p>This page reloads itself; when the code expires a fresh one "
        "replaces it. After scanning, the page reports success within a "
        "few seconds.</p></body></html>",
        encoding="utf-8",
    )
    return path


def _write_paired_html() -> None:
    """Overwrite the QR page with a success message (shown after pairing)."""
    path = _qr_html_path()
    if not path.exists():
        return
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>WhatsApp linked</title></head>"
        "<body style='background:#000;color:#fff;font-family:sans-serif;"
        "text-align:center;padding-top:24px'><h2>&#10003; WhatsApp linked "
        "successfully</h2><p>You can close this page.</p></body></html>",
        encoding="utf-8",
    )


class WhatsAppAgent(BaseChannelAgent):
    """Channel agent for a personal WhatsApp account (QR-paired bridge).

    Tasks run on the kiss-web daemon's agent (which supplies bash, file
    editing, and browser automation) with tools for searching contacts,
    reading and searching the locally synced message history, sending
    text/files/voice messages, and downloading received media — all
    through the whatsapp-mcp Go bridge that pairs with the user's phone
    via a QR code.

    If the bridge is not yet set up, the authentication tools clone and
    build it, start it, render the pairing QR code into a local HTML page,
    and wait for the user to scan it.

    Example::

        agent = WhatsAppAgent()
        result = agent.run(
            prompt_template="Send 'Hello!' to +14155238886",
        )
    """

    channel_system_prompt = (
        "\n\n## WhatsApp Pairing\n"
        "WhatsApp pairing flow (only when check_whatsapp_auth() reports "
        "not paired): call authenticate_whatsapp() to clone and build the "
        "bridge, then start_whatsapp_bridge(). If it reports a QR page, "
        "call show_browser(), open the page with go_to_url('file://...'), "
        "ask the user to scan the QR code with their phone (WhatsApp -> "
        "Settings -> Linked devices -> Link a device), and call "
        "wait_for_whatsapp_pairing() until it reports success. Message "
        "history syncs for a few minutes after first pairing."
    )

    def __init__(self) -> None:
        super().__init__("WhatsApp Agent")
        self._backend = WhatsAppChannelBackend()
        _apply_config(self._backend)

    def _is_authenticated(self) -> bool:
        """Return True if a WhatsApp device session exists (QR pairing done)."""
        return bool(self._backend._is_paired())

    def _get_auth_tools(self) -> list:
        """Return WhatsApp bridge setup, pairing, and lifecycle tool functions."""
        agent = self

        def check_whatsapp_auth() -> str:
            """Check whether WhatsApp is paired and the bridge is running.

            Reports the whatsapp-mcp clone, the bridge build, the bridge
            process, and the QR pairing state, with the next step to take.

            Returns:
                JSON status report with a next_step instruction.
            """
            backend = agent._backend
            binary = backend.bridge_dir / _BRIDGE_BINARY_NAME
            status = {
                "repo_dir": str(backend.repo_dir),
                "repo_cloned": (backend.repo_dir / "whatsapp-bridge" / "main.go").exists(),
                "bridge_built": binary.exists(),
                "bridge_running": backend._bridge_running(),
                "paired": backend._is_paired(),
                "messages_synced": backend.messages_db.exists(),
                "go_installed": shutil.which("go") is not None,
            }
            if not status["repo_cloned"] or not status["bridge_built"]:
                status["next_step"] = (
                    "Call authenticate_whatsapp() to clone and build the "
                    "whatsapp-mcp bridge (requires git and Go >= 1.24 with "
                    "gcc for CGO)."
                )
            elif not status["bridge_running"]:
                status["next_step"] = "Call start_whatsapp_bridge()."
            elif not status["paired"]:
                status["next_step"] = (
                    "Call get_whatsapp_qr_code(), open the returned QR page "
                    "in the browser for the user to scan with their phone, "
                    "then wait_for_whatsapp_pairing()."
                )
            else:
                status["next_step"] = "Ready. Use the whatsapp_* messaging tools."
            return json.dumps(status, indent=2)

        def authenticate_whatsapp(
            repo_dir: str = "", bridge_port: str = "", rebuild: bool = False
        ) -> str:
            """Set up the WhatsApp bridge: clone whatsapp-mcp and build it.

            Clones https://github.com/lharries/whatsapp-mcp (reusing an
            existing ~/.kiss/connectors/whatsapp-mcp clone when present),
            upgrades its whatsmeow dependency to the latest release
            (WhatsApp rejects the upstream pin as "Client outdated"),
            builds the Go bridge (CGO enabled), and saves the config.
            Pairing itself happens afterwards via start_whatsapp_bridge()
            and the QR code.

            Args:
                repo_dir: Optional custom path for the whatsapp-mcp clone.
                bridge_port: Optional REST port of a patched bridge
                    (the upstream bridge always uses 8080).
                rebuild: Force a whatsmeow upgrade and rebuild even when a
                    bridge binary exists (use when the bridge log reports
                    "Client outdated").

            Returns:
                JSON string with the setup result and the next step.
            """
            backend = agent._backend
            if repo_dir.strip():
                backend._repo_dir = repo_dir.strip()
            if bridge_port.strip():
                try:
                    backend._bridge_port = int(bridge_port)
                except ValueError:
                    return json.dumps(
                        {"ok": False, "error": f"Invalid bridge_port: {bridge_port}"}
                    )
            repo = backend.repo_dir
            if not (repo / "whatsapp-bridge" / "main.go").exists():
                if shutil.which("git") is None:
                    return json.dumps({"ok": False, "error": "git is not installed."})
                try:
                    repo.parent.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    return json.dumps(
                        {"ok": False, "error": f"Cannot create {repo.parent}: {e}"}
                    )
                clone = subprocess.run(
                    ["git", "clone", "--depth", "1", _BRIDGE_REPO_URL, str(repo)],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if clone.returncode != 0:
                    return json.dumps(
                        {"ok": False, "error": f"git clone failed: {clone.stderr[-1000:]}"}
                    )
            binary = backend.bridge_dir / _BRIDGE_BINARY_NAME
            if rebuild:
                binary.unlink(missing_ok=True)
            if not binary.exists():
                if shutil.which("go") is None:
                    return json.dumps(
                        {
                            "ok": False,
                            "error": "Go is not installed. Install the latest Go "
                            "(https://go.dev/doc/install; macOS: brew install go) "
                            "and gcc (the bridge uses go-sqlite3, a CGO package), "
                            "then call authenticate_whatsapp() again.",
                        }
                    )
                error = _modernize_bridge_source(backend.bridge_dir)
                if error:
                    return json.dumps({"ok": False, "error": error})
                build = subprocess.run(
                    ["go", "build", "-o", _BRIDGE_BINARY_NAME, "."],
                    cwd=str(backend.bridge_dir),
                    capture_output=True,
                    text=True,
                    timeout=900,
                    env={**os.environ, "CGO_ENABLED": "1"},
                )
                if build.returncode != 0:
                    return json.dumps(
                        {"ok": False, "error": f"go build failed: {build.stderr[-2000:]}"}
                    )
            _config.save(
                {
                    "repo_dir": str(repo),
                    "bridge_port": str(backend._bridge_port),
                }
            )
            return json.dumps(
                {
                    "ok": True,
                    "message": "WhatsApp bridge is cloned and built.",
                    "next_step": "Call start_whatsapp_bridge().",
                }
            )

        def start_whatsapp_bridge() -> str:
            """Start the WhatsApp bridge process (detached, survives this task).

            If the device is already paired the bridge simply reconnects;
            otherwise the bridge prints a pairing QR code, which is
            rendered into a local HTML page for the user to scan.

            Returns:
                JSON string with the bridge status; when pairing is needed
                it includes qr_page (a file path to open in the browser).
            """
            backend = agent._backend
            if backend._bridge_running():
                return json.dumps({"ok": True, "message": "Bridge already running."})
            pid = _bridge_pid()
            if _pid_alive(pid):
                # The bridge opens its REST port only after pairing, so a
                # live PID without a REST answer means pairing is pending.
                return json.dumps(
                    {
                        "ok": True,
                        "message": f"Bridge (pid {pid}) is already starting or waiting "
                        "for its QR scan. Call get_whatsapp_qr_code() and "
                        "wait_for_whatsapp_pairing() instead of starting it again.",
                    }
                )
            binary = backend.bridge_dir / _BRIDGE_BINARY_NAME
            if not binary.exists():
                return json.dumps(
                    {
                        "ok": False,
                        "error": "Bridge not built. Call authenticate_whatsapp() first.",
                    }
                )
            log_path = _bridge_log_path()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("", encoding="utf-8")
            with open(log_path, "ab") as log_fp:
                proc = subprocess.Popen(
                    [str(binary)],
                    cwd=str(backend.bridge_dir),
                    stdout=log_fp,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            _bridge_pid_path().write_text(str(proc.pid), encoding="utf-8")
            deadline = time.time() + 60
            while time.time() < deadline:
                time.sleep(1)
                text = log_path.read_text(encoding="utf-8", errors="replace")
                if _CONNECTED_MARKER in text or _PAIRED_MARKER in text:
                    return json.dumps(
                        {"ok": True, "message": "Bridge started and connected to WhatsApp."}
                    )
                qr = _extract_last_qr(text)
                if qr:
                    page = _write_qr_html(qr)
                    return json.dumps(
                        {
                            "ok": True,
                            "pairing_needed": True,
                            "qr_page": str(page),
                            "message": "Pairing needed. Call show_browser(), open "
                            f"go_to_url('file://{page}'), ask the user to scan the "
                            "QR code with WhatsApp on their phone (Settings -> "
                            "Linked devices -> Link a device), then call "
                            "wait_for_whatsapp_pairing().",
                        }
                    )
                if "Client outdated" in text:
                    with contextlib.suppress(OSError):
                        os.killpg(proc.pid, signal.SIGTERM)
                    _bridge_pid_path().unlink(missing_ok=True)
                    return json.dumps(
                        {
                            "ok": False,
                            "error": "WhatsApp rejected the bridge as outdated. Call "
                            "authenticate_whatsapp(rebuild=True) to upgrade whatsmeow "
                            "and rebuild, then start_whatsapp_bridge() again.",
                        }
                    )
                if proc.poll() is not None:
                    _bridge_pid_path().unlink(missing_ok=True)
                    return json.dumps(
                        {
                            "ok": False,
                            "error": f"Bridge exited (code {proc.returncode}). "
                            f"Log tail: {text[-2000:]}",
                        }
                    )
            return json.dumps(
                {
                    "ok": False,
                    "error": "Bridge did not report a connection or QR code within "
                    f"60s. Log tail: "
                    f"{log_path.read_text(encoding='utf-8', errors='replace')[-2000:]}",
                }
            )

        def get_whatsapp_qr_code() -> str:
            """Refresh the QR pairing page from the latest bridge output.

            The bridge rotates the QR code periodically; this re-extracts
            the newest code from the bridge log and rewrites the HTML page.

            Returns:
                JSON string with the qr_page path, or an error.
            """
            log_path = _bridge_log_path()
            if not log_path.exists():
                return json.dumps(
                    {"ok": False, "error": "No bridge log. Call start_whatsapp_bridge() first."}
                )
            text = log_path.read_text(encoding="utf-8", errors="replace")
            if _PAIRED_MARKER in text or _CONNECTED_MARKER in text:
                _write_paired_html()
                return json.dumps({"ok": True, "message": "Already paired and connected."})
            qr = _extract_last_qr(text)
            if not qr:
                return json.dumps(
                    {
                        "ok": False,
                        "error": "No QR code in the bridge log (yet). "
                        f"Log tail: {text[-1000:]}",
                    }
                )
            page = _write_qr_html(qr)
            return json.dumps(
                {
                    "ok": True,
                    "qr_page": str(page),
                    "message": f"Open file://{page} in the browser and ask the user "
                    "to scan it with WhatsApp on their phone.",
                }
            )

        def wait_for_whatsapp_pairing(timeout: int = 120) -> str:
            """Wait for the user to scan the QR code and complete pairing.

            Polls the bridge log, refreshing the QR page whenever the
            bridge rotates the code, until pairing succeeds or *timeout*
            elapses (call again to keep waiting — the bridge itself gives
            up after ~3 minutes and must then be restarted).

            Args:
                timeout: Maximum seconds to wait. Default: 120.

            Returns:
                JSON string with the pairing result.
            """
            log_path = _bridge_log_path()
            if not log_path.exists():
                return json.dumps(
                    {"ok": False, "error": "No bridge log. Call start_whatsapp_bridge() first."}
                )
            last_qr = ""
            deadline = time.time() + max(timeout, 1)
            while True:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                if _PAIRED_MARKER in text or _CONNECTED_MARKER in text:
                    _write_paired_html()
                    return json.dumps(
                        {"ok": True, "message": "WhatsApp paired and connected. Message "
                         "history now syncs in the background (takes a few minutes)."}
                    )
                if _QR_TIMEOUT_MARKER in text:
                    return json.dumps(
                        {
                            "ok": False,
                            "error": "The bridge timed out waiting for the QR scan. "
                            "Call start_whatsapp_bridge() to get a fresh QR code.",
                        }
                    )
                if not _pid_alive(_bridge_pid()):
                    return json.dumps(
                        {
                            "ok": False,
                            "error": "The bridge process is no longer running. Call "
                            f"start_whatsapp_bridge() again. Log tail: {text[-1000:]}",
                        }
                    )
                qr = _extract_last_qr(text)
                if qr and qr != last_qr:
                    _write_qr_html(qr)
                    last_qr = qr
                if time.time() >= deadline:
                    return json.dumps(
                        {
                            "ok": False,
                            "error": "Still waiting for the QR scan. Call "
                            "wait_for_whatsapp_pairing() again, or "
                            "get_whatsapp_qr_code() to refresh the page.",
                        }
                    )
                time.sleep(2)

        def stop_whatsapp_bridge() -> str:
            """Stop the WhatsApp bridge process.

            Note that the bridge only syncs messages while it runs; leave
            it running if the channel poller monitors WhatsApp.

            Returns:
                JSON string with the stop result.
            """
            pid = _bridge_pid()
            if pid <= 0:
                return json.dumps({"ok": False, "error": "No recorded bridge PID."})
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                _bridge_pid_path().unlink(missing_ok=True)
                return json.dumps({"ok": True, "message": "Bridge was not running."})
            except PermissionError as e:
                return json.dumps({"ok": False, "error": f"Could not stop bridge: {e}"})
            _bridge_pid_path().unlink(missing_ok=True)
            return json.dumps({"ok": True, "message": f"Bridge (pid {pid}) stopped."})

        def clear_whatsapp_auth() -> str:
            """Unpair WhatsApp: stop the bridge and delete the local session.

            Deletes the session and message databases
            (``whatsapp-bridge/store/``) and the saved config. Also remove
            the linked device on the phone (WhatsApp -> Settings -> Linked
            devices).

            Returns:
                Status message.
            """
            backend = agent._backend
            pid = _bridge_pid()
            if backend._bridge_running() and not _pid_alive(pid):
                return (
                    "Refusing to clear: a WhatsApp bridge is running on port "
                    f"{backend._bridge_port} that this agent did not start (e.g. the "
                    "connectors bridge). Stop it first, then call "
                    "clear_whatsapp_auth() again."
                )
            if pid > 0:
                with contextlib.suppress(OSError):
                    os.killpg(pid, signal.SIGTERM)
                deadline = time.time() + 5
                while _pid_alive(pid) and time.time() < deadline:
                    time.sleep(0.2)
                _bridge_pid_path().unlink(missing_ok=True)
            store = backend.bridge_dir / "store"
            try:
                if store.exists():
                    shutil.rmtree(store)
            except OSError as e:
                return f"Could not fully delete {store}: {e}"
            _qr_html_path().unlink(missing_ok=True)
            _config.clear()
            return (
                "WhatsApp session cleared. Also remove this device on the phone: "
                "WhatsApp -> Settings -> Linked devices."
            )

        return [
            check_whatsapp_auth,
            authenticate_whatsapp,
            start_whatsapp_bridge,
            get_whatsapp_qr_code,
            wait_for_whatsapp_pairing,
            stop_whatsapp_bridge,
            clear_whatsapp_auth,
        ]


def _make_backend() -> WhatsAppChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = WhatsAppChannelBackend()
    _apply_config(backend)
    if not backend._is_paired():
        print("Not paired. Run: kiss-whatsapp -t 'authenticate whatsapp'")
        sys.exit(1)
    return backend


def main() -> None:  # pragma: no cover – CLI entry point requires API
    """Run the WhatsAppAgent from the command line with chat persistence."""
    channel_main(
        WhatsAppAgent,
        "kiss-whatsapp",
        channel_name="WhatsApp",
        make_backend=_make_backend,
    )


def tools() -> list:
    """Return the WhatsApp channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    bridge state persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return WhatsAppAgent()._get_tools()


if __name__ == "__main__":
    main()
