# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Email Agent — channel agent for any IMAP/SMTP mailbox (pure stdlib).

Provides access to a generic email account: unread mail is polled over
IMAP4_SSL and replies are sent over SMTP (implicit SSL or STARTTLS),
Hermes-style.  Automated mail (no-reply senders, ``Auto-Submitted``,
``Precedence: bulk/junk/list``, mailing lists) is dropped from the
channel loop so the agent only answers real people.  Stores config in
``~/.kiss/third_party_agents/email/config.json``; the password is
typically an app password.

Usage::

    agent = EmailAgent()
    agent.run(prompt_template="List my unread emails")
"""

from __future__ import annotations

import email
import html
import imaplib
import json
import logging
import re
import smtplib
import sys
import threading
from email.header import decode_header, make_header
from email.message import EmailMessage, Message
from email.utils import formatdate, make_msgid, parseaddr, parsedate_to_datetime
from pathlib import Path
from typing import Any

from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

logger = logging.getLogger(__name__)

_EMAIL_DIR = Path.home() / ".kiss" / "third_party_agents" / "email"
_config = ChannelConfig(_EMAIL_DIR, ("imap_host", "smtp_host", "email_address", "password"))

_AUTOMATED_FROM_MARKERS = ("noreply", "no-reply", "donotreply", "mailer-daemon")
_AUTOMATED_PRECEDENCE = ("bulk", "junk", "list")


def _decode_header_value(value: str) -> str:
    """Decode an RFC 2047 encoded header value to a readable string."""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


def _decode_payload(part: Message) -> str:
    """Decode one non-multipart MIME part's payload to text."""
    payload = part.get_payload(decode=True)
    if not isinstance(payload, bytes):
        return str(part.get_payload())
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _html_to_text(html_text: str) -> str:
    """Strip HTML tags from markup, returning readable plain text."""
    text = re.sub(r"(?is)<(script|style)\b.*?</\1\s*>", " ", html_text)
    text = re.sub(r"(?i)<br\s*/?>|</p\s*>|</div\s*>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ([,.;:!?])", r"\1", text)
    return re.sub(r" ?\n ?", "\n", text).strip()


def _message_body_text(msg: Message) -> str:
    """Extract the plain-text body from a parsed email message.

    Prefers the ``text/plain`` part; when the mail is HTML-only, the
    ``text/html`` part is stripped of tags as a fallback.
    """
    if msg.is_multipart():
        html_body = ""
        for part in msg.walk():
            if "attachment" in str(part.get("Content-Disposition", "")).lower():
                continue
            content_type = part.get_content_type()
            if content_type == "text/plain":
                return _decode_payload(part).strip()
            if content_type == "text/html" and not html_body:
                html_body = _decode_payload(part)
        return _html_to_text(html_body) if html_body else ""
    if msg.get_content_type() == "text/html":
        return _html_to_text(_decode_payload(msg))
    return _decode_payload(msg).strip()


def _is_automated_mail(msg: Message) -> bool:
    """Return True for automated mail the channel loop must drop.

    A message is automated when its From address contains a no-reply
    marker, it carries ``Auto-Submitted`` other than ``no``, its
    ``Precedence`` is bulk/junk/list, or it has a ``List-Id`` header.
    """
    from_addr = parseaddr(str(msg.get("From", "")))[1].lower()
    if any(marker in from_addr for marker in _AUTOMATED_FROM_MARKERS):
        return True
    auto_submitted = str(msg.get("Auto-Submitted", "")).strip().lower()
    if auto_submitted and auto_submitted != "no":
        return True
    if str(msg.get("Precedence", "")).strip().lower() in _AUTOMATED_PRECEDENCE:
        return True
    return msg.get("List-Id") is not None


def _normalize_mail(msg: Message, mailbox: str) -> dict[str, str]:
    """Normalize a parsed email message to the channel message shape.

    Args:
        msg: Parsed ``email.message.Message``.
        mailbox: IMAP mailbox the message came from.

    Returns:
        Dict with ``ts``, ``user`` (from address), ``username`` (from
        display name), ``text`` (subject + plain-text body),
        ``channel_id`` (mailbox), ``thread_ts`` (Message-ID),
        ``subject``, and ``reply_to`` (Reply-To address, may be empty).
    """
    username, from_addr = parseaddr(str(msg.get("From", "")))
    reply_to = parseaddr(str(msg.get("Reply-To", "")))[1]
    subject = _decode_header_value(str(msg.get("Subject", ""))).strip()
    ts = ""
    try:
        date = parsedate_to_datetime(str(msg.get("Date", "")))
        if date is not None:
            ts = str(date.timestamp())
    except Exception:
        ts = ""
    return {
        "ts": ts,
        "user": from_addr,
        "username": _decode_header_value(username),
        "text": f"Subject: {subject}\n\n{_message_body_text(msg)}",
        "channel_id": mailbox,
        "thread_ts": str(msg.get("Message-ID", "")).strip(),
        "subject": subject,
        "reply_to": reply_to,
    }


def _quote_mailbox(mailbox: str) -> str:
    """Quote a mailbox name for IMAP commands when it contains a space."""
    if " " in mailbox and not (mailbox.startswith('"') and mailbox.endswith('"')):
        return f'"{mailbox}"'
    return mailbox


def _truncate_text(text: str, limit: int) -> str:
    """Truncate text to *limit* characters, marking the cut."""
    if len(text) <= limit:
        return text
    return text[:limit] + "... [truncated]"


def _store_result(status: str) -> str:
    """JSON result for an IMAP STORE command status."""
    if status != "OK":
        return json.dumps({"ok": False, "error": f"STORE failed with status {status!r}."})
    return json.dumps({"ok": True})


def _angle(message_id: str) -> str:
    """Wrap a Message-ID in angle brackets if missing."""
    message_id = message_id.strip()
    if message_id and not message_id.startswith("<"):
        return f"<{message_id}>"
    return message_id


def _build_outbound(
    from_addr: str, to_addr: str, subject: str, body: str, in_reply_to: str = ""
) -> EmailMessage:
    """Build an outbound MIME message, threading it when replying.

    Args:
        from_addr: Sender address.
        to_addr: Recipient address.
        subject: Subject line; prefixed with ``Re: `` when replying
            (unless already present).
        body: Plain-text body.
        in_reply_to: Message-ID being replied to; sets ``In-Reply-To``
            and ``References``.

    Returns:
        A ready-to-send ``EmailMessage`` with ``Date`` (RFC 5322
        requires it) and a fresh ``Message-ID`` always set.
    """
    msg = EmailMessage()
    if in_reply_to:
        message_id = _angle(in_reply_to)
        msg["In-Reply-To"] = message_id
        msg["References"] = message_id
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid()
    msg.set_content(body)
    return msg


class EmailChannelBackend(ToolMethodBackend):
    """Channel backend for a generic IMAP/SMTP email account.

    Polls UNSEEN mail over IMAP4_SSL and sends threaded replies over
    SMTP (implicit SSL or STARTTLS).  Pure stdlib.
    """

    def __init__(self) -> None:
        self._cfg: dict[str, str] = {}
        self._threads: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Load the stored email config."""
        cfg = _config.load()
        if not cfg:  # pragma: no branch
            self._connection_info = "No email config found."
            return False
        self._cfg = cfg
        self._connection_info = f"Email configured for {cfg['email_address']}"
        return True

    def _imap_login(self) -> imaplib.IMAP4_SSL:
        """Open and authenticate an IMAP4_SSL connection."""
        port = int(self._cfg.get("imap_port") or "993")
        imap = imaplib.IMAP4_SSL(self._cfg["imap_host"], port, timeout=30)
        imap.login(self._cfg["email_address"], self._cfg["password"])
        return imap

    def _imap_logout(self, imap: imaplib.IMAP4_SSL) -> None:
        """Best-effort IMAP logout."""
        try:
            imap.logout()
        except Exception:
            logger.debug("IMAP logout failed", exc_info=True)

    def _smtp_send(self, msg: EmailMessage) -> None:
        """Send a MIME message over SMTP per the configured security mode."""
        host = self._cfg["smtp_host"]
        port = int(self._cfg.get("smtp_port") or "465")
        security = (self._cfg.get("smtp_security") or "ssl").lower()
        if security == "starttls":
            with smtplib.SMTP(host, port, timeout=30) as smtp:
                smtp.starttls()
                smtp.login(self._cfg["email_address"], self._cfg["password"])
                smtp.send_message(msg)
        else:
            with smtplib.SMTP_SSL(host, port, timeout=30) as smtp:
                smtp.login(self._cfg["email_address"], self._cfg["password"])
                smtp.send_message(msg)

    def _find_by_message_id(self, imap: imaplib.IMAP4_SSL, message_id: str) -> list[str]:
        """Search the selected mailbox for a Message-ID, returning sequence numbers."""
        status, data = imap.search(None, f'HEADER Message-ID "{_angle(message_id)}"')
        if status != "OK" or not data or not isinstance(data[0], bytes):
            return []
        return data[0].decode("ascii").split()

    def _fetch_message(
        self, imap: imaplib.IMAP4_SSL, num: str, peek: bool = False
    ) -> Message | None:
        """Fetch one message by sequence number, optionally without marking it read."""
        parts = "(BODY.PEEK[])" if peek else "(RFC822)"
        status, fetched = imap.fetch(num, parts)
        if status != "OK" or not fetched or not isinstance(fetched[0], tuple):
            return None
        return email.message_from_bytes(fetched[0][1])

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll UNSEEN mail in a mailbox (default INBOX), dropping automated mail.

        Fetches with ``BODY.PEEK[]`` so polling does NOT mark mail
        ``\\Seen``; messages are marked read explicitly (e.g. via
        :meth:`mark_email_read`) only when appropriate, so no mail is
        lost if the process dies mid-task.
        """
        mailbox = channel_id or "INBOX"
        try:
            with self._lock:
                imap = self._imap_login()
                try:
                    imap.select(_quote_mailbox(mailbox))
                    status, data = imap.search(None, "UNSEEN")
                    nums = (
                        data[0].decode("ascii").split()
                        if status == "OK" and data and data[0]
                        else []
                    )
                    messages: list[dict[str, Any]] = []
                    for num in nums:
                        if len(messages) >= limit:
                            break
                        msg = self._fetch_message(imap, num, peek=True)
                        if msg is None or _is_automated_mail(msg):
                            continue
                        normalized = _normalize_mail(msg, mailbox)
                        self._cache_thread(normalized)
                        messages.append(normalized)
                    return messages, oldest
                finally:
                    self._imap_logout(imap)
        except Exception:
            logger.warning("Email poll failed", exc_info=True)
            return [], oldest

    def _cache_thread(self, normalized: dict[str, str]) -> None:
        """Cache the reply route for a polled message.

        Prefers the ``Reply-To`` address over ``From`` for the
        outbound recipient (standard email semantics).  When the mail
        has no Message-ID, ``thread_ts`` is set to ``ts`` and the
        cache is keyed by it so a threaded reply still resolves the
        sender.
        """
        sender = normalized.get("reply_to") or normalized["user"]
        if not normalized["thread_ts"]:
            normalized["thread_ts"] = normalized["ts"]
        if normalized["thread_ts"]:
            self._threads[normalized["thread_ts"]] = {
                "subject": normalized["subject"],
                "from": sender,
            }

    def _resolve_outbound(self, channel_id: str, thread_ts: str) -> tuple[str, str]:
        """Resolve the recipient and subject for :meth:`send_message`.

        ``ChannelRunner`` passes the polled mailbox as ``channel_id``;
        for a threaded reply the real recipient is the cached sender of
        the mail being replied to, so a ``channel_id`` that is not an
        email address falls back to that cached sender.

        Returns:
            ``(recipient, subject)`` tuple.
        """
        subject = "Message from KISS Email Agent"
        recipient = channel_id
        if thread_ts:
            thread = self._threads.get(thread_ts, {})
            subject = thread.get("subject", "") or "your message"
            if "@" not in recipient:
                recipient = thread.get("from", "") or recipient
        return recipient, subject

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send an email to *channel_id* (a recipient address).

        When ``thread_ts`` (the Message-ID being replied to) is given,
        the mail is threaded via ``In-Reply-To``/``References`` and the
        subject becomes ``Re: <original subject>``; if ``channel_id``
        is a mailbox name rather than an address (as in channel poll
        mode), the reply goes to the original sender.

        Raises:
            RuntimeError: If no valid recipient email address can be
                resolved (prevents sending to a bogus recipient such
                as a mailbox name).
        """
        recipient, subject = self._resolve_outbound(channel_id, thread_ts)
        if "@" not in recipient:
            raise RuntimeError(
                f"Cannot send email: no valid recipient address resolved for "
                f"channel {channel_id!r}, thread {thread_ts!r} (got {recipient!r})."
            )
        in_reply_to = thread_ts if "@" in thread_ts else ""
        msg = _build_outbound(
            self._cfg["email_address"], recipient, subject, text, in_reply_to=in_reply_to
        )
        with self._lock:
            self._smtp_send(msg)

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Return True if a polled message was sent from the agent's own address."""
        own = self._cfg.get("email_address", "").lower()
        return bool(own) and str(msg.get("user", "")).lower() == own

    def send_email(self, to: str, subject: str, body: str, in_reply_to: str = "") -> str:
        """Send an email over SMTP.

        Args:
            to: Recipient email address.
            subject: Subject line.
            body: Plain-text message body.
            in_reply_to: Optional Message-ID being replied to; threads
                the mail and prefixes the subject with "Re: ".

        Returns:
            JSON string with ok status.
        """
        try:
            msg = _build_outbound(
                self._cfg["email_address"], to, subject, body, in_reply_to=in_reply_to
            )
            with self._lock:
                self._smtp_send(msg)
            return json.dumps({"ok": True, "to": to, "subject": str(msg["Subject"])})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_unread_emails(self, mailbox: str = "INBOX", limit: int = 10) -> str:
        """List unread emails in a mailbox without marking them read.

        Args:
            mailbox: IMAP mailbox name. Default: "INBOX".
            limit: Maximum number of emails to return.

        Returns:
            JSON string with a list of emails (ts, user, username,
            subject, text, thread_ts = Message-ID, automated flag).
        """
        try:
            with self._lock:
                imap = self._imap_login()
                try:
                    imap.select(_quote_mailbox(mailbox))
                    status, data = imap.search(None, "UNSEEN")
                    nums = (
                        data[0].decode("ascii").split()
                        if status == "OK" and data and data[0]
                        else []
                    )
                    emails: list[dict[str, Any]] = []
                    for num in nums[:limit]:
                        msg = self._fetch_message(imap, num, peek=True)
                        if msg is None:
                            continue
                        normalized: dict[str, Any] = _normalize_mail(msg, mailbox)
                        normalized["text"] = _truncate_text(normalized["text"], 500)
                        normalized["automated"] = _is_automated_mail(msg)
                        emails.append(normalized)
                    return json.dumps({"ok": True, "emails": emails}, indent=2)
                finally:
                    self._imap_logout(imap)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def read_email(self, message_id: str, mailbox: str = "INBOX") -> str:
        """Read one email by Message-ID without marking it read.

        Args:
            message_id: The email's Message-ID (with or without angle brackets).
            mailbox: IMAP mailbox to search. Default: "INBOX".

        Returns:
            JSON string with the normalized email, or an error.
        """
        try:
            with self._lock:
                imap = self._imap_login()
                try:
                    imap.select(_quote_mailbox(mailbox))
                    nums = self._find_by_message_id(imap, message_id)
                    if not nums:
                        return json.dumps({"ok": False, "error": "Message not found."})
                    msg = self._fetch_message(imap, nums[0], peek=True)
                    if msg is None:
                        return json.dumps({"ok": False, "error": "Fetch failed."})
                    normalized = _normalize_mail(msg, mailbox)
                    normalized["text"] = _truncate_text(normalized["text"], 8000)
                    return json.dumps({"ok": True, "email": normalized}, indent=2)
                finally:
                    self._imap_logout(imap)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def mark_email_read(self, message_id: str, mailbox: str = "INBOX") -> str:
        """Mark an email as read (``\\Seen``) by Message-ID.

        Args:
            message_id: The email's Message-ID (with or without angle brackets).
            mailbox: IMAP mailbox to search. Default: "INBOX".

        Returns:
            JSON string with ok status.
        """
        try:
            with self._lock:
                imap = self._imap_login()
                try:
                    imap.select(_quote_mailbox(mailbox))
                    nums = self._find_by_message_id(imap, message_id)
                    if not nums:
                        return json.dumps({"ok": False, "error": "Message not found."})
                    status, _ = imap.store(nums[0], "+FLAGS", "\\Seen")
                    return _store_result(status)
                finally:
                    self._imap_logout(imap)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class EmailAgent(BaseChannelAgent):
    """Channel agent with generic IMAP/SMTP email tools."""

    channel_system_prompt = (
        "You are chatting with users over email. Inbound messages are "
        "unread emails; the message text starts with 'Subject: ...'. "
        "Reply with send_email, passing the sender's address as 'to' "
        "and the email's Message-ID as 'in_reply_to' so replies stay "
        "threaded. Write plain text (no HTML or Markdown) and never "
        "answer automated mail."
    )

    def __init__(self) -> None:
        super().__init__("Email Agent")
        self._backend = EmailChannelBackend()
        cfg = _config.load()
        if cfg:  # pragma: no branch
            self._backend._cfg = cfg

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._cfg)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_email_auth() -> str:
            """Check if the email account is configured.

            Returns:
                Configuration status or instructions.
            """
            if not agent._backend._cfg:  # pragma: no branch
                return (
                    "Not configured for email. Use authenticate_email() with your "
                    "IMAP host, SMTP host, email address, and password (usually an "
                    "app password, e.g. from Google Account > Security > App passwords)."
                )
            cfg = agent._backend._cfg
            return json.dumps(
                {
                    "ok": True,
                    "email_address": cfg["email_address"],
                    "imap_host": cfg["imap_host"],
                    "smtp_host": cfg["smtp_host"],
                    "smtp_security": cfg.get("smtp_security", "ssl"),
                }
            )

        def authenticate_email(
            imap_host: str,
            smtp_host: str,
            email_address: str,
            password: str,
            imap_port: str = "993",
            smtp_port: str = "465",
            smtp_security: str = "ssl",
        ) -> str:
            """Configure a generic email account (IMAP for reading, SMTP for sending).

            Args:
                imap_host: IMAP server hostname (e.g. "imap.gmail.com").
                smtp_host: SMTP server hostname (e.g. "smtp.gmail.com").
                email_address: The account's email address.
                password: Account password or app password.
                imap_port: IMAP SSL port. Default: "993".
                smtp_port: SMTP port. Default: "465".
                smtp_security: "ssl" (implicit TLS) or "starttls". Default: "ssl".

            Returns:
                Configuration result or error message.
            """
            fields = {
                "imap_host": imap_host.strip(),
                "smtp_host": smtp_host.strip(),
                "email_address": email_address.strip(),
                "password": password,
            }
            for key, value in fields.items():
                if not value:
                    return f"{key} cannot be empty."
            smtp_security = smtp_security.strip().lower()
            if smtp_security not in ("ssl", "starttls"):
                return "smtp_security must be 'ssl' or 'starttls'."
            for key, port in (("imap_port", imap_port), ("smtp_port", smtp_port)):
                if not str(port).strip().isdigit():
                    return f"{key} must be a port number."
            cfg = {
                **fields,
                "imap_port": str(imap_port).strip(),
                "smtp_port": str(smtp_port).strip(),
                "smtp_security": smtp_security,
            }
            _config.save(cfg)
            agent._backend._cfg = cfg
            return json.dumps({"ok": True, "message": "Email account configured."})

        def clear_email_auth() -> str:
            """Clear the stored email configuration.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._cfg = {}
            return "Email configuration cleared."

        return [check_email_auth, authenticate_email, clear_email_auth]


def _make_backend() -> EmailChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = EmailChannelBackend()
    cfg = _config.load()
    if not cfg:  # pragma: no branch
        print("Not configured. Run: kiss-email -t 'authenticate'")
        sys.exit(1)
    backend._cfg = cfg
    return backend


def main() -> None:
    """Run the EmailAgent from the command line with chat persistence."""
    channel_main(
        EmailAgent,
        "kiss-email",
        channel_name="Email",
        make_backend=_make_backend,
    )


def get_tools() -> list:
    """Return the Email channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return EmailAgent()._get_tools()


if __name__ == "__main__":
    main()
