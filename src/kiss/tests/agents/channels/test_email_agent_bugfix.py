# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests for confirmed bug fixes in the Email channel agent.

Covers: Reply-To preference for outbound recipients, safe handling of
mail without a Message-ID (thread caching by ts + RuntimeError on a
bogus recipient), RFC 5322 Date/Message-ID on outbound mail, HTML-only
body extraction, JSON validity of truncated large bodies, IMAP STORE
failure reporting, and mailbox-name quoting.  All tests exercise real
code paths on real ``email.message`` objects — no network, no mocks.
"""

from __future__ import annotations

import email
import json
import os
from collections.abc import Iterator
from email.utils import parsedate_to_datetime
from pathlib import Path

import pytest

from kiss.agents.third_party_agents.email_agent import (
    EmailChannelBackend,
    _build_outbound,
    _message_body_text,
    _normalize_mail,
    _quote_mailbox,
    _store_result,
    _truncate_text,
)


@pytest.fixture(autouse=True)
def _isolated_kiss_home(tmp_path: Path) -> Iterator[Path]:
    """Point ``KISS_HOME`` at a fresh temp dir so tests never touch ~/.kiss."""
    saved = os.environ.get("KISS_HOME")
    home = tmp_path / "kiss_home"
    os.environ["KISS_HOME"] = str(home)
    try:
        yield home
    finally:
        if saved is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = saved


def _raw_plain(**overrides: str) -> bytes:
    """Build a plain-text RFC822 message, allowing header overrides."""
    headers = {
        "From": "Alice Example <alice@example.com>",
        "To": "bot@example.com",
        "Subject": "Hello there",
        "Date": "Mon, 01 Jan 2024 12:00:00 +0000",
        "Message-ID": "<abc123@example.com>",
        "Content-Type": 'text/plain; charset="utf-8"',
    }
    headers.update(overrides)
    lines = [f"{k}: {v}" for k, v in headers.items() if v]
    return ("\r\n".join(lines) + "\r\n\r\nHi bot, please help.\r\n").encode("utf-8")


_RAW_HTML_ONLY_MULTIPART = b"""\
From: Carol <carol@example.com>\r
To: bot@example.com\r
Subject: HTML only\r
Date: Wed, 03 Jan 2024 09:00:00 +0000\r
Message-ID: <htmlonly7@example.com>\r
MIME-Version: 1.0\r
Content-Type: multipart/alternative; boundary="HTM"\r
\r
--HTM\r
Content-Type: text/html; charset="utf-8"\r
\r
<html><head><style>p { color: red; }</style>\r
<script>alert("x");</script></head>\r
<body><p>Hello &amp; welcome,</p><p>see the <b>report</b> now.</p></body></html>\r
--HTM--\r
"""


class TestReplyToPreference:
    """Fix 2: Reply-To is parsed and preferred for the outbound recipient."""

    def test_normalize_mail_parses_reply_to(self) -> None:
        """_normalize_mail exposes the Reply-To address."""
        msg = email.message_from_bytes(
            _raw_plain(**{"Reply-To": "Alice Work <alice.work@corp.example>"})
        )
        normalized = _normalize_mail(msg, "INBOX")
        assert normalized["reply_to"] == "alice.work@corp.example"
        assert normalized["user"] == "alice@example.com"

    def test_normalize_mail_without_reply_to(self) -> None:
        """Mail without Reply-To yields an empty reply_to field."""
        normalized = _normalize_mail(email.message_from_bytes(_raw_plain()), "INBOX")
        assert normalized["reply_to"] == ""

    def test_cached_thread_prefers_reply_to(self) -> None:
        """The thread cache routes replies to Reply-To, not From."""
        backend = EmailChannelBackend()
        msg = email.message_from_bytes(_raw_plain(**{"Reply-To": "alice.work@corp.example"}))
        backend._cache_thread(_normalize_mail(msg, "INBOX"))
        to, subject = backend._resolve_outbound("INBOX", "<abc123@example.com>")
        assert (to, subject) == ("alice.work@corp.example", "Hello there")

    def test_cached_thread_falls_back_to_from(self) -> None:
        """Without Reply-To, the From address is cached as the recipient."""
        backend = EmailChannelBackend()
        backend._cache_thread(_normalize_mail(email.message_from_bytes(_raw_plain()), "INBOX"))
        to, _ = backend._resolve_outbound("INBOX", "<abc123@example.com>")
        assert to == "alice@example.com"

    def test_is_from_bot_still_uses_from(self) -> None:
        """is_from_bot keys on the From address even when Reply-To differs."""
        backend = EmailChannelBackend()
        backend._cfg = {"email_address": "bot@example.com"}
        msg = email.message_from_bytes(
            _raw_plain(From="bot@example.com", **{"Reply-To": "human@example.com"})
        )
        assert backend.is_from_bot(_normalize_mail(msg, "INBOX")) is True


class TestMissingMessageId:
    """Fix 3: no-Message-ID mail is cached by ts; bogus recipients raise."""

    def test_cache_thread_keys_by_ts_when_no_message_id(self) -> None:
        """Without a Message-ID the thread is cached by ts and thread_ts=ts."""
        backend = EmailChannelBackend()
        normalized = _normalize_mail(
            email.message_from_bytes(_raw_plain(**{"Message-ID": ""})), "INBOX"
        )
        assert normalized["thread_ts"] == ""
        backend._cache_thread(normalized)
        assert normalized["thread_ts"] == normalized["ts"] != ""
        to, subject = backend._resolve_outbound("INBOX", normalized["ts"])
        assert (to, subject) == ("alice@example.com", "Hello there")

    def test_send_message_raises_on_invalid_recipient(self) -> None:
        """send_message raises RuntimeError instead of mailing 'INBOX'."""
        backend = EmailChannelBackend()
        backend._cfg = {"email_address": "bot@example.com"}
        with pytest.raises(RuntimeError, match="INBOX"):
            backend.send_message("INBOX", "hello", "")

    def test_send_message_raises_for_unknown_thread(self) -> None:
        """An uncached thread ts still cannot resolve to a mailbox recipient."""
        backend = EmailChannelBackend()
        backend._cfg = {"email_address": "bot@example.com"}
        with pytest.raises(RuntimeError, match="no valid recipient"):
            backend.send_message("Sent Items", "hello", "1704110400.0")


class TestOutboundRfc5322Headers:
    """Fix 4: every outbound message carries Date and Message-ID."""

    def test_fresh_mail_has_date_and_message_id(self) -> None:
        """Non-reply outbound mail gets a parseable Date and a Message-ID."""
        msg = _build_outbound("bot@example.com", "a@x.com", "Report", "body")
        assert parsedate_to_datetime(str(msg["Date"])) is not None
        message_id = str(msg["Message-ID"])
        assert message_id.startswith("<") and message_id.endswith(">")
        assert "@" in message_id

    def test_reply_has_date_and_distinct_message_id(self) -> None:
        """Replies get their own Message-ID, distinct from In-Reply-To."""
        msg = _build_outbound(
            "bot@example.com",
            "a@x.com",
            "Hello",
            "body",
            in_reply_to="<abc123@example.com>",
        )
        assert msg["Date"] is not None
        assert msg["Message-ID"] is not None
        assert str(msg["Message-ID"]) != str(msg["In-Reply-To"])


class TestHtmlOnlyBody:
    """Fix 5: HTML-only mail falls back to tag-stripped text."""

    def test_html_only_multipart_body(self) -> None:
        """A multipart mail with only text/html yields stripped text."""
        body = _message_body_text(email.message_from_bytes(_RAW_HTML_ONLY_MULTIPART))
        assert "Hello & welcome," in body
        assert "see the report now." in body
        assert "<" not in body
        assert "alert" not in body
        assert "color: red" not in body

    def test_html_only_single_part_body(self) -> None:
        """A non-multipart text/html mail is also stripped."""
        raw = _raw_plain(**{"Content-Type": 'text/html; charset="utf-8"'})
        raw = raw.replace(b"Hi bot, please help.", b"<p>Hi <b>bot</b>, please help.</p>")
        body = _message_body_text(email.message_from_bytes(raw))
        assert body == "Hi bot, please help."

    def test_plain_part_still_wins_over_html(self) -> None:
        """text/plain is still preferred when both parts exist."""
        raw = (
            b"From: Bob <bob@example.com>\r\nTo: bot@example.com\r\n"
            b"Subject: Both\r\nMIME-Version: 1.0\r\n"
            b'Content-Type: multipart/alternative; boundary="B"\r\n\r\n'
            b'--B\r\nContent-Type: text/html; charset="utf-8"\r\n\r\n'
            b"<p>html loses</p>\r\n"
            b'--B\r\nContent-Type: text/plain; charset="utf-8"\r\n\r\n'
            b"plain wins\r\n--B--\r\n"
        )
        assert _message_body_text(email.message_from_bytes(raw)) == "plain wins"


class TestJsonTruncation:
    """Fix 6: bodies are truncated as text so tool JSON always parses."""

    def test_truncated_large_body_stays_valid_json(self) -> None:
        """A >8000-char body, truncated then dumped, survives json.loads."""
        raw = _raw_plain().replace(b"Hi bot, please help.", b"x" * 20000 + b'"quote')
        normalized = _normalize_mail(email.message_from_bytes(raw), "INBOX")
        normalized["text"] = _truncate_text(normalized["text"], 8000)
        payload = json.dumps({"ok": True, "email": normalized}, indent=2)
        parsed = json.loads(payload)
        assert parsed["ok"] is True
        assert parsed["email"]["text"].endswith("... [truncated]")
        assert len(parsed["email"]["text"]) <= 8000 + len("... [truncated]")

    def test_short_text_is_untouched(self) -> None:
        """Short bodies are returned verbatim without a truncation marker."""
        assert _truncate_text("short", 8000) == "short"


class TestStoreStatus:
    """Fix 7: mark_email_read reports a failed IMAP STORE."""

    def test_store_failure_reports_status(self) -> None:
        """A non-OK STORE status yields ok:false with the status."""
        result = json.loads(_store_result("NO"))
        assert result["ok"] is False
        assert "NO" in result["error"]

    def test_store_success_reports_ok(self) -> None:
        """An OK STORE status yields ok:true."""
        assert json.loads(_store_result("OK")) == {"ok": True}


class TestMailboxQuoting:
    """Fix 8: mailbox names with spaces are quoted for IMAP commands."""

    def test_mailbox_with_space_is_quoted(self) -> None:
        """Names containing spaces get wrapped in double quotes."""
        assert _quote_mailbox("Sent Items") == '"Sent Items"'

    def test_plain_mailbox_is_untouched(self) -> None:
        """Names without spaces pass through unchanged."""
        assert _quote_mailbox("INBOX") == "INBOX"

    def test_already_quoted_mailbox_is_untouched(self) -> None:
        """Pre-quoted names are not double-wrapped."""
        assert _quote_mailbox('"Sent Items"') == '"Sent Items"'
