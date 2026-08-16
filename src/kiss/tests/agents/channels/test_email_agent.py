# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the generic IMAP/SMTP Email channel agent.

Exercises real code paths on real ``email.message`` objects — no
mocks, no test doubles, and no network calls.  Config state is
isolated by pointing ``KISS_HOME`` at a per-test temporary directory
(the same isolation pattern the other channel tests rely on).
"""

from __future__ import annotations

import email
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import kiss.agents.third_party_agents.email_agent as email_agent_mod
from kiss.agents.third_party_agents.email_agent import (
    EmailAgent,
    EmailChannelBackend,
    _build_outbound,
    _config,
    _is_automated_mail,
    _normalize_mail,
    get_tools,
)

_AUTH_TRIO = {"check_email_auth", "authenticate_email", "clear_email_auth"}
_BACKEND_TOOLS = {"send_email", "list_unread_emails", "read_email", "mark_email_read"}


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


def _tool(agent: EmailAgent, name: str):
    """Return the agent tool callable with the given name."""
    for tool in agent._get_tools():
        if tool.__name__ == name:
            return tool
    raise AssertionError(f"tool {name!r} not found")


def _authenticate(agent: EmailAgent) -> None:
    """Configure the agent with valid test credentials."""
    result = _tool(agent, "authenticate_email")(
        imap_host="imap.example.com",
        smtp_host="smtp.example.com",
        email_address="bot@example.com",
        password="app-password",
    )
    assert json.loads(result)["ok"] is True


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


_RAW_MULTIPART = b"""\
From: Bob <bob@example.com>\r
To: bot@example.com\r
Subject: =?utf-8?b?SMOpbGxv?=\r
Date: Tue, 02 Jan 2024 08:30:00 +0000\r
Message-ID: <multi42@example.com>\r
MIME-Version: 1.0\r
Content-Type: multipart/alternative; boundary="XYZ"\r
\r
--XYZ\r
Content-Type: text/plain; charset="utf-8"\r
\r
plain body wins\r
--XYZ\r
Content-Type: text/html; charset="utf-8"\r
\r
<p>html body loses</p>\r
--XYZ--\r
"""


class TestAuthFlow:
    """Agent instantiation and the authentication tool trio."""

    def test_unauthenticated_agent(self) -> None:
        """A fresh agent is unauthenticated and exposes only the auth trio."""
        agent = EmailAgent()
        assert agent.name == "Email Agent"
        assert agent._is_authenticated() is False
        assert {t.__name__ for t in agent._get_tools()} == _AUTH_TRIO

    def test_authenticate_persists_config(self, _isolated_kiss_home: Path) -> None:
        """authenticate_email persists a 0600 config with port/security defaults."""
        agent = EmailAgent()
        _authenticate(agent)
        path = _config.path
        assert path.exists()
        assert path.is_relative_to(_isolated_kiss_home)
        if sys.platform != "win32":
            assert path.stat().st_mode & 0o777 == 0o600
        cfg = json.loads(path.read_text(encoding="utf-8"))
        assert cfg["imap_host"] == "imap.example.com"
        assert cfg["smtp_host"] == "smtp.example.com"
        assert cfg["email_address"] == "bot@example.com"
        assert cfg["password"] == "app-password"
        assert cfg["imap_port"] == "993"
        assert cfg["smtp_port"] == "465"
        assert cfg["smtp_security"] == "ssl"

    def test_check_and_clear_auth(self) -> None:
        """check_email_auth reports the config; clear_email_auth removes it."""
        agent = EmailAgent()
        assert "Not configured" in _tool(agent, "check_email_auth")()
        _authenticate(agent)
        status = json.loads(_tool(agent, "check_email_auth")())
        assert status["ok"] is True
        assert status["email_address"] == "bot@example.com"
        assert status["smtp_security"] == "ssl"
        assert "cleared" in _tool(agent, "clear_email_auth")()
        assert not _config.path.exists()
        assert agent._is_authenticated() is False

    def test_authenticated_agent_exposes_backend_tools(self) -> None:
        """After authentication a fresh agent exposes the backend tool methods."""
        _authenticate(EmailAgent())
        agent = EmailAgent()
        assert agent._is_authenticated() is True
        names = {t.__name__ for t in agent._get_tools()}
        assert _AUTH_TRIO <= names
        assert _BACKEND_TOOLS <= names

    def test_authenticate_rejects_bad_input(self) -> None:
        """Empty fields, bad security modes, and bad ports are rejected unsaved."""
        agent = EmailAgent()
        auth = _tool(agent, "authenticate_email")
        assert auth("", "s", "e@x.com", "p") == "imap_host cannot be empty."
        assert auth("i", "s", "e@x.com", "p", smtp_security="tls") == (
            "smtp_security must be 'ssl' or 'starttls'."
        )
        assert auth("i", "s", "e@x.com", "p", imap_port="abc") == "imap_port must be a port number."
        assert auth("i", "s", "e@x.com", "p", smtp_port="") == "smtp_port must be a port number."
        assert not _config.path.exists()
        assert agent._is_authenticated() is False

    def test_authenticate_accepts_starttls(self) -> None:
        """Custom ports and starttls security are persisted verbatim."""
        agent = EmailAgent()
        result = _tool(agent, "authenticate_email")(
            imap_host="imap.example.com",
            smtp_host="smtp.example.com",
            email_address="bot@example.com",
            password="pw",
            imap_port="1993",
            smtp_port="587",
            smtp_security="starttls",
        )
        assert json.loads(result)["ok"] is True
        cfg = json.loads(_config.path.read_text(encoding="utf-8"))
        assert (cfg["imap_port"], cfg["smtp_port"], cfg["smtp_security"]) == (
            "1993",
            "587",
            "starttls",
        )

    def test_module_get_tools(self) -> None:
        """The module-level get_tools() returns a non-empty tool list."""
        tools = get_tools()
        assert tools
        assert _AUTH_TRIO <= {t.__name__ for t in tools}


class TestMailNormalization:
    """RFC822 parsing and normalization on real email.message objects."""

    def test_plain_message(self) -> None:
        """A plain-text mail normalizes to the channel message shape."""
        msg = email.message_from_bytes(_raw_plain())
        normalized = _normalize_mail(msg, "INBOX")
        assert normalized["user"] == "alice@example.com"
        assert normalized["username"] == "Alice Example"
        assert normalized["subject"] == "Hello there"
        assert normalized["text"] == "Subject: Hello there\n\nHi bot, please help."
        assert normalized["channel_id"] == "INBOX"
        assert normalized["thread_ts"] == "<abc123@example.com>"
        assert normalized["ts"] == "1704110400.0"

    def test_multipart_prefers_plain_and_decodes_subject(self) -> None:
        """Multipart mail yields the text/plain body and RFC 2047 subjects decode."""
        msg = email.message_from_bytes(_RAW_MULTIPART)
        normalized = _normalize_mail(msg, "Work")
        assert normalized["subject"] == "H\u00e9llo"
        assert normalized["text"] == "Subject: H\u00e9llo\n\nplain body wins"
        assert normalized["channel_id"] == "Work"
        assert normalized["thread_ts"] == "<multi42@example.com>"

    def test_missing_headers_are_tolerated(self) -> None:
        """Mail without Date/Message-ID/Subject still normalizes."""
        msg = email.message_from_bytes(_raw_plain(Date="", Subject="", **{"Message-ID": ""}))
        normalized = _normalize_mail(msg, "INBOX")
        assert normalized["ts"] == ""
        assert normalized["thread_ts"] == ""
        assert normalized["text"] == "Subject: \n\nHi bot, please help."


class TestAutomatedMailFiltering:
    """Each Hermes-style automated-mail drop rule, plus the pass case."""

    def test_human_mail_is_not_automated(self) -> None:
        """Ordinary human mail passes the filter."""
        assert _is_automated_mail(email.message_from_bytes(_raw_plain())) is False

    @pytest.mark.parametrize(
        "sender",
        [
            "noreply@example.com",
            "no-reply@example.com",
            "donotreply@example.com",
            "MAILER-DAEMON@example.com",
            "Bank <alerts.noreply@bank.example>",
        ],
    )
    def test_noreply_senders(self, sender: str) -> None:
        """No-reply style From addresses are automated."""
        assert _is_automated_mail(email.message_from_bytes(_raw_plain(From=sender))) is True

    def test_auto_submitted(self) -> None:
        """Auto-Submitted != 'no' is automated; 'no' is not."""
        auto = email.message_from_bytes(_raw_plain(**{"Auto-Submitted": "auto-generated"}))
        assert _is_automated_mail(auto) is True
        manual = email.message_from_bytes(_raw_plain(**{"Auto-Submitted": "No"}))
        assert _is_automated_mail(manual) is False

    @pytest.mark.parametrize("precedence", ["bulk", "junk", "list", "Bulk"])
    def test_precedence(self, precedence: str) -> None:
        """Precedence bulk/junk/list is automated."""
        msg = email.message_from_bytes(_raw_plain(Precedence=precedence))
        assert _is_automated_mail(msg) is True

    def test_precedence_first_class_is_fine(self) -> None:
        """Other Precedence values are not automated."""
        msg = email.message_from_bytes(_raw_plain(Precedence="first-class"))
        assert _is_automated_mail(msg) is False

    def test_list_id(self) -> None:
        """A List-Id header marks mailing-list mail as automated."""
        msg = email.message_from_bytes(_raw_plain(**{"List-Id": "<dev.lists.example.com>"}))
        assert _is_automated_mail(msg) is True


class TestOutboundConstruction:
    """Reply-header construction for outbound MIME messages."""

    def test_reply_sets_threading_headers(self) -> None:
        """Replies carry In-Reply-To/References and a 'Re: ' subject."""
        msg = _build_outbound(
            "bot@example.com",
            "alice@example.com",
            "Hello there",
            "On it!",
            in_reply_to="<abc123@example.com>",
        )
        assert msg["In-Reply-To"] == "<abc123@example.com>"
        assert msg["References"] == "<abc123@example.com>"
        assert msg["Subject"] == "Re: Hello there"
        assert msg["From"] == "bot@example.com"
        assert msg["To"] == "alice@example.com"
        assert msg.get_content().strip() == "On it!"

    def test_reply_adds_angle_brackets_and_keeps_re(self) -> None:
        """Bare Message-IDs get angle brackets; 'Re:' subjects are not doubled."""
        msg = _build_outbound(
            "bot@example.com", "a@x.com", "RE: Hello", "body", in_reply_to="abc123@example.com"
        )
        assert msg["In-Reply-To"] == "<abc123@example.com>"
        assert msg["Subject"] == "RE: Hello"

    def test_fresh_mail_has_no_threading_headers(self) -> None:
        """Non-replies have no threading headers and an unmodified subject."""
        msg = _build_outbound("bot@example.com", "a@x.com", "Weekly report", "body")
        assert msg["In-Reply-To"] is None
        assert msg["References"] is None
        assert msg["Subject"] == "Weekly report"


class TestBackendBehavior:
    """Backend logic that needs no network."""

    def test_is_from_bot(self) -> None:
        """is_from_bot matches the configured address case-insensitively."""
        backend = EmailChannelBackend()
        assert backend.is_from_bot({"user": "bot@example.com"}) is False
        backend._cfg = {"email_address": "Bot@Example.com"}
        assert backend.is_from_bot({"user": "bot@example.com"}) is True
        assert backend.is_from_bot({"user": "alice@example.com"}) is False

    def test_connect_without_config(self) -> None:
        """connect() fails cleanly when no config is stored."""
        backend = EmailChannelBackend()
        assert backend.connect() is False
        assert backend.connection_info == "No email config found."

    def test_connect_with_config(self) -> None:
        """connect() loads the persisted config."""
        _authenticate(EmailAgent())
        backend = EmailChannelBackend()
        assert backend.connect() is True
        assert backend.connection_info == "Email configured for bot@example.com"
        assert backend._cfg["imap_host"] == "imap.example.com"

    def test_resolve_outbound_reply_to_cached_sender(self) -> None:
        """A mailbox channel_id resolves to the cached thread sender and subject."""
        backend = EmailChannelBackend()
        backend._threads["<abc123@example.com>"] = {
            "subject": "Hello there",
            "from": "alice@example.com",
        }
        to, subject = backend._resolve_outbound("INBOX", "<abc123@example.com>")
        assert (to, subject) == ("alice@example.com", "Hello there")

    def test_resolve_outbound_explicit_recipient_wins(self) -> None:
        """An explicit address channel_id is kept even when replying."""
        backend = EmailChannelBackend()
        backend._threads["<abc123@example.com>"] = {"subject": "Hi", "from": "a@x.com"}
        to, subject = backend._resolve_outbound("carol@example.com", "<abc123@example.com>")
        assert (to, subject) == ("carol@example.com", "Hi")

    def test_resolve_outbound_uncached_and_fresh(self) -> None:
        """Uncached replies get a fallback subject; fresh mail gets the default."""
        backend = EmailChannelBackend()
        assert backend._resolve_outbound("INBOX", "<gone@example.com>") == (
            "INBOX",
            "your message",
        )
        assert backend._resolve_outbound("bob@example.com", "") == (
            "bob@example.com",
            "Message from KISS Email Agent",
        )

    def test_tool_method_discovery(self) -> None:
        """Channel protocol methods are hidden; email tools are exposed."""
        backend = EmailChannelBackend()
        names = {t.__name__ for t in backend.get_tool_methods()}
        assert names == _BACKEND_TOOLS

    def test_send_email_reports_errors_as_json(self) -> None:
        """Tool methods never raise: an unconfigured send returns an error JSON."""
        backend = EmailChannelBackend()
        result = json.loads(backend.send_email("a@x.com", "hi", "body"))
        assert result["ok"] is False
        assert result["error"]

    def test_make_backend_exits_without_config(self, capsys: pytest.CaptureFixture) -> None:
        """_make_backend prints a hint and exits when unconfigured."""
        with pytest.raises(SystemExit):
            email_agent_mod._make_backend()
        assert "Not configured" in capsys.readouterr().out

    def test_make_backend_with_config(self) -> None:
        """_make_backend returns a configured backend."""
        _authenticate(EmailAgent())
        backend = email_agent_mod._make_backend()
        assert backend._cfg["email_address"] == "bot@example.com"
