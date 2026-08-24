# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end reproduction and fix test for email redelivery (G-RC3).

Email poll mode searches ``UNSEEN`` with a peek fetch and never
advances a cursor, so before the fix every still-unread email spawned a
fresh task and a fresh SMTP reply on EVERY tick.  The fix: the channel
runner now calls an optional ``ack_message`` backend hook after
handling a message, and the email backend implements it by marking the
mail ``\\Seen``.

These tests drive the REAL ``EmailChannelBackend`` — stdlib
``imaplib.IMAP4_SSL`` and ``smtplib.SMTP_SSL`` clients — against a
minimal in-test IMAP and SMTP server pair speaking real TLS with a
self-signed certificate (generated with ``cryptography``).  The
servers are real socket servers, not mocks of any KISS code.

The runner's ``_launch_task`` is overridden with a real in-test
implementation (the same pattern as ``LaunchOutcomeRunner`` in
``test_hermes_runner.py``): launching actual kiss-web daemon tasks is
out of scope for a redelivery test.
"""

from __future__ import annotations

import datetime
import ipaddress
import logging
import os
import socket
import ssl
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.third_party_agents._channel_agent_utils import ChannelRunner
from kiss.agents.third_party_agents.email_agent import EmailChannelBackend, _config

_RAW_MAIL = (
    b"From: Alice Example <alice@example.com>\r\n"
    b"To: bot@example.com\r\n"
    b"Subject: Need help\r\n"
    b"Date: Mon, 01 Jan 2024 12:00:00 +0000\r\n"
    b"Message-ID: <need-help-1@example.com>\r\n"
    b"Content-Type: text/plain; charset=\"utf-8\"\r\n"
    b"\r\n"
    b"Hi bot, please help.\r\n"
)


@pytest.fixture(autouse=True)
def _isolated_kiss_home(tmp_path: Path) -> Iterator[Path]:
    """Point KISS_HOME at a fresh temp dir so tests never touch ~/.kiss."""
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


def _make_ssl_context(tmp_path: Path) -> ssl.SSLContext:
    """Generate a self-signed localhost certificate and server context."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(cert_path), str(key_path))
    return context


class _ImapLiteServer:
    """Threaded IMAP4rev1-subset server over TLS with one mailbox.

    Supports exactly what ``EmailChannelBackend`` uses: CAPABILITY,
    LOGIN, SELECT, SEARCH UNSEEN, SEARCH HEADER Message-ID, FETCH
    (BODY.PEEK[] / RFC822), STORE +FLAGS \\Seen, and LOGOUT.
    """

    def __init__(self, ssl_context: ssl.SSLContext) -> None:
        self.messages: list[dict[str, Any]] = []
        self.stored_flags: list[str] = []
        self._ssl_context = ssl_context
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def add_message(self, raw: bytes, message_id: str) -> None:
        """Add an unread message to the mailbox."""
        self.messages.append({"raw": raw, "id": message_id, "seen": False})

    def unseen(self) -> list[int]:
        """Return 1-based sequence numbers of unread messages."""
        return [i + 1 for i, m in enumerate(self.messages) if not m["seen"]]

    def close(self) -> None:
        """Stop accepting connections."""
        self._stopping.set()
        self._sock.close()

    def _accept_loop(self) -> None:
        """Accept and serve connections until closed."""
        while not self._stopping.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(
                target=self._serve, args=(conn,), daemon=True
            ).start()

    def _serve(self, conn: socket.socket) -> None:
        """Serve one IMAP connection."""
        try:
            tls = self._ssl_context.wrap_socket(conn, server_side=True)
        except (OSError, ssl.SSLError):
            conn.close()
            return
        try:
            tls.sendall(b"* OK IMAP4rev1 Service Ready\r\n")
            fp = tls.makefile("rwb")
            while True:
                line = fp.readline()
                if not line:
                    return
                parts = line.decode("utf-8", errors="replace").strip().split(" ", 2)
                tag = parts[0]
                cmd = parts[1].upper() if len(parts) > 1 else ""
                args = parts[2] if len(parts) > 2 else ""
                if cmd == "CAPABILITY":
                    fp.write(b"* CAPABILITY IMAP4rev1\r\n")
                    fp.write(f"{tag} OK CAPABILITY completed\r\n".encode())
                elif cmd == "LOGIN":
                    fp.write(f"{tag} OK LOGIN completed\r\n".encode())
                elif cmd == "SELECT":
                    fp.write(f"* {len(self.messages)} EXISTS\r\n".encode())
                    fp.write(b"* FLAGS (\\Seen)\r\n")
                    fp.write(f"{tag} OK [READ-WRITE] SELECT completed\r\n".encode())
                elif cmd == "SEARCH":
                    fp.write(self._search(args))
                    fp.write(f"{tag} OK SEARCH completed\r\n".encode())
                elif cmd == "FETCH":
                    num = int(args.split(" ", 1)[0])
                    raw = self.messages[num - 1]["raw"]
                    fp.write(f"* {num} FETCH (BODY[] {{{len(raw)}}}\r\n".encode())
                    fp.write(raw)
                    fp.write(b")\r\n")
                    fp.write(f"{tag} OK FETCH completed\r\n".encode())
                elif cmd == "STORE":
                    num_str, rest = args.split(" ", 1)
                    self.stored_flags.append(rest)
                    if "\\Seen" in rest:  # pragma: no branch
                        self.messages[int(num_str) - 1]["seen"] = True
                    fp.write(f"{tag} OK STORE completed\r\n".encode())
                elif cmd == "LOGOUT":
                    fp.write(b"* BYE\r\n")
                    fp.write(f"{tag} OK LOGOUT completed\r\n".encode())
                    fp.flush()
                    return
                else:
                    fp.write(f"{tag} OK {cmd} ignored\r\n".encode())
                fp.flush()
        except (OSError, ssl.SSLError, ValueError, IndexError):
            pass
        finally:
            tls.close()

    def _search(self, args: str) -> bytes:
        """Answer SEARCH UNSEEN and SEARCH HEADER Message-ID queries."""
        upper = args.upper()
        if "UNSEEN" in upper:
            hits = self.unseen()
        elif "HEADER MESSAGE-ID" in upper:
            wanted = args.split('"')[1]
            hits = [
                i + 1
                for i, m in enumerate(self.messages)
                if m["id"].strip("<>") == wanted.strip("<>")
            ]
        else:
            hits = []
        listing = (" " + " ".join(str(n) for n in hits)) if hits else ""
        return f"* SEARCH{listing}\r\n".encode()


class _SmtpLiteServer:
    """Threaded SMTP-subset server over implicit TLS recording deliveries."""

    def __init__(self, ssl_context: ssl.SSLContext) -> None:
        self.deliveries: list[bytes] = []
        self._ssl_context = ssl_context
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stop accepting connections."""
        self._stopping.set()
        self._sock.close()

    def _accept_loop(self) -> None:
        """Accept and serve connections until closed."""
        while not self._stopping.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(
                target=self._serve, args=(conn,), daemon=True
            ).start()

    def _serve(self, conn: socket.socket) -> None:
        """Serve one SMTP connection."""
        try:
            tls = self._ssl_context.wrap_socket(conn, server_side=True)
        except (OSError, ssl.SSLError):
            conn.close()
            return
        try:
            fp = tls.makefile("rwb")
            fp.write(b"220 rr-test SMTP\r\n")
            fp.flush()
            while True:
                line = fp.readline()
                if not line:
                    return
                verb = line.decode("utf-8", errors="replace").strip().upper()
                if verb.startswith("EHLO") or verb.startswith("HELO"):
                    fp.write(b"250-rr-test\r\n250 AUTH PLAIN LOGIN\r\n")
                elif verb.startswith("AUTH"):
                    fp.write(b"235 2.7.0 accepted\r\n")
                elif verb.startswith("MAIL") or verb.startswith("RCPT"):
                    fp.write(b"250 OK\r\n")
                elif verb.startswith("DATA"):
                    fp.write(b"354 go ahead\r\n")
                    fp.flush()
                    body = b""
                    while not body.endswith(b"\r\n.\r\n"):
                        chunk = fp.readline()
                        if not chunk:
                            return
                        body += chunk
                    self.deliveries.append(body)
                    fp.write(b"250 OK delivered\r\n")
                elif verb.startswith("QUIT"):
                    fp.write(b"221 bye\r\n")
                    fp.flush()
                    return
                else:
                    fp.write(b"250 OK\r\n")
                fp.flush()
        except (OSError, ssl.SSLError):
            pass
        finally:
            tls.close()


class CountingRunner(ChannelRunner):
    """Runner whose task launch is a real in-test implementation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.launched_prompts: list[str] = []

    def _launch_task(
        self, channel_id: str, thread_ts: str, prompt: str, last_reply_ts: str
    ) -> str:
        """Record the launch and report a successful task."""
        self.launched_prompts.append(prompt)
        return "success: true\nsummary: handled your request\n"


@pytest.fixture()
def mail_stack(
    tmp_path: Path,
) -> Iterator[tuple[EmailChannelBackend, _ImapLiteServer, _SmtpLiteServer]]:
    """A configured email backend wired to live IMAP/SMTP-lite servers."""
    context = _make_ssl_context(tmp_path)
    imap = _ImapLiteServer(context)
    smtp = _SmtpLiteServer(context)
    _config.save(
        {
            "imap_host": "127.0.0.1",
            "imap_port": str(imap.port),
            "smtp_host": "127.0.0.1",
            "smtp_port": str(smtp.port),
            "smtp_security": "ssl",
            "email_address": "bot@example.com",
            "password": "app-password",
        }
    )
    backend = EmailChannelBackend()
    try:
        yield backend, imap, smtp
    finally:
        imap.close()
        smtp.close()


class TestEmailRedeliveryStops:
    """The G-RC3 reproduction: one mail, two ticks, exactly one task+reply."""

    def test_two_ticks_process_one_mail_once(
        self, mail_stack: tuple[EmailChannelBackend, _ImapLiteServer, _SmtpLiteServer]
    ) -> None:
        """Tick 1 handles, replies, and acks; tick 2 redelivers nothing."""
        backend, imap, smtp = mail_stack
        imap.add_message(_RAW_MAIL, "<need-help-1@example.com>")
        runner = CountingRunner(
            backend=backend, channel_name="", agent_name="RR Email Test"
        )

        assert runner.run_once() == 1
        assert len(runner.launched_prompts) == 1
        assert "please help" in runner.launched_prompts[0]
        assert len(smtp.deliveries) == 1
        reply = smtp.deliveries[0].decode("utf-8", errors="replace")
        assert "handled your request" in reply
        assert "In-Reply-To: <need-help-1@example.com>" in reply
        # The ack marked the mail read on the server.
        assert imap.unseen() == []
        assert any("\\Seen" in flags for flags in imap.stored_flags)

        # Pre-fix, the still-unread mail was handled again every tick.
        assert runner.run_once() == 0
        assert len(runner.launched_prompts) == 1
        assert len(smtp.deliveries) == 1

    def test_new_mail_after_ack_is_still_picked_up(
        self, mail_stack: tuple[EmailChannelBackend, _ImapLiteServer, _SmtpLiteServer]
    ) -> None:
        """Acking one mail must not suppress genuinely new mail."""
        backend, imap, smtp = mail_stack
        imap.add_message(_RAW_MAIL, "<need-help-1@example.com>")
        runner = CountingRunner(
            backend=backend, channel_name="", agent_name="RR Email Test"
        )
        assert runner.run_once() == 1
        second = _RAW_MAIL.replace(b"need-help-1", b"need-help-2")
        imap.add_message(second, "<need-help-2@example.com>")
        assert runner.run_once() == 1
        assert len(runner.launched_prompts) == 2
        assert len(smtp.deliveries) == 2
        assert imap.unseen() == []


class TestEmailAckMessage:
    """Direct branches of the email ack hook."""

    def test_ack_without_message_id_is_a_noop(
        self, mail_stack: tuple[EmailChannelBackend, _ImapLiteServer, _SmtpLiteServer]
    ) -> None:
        """A message without a Message-ID cannot be acked; nothing happens."""
        backend, imap, _ = mail_stack
        assert backend.connect() is True
        backend.ack_message("INBOX", {"ts": "1", "thread_ts": ""})
        assert imap.stored_flags == []

    def test_ack_unknown_message_logs_warning(
        self,
        mail_stack: tuple[EmailChannelBackend, _ImapLiteServer, _SmtpLiteServer],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An ack for a vanished mail warns instead of raising."""
        backend, imap, _ = mail_stack
        assert backend.connect() is True
        with caplog.at_level(logging.WARNING):
            backend.ack_message(
                "INBOX", {"ts": "1", "thread_ts": "<gone@example.com>"}
            )
        assert "Could not mark email" in caplog.text
        assert imap.unseen() == []


class _NoAckBackend:
    """Minimal real backend without an ack hook (cursor-based platforms)."""

    _connection_info = "no-ack backend"

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages
        self.sent: list[str] = []

    @property
    def connection_info(self) -> str:
        """Connection status string."""
        return self._connection_info

    def connect(self) -> bool:
        """Always connected."""
        return True

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 50
    ) -> tuple[list[dict[str, Any]], str]:
        """Serve the configured messages once."""
        messages, self.messages = self.messages, []
        return messages, oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Record the reply."""
        self.sent.append(text)

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """No bot messages in these tests."""
        return False

    def strip_bot_mention(self, text: str) -> str:
        """No mention stripping."""
        return text

    def disconnect(self) -> None:
        """Nothing to release."""


class _RaisingAckBackend(_NoAckBackend):
    """Real backend whose ack hook always fails."""

    def ack_message(self, channel_id: str, msg: dict[str, Any]) -> None:
        """Simulate a platform error during the ack."""
        raise ConnectionError("ack transport down")


class TestRunnerAckDispatch:
    """ChannelRunner._ack_message branch behavior."""

    def test_backend_without_hook_is_untouched(self) -> None:
        """Cursor-based backends need no ack and still process normally."""
        backend = _NoAckBackend([{"ts": "1", "user": "alice", "text": "hi"}])
        runner = CountingRunner(backend=backend, channel_name="", agent_name="t")
        assert runner.run_once() == 1
        assert len(backend.sent) == 1

    def test_ack_failure_is_logged_not_raised(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An ack that raises never fails the tick (reply already sent)."""
        backend = _RaisingAckBackend([{"ts": "1", "user": "alice", "text": "hi"}])
        runner = CountingRunner(backend=backend, channel_name="", agent_name="t")
        with caplog.at_level(logging.WARNING):
            assert runner.run_once() == 1
        assert len(backend.sent) == 1
        assert "ack_message failed" in caplog.text
