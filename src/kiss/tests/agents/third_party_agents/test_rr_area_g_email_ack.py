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

The runner is the real ``ChannelRunner``: its ``_launch_task`` goes over
the wire (``run_agent_via_kiss_web`` → ``sorcar.run``) to a
:class:`RecordingDaemon` that records every ``run`` command and answers
with a scripted result text, so the prompts asserted below are the ones
the daemon actually received.
"""

from __future__ import annotations

import datetime
import ipaddress
import logging
import socket
import ssl
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.third_party_agents import _kiss_web_launcher
from kiss.agents.third_party_agents._channel_agent_utils import ChannelRunner
from kiss.agents.third_party_agents.email.email_sea import EmailChannelBackend, _config
from kiss.agents.third_party_agents.irc import irc_sea
from kiss.tests.agents.third_party_agents.recording_daemon import RecordingDaemon

_REPLY_TEXT = "handled your request"
# Idle limit on every accepted connection: a peer that connects and then
# goes silent (mid TLS handshake or mid command) cannot park a handler
# thread forever.  Far above any in-test exchange, and ``close()`` does
# not wait it out: it shuts the connections down and joins the handlers.
_CONNECTION_TIMEOUT = 30.0

_RAW_MAIL = (
    b"From: Alice Example <alice@example.com>\r\n"
    b"To: bot@example.com\r\n"
    b"Subject: Need help\r\n"
    b"Date: Mon, 01 Jan 2024 12:00:00 +0000\r\n"
    b"Message-ID: <need-help-1@example.com>\r\n"
    b'Content-Type: text/plain; charset="utf-8"\r\n'
    b"\r\n"
    b"Hi bot, please help.\r\n"
)


@pytest.fixture(autouse=True)
def _isolated_kiss_home(isolated_kiss_home: Path) -> Path:
    """Apply the shared per-test ``KISS_HOME`` isolation to every test here."""
    return isolated_kiss_home


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
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
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


class _TlsLiteServer:
    """Loopback TLS listener that serves each connection on its own thread.

    Subclasses implement ``_serve(tls)`` for a handshaken TLS socket.
    Every accepted connection gets ``_CONNECTION_TIMEOUT`` before it is
    wrapped, and the wrapped socket plus its handler thread are tracked
    from the accept loop (the handshake itself runs on the handler, so
    a silent peer parks only that thread, on a tracked socket).
    ``close()`` wakes the accept thread with ``shutdown(SHUT_RDWR)``
    (closing the fd alone does not interrupt a blocked ``accept()`` on
    Linux), shuts down every accepted connection the same way, and
    joins the accept thread and every handler with a bound, so no test
    leaves a thread parked on a dead or silent socket.
    """

    def __init__(self, ssl_context: ssl.SSLContext) -> None:
        self._ssl_context = ssl_context
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._stopping = threading.Event()
        self._handlers: list[tuple[ssl.SSLSocket, threading.Thread]] = []
        self._handlers_lock = threading.Lock()
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stop accepting, shut down every connection and join all threads."""
        self._stopping.set()
        with suppress(OSError):
            self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()
        self._thread.join(timeout=5)
        with self._handlers_lock:
            handlers = list(self._handlers)
        for tls, thread in handlers:
            with suppress(OSError):
                tls.shutdown(socket.SHUT_RDWR)
            thread.join(timeout=5)
            assert not thread.is_alive(), "connection handler did not stop"

    def _accept_loop(self) -> None:
        """Accept and serve connections until closed."""
        while not self._stopping.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            try:
                conn.settimeout(_CONNECTION_TIMEOUT)
                tls = self._ssl_context.wrap_socket(
                    conn, server_side=True, do_handshake_on_connect=False
                )
            except OSError:
                # A peer that resets right after connecting must not
                # take the accept thread down with it.
                conn.close()
                continue
            thread = threading.Thread(target=self._handle, args=(tls,), daemon=True)
            with self._handlers_lock:
                self._handlers.append((tls, thread))
            thread.start()

    def _handle(self, tls: ssl.SSLSocket) -> None:
        """Handshake, serve the protocol, and always close the socket."""
        try:
            tls.do_handshake()
            self._serve(tls)
        except (OSError, ssl.SSLError, ValueError, IndexError):
            pass
        finally:
            tls.close()

    def _serve(self, tls: ssl.SSLSocket) -> None:
        """Serve one handshaken connection (protocol-specific)."""
        raise NotImplementedError


class _ImapLiteServer(_TlsLiteServer):
    """Threaded IMAP4rev1-subset server over TLS with one mailbox.

    Supports exactly what ``EmailChannelBackend`` uses: CAPABILITY,
    LOGIN, SELECT, SEARCH UNSEEN, SEARCH HEADER Message-ID, FETCH
    (BODY.PEEK[] / RFC822), STORE +FLAGS \\Seen, and LOGOUT.
    """

    def __init__(self, ssl_context: ssl.SSLContext) -> None:
        self.messages: list[dict[str, Any]] = []
        self.stored_flags: list[str] = []
        super().__init__(ssl_context)

    def add_message(self, raw: bytes, message_id: str) -> None:
        """Add an unread message to the mailbox."""
        self.messages.append({"raw": raw, "id": message_id, "seen": False})

    def unseen(self) -> list[int]:
        """Return 1-based sequence numbers of unread messages."""
        return [i + 1 for i, m in enumerate(self.messages) if not m["seen"]]

    def _serve(self, tls: ssl.SSLSocket) -> None:
        """Serve one IMAP connection."""
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


class _SmtpLiteServer(_TlsLiteServer):
    """Threaded SMTP-subset server over implicit TLS recording deliveries."""

    def __init__(self, ssl_context: ssl.SSLContext) -> None:
        self.deliveries: list[bytes] = []
        super().__init__(ssl_context)

    def _serve(self, tls: ssl.SSLSocket) -> None:
        """Serve one SMTP connection."""
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


@pytest.fixture()
def daemon(monkeypatch: pytest.MonkeyPatch) -> Iterator[RecordingDaemon]:
    """A recording daemon every real ``ChannelRunner._launch_task`` reaches.

    ``run_agent_via_kiss_web`` resolves its endpoint file from the
    module-level ``_ENDPOINT_FILE_OVERRIDE`` when the caller passes
    none, so the runner's launch travels the real wire to this stand-in,
    which answers every task with ``_REPLY_TEXT``.
    """
    stand_in = RecordingDaemon(text=_REPLY_TEXT, chat_id="chat-email")
    monkeypatch.setattr(_kiss_web_launcher, "_ENDPOINT_FILE_OVERRIDE", str(stand_in.endpoint_file))
    try:
        yield stand_in
    finally:
        stand_in.close()


def _launched_prompts(daemon: RecordingDaemon) -> list[str]:
    """The prompts of the ``run`` commands *daemon* received, in order."""
    return [str(command["prompt"]) for command in daemon.run_commands]


@pytest.fixture()
def mail_stack(
    tmp_path: Path,
) -> Iterator[tuple[EmailChannelBackend, _ImapLiteServer, _SmtpLiteServer]]:
    """A configured email backend wired to live IMAP/SMTP-lite servers.

    Each server registers its own ``close`` as soon as it exists, so a
    failure while starting the second one still stops the first.
    """
    with ExitStack() as stack:
        context = _make_ssl_context(tmp_path)
        imap = _ImapLiteServer(context)
        stack.callback(imap.close)
        smtp = _SmtpLiteServer(context)
        stack.callback(smtp.close)
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
        yield EmailChannelBackend(), imap, smtp


class TestEmailRedeliveryStops:
    """The G-RC3 reproduction: one mail, two ticks, exactly one task+reply."""

    def test_two_ticks_process_one_mail_once(
        self,
        mail_stack: tuple[EmailChannelBackend, _ImapLiteServer, _SmtpLiteServer],
        daemon: RecordingDaemon,
    ) -> None:
        """Tick 1 handles, replies, and acks; tick 2 redelivers nothing."""
        backend, imap, smtp = mail_stack
        imap.add_message(_RAW_MAIL, "<need-help-1@example.com>")
        runner = ChannelRunner(backend=backend, channel_name="", agent_name="RR Email Test")

        assert runner.run_once() == 1
        assert len(_launched_prompts(daemon)) == 1
        assert "please help" in _launched_prompts(daemon)[0]
        assert len(smtp.deliveries) == 1
        reply = smtp.deliveries[0].decode("utf-8", errors="replace")
        assert _REPLY_TEXT in reply
        assert "In-Reply-To: <need-help-1@example.com>" in reply
        # The ack marked the mail read on the server.
        assert imap.unseen() == []
        assert any("\\Seen" in flags for flags in imap.stored_flags)

        # Pre-fix, the still-unread mail was handled again every tick.
        assert runner.run_once() == 0
        assert len(_launched_prompts(daemon)) == 1
        assert len(smtp.deliveries) == 1

    def test_new_mail_after_ack_is_still_picked_up(
        self,
        mail_stack: tuple[EmailChannelBackend, _ImapLiteServer, _SmtpLiteServer],
        daemon: RecordingDaemon,
    ) -> None:
        """Acking one mail must not suppress genuinely new mail."""
        backend, imap, smtp = mail_stack
        imap.add_message(_RAW_MAIL, "<need-help-1@example.com>")
        runner = ChannelRunner(backend=backend, channel_name="", agent_name="RR Email Test")
        assert runner.run_once() == 1
        second = _RAW_MAIL.replace(b"need-help-1", b"need-help-2")
        imap.add_message(second, "<need-help-2@example.com>")
        assert runner.run_once() == 1
        assert len(_launched_prompts(daemon)) == 2
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
            backend.ack_message("INBOX", {"ts": "1", "thread_ts": "<gone@example.com>"})
        assert "Could not mark email" in caplog.text
        assert imap.unseen() == []


def _wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Poll *predicate* until true or *timeout* elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _recv_line(conn: socket.socket) -> bytes:
    """Read one CRLF-terminated line from *conn*."""
    line = b""
    while not line.endswith(b"\r\n"):
        chunk = conn.recv(1)
        if not chunk:
            break
        line += chunk
    return line


class TestRunnerAckDispatch:
    """ChannelRunner._ack_message with a backend that has no ack hook.

    ``IRCChannelBackend`` is a real cursor-free backend without
    ``ack_message``; it is driven against a loopback IRC listener.

    The runner's other branch — an ``ack_message`` that raises, logged
    as ``ack_message failed`` — is unreachable without a test double:
    the only product ``ack_message`` (email) routes every IMAP failure
    through ``mark_email_read``, which converts any exception into an
    ``ok:false`` result that ``ack_message`` logs as ``Could not mark
    email`` (covered by ``test_ack_unknown_message_logs_warning``).
    """

    def test_backend_without_hook_is_untouched(self, daemon: RecordingDaemon) -> None:
        """A backend without ``ack_message`` processes and replies normally."""
        with socket.create_server(("127.0.0.1", 0)) as listener:
            listener.settimeout(5.0)
            irc_sea._config.save(
                {"server": "127.0.0.1", "port": str(listener.getsockname()[1]), "nick": "bot"}
            )
            backend = irc_sea.IRCChannelBackend()
            try:
                # Deliver one PRIVMSG on a first connection; the reader
                # thread queues it for the next poll.
                assert backend.connect() is True, backend.connection_info
                first, _ = listener.accept()
                with first:
                    first.settimeout(5.0)
                    first.sendall(b":alice!a@example.com PRIVMSG #general :hi\r\n")
                    assert _wait_for(lambda: not backend._message_queue.empty())
                    runner = ChannelRunner(backend=backend, channel_name="#general", agent_name="t")
                    # run_once() reconnects, joins, polls the queued message,
                    # launches the task and replies on the new connection.
                    assert runner.run_once() == 1
                assert len(_launched_prompts(daemon)) == 1
                assert "hi" in _launched_prompts(daemon)[0]
                second, _ = listener.accept()
                with second:
                    second.settimeout(5.0)
                    lines = [_recv_line(second) for _ in range(4)]
                assert lines[0] == b"NICK bot\r\n"
                assert lines[2] == b"JOIN #general\r\n"
                assert lines[3].startswith(b"PRIVMSG #general :")
                assert _REPLY_TEXT.encode() in lines[3]
            finally:
                backend.disconnect()
