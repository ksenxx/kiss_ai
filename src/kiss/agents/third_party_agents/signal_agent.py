# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Signal Agent — channel agent with Signal CLI tools.

Uses signal-cli subprocess to send/receive Signal messages. Stores
configuration in ``~/.kiss/third_party_agents/signal/config.json``.

Connecting works like linking Signal Desktop: ``authenticate_signal()``
without a phone number runs ``signal-cli link`` and renders the linking
QR code; the user scans it from Signal on their phone (Settings > Linked
devices) and ``finish_signal_auth()`` records the linked account.  No
registration code or password is ever typed into the agent.

Usage::

    agent = SignalAgent()
    agent.run(prompt_template="Send 'Hello!' to +14155238886")
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from html import escape
from pathlib import Path
from typing import Any

from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
    write_private_file,
)
from kiss.agents.third_party_agents._device_auth import ConsentSession
from kiss.core.processes import kill_process_group, popen_process_group

logger = logging.getLogger(__name__)

_SIGNAL_DIR = Path.home() / ".kiss" / "third_party_agents" / "signal"
_config = ChannelConfig(_SIGNAL_DIR, ("phone_number",))
_LINK_DEVICE_NAME = "KISS Sorcar"
# ``signal-cli link`` prints the provisioning URI first; once the phone
# has scanned it the command exits, newer versions announcing the account.
_LINK_URI_RE = re.compile(r"(sgnl://linkdevice\S+|tsdevice:/\S+)")
_ASSOCIATED_RE = re.compile(r"Associated with:\s*(\+\d{6,15})")
_ACCOUNT_RE = re.compile(r"Number:\s*(\+\d{6,15})")
# The provisioning URI is honored by the Signal servers for a few
# minutes only; the session stops waiting after this.
_LINK_LIFETIME = 10 * 60.0


def _qr_rows(text: str) -> list[list[bool]]:
    """Render *text* as a QR module matrix (True = dark module).

    Args:
        text: The payload to encode.

    Returns:
        Square matrix of modules including a quiet zone.
    """
    import qrcode
    from qrcode.constants import ERROR_CORRECT_M

    # A four-module quiet zone on every side, as the QR standard requires.
    qr = qrcode.QRCode(error_correction=ERROR_CORRECT_M, border=4)
    qr.add_data(text)
    qr.make(fit=True)
    return [[bool(cell) for cell in row] for row in qr.get_matrix()]


def _qr_text(rows: list[list[bool]]) -> str:
    """Render a module matrix as Unicode half-block text (dark = block).

    Args:
        rows: The QR module matrix.

    Returns:
        One line per two module rows, readable in any monospace font.
    """
    blocks = {
        (False, False): " ",
        (True, False): "\u2580",
        (False, True): "\u2584",
        (True, True): "\u2588",
    }
    if len(rows) % 2:
        rows = [*rows, [False] * len(rows[0])]
    lines = []
    for top, bottom in zip(rows[0::2], rows[1::2], strict=True):
        lines.append("".join(blocks[(a, b)] for a, b in zip(top, bottom, strict=True)))
    return "\n".join(lines)


def _qr_svg(rows: list[list[bool]], scale: int = 8) -> str:
    """Render a module matrix as an SVG document (black on white).

    Args:
        rows: The QR module matrix.
        scale: Pixels per module.

    Returns:
        The SVG source.
    """
    size = len(rows) * scale
    cells = "".join(
        f'<rect x="{x * scale}" y="{y * scale}" width="{scale}" height="{scale}"/>'
        for y, row in enumerate(rows)
        for x, dark in enumerate(row)
        if dark
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" shape-rendering="crispEdges">'
        f'<rect width="{size}" height="{size}" fill="#fff"/><g fill="#000">{cells}</g></svg>'
    )


def _write_link_page(uri: str) -> Path:
    """Write the linking QR page and return its path.

    The page carries the QR as an inline SVG (black on white, the
    polarity a phone camera expects) and the provisioning URI itself.
    It lives next to the channel config (so ``KISS_HOME`` isolation
    applies), is created 0600 since the URI is a one-time linking
    secret, and is deleted again when the session ends.

    Args:
        uri: The ``sgnl://linkdevice`` provisioning URI.

    Returns:
        Path of the written HTML file.
    """
    rows = _qr_rows(uri)
    path = _config.path.parent / "link-qr.html"
    write_private_file(
        path,
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Link Signal</title></head>"
        "<body style='font-family:sans-serif;text-align:center;padding-top:24px'>"
        "<h2>Link this computer to Signal</h2>"
        "<p>On your phone: Signal &rarr; Settings &rarr; Linked devices "
        "&rarr; Link new device &mdash; then scan this code:</p>"
        f"{_qr_svg(rows)}"
        f"<p style='font-size:12px;color:#555;word-break:break-all'>{escape(uri)}</p>"
        "</body></html>",
    )
    return path


def _cli_executable(signal_cli: str) -> str:
    """Resolve the signal-cli command the way a shell would.

    Windows distributes signal-cli as a ``signal-cli.bat`` launcher, and
    ``subprocess`` alone only ever appends ``.exe`` to a bare name, so the
    launcher is found through ``shutil.which`` (which honours ``PATHEXT``).
    A name that resolves nowhere is returned unchanged so the caller's
    error still names exactly what the user configured.

    Args:
        signal_cli: Bare command name or path of the signal-cli binary.

    Returns:
        The path to run, or *signal_cli* itself when nothing resolves.
    """
    return shutil.which(signal_cli) or signal_cli


class SignalLinkSession(ConsentSession):
    """A pending ``signal-cli link`` waiting for the phone to scan the QR."""

    def __init__(self, signal_cli: str) -> None:
        """Start ``signal-cli link`` and capture its provisioning URI.

        Args:
            signal_cli: Path of the signal-cli binary.

        Raises:
            RuntimeError: When signal-cli cannot be started or prints no
                provisioning URI within a minute.
        """
        try:
            # stderr is merged into stdout: both are drained by one reader
            # thread, so a chatty signal-cli can never fill a pipe and
            # stall (its log output goes to stderr).  signal-cli is a
            # launcher script in front of java, so it runs in its own
            # process group and is ended as a group (see _terminate).
            self._process = popen_process_group(
                [_cli_executable(signal_cli), "link", "-n", _LINK_DEVICE_NAME],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as e:
            raise RuntimeError(f"could not start {signal_cli!r}: {e}") from None
        self._output: list[str] = []
        self._output_done = threading.Event()
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()
        deadline = time.monotonic() + 60.0
        while (
            time.monotonic() < deadline and not self._find_uri() and not self._output_done.is_set()
        ):
            self._output_done.wait(0.1)
        uri = self._find_uri()
        if not uri:
            self._reap()
            raise RuntimeError(
                f"signal-cli link failed: {self._last_message('no provisioning URI')}"
            )
        super().__init__("signal", _LINK_LIFETIME, 1.0)
        self.signal_cli = signal_cli
        self.verification_uri = uri
        self.page = _write_link_page(uri)
        self.qr_text = _qr_text(_qr_rows(uri))

    def _read_output(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            self._output.append(line.rstrip("\n"))
        self._output_done.set()

    def _find_uri(self) -> str:
        for line in list(self._output):
            match = _LINK_URI_RE.search(line)
            if match:
                return match.group(1)
        return ""

    def _last_message(self, default: str) -> str:
        """Return the last non-URI output line (signal-cli's error text)."""
        lines = [ln.strip() for ln in self._output if ln.strip() and not _LINK_URI_RE.search(ln)]
        return (lines[-1] if lines else default)[:300]

    def _terminate(self, sig: int = signal.SIGTERM) -> None:
        """Signal the whole ``signal-cli link`` process group, if still alive.

        Signalling only the launcher would leave the java process it
        started running (and holding the output pipe open); on Windows
        that is the only way to end ``signal-cli.bat`` at all.

        Args:
            sig: POSIX signal to send; Windows always ends the tree.
        """
        if self._process.poll() is None:
            try:
                kill_process_group(self._process.pid, sig)
            except ProcessLookupError:
                pass
            except OSError:
                self._process.kill()

    def _reap(self) -> None:
        """End the process if still running, wait for it, and release its pipe."""
        if self._process.poll() is None:
            self._terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                # Windows has no SIGKILL; there the group is force-ended anyway.
                self._terminate(getattr(signal, "SIGKILL", signal.SIGTERM))
                self._process.wait(timeout=5.0)
        self._reader.join(timeout=5.0)
        if self._process.stdout is not None:
            self._process.stdout.close()

    def _run(self) -> None:
        """Poll like any consent session, then always reap the process."""
        try:
            super()._run()
        finally:
            self._reap()
            page = getattr(self, "page", None)
            if page is not None:
                # The provisioning URI is single-use; do not leave it on disk.
                Path(page).unlink(missing_ok=True)

    def _poll_once(self) -> dict[str, Any] | None:
        """Report the linked account once ``signal-cli link`` has exited."""
        code = self._process.poll()
        if code is None:
            return None
        self._output_done.wait(5.0)
        if code != 0:
            raise RuntimeError(f"linking failed: {self._last_message(f'exit code {code}')}")
        match = _ASSOCIATED_RE.search("\n".join(self._output))
        number = match.group(1) if match else _single_account(self.signal_cli)
        if not number:
            raise RuntimeError(
                "linked, but signal-cli did not report the account number; run "
                "authenticate_signal(phone_number=...) with the linked number"
            )
        return {"phone_number": number}

    def cancel(self) -> None:
        """Stop waiting and end the ``signal-cli link`` process.

        The poll thread reaps the process on its way out.
        """
        super().cancel()
        self._terminate()


def _single_account(signal_cli: str) -> str:
    """Return the only account signal-cli knows, or ``""``.

    Args:
        signal_cli: Path of the signal-cli binary.

    Returns:
        The E.164 number when exactly one account is registered/linked.
    """
    try:
        result = subprocess.run(
            [_cli_executable(signal_cli), "listAccounts"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    numbers = _ACCOUNT_RE.findall(result.stdout)
    return numbers[0] if len(numbers) == 1 else ""


def _link_instructions(session: SignalLinkSession) -> str:
    """Build the agent-facing hand-off text for a started link session.

    The QR code is included as a fenced code block so that Markdown
    renderers keep it monospaced and its rows intact.
    """
    return (
        "Connect Signal the way Signal Desktop links: the USER scans a QR code "
        "with their phone; you only relay it. Never ask for the user's phone "
        "number, PIN, or a verification code. Steps: 1) Call ask_user_question() "
        "showing the user the QR code below EXACTLY as given, inside the same "
        "fenced code block (it must stay monospaced with every row intact; if the "
        "chat uses a dark theme and the phone cannot read it, tell the user to open "
        f"{session.page} on this computer, which shows it black on white). Ask them "
        "to open Signal on their phone > Settings > Linked devices > Link new "
        "device, scan the code, and reply here when done (the code is valid for "
        f"about {max(session.expires_in // 60, 1)} minutes). 2) Call "
        "finish_signal_auth(); if it returns 'pending', wait a few seconds and call "
        "it again. Nothing has to be pasted back.\n\n```text\n" + session.qr_text + "\n```"
    )


class SignalChannelBackend(ToolMethodBackend):
    """Channel backend for Signal via signal-cli."""

    # Cap on parked foreign-sender envelopes kept in the channel state;
    # beyond it the OLDEST foreign envelopes are dropped (with a log).
    _QUEUE_MAX = 500

    def __init__(self) -> None:
        self._phone_number: str = ""
        self._signal_cli: str = "signal-cli"
        self._connection_info: str = ""
        self._channel_state: dict[str, Any] | None = None

    def bind_channel_state(self, state: dict[str, Any]) -> None:
        """Receive the channel runner's state dict for envelope parking.

        Called by ``ChannelRunner._run_tick`` right before
        :meth:`poll_messages`.  ``signal-cli receive`` is destructive
        (the server acks every delivered envelope), so envelopes that
        cannot be returned this tick — foreign senders, overflow past
        the poll limit — are parked in ``state["pending_envelopes"]``,
        which the runner persists via ``save_channel_state``.

        Args:
            state: The runner's loaded channel-state dict.
        """
        self._channel_state = state

    def connect(self) -> bool:
        """Load Signal config."""
        cfg = _config.load()
        if not cfg:  # pragma: no branch
            self._connection_info = "No Signal config found."
            return False
        self._phone_number = cfg["phone_number"]
        self._signal_cli = cfg.get("signal_cli_path", "signal-cli")
        self._connection_info = f"Signal configured for {self._phone_number}"
        return True

    def _run_cli(self, *args: str, timeout: int = 30) -> tuple[str, str, int]:
        """Run signal-cli command and return (stdout, stderr, returncode)."""
        cmd = [_cli_executable(self._signal_cli), "-u", self._phone_number, *args]
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", timeout=timeout
        )
        return result.stdout, result.stderr, result.returncode

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Receive pending Signal messages via signal-cli.

        ``signal-cli receive`` is a DESTRUCTIVE read: the server acks
        every delivered envelope, so any envelope this method consumes
        but does not surface would be lost permanently.  At the same
        time the channel runner replies to ``channel_id`` (the
        configured contact), so surfacing another sender's envelope
        would leak the reply to the wrong contact.  Therefore:

        - With no *channel_id*, ALL received text messages are
          returned (legacy behaviour; the runner's allow-list decides
          which ones to act on).
        - With a *channel_id*, only that sender's envelopes are
          returned (at most *limit* per tick, queued ones first).  The
          rest of the consumed envelopes are parked in the channel
          state bound via :meth:`bind_channel_state`: matching overflow
          beyond *limit* is delivered on subsequent ticks, and
          foreign-sender envelopes are kept (capped at ``_QUEUE_MAX``,
          oldest dropped with a log) so they can be delivered if the
          contact is later monitored.
        - With a *channel_id* but NO bound state (a custom stateless
          runner), foreign-sender envelopes are DROPPED with a warning
          log — cross-contact reply leakage must be impossible — and
          *limit* is ignored so matching envelopes are never truncated.
        """
        try:
            stdout, _, _ = self._run_cli("receive", "--output=json", "--timeout", "5")
        except Exception:
            return [], oldest
        received: list[dict[str, Any]] = []
        for line in stdout.strip().split("\n"):  # pragma: no branch
            if not line.strip():  # pragma: no branch
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = data.get("envelope", {}).get("dataMessage", {})
            sender = data.get("envelope", {}).get("source", "")
            if not msg.get("message"):  # pragma: no branch
                continue
            received.append(
                {
                    "ts": str(data.get("envelope", {}).get("timestamp", "")),
                    "user": sender,
                    "text": msg["message"],
                }
            )
        if not channel_id:
            return received, oldest
        state = self._channel_state
        queued: list[dict[str, Any]] = []
        if state is not None:
            queued = [e for e in state.get("pending_envelopes", []) if isinstance(e, dict)]
        matching = [e for e in queued if e.get("user") == channel_id]
        matching += [e for e in received if e["user"] == channel_id]
        if state is None:
            dropped = [e for e in received if e["user"] != channel_id]
            if dropped:
                logger.warning(
                    "Dropping %d Signal envelope(s) from sender(s) other than "
                    "the monitored contact %s (no channel state to park them in): %s",
                    len(dropped),
                    channel_id,
                    sorted({str(e["user"]) for e in dropped}),
                )
            return matching, oldest
        overflow = matching[max(limit, 0) :] if limit > 0 else []
        delivered = matching[: max(limit, 0)] if limit > 0 else matching
        foreign = [e for e in queued if e.get("user") != channel_id]
        foreign += [
            e for e in received if e["user"] != channel_id and e["user"] != self._phone_number
        ]
        if len(foreign) > self._QUEUE_MAX:
            logger.warning(
                "Dropping %d oldest foreign-sender Signal envelope(s): "
                "the pending-envelope queue is capped at %d",
                len(foreign) - self._QUEUE_MAX,
                self._QUEUE_MAX,
            )
            foreign = foreign[-self._QUEUE_MAX :]
        state["pending_envelopes"] = overflow + foreign
        return delivered, oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a Signal message.

        Raises:
            RuntimeError: If signal-cli exits nonzero or reports an error on stderr.
        """
        _, stderr, returncode = self._run_cli("send", "-m", text, channel_id)
        if returncode != 0 or (stderr and "error" in stderr.lower()):
            raise RuntimeError(f"signal-cli send failed (exit {returncode}): {stderr.strip()}")

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Check if a message is from the bot."""
        return bool(msg.get("user", "") == self._phone_number)

    def send_signal_message(self, recipient: str, message: str) -> str:
        """Send a Signal text message.

        Args:
            recipient: Recipient phone number in E.164 format.
            message: Message text to send.

        Returns:
            JSON string with ok status.
        """
        try:
            self.send_message(recipient, message)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def receive_messages(self, timeout: int = 5) -> str:
        """Receive pending Signal messages.

        Args:
            timeout: Seconds to wait for messages. Default: 5.

        Returns:
            JSON string with list of received messages.
        """
        try:
            stdout, _, _ = self._run_cli(
                "receive",
                "--output=json",
                "--timeout",
                str(timeout),
                timeout=max(30, timeout + 10),
            )
            messages = []
            for line in stdout.strip().split("\n"):  # pragma: no branch
                if not line.strip():  # pragma: no branch
                    continue
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            return json.dumps({"ok": True, "messages": messages}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_attachment(self, recipient: str, message: str, file_path: str) -> str:
        """Send a Signal message with an attachment.

        Args:
            recipient: Recipient phone number.
            message: Message text.
            file_path: Local path to the file to attach.

        Returns:
            JSON string with ok status.
        """
        try:
            _, stderr, returncode = self._run_cli("send", "-m", message, "-a", file_path, recipient)
            if returncode != 0 or (stderr and "error" in stderr.lower()):
                return json.dumps({"ok": False, "error": stderr.strip()})
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_contacts(self) -> str:
        """List Signal contacts.

        Returns:
            JSON string with contact list.
        """
        try:
            stdout, _, _ = self._run_cli("listContacts", "--output=json")
            try:
                contacts = json.loads(stdout)
            except json.JSONDecodeError:
                contacts = []
            return json.dumps({"ok": True, "contacts": contacts}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_groups(self) -> str:
        """List Signal groups.

        Returns:
            JSON string with group list.
        """
        try:
            stdout, _, _ = self._run_cli("listGroups", "--output=json")
            try:
                groups = json.loads(stdout)
            except json.JSONDecodeError:
                groups = []
            return json.dumps({"ok": True, "groups": groups}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class SignalAgent(BaseChannelAgent):
    """Channel agent with Signal CLI tools."""

    channel_system_prompt = (
        "## Signal Authentication\n"
        "1. Call check_signal_auth() first; if it reports ok, use the tools and never "
        "re-run authentication over a working configuration.\n"
        "2. To connect, call authenticate_signal() with no phone number. It runs "
        "`signal-cli link` and returns status 'consent_required' with a QR code "
        "(monospace text, also written to link-qr.html) and its sgnl:// URI.\n"
        "3. The USER completes the linking, exactly like linking Signal Desktop: call "
        "ask_user_question() showing the QR code, telling them to open Signal on their "
        "phone > Settings > Linked devices > Link new device, scan it, and reply when "
        "done. Never ask for the user's phone number, Signal PIN, or an SMS "
        "verification code, and never run `signal-cli register` or `verify`.\n"
        "4. Then call finish_signal_auth(); if it returns 'pending', wait a few seconds "
        "and call it again. Confirm the result with check_signal_auth(). Only when "
        "signal-cli is already registered or linked on this machine may you instead "
        "record its number with authenticate_signal(phone_number=...)."
    )

    def __init__(self) -> None:
        super().__init__("Signal Agent")
        self._backend = SignalChannelBackend()
        cfg = _config.load()
        if cfg:  # pragma: no branch
            self._backend._phone_number = cfg["phone_number"]
            self._backend._signal_cli = cfg.get("signal_cli_path", "signal-cli")

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._phone_number)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_signal_auth() -> str:
            """Check if Signal is configured and signal-cli is available.

            Returns:
                Configuration status or instructions.
            """
            if not agent._backend._phone_number:  # pragma: no branch
                return (
                    "Not configured for Signal. Call authenticate_signal() with no phone "
                    "number to link this computer the way Signal Desktop links: it runs "
                    "`signal-cli link` and returns a QR code the user scans with Signal "
                    "on their phone (Settings > Linked devices > Link new device); then "
                    "call finish_signal_auth(). Never ask for the user's PIN or a "
                    "verification code. Requires signal-cli "
                    "(https://github.com/AsamK/signal-cli) to be installed; if it is "
                    "already registered or linked here, record its number with "
                    "authenticate_signal(phone_number=...)."
                )
            try:
                result = subprocess.run(
                    [_cli_executable(agent._backend._signal_cli), "--version"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=10,
                )
                return json.dumps(
                    {
                        "ok": True,
                        "phone_number": agent._backend._phone_number,
                        "signal_cli_version": result.stdout.strip(),
                    }
                )
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_signal(phone_number: str = "", signal_cli_path: str = "signal-cli") -> str:
            """Link this computer to the user's Signal account, or record a number.

            Without ``phone_number`` this runs ``signal-cli link`` and
            returns a ``consent_required`` answer carrying the linking QR
            code (as monospace text, and written to ``link-qr.html``):
            show it to the user (ask_user_question) to scan from Signal on
            their phone, then call finish_signal_auth().  With
            ``phone_number`` the already registered/linked signal-cli
            account is recorded directly.

            Args:
                phone_number: Optional Signal number in E.164 format of an
                    account signal-cli already has.
                signal_cli_path: Path to signal-cli binary. Default: "signal-cli".

            Returns:
                A consent_required JSON answer, a configuration result, or
                an error message.
            """
            if phone_number and not phone_number.strip():
                return "phone_number cannot be empty."
            signal_cli = signal_cli_path.strip() or "signal-cli"
            if phone_number.strip():
                ConsentSession.cancel_active("signal")
                _config.save({"phone_number": phone_number.strip(), "signal_cli_path": signal_cli})
                agent._backend._phone_number = phone_number.strip()
                agent._backend._signal_cli = signal_cli
                return json.dumps(
                    {
                        "ok": True,
                        "message": "Signal configured.",
                        "phone_number": phone_number.strip(),
                    }
                )
            try:
                session = SignalLinkSession(signal_cli)
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})
            session.register()
            return json.dumps(
                {
                    "ok": True,
                    "status": "consent_required",
                    "verification_uri": session.verification_uri,
                    "qr_page": str(session.page),
                    "qr_text": session.qr_text,
                    "expires_in": session.expires_in,
                    "instructions": _link_instructions(session),
                }
            )

        def finish_signal_auth() -> str:
            """Complete a linking started by authenticate_signal().

            Call after the user reports that they scanned the QR code;
            the linked account signal-cli reports is stored.

            Returns:
                The configuration result, a pending status while the phone
                has not scanned the code yet, or an error message.
            """
            session, status = ConsentSession.finish("signal")
            if status == "pending":
                return json.dumps(
                    {
                        "ok": False,
                        "status": "pending",
                        "error": "The phone has not linked yet; ask the user to scan the "
                        "code, then call this tool again.",
                    }
                )
            if not isinstance(session, SignalLinkSession) or session.result is None:
                return json.dumps({"ok": False, "error": f"Signal linking failed: {status}"})
            number = str(session.result["phone_number"])
            _config.save({"phone_number": number, "signal_cli_path": session.signal_cli})
            agent._backend._phone_number = number
            agent._backend._signal_cli = session.signal_cli
            return json.dumps({"ok": True, "message": "Signal linked.", "phone_number": number})

        def clear_signal_auth() -> str:
            """Clear the stored Signal configuration.

            Returns:
                Status message.
            """
            ConsentSession.cancel_active("signal")
            _config.clear()
            agent._backend._phone_number = ""
            return "Signal configuration cleared."

        return [check_signal_auth, authenticate_signal, finish_signal_auth, clear_signal_auth]


def _make_backend() -> SignalChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = SignalChannelBackend()
    cfg = _config.load()
    if not cfg:  # pragma: no branch
        print("Not configured. Run: kiss-signal -t 'authenticate'")
        sys.exit(1)
    backend._phone_number = cfg["phone_number"]
    backend._signal_cli = cfg.get("signal_cli_path", "signal-cli")
    return backend


def main() -> None:
    """Run the SignalAgent from the command line with chat persistence."""
    channel_main(
        SignalAgent,
        "kiss-signal",
        channel_name="Signal",
        make_backend=_make_backend,
    )


def tools() -> list:
    """Return the Signal channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return SignalAgent()._get_tools()


if __name__ == "__main__":
    main()
