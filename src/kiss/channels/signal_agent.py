"""Signal Agent — StatefulSorcarAgent extension with Signal CLI tools.

Uses signal-cli subprocess to send/receive Signal messages. Stores
configuration in ``~/.kiss/channels/signal/config.json``.

Usage::

    agent = SignalAgent()
    agent.run(prompt_template="Send 'Hello!' to +14155238886")
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.channels._backend_utils import wait_for_matching_message
from kiss.channels._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

_SIGNAL_DIR = Path.home() / ".kiss" / "channels" / "signal"
_config = ChannelConfig(_SIGNAL_DIR, ("phone_number",))


class SignalChannelBackend(ToolMethodBackend):
    """Channel backend for Signal via signal-cli."""

    def __init__(self) -> None:
        self._phone_number: str = ""
        self._signal_cli: str = "signal-cli"
        self._connection_info: str = ""

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

    def _run_cli(self, *args: str) -> tuple[str, str]:
        """Run signal-cli command and return (stdout, stderr)."""
        cmd = [self._signal_cli, "-u", self._phone_number, *args]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout, result.stderr

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Receive pending Signal messages via signal-cli."""
        try:
            stdout, _ = self._run_cli("receive", "--output=json", "--timeout", "5")
            messages: list[dict[str, Any]] = []
            for line in stdout.strip().split("\n"):  # pragma: no branch
                if not line.strip():  # pragma: no branch
                    continue
                try:
                    data = json.loads(line)
                    msg = data.get("envelope", {}).get("dataMessage", {})
                    sender = data.get("envelope", {}).get("source", "")
                    if msg.get("message"):  # pragma: no branch
                        messages.append(
                            {
                                "ts": str(data.get("envelope", {}).get("timestamp", "")),
                                "user": sender,
                                "text": msg["message"],
                            }
                        )
                except json.JSONDecodeError:
                    pass
            return messages, oldest
        except Exception:
            return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a Signal message."""
        self._run_cli("send", "-m", text, channel_id)

    def wait_for_reply(
        self,
        channel_id: str,
        thread_ts: str,
        user_id: str,
        timeout_seconds: float = 300.0,
        stop_event: threading.Event | None = None,
    ) -> str | None:
        """Poll for a reply from a specific user."""
        return wait_for_matching_message(
            poll=lambda: self.poll_messages(channel_id, "")[0],
            matches=lambda msg: msg.get("user") == user_id,
            extract_text=lambda msg: str(msg.get("text", "")),
            timeout_seconds=timeout_seconds,
            stop_event=stop_event,
            poll_interval=3.0,
        )

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
            _, stderr = self._run_cli("send", "-m", message, recipient)
            if stderr and "error" in stderr.lower():  # pragma: no branch
                return json.dumps({"ok": False, "error": stderr.strip()})
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
            stdout, _ = self._run_cli("receive", "--output=json", "--timeout", str(timeout))
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
            _, stderr = self._run_cli("send", "-m", message, "-a", file_path, recipient)
            if stderr and "error" in stderr.lower():  # pragma: no branch
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
            stdout, _ = self._run_cli("listContacts", "--output=json")
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
            stdout, _ = self._run_cli("listGroups", "--output=json")
            try:
                groups = json.loads(stdout)
            except json.JSONDecodeError:
                groups = []
            return json.dumps({"ok": True, "groups": groups}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class SignalAgent(BaseChannelAgent, StatefulSorcarAgent):
    """StatefulSorcarAgent extended with Signal CLI tools."""

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
                    "Not configured for Signal. Use authenticate_signal(phone_number=...) "
                    "to configure. Requires signal-cli to be installed and registered."
                )
            try:
                result = subprocess.run(
                    [agent._backend._signal_cli, "--version"],
                    capture_output=True,
                    text=True,
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

        def authenticate_signal(phone_number: str, signal_cli_path: str = "signal-cli") -> str:
            """Configure Signal with a phone number and signal-cli path.

            Args:
                phone_number: Your Signal phone number in E.164 format.
                signal_cli_path: Path to signal-cli binary. Default: "signal-cli".

            Returns:
                Configuration result or error message.
            """
            phone_number = phone_number.strip()
            if not phone_number:  # pragma: no branch
                return "phone_number cannot be empty."
            _config.save(
                {"phone_number": phone_number.strip(), "signal_cli_path": signal_cli_path.strip()}
            )
            agent._backend._phone_number = phone_number
            agent._backend._signal_cli = signal_cli_path
            return json.dumps(
                {
                    "ok": True,
                    "message": "Signal configured.",
                    "phone_number": phone_number,
                }
            )

        def clear_signal_auth() -> str:
            """Clear the stored Signal configuration.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._phone_number = ""
            return "Signal configuration cleared."

        return [check_signal_auth, authenticate_signal, clear_signal_auth]


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


if __name__ == "__main__":
    main()
