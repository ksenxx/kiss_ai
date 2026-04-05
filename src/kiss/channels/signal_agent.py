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

from kiss.agents.sorcar.sorcar_agent import (
    _build_arg_parser,
    _resolve_task,
    cli_ask_user_question,
    cli_wait_for_user,
)
from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.channels._backend_utils import wait_for_matching_message

_SIGNAL_DIR = Path.home() / ".kiss" / "channels" / "signal"


def _config_path() -> Path:
    """Return the path to the stored Signal config file."""
    return _SIGNAL_DIR / "config.json"


def _load_config() -> dict[str, str] | None:
    """Load stored Signal config from disk."""
    path = _config_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and data.get("phone_number"):  # pragma: no branch
            return {
                "phone_number": data["phone_number"],
                "signal_cli_path": data.get("signal_cli_path", "signal-cli"),
            }
        return None
    except (json.JSONDecodeError, OSError):
        return None


def _save_config(phone_number: str, signal_cli_path: str = "signal-cli") -> None:
    """Save Signal config to disk with restricted permissions."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "phone_number": phone_number.strip(),
        "signal_cli_path": signal_cli_path.strip(),
    }, indent=2))
    if sys.platform != "win32":  # pragma: no branch
        path.chmod(0o600)


def _clear_config() -> None:
    """Delete the stored Signal config."""
    path = _config_path()
    if path.exists():  # pragma: no branch
        path.unlink()


class SignalChannelBackend:
    """ChannelBackend implementation for Signal via signal-cli."""

    def __init__(self) -> None:
        self._phone_number: str = ""
        self._signal_cli: str = "signal-cli"
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Load Signal config."""
        cfg = _load_config()
        if not cfg:  # pragma: no branch
            self._connection_info = "No Signal config found."
            return False
        self._phone_number = cfg["phone_number"]
        self._signal_cli = cfg.get("signal_cli_path", "signal-cli")
        self._connection_info = f"Signal configured for {self._phone_number}"
        return True

    @property
    def connection_info(self) -> str:
        """Human-readable connection status string."""
        return self._connection_info

    def find_channel(self, name: str) -> str | None:
        """Return phone number as channel ID."""
        return name if name else None

    def find_user(self, username: str) -> str | None:
        """Return username as user ID."""
        return username if username else None

    def join_channel(self, channel_id: str) -> None:
        """No-op for Signal."""

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
                        messages.append({
                            "ts": str(data.get("envelope", {}).get("timestamp", "")),
                            "user": sender,
                            "text": msg["message"],
                        })
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

    def disconnect(self) -> None:
        """Release backend resources before stop or reconnect."""

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Check if a message is from the bot."""
        return bool(msg.get("user", "") == self._phone_number)

    def strip_bot_mention(self, text: str) -> str:
        """Remove bot mentions from text."""
        return text

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
            stdout, _ = self._run_cli(
                "receive", "--output=json", "--timeout", str(timeout)
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
            _, stderr = self._run_cli(
                "send", "-m", message, "-a", file_path, recipient
            )
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

    def get_tool_methods(self) -> list:
        """Return list of bound tool methods for use by the LLM agent."""
        non_tool = frozenset({
            "connect", "find_channel", "find_user", "join_channel",
            "poll_messages", "send_message", "wait_for_reply",
            "is_from_bot", "strip_bot_mention", "disconnect", "get_tool_methods",
        })
        return [
            getattr(self, name)
            for name in sorted(dir(self))
            if not name.startswith("_")
            and name not in non_tool
            and callable(getattr(self, name))
        ]


class SignalAgent(StatefulSorcarAgent):
    """StatefulSorcarAgent extended with Signal CLI tools."""

    def __init__(self) -> None:
        super().__init__("Signal Agent")
        self._backend = SignalChannelBackend()
        cfg = _load_config()
        if cfg:  # pragma: no branch
            self._backend._phone_number = cfg["phone_number"]
            self._backend._signal_cli = cfg.get("signal_cli_path", "signal-cli")

    def _get_tools(self) -> list:
        """Return SorcarAgent tools + Signal auth tools + Signal API tools."""
        tools = super()._get_tools()
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
                    capture_output=True, text=True, timeout=10
                )
                return json.dumps({
                    "ok": True,
                    "phone_number": agent._backend._phone_number,
                    "signal_cli_version": result.stdout.strip(),
                })
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_signal(
            phone_number: str, signal_cli_path: str = "signal-cli"
        ) -> str:
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
            _save_config(phone_number, signal_cli_path)
            agent._backend._phone_number = phone_number
            agent._backend._signal_cli = signal_cli_path
            return json.dumps({
                "ok": True,
                "message": "Signal configured.",
                "phone_number": phone_number,
            })

        def clear_signal_auth() -> str:
            """Clear the stored Signal configuration.

            Returns:
                Status message.
            """
            _clear_config()
            agent._backend._phone_number = ""
            return "Signal configuration cleared."

        tools.extend([check_signal_auth, authenticate_signal, clear_signal_auth])

        if agent._backend._phone_number:  # pragma: no branch
            tools.extend(agent._backend.get_tool_methods())

        return tools


def main() -> None:
    """Run the SignalAgent from the command line with chat persistence."""
    import sys
    import time as time_mod

    if len(sys.argv) <= 1:  # pragma: no branch
        print("Usage: kiss-signal [-m MODEL] [-t TASK] [-n] [--daemon]")
        sys.exit(1)

    parser = _build_arg_parser()
    parser.add_argument("-n", "--new", action="store_true", help="Start a new chat session")
    parser.add_argument("--daemon", action="store_true", help="Run as background daemon")
    parser.add_argument("--daemon-channel", default="", help="Phone number to monitor")
    parser.add_argument("--allow-users", default="", help="Comma-separated phone numbers to allow")
    args = parser.parse_args()

    if args.daemon:  # pragma: no branch
        from kiss.channels.background_agent import ChannelDaemon

        backend = SignalChannelBackend()
        cfg = _load_config()
        if not cfg:  # pragma: no branch
            print("Not configured. Run: kiss-signal -t 'authenticate'")
            sys.exit(1)
        backend._phone_number = cfg["phone_number"]
        backend._signal_cli = cfg.get("signal_cli_path", "signal-cli")
        allow_users = [u.strip() for u in args.allow_users.split(",") if u.strip()] or None
        daemon = ChannelDaemon(
            backend=backend,
            channel_name=args.daemon_channel,
            agent_name="Signal Background Agent",
            extra_tools=backend.get_tool_methods(),
            model_name=args.model_name,
            max_budget=args.max_budget,
            work_dir=args.work_dir or str(Path.home() / ".kiss" / "daemon_work"),
            allow_users=allow_users,
        )
        print("Starting Signal daemon... (Ctrl+C to stop)")
        try:
            daemon.run()
        except KeyboardInterrupt:
            print("Daemon stopped.")
        return

    agent = SignalAgent()
    task_description = _resolve_task(args)
    work_dir = args.work_dir or str(Path(".").resolve())
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    if args.new:  # pragma: no branch
        agent.new_chat()
    else:
        agent.resume_chat(task_description)

    model_config: dict[str, Any] = {}
    if args.endpoint:  # pragma: no branch
        model_config["base_url"] = args.endpoint

    run_kwargs: dict[str, Any] = {
        "prompt_template": task_description,
        "model_name": args.model_name,
        "max_budget": args.max_budget,
        "model_config": model_config,
        "work_dir": work_dir,
        "verbose": args.verbose,
        "wait_for_user_callback": cli_wait_for_user,
        "ask_user_question_callback": cli_ask_user_question,
    }

    start_time = time_mod.time()
    agent.run(**run_kwargs)
    elapsed = time_mod.time() - start_time

    print(f"Time: {elapsed:.1f}s")
    print(f"Cost: ${agent.budget_used:.4f}")
    print(f"Total tokens: {agent.total_tokens_used}")


if __name__ == "__main__":
    main()
