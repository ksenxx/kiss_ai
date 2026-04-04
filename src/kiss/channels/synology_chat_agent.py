"""Synology Chat Agent — StatefulSorcarAgent extension with Synology Chat webhook API.

Provides access to Synology Chat via incoming and outgoing webhooks.
Stores config in ``~/.kiss/channels/synology/config.json``.

Usage::

    agent = SynologyChatAgent()
    agent.run(prompt_template="Send a message to the team")
"""

from __future__ import annotations

import json
import logging
import queue
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import requests

from kiss.agents.sorcar.sorcar_agent import (
    _build_arg_parser,
    _resolve_task,
    cli_ask_user_question,
    cli_wait_for_user,
)
from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.channels._backend_utils import (
    ThreadedHTTPServer,
    stop_http_server,
    wait_for_matching_message,
)

logger = logging.getLogger(__name__)

_DEFAULT_WEBHOOK_PORT = 18083

_SYNOLOGY_DIR = Path.home() / ".kiss" / "channels" / "synology"


def _config_path() -> Path:
    """Return the path to the stored Synology config file."""
    return _SYNOLOGY_DIR / "config.json"


def _load_config() -> dict[str, str] | None:
    """Load stored Synology config from disk."""
    path = _config_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and data.get("webhook_url"):  # pragma: no branch
            return {
                "webhook_url": data["webhook_url"],
                "token": data.get("token", ""),
            }
        return None
    except (json.JSONDecodeError, OSError):
        return None


def _save_config(webhook_url: str, token: str = "") -> None:
    """Save Synology config to disk with restricted permissions."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "webhook_url": webhook_url.strip(),
        "token": token.strip(),
    }, indent=2))
    if sys.platform != "win32":  # pragma: no branch
        path.chmod(0o600)


def _clear_config() -> None:
    """Delete the stored Synology config."""
    path = _config_path()
    if path.exists():  # pragma: no branch
        path.unlink()


class SynologyChatChannelBackend:
    """ChannelBackend implementation for Synology Chat webhooks.

    Sends messages via the incoming webhook URL. Receives messages
    via a webhook server (outgoing webhook from Synology Chat).
    """

    def __init__(self) -> None:
        self._webhook_url: str = ""
        self._token: str = ""
        self._message_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._webhook_server: ThreadedHTTPServer | None = None
        self._webhook_thread: threading.Thread | None = None
        self._send_lock = threading.Lock()
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Load Synology config and start webhook server."""
        cfg = _load_config()
        if not cfg:  # pragma: no branch
            self._connection_info = "No Synology Chat config found."
            return False
        self._webhook_url = cfg["webhook_url"]
        self._token = cfg.get("token", "")
        self._connection_info = "Synology Chat configured"
        if not self._start_webhook_server():  # pragma: no branch
            return False
        return True

    def _start_webhook_server(self, port: int = _DEFAULT_WEBHOOK_PORT) -> bool:
        """Start the outgoing webhook HTTP server."""
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    # Synology Chat sends URL-encoded payload
                    params = parse_qs(body.decode("utf-8"))
                    payload_str = params.get("payload", ["{}"])[0]
                    payload = json.loads(payload_str)
                    token = payload.get("token", "")
                    if not backend._token or token == backend._token:  # pragma: no branch
                        backend._message_queue.put({
                            "ts": str(payload.get("timestamp", "")),
                            "user": payload.get("user_id", ""),
                            "text": payload.get("text", ""),
                            "channel_id": payload.get("channel_id", ""),
                        })
                except Exception:
                    pass
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args: Any) -> None:  # type: ignore[override]
                pass

        self.disconnect()
        try:
            self._webhook_server = ThreadedHTTPServer(("0.0.0.0", port), Handler)
            self._webhook_thread = threading.Thread(
                target=self._webhook_server.serve_forever, daemon=True
            )
            self._webhook_thread.start()
            logger.info("Synology Chat webhook server started on port %d", port)
            return True
        except OSError as e:
            self._connection_info = f"Synology webhook bind failed: {e}"
            logger.warning("Could not start Synology webhook server: %s", e)
            self._webhook_server = None
            self._webhook_thread = None
            return False

    @property
    def connection_info(self) -> str:
        """Human-readable connection status string."""
        return self._connection_info

    def find_channel(self, name: str) -> str | None:
        """Return channel name."""
        return name if name else None

    def find_user(self, username: str) -> str | None:
        """Return username as user ID."""
        return username if username else None

    def join_channel(self, channel_id: str) -> None:
        """No-op for Synology Chat."""

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Drain the webhook message queue."""
        messages: list[dict[str, Any]] = []
        while not self._message_queue.empty() and len(messages) < limit:  # pragma: no branch
            msg = self._message_queue.get_nowait()
            if not channel_id or msg.get("channel_id") == channel_id:  # pragma: no branch
                messages.append(msg)
        return messages, oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a Synology Chat message via incoming webhook."""
        with self._send_lock:
            payload = {"text": text}
            if channel_id:  # pragma: no branch
                payload["channel_id"] = channel_id
            requests.post(self._webhook_url, params={"payload": json.dumps(payload)}, timeout=30)

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
            poll_interval=2.0,
        )

    def disconnect(self) -> None:
        """Stop the embedded webhook server and release backend resources."""
        self._webhook_server, self._webhook_thread = stop_http_server(
            self._webhook_server, self._webhook_thread
        )

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Check if message is from the bot."""
        return False

    def strip_bot_mention(self, text: str) -> str:
        """Remove bot mentions from text."""
        return text

    def post_message(self, text: str, user_ids: str = "") -> str:
        """Send a message to Synology Chat via incoming webhook.

        Args:
            text: Message text.
            user_ids: Comma-separated user IDs to send to (optional).
                If empty, sends to the default channel.

        Returns:
            JSON string with ok status.
        """
        try:
            payload: dict[str, Any] = {"text": text}
            if user_ids:  # pragma: no branch
                payload["user_ids"] = [u.strip() for u in user_ids.split(",") if u.strip()]
            with self._send_lock:
                resp = requests.post(
                    self._webhook_url,
                    params={"payload": json.dumps(payload)},
                    timeout=30,
                )
            return json.dumps({"ok": resp.status_code == 200})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_file_message(self, text: str, file_url: str) -> str:
        """Send a message with a file attachment.

        Args:
            text: Message text.
            file_url: URL of the file to attach.

        Returns:
            JSON string with ok status.
        """
        try:
            payload = {"text": text, "file_url": file_url}
            with self._send_lock:
                resp = requests.post(
                    self._webhook_url,
                    params={"payload": json.dumps(payload)},
                    timeout=30,
                )
            return json.dumps({"ok": resp.status_code == 200})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_tool_methods(self) -> list:
        """Return list of bound tool methods for use by the LLM agent."""
        non_tool = frozenset({
            "connect", "find_channel", "find_user", "join_channel",
            "poll_messages", "send_message", "wait_for_reply",
            "is_from_bot", "strip_bot_mention", "get_tool_methods",
        })
        return [
            getattr(self, name)
            for name in sorted(dir(self))
            if not name.startswith("_")
            and name not in non_tool
            and callable(getattr(self, name))
        ]


class SynologyChatAgent(StatefulSorcarAgent):
    """StatefulSorcarAgent extended with Synology Chat webhook tools."""

    def __init__(self) -> None:
        super().__init__("Synology Chat Agent")
        self._backend = SynologyChatChannelBackend()
        cfg = _load_config()
        if cfg:  # pragma: no branch
            self._backend._webhook_url = cfg["webhook_url"]
            self._backend._token = cfg.get("token", "")

    def _get_tools(self) -> list:
        """Return SorcarAgent tools + Synology auth tools + API tools."""
        tools = super()._get_tools()
        agent = self

        def check_synology_auth() -> str:
            """Check if Synology Chat is configured.

            Returns:
                Configuration status or instructions.
            """
            if not agent._backend._webhook_url:  # pragma: no branch
                return (
                    "Not configured for Synology Chat. Use authenticate_synology() to configure.\n"
                    "You need the incoming webhook URL from Synology Chat integration settings."
                )
            return json.dumps({
                "ok": True,
                "webhook_url": agent._backend._webhook_url[:50] + "...",
            })

        def authenticate_synology(webhook_url: str, token: str = "") -> str:
            """Configure Synology Chat webhook.

            Args:
                webhook_url: Synology Chat incoming webhook URL.
                token: Optional outgoing webhook token for verification.

            Returns:
                Configuration result or error message.
            """
            if not webhook_url.strip():  # pragma: no branch
                return "webhook_url cannot be empty."
            agent._backend._webhook_url = webhook_url.strip()
            agent._backend._token = token.strip()
            _save_config(webhook_url, token)
            return json.dumps({"ok": True, "message": "Synology Chat configured."})

        def clear_synology_auth() -> str:
            """Clear the stored Synology Chat configuration.

            Returns:
                Status message.
            """
            _clear_config()
            agent._backend._webhook_url = ""
            agent._backend._token = ""
            return "Synology Chat configuration cleared."

        tools.extend([check_synology_auth, authenticate_synology, clear_synology_auth])

        if agent._backend._webhook_url:  # pragma: no branch
            tools.extend(agent._backend.get_tool_methods())

        return tools


def main() -> None:
    """Run the SynologyChatAgent from the command line with chat persistence."""
    import sys
    import time as time_mod

    if len(sys.argv) <= 1:  # pragma: no branch
        print("Usage: kiss-synology [-m MODEL] [-t TASK] [-n] [--daemon]")
        sys.exit(1)

    parser = _build_arg_parser()
    parser.add_argument("-n", "--new", action="store_true", help="Start a new chat session")
    parser.add_argument("--daemon", action="store_true", help="Run as background daemon")
    parser.add_argument("--allow-users", default="", help="Comma-separated user IDs to allow")
    args = parser.parse_args()

    if args.daemon:  # pragma: no branch
        from kiss.channels.background_agent import ChannelDaemon

        backend = SynologyChatChannelBackend()
        cfg = _load_config()
        if not cfg:  # pragma: no branch
            print("Not configured. Run: kiss-synology -t 'authenticate'")
            sys.exit(1)
        backend._webhook_url = cfg["webhook_url"]
        backend._token = cfg.get("token", "")
        allow_users = [u.strip() for u in args.allow_users.split(",") if u.strip()] or None
        daemon = ChannelDaemon(
            backend=backend,
            channel_name="",
            agent_name="Synology Chat Background Agent",
            extra_tools=backend.get_tool_methods(),
            model_name=args.model_name,
            max_budget=args.max_budget,
            work_dir=args.work_dir or str(Path.home() / ".kiss" / "daemon_work"),
            poll_interval=1.0,
            allow_users=allow_users,
        )
        print("Starting Synology Chat daemon... (Ctrl+C to stop)")
        try:
            daemon.run()
        except KeyboardInterrupt:
            print("Daemon stopped.")
        return

    agent = SynologyChatAgent()
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
        "headless": args.headless,
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
