"""Zalo Agent — StatefulSorcarAgent extension with Zalo Official Account API tools.

Provides authenticated access to Zalo OA via access token. Covers both
extensions/zalo/ (OA API) and extensions/zalouser/ (personal). Stores
config in ``~/.kiss/channels/zalo/config.json``.

Usage::

    agent = ZaloAgent()
    agent.run(prompt_template="Get OA info")
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

_DEFAULT_WEBHOOK_PORT = 18082

_ZALO_DIR = Path.home() / ".kiss" / "channels" / "zalo"
_API_BASE = "https://openapi.zalo.me/v2.0/oa"


def _config_path() -> Path:
    """Return the path to the stored Zalo config file."""
    return _ZALO_DIR / "config.json"


def _load_config() -> dict[str, str] | None:
    """Load stored Zalo config from disk."""
    path = _config_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and data.get("access_token"):  # pragma: no branch
            return {
                "access_token": data["access_token"],
                "oa_id": data.get("oa_id", ""),
            }
        return None
    except (json.JSONDecodeError, OSError):
        return None


def _save_config(access_token: str, oa_id: str = "") -> None:
    """Save Zalo config to disk with restricted permissions."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "access_token": access_token.strip(),
        "oa_id": oa_id.strip(),
    }, indent=2))
    if sys.platform != "win32":  # pragma: no branch
        path.chmod(0o600)


def _clear_config() -> None:
    """Delete the stored Zalo config."""
    path = _config_path()
    if path.exists():  # pragma: no branch
        path.unlink()


class ZaloChannelBackend:
    """ChannelBackend implementation for Zalo OA API.

    Uses webhook queue pattern for receiving inbound messages.
    """

    def __init__(self) -> None:
        self._access_token: str = ""
        self._oa_id: str = ""
        self._message_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._webhook_server: ThreadedHTTPServer | None = None
        self._webhook_thread: threading.Thread | None = None
        self._connection_info: str = ""

    def _headers(self) -> dict[str, str]:
        return {"access_token": self._access_token}

    def connect(self) -> bool:
        """Load Zalo config and start webhook server."""
        cfg = _load_config()
        if not cfg:  # pragma: no branch
            self._connection_info = "No Zalo config found."
            return False
        self._access_token = cfg["access_token"]
        self._oa_id = cfg.get("oa_id", "")
        self._connection_info = "Zalo OA configured"
        if not self._start_webhook_server():  # pragma: no branch
            return False
        return True

    def _start_webhook_server(self, port: int = _DEFAULT_WEBHOOK_PORT) -> bool:
        """Start the webhook HTTP server."""
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    data = json.loads(body)
                    event_name = data.get("event_name", "")
                    if event_name == "user_send_text":  # pragma: no branch
                        sender = data.get("sender", {})
                        message = data.get("message", {})
                        backend._message_queue.put({
                            "ts": str(data.get("timestamp", "")),
                            "user": sender.get("id", ""),
                            "text": message.get("text", ""),
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
            logger.info("Zalo webhook server started on port %d", port)
            return True
        except OSError as e:
            self._connection_info = f"Zalo webhook bind failed: {e}"
            logger.warning("Could not start Zalo webhook server: %s", e)
            self._webhook_server = None
            self._webhook_thread = None
            return False

    @property
    def connection_info(self) -> str:
        """Human-readable connection status string."""
        return self._connection_info

    def find_channel(self, name: str) -> str | None:
        """Return channel name as user ID."""
        return name if name else None

    def find_user(self, username: str) -> str | None:
        """Return username as user ID."""
        return username if username else None

    def join_channel(self, channel_id: str) -> None:
        """No-op for Zalo."""

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Drain the webhook message queue."""
        messages: list[dict[str, Any]] = []
        while not self._message_queue.empty() and len(messages) < limit:  # pragma: no branch
            messages.append(self._message_queue.get_nowait())
        return messages, oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a Zalo text message."""
        self.send_text_message(channel_id, text)

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

    def send_text_message(self, to_user_id: str, text: str) -> str:
        """Send a text message to a Zalo user.

        Args:
            to_user_id: Zalo user ID.
            text: Message text.

        Returns:
            JSON string with ok status.
        """
        try:
            resp = requests.post(
                f"{_API_BASE}/message/text",
                headers=self._headers(),
                json={"recipient": {"user_id": to_user_id}, "message": {"text": text}},
                timeout=30,
            )
            data = resp.json()
            if data.get("error") == 0:  # pragma: no branch
                msg_id = data.get("data", {}).get("message_id", "")
                return json.dumps({"ok": True, "message_id": msg_id})
            return json.dumps({"ok": False, "error": data.get("message", "Unknown error")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_image_message(
        self, to_user_id: str, image_url: str, caption: str = ""
    ) -> str:
        """Send an image message to a Zalo user.

        Args:
            to_user_id: Zalo user ID.
            image_url: URL of the image to send.
            caption: Optional image caption.

        Returns:
            JSON string with ok status.
        """
        try:
            attachment: dict[str, Any] = {
                "type": "template",
                "payload": {
                    "template_type": "media",
                    "elements": [{"media_type": "image", "url": image_url}],
                },
            }
            msg: dict[str, Any] = {"attachment": attachment}
            if caption:  # pragma: no branch
                msg["text"] = caption
            resp = requests.post(
                f"{_API_BASE}/message",
                headers=self._headers(),
                json={"recipient": {"user_id": to_user_id}, "message": msg},
                timeout=30,
            )
            data = resp.json()
            return json.dumps({
                "ok": data.get("error") == 0,
                "message": data.get("message", ""),
            })
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_follower_profile(self, user_id: str) -> str:
        """Get a Zalo follower's profile.

        Args:
            user_id: Zalo user ID.

        Returns:
            JSON string with user profile.
        """
        try:
            resp = requests.get(
                f"{_API_BASE}/getprofile",
                headers=self._headers(),
                params={"user_id": user_id},
                timeout=30,
            )
            data = resp.json()
            if data.get("error") == 0:  # pragma: no branch
                return json.dumps({"ok": True, "profile": data.get("data", {})}, indent=2)[:8000]
            return json.dumps({"ok": False, "error": data.get("message", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_followers(self, offset: int = 0, count: int = 50) -> str:
        """Get followers of the Zalo OA.

        Args:
            offset: Pagination offset. Default: 0.
            count: Number of followers to return (max 50). Default: 50.

        Returns:
            JSON string with follower list.
        """
        try:
            resp = requests.get(
                f"{_API_BASE}/getfollowers",
                headers=self._headers(),
                params={"offset": offset, "count": min(count, 50)},
                timeout=30,
            )
            data = resp.json()
            if data.get("error") == 0:  # pragma: no branch
                return json.dumps({"ok": True, **data.get("data", {})}, indent=2)[:8000]
            return json.dumps({"ok": False, "error": data.get("message", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_oa_info(self) -> str:
        """Get Zalo Official Account information.

        Returns:
            JSON string with OA info (name, id, description, etc).
        """
        try:
            resp = requests.get(
                f"{_API_BASE}/getoa",
                headers=self._headers(),
                timeout=30,
            )
            data = resp.json()
            if data.get("error") == 0:  # pragma: no branch
                return json.dumps({"ok": True, "oa": data.get("data", {})}, indent=2)[:8000]
            return json.dumps({"ok": False, "error": data.get("message", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_recent_messages(self, offset: int = 0, count: int = 10) -> str:
        """Get recent messages from the OA.

        Args:
            offset: Pagination offset. Default: 0.
            count: Number of messages. Default: 10.

        Returns:
            JSON string with message list.
        """
        try:
            resp = requests.get(
                f"{_API_BASE}/listrecentchat",
                headers=self._headers(),
                params={"offset": offset, "count": count},
                timeout=30,
            )
            data = resp.json()
            if data.get("error") == 0:  # pragma: no branch
                return json.dumps(
                    {"ok": True, "conversations": data.get("data", {})}, indent=2
                )[:8000]
            return json.dumps({"ok": False, "error": data.get("message", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_conversation(
        self, user_id: str, offset: int = 0, count: int = 20
    ) -> str:
        """Get conversation history with a specific user.

        Args:
            user_id: Zalo user ID.
            offset: Pagination offset. Default: 0.
            count: Number of messages. Default: 20.

        Returns:
            JSON string with conversation messages.
        """
        try:
            resp = requests.get(
                f"{_API_BASE}/conversation",
                headers=self._headers(),
                params={"user_id": user_id, "offset": str(offset), "count": str(count)},
                timeout=30,
            )
            data = resp.json()
            if data.get("error") == 0:  # pragma: no branch
                return json.dumps({"ok": True, "messages": data.get("data", {})}, indent=2)[:8000]
            return json.dumps({"ok": False, "error": data.get("message", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def upload_image(self, file_path: str) -> str:
        """Upload an image file to Zalo.

        Args:
            file_path: Local path to the image file.

        Returns:
            JSON string with ok status and attachment_id.
        """
        try:
            with open(file_path, "rb") as f:
                resp = requests.post(
                    f"{_API_BASE}/upload/image",
                    headers=self._headers(),
                    files={"file": (Path(file_path).name, f)},
                    timeout=60,
                )
            data = resp.json()
            if data.get("error") == 0:  # pragma: no branch
                attachment_id = data.get("data", {}).get("attachment_id", "")
                return json.dumps({"ok": True, "attachment_id": attachment_id})
            return json.dumps({"ok": False, "error": data.get("message", "")})
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


class ZaloAgent(StatefulSorcarAgent):
    """StatefulSorcarAgent extended with Zalo OA API tools."""

    def __init__(self) -> None:
        super().__init__("Zalo Agent")
        self._backend = ZaloChannelBackend()
        cfg = _load_config()
        if cfg:  # pragma: no branch
            self._backend._access_token = cfg["access_token"]
            self._backend._oa_id = cfg.get("oa_id", "")

    def _get_tools(self) -> list:
        """Return SorcarAgent tools + Zalo auth tools + Zalo API tools."""
        tools = super()._get_tools()
        agent = self

        def check_zalo_auth() -> str:
            """Check if Zalo credentials are configured and valid.

            Returns:
                Authentication status or instructions.
            """
            if not agent._backend._access_token:  # pragma: no branch
                return (
                    "Not authenticated with Zalo. Use authenticate_zalo(access_token=...) "
                    "to configure. Get a token from Zalo for Developers portal."
                )
            try:
                result = json.loads(agent._backend.get_oa_info())
                if result.get("ok"):  # pragma: no branch
                    oa = result.get("oa", {})
                    return json.dumps(
                        {"ok": True, "name": oa.get("name", ""), "oa_id": oa.get("oa_id", "")}
                    )
                return json.dumps({"ok": False, "error": "Authentication failed."})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_zalo(access_token: str, oa_id: str = "") -> str:
            """Store and validate Zalo OA credentials.

            Args:
                access_token: Zalo OA access token from developer portal.
                oa_id: Official Account ID (optional).

            Returns:
                Validation result or error message.
            """
            if not access_token.strip():  # pragma: no branch
                return "access_token cannot be empty."
            agent._backend._access_token = access_token.strip()
            agent._backend._oa_id = oa_id.strip()
            try:
                result = json.loads(agent._backend.get_oa_info())
                if result.get("ok"):  # pragma: no branch
                    _save_config(access_token, oa_id)
                    return json.dumps({"ok": True, "message": "Zalo credentials saved."})
                return json.dumps({"ok": False, "error": "Could not verify credentials."})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def clear_zalo_auth() -> str:
            """Clear the stored Zalo credentials.

            Returns:
                Status message.
            """
            _clear_config()
            agent._backend._access_token = ""
            agent._backend._oa_id = ""
            return "Zalo authentication cleared."

        tools.extend([check_zalo_auth, authenticate_zalo, clear_zalo_auth])

        if agent._backend._access_token:  # pragma: no branch
            tools.extend(agent._backend.get_tool_methods())

        return tools


def main() -> None:
    """Run the ZaloAgent from the command line with chat persistence."""
    import sys
    import time as time_mod

    if len(sys.argv) <= 1:  # pragma: no branch
        print("Usage: kiss-zalo [-m MODEL] [-t TASK] [-n] [--daemon]")
        sys.exit(1)

    parser = _build_arg_parser()
    parser.add_argument("-n", "--new", action="store_true", help="Start a new chat session")
    parser.add_argument("--daemon", action="store_true", help="Run as background daemon")
    parser.add_argument("--allow-users", default="", help="Comma-separated user IDs to allow")
    args = parser.parse_args()

    if args.daemon:  # pragma: no branch
        from kiss.channels.background_agent import ChannelDaemon

        backend = ZaloChannelBackend()
        cfg = _load_config()
        if not cfg:  # pragma: no branch
            print("Not authenticated. Run: kiss-zalo -t 'authenticate'")
            sys.exit(1)
        backend._access_token = cfg["access_token"]
        backend._oa_id = cfg.get("oa_id", "")
        allow_users = [u.strip() for u in args.allow_users.split(",") if u.strip()] or None
        daemon = ChannelDaemon(
            backend=backend,
            channel_name="",
            agent_name="Zalo Background Agent",
            extra_tools=backend.get_tool_methods(),
            model_name=args.model_name,
            max_budget=args.max_budget,
            work_dir=args.work_dir or str(Path.home() / ".kiss" / "daemon_work"),
            poll_interval=1.0,
            allow_users=allow_users,
        )
        print("Starting Zalo daemon... (Ctrl+C to stop)")
        try:
            daemon.run()
        except KeyboardInterrupt:
            print("Daemon stopped.")
        return

    agent = ZaloAgent()
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
