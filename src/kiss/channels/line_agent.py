"""LINE Agent — StatefulSorcarAgent extension with LINE Messaging API tools.

Provides authenticated access to LINE via channel access token. Uses webhook
queue pattern for receiving messages. Stores config in
``~/.kiss/channels/line/config.json``.

Usage::

    agent = LineAgent()
    agent.run(prompt_template="Send 'Hello!' to user U123456789")
"""

from __future__ import annotations

import json
import logging
import queue
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.sorcar_agent import (
    _build_arg_parser,
    _resolve_task,
    cli_ask_user_question,
    cli_wait_for_user,
)
from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent

logger = logging.getLogger(__name__)

_LINE_DIR = Path.home() / ".kiss" / "channels" / "line"


def _config_path() -> Path:
    """Return the path to the stored LINE config file."""
    return _LINE_DIR / "config.json"


def _load_config() -> dict[str, str] | None:
    """Load stored LINE config from disk."""
    path = _config_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and data.get("channel_access_token"):
            return {
                "channel_access_token": data["channel_access_token"],
                "channel_secret": data.get("channel_secret", ""),
            }
        return None
    except (json.JSONDecodeError, OSError):
        return None


def _save_config(channel_access_token: str, channel_secret: str = "") -> None:
    """Save LINE config to disk with restricted permissions."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "channel_access_token": channel_access_token.strip(),
        "channel_secret": channel_secret.strip(),
    }, indent=2))
    if sys.platform != "win32":
        path.chmod(0o600)


def _clear_config() -> None:
    """Delete the stored LINE config."""
    path = _config_path()
    if path.exists():
        path.unlink()


class LineChannelBackend:
    """ChannelBackend implementation for LINE Messaging API.

    Uses webhook queue pattern for receiving inbound messages.
    """

    def __init__(self) -> None:
        self._api: Any = None
        self._message_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._webhook_server: HTTPServer | None = None
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Authenticate with LINE and start webhook server."""
        cfg = _load_config()
        if not cfg:
            self._connection_info = "No LINE config found."
            return False
        try:
            from linebot.v3.messaging import ApiClient, Configuration, MessagingApi

            configuration = Configuration(access_token=cfg["channel_access_token"])
            self._api = MessagingApi(ApiClient(configuration))
            self._connection_info = "Connected to LINE"
            self._start_webhook_server()
            return True
        except Exception as e:
            self._connection_info = f"LINE connection failed: {e}"
            return False

    def _start_webhook_server(self, port: int = 8080) -> None:
        """Start webhook HTTP server."""
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    data = json.loads(body)
                    for event in data.get("events", []):
                        if event.get("type") == "message":
                            msg = event.get("message", {})
                            if msg.get("type") == "text":
                                source = event.get("source", {})
                                backend._message_queue.put({
                                    "ts": str(event.get("timestamp", "")),
                                    "user": source.get("userId", ""),
                                    "text": msg.get("text", ""),
                                    "reply_token": event.get("replyToken", ""),
                                    "group_id": source.get("groupId", ""),
                                    "room_id": source.get("roomId", ""),
                                })
                except Exception:
                    pass
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args: Any) -> None:  # type: ignore[override]
                pass

        try:
            self._webhook_server = HTTPServer(("0.0.0.0", port), Handler)
            thread = threading.Thread(
                target=self._webhook_server.serve_forever, daemon=True
            )
            thread.start()
            logger.info("LINE webhook server started on port %d", port)
        except OSError as e:
            logger.warning("Could not start LINE webhook server: %s", e)

    @property
    def connection_info(self) -> str:
        """Human-readable connection status string."""
        return self._connection_info

    def find_channel(self, name: str) -> str | None:
        """Return channel name as user/group ID."""
        return name if name else None

    def find_user(self, username: str) -> str | None:
        """Return username as user ID."""
        return username if username else None

    def join_channel(self, channel_id: str) -> None:
        """No-op for LINE."""

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Drain the webhook message queue."""
        messages: list[dict[str, Any]] = []
        while not self._message_queue.empty() and len(messages) < limit:
            messages.append(self._message_queue.get_nowait())
        return messages, oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a LINE push message."""
        if not self._api:
            return
        try:
            from linebot.v3.messaging import PushMessageRequest, TextMessage

            self._api.push_message(PushMessageRequest(
                to=channel_id,
                messages=[TextMessage(text=text)]
            ))
        except Exception:
            pass

    def wait_for_reply(self, channel_id: str, thread_ts: str, user_id: str) -> str:
        """Poll for a reply from a specific user."""
        import time
        while True:
            time.sleep(2.0)
            msgs, _ = self.poll_messages(channel_id, "")
            for msg in msgs:
                if msg.get("user") == user_id:
                    return str(msg.get("text", ""))

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Check if message is from the bot."""
        return False

    def strip_bot_mention(self, text: str) -> str:
        """Remove bot mentions from text."""
        return text

    def push_text_message(self, to: str, text: str) -> str:
        """Send a push text message to a LINE user or group.

        Args:
            to: Target user ID, group ID, or room ID.
            text: Message text (up to 5000 characters).

        Returns:
            JSON string with ok status.
        """
        assert self._api is not None
        try:
            from linebot.v3.messaging import PushMessageRequest, TextMessage

            self._api.push_message(PushMessageRequest(to=to, messages=[TextMessage(text=text)]))
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def reply_message(self, reply_token: str, messages_json: str) -> str:
        """Reply to a message using the reply token.

        Args:
            reply_token: Reply token from an inbound message event.
            messages_json: JSON array of message objects. Example:
                '[{"type":"text","text":"Hello!"}]'

        Returns:
            JSON string with ok status.
        """
        assert self._api is not None
        try:
            from linebot.v3.messaging import ReplyMessageRequest, TextMessage

            msgs_data = json.loads(messages_json)
            messages = [
                TextMessage(text=m.get("text", ""))
                for m in msgs_data
                if m.get("type") == "text"
            ]
            self._api.reply_message(ReplyMessageRequest(replyToken=reply_token, messages=messages))
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_profile(self, user_id: str) -> str:
        """Get a LINE user's profile.

        Args:
            user_id: LINE user ID.

        Returns:
            JSON string with user profile (displayName, pictureUrl, statusMessage).
        """
        assert self._api is not None
        try:
            profile = self._api.get_profile(user_id)
            return json.dumps({
                "ok": True,
                "display_name": profile.display_name,
                "user_id": profile.user_id,
                "picture_url": profile.picture_url or "",
                "status_message": profile.status_message or "",
            })
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_quota(self) -> str:
        """Get the LINE messaging quota for the current month.

        Returns:
            JSON string with quota information.
        """
        assert self._api is not None
        try:
            quota = self._api.get_message_quota()
            return json.dumps({
                "ok": True,
                "type": quota.type,
                "value": quota.value if hasattr(quota, "value") else None,
            })
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def leave_group(self, group_id: str) -> str:
        """Leave a LINE group.

        Args:
            group_id: Group ID to leave.

        Returns:
            JSON string with ok status.
        """
        assert self._api is not None
        try:
            self._api.leave_group(group_id)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def push_image_message(
        self, to: str, image_url: str, preview_url: str
    ) -> str:
        """Send a push image message.

        Args:
            to: Target user ID, group ID, or room ID.
            image_url: URL of the full-size image.
            preview_url: URL of the preview image.

        Returns:
            JSON string with ok status.
        """
        assert self._api is not None
        try:
            from linebot.v3.messaging import ImageMessage, PushMessageRequest

            self._api.push_message(PushMessageRequest(
                to=to,
                messages=[ImageMessage(originalContentUrl=image_url, previewImageUrl=preview_url)]
            ))
            return json.dumps({"ok": True})
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


class LineAgent(StatefulSorcarAgent):
    """StatefulSorcarAgent extended with LINE Messaging API tools."""

    def __init__(self) -> None:
        super().__init__("LINE Agent")
        self._backend = LineChannelBackend()
        cfg = _load_config()
        if cfg:
            try:
                from linebot.v3.messaging import ApiClient, Configuration, MessagingApi

                configuration = Configuration(access_token=cfg["channel_access_token"])
                self._backend._api = MessagingApi(ApiClient(configuration))
            except Exception:
                pass

    def _get_tools(self) -> list:
        """Return SorcarAgent tools + LINE auth tools + LINE API tools."""
        tools = super()._get_tools()
        agent = self

        def check_line_auth() -> str:
            """Check if LINE credentials are configured and valid.

            Returns:
                Authentication status or instructions.
            """
            if agent._backend._api is None:
                return (
                    "Not authenticated with LINE. Use authenticate_line(channel_access_token=...) "
                    "to configure. Get a token from LINE Developers Console."
                )
            try:
                quota = json.loads(agent._backend.get_quota())
                if quota.get("ok"):
                    return json.dumps({"ok": True, "quota": quota})
                return json.dumps({"ok": True, "message": "LINE authenticated."})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_line(
            channel_access_token: str, channel_secret: str = ""
        ) -> str:
            """Store and validate LINE channel credentials.

            Args:
                channel_access_token: LINE channel access token from Developers Console.
                channel_secret: LINE channel secret (optional, for webhook verification).

            Returns:
                Validation result or error message.
            """
            if not channel_access_token.strip():
                return "channel_access_token cannot be empty."
            try:
                from linebot.v3.messaging import ApiClient, Configuration, MessagingApi

                configuration = Configuration(access_token=channel_access_token.strip())
                api = MessagingApi(ApiClient(configuration))
                agent._backend._api = api
                _save_config(channel_access_token, channel_secret)
                return json.dumps({"ok": True, "message": "LINE credentials saved."})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def clear_line_auth() -> str:
            """Clear the stored LINE credentials.

            Returns:
                Status message.
            """
            _clear_config()
            agent._backend._api = None
            return "LINE authentication cleared."

        tools.extend([check_line_auth, authenticate_line, clear_line_auth])

        if agent._backend._api is not None:
            tools.extend(agent._backend.get_tool_methods())

        return tools


def main() -> None:
    """Run the LineAgent from the command line with chat persistence."""
    import os
    import sys
    import time as time_mod

    if len(sys.argv) <= 1:
        print("Usage: kiss-line [-m MODEL] [-t TASK] [-n] [--daemon]")
        sys.exit(1)

    parser = _build_arg_parser()
    parser.add_argument("-n", "--new", action="store_true", help="Start a new chat session")
    parser.add_argument("--daemon", action="store_true", help="Run as background daemon")
    parser.add_argument("--daemon-channel", default="", help="User/group ID to monitor")
    parser.add_argument("--allow-users", default="", help="Comma-separated user IDs to allow")
    args = parser.parse_args()

    if args.daemon:
        from kiss.channels.background_agent import ChannelDaemon

        backend = LineChannelBackend()
        cfg = _load_config()
        if not cfg:
            print("Not authenticated. Run: kiss-line -t 'authenticate'")
            sys.exit(1)
        from linebot.v3.messaging import ApiClient, Configuration, MessagingApi
        configuration = Configuration(access_token=cfg["channel_access_token"])
        backend._api = MessagingApi(ApiClient(configuration))
        allow_users = [u.strip() for u in args.allow_users.split(",") if u.strip()] or None
        daemon = ChannelDaemon(
            backend=backend,
            channel_name=args.daemon_channel,
            agent_name="LINE Background Agent",
            extra_tools=backend.get_tool_methods(),
            model_name=args.model_name,
            max_budget=args.max_budget,
            work_dir=args.work_dir or str(Path.home() / ".kiss" / "daemon_work"),
            poll_interval=1.0,
            allow_users=allow_users,
        )
        print("Starting LINE daemon... (Ctrl+C to stop)")
        try:
            daemon.run()
        except KeyboardInterrupt:
            print("Daemon stopped.")
        return

    agent = LineAgent()
    task_description = _resolve_task(args)
    work_dir = args.work_dir or str(Path(".").resolve())
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    if args.new:
        agent.new_chat()
    else:
        agent.resume_chat(task_description)

    model_config: dict[str, Any] = {}
    if args.endpoint:
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

    old_cwd = os.getcwd()
    os.chdir(work_dir)
    start_time = time_mod.time()
    try:
        agent.run(**run_kwargs)
    finally:
        os.chdir(old_cwd)
    elapsed = time_mod.time() - start_time

    print(f"Time: {elapsed:.1f}s")
    print(f"Cost: ${agent.budget_used:.4f}")
    print(f"Total tokens: {agent.total_tokens_used}")


if __name__ == "__main__":
    main()
