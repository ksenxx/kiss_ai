"""Feishu/Lark Agent — StatefulSorcarAgent extension with Feishu Open Platform tools.

Provides authenticated access to Feishu/Lark via app_id and app_secret.
Stores config in ``~/.kiss/channels/feishu/config.json``.

Usage::

    agent = FeishuAgent()
    agent.run(prompt_template="List all chats")
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
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

logger = logging.getLogger(__name__)

_FEISHU_DIR = Path.home() / ".kiss" / "channels" / "feishu"


def _config_path() -> Path:
    """Return the path to the stored Feishu config file."""
    return _FEISHU_DIR / "config.json"


def _load_config() -> dict[str, str] | None:
    """Load stored Feishu config from disk."""
    path = _config_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if (  # pragma: no branch
            isinstance(data, dict) and data.get("app_id") and data.get("app_secret")
        ):
            return {"app_id": data["app_id"], "app_secret": data["app_secret"]}
        return None
    except (json.JSONDecodeError, OSError):
        return None


def _save_config(app_id: str, app_secret: str) -> None:
    """Save Feishu config to disk with restricted permissions."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"app_id": app_id.strip(), "app_secret": app_secret.strip()}, indent=2
    ))
    if sys.platform != "win32":  # pragma: no branch
        path.chmod(0o600)


def _clear_config() -> None:
    """Delete the stored Feishu config."""
    path = _config_path()
    if path.exists():  # pragma: no branch
        path.unlink()


class FeishuChannelBackend:
    """ChannelBackend implementation for Feishu/Lark Open Platform."""

    def __init__(self) -> None:
        self._client: Any = None
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Authenticate with Feishu using stored app credentials."""
        cfg = _load_config()
        if not cfg:  # pragma: no branch
            self._connection_info = "No Feishu config found."
            return False
        try:
            import lark_oapi as lark

            self._client = (
                lark.Client.builder()
                .app_id(cfg["app_id"])
                .app_secret(cfg["app_secret"])
                .build()
            )
            self._connection_info = f"Connected with app_id {cfg['app_id']}"
            return True
        except Exception as e:
            self._connection_info = f"Feishu connection failed: {e}"
            return False

    @property
    def connection_info(self) -> str:
        """Human-readable connection status string."""
        return self._connection_info

    def find_channel(self, name: str) -> str | None:
        """Return channel name as chat ID."""
        return name if name else None

    def find_user(self, username: str) -> str | None:
        """Return username as user ID."""
        return username if username else None

    def join_channel(self, channel_id: str) -> None:
        """No-op for Feishu bots."""

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll Feishu chat for new messages."""
        if not self._client or not channel_id:  # pragma: no branch
            return [], oldest
        try:
            from lark_oapi.api.im.v1 import ListMessageRequest

            req = (
                ListMessageRequest.builder()
                .container_id(channel_id)
                .container_id_type("chat_id")
                .page_size(limit)
                .build()
            )
            resp = self._client.im.v1.message.list(req)
            if not resp.success():  # pragma: no branch
                return [], oldest
            messages: list[dict[str, Any]] = []
            new_oldest = oldest
            for item in resp.data.items or []:  # pragma: no branch
                ts = item.create_time or ""
                if oldest and ts <= oldest:  # pragma: no branch
                    continue
                new_oldest = ts
                body = item.body
                text = ""
                if body:  # pragma: no branch
                    try:
                        content = json.loads(body.content or "{}")
                        text = content.get("text", "")
                    except Exception:
                        text = str(body.content or "")
                messages.append({
                    "ts": ts,
                    "user": item.sender.id if item.sender else "",
                    "text": text,
                    "message_id": item.message_id or "",
                })
            return messages, new_oldest
        except Exception:
            return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a Feishu message."""
        if not self._client:  # pragma: no branch
            return
        try:
            from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

            body = (
                CreateMessageRequestBody.builder()
                .receive_id(channel_id)
                .receive_id_type("chat_id")
                .msg_type("text")
                .content(json.dumps({"text": text}))
                .build()
            )
            req = CreateMessageRequest.builder().request_body(body).build()
            self._client.im.v1.message.create(req)
        except Exception:
            pass

    def wait_for_reply(
        self,
        channel_id: str,
        thread_ts: str,
        user_id: str,
        timeout_seconds: float = 300.0,
        stop_event: threading.Event | None = None,
    ) -> str | None:
        """Poll for a reply from a specific user."""
        oldest = str(int(time.time() * 1000))

        def poll() -> list[dict[str, Any]]:
            nonlocal oldest
            msgs, oldest = self.poll_messages(channel_id, oldest)
            return msgs

        return wait_for_matching_message(
            poll=poll,
            matches=lambda msg: msg.get("user") == user_id,
            extract_text=lambda msg: str(msg.get("text", "")),
            timeout_seconds=timeout_seconds,
            stop_event=stop_event,
            poll_interval=3.0,
        )

    def disconnect(self) -> None:
        """Release backend resources before stop or reconnect."""

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Check if message is from the bot."""
        return False

    def strip_bot_mention(self, text: str) -> str:
        """Remove bot mentions from text."""
        return text

    def send_text_message(
        self, receive_id: str, text: str, receive_id_type: str = "chat_id"
    ) -> str:
        """Send a text message to a Feishu chat or user.

        Args:
            receive_id: Chat ID, user ID, or open ID depending on receive_id_type.
            text: Message text.
            receive_id_type: "chat_id", "user_id", "open_id", or "email".
                Default: "chat_id".

        Returns:
            JSON string with ok status and message id.
        """
        assert self._client is not None
        try:
            from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

            body = (
                CreateMessageRequestBody.builder()
                .receive_id(receive_id)
                .receive_id_type(receive_id_type)
                .msg_type("text")
                .content(json.dumps({"text": text}))
                .build()
            )
            req = CreateMessageRequest.builder().request_body(body).build()
            resp = self._client.im.v1.message.create(req)
            if not resp.success():  # pragma: no branch
                return json.dumps({"ok": False, "error": resp.msg})
            return json.dumps({"ok": True, "message_id": resp.data.message_id or ""})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def reply_message(self, message_id: str, text: str) -> str:
        """Reply to an existing Feishu message.

        Args:
            message_id: ID of the message to reply to.
            text: Reply text.

        Returns:
            JSON string with ok status and reply message id.
        """
        assert self._client is not None
        try:
            from lark_oapi.api.im.v1 import ReplyMessageRequest, ReplyMessageRequestBody

            body = (
                ReplyMessageRequestBody.builder()
                .content(json.dumps({"text": text}))
                .msg_type("text")
                .build()
            )
            req = ReplyMessageRequest.builder().message_id(message_id).request_body(body).build()
            resp = self._client.im.v1.message.reply(req)
            if not resp.success():  # pragma: no branch
                return json.dumps({"ok": False, "error": resp.msg})
            return json.dumps({"ok": True, "message_id": resp.data.message_id or ""})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def delete_message(self, message_id: str) -> str:
        """Delete a Feishu message.

        Args:
            message_id: Message ID to delete.

        Returns:
            JSON string with ok status.
        """
        assert self._client is not None
        try:
            from lark_oapi.api.im.v1 import DeleteMessageRequest

            req = DeleteMessageRequest.builder().message_id(message_id).build()
            resp = self._client.im.v1.message.delete(req)
            if not resp.success():  # pragma: no branch
                return json.dumps({"ok": False, "error": resp.msg})
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_messages(
        self,
        container_id: str,
        start_time: str = "",
        end_time: str = "",
        page_size: int = 20,
    ) -> str:
        """List messages in a Feishu chat.

        Args:
            container_id: Chat ID.
            start_time: Start Unix timestamp (seconds). Optional.
            end_time: End Unix timestamp (seconds). Optional.
            page_size: Maximum messages to return. Default: 20.

        Returns:
            JSON string with message list.
        """
        assert self._client is not None
        try:
            from lark_oapi.api.im.v1 import ListMessageRequest

            req = (
                ListMessageRequest.builder()
                .container_id(container_id)
                .container_id_type("chat_id")
                .page_size(page_size)
                .build()
            )
            resp = self._client.im.v1.message.list(req)
            if not resp.success():  # pragma: no branch
                return json.dumps({"ok": False, "error": resp.msg})
            messages = []
            for item in resp.data.items or []:  # pragma: no branch
                messages.append({
                    "message_id": item.message_id or "",
                    "msg_type": item.msg_type or "",
                    "create_time": item.create_time or "",
                    "sender_id": item.sender.id if item.sender else "",
                })
            return json.dumps({"ok": True, "messages": messages}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_chats(self, page_size: int = 20, page_token: str = "") -> str:
        """List Feishu chats the bot is a member of.

        Args:
            page_size: Maximum chats to return. Default: 20.
            page_token: Pagination token.

        Returns:
            JSON string with chat list (chat_id, name, description).
        """
        assert self._client is not None
        try:
            from lark_oapi.api.im.v1 import ListChatRequest

            req_builder = ListChatRequest.builder().page_size(page_size)
            if page_token:  # pragma: no branch
                req_builder = req_builder.page_token(page_token)
            resp = self._client.im.v1.chat.list(req_builder.build())
            if not resp.success():  # pragma: no branch
                return json.dumps({"ok": False, "error": resp.msg})
            chats = [
                {
                    "chat_id": c.chat_id or "",
                    "name": c.name or "",
                    "description": c.description or "",
                }
                for c in (resp.data.items or [])
            ]
            result: dict[str, Any] = {"ok": True, "chats": chats}
            if resp.data.page_token:  # pragma: no branch
                result["page_token"] = resp.data.page_token
            return json.dumps(result, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_chat(self, chat_id: str) -> str:
        """Get information about a Feishu chat.

        Args:
            chat_id: Chat ID.

        Returns:
            JSON string with chat details.
        """
        assert self._client is not None
        try:
            from lark_oapi.api.im.v1 import GetChatRequest

            req = GetChatRequest.builder().chat_id(chat_id).build()
            resp = self._client.im.v1.chat.get(req)
            if not resp.success():  # pragma: no branch
                return json.dumps({"ok": False, "error": resp.msg})
            data = resp.data
            return json.dumps({
                "ok": True,
                "chat_id": data.chat_id or "",
                "name": data.name or "",
                "description": data.description or "",
                "owner_id": data.owner_id or "",
            }, indent=2)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_user_info(self, user_id: str, user_id_type: str = "open_id") -> str:
        """Get Feishu user information.

        Args:
            user_id: User ID.
            user_id_type: ID type ("open_id", "user_id", "union_id"). Default: "open_id".

        Returns:
            JSON string with user info.
        """
        assert self._client is not None
        try:
            from lark_oapi.api.contact.v3 import GetUserRequest

            req = GetUserRequest.builder().user_id(user_id).user_id_type(user_id_type).build()
            resp = self._client.contact.v3.user.get(req)
            if not resp.success():  # pragma: no branch
                return json.dumps({"ok": False, "error": resp.msg})
            user = resp.data.user
            return json.dumps({
                "ok": True,
                "name": user.name or "" if user else "",
                "email": user.email or "" if user else "",
                "open_id": user.open_id or "" if user else "",
            })
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


class FeishuAgent(StatefulSorcarAgent):
    """StatefulSorcarAgent extended with Feishu/Lark Open Platform tools."""

    def __init__(self) -> None:
        super().__init__("Feishu Agent")
        self._backend = FeishuChannelBackend()
        cfg = _load_config()
        if cfg:  # pragma: no branch
            try:
                import lark_oapi as lark

                self._backend._client = (
                    lark.Client.builder()
                    .app_id(cfg["app_id"])
                    .app_secret(cfg["app_secret"])
                    .build()
                )
            except Exception:
                pass

    def _get_tools(self) -> list:
        """Return SorcarAgent tools + Feishu auth tools + Feishu API tools."""
        tools = super()._get_tools()
        agent = self

        def check_feishu_auth() -> str:
            """Check if Feishu credentials are configured and valid.

            Returns:
                Authentication status or instructions.
            """
            if agent._backend._client is None:  # pragma: no branch
                return (
                    "Not authenticated with Feishu. "
                    "Use authenticate_feishu(app_id=..., app_secret=...) "
                    "to configure. Get credentials from the Feishu Open Platform developer console."
                )
            try:
                resp = agent._backend.list_chats(page_size=1)
                data = json.loads(resp)
                if data.get("ok"):  # pragma: no branch
                    return json.dumps({"ok": True, "message": "Feishu authenticated."})
                return resp
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_feishu(app_id: str, app_secret: str) -> str:
            """Store and validate Feishu app credentials.

            Args:
                app_id: Feishu app ID from developer console.
                app_secret: Feishu app secret from developer console.

            Returns:
                Validation result or error message.
            """
            for val, name in [(app_id, "app_id"), (app_secret, "app_secret")]:  # pragma: no branch
                if not val.strip():  # pragma: no branch
                    return f"{name} cannot be empty."
            try:
                import lark_oapi as lark

                client = (
                    lark.Client.builder()
                    .app_id(app_id.strip())
                    .app_secret(app_secret.strip())
                    .build()
                )
                agent._backend._client = client
                _save_config(app_id, app_secret)
                return json.dumps({"ok": True, "message": "Feishu credentials saved."})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def clear_feishu_auth() -> str:
            """Clear the stored Feishu credentials.

            Returns:
                Status message.
            """
            _clear_config()
            agent._backend._client = None
            return "Feishu authentication cleared."

        tools.extend([check_feishu_auth, authenticate_feishu, clear_feishu_auth])

        if agent._backend._client is not None:  # pragma: no branch
            tools.extend(agent._backend.get_tool_methods())

        return tools


def main() -> None:
    """Run the FeishuAgent from the command line with chat persistence."""
    import sys
    import time as time_mod

    if len(sys.argv) <= 1:  # pragma: no branch
        print("Usage: kiss-feishu [-m MODEL] [-t TASK] [-n] [--daemon]")
        sys.exit(1)

    parser = _build_arg_parser()
    parser.add_argument("-n", "--new", action="store_true", help="Start a new chat session")
    parser.add_argument("--daemon", action="store_true", help="Run as background daemon")
    parser.add_argument("--daemon-channel", default="", help="Chat ID to monitor")
    parser.add_argument("--allow-users", default="", help="Comma-separated user IDs to allow")
    args = parser.parse_args()

    if args.daemon:  # pragma: no branch
        from kiss.channels.background_agent import ChannelDaemon

        backend = FeishuChannelBackend()
        cfg = _load_config()
        if not cfg:  # pragma: no branch
            print("Not authenticated. Run: kiss-feishu -t 'authenticate'")
            sys.exit(1)
        import lark_oapi as lark
        backend._client = (
            lark.Client.builder()
            .app_id(cfg["app_id"])
            .app_secret(cfg["app_secret"])
            .build()
        )
        allow_users = [u.strip() for u in args.allow_users.split(",") if u.strip()] or None
        daemon = ChannelDaemon(
            backend=backend,
            channel_name=args.daemon_channel,
            agent_name="Feishu Background Agent",
            extra_tools=backend.get_tool_methods(),
            model_name=args.model_name,
            max_budget=args.max_budget,
            work_dir=args.work_dir or str(Path.home() / ".kiss" / "daemon_work"),
            allow_users=allow_users,
        )
        print("Starting Feishu daemon... (Ctrl+C to stop)")
        try:
            daemon.run()
        except KeyboardInterrupt:
            print("Daemon stopped.")
        return

    agent = FeishuAgent()
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
