"""Feishu/Lark Agent — StatefulSorcarAgent extension with Feishu Open Platform tools.

Provides authenticated access to Feishu/Lark via app_id and app_secret.
Stores config in ``~/.kiss/channels/feishu/config.json``.

Usage::

    agent = FeishuAgent()
    agent.run(prompt_template="List all chats")
"""

from __future__ import annotations

import json
import sys
import threading
import time
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

_FEISHU_DIR = Path.home() / ".kiss" / "channels" / "feishu"
_config = ChannelConfig(
    _FEISHU_DIR,
    (
        "app_id",
        "app_secret",
    ),
)


class FeishuChannelBackend(ToolMethodBackend):
    """Channel backend for Feishu/Lark Open Platform."""

    def __init__(self) -> None:
        self._client: Any = None
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Authenticate with Feishu using stored app credentials."""
        cfg = _config.load()
        if not cfg:  # pragma: no branch
            self._connection_info = "No Feishu config found."
            return False
        try:
            import lark_oapi as lark

            self._client = (
                lark.Client.builder().app_id(cfg["app_id"]).app_secret(cfg["app_secret"]).build()
            )
            self._connection_info = f"Connected with app_id {cfg['app_id']}"
            return True
        except Exception as e:
            self._connection_info = f"Feishu connection failed: {e}"
            return False

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
                messages.append(
                    {
                        "ts": ts,
                        "user": item.sender.id if item.sender else "",
                        "text": text,
                        "message_id": item.message_id or "",
                    }
                )
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
                messages.append(
                    {
                        "message_id": item.message_id or "",
                        "msg_type": item.msg_type or "",
                        "create_time": item.create_time or "",
                        "sender_id": item.sender.id if item.sender else "",
                    }
                )
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
            return json.dumps(
                {
                    "ok": True,
                    "chat_id": data.chat_id or "",
                    "name": data.name or "",
                    "description": data.description or "",
                    "owner_id": data.owner_id or "",
                },
                indent=2,
            )
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
            return json.dumps(
                {
                    "ok": True,
                    "name": user.name or "" if user else "",
                    "email": user.email or "" if user else "",
                    "open_id": user.open_id or "" if user else "",
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class FeishuAgent(BaseChannelAgent, StatefulSorcarAgent):
    """StatefulSorcarAgent extended with Feishu/Lark Open Platform tools."""

    def __init__(self) -> None:
        super().__init__("Feishu Agent")
        self._backend = FeishuChannelBackend()
        cfg = _config.load()
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

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return self._backend._client is not None

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
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
                return str(resp)
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
                _config.save({"app_id": app_id.strip(), "app_secret": app_secret.strip()})
                return json.dumps({"ok": True, "message": "Feishu credentials saved."})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def clear_feishu_auth() -> str:
            """Clear the stored Feishu credentials.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._client = None
            return "Feishu authentication cleared."

        return [check_feishu_auth, authenticate_feishu, clear_feishu_auth]


def _make_backend() -> FeishuChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = FeishuChannelBackend()
    cfg = _config.load()
    if not cfg:  # pragma: no branch
        print("Not authenticated. Run: kiss-feishu -t 'authenticate'")
        sys.exit(1)
    import lark_oapi as lark

    backend._client = (
        lark.Client.builder().app_id(cfg["app_id"]).app_secret(cfg["app_secret"]).build()
    )
    return backend


def main() -> None:
    """Run the FeishuAgent from the command line with chat persistence."""
    channel_main(
        FeishuAgent,
        "kiss-feishu",
        channel_name="Feishu",
        make_backend=_make_backend,
    )


if __name__ == "__main__":
    main()
