# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""LINE Agent — channel agent with LINE Messaging API tools.

Provides authenticated access to LINE via channel access token. Uses webhook
queue pattern for receiving messages. Stores config in
``~/.kiss/third_party_agents/line/config.json``.

Usage::

    agent = LineAgent()
    agent.run(prompt_template="Send 'Hello!' to user U123456789")
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from kiss.agents.third_party_agents._backend_utils import (
    ThreadedHTTPServer,
    drain_queue_messages,
    start_http_server,
    stop_http_server,
)
from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

logger = logging.getLogger(__name__)

_DEFAULT_WEBHOOK_PORT = 18081

_LINE_DIR = Path.home() / ".kiss" / "third_party_agents" / "line"
_LINE_API_BASE = "https://api.line.me"
_config = ChannelConfig(_LINE_DIR, ("channel_access_token",))


def _scrub_config_token() -> None:
    """Remove a vault-migrated ``channel_access_token`` from config.json.

    Finishes the Muse migration automatically: the ``channel_secret``
    (an inbound webhook-verification secret that never leaves this
    machine) is kept and the file is deleted when nothing but the
    access token was stored.
    """
    try:
        cfg = json.loads(_config.path.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(cfg, dict) or "channel_access_token" not in cfg:
        return
    kept = {k: str(v) for k, v in cfg.items() if k != "channel_access_token" and v}
    if kept:
        from kiss.agents.third_party_agents._channel_agent_utils import save_json_config

        save_json_config(_config.path, kept)
    else:
        _config.clear()


class _SdkLineApi:
    """LINE Messaging API adapter over the official ``linebot`` SDK.

    The backend talks to one of two duck-typed adapters — this one
    (legacy mode, real token in-process) or :class:`_MuseLineApi`
    (Muse mode, surrogate + daemon boundary) — through the same small
    set of semantic methods, so the tool methods contain no SDK
    imports or mode branches.
    """

    def __init__(self, channel_access_token: str) -> None:
        from linebot.v3.messaging import ApiClient, Configuration, MessagingApi

        self._api = MessagingApi(ApiClient(Configuration(access_token=channel_access_token)))

    def push_text(self, to: str, text: str) -> None:
        """Push one text message to a user, group, or room."""
        from linebot.v3.messaging import PushMessageRequest, TextMessage

        self._api.push_message(PushMessageRequest(to=to, messages=[TextMessage(text=text)]))

    def push_image(self, to: str, image_url: str, preview_url: str) -> None:
        """Push one image message."""
        from linebot.v3.messaging import ImageMessage, PushMessageRequest

        self._api.push_message(
            PushMessageRequest(
                to=to,
                messages=[ImageMessage(originalContentUrl=image_url, previewImageUrl=preview_url)],
            )
        )

    def reply_texts(self, reply_token: str, texts: list[str]) -> None:
        """Reply to an inbound event with text messages."""
        from linebot.v3.messaging import ReplyMessageRequest, TextMessage

        self._api.reply_message(
            ReplyMessageRequest(
                replyToken=reply_token, messages=[TextMessage(text=t) for t in texts]
            )
        )

    def get_profile(self, user_id: str) -> dict[str, Any]:
        """Return a user's profile as a plain dict."""
        profile = self._api.get_profile(user_id)
        return {
            "display_name": profile.display_name,
            "user_id": profile.user_id,
            "picture_url": profile.picture_url or "",
            "status_message": profile.status_message or "",
        }

    def get_quota(self) -> dict[str, Any]:
        """Return the monthly message quota as a plain dict."""
        quota = self._api.get_message_quota()
        return {"type": quota.type, "value": quota.value if hasattr(quota, "value") else None}

    def leave_group(self, group_id: str) -> None:
        """Leave a group."""
        self._api.leave_group(group_id)


class _MuseLineApi:
    """LINE Messaging API adapter that executes at the Muse boundary.

    Sends a surrogate bearer with every REST call; the daemon swaps in
    the real channel access token, so this process never holds it.
    """

    def __init__(self, surrogate: str) -> None:
        from kiss.agents.third_party_agents.muse_auth.client import MuseBoundarySession

        self._api_base = os.environ.get("LINE_API_BASE", _LINE_API_BASE)
        self._surrogate = surrogate
        self._session = MuseBoundarySession("line")

    def _call(self, method: str, path: str, json_body: Any = None) -> dict[str, Any]:
        """Execute one REST call at the boundary, raising on HTTP errors.

        Args:
            method: HTTP method.
            path: API path under ``https://api.line.me``.
            json_body: Optional JSON body.

        Returns:
            The decoded JSON response (``{}`` for empty bodies).

        Raises:
            RuntimeError: On any HTTP error status (mirroring the SDK,
                which raises ``ApiException`` on non-2xx responses).
        """
        resp = self._session.request(
            method,
            f"{self._api_base}{path}",
            headers={"Authorization": f"Bearer {self._surrogate}"},
            json=json_body,
            timeout=30,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"LINE API {method} {path} failed: HTTP {resp.status_code} {resp.text[:300]}"
            )
        return dict(resp.json()) if resp.content else {}

    def push_text(self, to: str, text: str) -> None:
        """Push one text message to a user, group, or room."""
        self._call(
            "POST",
            "/v2/bot/message/push",
            {"to": to, "messages": [{"type": "text", "text": text}]},
        )

    def push_image(self, to: str, image_url: str, preview_url: str) -> None:
        """Push one image message."""
        self._call(
            "POST",
            "/v2/bot/message/push",
            {
                "to": to,
                "messages": [
                    {
                        "type": "image",
                        "originalContentUrl": image_url,
                        "previewImageUrl": preview_url,
                    }
                ],
            },
        )

    def reply_texts(self, reply_token: str, texts: list[str]) -> None:
        """Reply to an inbound event with text messages."""
        self._call(
            "POST",
            "/v2/bot/message/reply",
            {
                "replyToken": reply_token,
                "messages": [{"type": "text", "text": t} for t in texts],
            },
        )

    def get_profile(self, user_id: str) -> dict[str, Any]:
        """Return a user's profile as a plain dict."""
        data = self._call("GET", f"/v2/bot/profile/{user_id}")
        return {
            "display_name": data.get("displayName", ""),
            "user_id": data.get("userId", ""),
            "picture_url": data.get("pictureUrl", "") or "",
            "status_message": data.get("statusMessage", "") or "",
        }

    def get_quota(self) -> dict[str, Any]:
        """Return the monthly message quota as a plain dict."""
        data = self._call("GET", "/v2/bot/message/quota")
        return {"type": data.get("type", ""), "value": data.get("value")}

    def leave_group(self, group_id: str) -> None:
        """Leave a group."""
        self._call("POST", f"/v2/bot/group/{group_id}/leave")


class LineChannelBackend(ToolMethodBackend):
    """Channel backend for LINE Messaging API.

    Uses webhook queue pattern for receiving inbound messages.
    """

    def __init__(self) -> None:
        self._api: Any = None
        self._message_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._webhook_server: ThreadedHTTPServer | None = None
        self._webhook_thread: threading.Thread | None = None
        self._connection_info: str = ""

    def _wire_muse(self) -> bool:
        """Acquire a LINE surrogate and wire the boundary adapter.

        A ``channel_access_token`` still in the legacy config is the
        newest user intent (initial migration, or a rotation done while
        Muse was off): it is enrolled as a bearer credential replacing
        any vault entry, and scrubbed from ``config.json`` (the inbound
        ``channel_secret`` survives) only after the vault holds it.
        No network round trip happens here.

        Returns:
            True when the backend holds a surrogate-backed adapter.
        """
        from kiss.agents.third_party_agents.muse_auth.client import (
            mint_surrogate,
            store_credentials,
        )

        cfg = _config.load_metadata() or {}
        token = cfg.get("channel_access_token", "")
        if token:
            store_credentials("line", {"kind": "bearer", "token": token}, [])
        handle = mint_surrogate("line")
        if handle is None:
            self._connection_info = "No LINE credential in the Muse vault or config."
            return False
        _scrub_config_token()
        self._api = _MuseLineApi(handle.token)
        return True

    def connect(self) -> bool:
        """Authenticate with LINE and start webhook server."""
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        try:
            if muse_auth_enabled():
                if not self._wire_muse():
                    return False
            else:
                cfg = _config.load()
                if not cfg:  # pragma: no branch
                    self._connection_info = "No LINE config found."
                    return False
                self._api = _SdkLineApi(cfg["channel_access_token"])
            self._connection_info = "Connected to LINE"
            if not self._start_webhook_server():  # pragma: no branch
                return False
            return True
        except Exception as e:
            self._connection_info = f"LINE connection failed: {e}"
            return False

    def _start_webhook_server(self, port: int = _DEFAULT_WEBHOOK_PORT) -> bool:
        """Start webhook HTTP server."""
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    data = json.loads(body)
                    for event in data.get("events", []):  # pragma: no branch
                        if event.get("type") == "message":  # pragma: no branch
                            msg = event.get("message", {})
                            if msg.get("type") == "text":  # pragma: no branch
                                source = event.get("source", {})
                                backend._message_queue.put(
                                    {
                                        "ts": str(event.get("timestamp", "")),
                                        "user": source.get("userId", ""),
                                        "text": msg.get("text", ""),
                                        "reply_token": event.get("replyToken", ""),
                                        "group_id": source.get("groupId", ""),
                                        "room_id": source.get("roomId", ""),
                                    }
                                )
                except Exception:
                    pass
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args: Any) -> None:  # type: ignore[override]
                pass

        self.disconnect()
        self._webhook_server, self._webhook_thread, error = start_http_server(
            ("0.0.0.0", port),
            Handler,
            log=logger,
            started_log="LINE webhook server started on port %d",
            error_prefix="LINE webhook bind failed",
            error_log="Could not start LINE webhook server: %s",
        )
        if error is not None:
            self._connection_info = error
            return False
        return True

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Drain the webhook message queue, filtered by source when given."""

        def keep(msg: dict[str, Any]) -> bool:
            if not channel_id:
                return True
            return channel_id in (
                msg.get("user", ""),
                msg.get("group_id", ""),
                msg.get("room_id", ""),
            )

        return drain_queue_messages(self._message_queue, limit=limit, keep=keep), oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a LINE push message.

        Raises:
            RuntimeError: If the LINE API client is not configured.
        """
        if not self._api:
            raise RuntimeError("Not connected to LINE")
        self._api.push_text(channel_id, text)

    def disconnect(self) -> None:
        """Stop the embedded webhook server and release backend resources."""
        self._webhook_server, self._webhook_thread = stop_http_server(
            self._webhook_server, self._webhook_thread
        )

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
            self._api.push_text(to, text)
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
            msgs_data = json.loads(messages_json)
            texts = [m.get("text", "") for m in msgs_data if m.get("type") == "text"]
            self._api.reply_texts(reply_token, texts)
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
            return json.dumps({"ok": True, **self._api.get_profile(user_id)})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_quota(self) -> str:
        """Get the LINE messaging quota for the current month.

        Returns:
            JSON string with quota information.
        """
        assert self._api is not None
        try:
            return json.dumps({"ok": True, **self._api.get_quota()})
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

    def push_image_message(self, to: str, image_url: str, preview_url: str) -> str:
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
            self._api.push_image(to, image_url, preview_url)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


def _muse_authenticate(
    backend: LineChannelBackend, channel_access_token: str, channel_secret: str
) -> str:
    """Enroll a LINE channel access token into the Muse vault and validate it.

    The plaintext token goes straight into the vault as a bearer
    credential, atomically replacing any previous enrollment, and is
    never written to ``config.json`` (only the inbound
    ``channel_secret`` webhook-verification metadata is — written
    first, so a failed enrollment leaves no access token on disk).
    Validation runs the message-quota read through the daemon boundary,
    so it is audited; an invalid token leaves the vault empty.

    Args:
        backend: The agent's LINE backend to (re)wire.
        channel_access_token: LINE channel access token.
        channel_secret: Optional channel secret (inbound verification).

    Returns:
        JSON string with the validation result.
    """
    import contextlib

    from kiss.agents.third_party_agents._channel_agent_utils import save_json_config
    from kiss.agents.third_party_agents.muse_auth.client import (
        clear_credentials,
        store_credentials,
    )

    try:
        if channel_secret:
            save_json_config(_config.path, {"channel_secret": channel_secret})
        else:
            _config.clear()
        store_credentials("line", {"kind": "bearer", "token": channel_access_token}, [])
        if backend._wire_muse():  # pragma: no branch - credential was just stored
            result = json.loads(backend.get_quota())
            if result.get("ok"):
                return json.dumps({"ok": True, "message": "LINE credentials saved (Muse-auth)."})
            error = json.dumps({"ok": False, "error": str(result)})
        else:  # pragma: no cover - defense in depth
            error = json.dumps({"ok": False, "error": backend._connection_info})
    except Exception as e:
        error = json.dumps({"ok": False, "error": str(e)})
    # Roll the vault back so a bad token is not left enrolled.
    with contextlib.suppress(Exception):
        clear_credentials("line")
    backend._api = None
    return error


class LineAgent(BaseChannelAgent):
    """Channel agent with LINE Messaging API tools."""

    def __init__(self) -> None:
        super().__init__("LINE Agent")
        self._backend = LineChannelBackend()
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        if muse_auth_enabled():
            from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError

            # Muse-auth mode: wire a vault surrogate and the boundary
            # adapter (no network round trip); the real token never
            # enters this process once migrated.  A daemon failure
            # leaves the agent constructible (fail closed, tokenless)
            # so its authenticate/clear tools stay available.
            try:
                self._backend._wire_muse()
            except MuseAuthError as e:
                self._backend._api = None
                self._backend._connection_info = f"Muse-auth wiring failed: {e}"
            return
        cfg = _config.load()
        if cfg:  # pragma: no branch
            try:
                self._backend._api = _SdkLineApi(cfg["channel_access_token"])
            except Exception:
                pass

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return self._backend._api is not None

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_line_auth() -> str:
            """Check if LINE credentials are configured and valid.

            Returns:
                Authentication status or instructions.
            """
            if agent._backend._api is None:  # pragma: no branch
                return (
                    "Not authenticated with LINE. Use authenticate_line(channel_access_token=...) "
                    "to configure. Get a token from https://developers.line.biz/console/\n"
                    "Create a Messaging API channel, then find the Channel access token "
                    "on the channel's 'Messaging API' tab."
                )
            try:
                quota = json.loads(agent._backend.get_quota())
                if quota.get("ok"):  # pragma: no branch
                    return json.dumps({"ok": True, "quota": quota})
                return json.dumps({"ok": True, "message": "LINE authenticated."})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_line(channel_access_token: str, channel_secret: str = "") -> str:
            """Store and validate LINE channel credentials.

            Args:
                channel_access_token: LINE channel access token from Developers Console.
                channel_secret: LINE channel secret (optional, for webhook verification).

            Returns:
                Validation result or error message.
            """
            if not channel_access_token.strip():  # pragma: no branch
                return "channel_access_token cannot be empty."
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if muse_auth_enabled():
                return _muse_authenticate(
                    agent._backend, channel_access_token.strip(), channel_secret.strip()
                )
            try:
                agent._backend._api = _SdkLineApi(channel_access_token.strip())
                _config.save(
                    {
                        "channel_access_token": channel_access_token.strip(),
                        "channel_secret": channel_secret.strip(),
                    }
                )
                return json.dumps({"ok": True, "message": "LINE credentials saved."})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def clear_line_auth() -> str:
            """Clear the stored LINE credentials.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._api = None
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if muse_auth_enabled():
                from kiss.agents.third_party_agents.muse_auth.client import clear_credentials

                clear_credentials("line")
            return "LINE authentication cleared."

        return [check_line_auth, authenticate_line, clear_line_auth]


def _make_backend() -> LineChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = LineChannelBackend()
    from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

    if muse_auth_enabled():
        if backend._wire_muse():
            return backend
        print("Not authenticated. Run: kiss-line -t 'authenticate'")
        sys.exit(1)
    cfg = _config.load()
    if not cfg:  # pragma: no branch
        print("Not authenticated. Run: kiss-line -t 'authenticate'")
        sys.exit(1)
    backend._api = _SdkLineApi(cfg["channel_access_token"])
    return backend


def main() -> None:
    """Run the LineAgent from the command line with chat persistence."""
    channel_main(
        LineAgent,
        "kiss-line",
        channel_name="LINE",
        make_backend=_make_backend,
    )


def tools() -> list:
    """Return the LINE channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return LineAgent()._get_tools()


if __name__ == "__main__":
    main()
