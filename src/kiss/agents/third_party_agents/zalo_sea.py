# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Zalo Agent — channel agent with Zalo Official Account API tools.

Provides authenticated access to Zalo OA via access token. Covers both
extensions/zalo/ (OA API) and extensions/zalouser/ (personal). Stores
config in ``~/.kiss/third_party_agents/zalo/config.json``.

Usage::

    agent = ZaloAgent()
    agent.run(prompt_template="Get OA info")
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

import requests

from kiss.agents.third_party_agents._backend_utils import (
    ThreadedHTTPServer,
    drain_queue_messages,
    start_http_server,
    stop_http_server,
)
from kiss.agents.third_party_agents._browser_handoff import portal_handoff
from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

logger = logging.getLogger(__name__)

_DEFAULT_WEBHOOK_PORT = 18082

_ZALO_DIR = Path.home() / ".kiss" / "third_party_agents" / "zalo"
_API_BASE = "https://openapi.zalo.me/v2.0/oa"
_config = ChannelConfig(_ZALO_DIR, ("access_token",))


def _scrub_config_token() -> None:
    """Remove a vault-migrated ``access_token`` from config.json.

    Finishes the Muse migration automatically: the non-secret ``oa_id``
    metadata is kept and the file is deleted when nothing but the token
    was stored.
    """
    try:
        cfg = json.loads(_config.path.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(cfg, dict) or "access_token" not in cfg:
        return
    kept = {k: str(v) for k, v in cfg.items() if k != "access_token" and v}
    if kept:
        from kiss.agents.third_party_agents._channel_agent_utils import save_json_config

        save_json_config(_config.path, kept)
    else:
        _config.clear()


class ZaloChannelBackend(ToolMethodBackend):
    """Channel backend for Zalo OA API.

    Uses webhook queue pattern for receiving inbound messages.
    """

    def __init__(self, api_base: str = "") -> None:
        self._api_base_override = api_base
        self._access_token: str = ""
        self._oa_id: str = ""
        self._http: Any = requests
        self._muse: bool = False
        self._message_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._webhook_server: ThreadedHTTPServer | None = None
        self._webhook_thread: threading.Thread | None = None
        self._connection_info: str = ""


    def _api_base(self) -> str:
        """Return the OA API base URL, resolved at call time.

        Precedence: the constructor override, then ``$ZALO_API_BASE``,
        then the module-level ``_API_BASE`` — read per call so tests
        that repoint the module global after constructing a backend
        still take effect.

        Returns:
            The API base URL string.
        """
        return self._api_base_override or os.environ.get("ZALO_API_BASE") or _API_BASE

    def _headers(self) -> dict[str, str]:
        if self._muse:
            # ``_access_token`` holds a surrogate: the daemon swaps this
            # bearer for the real ``access_token`` header (a header-kind
            # vault credential) at the network boundary.
            return {"Authorization": f"Bearer {self._access_token}"}
        return {"access_token": self._access_token}

    def _wire_muse(self) -> bool:
        """Acquire a Zalo surrogate and wire the boundary session.

        An ``access_token`` still in the legacy config is the newest
        user intent (initial migration, or a rotation done while Muse
        was off): it is enrolled as a header-kind credential (Zalo sends
        the token in an ``access_token`` request header) replacing any
        vault entry, and scrubbed from ``config.json`` only after the
        vault holds it.  No network round trip happens here.

        Returns:
            True when the backend holds a surrogate and boundary session.
        """
        from kiss.agents.third_party_agents.muse_auth.client import (
            MuseBoundarySession,
            mint_surrogate,
            store_credentials,
        )

        cfg = _config.load_metadata() or {}
        token = cfg.get("access_token", "")
        if token:
            store_credentials(
                "zalo", {"kind": "header", "header": "access_token", "token": token}, []
            )
        handle = mint_surrogate("zalo")
        if handle is None:
            self._connection_info = "No Zalo credential in the Muse vault or config."
            return False
        _scrub_config_token()
        self._oa_id = cfg.get("oa_id", "")
        self._access_token = handle.token
        self._http = MuseBoundarySession("zalo")
        self._muse = True
        return True

    def connect(self) -> bool:
        """Load Zalo config and start webhook server."""
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        if muse_auth_enabled():
            if not self._wire_muse():
                return False
        else:
            cfg = _config.load()
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
                        backend._message_queue.put(
                            {
                                "ts": str(data.get("timestamp", "")),
                                "user": sender.get("id", ""),
                                "text": message.get("text", ""),
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
            started_log="Zalo webhook server started on port %d",
            error_prefix="Zalo webhook bind failed",
            error_log="Could not start Zalo webhook server: %s",
        )
        if error is not None:
            self._connection_info = error
            return False
        return True

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Drain the webhook message queue, filtered by sender when given."""

        def keep(msg: dict[str, Any]) -> bool:
            return not channel_id or msg.get("user", "") == channel_id

        return drain_queue_messages(self._message_queue, limit=limit, keep=keep), oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a Zalo text message.

        Raises:
            RuntimeError: If the Zalo API reports a send failure.
        """
        result = json.loads(self.send_text_message(channel_id, text))
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Zalo send failed"))

    def disconnect(self) -> None:
        """Stop the embedded webhook server and release backend resources."""
        self._webhook_server, self._webhook_thread = stop_http_server(
            self._webhook_server, self._webhook_thread
        )

    def send_text_message(self, to_user_id: str, text: str) -> str:
        """Send a text message to a Zalo user.

        Args:
            to_user_id: Zalo user ID.
            text: Message text.

        Returns:
            JSON string with ok status.
        """
        try:
            resp = self._http.post(
                f"{self._api_base()}/message/text",
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

    def send_image_message(self, to_user_id: str, image_url: str, caption: str = "") -> str:
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
            resp = self._http.post(
                f"{self._api_base()}/message",
                headers=self._headers(),
                json={"recipient": {"user_id": to_user_id}, "message": msg},
                timeout=30,
            )
            data = resp.json()
            return json.dumps(
                {
                    "ok": data.get("error") == 0,
                    "message": data.get("message", ""),
                }
            )
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
            resp = self._http.get(
                f"{self._api_base()}/getprofile",
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
            resp = self._http.get(
                f"{self._api_base()}/getfollowers",
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
            resp = self._http.get(
                f"{self._api_base()}/getoa",
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
            resp = self._http.get(
                f"{self._api_base()}/listrecentchat",
                headers=self._headers(),
                params={"offset": offset, "count": count},
                timeout=30,
            )
            data = resp.json()
            if data.get("error") == 0:  # pragma: no branch
                return json.dumps({"ok": True, "conversations": data.get("data", {})}, indent=2)[
                    :8000
                ]
            return json.dumps({"ok": False, "error": data.get("message", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_conversation(self, user_id: str, offset: int = 0, count: int = 20) -> str:
        """Get conversation history with a specific user.

        Args:
            user_id: Zalo user ID.
            offset: Pagination offset. Default: 0.
            count: Number of messages. Default: 20.

        Returns:
            JSON string with conversation messages.
        """
        try:
            resp = self._http.get(
                f"{self._api_base()}/conversation",
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
                resp = self._http.post(
                    f"{self._api_base()}/upload/image",
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


def _muse_authenticate(backend: ZaloChannelBackend, access_token: str, oa_id: str) -> str:
    """Enroll a Zalo OA token into the Muse vault and validate it.

    The plaintext token goes straight into the vault as a header-kind
    credential (Zalo's ``access_token`` request header), atomically
    replacing any previous enrollment, and is never written to
    ``config.json`` (only the non-secret ``oa_id`` metadata is — written
    first, so a failed enrollment leaves no token on disk).  Validation
    runs ``/getoa`` through the daemon boundary, so it is audited; an
    invalid token leaves the vault empty.

    Args:
        backend: The agent's Zalo backend to (re)wire.
        access_token: Zalo OA access token.
        oa_id: Optional Official Account ID metadata.

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
        if oa_id:
            save_json_config(_config.path, {"oa_id": oa_id})
        else:
            _config.clear()
        store_credentials(
            "zalo", {"kind": "header", "header": "access_token", "token": access_token}, []
        )
        if backend._wire_muse():  # pragma: no branch - credential was just stored
            result = json.loads(backend.get_oa_info())
            if result.get("ok"):
                return json.dumps({"ok": True, "message": "Zalo credentials saved (Muse-auth)."})
            error = json.dumps({"ok": False, "error": "Could not verify credentials."})
        else:  # pragma: no cover - defense in depth
            error = json.dumps({"ok": False, "error": backend._connection_info})
    except Exception as e:
        error = json.dumps({"ok": False, "error": str(e)})
    # Roll the vault back so a bad token is not left enrolled.
    with contextlib.suppress(Exception):
        clear_credentials("zalo")
    backend._access_token = ""
    backend._http = requests
    backend._muse = False
    return error


class ZaloAgent(BaseChannelAgent):
    """Channel agent with Zalo OA API tools."""

    def __init__(self) -> None:
        super().__init__("Zalo Agent")
        self._backend = ZaloChannelBackend()
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        if muse_auth_enabled():
            from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError

            # Muse-auth mode: wire a vault surrogate and the boundary
            # session (no network round trip); the real token never
            # enters this process once migrated.  A daemon failure
            # leaves the agent constructible (fail closed, tokenless)
            # so its authenticate/clear tools stay available.
            try:
                self._backend._wire_muse()
            except MuseAuthError as e:
                self._backend._access_token = ""
                self._backend._connection_info = f"Muse-auth wiring failed: {e}"
            return
        cfg = _config.load()
        if cfg:  # pragma: no branch
            self._backend._access_token = cfg["access_token"]
            self._backend._oa_id = cfg.get("oa_id", "")

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._access_token)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_zalo_auth() -> str:
            """Check if Zalo credentials are configured and valid.

            Returns:
                Authentication status or instructions.
            """
            if not agent._backend._access_token:  # pragma: no branch
                return (
                    "Not authenticated with Zalo. Use authenticate_zalo(access_token=...) "
                    "to configure. Get a token from https://developers.zalo.me/ — "
                    "create an Official Account app and find the access token in its settings."
                    + "\n"
                    + portal_handoff("https://developers.zalo.me/")
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
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if muse_auth_enabled():
                return _muse_authenticate(agent._backend, access_token.strip(), oa_id.strip())
            agent._backend._access_token = access_token.strip()
            agent._backend._oa_id = oa_id.strip()
            try:
                result = json.loads(agent._backend.get_oa_info())
                if result.get("ok"):  # pragma: no branch
                    _config.save({"access_token": access_token.strip(), "oa_id": oa_id.strip()})
                    return json.dumps({"ok": True, "message": "Zalo credentials saved."})
                return json.dumps({"ok": False, "error": "Could not verify credentials."})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def clear_zalo_auth() -> str:
            """Clear the stored Zalo credentials.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._access_token = ""
            agent._backend._oa_id = ""
            agent._backend._http = requests
            agent._backend._muse = False
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if muse_auth_enabled():
                from kiss.agents.third_party_agents.muse_auth.client import clear_credentials

                clear_credentials("zalo")
            return "Zalo authentication cleared."

        return [check_zalo_auth, authenticate_zalo, clear_zalo_auth]


def _make_backend() -> ZaloChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = ZaloChannelBackend()
    from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

    if muse_auth_enabled():
        if backend._wire_muse():
            return backend
        print("Not authenticated. Run: kiss-zalo -t 'authenticate'")
        sys.exit(1)
    cfg = _config.load()
    if not cfg:  # pragma: no branch
        print("Not authenticated. Run: kiss-zalo -t 'authenticate'")
        sys.exit(1)
    backend._access_token = cfg["access_token"]
    backend._oa_id = cfg.get("oa_id", "")
    return backend


def main() -> None:
    """Run the ZaloAgent from the command line with chat persistence."""
    channel_main(
        ZaloAgent,
        "kiss-zalo",
        channel_name="Zalo",
        make_backend=_make_backend,
    )


def tools() -> list:
    """Return the Zalo channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return ZaloAgent()._get_tools()


if __name__ == "__main__":
    main()
