# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""BlueBubbles Agent — channel agent with BlueBubbles REST API tools.

Provides access to iMessage via the BlueBubbles server running on a local Mac.
macOS only. Stores config in ``~/.kiss/third_party_agents/bluebubbles/config.json``.

Usage::

    agent = BlueBubblesAgent()
    agent.run(prompt_template="List recent iMessage conversations")
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

_BB_DIR = Path.home() / ".kiss" / "third_party_agents" / "bluebubbles"

_PLATFORM_ERROR = json.dumps(
    {
        "ok": False,
        "error": "BlueBubbles requires macOS with a running BlueBubbles server.",
    }
)
_config = ChannelConfig(
    _BB_DIR,
    (
        "server_url",
        "password",
    ),
)


def _scrub_config_password() -> None:
    """Remove a vault-migrated ``password`` from config.json.

    Finishes the Muse migration automatically: the non-secret
    ``server_url`` metadata is kept and the file is deleted when
    nothing but the password was stored.
    """
    try:
        cfg = json.loads(_config.path.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(cfg, dict) or "password" not in cfg:
        return
    kept = {k: str(v) for k, v in cfg.items() if k != "password" and v}
    if kept:
        from kiss.agents.third_party_agents._channel_agent_utils import save_json_config

        save_json_config(_config.path, kept)
    else:
        _config.clear()


class BlueBubblesChannelBackend(ToolMethodBackend):
    """Channel backend for BlueBubbles REST API."""

    def __init__(self) -> None:
        self._server_url: str = ""
        self._password: str = ""
        self._http: Any = requests
        self._muse: bool = False
        self._surrogate: str = ""
        self._last_ts: float = 0.0
        self._connection_info: str = ""

    def _url(self, path: str) -> str:
        return f"{self._server_url}{path}"

    def _params(self) -> dict[str, str]:
        # In Muse mode the password never enters this process: the
        # daemon splices the real ``password=`` query parameter (a
        # query-kind vault credential) into the URL at the boundary.
        if self._muse:
            return {}
        return {"password": self._password}

    def _headers(self) -> dict[str, str]:
        """Return per-request headers: the surrogate bearer in Muse mode.

        Returns:
            ``{"Authorization": "Bearer <surrogate>"}`` in Muse mode
            (the daemon validates and strips it), else ``{}``.
        """
        if self._muse:
            return {"Authorization": f"Bearer {self._surrogate}"}
        return {}

    def _wire_muse(self) -> bool:
        """Acquire a BlueBubbles surrogate and wire the boundary session.

        A ``password`` still in the legacy config is the newest user
        intent (initial migration, or a rotation done while Muse was
        off): it is enrolled as a query-kind credential (BlueBubbles
        authenticates with a ``password=`` query parameter) bound to the
        configured server origin (flagged as a consent-scoped insecure
        host when the URL is plain ``http`` to a non-loopback host), and
        scrubbed from ``config.json`` only after the vault holds it.
        No network round trip happens here.

        Returns:
            True when the backend holds a surrogate and boundary session.
        """
        from kiss.agents.third_party_agents.muse_auth._common import (
            insecure_origin_hosts,
            origin_hosts,
            valid_http_url,
        )
        from kiss.agents.third_party_agents.muse_auth.client import (
            MuseBoundarySession,
            mint_surrogate,
            store_credentials,
        )

        cfg = _config.load_metadata() or {}
        server_url = str(cfg.get("server_url") or "").rstrip("/")
        if not server_url:
            self._connection_info = "No BlueBubbles config found."
            return False
        # Validate before any credential state changes: a malformed
        # legacy URL must not auto-migrate the password into a host
        # scope Sentinel can never match, nor scrub the plaintext copy.
        if not valid_http_url(server_url):
            self._connection_info = (
                f"BlueBubbles server URL {server_url!r} is not a valid http(s):// URL; "
                "fix config.json and reconnect."
            )
            return False
        password = str(cfg.get("password") or "")
        if password:
            store_credentials(
                "bluebubbles",
                {"kind": "query", "param": "password", "token": password},
                [],
                hosts=origin_hosts(server_url),
                insecure_hosts=insecure_origin_hosts(server_url),
            )
        handle = mint_surrogate("bluebubbles")
        if handle is None:
            self._connection_info = "No BlueBubbles credential in the Muse vault or config."
            return False
        _scrub_config_password()
        self._server_url = server_url
        self._surrogate = handle.token
        self._http = MuseBoundarySession("bluebubbles")
        self._muse = True
        return True

    def connect(self) -> bool:
        """Connect to BlueBubbles server."""
        if sys.platform != "darwin":  # pragma: no branch
            self._connection_info = "BlueBubbles requires macOS."
            return False
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        if muse_auth_enabled():
            # Vault-first surrogate wiring; validation below runs the
            # /server/info read through the daemon boundary (audited).
            if not self._wire_muse():
                return False
        else:
            cfg = _config.load()
            if not cfg:  # pragma: no branch
                self._connection_info = "No BlueBubbles config found."
                return False
            self._server_url = cfg["server_url"].rstrip("/")
            self._password = cfg["password"]
        try:
            resp = self._http.get(
                self._url("/api/v1/server/info"),
                params=self._params(),
                headers=self._headers(),
                timeout=10,
            )
            data = resp.json()
            if data.get("status") == 200:  # pragma: no branch
                self._connection_info = f"Connected to BlueBubbles at {self._server_url}"
                self._last_ts = time.time() * 1000
                return True
            self._connection_info = f"BlueBubbles auth failed: {data}"
            return False
        except Exception as e:
            self._connection_info = f"BlueBubbles connection failed: {e}"
            return False

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll BlueBubbles for new messages in a chat.

        ``oldest`` is a millisecond ``dateCreated`` cursor; ``""``/``"0"``
        means fetch the most recent messages without an after-filter.
        Returns the messages (oldest first) filtered to ``channel_id`` and
        the max ``dateCreated`` seen as the new cursor.
        """
        try:
            cursor = int(float(oldest)) if oldest not in ("", "0") else 0
            body: dict[str, Any] = {"limit": limit, "with": ["chat"], "sort": "DESC"}
            if channel_id:  # pragma: no branch
                body["chatGuid"] = channel_id
            if cursor:
                body["after"] = cursor
            resp = self._http.post(
                self._url("/api/v1/message/query"),
                params=self._params(),
                headers=self._headers(),
                json=body,
                timeout=10,
            )
            data = resp.json()
            new_cursor = cursor
            messages: list[dict[str, Any]] = []
            for msg in data.get("data", []):  # pragma: no branch
                ts = int(msg.get("dateCreated", 0) or 0)
                new_cursor = max(new_cursor, ts)
                if ts > self._last_ts:  # pragma: no branch
                    self._last_ts = ts
                chats = msg.get("chats") or []
                chat_guid = chats[0].get("guid", "") if chats else channel_id
                messages.append(
                    {
                        "ts": str(ts),
                        "user": (msg.get("sender") or {}).get("address", ""),
                        "text": msg.get("text", "") or "",
                        "guid": msg.get("guid", ""),
                        "chat_guid": chat_guid,
                        "is_from_me": bool(msg.get("isFromMe")),
                    }
                )
            messages.reverse()
            return messages, str(new_cursor) if new_cursor else oldest
        except Exception:
            return [], oldest

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Check whether a polled message was sent by the bot itself (isFromMe).

        Args:
            msg: Message dict from :meth:`poll_messages`.

        Returns:
            True if the message was sent from this account.
        """
        return bool(msg.get("is_from_me"))

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a BlueBubbles message.

        Raises:
            RuntimeError: If the server reports a non-200 status.
        """
        resp = self._http.post(
            self._url("/api/v1/message/text"),
            params=self._params(),
                headers=self._headers(),
            json={"chatGuid": channel_id, "message": text, "method": "private-api"},
            timeout=30,
        )
        data = resp.json()
        if resp.status_code >= 400 or data.get("status") != 200:
            raise RuntimeError(f"BlueBubbles send failed: {data}")

    def list_chats(self, limit: int = 25, offset: int = 0) -> str:
        """List recent iMessage conversations.

        Args:
            limit: Maximum chats to return. Default: 25.
            offset: Pagination offset. Default: 0.

        Returns:
            JSON string with chat list.
        """
        if sys.platform != "darwin":  # pragma: no branch
            return _PLATFORM_ERROR
        try:
            resp = self._http.get(
                self._url("/api/v1/chat"),
                params={**self._params(), "limit": str(limit), "offset": str(offset)},
                headers=self._headers(),
                timeout=10,
            )
            data = resp.json()
            chats = [
                {
                    "guid": c.get("guid", ""),
                    "display_name": c.get("displayName", ""),
                    "participants": [p.get("address", "") for p in c.get("participants", [])],
                }
                for c in data.get("data", [])
            ]
            return json.dumps({"ok": True, "chats": chats}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_chat(self, chat_guid: str) -> str:
        """Get a specific iMessage conversation.

        Args:
            chat_guid: Chat GUID (from list_chats).

        Returns:
            JSON string with chat details.
        """
        if sys.platform != "darwin":  # pragma: no branch
            return _PLATFORM_ERROR
        try:
            resp = self._http.get(
                self._url(f"/api/v1/chat/{chat_guid}"), params=self._params(),
                headers=self._headers(), timeout=10
            )
            return json.dumps({"ok": True, "chat": resp.json().get("data", {})}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_chat_messages(
        self, chat_guid: str, limit: int = 25, before: str = "", after: str = ""
    ) -> str:
        """Get messages from a specific conversation.

        Args:
            chat_guid: Chat GUID.
            limit: Maximum messages to return. Default: 25.
            before: Return messages before this timestamp (ms).
            after: Return messages after this timestamp (ms).

        Returns:
            JSON string with message list.
        """
        if sys.platform != "darwin":  # pragma: no branch
            return _PLATFORM_ERROR
        try:
            params: dict[str, Any] = {**self._params(), "limit": limit}
            if before:  # pragma: no branch
                params["before"] = before
            if after:  # pragma: no branch
                params["after"] = after
            resp = self._http.get(
                self._url(f"/api/v1/chat/{chat_guid}/message"),
                params=params,
                headers=self._headers(),
                timeout=10
            )
            messages = [
                {
                    "guid": m.get("guid", ""),
                    "text": m.get("text", "") or "",
                    "sender": m.get("sender", {}).get("address", ""),
                    "date_created": m.get("dateCreated", 0),
                    "is_from_me": m.get("isFromMe", False),
                }
                for m in resp.json().get("data", [])
            ]
            return json.dumps({"ok": True, "messages": messages}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def post_message(self, chat_guid: str, text: str) -> str:
        """Send a message to an iMessage conversation.

        Args:
            chat_guid: Chat GUID to send to.
            text: Message text.

        Returns:
            JSON string with ok status.
        """
        if sys.platform != "darwin":  # pragma: no branch
            return _PLATFORM_ERROR
        try:
            resp = self._http.post(
                self._url("/api/v1/message/text"),
                params=self._params(),
                headers=self._headers(),
                json={"chatGuid": chat_guid, "message": text, "method": "private-api"},
                timeout=30,
            )
            data = resp.json()
            if data.get("status") == 200:  # pragma: no branch
                return json.dumps({"ok": True})
            return json.dumps({"ok": False, "error": str(data)})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_server_info(self) -> str:
        """Get BlueBubbles server information.

        Returns:
            JSON string with server info.
        """
        if sys.platform != "darwin":  # pragma: no branch
            return _PLATFORM_ERROR
        try:
            resp = self._http.get(
                self._url("/api/v1/server/info"),
                params=self._params(),
                headers=self._headers(),
                timeout=10,
            )
            data = resp.json()
            # BlueBubbles wraps every response in a numeric ``status``
            # envelope; a decodable body alone (e.g. an HTTP 401 with
            # {"status": 401, "message": "bad password"}) is a failure.
            if resp.status_code == 200 and data.get("status") == 200:
                return json.dumps({"ok": True, "info": data.get("data", {})}, indent=2)[:8000]
            return json.dumps(
                {"ok": False, "error": f"HTTP {resp.status_code}: {str(data)[:500]}"}
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def mark_chat_read(self, chat_guid: str) -> str:
        """Mark a chat as read.

        Args:
            chat_guid: Chat GUID to mark as read.

        Returns:
            JSON string with ok status.
        """
        if sys.platform != "darwin":  # pragma: no branch
            return _PLATFORM_ERROR
        try:
            resp = self._http.post(
                self._url(f"/api/v1/chat/{chat_guid}/read"), params=self._params(),
                headers=self._headers(), timeout=10
            )
            return json.dumps({"ok": resp.json().get("status") == 200})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


def _muse_authenticate(backend: BlueBubblesChannelBackend, server_url: str, password: str) -> str:
    """Enroll a BlueBubbles password into the Muse vault and validate it.

    The plaintext password goes straight into the vault as a query-kind
    credential (the daemon splices ``password=`` into the URL at the
    boundary) bound to the configured server origin, and is never
    written to ``config.json`` (only the non-secret ``server_url``
    metadata is — written first, so a failed enrollment leaves no
    password on disk).  Validation runs the ``/server/info`` read
    through the daemon boundary, so it is audited; an invalid password
    leaves the vault empty.

    Args:
        backend: The agent's BlueBubbles backend to (re)wire.
        server_url: BlueBubbles server base URL.
        password: BlueBubbles server password.

    Returns:
        JSON string with the validation result.
    """
    import contextlib

    from kiss.agents.third_party_agents._channel_agent_utils import save_json_config
    from kiss.agents.third_party_agents.muse_auth._common import (
        insecure_origin_hosts,
        origin_hosts,
        valid_http_url,
    )
    from kiss.agents.third_party_agents.muse_auth.client import (
        clear_credentials,
        store_credentials,
    )

    if not valid_http_url(server_url):
        return json.dumps(
            {"ok": False, "error": f"{server_url!r} is not a valid http(s):// server URL."}
        )
    try:
        save_json_config(_config.path, {"server_url": server_url})
        store_credentials(
            "bluebubbles",
            {"kind": "query", "param": "password", "token": password},
            [],
            hosts=origin_hosts(server_url),
            insecure_hosts=insecure_origin_hosts(server_url),
        )
        if backend._wire_muse():  # pragma: no branch - credential was just stored
            result = json.loads(backend.get_server_info())
            if result.get("ok"):
                return json.dumps({"ok": True, "message": "BlueBubbles configured (Muse-auth)."})
            error = json.dumps(
                {"ok": False, "error": "Could not connect to BlueBubbles server."}
            )
        else:  # pragma: no cover - defense in depth
            error = json.dumps({"ok": False, "error": backend._connection_info})
    except Exception as e:
        error = json.dumps({"ok": False, "error": str(e)})
    # Roll the vault back so a bad password is not left enrolled.
    with contextlib.suppress(Exception):
        clear_credentials("bluebubbles")
    backend._server_url = ""
    backend._surrogate = ""
    backend._http = requests
    backend._muse = False
    return error


class BlueBubblesAgent(BaseChannelAgent):
    """Channel agent with BlueBubbles REST API tools (macOS only)."""

    def __init__(self) -> None:
        super().__init__("BlueBubbles Agent")
        self._backend = BlueBubblesChannelBackend()
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        if muse_auth_enabled():
            from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError

            # Muse-auth mode: wire a vault surrogate and the boundary
            # session (no network round trip); the real password never
            # enters this process once migrated.  A daemon failure
            # leaves the agent constructible (fail closed) so its
            # authenticate/clear tools stay available.
            try:
                self._backend._wire_muse()
            except MuseAuthError as e:
                self._backend._server_url = ""
                self._backend._connection_info = f"Muse-auth wiring failed: {e}"
            return
        cfg = _config.load()
        if cfg:  # pragma: no branch
            self._backend._server_url = cfg["server_url"].rstrip("/")
            self._backend._password = cfg["password"]

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._server_url)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_bluebubbles_auth() -> str:
            """Check if BlueBubbles is configured and reachable.

            Returns:
                Connection status or instructions.
            """
            if not agent._backend._server_url:  # pragma: no branch
                return (
                    "Not configured for BlueBubbles. "
                    "Use authenticate_bluebubbles(server_url=..., password=...) "
                    "to configure.\n"
                    "Requires BlueBubbles server (https://bluebubbles.app) running on a Mac. "
                    "Find the server URL and password in the BlueBubbles server app settings."
                )
            try:
                result = json.loads(agent._backend.get_server_info())
                if result.get("ok"):  # pragma: no branch
                    return json.dumps(
                        {
                            "ok": True,
                            "server_url": agent._backend._server_url,
                        }
                    )
                return json.dumps({"ok": False, "error": result.get("error", "Unknown error")})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_bluebubbles(server_url: str, password: str) -> str:
            """Configure BlueBubbles connection.

            Args:
                server_url: BlueBubbles server URL (e.g. "http://localhost:1234").
                password: BlueBubbles server password.

            Returns:
                Connection result or error message.
            """
            if sys.platform != "darwin":  # pragma: no branch
                return _PLATFORM_ERROR
            for val, name in [(server_url, "server_url"), (password, "password")]:
                if not val.strip():  # pragma: no branch
                    return f"{name} cannot be empty."
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if muse_auth_enabled():
                return _muse_authenticate(
                    agent._backend, server_url.strip().rstrip("/"), password.strip()
                )
            agent._backend._server_url = server_url.strip().rstrip("/")
            agent._backend._password = password.strip()
            result = json.loads(agent._backend.get_server_info())
            if result.get("ok"):  # pragma: no branch
                _config.save({"server_url": server_url.strip(), "password": password.strip()})
                return json.dumps({"ok": True, "message": "BlueBubbles configured."})
            return json.dumps({"ok": False, "error": "Could not connect to BlueBubbles server."})

        def clear_bluebubbles_auth() -> str:
            """Clear the stored BlueBubbles configuration.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._server_url = ""
            agent._backend._password = ""
            agent._backend._surrogate = ""
            agent._backend._http = requests
            agent._backend._muse = False
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if muse_auth_enabled():
                from kiss.agents.third_party_agents.muse_auth.client import clear_credentials

                clear_credentials("bluebubbles")
            return "BlueBubbles configuration cleared."

        return [check_bluebubbles_auth, authenticate_bluebubbles, clear_bluebubbles_auth]


def _make_backend() -> BlueBubblesChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = BlueBubblesChannelBackend()
    from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

    if muse_auth_enabled():
        if backend._wire_muse():
            return backend
        print("Not configured. Run: kiss-bluebubbles -t 'authenticate'")
        sys.exit(1)
    cfg = _config.load()
    if not cfg:  # pragma: no branch
        print("Not configured. Run: kiss-bluebubbles -t 'authenticate'")
        sys.exit(1)
    backend._server_url = cfg["server_url"].rstrip("/")
    backend._password = cfg["password"]
    return backend


def main() -> None:
    """Run the BlueBubblesAgent from the command line with chat persistence."""
    channel_main(
        BlueBubblesAgent,
        "kiss-bluebubbles",
        channel_name="BlueBubbles",
        make_backend=_make_backend,
    )


def tools() -> list:
    """Return the BlueBubbles channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return BlueBubblesAgent()._get_tools()


if __name__ == "__main__":
    main()
