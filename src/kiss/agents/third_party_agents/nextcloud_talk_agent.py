# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Nextcloud Talk Agent — channel agent with Nextcloud Talk API tools.

Provides authenticated access to Nextcloud Talk via username/password.
Stores config in ``~/.kiss/third_party_agents/nextcloud/config.json``.

Usage::

    agent = NextcloudTalkAgent()
    agent.run(prompt_template="List all rooms")
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import requests

from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
    save_json_config,
)

logger = logging.getLogger(__name__)

_NEXTCLOUD_DIR = Path.home() / ".kiss" / "third_party_agents" / "nextcloud"
_config = ChannelConfig(
    _NEXTCLOUD_DIR,
    (
        "url",
        "username",
        "password",
    ),
)


def _basic_credential(username: str, password: str) -> str:
    """Return the ``Authorization: Basic`` value for a username/password.

    Nextcloud authenticates with HTTP Basic auth, which is one header —
    so the Muse vault stores it as a header-kind credential and the
    daemon emits exactly this value at the boundary.

    Args:
        username: Nextcloud login name.
        password: Nextcloud password or app password.

    Returns:
        ``Basic <base64(username:password)>``.
    """
    import base64

    return "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()


def _scrub_config_password() -> None:
    """Remove a vault-migrated ``password`` from config.json.

    Finishes the Muse migration automatically: the non-secret ``url``/
    ``username`` metadata is kept (``username`` is needed to recognize
    the bot's own messages) and the file is deleted when nothing but
    the password was stored.
    """
    try:
        cfg = json.loads(_config.path.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(cfg, dict) or "password" not in cfg:
        return
    kept = {k: str(v) for k, v in cfg.items() if k != "password" and v}
    if kept:
        save_json_config(_config.path, kept)
    else:
        _config.clear()


class NextcloudTalkChannelBackend(ToolMethodBackend):
    """Channel backend for Nextcloud Talk REST API."""

    def __init__(self) -> None:
        self._url: str = ""
        self._auth: tuple[str, str] = ("", "")
        self._http: Any = requests
        self._muse: bool = False
        self._surrogate: str = ""
        self._last_message_id: int = 0
        self._connection_info: str = ""

    def _base(self) -> str:
        return f"{self._url}/ocs/v2.php/apps/spreed/api/v4"

    def _headers(self) -> dict[str, str]:
        return {"OCS-APIRequest": "true", "Accept": "application/json"}

    def _auth_kwargs(self) -> dict[str, Any]:
        """Return the per-request credential kwargs for the current mode.

        Legacy mode sends HTTP Basic auth directly; Muse mode sends a
        surrogate bearer that the daemon swaps for the real
        ``Authorization: Basic`` header at the network boundary.

        Returns:
            Keyword arguments carrying headers (and legacy ``auth``).
        """
        if self._muse:
            return {
                "headers": {**self._headers(), "Authorization": f"Bearer {self._surrogate}"}
            }
        return {"auth": self._auth, "headers": self._headers()}

    def _validate_credentials(self) -> tuple[bool, str]:
        """Check the credentials with a ``/room`` read, strictly.

        OCS error responses also arrive inside an ``ocs`` envelope (an
        HTTP 401 carries ``meta.status == "failure"``), so envelope
        presence proves nothing: both the HTTP status and the OCS meta
        statuscode must signal success.

        Returns:
            ``(True, "")`` when the credentials are valid, else
            ``(False, detail)`` with a credential-free failure detail.
        """
        resp = self._http.get(f"{self._base()}/room", timeout=30, **self._auth_kwargs())
        try:
            meta = resp.json().get("ocs", {}).get("meta", {})
        except ValueError:
            meta = {}
        statuscode = meta.get("statuscode")
        if resp.status_code == 200 and statuscode in (200, 201):
            return True, ""
        return False, f"HTTP {resp.status_code}, OCS statuscode {statuscode!r}"

    def _get(self, path: str, params: dict | None = None) -> dict[str, Any]:  # type: ignore[type-arg]
        resp = self._http.get(
            f"{self._base()}{path}",
            params=params,
            timeout=30,
            **self._auth_kwargs(),
        )
        return resp.json()  # type: ignore[no-any-return]

    def _post(self, path: str, data: dict | None = None) -> dict[str, Any]:  # type: ignore[type-arg]
        resp = self._http.post(
            f"{self._base()}{path}",
            json=data,
            timeout=30,
            **self._auth_kwargs(),
        )
        return resp.json()  # type: ignore[no-any-return]

    def _wire_muse(self) -> bool:
        """Acquire a Nextcloud surrogate and wire the boundary session.

        A ``password`` still in the legacy config is the newest user
        intent (initial migration, or a rotation done while Muse was
        off): together with ``username`` it is enrolled as a header-kind
        Basic credential bound to the configured server origin (flagged
        as a consent-scoped insecure host when the URL is plain
        ``http``), and scrubbed from ``config.json`` only after the
        vault holds it.  No network round trip happens here.

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
        url = str(cfg.get("url") or "").rstrip("/")
        if not url:
            self._connection_info = "No Nextcloud config found."
            return False
        # Validate before any credential state changes: a malformed
        # legacy URL must not auto-migrate the password into a host
        # scope Sentinel can never match, nor scrub the plaintext copy.
        if not valid_http_url(url):
            self._connection_info = (
                f"Nextcloud URL {url!r} is not a valid http(s):// URL; "
                "fix config.json and reconnect."
            )
            return False
        username = str(cfg.get("username") or "")
        password = str(cfg.get("password") or "")
        if password and username:
            store_credentials(
                "nextcloud",
                {
                    "kind": "header",
                    "header": "Authorization",
                    "token": _basic_credential(username, password),
                },
                [],
                hosts=origin_hosts(url),
                insecure_hosts=insecure_origin_hosts(url),
            )
        handle = mint_surrogate("nextcloud")
        if handle is None:
            self._connection_info = "No Nextcloud credential in the Muse vault or config."
            return False
        _scrub_config_password()
        self._url = url
        # Keep the username (for is_from_bot); the password never
        # lives in this process once migrated.
        self._auth = (username, "")
        self._surrogate = handle.token
        self._http = MuseBoundarySession("nextcloud")
        self._muse = True
        return True

    def connect(self) -> bool:
        """Authenticate with Nextcloud Talk."""
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        if muse_auth_enabled():
            # Vault-first surrogate wiring; validation below runs the
            # /room read through the daemon boundary (audited).
            if not self._wire_muse():
                return False
        else:
            cfg = _config.load()
            if not cfg:  # pragma: no branch
                self._connection_info = "No Nextcloud config found."
                return False
            self._url = cfg["url"].rstrip("/")
            self._auth = (cfg["username"], cfg["password"])
        try:
            ok, detail = self._validate_credentials()
            if ok:
                self._connection_info = f"Connected to {self._url} as {self._auth[0]}"
                return True
            self._connection_info = f"Nextcloud auth failed: {detail}"
            return False
        except Exception as e:
            self._connection_info = f"Nextcloud connection failed: {e}"
            return False

    def join_channel(self, channel_id: str) -> None:
        """Join a Nextcloud Talk room via the self-join endpoint."""
        try:
            result = self._post(f"/room/{channel_id}/participants/active")
            statuscode = result.get("ocs", {}).get("meta", {}).get("statuscode", 0)
            if statuscode not in (200, 201):
                logger.warning("Nextcloud Talk join failed for %s: %s", channel_id, result)
        except Exception:
            logger.warning("Nextcloud Talk join failed for %s", channel_id, exc_info=True)

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll a Nextcloud Talk room for new messages.

        Fetches the latest messages (``lookIntoFuture=0`` without
        ``lastKnownMessageId``, which would page backwards) and filters
        client-side to message ids greater than the numeric ``oldest``
        cursor (``""``/``"0"`` = no cursor).  ``ts`` carries the Talk
        message id (used as ``replyTo`` by :meth:`send_message`); the unix
        epoch is kept under ``timestamp``.  Returns max id as the cursor.
        """
        if not channel_id:  # pragma: no branch
            return [], oldest
        try:
            cursor = int(oldest) if oldest not in ("", "0") else 0
        except ValueError:
            cursor = 0
        try:
            params: dict[str, Any] = {"lookIntoFuture": 0, "limit": limit}
            result = self._get(f"/chat/{channel_id}", params=params)
            msgs = result.get("ocs", {}).get("data", [])
            messages: list[dict[str, Any]] = []
            max_id = cursor
            for msg in msgs:  # pragma: no branch
                msg_id = int(msg.get("id", 0) or 0)
                if msg_id <= cursor:
                    continue
                max_id = max(max_id, msg_id)
                if msg_id > self._last_message_id:  # pragma: no branch
                    self._last_message_id = msg_id
                messages.append(
                    {
                        "ts": str(msg_id),
                        "user": msg.get("actorId", ""),
                        "text": msg.get("message", ""),
                        "id": str(msg_id),
                        "timestamp": str(msg.get("timestamp", "")),
                    }
                )
            messages.sort(key=lambda m: int(m["ts"]))
            return messages, str(max_id) if max_id else oldest
        except Exception:
            return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a Nextcloud Talk message.

        Raises:
            RuntimeError: If the OCS response reports failure.
        """
        kwargs: dict[str, Any] = {"message": text, "replyTo": 0}
        if thread_ts:  # pragma: no branch
            kwargs["replyTo"] = int(thread_ts)
        result = self._post(f"/chat/{channel_id}", kwargs)
        statuscode = result.get("ocs", {}).get("meta", {}).get("statuscode", 0)
        if statuscode not in (200, 201):
            raise RuntimeError(f"Nextcloud Talk send failed: {result}")

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Check if message is from the bot."""
        return bool(msg.get("user", "") == self._auth[0])

    def list_rooms(self) -> str:
        """List Nextcloud Talk rooms.

        Returns:
            JSON string with room list (token, displayName, type).
        """
        try:
            result = self._get("/room")
            rooms = [
                {
                    "token": r.get("token", ""),
                    "display_name": r.get("displayName", ""),
                    "type": r.get("type", 0),
                    "participants": r.get("participantCount", 0),
                }
                for r in result.get("ocs", {}).get("data", [])
            ]
            return json.dumps({"ok": True, "rooms": rooms}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_room(self, token: str) -> str:
        """Get information about a Nextcloud Talk room.

        Args:
            token: Room token.

        Returns:
            JSON string with room details.
        """
        try:
            result = self._get(f"/room/{token}")
            room_data = result.get("ocs", {}).get("data", {})
            return json.dumps({"ok": True, "room": room_data}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def create_room(self, room_type: int = 3, invite: str = "", room_name: str = "") -> str:
        """Create a Nextcloud Talk room.

        Args:
            room_type: 1=one-to-one, 2=group, 3=public. Default: 3.
            invite: User ID, group ID, or circle ID to invite.
            room_name: Room display name.

        Returns:
            JSON string with room token.
        """
        try:
            data: dict[str, Any] = {"roomType": room_type}
            if invite:  # pragma: no branch
                data["invite"] = invite
            if room_name:  # pragma: no branch
                data["roomName"] = room_name
            result = self._post("/room", data)
            room = result.get("ocs", {}).get("data", {})
            return json.dumps(
                {
                    "ok": True,
                    "token": room.get("token", ""),
                    "name": room.get("displayName", ""),
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_participants(self, token: str) -> str:
        """List participants in a room.

        Args:
            token: Room token.

        Returns:
            JSON string with participant list.
        """
        try:
            result = self._get(f"/room/{token}/participants")
            participants = result.get("ocs", {}).get("data", [])
            return json.dumps({"ok": True, "participants": participants}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_messages(
        self,
        token: str,
        look_into_future: int = 0,
        limit: int = 100,
        last_known_message_id: int = 0,
    ) -> str:
        """List messages in a Nextcloud Talk room.

        Args:
            token: Room token.
            look_into_future: 0 for history, 1 for new messages. Default: 0.
            limit: Maximum messages. Default: 100.
            last_known_message_id: Last message ID seen (for pagination).

        Returns:
            JSON string with message list.
        """
        try:
            params: dict[str, Any] = {
                "lookIntoFuture": look_into_future,
                "limit": limit,
            }
            if last_known_message_id:  # pragma: no branch
                params["lastKnownMessageId"] = last_known_message_id
            result = self._get(f"/chat/{token}", params=params)
            messages = result.get("ocs", {}).get("data", [])
            return json.dumps({"ok": True, "messages": messages}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def post_message(self, token: str, message: str, reply_to: int = 0) -> str:
        """Post a message to a Nextcloud Talk room.

        Args:
            token: Room token.
            message: Message text.
            reply_to: Message ID to reply to. Default: 0 (no reply).

        Returns:
            JSON string with ok status and message id.
        """
        try:
            result = self._post(f"/chat/{token}", {"message": message, "replyTo": reply_to})
            msg_data = result.get("ocs", {}).get("data", {})
            return json.dumps({"ok": True, "id": msg_data.get("id", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def set_room_name(self, token: str, name: str) -> str:
        """Set the name of a Nextcloud Talk room.

        Args:
            token: Room token.
            name: New room name.

        Returns:
            JSON string with ok status.
        """
        try:
            resp = self._http.put(
                f"{self._base()}/room/{token}/name",
                json={"roomName": name},
                timeout=30,
                **self._auth_kwargs(),
            )
            return json.dumps({"ok": resp.status_code == 200})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def delete_message(self, token: str, message_id: int) -> str:
        """Delete a message from a room.

        Args:
            token: Room token.
            message_id: Message ID to delete.

        Returns:
            JSON string with ok status.
        """
        try:
            resp = self._http.delete(
                f"{self._base()}/chat/{token}/{message_id}",
                timeout=30,
                **self._auth_kwargs(),
            )
            return json.dumps({"ok": resp.status_code == 200})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


def _muse_authenticate(
    backend: NextcloudTalkChannelBackend, url: str, username: str, password: str
) -> str:
    """Enroll Nextcloud credentials into the Muse vault and validate them.

    The password goes straight into the vault as a header-kind Basic
    credential bound to the configured server origin, and is never
    written to ``config.json`` (only the non-secret ``url``/``username``
    metadata is — written first, so a failed enrollment leaves no
    password on disk).  Validation runs the ``/room`` read through the
    daemon boundary, so it is audited; invalid credentials leave the
    vault empty.

    Args:
        backend: The agent's Nextcloud backend to (re)wire.
        url: Nextcloud server base URL.
        username: Nextcloud login name.
        password: Nextcloud password or app password.

    Returns:
        JSON string with the validation result.
    """
    import contextlib

    from kiss.agents.third_party_agents.muse_auth._common import (
        insecure_origin_hosts,
        origin_hosts,
        valid_http_url,
    )
    from kiss.agents.third_party_agents.muse_auth.client import (
        clear_credentials,
        store_credentials,
    )

    if not valid_http_url(url):
        return json.dumps(
            {"ok": False, "error": f"{url!r} is not a valid http(s):// server URL."}
        )
    try:
        save_json_config(_config.path, {"url": url, "username": username})
        store_credentials(
            "nextcloud",
            {
                "kind": "header",
                "header": "Authorization",
                "token": _basic_credential(username, password),
            },
            [],
            hosts=origin_hosts(url),
            insecure_hosts=insecure_origin_hosts(url),
        )
        if backend._wire_muse():  # pragma: no branch - credential was just stored
            # The same strict check connect() uses: HTTP status AND OCS
            # meta statuscode (list_rooms() would swallow a 401 into an
            # empty room list, and error envelopes also carry "ocs").
            ok, detail = backend._validate_credentials()
            if ok:
                return json.dumps(
                    {"ok": True, "message": "Nextcloud credentials saved (Muse-auth)."}
                )
            error = json.dumps({"ok": False, "error": f"Authentication failed: {detail}"})
        else:  # pragma: no cover - defense in depth
            error = json.dumps({"ok": False, "error": backend._connection_info})
    except Exception as e:
        error = json.dumps({"ok": False, "error": str(e)})
    # Roll the vault back so bad credentials are not left enrolled.
    with contextlib.suppress(Exception):
        clear_credentials("nextcloud")
    backend._url = ""
    backend._auth = ("", "")
    backend._surrogate = ""
    backend._http = requests
    backend._muse = False
    return error


class NextcloudTalkAgent(BaseChannelAgent):
    """Channel agent with Nextcloud Talk API tools."""

    def __init__(self) -> None:
        super().__init__("Nextcloud Talk Agent")
        self._backend = NextcloudTalkChannelBackend()
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
                self._backend._url = ""
                self._backend._connection_info = f"Muse-auth wiring failed: {e}"
            return
        cfg = _config.load()
        if cfg:  # pragma: no branch
            self._backend._url = cfg["url"].rstrip("/")
            self._backend._auth = (cfg["username"], cfg["password"])

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._url)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_nextcloud_auth() -> str:
            """Check if Nextcloud Talk credentials are configured and valid.

            Returns:
                Authentication status or instructions.
            """
            if not agent._backend._url:  # pragma: no branch
                return (
                    "Not authenticated with Nextcloud Talk. "
                    "Use authenticate_nextcloud(url=..., username=..., password=...) "
                    "to configure.\n"
                    "You need: Nextcloud server URL (e.g. 'https://cloud.example.com'), "
                    "username, and password (or an app password from "
                    "Settings > Security > Devices & sessions)."
                )
            try:
                result = json.loads(agent._backend.list_rooms())
                if result.get("ok"):  # pragma: no branch
                    return json.dumps({"ok": True, "room_count": len(result.get("rooms", []))})
                return json.dumps({"ok": False, "error": "Authentication failed."})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_nextcloud(url: str, username: str, password: str) -> str:
            """Store and validate Nextcloud Talk credentials.

            Args:
                url: Nextcloud server URL (e.g. "https://nextcloud.example.com").
                username: Nextcloud username.
                password: Nextcloud password or app password.

            Returns:
                Validation result or error message.
            """
            for val, name in [(url, "url"), (username, "username"), (password, "password")]:
                if not val.strip():  # pragma: no branch
                    return f"{name} cannot be empty."
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if muse_auth_enabled():
                return _muse_authenticate(
                    agent._backend,
                    url.strip().rstrip("/"),
                    username.strip(),
                    password.strip(),
                )
            agent._backend._url = url.strip().rstrip("/")
            agent._backend._auth = (username.strip(), password.strip())
            try:
                result = agent._backend.list_rooms()
                data = json.loads(result)
                if data.get("ok"):  # pragma: no branch
                    _config.save(
                        {
                            "url": url.strip().rstrip("/"),
                            "username": username.strip(),
                            "password": password.strip(),
                        }
                    )
                    return json.dumps({"ok": True, "message": "Nextcloud credentials saved."})
                return json.dumps({"ok": False, "error": "Authentication failed."})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def clear_nextcloud_auth() -> str:
            """Clear the stored Nextcloud credentials.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._url = ""
            agent._backend._auth = ("", "")
            agent._backend._surrogate = ""
            agent._backend._http = requests
            agent._backend._muse = False
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if muse_auth_enabled():
                from kiss.agents.third_party_agents.muse_auth.client import clear_credentials

                clear_credentials("nextcloud")
            return "Nextcloud authentication cleared."

        return [check_nextcloud_auth, authenticate_nextcloud, clear_nextcloud_auth]


def _make_backend() -> NextcloudTalkChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = NextcloudTalkChannelBackend()
    from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

    if muse_auth_enabled():
        if backend._wire_muse():
            return backend
        print("Not authenticated. Run: kiss-nextcloud -t 'authenticate'")
        sys.exit(1)
    cfg = _config.load()
    if not cfg:  # pragma: no branch
        print("Not authenticated. Run: kiss-nextcloud -t 'authenticate'")
        sys.exit(1)
    backend._url = cfg["url"].rstrip("/")
    backend._auth = (cfg["username"], cfg["password"])
    return backend


def main() -> None:
    """Run the NextcloudTalkAgent from the command line with chat persistence."""
    channel_main(
        NextcloudTalkAgent,
        "kiss-nextcloud",
        channel_name="Nextcloud Talk",
        make_backend=_make_backend,
    )


def tools() -> list:
    """Return the Nextcloud Talk channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return NextcloudTalkAgent()._get_tools()


if __name__ == "__main__":
    main()
