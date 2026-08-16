# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""SimpleX Chat Agent — channel agent for the simplex-chat CLI WebSocket API.

Talks to a locally running ``simplex-chat`` CLI started with a WebSocket
port (e.g. ``simplex-chat -p 5225``, giving ``ws://127.0.0.1:5225``).
The client sends JSON frames ``{"corrId": "<n>", "cmd": "<command>"}``;
the CLI replies with frames carrying the matching ``corrId`` and pushes
unsolicited events (``newChatItems`` for inbound messages).  Stores config
in ``~/.kiss/third_party_agents/simplex/config.json``.

Usage::

    agent = SimpleXAgent()
    agent.run(prompt_template="Send 'Hello!' to contact alice")
"""

from __future__ import annotations

import json
import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as ws_connect

from kiss.agents.third_party_agents._backend_utils import drain_queue_messages
from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

logger = logging.getLogger(__name__)

_DEFAULT_WS_URL = "ws://127.0.0.1:5225"
_COMMAND_TIMEOUT = 15.0
_PUMP_TIMEOUT = 0.2

_SIMPLEX_DIR = Path.home() / ".kiss" / "third_party_agents" / "simplex"
_config = ChannelConfig(_SIMPLEX_DIR, ("ws_url",))


def _resp_of(frame: dict[str, Any]) -> dict[str, Any]:
    """Return the response object of a frame, unwrapping Right/Left envelopes.

    Newer simplex-chat versions wrap the response as ``{"Right": {...}}``
    (or ``{"Left": {...}}`` for errors); older versions send it directly.

    Args:
        frame: Decoded JSON frame from the CLI.

    Returns:
        The inner response dict (possibly empty).
    """
    resp = frame.get("resp")
    if not isinstance(resp, dict):
        return {}
    for envelope in ("Right", "Left"):
        inner = resp.get(envelope)
        if isinstance(inner, dict):
            return inner
    return resp


def _find_address(obj: Any) -> str:
    """Recursively find a SimpleX contact address string in a response.

    Looks for the ``connReqContact`` / ``connLinkContact`` keys that
    address-related responses carry at varying nesting depths.

    Args:
        obj: Response fragment (dict, list, or scalar).

    Returns:
        The first address string found, or ``""``.
    """
    if isinstance(obj, str):
        return ""
    if isinstance(obj, dict):
        for key in ("connReqContact", "connLinkContact"):
            value = obj.get(key)
            if isinstance(value, str) and value:
                return value
        for value in obj.values():
            found = _find_address(value)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_address(value)
            if found:
                return found
    return ""


class SimpleXChannelBackend(ToolMethodBackend):
    """Channel backend for the simplex-chat CLI WebSocket API.

    Maintains a single synchronous WebSocket connection, a monotonically
    increasing ``corrId`` counter for command/response matching, and a
    queue of normalized inbound messages built from unsolicited
    ``newChatItems`` events.
    """

    def __init__(self) -> None:
        self._ws_url: str = ""
        self._ws: Any = None
        self._corr_id: int = 0
        self._message_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._io_lock = threading.Lock()
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Load SimpleX config and open the WebSocket connection."""
        cfg = _config.load()
        if not cfg:  # pragma: no branch
            self._connection_info = "No SimpleX Chat config found."
            return False
        self._ws_url = cfg["ws_url"]
        try:
            self._ws = ws_connect(self._ws_url, open_timeout=10)
            self._connection_info = f"Connected to simplex-chat at {self._ws_url}"
            return True
        except Exception as e:
            self._connection_info = f"SimpleX connect failed: {e}"
            logger.warning("Could not connect to simplex-chat: %s", e)
            self._ws = None
            return False

    def _ensure_connected(self) -> None:
        """Open the WebSocket if it is not already open.

        Raises:
            RuntimeError: If no ws_url is configured or the connection fails.
        """
        if self._ws is not None:
            return
        if not self._ws_url:
            raise RuntimeError("SimpleX Chat not configured (no ws_url).")
        self._ws = ws_connect(self._ws_url, open_timeout=10)
        self._connection_info = f"Connected to simplex-chat at {self._ws_url}"

    def _handle_event(self, frame: dict[str, Any]) -> None:
        """Queue normalized inbound messages from an unsolicited event frame.

        Only ``newChatItems`` events with received-direction chat items
        (``directRcv`` / ``groupRcv``) are queued.

        Args:
            frame: Decoded JSON frame without a matching ``corrId``.
        """
        resp = _resp_of(frame)
        if resp.get("type") != "newChatItems":
            return
        for entry in resp.get("chatItems", []):
            if not isinstance(entry, dict):
                continue
            msg = self._normalize_chat_item(entry)
            if msg is not None:
                self._message_queue.put(msg)

    def _normalize_chat_item(self, entry: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize one ``newChatItems`` entry into a message dict.

        Args:
            entry: One element of the event's ``chatItems`` list, with
                ``chatInfo`` and ``chatItem`` keys.

        Returns:
            Dict with ``ts``, ``user``, ``text``, ``channel_id``,
            ``thread_ts`` and ``direction`` keys, or ``None`` for
            non-received directions or malformed entries.
        """
        chat_item = entry.get("chatItem") or {}
        chat_dir = chat_item.get("chatDir") or {}
        direction = str(chat_dir.get("type", ""))
        if direction not in ("directRcv", "groupRcv"):
            return None
        chat_info = entry.get("chatInfo") or {}
        if direction == "groupRcv":
            channel = str((chat_info.get("groupInfo") or {}).get("localDisplayName", ""))
            user = str((chat_dir.get("groupMember") or {}).get("localDisplayName", ""))
        else:
            channel = str((chat_info.get("contact") or {}).get("localDisplayName", ""))
            user = channel
        meta = chat_item.get("meta") or {}
        content = chat_item.get("content") or {}
        text = str((content.get("msgContent") or {}).get("text", ""))
        return {
            "ts": str(meta.get("itemTs") or time.time()),
            "user": user,
            "username": user,
            "text": text,
            "channel_id": channel,
            "thread_ts": str(meta.get("itemId", "")),
            "direction": direction,
        }

    def _send_cmd(self, cmd: str, timeout: float = _COMMAND_TIMEOUT) -> dict[str, Any]:
        """Send a command frame and wait for its correlated response.

        Unsolicited event frames received while waiting are queued.

        Args:
            cmd: simplex-chat command string (e.g. ``"/contacts"``).
            timeout: Seconds to wait for the matching response.

        Returns:
            The full response frame.

        Raises:
            RuntimeError: If not connected, the connection drops, or the
                response does not arrive within *timeout*.
        """
        with self._io_lock:
            if self._ws is None:
                raise RuntimeError("Not connected to simplex-chat.")
            self._corr_id += 1
            corr_id = str(self._corr_id)
            self._ws.send(json.dumps({"corrId": corr_id, "cmd": cmd}))
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"SimpleX response timeout for command: {cmd!r}")
                try:
                    raw = self._ws.recv(timeout=remaining)
                except TimeoutError:
                    raise RuntimeError(f"SimpleX response timeout for command: {cmd!r}") from None
                except ConnectionClosed as e:
                    self._ws = None
                    raise RuntimeError(f"SimpleX connection closed: {e}") from e
                frame = self._decode_frame(raw)
                if frame is None:
                    continue
                if str(frame.get("corrId", "")) == corr_id:
                    return frame
                self._handle_event(frame)

    def _pump_events(self, timeout: float = _PUMP_TIMEOUT) -> None:
        """Read pending frames non-blockingly and queue unsolicited events.

        Args:
            timeout: Per-read timeout; the pump stops on the first idle read.
        """
        with self._io_lock:
            if self._ws is None:
                return
            while True:
                try:
                    raw = self._ws.recv(timeout=timeout)
                except TimeoutError:
                    return
                except ConnectionClosed:
                    self._ws = None
                    return
                frame = self._decode_frame(raw)
                if frame is not None:
                    self._handle_event(frame)

    @staticmethod
    def _decode_frame(raw: str | bytes) -> dict[str, Any] | None:
        """Decode one WebSocket frame into a dict, or ``None`` if malformed.

        Args:
            raw: Raw frame payload.

        Returns:
            Decoded dict, or ``None`` for non-JSON / non-dict payloads.
        """
        try:
            frame = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        return frame if isinstance(frame, dict) else None

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Pump pending frames, then drain queued inbound messages.

        Drained messages not matching ``channel_id`` are discarded.

        Args:
            channel_id: Contact or group display name filter ("" for all).
            oldest: Opaque cursor, returned unchanged.
            limit: Maximum number of messages to return.

        Returns:
            Tuple of (normalized message dicts, unchanged cursor).
        """
        self._pump_events()
        messages = drain_queue_messages(
            self._message_queue,
            limit=limit,
            keep=lambda msg: not channel_id or msg.get("channel_id") == channel_id,
        )
        return messages, oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a text message to a contact or group.

        SimpleX chats have no threading, so ``thread_ts`` is ignored.

        Args:
            channel_id: Contact or group display name.
            text: Message text.
            thread_ts: Ignored.

        Raises:
            RuntimeError: If the CLI reports a command error.
        """
        self._ensure_connected()
        frame = self._send_cmd(f"@'{channel_id}' {text}")
        rtype = str(_resp_of(frame).get("type", ""))
        if "chatCmdError" in rtype:
            raise RuntimeError(f"SimpleX send failed: {json.dumps(_resp_of(frame))[:500]}")

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Return True for sent-direction chat items (the bot's own messages).

        Args:
            msg: Message dict from :meth:`poll_messages`.

        Returns:
            Whether the message direction is a sent direction
            (``directSnd`` / ``groupSnd``).
        """
        return str(msg.get("direction", "")).endswith("Snd")

    def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                logger.debug("SimpleX close failed", exc_info=True)
            self._ws = None

    def send_simplex_message(self, contact: str, text: str) -> str:
        """Send a text message to a SimpleX contact or group.

        Args:
            contact: Contact or group display name (e.g. "alice").
            text: Message text.

        Returns:
            JSON string with ok status.
        """
        try:
            self.send_message(contact, text)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_simplex_contacts(self) -> str:
        """List the SimpleX contacts known to the connected CLI profile.

        Returns:
            JSON string with ok status and a list of contact display names.
        """
        try:
            self._ensure_connected()
            resp = _resp_of(self._send_cmd("/contacts"))
            contacts = [
                str(c.get("localDisplayName", ""))
                for c in resp.get("contacts", [])
                if isinstance(c, dict)
            ]
            return json.dumps({"ok": True, "contacts": contacts})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_simplex_address(self) -> str:
        """Create (or fetch the existing) SimpleX contact address.

        Sends ``/address``; when creation fails because an address
        already exists, falls back to ``/show_address``.

        Returns:
            JSON string with ok status and the contact address.
        """
        try:
            self._ensure_connected()
            resp = _resp_of(self._send_cmd("/address"))
            if "chatCmdError" in str(resp.get("type", "")):
                resp = _resp_of(self._send_cmd("/show_address"))
                if "chatCmdError" in str(resp.get("type", "")):
                    return json.dumps({"ok": False, "error": json.dumps(resp)[:500]})
            address = _find_address(resp)
            if not address:
                return json.dumps({"ok": False, "error": "No address in response."})
            return json.dumps({"ok": True, "address": address})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class SimpleXAgent(BaseChannelAgent):
    """Channel agent with simplex-chat CLI WebSocket tools."""

    channel_system_prompt = (
        "You are chatting via SimpleX Chat through a local simplex-chat CLI. "
        "Contacts and groups are addressed by display name. Messages are "
        "plain text; there is no threading, no reactions, and no file upload "
        "through these tools."
    )

    def __init__(self) -> None:
        super().__init__("SimpleX Agent")
        self._backend = SimpleXChannelBackend()
        cfg = _config.load()
        if cfg:  # pragma: no branch
            self._backend._ws_url = cfg["ws_url"]

    def _is_authenticated(self) -> bool:
        """Return True if the backend is configured."""
        return bool(self._backend._ws_url)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_simplex_auth() -> str:
            """Check if SimpleX Chat is configured.

            Returns:
                Configuration status or instructions.
            """
            if not agent._backend._ws_url:  # pragma: no branch
                return (
                    "Not configured for SimpleX Chat. Use authenticate_simplex() to "
                    "configure. Start the simplex-chat CLI with a WebSocket port "
                    "(e.g. `simplex-chat -p 5225`) and pass its URL "
                    "(default ws://127.0.0.1:5225)."
                )
            return json.dumps({"ok": True, "ws_url": agent._backend._ws_url})

        def authenticate_simplex(ws_url: str = _DEFAULT_WS_URL) -> str:
            """Configure the simplex-chat CLI WebSocket URL.

            Args:
                ws_url: WebSocket URL of the running simplex-chat CLI
                    (default "ws://127.0.0.1:5225", from `simplex-chat -p 5225`).

            Returns:
                Configuration result or error message.
            """
            if not ws_url.strip():  # pragma: no branch
                return "ws_url cannot be empty."
            agent._backend._ws_url = ws_url.strip()
            _config.save({"ws_url": ws_url.strip()})
            return json.dumps({"ok": True, "message": "SimpleX Chat configured."})

        def clear_simplex_auth() -> str:
            """Clear the stored SimpleX Chat configuration.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend.disconnect()
            agent._backend._ws_url = ""
            return "SimpleX Chat configuration cleared."

        return [check_simplex_auth, authenticate_simplex, clear_simplex_auth]


def _make_backend() -> SimpleXChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = SimpleXChannelBackend()
    cfg = _config.load()
    if not cfg:  # pragma: no branch
        print("Not configured. Run: kiss-simplex -t 'authenticate'")
        sys.exit(1)
    backend._ws_url = cfg["ws_url"]
    return backend


def main() -> None:
    """Run the SimpleXAgent from the command line with chat persistence."""
    channel_main(
        SimpleXAgent,
        "kiss-simplex",
        channel_name="SimpleX Chat",
        make_backend=_make_backend,
    )


def get_tools() -> list:
    """Return the SimpleX Chat channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return SimpleXAgent()._get_tools()


if __name__ == "__main__":
    main()
