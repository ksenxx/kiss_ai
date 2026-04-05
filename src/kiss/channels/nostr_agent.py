"""Nostr Agent — StatefulSorcarAgent extension with Nostr protocol tools.

Provides access to the Nostr decentralized protocol via pynostr.
Stores config in ``~/.kiss/channels/nostr/config.json``.

Usage::

    agent = NostrAgent()
    agent.run(prompt_template="Post a note saying hello")
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.channels._channel_agent_utils import (
    BaseChannelAgent,
    ToolMethodBackend,
    channel_main,
    clear_json_config,
    load_json_config,
    save_json_config,
)

_NOSTR_DIR = Path.home() / ".kiss" / "channels" / "nostr"


def _config_path() -> Path:
    """Return the path to the stored Nostr config file."""
    return _NOSTR_DIR / "config.json"


def _load_config() -> dict[str, str] | None:
    """Load stored Nostr config from disk."""
    return load_json_config(_config_path(), ("private_key",))


def _save_config(private_key: str, relays: str) -> None:
    """Save Nostr config to disk with restricted permissions."""
    save_json_config(_config_path(), {"private_key": private_key.strip(), "relays": relays.strip()})


def _clear_config() -> None:
    """Delete the stored Nostr config."""
    clear_json_config(_config_path())


class NostrChannelBackend(ToolMethodBackend):
    """ChannelBackend implementation for Nostr protocol via pynostr."""

    def __init__(self) -> None:
        self._private_key: Any = None
        self._public_key: str = ""
        self._relays: list[str] = []
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Load Nostr keys from stored config."""
        cfg = _load_config()
        if not cfg:  # pragma: no branch
            self._connection_info = "No Nostr config found."
            return False
        try:
            from pynostr.key import PrivateKey

            pk_str = cfg["private_key"]
            if pk_str.startswith("nsec"):  # pragma: no branch
                self._private_key = PrivateKey.from_nsec(pk_str)
            else:
                self._private_key = PrivateKey.from_hex(pk_str)
            self._public_key = self._private_key.public_key.hex()
            relays_str = cfg.get("relays", "wss://relay.damus.io")
            self._relays = [r.strip() for r in relays_str.split(",") if r.strip()]
            self._connection_info = f"Nostr key loaded, pubkey: {self._public_key[:16]}..."
            return True
        except Exception as e:
            self._connection_info = f"Nostr key load failed: {e}"
            return False

    @property
    def connection_info(self) -> str:
        """Human-readable connection status string."""
        return self._connection_info

    def find_channel(self, name: str) -> str | None:
        """Return channel/relay name."""
        return name if name else None

    def find_user(self, username: str) -> str | None:
        """Return username as public key."""
        return username if username else None

    def join_channel(self, channel_id: str) -> None:
        """No-op for Nostr."""

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll Nostr relays for new events (basic implementation)."""
        return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Publish a Nostr note."""
        self.publish_note(text)

    def wait_for_reply(
        self,
        channel_id: str,
        thread_ts: str,
        user_id: str,
        timeout_seconds: float = 300.0,
        stop_event: threading.Event | None = None,
    ) -> str | None:
        """Reply waiting is not currently supported for Nostr."""
        return None

    def disconnect(self) -> None:
        """Release backend resources before stop or reconnect."""

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Check if event is from this key."""
        return bool(msg.get("pubkey", "") == self._public_key)

    def strip_bot_mention(self, text: str) -> str:
        """Remove bot mentions from text."""
        return text

    def _publish_event(self, kind: int, content: str, tags: list | None = None) -> dict[str, Any]:
        """Publish a Nostr event to all configured relays."""
        from pynostr.event import Event
        from pynostr.relay_manager import RelayManager

        event = Event(kind=kind, content=content, tags=tags or [])
        event.sign(self._private_key.hex())

        manager = RelayManager()
        for relay_url in self._relays:  # pragma: no branch
            manager.add_relay(relay_url)

        manager.open_connections({"cert_reqs": False})
        time.sleep(1.0)
        manager.publish_event(event)
        time.sleep(1.0)
        manager.close_connections()
        return {"event_id": event.id}

    def publish_note(self, content: str) -> str:
        """Publish a text note (kind 1) to Nostr.

        Args:
            content: Note content text.

        Returns:
            JSON string with ok status and event id.
        """
        if not self._private_key:  # pragma: no branch
            return json.dumps({"ok": False, "error": "Not authenticated"})
        try:
            result = self._publish_event(1, content)
            return json.dumps({"ok": True, "event_id": result.get("event_id", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def publish_reply(self, content: str, reply_to_event_id: str) -> str:
        """Publish a reply to an existing Nostr event.

        Args:
            content: Reply content.
            reply_to_event_id: Event ID to reply to.

        Returns:
            JSON string with ok status and event id.
        """
        if not self._private_key:  # pragma: no branch
            return json.dumps({"ok": False, "error": "Not authenticated"})
        try:
            tags = [["e", reply_to_event_id, "", "reply"]]
            result = self._publish_event(1, content, tags)
            return json.dumps({"ok": True, "event_id": result.get("event_id", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_dm(self, recipient_pubkey: str, content: str) -> str:
        """Send an encrypted direct message (NIP-04).

        Args:
            recipient_pubkey: Recipient's public key (hex).
            content: Message content (will be encrypted).

        Returns:
            JSON string with ok status and event id.
        """
        if not self._private_key:  # pragma: no branch
            return json.dumps({"ok": False, "error": "Not authenticated"})
        try:
            from pynostr.encrypted_dm import EncryptedDirectMessage

            dm = EncryptedDirectMessage(
                recipient_pubkey=recipient_pubkey,
            )
            dm.encrypt(self._private_key.hex(), cleartext_content=content)
            event = dm.to_event()
            event.sign(self._private_key.hex())

            from pynostr.relay_manager import RelayManager
            manager = RelayManager()
            for relay_url in self._relays:  # pragma: no branch
                manager.add_relay(relay_url)
            manager.open_connections({"cert_reqs": False})
            time.sleep(1.0)
            manager.publish_event(event)
            time.sleep(1.0)
            manager.close_connections()
            return json.dumps({"ok": True, "event_id": event.id})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_profile(self) -> str:
        """Get the current user's Nostr profile.

        Returns:
            JSON string with public key info.
        """
        if not self._private_key:  # pragma: no branch
            return json.dumps({"ok": False, "error": "Not authenticated"})
        try:

            pub = self._private_key.public_key
            return json.dumps({
                "ok": True,
                "pubkey_hex": pub.hex(),
                "pubkey_npub": pub.bech32(),
            })
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def set_profile(
        self,
        name: str = "",
        about: str = "",
        picture: str = "",
        nip05: str = "",
    ) -> str:
        """Set the Nostr user profile (kind 0).

        Args:
            name: Display name.
            about: Bio/about text.
            picture: Profile picture URL.
            nip05: NIP-05 identifier (user@domain.com).

        Returns:
            JSON string with ok status and event id.
        """
        if not self._private_key:  # pragma: no branch
            return json.dumps({"ok": False, "error": "Not authenticated"})
        try:
            profile: dict[str, str] = {}
            if name:  # pragma: no branch
                profile["name"] = name
            if about:  # pragma: no branch
                profile["about"] = about
            if picture:  # pragma: no branch
                profile["picture"] = picture
            if nip05:  # pragma: no branch
                profile["nip05"] = nip05
            result = self._publish_event(0, json.dumps(profile))
            return json.dumps({"ok": True, "event_id": result.get("event_id", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_relays(self) -> str:
        """List configured Nostr relays.

        Returns:
            JSON string with relay list.
        """
        return json.dumps({"ok": True, "relays": self._relays})

    def add_relay(self, relay_url: str) -> str:
        """Add a Nostr relay to the configuration.

        Args:
            relay_url: WebSocket URL of the relay (wss://...).

        Returns:
            JSON string with ok status.
        """
        if relay_url not in self._relays:  # pragma: no branch
            self._relays.append(relay_url)
        cfg = _load_config()
        if cfg:  # pragma: no branch
            cfg["relays"] = ",".join(self._relays)
            _save_config(cfg["private_key"], cfg["relays"])
        return json.dumps({"ok": True, "relays": self._relays})

    def remove_relay(self, relay_url: str) -> str:
        """Remove a Nostr relay from the configuration.

        Args:
            relay_url: Relay URL to remove.

        Returns:
            JSON string with ok status.
        """
        self._relays = [r for r in self._relays if r != relay_url]
        cfg = _load_config()
        if cfg:  # pragma: no branch
            cfg["relays"] = ",".join(self._relays)
            _save_config(cfg["private_key"], cfg["relays"])
        return json.dumps({"ok": True, "relays": self._relays})



class NostrAgent(BaseChannelAgent, StatefulSorcarAgent):
    """StatefulSorcarAgent extended with Nostr protocol tools."""

    def __init__(self) -> None:
        super().__init__("Nostr Agent")
        self._backend = NostrChannelBackend()
        cfg = _load_config()
        if cfg:  # pragma: no branch
            try:
                from pynostr.key import PrivateKey

                pk_str = cfg["private_key"]
                if pk_str.startswith("nsec"):  # pragma: no branch
                    self._backend._private_key = PrivateKey.from_nsec(pk_str)
                else:
                    self._backend._private_key = PrivateKey.from_hex(pk_str)
                self._backend._public_key = self._backend._private_key.public_key.hex()
                relays_str = cfg.get("relays", "wss://relay.damus.io")
                self._backend._relays = [r.strip() for r in relays_str.split(",") if r.strip()]
            except Exception:
                pass

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return self._backend._private_key is not None

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self


        def check_nostr_auth() -> str:
            """Check if Nostr key is configured.

            Returns:
                Key status or instructions.
            """
            if agent._backend._private_key is None:  # pragma: no branch
                return (
                    "Not configured for Nostr. Use authenticate_nostr(private_key=...) "
                    "to configure. Provide an nsec... key or hex private key."
                )
            return json.loads(agent._backend.get_profile()).get("ok") and json.dumps({
                "ok": True,
                "pubkey": agent._backend._public_key[:16] + "...",
                "relays": agent._backend._relays,
            }) or json.dumps({"ok": False, "error": "Key error"})

        def authenticate_nostr(
            private_key: str, relays: str = "wss://relay.damus.io"
        ) -> str:
            """Configure Nostr with a private key.

            Args:
                private_key: nsec bech32 or hex private key.
                relays: Comma-separated relay WebSocket URLs.

            Returns:
                Configuration result or error message.
            """
            if not private_key.strip():  # pragma: no branch
                return "private_key cannot be empty."
            try:
                from pynostr.key import PrivateKey

                pk_str = private_key.strip()
                if pk_str.startswith("nsec"):  # pragma: no branch
                    pk = PrivateKey.from_nsec(pk_str)
                else:
                    pk = PrivateKey.from_hex(pk_str)
                agent._backend._private_key = pk
                agent._backend._public_key = pk.public_key.hex()
                agent._backend._relays = [r.strip() for r in relays.split(",") if r.strip()]
                _save_config(pk_str, relays)
                return json.dumps({
                    "ok": True,
                    "message": "Nostr key configured.",
                    "pubkey": pk.public_key.hex()[:16] + "...",
                })
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def clear_nostr_auth() -> str:
            """Clear the stored Nostr configuration.

            Returns:
                Status message.
            """
            _clear_config()
            agent._backend._private_key = None
            agent._backend._public_key = ""
            return "Nostr configuration cleared."

        return [check_nostr_auth, authenticate_nostr, clear_nostr_auth]


def main() -> None:
    """Run the NostrAgent from the command line with chat persistence."""
    channel_main(NostrAgent, "kiss-nostr")

if __name__ == "__main__":
    main()
