# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""ntfy Agent — channel agent for the ntfy.sh pub-sub notification service.

Provides publish/subscribe access to an ntfy topic over plain HTTP:
messages are published by POSTing to ``{server}/{topic}`` and read back
by polling ``{server}/{topic}/json?poll=1``.  Loop prevention uses an
echo tag: every message the agent publishes is tagged (default
``kiss-sorcar``) and tagged messages are treated as bot messages.

Stores config in ``~/.kiss/third_party_agents/ntfy/config.json`` with a
required ``topic`` and optional ``server`` (default ``https://ntfy.sh``),
``token`` (sent as ``Authorization: Bearer``) and ``echo_tag``.

In Muse-auth mode (``KISS_MUSE_AUTH=1``) a configured token lives in
the Muse vault, enrolled together with a self-hosted server's host
(flagged as a consent-scoped insecure host when the server URL is
plain ``http://``), and every API call runs at the daemon boundary;
the plaintext token is scrubbed from ``config.json`` after enrollment.
A tokenless configuration (the public ntfy.sh works without one) has
no credential to protect and stays on the legacy direct path.

Usage::

    agent = NtfyAgent()
    agent.run(prompt_template="Notify me that the build finished")
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
    save_json_config,
)

logger = logging.getLogger(__name__)

_DEFAULT_SERVER = "https://ntfy.sh"
_DEFAULT_ECHO_TAG = "kiss-sorcar"

_NTFY_DIR = Path.home() / ".kiss" / "third_party_agents" / "ntfy"
_config = ChannelConfig(_NTFY_DIR, ("topic",))


def _scrub_config_token() -> None:
    """Remove a vault-migrated ``token`` from config.json.

    Finishes the Muse migration automatically: the non-secret ``topic``,
    ``server`` and ``echo_tag`` keys are kept so the config file stays
    loadable while never containing the secret again.
    """
    try:
        cfg = json.loads(_config.path.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(cfg, dict) or not cfg.get("token"):
        return
    kept = {k: str(v) for k, v in cfg.items() if k != "token" and v}
    save_json_config(_config.path, kept)


def _extra_hosts(server: str) -> tuple[str, ...]:
    """Return the Muse enrollment origins for an ntfy server URL.

    An ntfy access token belongs to exactly one server, so the
    credential is origin-bound: only the configured server's origin —
    host AND port, because the same hostname on another port is a
    different server — is enrolled (there is no built-in allowlist),
    and a self-hosted instance's token can never be spent against
    public ``ntfy.sh``.

    Args:
        server: Configured ntfy server base URL.

    Returns:
        ``("host:port",)``, or ``()`` when the URL has no host.
    """
    from kiss.agents.third_party_agents.muse_auth._common import url_origin_entry

    entry = url_origin_entry(server)
    return (entry,) if entry else ()


def _insecure_extra_hosts(server: str) -> tuple[str, ...]:
    """Return the origins to enroll as consent-scoped plain-HTTP origins.

    Args:
        server: Configured ntfy server base URL.

    Returns:
        ``("host:port",)`` for a plain-HTTP non-loopback server, else ``()``.
    """
    from kiss.agents.third_party_agents.muse_auth._common import (
        canonical_host,
        is_loopback_host,
        url_origin_entry,
    )

    parsed = urlparse(server)
    host = canonical_host(parsed.hostname or "")
    if parsed.scheme == "http" and host and not is_loopback_host(host):
        return (url_origin_entry(server),)
    return ()


class NtfyChannelBackend(ToolMethodBackend):
    """Channel backend for the ntfy HTTP pub-sub API.

    Publishes by POSTing plain text to ``{server}/{topic}`` and polls
    with ``GET {server}/{topic}/json?poll=1&since=...``.  Messages the
    backend publishes carry the echo tag so :meth:`is_from_bot` can
    filter them out of subsequent polls.
    """

    def __init__(self) -> None:
        self._server: str = _DEFAULT_SERVER
        self._topic: str = ""
        self._token: str = ""
        self._echo_tag: str = _DEFAULT_ECHO_TAG
        self._http: Any = requests
        self._muse: bool = False
        self._muse_error: str = ""
        self._send_lock = threading.Lock()
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Load the ntfy config and mark the backend as connected."""
        cfg = _config.load()
        if not cfg:  # pragma: no branch
            self._connection_info = "No ntfy config found."
            return False
        self._apply_config(cfg)
        if self._muse_error:
            # The configured token could not be wired through the Muse
            # boundary: the backend is deliberately tokenless, so do not
            # claim a working connection.
            self._connection_info = self._muse_error
            return False
        suffix = " (Muse-auth)" if self._muse else ""
        self._connection_info = f"ntfy configured: {self._server}/{self._topic}{suffix}"
        return True

    def _apply_config(self, cfg: dict[str, str]) -> None:
        """Copy persisted config values onto the backend, applying defaults.

        In Muse-auth mode a configured (or previously vault-enrolled)
        token is swapped for a surrogate and requests are rewired
        through the daemon boundary; a tokenless configuration stays on
        the legacy direct path because there is no credential to
        protect.
        """
        self._topic = cfg["topic"]
        self._server = (cfg.get("server") or _DEFAULT_SERVER).rstrip("/")
        self._token = cfg.get("token", "")
        self._echo_tag = cfg.get("echo_tag") or _DEFAULT_ECHO_TAG
        self._http = requests
        self._muse = False
        self._muse_error = ""
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        if muse_auth_enabled():
            from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError

            # An explicit ``token: ""`` in config.json (a tokenless
            # rotation done while Muse was off) is authoritative user
            # intent: it overrides any stale vault entry.  A scrubbed
            # Muse config has NO token key, so this only fires for a
            # deliberately-removed token.
            explicit_tokenless = "token" in cfg and not cfg["token"]
            try:
                self._wire_muse(explicit_tokenless=explicit_tokenless)
            except MuseAuthError as e:
                # Fail closed: in Muse mode the real token must never
                # be used on the direct path, so a failed wiring leaves
                # the backend tokenless rather than falling back to it.
                # The recorded error keeps connect() and the check tool
                # honest about the broken credential.
                self._token = ""
                self._muse_error = f"ntfy Muse-auth wiring failed: {e}"
                self._connection_info = self._muse_error

    def _wire_muse(self, explicit_tokenless: bool = False) -> None:
        """Enroll/mint the ntfy token surrogate and wire the boundary session.

        A plaintext token in the just-read config is the newest user
        intent (initial migration, or a rotation done while Muse was
        off): it replaces any vault enrollment, and is scrubbed from
        ``config.json`` only after the vault holds it.  Without a token
        anywhere the backend stays legacy (nothing to protect).

        Args:
            explicit_tokenless: True when config.json carries an
                explicit empty ``token`` (the user deliberately removed
                it while Muse was off); any stale vault entry is dropped
                so it cannot be silently revived.

        Raises:
            MuseAuthError: When the configured server URL is malformed,
                the daemon rejects the enrollment, or the daemon is
                unreachable (the caller resets the backend tokenless).
        """
        from kiss.agents.third_party_agents.muse_auth._common import valid_http_url
        from kiss.agents.third_party_agents.muse_auth.client import (
            MuseAuthError,
            MuseBoundarySession,
            clear_credentials,
            mint_surrogate,
            store_credentials,
        )

        # Validate before any credential state changes: a malformed
        # legacy server URL must not auto-migrate the token into a host
        # scope Sentinel can never match, nor scrub the plaintext copy.
        if not valid_http_url(self._server):
            raise MuseAuthError(
                f"configured ntfy server {self._server!r} is not a valid http(s):// URL; "
                "fix config.json and reconnect"
            )
        if explicit_tokenless:
            # Honor the deliberate removal: drop any stale vault entry
            # and stay on the legacy tokenless direct path.
            clear_credentials("ntfy")
            return
        if self._token:
            store_credentials(
                "ntfy",
                {"kind": "bearer", "token": self._token},
                [],
                hosts=_extra_hosts(self._server),
                insecure_hosts=_insecure_extra_hosts(self._server),
            )
        handle = mint_surrogate("ntfy")
        if handle is None:
            return
        _scrub_config_token()
        self._token = handle.token
        self._http = MuseBoundarySession("ntfy")
        self._muse = True

    def _auth_headers(self) -> dict[str, str]:
        """Return HTTP headers with Bearer authorization when a token is set."""
        if self._token:
            return {"Authorization": f"Bearer {self._token}"}
        return {}

    def _echo_tags(self) -> list[str]:
        """Return the echo tag as a list of stripped comma-separated sub-tags."""
        return [t.strip() for t in self._echo_tag.split(",") if t.strip()]

    def _fetch_messages(
        self, topic: str, oldest: str, limit: int
    ) -> tuple[list[dict[str, Any]], str]:
        """Fetch and normalize messages from an ntfy topic, raising on failure.

        Fetches ``{server}/{topic}/json?poll=1&since={oldest or 'all'}``,
        parses the newline-delimited JSON stream and keeps only
        ``message`` events.  When a message carries a title, the title is
        prepended to the normalized ``text`` so downstream consumers that
        only read ``text`` still see it.

        Args:
            topic: Topic to poll.
            oldest: ``since`` cursor (unix time as string, or empty for all).
            limit: Maximum number of messages to return.

        Returns:
            Tuple of (normalized message dicts, newest seen time as string).

        Raises:
            RuntimeError: If the poll request returns a non-200 status.
            requests.RequestException: If the HTTP request itself fails.
        """
        resp = self._http.get(
            f"{self._server}/{topic}/json",
            params={"poll": "1", "since": oldest or "all"},
            headers=self._auth_headers(),
            timeout=30,
        )
        if resp.status_code != 200:
            # Include the body: a Muse-auth denial carries the grant
            # instructions the user needs to approve the action.
            raise RuntimeError(
                f"ntfy poll failed: HTTP {resp.status_code}: {resp.text[:300]}"
            )
        messages: list[dict[str, Any]] = []
        newest = oldest
        for line in resp.text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("event") != "message":
                continue
            event_time = int(event.get("time", 0) or 0)
            event_topic = str(event.get("topic", "") or topic)
            title = str(event.get("title", "") or "")
            text = str(event.get("message", "") or "")
            if title:
                text = f"{title}\n\n{text}"
            messages.append(
                {
                    "ts": str(event_time),
                    "user": event_topic,
                    "text": text,
                    "title": title,
                    "channel_id": event_topic,
                    "tags": list(event.get("tags") or []),
                }
            )
            if not newest or event_time > int(newest):
                newest = str(event_time)
            if len(messages) >= limit:
                break
        return messages, newest

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll an ntfy topic for new messages, swallowing failures.

        Args:
            channel_id: Topic to poll when no topic is configured.
            oldest: ``since`` cursor (unix time as string, or empty for all).
            limit: Maximum number of messages to return.

        Returns:
            Tuple of (normalized message dicts, newest seen time as string);
            ``([], oldest)`` on any failure.
        """
        topic = self._topic or channel_id
        try:
            return self._fetch_messages(topic, oldest, limit)
        except Exception:
            logger.debug("ntfy poll failed for topic %s", topic, exc_info=True)
            return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Publish a plain-text message to the ntfy topic.

        ntfy has no threads, so ``thread_ts`` is ignored.  The message
        is tagged with the echo tag for loop prevention.

        Args:
            channel_id: Topic to publish to when no topic is configured.
            text: Message body.
            thread_ts: Ignored (ntfy has no threading).

        Raises:
            RuntimeError: If the publish request returns a non-2xx status.
        """
        topic = self._topic or channel_id
        headers = self._auth_headers()
        headers["X-Tags"] = ",".join(self._echo_tags())
        with self._send_lock:
            resp = self._http.post(
                f"{self._server}/{topic}", data=text.encode("utf-8"), headers=headers, timeout=30
            )
        if not 200 <= resp.status_code < 300:
            raise RuntimeError(
                f"ntfy publish failed: HTTP {resp.status_code}: {resp.text[:300]}"
            )

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Return True when a polled message carries the echo tag.

        Args:
            msg: Message dict from :meth:`poll_messages`.

        Returns:
            Whether the message was published by this agent.
        """
        tags = msg.get("tags") or []
        return any(sub_tag in tags for sub_tag in self._echo_tags())

    def publish_notification(
        self,
        message: str,
        title: str = "",
        priority: str = "",
        tags: str = "",
        click_url: str = "",
    ) -> str:
        """Publish a notification to the configured ntfy topic.

        Args:
            message: Notification body text.
            title: Optional notification title.
            priority: Optional priority (``min``, ``low``, ``default``,
                ``high`` or ``urgent``, or 1-5).
            tags: Optional comma-separated tags/emoji shortcodes.
            click_url: Optional URL opened when the notification is tapped.

        Returns:
            JSON string with ok status and the published message id.
        """
        try:
            all_tags = [t.strip() for t in tags.split(",") if t.strip()]
            all_tags.extend(self._echo_tags())
            headers = self._auth_headers()
            headers["X-Tags"] = ",".join(all_tags)
            if title:
                headers["X-Title"] = title
            if priority:
                headers["X-Priority"] = priority
            if click_url:
                headers["X-Click"] = click_url
            with self._send_lock:
                resp = self._http.post(
                    f"{self._server}/{self._topic}",
                    data=message.encode("utf-8"),
                    headers=headers,
                    timeout=30,
                )
            if not 200 <= resp.status_code < 300:
                return json.dumps(
                    {
                        "ok": False,
                        "error": (
                            f"ntfy publish failed: HTTP {resp.status_code}: "
                            f"{resp.text[:300]}"
                        ),
                    }
                )
            try:
                message_id = str(resp.json().get("id", ""))
            except ValueError:
                message_id = ""
            return json.dumps({"ok": True, "id": message_id})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def poll_topic(self, since: str = "all", limit: int = 10) -> str:
        """Read recent messages from the configured ntfy topic.

        Args:
            since: ``since`` cursor — ``all`` for the full cache, or a
                unix timestamp string returned by a previous poll.
            limit: Maximum number of messages to return.

        Returns:
            JSON object with ``ok`` status: on success ``messages`` holds
            message dicts with ``ts``, ``user``, ``text``, ``title``,
            ``channel_id`` and ``tags`` keys; on failure ``error`` holds
            the failure reason.
        """
        try:
            messages, _ = self._fetch_messages(self._topic, "" if since == "all" else since, limit)
            return json.dumps({"ok": True, "messages": messages})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class NtfyAgent(BaseChannelAgent):
    """Channel agent with ntfy pub-sub notification tools."""

    channel_system_prompt = (
        "You are chatting via ntfy (https://ntfy.sh), a topic-based HTTP "
        "pub-sub notification service. Messages are plain text published "
        "to a topic; there are no threads, users or rich formatting. Use "
        "publish_notification to send notifications (optionally with a "
        "title, priority, tags and click URL) and poll_topic to read "
        "recent messages from the topic."
    )

    def __init__(self) -> None:
        super().__init__("Ntfy Agent")
        self._backend = NtfyChannelBackend()
        cfg = _config.load()
        if cfg:  # pragma: no branch
            self._backend._apply_config(cfg)

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated.

        A configured token whose Muse-boundary wiring failed leaves the
        backend deliberately tokenless: it is NOT authenticated, so the
        backend tools stay hidden and no request can slip out on the
        direct transport without a credential.
        """
        return bool(self._backend._topic) and not self._backend._muse_error

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_ntfy_auth() -> str:
            """Check if ntfy is configured.

            Returns:
                Configuration status or instructions.
            """
            if not agent._backend._topic:  # pragma: no branch
                return (
                    "Not configured for ntfy. Use authenticate_ntfy() to configure.\n"
                    "You need a topic name; the public https://ntfy.sh server "
                    "works without a token, self-hosted servers may need an "
                    "access token."
                )
            if agent._backend._muse_error:
                return json.dumps({"ok": False, "error": agent._backend._muse_error})
            return json.dumps(
                {
                    "ok": True,
                    "server": agent._backend._server,
                    "topic": agent._backend._topic,
                    "echo_tag": agent._backend._echo_tag,
                }
            )

        def authenticate_ntfy(
            topic: str, server: str = "", token: str = "", echo_tag: str = ""
        ) -> str:
            """Configure the ntfy topic and server.

            Args:
                topic: ntfy topic name to publish to and poll.
                server: ntfy server base URL (default ``https://ntfy.sh``).
                token: Optional access token sent as ``Authorization: Bearer``.
                echo_tag: Tag marking the agent's own messages for loop
                    prevention (default ``kiss-sorcar``).

            Returns:
                Configuration result or error message.
            """
            if not topic.strip():  # pragma: no branch
                return "topic cannot be empty."
            cfg = {
                "topic": topic.strip(),
                "server": (server.strip() or _DEFAULT_SERVER).rstrip("/"),
                "token": token.strip(),
                "echo_tag": echo_tag.strip() or _DEFAULT_ECHO_TAG,
            }
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if muse_auth_enabled():
                # Transactional: the token goes straight into the vault
                # (never onto disk) BEFORE any state changes, so a
                # failed enrollment leaves the previous credential and
                # config untouched.  A deliberately tokenless setup
                # drops any stale vault credential instead.
                from kiss.agents.third_party_agents.muse_auth._common import valid_http_url
                from kiss.agents.third_party_agents.muse_auth.client import (
                    MuseAuthError,
                    clear_credentials,
                    store_credentials,
                )

                if not valid_http_url(cfg["server"]):
                    return "server must be an http(s):// URL with a hostname and valid port."
                persisted = {k: v for k, v in cfg.items() if k != "token"}
                try:
                    # Write the non-secret metadata FIRST: it never holds
                    # the token, so a failure here leaves the prior vault
                    # credential and config intact (transactional order).
                    _config.save(persisted)
                    if cfg["token"]:
                        store_credentials(
                            "ntfy",
                            {"kind": "bearer", "token": cfg["token"]},
                            [],
                            hosts=_extra_hosts(cfg["server"]),
                            insecure_hosts=_insecure_extra_hosts(cfg["server"]),
                        )
                    else:
                        clear_credentials("ntfy")
                    agent._backend._apply_config(persisted)
                except (MuseAuthError, OSError) as e:
                    return json.dumps({"ok": False, "error": str(e)})
                if agent._backend._muse_error:  # pragma: no cover - daemon race
                    return json.dumps({"ok": False, "error": agent._backend._muse_error})
                return json.dumps({"ok": True, "message": "ntfy configured."})
            _config.save(cfg)
            agent._backend._apply_config(cfg)
            return json.dumps({"ok": True, "message": "ntfy configured."})

        def clear_ntfy_auth() -> str:
            """Clear the stored ntfy configuration.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._topic = ""
            agent._backend._server = _DEFAULT_SERVER
            agent._backend._token = ""
            agent._backend._echo_tag = _DEFAULT_ECHO_TAG
            agent._backend._http = requests
            agent._backend._muse = False
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if muse_auth_enabled():
                from kiss.agents.third_party_agents.muse_auth.client import clear_credentials

                clear_credentials("ntfy")
            return "ntfy configuration cleared."

        return [check_ntfy_auth, authenticate_ntfy, clear_ntfy_auth]


def _make_backend() -> NtfyChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = NtfyChannelBackend()
    cfg = _config.load()
    if not cfg:  # pragma: no branch
        print("Not configured. Run: kiss-ntfy -t 'authenticate'")
        sys.exit(1)
    backend._apply_config(cfg)
    return backend


def main() -> None:
    """Run the NtfyAgent from the command line with chat persistence."""
    channel_main(NtfyAgent, "kiss-ntfy", channel_name="ntfy", make_backend=_make_backend)


def tools() -> list:
    """Return the ntfy channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return NtfyAgent()._get_tools()


if __name__ == "__main__":
    main()
