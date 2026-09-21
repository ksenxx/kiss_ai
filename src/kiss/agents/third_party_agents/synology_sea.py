# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Synology Chat Agent — channel agent with Synology Chat webhook API.

Provides access to Synology Chat via incoming and outgoing webhooks.
Stores config in ``~/.kiss/third_party_agents/synology/config.json``.

Usage::

    agent = SynologyChatAgent()
    agent.run(prompt_template="Send a message to the team")
"""

from __future__ import annotations

import json
import logging
import queue
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import requests

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

_DEFAULT_WEBHOOK_PORT = 18083

_SYNOLOGY_DIR = Path.home() / ".kiss" / "third_party_agents" / "synology"
_config = ChannelConfig(_SYNOLOGY_DIR, ("webhook_url",))


def _embedded_token(webhook_url: str) -> str:
    """Return the ``token`` query parameter embedded in a webhook URL.

    Synology Chat incoming-webhook URLs carry their secret as a
    ``token=...`` query parameter; in Muse mode that secret moves into
    the vault and the stored URL keeps only the non-secret parts.

    Args:
        webhook_url: The configured incoming webhook URL.

    Returns:
        The embedded token, or ``""`` when the URL carries none.
    """
    from urllib.parse import parse_qs, urlsplit

    values = parse_qs(urlsplit(webhook_url).query).get("token", [])
    return values[0] if values else ""


def _scrub_config_webhook_token() -> None:
    """Remove the vault-migrated embedded ``token`` from the webhook URL.

    Finishes the Muse migration automatically: ``config.json`` keeps the
    webhook URL minus its secret query parameter (plus the separate,
    inbound-only outgoing-webhook verification ``token``, which never
    leaves this machine).
    """
    from kiss.agents.third_party_agents.muse_auth._common import strip_url_query_param

    try:
        cfg = json.loads(_config.path.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(cfg, dict) or not _embedded_token(str(cfg.get("webhook_url", ""))):
        return
    cfg["webhook_url"] = strip_url_query_param(str(cfg["webhook_url"]), "token")
    # Non-secret migration marker: distinguishes a Muse-scrubbed URL
    # (which may reuse its vault credential) from an explicitly
    # tokenless webhook URL (which must never revive a stale one).
    cfg["muse"] = "1"
    from kiss.agents.third_party_agents._channel_agent_utils import save_json_config

    save_json_config(_config.path, {k: str(v) for k, v in cfg.items()})


class SynologyChatChannelBackend(ToolMethodBackend):
    """Channel backend for Synology Chat webhooks.

    Sends messages via the incoming webhook URL. Receives messages
    via a webhook server (outgoing webhook from Synology Chat).
    """

    def __init__(self) -> None:
        self._webhook_url: str = ""
        self._token: str = ""
        self._http: Any = requests
        self._muse: bool = False
        self._surrogate: str = ""
        self._message_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._webhook_server: ThreadedHTTPServer | None = None
        self._webhook_thread: threading.Thread | None = None
        self._send_lock = threading.Lock()
        self._connection_info: str = ""

    def _headers(self) -> dict[str, str]:
        """Return per-request headers: the surrogate bearer in Muse mode.

        Returns:
            ``{"Authorization": "Bearer <surrogate>"}`` in Muse mode
            (the daemon validates and strips it, then splices the real
            ``token=`` query parameter into the webhook URL), else
            ``{}``.
        """
        if self._muse:
            return {"Authorization": f"Bearer {self._surrogate}"}
        return {}

    def _wire_direct(self, webhook_url: str) -> None:
        """Wire the legacy direct path, resetting any earlier Muse state.

        A re-wire after a tokenless rotation must not keep sending
        through the boundary with a stale surrogate.

        Args:
            webhook_url: The tokenless incoming webhook URL to use.
        """
        self._webhook_url = webhook_url
        self._muse = False
        self._surrogate = ""
        self._http = requests

    def _wire_muse(self) -> bool:
        """Wire the Muse boundary for the configured incoming webhook.

        A ``token=`` query parameter still embedded in the configured
        webhook URL is the newest user intent (initial migration, or a
        rotation done while Muse was off): it is enrolled as a
        query-kind credential bound to the webhook's origin (flagged as
        a consent-scoped insecure host when the URL is plain ``http`` to
        a non-loopback host), and the stored URL keeps only the
        non-secret parts plus a ``muse`` migration marker.  A tokenless
        webhook URL without that marker is explicit user intent to run
        without a secret and stays on the legacy direct path — a stale
        vault credential is never revived into it.  No network round
        trip happens here.

        Returns:
            True when the backend is usable (boundary-wired or
            tokenless-legacy); False on a missing/invalid config.
        """
        from kiss.agents.third_party_agents.muse_auth._common import (
            insecure_origin_hosts,
            origin_hosts,
            strip_url_query_param,
            valid_http_url,
        )
        from kiss.agents.third_party_agents.muse_auth.client import (
            MuseBoundarySession,
            mint_surrogate,
            store_credentials,
        )

        cfg = _config.load_metadata() or {}
        webhook_url = str(cfg.get("webhook_url") or "")
        if not webhook_url:
            self._connection_info = "No Synology Chat config found."
            return False
        # Validate before any credential state changes: a malformed
        # legacy URL must not auto-migrate the token into a host scope
        # Sentinel can never match, nor scrub it from the stored URL.
        if not valid_http_url(webhook_url):
            self._connection_info = (
                f"Synology webhook URL {webhook_url!r} is not a valid http(s):// URL; "
                "fix config.json and reconnect."
            )
            return False
        self._token = str(cfg.get("token") or "")
        embedded = _embedded_token(webhook_url)
        if embedded:
            store_credentials(
                "synology",
                {"kind": "query", "param": "token", "token": embedded},
                [],
                hosts=origin_hosts(webhook_url),
                insecure_hosts=insecure_origin_hosts(webhook_url),
            )
        elif str(cfg.get("muse") or "") != "1":
            # A tokenless webhook URL without the Muse migration marker
            # is explicit user intent to run without a webhook secret
            # (a scrubbed URL always carries the marker).  Never revive
            # a stale vault credential into it.
            self._wire_direct(webhook_url)
            return True
        handle = mint_surrogate("synology")
        if handle is None:
            # No embedded token and no vault enrollment: the webhook URL
            # itself is the whole configuration; keep the legacy direct
            # path.
            self._wire_direct(webhook_url)
            return True
        _scrub_config_webhook_token()
        self._webhook_url = strip_url_query_param(webhook_url, "token")
        self._surrogate = handle.token
        self._http = MuseBoundarySession("synology")
        self._muse = True
        return True

    def connect(self) -> bool:
        """Load Synology config and start webhook server."""
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        if muse_auth_enabled():
            if not self._wire_muse():
                return False
        else:
            cfg = _config.load()
            if not cfg:  # pragma: no branch
                self._connection_info = "No Synology Chat config found."
                return False
            self._webhook_url = cfg["webhook_url"]
            self._token = cfg.get("token", "")
        self._connection_info = "Synology Chat configured"
        if not self._start_webhook_server():  # pragma: no branch
            return False
        return True

    def _start_webhook_server(self, port: int = _DEFAULT_WEBHOOK_PORT) -> bool:
        """Start the outgoing webhook HTTP server."""
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    params = parse_qs(body.decode("utf-8"))
                    if "payload" in params:
                        payload: dict[str, Any] = json.loads(params["payload"][0])
                    else:
                        payload = {k: v[0] for k, v in params.items()}
                    token = str(payload.get("token", ""))
                    if not backend._token or token == backend._token:  # pragma: no branch
                        backend._message_queue.put(
                            {
                                "ts": str(payload.get("timestamp", "")),
                                "user": str(payload.get("user_id", "")),
                                "username": str(payload.get("username", "")),
                                "text": str(payload.get("text", "")),
                                "channel_id": str(payload.get("channel_id", "")),
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
            started_log="Synology Chat webhook server started on port %d",
            error_prefix="Synology webhook bind failed",
            error_log="Could not start Synology webhook server: %s",
        )
        if error is not None:
            self._connection_info = error
            return False
        return True

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Drain the webhook message queue.

        Drained messages not matching ``channel_id`` are discarded.
        """
        messages = drain_queue_messages(
            self._message_queue,
            limit=limit,
            keep=lambda msg: not channel_id or msg.get("channel_id") == channel_id,
        )
        return messages, oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a Synology Chat message via the incoming webhook.

        Incoming webhooks are bound to a fixed channel at creation, so
        ``channel_id`` is ignored.

        Raises:
            RuntimeError: If the webhook request fails.
        """
        with self._send_lock:
            resp = self._http.post(
                self._webhook_url,
                data={"payload": json.dumps({"text": text})},
                headers=self._headers(),
                timeout=30,
            )
        if resp.status_code != 200:
            raise RuntimeError(f"Synology Chat send failed: HTTP {resp.status_code}")
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if isinstance(data, dict) and data.get("success") is False:
            raise RuntimeError(f"Synology Chat send failed: {data}")

    def disconnect(self) -> None:
        """Stop the embedded webhook server and release backend resources."""
        self._webhook_server, self._webhook_thread = stop_http_server(
            self._webhook_server, self._webhook_thread
        )

    def post_message(self, text: str, user_ids: str = "") -> str:
        """Send a message to Synology Chat via incoming webhook.

        Args:
            text: Message text.
            user_ids: Comma-separated user IDs to send to (optional).
                If empty, sends to the default channel.

        Returns:
            JSON string with ok status.
        """
        try:
            payload: dict[str, Any] = {"text": text}
            if user_ids:  # pragma: no branch
                payload["user_ids"] = [u.strip() for u in user_ids.split(",") if u.strip()]
            with self._send_lock:
                resp = self._http.post(
                    self._webhook_url,
                    data={"payload": json.dumps(payload)},
                    headers=self._headers(),
                    timeout=30,
                )
            return json.dumps({"ok": resp.status_code == 200})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_file_message(self, text: str, file_url: str) -> str:
        """Send a message with a file attachment.

        Args:
            text: Message text.
            file_url: URL of the file to attach.

        Returns:
            JSON string with ok status.
        """
        try:
            payload = {"text": text, "file_url": file_url}
            with self._send_lock:
                resp = self._http.post(
                    self._webhook_url,
                    data={"payload": json.dumps(payload)},
                    headers=self._headers(),
                    timeout=30,
                )
            return json.dumps({"ok": resp.status_code == 200})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


def _muse_authenticate(
    backend: SynologyChatChannelBackend, webhook_url: str, outgoing_token: str
) -> str:
    """Enroll a Synology webhook secret into the Muse vault.

    The ``token=`` query parameter embedded in the incoming webhook URL
    goes straight into the vault as a query-kind credential bound to the
    webhook's origin; ``config.json`` stores the URL without it (written
    first, so a failed enrollment leaves no new secret on disk).  The
    separate *outgoing_token* only verifies inbound webhook posts and
    never leaves this machine, so it stays in the config.  A webhook URL
    with no embedded token has no secret to protect: it is stored as-is
    on the legacy direct path and any stale vault credential is cleared
    so it cannot be revived into the new URL.  On failure the pre-call
    config is restored.  Like the legacy tool, no validation request is
    sent — an incoming webhook has no read endpoint.

    Args:
        backend: The agent's Synology backend to (re)wire.
        webhook_url: Synology Chat incoming webhook URL.
        outgoing_token: Optional outgoing-webhook verification token.

    Returns:
        JSON string with the configuration result.
    """
    import contextlib
    import os

    from kiss.agents.third_party_agents.muse_auth._common import (
        insecure_origin_hosts,
        origin_hosts,
        strip_url_query_param,
        valid_credential_value,
        valid_http_url,
    )
    from kiss.agents.third_party_agents.muse_auth.client import (
        clear_credentials,
        store_credentials,
    )

    if not valid_http_url(webhook_url):
        return json.dumps(
            {"ok": False, "error": f"{webhook_url!r} is not a valid http(s):// webhook URL."}
        )
    embedded = _embedded_token(webhook_url)
    if embedded and not valid_credential_value(embedded):
        # Pre-validate with the vault's own rule so a doomed enrollment
        # never mutates the stored configuration first.
        return json.dumps(
            {
                "ok": False,
                "error": "the token= parameter embedded in the webhook URL contains "
                "control characters or stray whitespace; regenerate the webhook and retry.",
            }
        )
    stored_url = strip_url_query_param(webhook_url, "token") if embedded else webhook_url
    try:
        prev_raw: str | None = _config.path.read_text()
    except OSError:
        prev_raw = None
    try:
        cfg = {"webhook_url": stored_url}
        if embedded:
            cfg["muse"] = "1"
        if outgoing_token:
            cfg["token"] = outgoing_token
        _config.save(cfg)
        if embedded:
            store_credentials(
                "synology",
                {"kind": "query", "param": "token", "token": embedded},
                [],
                hosts=origin_hosts(webhook_url),
                insecure_hosts=insecure_origin_hosts(webhook_url),
            )
        else:
            # Explicit tokenless configuration: drop any stale vault
            # credential instead of letting a later wire revive it into
            # the new URL.
            clear_credentials("synology")
        if backend._wire_muse():  # pragma: no branch - config was just saved
            return json.dumps({"ok": True, "message": "Synology Chat configured (Muse-auth)."})
        return json.dumps(  # pragma: no cover - defense in depth
            {"ok": False, "error": backend._connection_info}
        )
    except Exception as e:
        # Restore the pre-call config so a failed enrollment leaves no
        # half-migrated state.  The restored bytes are exactly what was
        # already on disk, so no new secret lands in the file.
        with contextlib.suppress(Exception):
            if prev_raw is None:
                _config.clear()
            else:
                _config.path.write_text(prev_raw)
                os.chmod(_config.path, 0o600)
        return json.dumps({"ok": False, "error": str(e)})


class SynologyChatAgent(BaseChannelAgent):
    """Channel agent with Synology Chat webhook tools."""

    def __init__(self) -> None:
        super().__init__("Synology Chat Agent")
        self._backend = SynologyChatChannelBackend()
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        if muse_auth_enabled():
            from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError

            # Muse-auth mode: wire a vault surrogate and the boundary
            # session (no network round trip); the webhook secret never
            # enters this process once migrated.  A daemon failure
            # leaves the agent constructible (fail closed) so its
            # authenticate/clear tools stay available.
            try:
                self._backend._wire_muse()
            except MuseAuthError as e:
                self._backend._webhook_url = ""
                self._backend._connection_info = f"Muse-auth wiring failed: {e}"
            return
        cfg = _config.load()
        if cfg:  # pragma: no branch
            self._backend._webhook_url = cfg["webhook_url"]
            self._backend._token = cfg.get("token", "")

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._webhook_url)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_synology_auth() -> str:
            """Check if Synology Chat is configured.

            Returns:
                Configuration status or instructions.
            """
            if not agent._backend._webhook_url:  # pragma: no branch
                return (
                    "Not configured for Synology Chat. Use authenticate_synology() to configure.\n"
                    "You need the incoming webhook URL from Synology Chat > "
                    "Integration > Incoming Webhooks > Create."
                )
            return json.dumps(
                {
                    "ok": True,
                    "webhook_url": agent._backend._webhook_url[:50] + "...",
                }
            )

        def authenticate_synology(webhook_url: str, token: str = "") -> str:
            """Configure Synology Chat webhook.

            Args:
                webhook_url: Synology Chat incoming webhook URL.
                token: Optional outgoing webhook token for verification.

            Returns:
                Configuration result or error message.
            """
            if not webhook_url.strip():  # pragma: no branch
                return "webhook_url cannot be empty."
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if muse_auth_enabled():
                return _muse_authenticate(agent._backend, webhook_url.strip(), token.strip())
            agent._backend._webhook_url = webhook_url.strip()
            agent._backend._token = token.strip()
            _config.save({"webhook_url": webhook_url.strip(), "token": token.strip()})
            return json.dumps({"ok": True, "message": "Synology Chat configured."})

        def clear_synology_auth() -> str:
            """Clear the stored Synology Chat configuration.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._webhook_url = ""
            agent._backend._token = ""
            agent._backend._surrogate = ""
            agent._backend._http = requests
            agent._backend._muse = False
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if muse_auth_enabled():
                from kiss.agents.third_party_agents.muse_auth.client import clear_credentials

                clear_credentials("synology")
            return "Synology Chat configuration cleared."

        return [check_synology_auth, authenticate_synology, clear_synology_auth]


def _make_backend() -> SynologyChatChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = SynologyChatChannelBackend()
    from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

    if muse_auth_enabled():
        if backend._wire_muse():
            return backend
        print("Not configured. Run: kiss-synology -t 'authenticate'")
        sys.exit(1)
    cfg = _config.load()
    if not cfg:  # pragma: no branch
        print("Not configured. Run: kiss-synology -t 'authenticate'")
        sys.exit(1)
    backend._webhook_url = cfg["webhook_url"]
    backend._token = cfg.get("token", "")
    return backend


def main() -> None:
    """Run the SynologyChatAgent from the command line with chat persistence."""
    channel_main(
        SynologyChatAgent,
        "kiss-synology",
        channel_name="Synology Chat",
        make_backend=_make_backend,
    )


def tools() -> list:
    """Return the Synology Chat channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return SynologyChatAgent()._get_tools()


if __name__ == "__main__":
    main()
