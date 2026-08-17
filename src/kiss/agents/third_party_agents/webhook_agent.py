# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Webhook Routes Agent — channel agent for Hermes-style inbound webhooks.

Runs an embedded HTTP server that accepts ``POST /hook/<route-name>``
requests from external systems (GitHub, CI, monitoring, ...), verifies
an HMAC-SHA256 signature per route, and turns each accepted event into
a normalized channel message rendered from the route's prompt template.
Stores config in ``~/.kiss/third_party_agents/webhook/config.json``
(``port`` plus a JSON-encoded ``routes`` map).

Two signature schemes are supported per route:

- ``github``: ``X-Hub-Signature-256`` must equal
  ``"sha256=" + HMAC-SHA256(secret, raw_body)`` hex digest.
- ``generic``: ``X-Kiss-Timestamp`` (unix seconds, rejected when skewed
  more than 300 s) and ``X-Kiss-Signature`` equal to
  ``HMAC-SHA256(secret, f"{timestamp}." + raw_body)`` hex digest.

Additional per-route protections: 1 MB body cap (413, with missing /
malformed ``Content-Length`` rejected as 411/400 before reading),
duplicate delivery-id suppression via ``X-GitHub-Delivery`` /
``X-Kiss-Delivery`` scoped per route and recorded only after the event
is queued or delivered (LRU of 256), rate limit of 60 accepted events
per route per minute (429), and optional payload filters (dot-path ->
expected string) that silently drop non-matching events.  Routes with
``deliver_module`` set are deliver-only: the rendered text is pushed
through that channel module's backend instead of being queued for the
agent, and delivery failure returns 502 so the sender can retry.

Usage::

    agent = WebhookAgent()
    agent.run(prompt_template="Add a webhook route for GitHub pushes")
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import logging
import math
import queue
import re
import sys
import threading
import time
from collections import OrderedDict, deque
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from kiss.agents.third_party_agents._backend_utils import (
    ThreadedHTTPServer,
    drain_queue_messages,
    drain_request_body,
    stop_http_server,
)
from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

logger = logging.getLogger(__name__)

_DEFAULT_PORT = "18090"
_MAX_BODY_BYTES = 1024 * 1024
_MAX_DEDUP_IDS = 256
_RATE_LIMIT_EVENTS = 60
_RATE_LIMIT_WINDOW_SECONDS = 60.0
_MAX_TIMESTAMP_SKEW_SECONDS = 300.0
_ROUTE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_][A-Za-z0-9_.\-]*)\}")
_MISSING = object()

_WEBHOOK_DIR = Path.home() / ".kiss" / "third_party_agents" / "webhook"
_config = ChannelConfig(_WEBHOOK_DIR, ("port",))


def _parse_routes(cfg: dict[str, str] | None) -> dict[str, dict[str, Any]]:
    """Decode the JSON-encoded ``routes`` config value.

    Args:
        cfg: Loaded config dict, or ``None`` when unconfigured.

    Returns:
        Route-name -> route-definition map (empty on any problem).
    """
    if not cfg:
        return {}
    try:
        routes = json.loads(cfg.get("routes") or "{}")
    except ValueError:
        return {}
    if not isinstance(routes, dict):
        return {}
    return {str(k): v for k, v in routes.items() if isinstance(v, dict)}


def _save_routes(routes: dict[str, dict[str, Any]]) -> None:
    """Persist *routes* into the JSON-encoded ``routes`` config value.

    Args:
        routes: Route-name -> route-definition map to persist.

    Raises:
        RuntimeError: If the webhook config does not exist yet.
    """
    cfg = _config.load()
    if not cfg:
        raise RuntimeError("Not configured. Use authenticate_webhook() first.")
    cfg["routes"] = json.dumps(routes)
    _config.save(cfg)


def _lookup(payload: Any, dot_path: str) -> Any:
    """Resolve a dot-path (e.g. ``pull_request.user.login``) in a JSON payload.

    Dict keys and integer list indices are supported.

    Args:
        payload: Decoded JSON payload.
        dot_path: Dot-separated path.

    Returns:
        The value at the path, or the module ``_MISSING`` sentinel.
    """
    current = payload
    for part in dot_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return _MISSING
    return current


def _render_template(template: str, payload: Any) -> str:
    """Render ``{dot.path}`` placeholders in *template* from *payload*.

    ``{payload}`` expands to the full compact JSON; unknown placeholders
    are left verbatim.

    Args:
        template: Prompt template with ``{...}`` placeholders.
        payload: Decoded JSON payload.

    Returns:
        The rendered text.
    """
    compact = json.dumps(payload, separators=(",", ":"))

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key == "payload":
            return compact
        value = _lookup(payload, key)
        return match.group(0) if value is _MISSING else str(value)

    return _PLACEHOLDER_RE.sub(_replace, template)


class WebhookChannelBackend(ToolMethodBackend):
    """Channel backend embedding an HTTP server for inbound webhooks.

    Each configured route (``POST /hook/<name>``) verifies an HMAC
    signature, applies payload filters, renders a prompt template, and
    queues the result as a channel message (or delivers it through
    another channel module for deliver-only routes).
    """

    def __init__(self) -> None:
        self._port: str = ""
        self._bound_port: int = 0
        self._routes: dict[str, dict[str, Any]] = {}
        self._message_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._server: ThreadedHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._seen_delivery_ids: OrderedDict[str, None] = OrderedDict()
        self._inflight_delivery_ids: set[str] = set()
        self._accept_times: dict[str, deque[float]] = {}
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Load the webhook config and start the embedded HTTP server."""
        cfg = _config.load()
        if not cfg:
            self._connection_info = "No webhook config found."
            return False
        self._port = cfg["port"]
        with self._state_lock:
            self._routes = _parse_routes(cfg)
        return self._start_server()

    def _start_server(self) -> bool:
        """Bind and start the webhook HTTP server on the configured port."""
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.unread_body_length: int | None = None
                status = backend._handle_request(self)
                self.send_response(status)
                self.end_headers()
                # A rejected request's body was never read; drain it
                # (bounded) AFTER the response so the client reads the
                # status code instead of a connection reset — see
                # ``drain_request_body``.
                drain_request_body(self, self.unread_body_length)

            def log_message(self, *args: Any) -> None:  # type: ignore[override]
                pass

        self.disconnect()
        try:
            self._server = ThreadedHTTPServer(("0.0.0.0", int(self._port)), Handler)
        except (OSError, ValueError) as e:
            self._connection_info = f"Webhook server bind failed: {e}"
            logger.warning("Could not start webhook server: %s", e)
            self._server = None
            self._server_thread = None
            return False
        self._bound_port = self._server.server_address[1]
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        self._connection_info = (
            f"Webhook server listening on port {self._bound_port} ({len(self._routes)} route(s))"
        )
        logger.info("%s", self._connection_info)
        return True

    def _handle_request(self, handler: BaseHTTPRequestHandler) -> int:
        """Process one inbound POST request and return the HTTP status code.

        A rejection leaves the request body unread; its parsed length is
        recorded on ``handler.unread_body_length`` so ``do_POST`` can
        drain it after sending the response (see
        :func:`drain_request_body`) and the client reads the status code
        instead of a connection reset.
        """
        path = urlsplit(handler.path).path
        raw_length = handler.headers.get("Content-Length")
        length: int | None = None
        if raw_length is not None:
            raw_length = raw_length.strip()
            if raw_length.isascii() and raw_length.isdecimal() and len(raw_length) <= 10:
                length = int(raw_length)
        handler.unread_body_length = length  # type: ignore[attr-defined]
        if not path.startswith("/hook/"):
            return 404
        name = path[len("/hook/") :]
        with self._state_lock:
            route = self._routes.get(name)
        if route is None:
            return 404
        if raw_length is None:
            return 411
        if length is None:
            return 400
        if length > _MAX_BODY_BYTES:
            return 413
        body = handler.rfile.read(length)
        handler.unread_body_length = None  # type: ignore[attr-defined]
        return self._handle_event(name, route, handler.headers, body)

    def _handle_event(self, name: str, route: dict[str, Any], headers: Any, body: bytes) -> int:
        """Verify, dedup, rate-limit, filter, render, and dispatch one event."""
        if not self._verify_signature(route, headers, body):
            logger.warning("Webhook route %r: rejected bad/missing signature", name)
            return 401
        delivery_id = headers.get("X-GitHub-Delivery") or headers.get("X-Kiss-Delivery") or ""
        dedup_key = f"{name}:{delivery_id}" if delivery_id else ""
        now = time.time()
        with self._state_lock:
            if dedup_key and (
                dedup_key in self._seen_delivery_ids or dedup_key in self._inflight_delivery_ids
            ):
                logger.info("Webhook route %r: dropped duplicate delivery %s", name, delivery_id)
                return 200
            window = self._accept_times.setdefault(name, deque())
            while window and now - window[0] > _RATE_LIMIT_WINDOW_SECONDS:
                window.popleft()
            if len(window) >= _RATE_LIMIT_EVENTS:
                logger.warning("Webhook route %r: rate limit exceeded", name)
                return 429
            window.append(now)
            if dedup_key:
                # Reserve the id while this event is in flight so a
                # concurrent retry cannot trigger a duplicate delivery.
                self._inflight_delivery_ids.add(dedup_key)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._release_delivery(dedup_key)
            return 400
        filters = route.get("filters") or {}
        for dot_path, expected in filters.items():
            value = _lookup(payload, dot_path)
            if value is _MISSING or str(value) != str(expected):
                logger.info("Webhook route %r: dropped event (filter %r)", name, dot_path)
                self._record_delivery(dedup_key)
                return 200
        text = _render_template(str(route.get("prompt_template") or "{payload}"), payload)
        if route.get("deliver_module"):
            delivered = self._deliver(
                str(route["deliver_module"]), str(route.get("deliver_channel", "")), text
            )
            if not delivered:
                self._release_delivery(dedup_key)
                return 502
            self._record_delivery(dedup_key)
            return 202
        self._message_queue.put(
            {"ts": f"{now:.6f}", "user": name, "text": text, "channel_id": name}
        )
        self._record_delivery(dedup_key)
        logger.info("Webhook route %r: accepted event", name)
        return 200

    def _record_delivery(self, dedup_key: str) -> None:
        """Mark a terminally handled delivery id as processed (LRU-bounded)."""
        if not dedup_key:
            return
        with self._state_lock:
            self._inflight_delivery_ids.discard(dedup_key)
            self._seen_delivery_ids[dedup_key] = None
            if len(self._seen_delivery_ids) > _MAX_DEDUP_IDS:
                self._seen_delivery_ids.popitem(last=False)

    def _release_delivery(self, dedup_key: str) -> None:
        """Release an in-flight delivery id after a retryable failure."""
        if not dedup_key:
            return
        with self._state_lock:
            self._inflight_delivery_ids.discard(dedup_key)

    def _verify_signature(self, route: dict[str, Any], headers: Any, body: bytes) -> bool:
        """Check the route's HMAC signature scheme against the request."""
        secret = str(route.get("secret", ""))
        if route.get("kind") == "github":
            signature = headers.get("X-Hub-Signature-256", "") or ""
            expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
            return bool(signature) and hmac.compare_digest(signature, expected)
        timestamp = headers.get("X-Kiss-Timestamp", "") or ""
        signature = headers.get("X-Kiss-Signature", "") or ""
        try:
            ts_value = float(timestamp)
        except ValueError:
            return False
        if not math.isfinite(ts_value):
            return False
        if abs(time.time() - ts_value) > _MAX_TIMESTAMP_SKEW_SECONDS:
            return False
        expected = hmac.new(
            secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256
        ).hexdigest()
        return bool(signature) and hmac.compare_digest(signature, expected)

    def _deliver(self, module_name: str, channel: str, text: str) -> bool:
        """Push *text* through another channel module's backend (audit-logged).

        Returns:
            True when the message was sent, False on any failure (import
            error, ``sys.exit`` from the backend factory, ``connect()``
            not returning True, or a send error).  The delivery backend
            is always disconnected.
        """
        delivery_backend = None
        try:
            module = importlib.import_module(module_name)
            delivery_backend = module._make_backend()
            if delivery_backend.connect() is not True:
                logger.error(
                    "Webhook delivery via %s to %r failed: connect() failed", module_name, channel
                )
                return False
            delivery_backend.send_message(channel, text)
            logger.info("Webhook delivery via %s to %r succeeded", module_name, channel)
            return True
        except (Exception, SystemExit):
            logger.error(
                "Webhook delivery via %s to %r failed", module_name, channel, exc_info=True
            )
            return False
        finally:
            if delivery_backend is not None:
                try:
                    delivery_backend.disconnect()
                except Exception:
                    logger.warning(
                        "Webhook delivery backend %s disconnect failed", module_name, exc_info=True
                    )

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Drain queued webhook events.

        Drained messages not matching ``channel_id`` (a route name) are
        discarded.
        """
        messages = drain_queue_messages(
            self._message_queue,
            limit=limit,
            keep=lambda msg: not channel_id or msg.get("channel_id") == channel_id,
        )
        return messages, oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a reply for a webhook route.

        Webhooks are inbound-only, so replies are pushed through the
        route's ``deliver_module``/``deliver_channel`` when configured,
        and otherwise logged and dropped.
        """
        with self._state_lock:
            route = self._routes.get(channel_id)
        if route and route.get("deliver_module"):
            self._deliver(str(route["deliver_module"]), str(route.get("deliver_channel", "")), text)
            return
        logger.info(
            "Webhook route %r has no delivery target; reply dropped: %.200s", channel_id, text
        )

    def disconnect(self) -> None:
        """Stop the embedded webhook server and release backend resources."""
        self._server, self._server_thread = stop_http_server(self._server, self._server_thread)
        self._bound_port = 0

    def add_webhook_route(
        self,
        name: str,
        secret: str,
        kind: str = "generic",
        prompt_template: str = "{payload}",
        filters_json: str = "",
        deliver_module: str = "",
        deliver_channel: str = "",
    ) -> str:
        """Add (or replace) an inbound webhook route at ``POST /hook/<name>``.

        Args:
            name: Route name (letters, digits, ``_``, ``-``); also the URL path
                segment and the channel/user id of queued messages.
            secret: Shared HMAC-SHA256 secret used to verify requests.
            kind: Signature scheme — ``"github"`` (``X-Hub-Signature-256``) or
                ``"generic"`` (``X-Kiss-Timestamp`` + ``X-Kiss-Signature``).
            prompt_template: Text template; ``{a.b.c}`` placeholders are
                replaced from the JSON payload, ``{payload}`` by the full
                compact JSON.
            filters_json: Optional JSON object of dot-path -> expected string;
                events where any filter does not match are dropped.
            deliver_module: Optional channel module (e.g.
                ``kiss.agents.third_party_agents.synology_chat_agent``); when
                set the route is deliver-only: rendered text is sent through
                that module instead of being queued for the agent.
            deliver_channel: Channel id passed to the delivery module.

        Returns:
            JSON string with ok status.
        """
        try:
            if not _ROUTE_NAME_RE.match(name):
                return json.dumps({"ok": False, "error": "Route name must match [A-Za-z0-9_-]+."})
            if not secret:
                return json.dumps({"ok": False, "error": "secret cannot be empty."})
            if kind not in ("generic", "github"):
                return json.dumps({"ok": False, "error": "kind must be 'generic' or 'github'."})
            filters: dict[str, str] = {}
            if filters_json:
                parsed = json.loads(filters_json)
                if not isinstance(parsed, dict):
                    return json.dumps({"ok": False, "error": "filters_json must be a JSON object."})
                filters = {str(k): str(v) for k, v in parsed.items()}
            route = {
                "secret": secret,
                "kind": kind,
                "prompt_template": prompt_template,
                "filters": filters,
                "deliver_module": deliver_module,
                "deliver_channel": deliver_channel,
            }
            with self._state_lock:
                routes = dict(self._routes)
                routes[name] = route
                _save_routes(routes)
                self._routes = routes
            return json.dumps(
                {"ok": True, "message": f"Route {name!r} saved.", "path": f"/hook/{name}"}
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def remove_webhook_route(self, name: str) -> str:
        """Remove an inbound webhook route.

        Args:
            name: Route name to remove.

        Returns:
            JSON string with ok status.
        """
        try:
            with self._state_lock:
                if name not in self._routes:
                    return json.dumps({"ok": False, "error": f"No route named {name!r}."})
                routes = dict(self._routes)
                del routes[name]
                _save_routes(routes)
                self._routes = routes
            return json.dumps({"ok": True, "message": f"Route {name!r} removed."})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_webhook_routes(self) -> str:
        """List the configured inbound webhook routes (secrets redacted).

        Returns:
            JSON string with ok status and a ``routes`` list.
        """
        try:
            with self._state_lock:
                routes = [
                    {
                        "name": name,
                        "path": f"/hook/{name}",
                        "kind": route.get("kind", "generic"),
                        "prompt_template": route.get("prompt_template", "{payload}"),
                        "filters": route.get("filters") or {},
                        "deliver_module": route.get("deliver_module", ""),
                        "deliver_channel": route.get("deliver_channel", ""),
                    }
                    for name, route in sorted(self._routes.items())
                ]
            return json.dumps({"ok": True, "routes": routes})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class WebhookAgent(BaseChannelAgent):
    """Channel agent with inbound webhook route tools."""

    channel_system_prompt = (
        "You are managing inbound webhooks. External systems POST signed "
        "JSON events to /hook/<route-name> on this machine; each accepted "
        "event becomes a message whose text is the route's rendered prompt "
        "template. Use add_webhook_route / list_webhook_routes / "
        "remove_webhook_route to manage routes. Webhooks are inbound-only: "
        "replies are delivered only through a route's configured delivery "
        "channel module."
    )

    def __init__(self) -> None:
        super().__init__("Webhook Agent")
        self._backend = WebhookChannelBackend()
        cfg = _config.load()
        if cfg:
            self._backend._port = cfg["port"]
            self._backend._routes = _parse_routes(cfg)

    def _is_authenticated(self) -> bool:
        """Return True if the backend is configured."""
        return bool(self._backend._port)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_webhook_auth() -> str:
            """Check if the webhook listener is configured.

            Returns:
                Configuration status or instructions.
            """
            if not agent._backend._port:
                return (
                    "Not configured for webhooks. Use authenticate_webhook() to "
                    "configure the listener port (default 18090), then add "
                    "routes with add_webhook_route()."
                )
            return json.dumps(
                {
                    "ok": True,
                    "port": agent._backend._port,
                    "routes": sorted(agent._backend._routes),
                }
            )

        def authenticate_webhook(port: str = "18090") -> str:
            """Configure the inbound webhook listener.

            Args:
                port: TCP port for the webhook HTTP server (default 18090).

            Returns:
                Configuration result or error message.
            """
            port = port.strip()
            if not port.isdigit() or int(port) > 65535:
                return "port must be a number between 0 and 65535."
            cfg = _config.load()
            routes = cfg.get("routes", "{}") if cfg else "{}"
            _config.save({"port": port, "routes": routes})
            agent._backend._port = port
            agent._backend._routes = _parse_routes({"port": port, "routes": routes})
            return json.dumps(
                {"ok": True, "message": f"Webhook listener configured on port {port}."}
            )

        def clear_webhook_auth() -> str:
            """Clear the stored webhook configuration and routes.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._port = ""
            agent._backend._routes = {}
            return "Webhook configuration cleared."

        return [check_webhook_auth, authenticate_webhook, clear_webhook_auth]


def _make_backend() -> WebhookChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = WebhookChannelBackend()
    cfg = _config.load()
    if not cfg:
        print("Not configured. Run: kiss-webhook -t 'authenticate'")
        sys.exit(1)
    backend._port = cfg["port"]
    backend._routes = _parse_routes(cfg)
    return backend


def main() -> None:
    """Run the WebhookAgent from the command line with chat persistence."""
    channel_main(
        WebhookAgent,
        "kiss-webhook",
        channel_name="Webhook",
        make_backend=_make_backend,
    )


def get_tools() -> list:
    """Return the webhook channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return WebhookAgent()._get_tools()


if __name__ == "__main__":
    main()
