# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A real local HTTP server speaking the subset of Composio's v3.1 API KISS uses.

The Composio SDK honors ``COMPOSIO_BASE_URL``, so pointing it at this
server exercises the real SDK end to end: auth-config lookup/creation,
Connect Link creation, connected-account status/deletion and the tool
proxy.  The proxy really forwards each request to its ``endpoint`` (a
local server emulating a Google API) and injects
``Authorization: Bearer <token>``, the way Composio injects the user's
Google token.  Non-JSON, non-text answers come back as ``binary_data``
download links served by this same server, like Composio's file URLs.
"""

from __future__ import annotations

import base64
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from typing import Any, cast

import requests

from kiss.agents.third_party_agents import _composio_google as composio_google
from kiss.agents.third_party_agents._backend_utils import ThreadedHTTPServer, stop_http_server

API_KEY = "test-composio-key"
TOKEN = "composio-injected-token"


class FakeComposioServer(ThreadedHTTPServer):
    """Composio API emulator with inspectable state.

    Attributes:
        auth_configs: Existing auth configs (``id``/``created_at``).
        accounts: Connected-account ID -> status.
        deleted: Connected-account IDs deleted through the API.
        requests_log: ``(method, path, json_body)`` of every API call.
        proxied: Every request the proxy forwarded (method, url, headers, body).
        link_status: Status new connected accounts start in.
        token: Bearer token the proxy injects upstream.
        downloads: Binary proxy answers, served at ``/download/<index>``.
        upstream_overrides: Origin -> replacement origin applied to proxied
            endpoints, so clients hard-wired to Google hosts (googleapiclient
            services) reach a local emulator instead.
    """

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.auth_configs: list[dict[str, str]] = []
        self.accounts: dict[str, str] = {}
        self.deleted: list[str] = []
        self.requests_log: list[tuple[str, str, Any]] = []
        self.proxied: list[dict[str, Any]] = []
        self.link_status = "INITIATED"
        self.token = TOKEN
        self.downloads: list[tuple[str, bytes]] = []
        self.upstream_overrides: dict[str, str] = {}
        self._counter = 0

    @property
    def base_url(self) -> str:
        """Return the server's base URL."""
        return f"http://127.0.0.1:{self.server_address[1]}"

    def next_id(self, prefix: str) -> str:
        """Return a fresh object ID with *prefix*."""
        self._counter += 1
        return f"{prefix}_{self._counter}"


class _Handler(BaseHTTPRequestHandler):
    """Routes the Composio API calls the SDK makes."""

    server: FakeComposioServer  # type: ignore[assignment]

    def log_message(self, *_args: Any) -> None:  # type: ignore[override]
        """Silence request logging."""

    def _reply(self, status: int, payload: Any) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> Any:
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        return json.loads(raw) if raw else None

    def _route(self) -> None:
        body = self._body()
        server = self.server
        path = self.path.split("?", 1)[0]
        if path.startswith("/download/"):
            content_type, data = server.downloads[int(path.rsplit("/", 1)[1])]
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        server.requests_log.append((self.command, self.path, body))
        if self.headers.get("x-api-key") != API_KEY:
            self._reply(401, {"error": {"message": "invalid api key"}})
            return
        if path == "/api/v3.1/auth_configs" and self.command == "GET":
            items = [{"id": c["id"], "created_at": c["created_at"]} for c in server.auth_configs]
            self._reply(200, {"items": items, "current_page": 1,
                              "total_items": len(items), "total_pages": 1})
        elif path == "/api/v3.1/auth_configs" and self.command == "POST":
            new = {"id": server.next_id("ac"), "created_at": "2026-09-26T00:00:00Z"}
            server.auth_configs.append(new)
            self._reply(201, {"auth_config": {"id": new["id"], "auth_scheme": "OAUTH2",
                                              "is_composio_managed": True},
                              "toolkit": {"slug": body["toolkit"]["slug"]}})
        elif path == "/api/v3.1/connected_accounts" and self.command == "GET":
            self._reply(200, {"items": [], "current_page": 1, "total_items": 0, "total_pages": 1})
        elif path == "/api/v3.1/connected_accounts/link" and self.command == "POST":
            account = server.next_id("ca")
            server.accounts[account] = server.link_status
            self._reply(201, {"connected_account_id": account, "link_token": "lt",
                              "redirect_url": f"https://connect.composio.dev/link/{account}",
                              "expires_at": "2099-01-01T00:00:00Z"})
        elif path.startswith("/api/v3.1/connected_accounts/"):
            self._account(path.rsplit("/", 1)[1])
        elif path == "/api/v3.1/tools/execute/proxy" and self.command == "POST":
            self._proxy(body)
        else:
            self._reply(404, {"error": {"message": f"no route {self.command} {path}"}})

    def _account(self, account: str) -> None:
        server = self.server
        if account not in server.accounts:
            self._reply(404, {"error": {"message": "connected account not found"}})
        elif self.command == "DELETE":
            del server.accounts[account]
            server.deleted.append(account)
            self._reply(200, {"success": True})
        else:
            self._reply(200, {"id": account, "status": server.accounts[account]})

    def _proxy(self, body: dict[str, Any]) -> None:
        server = self.server
        if server.accounts.get(body.get("connected_account_id", "")) != "ACTIVE":
            self._reply(400, {"error": {"message": "connected account is not active"}})
            return
        params = [(p["name"], p["value"]) for p in body.get("parameters") or []
                  if p["type"] == "query"]
        headers = {p["name"]: p["value"] for p in body.get("parameters") or []
                   if p["type"] == "header"}
        headers["Authorization"] = f"Bearer {server.token}"
        endpoint = body["endpoint"]
        for origin, replacement in server.upstream_overrides.items():
            if endpoint.startswith(origin):
                endpoint = replacement + endpoint[len(origin):]
        payload = body.get("body")
        data: bytes | None = None
        if payload is not None:
            headers.setdefault("Content-Type", "application/json")
            data = json.dumps(payload).encode()
        binary = body.get("binary_body")
        if binary is not None:
            # Like Composio: the base64 body is sent raw with its content type.
            data = base64.b64decode(binary["base64"])
            headers["Content-Type"] = binary.get("content_type") or "application/octet-stream"
        server.proxied.append({"method": body["method"], "url": endpoint,
                               "params": params, "headers": headers, "body": payload,
                               "binary_body": binary})
        try:
            upstream = requests.request(
                body["method"], endpoint, params=params, headers=headers, data=data, timeout=30,
            )
        except requests.RequestException as e:
            self._reply(400, {"error": {"message": f"upstream request failed: {e}"}})
            return
        content_type = upstream.headers.get("Content-Type", "")
        answer: dict[str, Any] = {"status": upstream.status_code, "headers": {
            k: v for k, v in upstream.headers.items()
            if k.lower() not in ("content-length", "transfer-encoding")
        }}
        if "json" in content_type:
            answer["data"] = upstream.json() if upstream.content else None
        elif not upstream.content or content_type.startswith("text/"):
            answer["data"] = upstream.text or None
        else:
            server.downloads.append((content_type, upstream.content))
            answer["binary_data"] = {
                "url": f"{server.base_url}/download/{len(server.downloads) - 1}",
                "content_type": content_type, "size": len(upstream.content),
            }
        self._reply(200, answer)

    do_GET = do_POST = do_DELETE = do_PATCH = do_PUT = _route  # noqa: N815


def start_fake_composio(monkeypatch: Any) -> Iterator[FakeComposioServer]:
    """Run a :class:`FakeComposioServer` and point the Composio SDK at it.

    Use from a pytest fixture: ``yield from start_fake_composio(monkeypatch)``.

    Args:
        monkeypatch: pytest's ``monkeypatch`` fixture (sets env vars).

    Yields:
        The running server.
    """
    server = FakeComposioServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("COMPOSIO_BASE_URL", server.base_url)
    monkeypatch.setenv("COMPOSIO_API_KEY", API_KEY)
    try:
        yield server
    finally:
        stop_http_server(server, thread)


def reset_state(service: str) -> None:
    """Forget *service*'s recorded Composio connection and any saved API key.

    Args:
        service: KISS service name.
    """
    composio_google._state_path(service).unlink(missing_ok=True)
    composio_google._api_key_path().unlink(missing_ok=True)


def connect(server: FakeComposioServer, service: str) -> str:
    """Connect *service* through the real start/finish flow.

    Args:
        server: The running fake Composio server.
        service: KISS service name.

    Returns:
        The new connected-account ID.
    """
    started = composio_google.start_connect(service, service)
    assert started["status"] == "consent_required", started
    account = str(started["verification_uri"]).rsplit("/", 1)[1]
    server.accounts[account] = "ACTIVE"
    finished = composio_google.finish_connect(service, service)
    assert finished["ok"] is True, finished
    return cast(str, composio_google.connected_account_id(service))
