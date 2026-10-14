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
Google token, unless a test installs a canned ``proxy_reply``.  Non-JSON,
non-text answers come back as ``binary_data`` download links served by
this same server, like Composio's file URLs.
"""

from __future__ import annotations

import base64
import json
import os
import threading
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

import requests

from kiss.agents.third_party_agents import _composio_google as composio_google
from kiss.agents.third_party_agents._backend_utils import ThreadedHTTPServer
from kiss.tests.agents.third_party_agents.recording_http import serving

API_KEY = "test-composio-key"
TOKEN = "composio-injected-token"
API = "/api/v3.1"


class FakeComposioServer(ThreadedHTTPServer):
    """Composio API emulator with inspectable state.

    Attributes:
        requests: Every request received, as ``{"method", "path", "query",
            "body"}`` (``query`` is ``parse_qs`` of the URL query).
        api_keys: ``x-api-key`` values accepted; any other key gets 401.
        auth_configs: Existing auth configs, listed verbatim by the API.
        accounts: Connected-account ID -> current status.
        status_queue: Connected-account ID -> statuses answered by its
            next status GETs, one per poll, before ``accounts`` applies
            (see :meth:`set_statuses`).
        undeletable: Connected-account IDs whose DELETE is refused (404).
        deleted: Connected-account IDs deleted through the API.
        next_account: When set, the ID the next Connect Link creates
            (otherwise ``ca_<n>``); a link for an existing ID keeps its status.
        token: Bearer token the proxy injects upstream.
        proxy_reply: When set, the proxy answers with it verbatim instead
            of forwarding to the endpoint.
        downloads: Binary proxy answers ``(content_type, data)``, served
            at ``/download/<index>`` (404 for an unknown index).
        upstream_overrides: Origin -> replacement origin applied to proxied
            endpoints, so clients hard-wired to Google hosts (googleapiclient
            services) reach a local emulator instead.
        status_polled: Per API key, set once that caller has polled a
            connected-account status (the ``finish_connect`` using the key
            has read its ``pending_id`` by then).  Concurrent callers are
            told apart by the key ``_client()`` reads from
            ``COMPOSIO_API_KEY`` when they start.
        status_holds: Per API key, an Event a status GET waits for before
            it is answered (503 if the wait expires): lets a test hold one
            caller's poll until another caller has finished.
        status_held: Per API key, set once such a poll is waiting.
    """

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.requests: list[dict[str, Any]] = []
        self.api_keys: set[str] = {API_KEY}
        self.auth_configs: list[dict[str, Any]] = []
        self.accounts: dict[str, str] = {}
        self.status_queue: dict[str, list[str]] = {}
        self.undeletable: set[str] = set()
        self.deleted: list[str] = []
        self.next_account: str | None = None
        self.token = TOKEN
        self.proxy_reply: dict[str, Any] | None = None
        self.downloads: list[tuple[str, bytes]] = []
        self.upstream_overrides: dict[str, str] = {}
        self.status_polled: defaultdict[str, threading.Event] = defaultdict(threading.Event)
        self.status_holds: dict[str, threading.Event] = {}
        self.status_held: defaultdict[str, threading.Event] = defaultdict(threading.Event)
        self._counter = 0

    @property
    def base_url(self) -> str:
        """Return the server's base URL."""
        return f"http://127.0.0.1:{self.server_address[1]}"

    def next_id(self, prefix: str) -> str:
        """Return a fresh object ID with *prefix*."""
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def paths(self, method: str) -> list[str]:
        """Return the paths of every recorded request made with *method*."""
        return [r["path"] for r in self.requests if r["method"] == method]

    def set_statuses(self, account: str, *statuses: str) -> None:
        """Answer *account*'s next status polls with *statuses* in turn; the last one sticks.

        Args:
            account: Connected-account ID.
            statuses: One status per poll, e.g. ``"INITIATED", "ACTIVE"``.
        """
        *queue, final = statuses
        self.accounts[account] = final
        self.status_queue[account] = queue


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

    def _route(self) -> None:
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body: Any = json.loads(raw) if raw else None
        server = self.server
        parts = urlsplit(self.path)
        path = parts.path
        server.requests.append(
            {"method": self.command, "path": path, "query": parse_qs(parts.query), "body": body}
        )
        if path.startswith("/download/"):
            self._download(int(path.rsplit("/", 1)[1]))
            return
        api_key = self.headers.get("x-api-key", "")
        if api_key not in server.api_keys:
            self._reply(401, {"error": {"message": "invalid api key"}})
            return
        if path == f"{API}/auth_configs" and self.command == "GET":
            items = server.auth_configs
            self._reply(
                200,
                {"items": items, "current_page": 1, "total_items": len(items), "total_pages": 1},
            )
        elif path == f"{API}/auth_configs" and self.command == "POST":
            new = {"id": server.next_id("ac"), "created_at": "2026-09-26T00:00:00Z"}
            server.auth_configs.append(new)
            self._reply(
                201,
                {
                    "auth_config": {
                        "id": new["id"],
                        "auth_scheme": "OAUTH2",
                        "is_composio_managed": True,
                    },
                    "toolkit": {"slug": body["toolkit"]["slug"]},
                },
            )
        elif path == f"{API}/connected_accounts" and self.command == "GET":
            self._reply(200, {"items": [], "current_page": 1, "total_items": 0, "total_pages": 1})
        elif path == f"{API}/connected_accounts/link" and self.command == "POST":
            account = server.next_account or server.next_id("ca")
            server.accounts.setdefault(account, "INITIATED")
            self._reply(
                201,
                {
                    "connected_account_id": account,
                    "link_token": "lt",
                    "redirect_url": f"https://connect.composio.dev/link/{account}",
                    "expires_at": "2099-01-01T00:00:00Z",
                },
            )
        elif path.startswith(f"{API}/connected_accounts/"):
            self._account(path.rsplit("/", 1)[1], api_key)
        elif path == f"{API}/tools/execute/proxy" and self.command == "POST":
            if server.proxy_reply is not None:
                self._reply(200, server.proxy_reply)
            else:
                self._proxy(body)
        else:
            self._reply(404, {"error": {"message": f"no route {self.command} {path}"}})

    def _download(self, index: int) -> None:
        if index >= len(self.server.downloads):
            self._reply(404, {"error": {"message": "no such download"}})
            return
        content_type, data = self.server.downloads[index]
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _account(self, account: str, api_key: str) -> None:
        server = self.server
        if account not in server.accounts:
            self._reply(404, {"error": {"message": "connected account not found", "status": 404}})
        elif self.command == "DELETE":
            if account in server.undeletable:
                self._reply(404, {"error": {"message": "not found", "status": 404}})
                return
            del server.accounts[account]
            server.status_queue.pop(account, None)
            server.deleted.append(account)
            self._reply(200, {"success": True})
        else:
            hold = server.status_holds.get(api_key)
            if hold is not None:
                server.status_held[api_key].set()
                if not hold.wait(timeout=15.0):
                    # An expired hold is a test failure, never a normal answer.
                    self._reply(503, {"error": {"message": "status poll hold expired"}})
                    return
            queue = server.status_queue.get(account)
            status = queue.pop(0) if queue else server.accounts[account]
            server.status_polled[api_key].set()
            self._reply(200, {"id": account, "status": status})

    def _proxy(self, body: dict[str, Any]) -> None:
        server = self.server
        if server.accounts.get(body.get("connected_account_id", "")) != "ACTIVE":
            self._reply(400, {"error": {"message": "connected account is not active"}})
            return
        # Like Composio: a repeated query name in ``parameters`` keeps only
        # its last value (the endpoint URL's own query string is kept whole).
        params = {
            p["name"]: p["value"] for p in body.get("parameters") or [] if p["type"] == "query"
        }
        headers = {
            p["name"]: p["value"] for p in body.get("parameters") or [] if p["type"] == "header"
        }
        headers["Authorization"] = f"Bearer {server.token}"
        endpoint = body["endpoint"]
        for origin, replacement in server.upstream_overrides.items():
            if endpoint.startswith(origin):
                endpoint = replacement + endpoint[len(origin) :]
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
        try:
            upstream = requests.request(
                body["method"],
                endpoint,
                params=params,
                headers=headers,
                data=data,
                timeout=30,
            )
        except requests.RequestException as e:
            self._reply(400, {"error": {"message": f"upstream request failed: {e}"}})
            return
        content_type = upstream.headers.get("Content-Type", "")
        answer: dict[str, Any] = {
            "status": upstream.status_code,
            "headers": {
                k: v
                for k, v in upstream.headers.items()
                if k.lower() not in ("content-length", "transfer-encoding")
            },
        }
        if "json" in content_type:
            answer["data"] = upstream.json() if upstream.content else None
        elif not upstream.content or content_type.startswith("text/"):
            answer["data"] = upstream.text or None
        else:
            server.downloads.append((content_type, upstream.content))
            answer["binary_data"] = {
                "url": f"{server.base_url}/download/{len(server.downloads) - 1}",
                "content_type": content_type,
                "size": len(upstream.content),
            }
        self._reply(200, answer)

    do_GET = do_POST = do_DELETE = do_PATCH = do_PUT = _route  # noqa: N815


def start_fake_composio(monkeypatch: Any) -> Iterator[FakeComposioServer]:
    """Run a :class:`FakeComposioServer` and point the Composio SDK at it.

    Use from a pytest fixture: ``yield from start_fake_composio(monkeypatch)``.
    A fixture that does more work around the server (connecting a service,
    building a backend) must use :func:`fake_composio` instead: a ``for``
    loop over this generator leaves shutdown to garbage collection when
    the loop body raises before the fixture's own ``yield``.

    A developer's shell may pin auth configs for any service
    (``KISS_COMPOSIO_AUTH_CONFIG_GOOGLE_CALENDAR=ac_...``) or the Composio
    user id; every such override is removed so the tests exercise the
    list/create path with the default user.

    Args:
        monkeypatch: pytest's ``monkeypatch`` fixture (sets env vars).

    Yields:
        The running server.
    """
    with serving(FakeComposioServer()) as server:
        monkeypatch.setenv("COMPOSIO_BASE_URL", server.base_url)
        monkeypatch.setenv("COMPOSIO_API_KEY", API_KEY)
        monkeypatch.delenv("KISS_COMPOSIO_USER_ID", raising=False)
        for name in [n for n in os.environ if n.startswith("KISS_COMPOSIO_AUTH_CONFIG_")]:
            monkeypatch.delenv(name)
        yield server


# :func:`start_fake_composio` as a context manager: ``with fake_composio(monkeypatch) as c:``.
fake_composio = contextmanager(start_fake_composio)


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
