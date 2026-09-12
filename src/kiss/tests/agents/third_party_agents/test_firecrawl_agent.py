# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Firecrawl channel agent.

Runs a REAL local HTTP server (stdlib ``ThreadedHTTPServer``) emulating
the Firecrawl v2 REST API endpoints — no mocks, patches, or fakes.
The server asserts the ``Authorization: Bearer`` header on every call,
returns canned JSON, and records requests (method, path, body) for
verification.

Config state is isolated per pytest process because the session
conftest points ``KISS_HOME`` at a temporary directory and
``ChannelConfig.path`` resolves ``$KISS_HOME`` lazily.
"""

from __future__ import annotations

import json
import shutil
import stat
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler
from typing import Any, cast
from urllib.parse import urlparse

import pytest

import kiss.agents.third_party_agents.firecrawl_agent as fc_mod
from kiss.agents.third_party_agents._backend_utils import (
    ThreadedHTTPServer,
    stop_http_server,
)
from kiss.agents.third_party_agents.firecrawl_agent import (
    _DEFAULT_BASE_URL,
    FirecrawlAgent,
    FirecrawlChannelBackend,
    _config,
)

_API_KEY = "fc-test-key"
_CRAWL_ID = "c0ffee00-1234-5678-9abc-def012345678"

_SCRAPE_DATA = {
    "markdown": "# Example Domain\n\nThis domain is for use in examples.",
    "links": ["https://www.iana.org/domains/example"],
    "metadata": {
        "title": "Example Domain",
        "statusCode": 200,
        "sourceURL": "https://example.com",
    },
}
_HTML_DATA = {
    "html": "<h1>Example</h1>",
    "rawHtml": "<html><body><h1>Example</h1></body></html>",
    "summary": "A short summary.",
    "metadata": {"title": "HTML Page", "statusCode": 200, "sourceURL": "https://htmlpage.example"},
}
_MAP_LINKS = [
    {"url": "https://example.com/", "title": "Home", "description": "The home page"},
    {"url": "https://example.com/blog", "title": "Blog"},
    "https://example.com/raw-string-entry",  # non-dict entries are skipped
]
_SEARCH_WEB = [
    {
        "url": "https://firecrawl.dev",
        "title": "Firecrawl",
        "description": "Web data API",
        "position": 1,
    },
    "raw-string-result",  # non-dict entries are skipped
]
_CRAWL_STATUS = {
    "success": True,
    "status": "completed",
    "completed": 2,
    "total": 2,
    "creditsUsed": 2,
    "data": [
        {
            "markdown": "page one content",
            "metadata": {"sourceURL": "https://example.com/1", "title": "Page 1"},
        },
        {"markdown": "page two, no metadata"},
        "raw-string-page",  # non-dict entries are skipped
    ],
}
# A crawl whose status response exceeded Firecrawl's 10MB page limit and
# therefore carries a "next" URL pointing at the following data page.
_CRAWL_ID_PAGED = "9a9a9a9a-1111-2222-3333-444455556666"
_CRAWL_NEXT_URL = f"https://api.firecrawl.dev/v2/crawl/{_CRAWL_ID_PAGED}?skip=1"
_CRAWL_STATUS_PAGED = {
    "success": True,
    "status": "scraping",
    "completed": 1,
    "total": 5,
    "creditsUsed": 1,
    "next": _CRAWL_NEXT_URL,
    "data": [
        {
            "markdown": "first paged page",
            "metadata": {"sourceURL": "https://example.com/p1", "title": "Paged 1"},
        }
    ],
}


class _FirecrawlRequestHandler(BaseHTTPRequestHandler):
    """Emulates the Firecrawl v2 REST API and records requests."""

    def _authorized(self) -> bool:
        return self.headers.get("Authorization") == f"Bearer {_API_KEY}"

    def _reply(self, status: int, body: str, content_type: str = "application/json") -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _record(self, body: dict[str, Any] | None) -> None:
        cast(_FirecrawlServer, self.server).requests.append(
            {
                "method": self.command,
                "path": self.path,
                "authorization": self.headers.get("Authorization", ""),
                "body": body,
            }
        )

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            body: dict[str, Any] | None = json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            body = None
        self._record(body)
        if not self._authorized():
            self._reply(401, json.dumps({"error": "Unauthorized: Invalid token"}))
            return
        path = urlparse(self.path).path
        url = (body or {}).get("url", "")
        if path == "/v2/scrape":
            if url == "https://fail.example":
                self._reply(500, json.dumps({"error": "An unexpected error occurred."}))
            elif url == "https://apifalse.example":
                self._reply(200, json.dumps({"success": False, "error": "denied by policy"}))
            elif url == "https://notjson.example":
                self._reply(200, "this is not json", content_type="text/plain")
            elif url == "https://array.example":
                self._reply(200, json.dumps([1, 2, 3]))
            elif url == "https://empty.example":
                self._reply(200, json.dumps({"success": True}))
            elif url == "https://htmlpage.example":
                self._reply(200, json.dumps({"success": True, "data": _HTML_DATA}))
            else:
                self._reply(200, json.dumps({"success": True, "data": _SCRAPE_DATA}))
        elif path == "/v2/map":
            self._reply(200, json.dumps({"success": True, "links": _MAP_LINKS}))
        elif path == "/v2/search":
            self._reply(200, json.dumps({"success": True, "data": {"web": _SEARCH_WEB}}))
        elif path == "/v2/crawl":
            self._reply(
                200,
                json.dumps({"success": True, "id": _CRAWL_ID, "url": f"/v2/crawl/{_CRAWL_ID}"}),
            )
        else:
            self._reply(404, json.dumps({"error": "Not found"}))

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        self._record(None)
        if not self._authorized():
            self._reply(401, json.dumps({"error": "Unauthorized: Invalid token"}))
            return
        path = urlparse(self.path).path
        if path == f"/v2/crawl/{_CRAWL_ID}":
            self._reply(200, json.dumps(_CRAWL_STATUS))
        elif path == f"/v2/crawl/{_CRAWL_ID_PAGED}":
            self._reply(200, json.dumps(_CRAWL_STATUS_PAGED))
        else:
            self._reply(404, json.dumps({"error": "Crawl job not found."}))

    def do_DELETE(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        self._record(None)
        if not self._authorized():
            self._reply(401, json.dumps({"error": "Unauthorized: Invalid token"}))
            return
        if urlparse(self.path).path == f"/v2/crawl/{_CRAWL_ID}":
            self._reply(200, json.dumps({"status": "cancelled"}))
        else:
            self._reply(404, json.dumps({"error": "Crawl job not found."}))

    def log_message(self, *args: Any) -> None:  # type: ignore[override]
        pass


class _FirecrawlServer(ThreadedHTTPServer):
    """ThreadedHTTPServer that records every request it receives."""

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, _FirecrawlRequestHandler)
        self.requests: list[dict[str, Any]] = []


@pytest.fixture()
def fc_server():
    """Start the emulated Firecrawl server on a free port; yield (base_url, server)."""
    server = _FirecrawlServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base_url, server
    finally:
        stop_http_server(server, thread)


@pytest.fixture()
def backend(fc_server):
    """A backend pointed at the emulated server with the valid API key."""
    base_url, server = fc_server
    b = FirecrawlChannelBackend()
    b._base_url = base_url
    b._api_key = _API_KEY
    return b, server


@pytest.fixture(autouse=True)
def _fresh_config():
    """Start and end every test with no persisted Firecrawl config."""
    _config.clear()
    yield
    _config.clear()


def test_agent_instantiation_unauthenticated() -> None:
    """A fresh agent is unauthenticated and exposes only the auth trio."""
    agent = FirecrawlAgent()
    assert agent.name == "Firecrawl Agent"
    assert agent._is_authenticated() is False
    names = [t.__name__ for t in agent._get_tools()]
    assert names == [
        "check_firecrawl_auth",
        "authenticate_firecrawl",
        "clear_firecrawl_auth",
    ]


def test_check_auth_unauthenticated_message() -> None:
    """check_firecrawl_auth explains how to configure when unauthenticated."""
    agent = FirecrawlAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    msg = tools["check_firecrawl_auth"]()
    assert "authenticate_firecrawl" in msg
    assert "https://www.firecrawl.dev" in msg


def test_authenticate_persists_config_and_exposes_tools() -> None:
    """authenticate_firecrawl persists config (0600) and unlocks backend tools."""
    agent = FirecrawlAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    result = json.loads(tools["authenticate_firecrawl"](_API_KEY))
    assert result["ok"] is True

    assert _config.path.exists()
    if sys.platform != "win32":
        assert stat.S_IMODE(_config.path.stat().st_mode) == 0o600
    saved = json.loads(_config.path.read_text(encoding="utf-8"))
    assert saved == {"api_key": _API_KEY}

    assert agent._is_authenticated() is True
    assert agent._backend._base_url == _DEFAULT_BASE_URL
    checked = json.loads(tools["check_firecrawl_auth"]())
    assert checked == {"ok": True, "base_url": _DEFAULT_BASE_URL}

    names = {t.__name__ for t in agent._get_tools()}
    assert {
        "firecrawl_scrape",
        "firecrawl_map",
        "firecrawl_search",
        "firecrawl_start_crawl",
        "firecrawl_get_crawl_status",
        "firecrawl_cancel_crawl",
    } <= names
    assert "connect" not in names  # channel protocol method, not an LLM tool

    cleared = tools["clear_firecrawl_auth"]()
    assert "cleared" in cleared.lower()
    assert not _config.path.exists()
    assert agent._is_authenticated() is False
    assert agent._backend._base_url == _DEFAULT_BASE_URL
    assert len(agent._get_tools()) == 3


def test_authenticate_stores_optional_base_url() -> None:
    """A self-hosted base_url is persisted and applied to the backend."""
    agent = FirecrawlAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    result = json.loads(
        tools["authenticate_firecrawl"](_API_KEY, "http://firecrawl.internal:3002")
    )
    assert result["ok"] is True
    saved = json.loads(_config.path.read_text(encoding="utf-8"))
    assert saved == {"api_key": _API_KEY, "base_url": "http://firecrawl.internal:3002"}
    assert agent._backend._base_url == "http://firecrawl.internal:3002"


def test_authenticate_persistence_failure_returns_ok_false() -> None:
    """authenticate_firecrawl returns ok:false and never raises when saving fails.

    The failure is forced end to end: a regular FILE occupies the path
    where the config DIRECTORY must be created, so ``_config.save``'s
    mkdir fails.  The agent must stay unauthenticated (key and base_url
    untouched) with the backend tools still locked.
    """
    agent = FirecrawlAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    config_dir = _config.path.parent
    if config_dir.is_dir():
        shutil.rmtree(config_dir)
    config_dir.parent.mkdir(parents=True, exist_ok=True)
    config_dir.write_text("occupies the config directory path", encoding="utf-8")
    try:
        result = json.loads(
            tools["authenticate_firecrawl"](_API_KEY, "http://firecrawl.internal:3002")
        )
        assert result["ok"] is False
        assert "could not save config" in result["error"]
        assert agent._is_authenticated() is False
        assert agent._backend._api_key == ""
        assert agent._backend._base_url == _DEFAULT_BASE_URL
        assert len(agent._get_tools()) == 3  # backend tools stay locked
    finally:
        config_dir.unlink()


def test_authenticate_rejects_empty_api_key() -> None:
    """authenticate_firecrawl refuses an empty or whitespace api_key."""
    agent = FirecrawlAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    assert "cannot be empty" in tools["authenticate_firecrawl"]("")
    assert "cannot be empty" in tools["authenticate_firecrawl"]("   ")
    assert not _config.path.exists()


def test_new_agent_loads_persisted_config() -> None:
    """A new agent picks up previously persisted credentials."""
    _config.save({"api_key": _API_KEY})
    agent = FirecrawlAgent()
    assert agent._is_authenticated() is True
    assert agent._backend._api_key == _API_KEY
    assert agent._backend._base_url == _DEFAULT_BASE_URL


def test_new_agent_loads_persisted_base_url() -> None:
    """A new agent picks up a persisted self-hosted base_url."""
    _config.save({"api_key": _API_KEY, "base_url": "http://firecrawl.internal:3002"})
    agent = FirecrawlAgent()
    assert agent._backend._base_url == "http://firecrawl.internal:3002"


def test_tools_module_function() -> None:
    """Module-level tools() returns a non-empty tool list."""
    tools = fc_mod.tools()
    assert len(tools) >= 3
    assert all(callable(t) for t in tools)


def test_connect_without_config_fails() -> None:
    """connect() fails cleanly when no config is persisted."""
    b = FirecrawlChannelBackend()
    assert b.connect() is False
    assert "No Firecrawl config" in b.connection_info


def test_connect_with_config_succeeds() -> None:
    """connect() loads persisted config; missing base_url falls back to cloud."""
    _config.save({"api_key": _API_KEY})
    b = FirecrawlChannelBackend()
    assert b.connect() is True
    assert b._api_key == _API_KEY
    assert b._base_url == _DEFAULT_BASE_URL
    assert _DEFAULT_BASE_URL in b.connection_info

    _config.save({"api_key": _API_KEY, "base_url": "http://firecrawl.internal:3002"})
    b2 = FirecrawlChannelBackend()
    assert b2.connect() is True
    assert b2._base_url == "http://firecrawl.internal:3002"


def test_scrape_happy_path(backend) -> None:
    """firecrawl_scrape POSTs /v2/scrape and condenses markdown + metadata."""
    b, server = backend
    result = json.loads(b.firecrawl_scrape("https://example.com"))
    assert result["ok"] is True
    assert result["title"] == "Example Domain"
    assert result["status_code"] == 200
    assert result["url"] == "https://example.com"
    assert result["markdown"].startswith("# Example Domain")
    assert result["links"] == ["https://www.iana.org/domains/example"]
    assert "html" not in result  # absent formats are not echoed back
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("POST", "/v2/scrape")
    assert req["authorization"] == f"Bearer {_API_KEY}"
    assert req["body"] == {
        "url": "https://example.com",
        "formats": ["markdown"],
        "onlyMainContent": True,
    }


def test_scrape_multiple_formats(backend) -> None:
    """Comma-separated formats are split and sent as a list; all condensed."""
    b, server = backend
    result = json.loads(
        b.firecrawl_scrape("https://htmlpage.example", "html, rawHtml,summary", False)
    )
    assert result["ok"] is True
    assert result["html"] == "<h1>Example</h1>"
    assert result["rawHtml"] == "<html><body><h1>Example</h1></body></html>"
    assert result["summary"] == "A short summary."
    assert "markdown" not in result
    assert server.requests[-1]["body"] == {
        "url": "https://htmlpage.example",
        "formats": ["html", "rawHtml", "summary"],
        "onlyMainContent": False,
    }


def test_scrape_empty_data_payload(backend) -> None:
    """A success reply with no data object still yields ok with defaults."""
    b, _ = backend
    result = json.loads(b.firecrawl_scrape("https://empty.example"))
    assert result == {"ok": True, "title": "", "status_code": 0, "url": "https://empty.example"}


def test_scrape_input_validation(backend) -> None:
    """Empty url or an all-empty formats list is refused without a request."""
    b, server = backend
    assert json.loads(b.firecrawl_scrape("  "))["ok"] is False
    assert json.loads(b.firecrawl_scrape("https://example.com", " , ,"))["ok"] is False
    assert server.requests == []


def test_map_happy_path(backend) -> None:
    """firecrawl_map POSTs /v2/map and returns condensed dict links only."""
    b, server = backend
    result = json.loads(b.firecrawl_map("https://example.com", search="blog", limit=50))
    assert result["ok"] is True
    assert result["count"] == 2  # the raw string entry is skipped
    assert result["links"] == [
        {"url": "https://example.com/", "title": "Home", "description": "The home page"},
        {"url": "https://example.com/blog", "title": "Blog"},
    ]
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("POST", "/v2/map")
    assert req["authorization"] == f"Bearer {_API_KEY}"
    assert req["body"] == {"url": "https://example.com", "limit": 50, "search": "blog"}


def test_map_omits_empty_search(backend) -> None:
    """An empty search is not sent in the request body."""
    b, server = backend
    assert json.loads(b.firecrawl_map("https://example.com"))["ok"] is True
    assert server.requests[-1]["body"] == {"url": "https://example.com", "limit": 100}


def test_map_input_validation(backend) -> None:
    """Empty url or non-positive limit is refused without a request."""
    b, server = backend
    assert json.loads(b.firecrawl_map(""))["ok"] is False
    assert json.loads(b.firecrawl_map("https://example.com", limit=0))["ok"] is False
    assert server.requests == []


def test_search_happy_path(backend) -> None:
    """firecrawl_search POSTs /v2/search with typed sources; condenses results."""
    b, server = backend
    result = json.loads(b.firecrawl_search("firecrawl docs", limit=3))
    assert result["ok"] is True
    assert result["web"] == [
        {"url": "https://firecrawl.dev", "title": "Firecrawl", "description": "Web data API"}
    ]
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("POST", "/v2/search")
    assert req["authorization"] == f"Bearer {_API_KEY}"
    assert req["body"] == {
        "query": "firecrawl docs",
        "limit": 3,
        "sources": [{"type": "web"}],
    }


def test_search_multiple_sources(backend) -> None:
    """Multiple sources are all requested; missing ones come back empty."""
    b, server = backend
    result = json.loads(b.firecrawl_search("news query", sources="web, news"))
    assert result["ok"] is True
    assert len(result["web"]) == 1
    assert result["news"] == []  # emulator returns no news array
    assert server.requests[-1]["body"]["sources"] == [{"type": "web"}, {"type": "news"}]


def test_search_input_validation(backend) -> None:
    """Empty query/sources, bad source names, and bad limits are refused."""
    b, server = backend
    assert json.loads(b.firecrawl_search(" "))["ok"] is False
    assert json.loads(b.firecrawl_search("q", limit=0))["ok"] is False
    assert json.loads(b.firecrawl_search("q", sources=" , "))["ok"] is False
    bad = json.loads(b.firecrawl_search("q", sources="web,videos"))
    assert bad["ok"] is False
    assert "videos" in bad["error"]
    assert server.requests == []


def test_start_crawl_happy_path(backend) -> None:
    """firecrawl_start_crawl POSTs /v2/crawl and returns the crawl id."""
    b, server = backend
    result = json.loads(b.firecrawl_start_crawl("https://example.com", limit=25))
    assert result == {"ok": True, "crawl_id": _CRAWL_ID}
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("POST", "/v2/crawl")
    assert req["authorization"] == f"Bearer {_API_KEY}"
    assert req["body"] == {"url": "https://example.com", "limit": 25}


def test_start_crawl_input_validation(backend) -> None:
    """Empty url or non-positive limit is refused without a request."""
    b, server = backend
    assert json.loads(b.firecrawl_start_crawl(""))["ok"] is False
    assert json.loads(b.firecrawl_start_crawl("https://example.com", limit=0))["ok"] is False
    assert server.requests == []


def test_get_crawl_status_happy_path(backend) -> None:
    """firecrawl_get_crawl_status GETs /v2/crawl/{id} and condenses pages."""
    b, server = backend
    result = json.loads(b.firecrawl_get_crawl_status(_CRAWL_ID))
    assert result["ok"] is True
    assert result["status"] == "completed"
    assert result["completed"] == 2
    assert result["total"] == 2
    assert result["credits_used"] == 2
    assert result["pages"] == [
        {"url": "https://example.com/1", "title": "Page 1", "markdown": "page one content"},
        {"url": "", "title": "", "markdown": "page two, no metadata"},
    ]  # the raw string page entry is skipped
    assert "next" not in result  # no pagination URL in the response -> key omitted
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("GET", f"/v2/crawl/{_CRAWL_ID}")
    assert req["authorization"] == f"Bearer {_API_KEY}"


def test_get_crawl_status_surfaces_next_url(backend) -> None:
    """A crawl-status response with a 'next' pagination URL surfaces it."""
    b, server = backend
    result = json.loads(b.firecrawl_get_crawl_status(_CRAWL_ID_PAGED))
    assert result["ok"] is True
    assert result["status"] == "scraping"
    assert result["completed"] == 1
    assert result["total"] == 5
    assert result["next"] == _CRAWL_NEXT_URL
    assert result["pages"] == [
        {"url": "https://example.com/p1", "title": "Paged 1", "markdown": "first paged page"}
    ]
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("GET", f"/v2/crawl/{_CRAWL_ID_PAGED}")


def test_cancel_crawl_happy_path(backend) -> None:
    """firecrawl_cancel_crawl DELETEs /v2/crawl/{id}."""
    b, server = backend
    result = json.loads(b.firecrawl_cancel_crawl(_CRAWL_ID))
    assert result == {"ok": True, "status": "cancelled"}
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("DELETE", f"/v2/crawl/{_CRAWL_ID}")
    assert req["authorization"] == f"Bearer {_API_KEY}"


def test_crawl_id_path_traversal_rejected(backend) -> None:
    """Unsafe crawl ids are refused up front — no HTTP request is made."""
    b, server = backend
    for attempt in (
        b.firecrawl_get_crawl_status(""),
        b.firecrawl_get_crawl_status("../scrape"),
        b.firecrawl_get_crawl_status("a/b"),
        b.firecrawl_get_crawl_status("a\\b"),
        b.firecrawl_cancel_crawl(""),
        b.firecrawl_cancel_crawl("../scrape"),
        b.firecrawl_cancel_crawl("a/b"),
        b.firecrawl_cancel_crawl("a\\b"),
    ):
        result = json.loads(attempt)
        assert result["ok"] is False
        assert "invalid crawl_id" in result["error"]
    assert server.requests == []  # nothing ever reached the server


def test_crawl_id_is_encoded_as_single_path_segment(backend) -> None:
    """Special characters in crawl_id are percent-encoded, never path-interpreted."""
    b, server = backend
    result = json.loads(b.firecrawl_get_crawl_status("a b%c"))
    assert result["ok"] is False  # emulator 404s the unknown id — that's fine
    assert server.requests[-1]["path"] == "/v2/crawl/a%20b%25c"


def test_api_level_error_returns_ok_false(backend) -> None:
    """A 200 reply with success:false yields ok:false with the API error."""
    b, _ = backend
    result = json.loads(b.firecrawl_scrape("https://apifalse.example"))
    assert result["ok"] is False
    assert "denied by policy" in result["error"]


def test_non_json_response_returns_ok_false(backend) -> None:
    """A non-JSON response body yields ok:false, not an exception."""
    b, _ = backend
    result = json.loads(b.firecrawl_scrape("https://notjson.example"))
    assert result["ok"] is False
    assert "non-JSON" in result["error"]


def test_non_object_json_response_returns_ok_false(backend) -> None:
    """A JSON-array response body yields ok:false, not an exception."""
    b, _ = backend
    result = json.loads(b.firecrawl_scrape("https://array.example"))
    assert result["ok"] is False
    assert "unexpected response shape" in result["error"]


def test_unauthorized_key_returns_ok_false(fc_server) -> None:
    """A 401 from the server yields ok:false JSON from every tool — no exception."""
    base_url, _ = fc_server
    b = FirecrawlChannelBackend()
    b._base_url = base_url
    b._api_key = "wrong-key"
    for call in (
        lambda: b.firecrawl_scrape("https://example.com"),
        lambda: b.firecrawl_map("https://example.com"),
        lambda: b.firecrawl_search("query"),
        lambda: b.firecrawl_start_crawl("https://example.com"),
        lambda: b.firecrawl_get_crawl_status(_CRAWL_ID),
        lambda: b.firecrawl_cancel_crawl(_CRAWL_ID),
    ):
        result = json.loads(call())
        assert result["ok"] is False
        assert "401" in result["error"]


def test_server_error_returns_ok_false(backend) -> None:
    """A 500 from the server yields ok:false JSON, not an exception."""
    b, _ = backend
    result = json.loads(b.firecrawl_scrape("https://fail.example"))
    assert result["ok"] is False
    assert "500" in result["error"]


def test_main_prints_usage_without_args() -> None:
    """Running the module CLI with no arguments prints usage and exits 1.

    Executed as a REAL subprocess (no argv patching), so the
    ``channel_main(...)`` call inside ``main()`` is exercised end to
    end.  In-process branch coverage cannot count this line because
    the coverage tracer does not follow the child process.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "kiss.agents.third_party_agents.firecrawl_agent"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1
    assert "Usage: kiss-firecrawl" in proc.stdout


def test_connection_refused_returns_ok_false() -> None:
    """Tools return ok:false when the server is unreachable — never raise."""
    b = FirecrawlChannelBackend()
    b._base_url = "http://127.0.0.1:9"  # discard port; nothing listens
    b._api_key = _API_KEY
    result = json.loads(b.firecrawl_scrape("https://example.com"))
    assert result["ok"] is False
    assert result["error"]
