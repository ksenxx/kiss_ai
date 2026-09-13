# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Brave Search channel agent.

Runs a REAL local HTTP server (stdlib ``ThreadedHTTPServer``) emulating
the Brave Search API endpoints — no mocks, patches, or fakes.  The
server asserts the ``X-Subscription-Token`` header on every call,
returns canned JSON, and records requests (method, path, parsed query
params, headers) for verification.

Config state is isolated per pytest process because the session
conftest points ``KISS_HOME`` at a temporary directory and
``ChannelConfig.path`` resolves ``$KISS_HOME`` lazily.

Branch-coverage note: ``main()`` and the ``__main__`` guard delegate
straight to ``channel_main`` (a CLI entry point that parses
``sys.argv`` and exits); exercising them requires a subprocess, not a
test double, and their behaviour is covered by the shared
``channel_main`` tests.  All other branches of the module are covered
below.
"""

from __future__ import annotations

import json
import shutil
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import pytest

import kiss.agents.third_party_agents.brave_search_agent as brave_mod
from kiss.agents.third_party_agents._backend_utils import (
    ThreadedHTTPServer,
    stop_http_server,
)
from kiss.agents.third_party_agents.brave_search_agent import (
    BraveSearchAgent,
    BraveSearchChannelBackend,
    _clamp,
    _config,
)

_API_KEY = "test-brave-key"

_WEB_RESPONSE = {
    "web": {
        "results": [
            {
                "title": "Python",
                "url": "https://python.org",
                "description": "The Python language",
                "age": "2 days ago",
                "profile": {"noise": "dropped"},
            }
        ]
    },
    "infobox": {
        "results": [
            {
                "title": "Python (language)",
                "long_desc": "A programming language",
                "url": "https://en.wikipedia.org/wiki/Python",
            }
        ]
    },
    "faq": {
        "results": [
            {
                "question": "What is Python?",
                "answer": "A language",
                "url": "https://faq.example",
            }
        ]
    },
}

_WEB_RESPONSE_PLAIN = {
    "web": {
        "results": [
            {"title": "Plain", "url": "https://plain.example", "description": "d"}
        ]
    }
}

_NEWS_RESPONSE = {
    "results": [
        {
            "title": "Climate summit opens",
            "url": "https://news.example/summit",
            "description": "World leaders meet",
            "age": "1 hour ago",
            "meta_url": {"hostname": "reuters.com"},
        }
    ]
}

_IMAGE_RESPONSE = {
    "results": [
        {
            "title": "Mountain",
            "url": "https://page.example/mountain",
            "source": "example.com",
            "properties": {"url": "https://img.example/mountain.jpg"},
            "thumbnail": {"src": "https://thumb.example/mountain.jpg"},
        }
    ]
}

_VIDEO_RESPONSE = {
    "results": [
        {
            "title": "Tutorial",
            "url": "https://video.example/tut",
            "description": "Learn things",
            "age": "3 days ago",
            "video": {"duration": "10:02", "creator": "Prof X"},
        }
    ]
}

# Big enough that the condensed JSON exceeds the 8000-char output cap.
_WEB_RESPONSE_HUGE = {
    "web": {
        "results": [
            {"title": f"R{i}", "url": f"https://e.example/{i}", "description": "x" * 80}
            for i in range(200)
        ]
    }
}


class _BraveRequestHandler(BaseHTTPRequestHandler):
    """Emulates the Brave Search API and records requests."""

    def _reply(self, status: int, body: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        parsed = urlparse(self.path)
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        cast(_BraveServer, self.server).requests.append(
            {
                "method": self.command,
                "path": parsed.path,
                "params": params,
                "token": self.headers.get("X-Subscription-Token", ""),
                "accept": self.headers.get("Accept", ""),
            }
        )
        if self.headers.get("X-Subscription-Token") != _API_KEY:
            self._reply(401, json.dumps({"message": "Unauthorized"}))
            return
        q = params.get("q", "")
        if q == "boom":
            self._reply(500, json.dumps({"message": "Internal Server Error"}))
            return
        if parsed.path == "/web/search":
            if q == "plain":
                self._reply(200, json.dumps(_WEB_RESPONSE_PLAIN))
            elif q == "huge":
                self._reply(200, json.dumps(_WEB_RESPONSE_HUGE))
            else:
                self._reply(200, json.dumps(_WEB_RESPONSE))
        elif parsed.path == "/news/search":
            self._reply(200, json.dumps(_NEWS_RESPONSE))
        elif parsed.path == "/images/search":
            self._reply(200, json.dumps(_IMAGE_RESPONSE))
        elif parsed.path == "/videos/search":
            self._reply(200, json.dumps(_VIDEO_RESPONSE))
        else:
            self._reply(404, json.dumps({"message": "Not found"}))

    def log_message(self, *args: Any) -> None:  # type: ignore[override]
        pass


class _BraveServer(ThreadedHTTPServer):
    """ThreadedHTTPServer that records every request it receives."""

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, _BraveRequestHandler)
        self.requests: list[dict[str, Any]] = []


@pytest.fixture()
def brave_server():
    """Start the emulated Brave API server on a free port; yield (base_url, server)."""
    server = _BraveServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base_url, server
    finally:
        stop_http_server(server, thread)


@pytest.fixture()
def backend(brave_server):
    """A backend pointed at the emulated server with the valid token."""
    base_url, server = brave_server
    b = BraveSearchChannelBackend()
    b._base_url = base_url
    b._api_key = _API_KEY
    return b, server


@pytest.fixture(autouse=True)
def _fresh_config():
    """Start and end every test with no persisted Brave Search config."""
    _config.clear()
    yield
    _config.clear()


def test_agent_instantiation_unauthenticated() -> None:
    """A fresh agent is unauthenticated and exposes only the auth trio."""
    agent = BraveSearchAgent()
    assert agent.name == "Brave Search Agent"
    assert agent._is_authenticated() is False
    names = [t.__name__ for t in agent._get_tools()]
    assert names == [
        "check_brave_search_auth",
        "authenticate_brave_search",
        "clear_brave_search_auth",
    ]


def test_check_auth_unauthenticated_message() -> None:
    """check_brave_search_auth explains how to configure when unauthenticated."""
    agent = BraveSearchAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    msg = tools["check_brave_search_auth"]()
    assert "authenticate_brave_search" in msg
    assert "https://api-dashboard.search.brave.com/" in msg


def test_authenticate_persists_config_and_exposes_tools() -> None:
    """authenticate_brave_search persists config (0600) and unlocks backend tools."""
    agent = BraveSearchAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    result = json.loads(tools["authenticate_brave_search"](f"  {_API_KEY}  "))
    assert result["ok"] is True

    assert _config.path.exists()
    if sys.platform != "win32":
        assert stat.S_IMODE(_config.path.stat().st_mode) == 0o600
    saved = json.loads(_config.path.read_text(encoding="utf-8"))
    assert saved == {"api_key": _API_KEY}

    assert agent._is_authenticated() is True
    checked = json.loads(tools["check_brave_search_auth"]())
    assert checked["ok"] is True

    names = {t.__name__ for t in agent._get_tools()}
    assert {
        "brave_web_search",
        "brave_news_search",
        "brave_image_search",
        "brave_video_search",
    } <= names
    assert "connect" not in names  # infrastructure method, not an LLM tool

    cleared = tools["clear_brave_search_auth"]()
    assert "cleared" in cleared.lower()
    assert not _config.path.exists()
    assert agent._is_authenticated() is False
    assert len(agent._get_tools()) == 3


def test_authenticate_persistence_failure_returns_ok_false() -> None:
    """authenticate_brave_search returns ok:false and never raises when saving fails.

    The failure is forced end to end: a regular FILE occupies the path
    where the config DIRECTORY must be created, so ``_config.save``'s
    mkdir fails.  The agent must stay unauthenticated with the backend
    tools still locked.
    """
    agent = BraveSearchAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    config_dir = _config.path.parent
    if config_dir.is_dir():
        shutil.rmtree(config_dir)
    config_dir.parent.mkdir(parents=True, exist_ok=True)
    config_dir.write_text("occupies the config directory path", encoding="utf-8")
    try:
        result = json.loads(tools["authenticate_brave_search"](_API_KEY))
        assert result["ok"] is False
        assert "could not save config" in result["error"]
        assert agent._is_authenticated() is False
        assert agent._backend._api_key == ""
        assert len(agent._get_tools()) == 3  # backend tools stay locked
    finally:
        config_dir.unlink()


def test_authenticate_rejects_empty_api_key() -> None:
    """authenticate_brave_search refuses an empty or blank api_key."""
    agent = BraveSearchAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    assert "cannot be empty" in tools["authenticate_brave_search"]("")
    assert "cannot be empty" in tools["authenticate_brave_search"]("   ")
    assert not _config.path.exists()


def test_new_agent_loads_persisted_config() -> None:
    """A new agent picks up previously persisted credentials."""
    _config.save({"api_key": _API_KEY})
    agent = BraveSearchAgent()
    assert agent._is_authenticated() is True
    assert agent._backend._api_key == _API_KEY


def test_tools_module_function() -> None:
    """Module-level tools() returns a non-empty tool list."""
    tools = brave_mod.tools()
    assert len(tools) >= 3
    assert all(callable(t) for t in tools)


def test_connect_without_config_fails() -> None:
    """connect() fails cleanly when no config is persisted."""
    b = BraveSearchChannelBackend()
    assert b.connect() is False
    assert "No Brave Search config" in b.connection_info


def test_connect_with_config_succeeds() -> None:
    """connect() loads persisted config into the backend."""
    _config.save({"api_key": _API_KEY})
    b = BraveSearchChannelBackend()
    assert b.connect() is True
    assert b._api_key == _API_KEY
    assert "configured" in b.connection_info


def test_clamp_bounds() -> None:
    """_clamp limits values to the inclusive range."""
    assert _clamp(0, 1, 20) == 1
    assert _clamp(100, 1, 20) == 20
    assert _clamp(10, 1, 20) == 10


def test_brave_web_search_happy_path(backend) -> None:
    """brave_web_search hits /web/search with the token and condenses results."""
    b, server = backend
    result = json.loads(b.brave_web_search("python language"))
    assert result["ok"] is True
    assert result["results"] == [
        {
            "title": "Python",
            "url": "https://python.org",
            "description": "The Python language",
            "age": "2 days ago",
        }
    ]
    assert result["infobox"] == [
        {
            "title": "Python (language)",
            "description": "A programming language",
            "url": "https://en.wikipedia.org/wiki/Python",
        }
    ]
    assert result["faq"] == [
        {"question": "What is Python?", "answer": "A language", "url": "https://faq.example"}
    ]
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("GET", "/web/search")
    assert req["token"] == _API_KEY
    assert req["accept"] == "application/json"
    assert req["params"] == {"q": "python language", "count": "10"}


def test_brave_web_search_without_infobox_or_faq(backend) -> None:
    """A response with no infobox/faq yields no infobox/faq keys."""
    b, _ = backend
    result = json.loads(b.brave_web_search("plain"))
    assert result["ok"] is True
    assert result["results"][0]["url"] == "https://plain.example"
    assert result["results"][0]["age"] == ""  # missing field condensed to ""
    assert "infobox" not in result
    assert "faq" not in result


def test_brave_web_search_optional_params_sent(backend) -> None:
    """Non-empty optional params are all forwarded as query params."""
    b, server = backend
    result = json.loads(
        b.brave_web_search(
            "plain", count=5, offset=2, country="DE", search_lang="de", freshness="pw"
        )
    )
    assert result["ok"] is True
    assert server.requests[-1]["params"] == {
        "q": "plain",
        "count": "5",
        "offset": "2",
        "country": "DE",
        "search_lang": "de",
        "freshness": "pw",
    }


def test_brave_web_search_omits_empty_optional_params(backend) -> None:
    """Empty/zero optional params never appear in the query string."""
    b, server = backend
    b.brave_web_search("plain")
    assert set(server.requests[-1]["params"]) == {"q", "count"}


def test_brave_web_search_clamps_count_and_offset(backend) -> None:
    """count is clamped to 1-20 and offset to 0-9 for web search."""
    b, server = backend
    b.brave_web_search("plain", count=999, offset=99)
    assert server.requests[-1]["params"]["count"] == "20"
    assert server.requests[-1]["params"]["offset"] == "9"
    b.brave_web_search("plain", count=-3)
    assert server.requests[-1]["params"]["count"] == "1"


def test_brave_web_search_empty_query(backend) -> None:
    """An empty or blank query yields ok:false without any HTTP request."""
    b, server = backend
    assert json.loads(b.brave_web_search("")) == {"ok": False, "error": "query cannot be empty"}
    assert json.loads(b.brave_web_search("   "))["ok"] is False
    assert server.requests == []


def test_brave_web_search_output_truncated(backend) -> None:
    """List-heavy responses are truncated to the 8000-char output cap."""
    b, _ = backend
    raw = b.brave_web_search("huge")
    assert len(raw) == 8000
    assert raw.startswith('{"ok": true')


def test_brave_news_search_happy_path(backend) -> None:
    """brave_news_search hits /news/search and condenses results with source."""
    b, server = backend
    result = json.loads(b.brave_news_search("climate summit"))
    assert result["ok"] is True
    assert result["results"] == [
        {
            "title": "Climate summit opens",
            "url": "https://news.example/summit",
            "description": "World leaders meet",
            "age": "1 hour ago",
            "source": "reuters.com",
        }
    ]
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("GET", "/news/search")
    assert req["token"] == _API_KEY
    assert req["params"] == {"q": "climate summit", "count": "10"}


def test_brave_news_search_freshness_and_clamp(backend) -> None:
    """news freshness is forwarded and count is clamped to 1-50."""
    b, server = backend
    b.brave_news_search("climate", count=500, freshness="pd")
    assert server.requests[-1]["params"] == {"q": "climate", "count": "50", "freshness": "pd"}


def test_brave_news_search_empty_query(backend) -> None:
    """An empty news query yields ok:false without any HTTP request."""
    b, server = backend
    assert json.loads(b.brave_news_search(""))["ok"] is False
    assert server.requests == []


def test_brave_image_search_happy_path(backend) -> None:
    """brave_image_search hits /images/search and condenses image results."""
    b, server = backend
    result = json.loads(b.brave_image_search("mountain"))
    assert result["ok"] is True
    assert result["results"] == [
        {
            "title": "Mountain",
            "page_url": "https://page.example/mountain",
            "image_url": "https://img.example/mountain.jpg",
            "thumbnail": "https://thumb.example/mountain.jpg",
            "source": "example.com",
        }
    ]
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("GET", "/images/search")
    assert req["token"] == _API_KEY
    assert req["params"] == {"q": "mountain", "count": "10"}


def test_brave_image_search_clamps_count(backend) -> None:
    """image count is clamped to 1-200."""
    b, server = backend
    b.brave_image_search("mountain", count=1000)
    assert server.requests[-1]["params"]["count"] == "200"


def test_brave_image_search_empty_query(backend) -> None:
    """An empty image query yields ok:false without any HTTP request."""
    b, server = backend
    assert json.loads(b.brave_image_search(" "))["ok"] is False
    assert server.requests == []


def test_brave_video_search_happy_path(backend) -> None:
    """brave_video_search hits /videos/search and condenses video results."""
    b, server = backend
    result = json.loads(b.brave_video_search("tutorial"))
    assert result["ok"] is True
    assert result["results"] == [
        {
            "title": "Tutorial",
            "url": "https://video.example/tut",
            "description": "Learn things",
            "age": "3 days ago",
            "duration": "10:02",
            "creator": "Prof X",
        }
    ]
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("GET", "/videos/search")
    assert req["token"] == _API_KEY
    assert req["params"] == {"q": "tutorial", "count": "10"}


def test_brave_video_search_clamps_count(backend) -> None:
    """video count is clamped to 1-50."""
    b, server = backend
    b.brave_video_search("tutorial", count=500)
    assert server.requests[-1]["params"]["count"] == "50"


def test_brave_video_search_empty_query(backend) -> None:
    """An empty video query yields ok:false without any HTTP request."""
    b, server = backend
    assert json.loads(b.brave_video_search(""))["ok"] is False
    assert server.requests == []


def test_wrong_token_returns_ok_false(brave_server) -> None:
    """A 401 from the server yields ok:false JSON from every tool — no exception."""
    base_url, _ = brave_server
    b = BraveSearchChannelBackend()
    b._base_url = base_url
    b._api_key = "wrong-token"
    for call in (
        lambda: b.brave_web_search("python"),
        lambda: b.brave_news_search("python"),
        lambda: b.brave_image_search("python"),
        lambda: b.brave_video_search("python"),
    ):
        result = json.loads(call())
        assert result["ok"] is False
        assert "401" in result["error"]


def test_server_error_returns_ok_false(backend) -> None:
    """A 500 from the server yields ok:false JSON from every tool, not an exception."""
    b, _ = backend
    for call in (
        lambda: b.brave_web_search("boom"),
        lambda: b.brave_news_search("boom"),
        lambda: b.brave_image_search("boom"),
        lambda: b.brave_video_search("boom"),
    ):
        result = json.loads(call())
        assert result["ok"] is False
        assert "500" in result["error"]


def test_connection_refused_returns_ok_false() -> None:
    """Tools return ok:false when the server is unreachable — never raise."""
    b = BraveSearchChannelBackend()
    b._base_url = "http://127.0.0.1:9"  # discard port; nothing listens
    b._api_key = _API_KEY
    for call in (
        lambda: b.brave_web_search("python"),
        lambda: b.brave_news_search("python"),
        lambda: b.brave_image_search("python"),
        lambda: b.brave_video_search("python"),
    ):
        result = json.loads(call())
        assert result["ok"] is False
        assert result["error"]


def test_default_base_url_is_real_api() -> None:
    """A fresh backend defaults to the real Brave Search API base URL."""
    b = BraveSearchChannelBackend()
    assert b._base_url == "https://api.search.brave.com/res/v1"
