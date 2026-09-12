# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Notion channel agent.

Runs a REAL local HTTP server (stdlib ``ThreadedHTTPServer``) emulating
the Notion REST API endpoints — no mocks, patches, or fakes.  The
server asserts the ``Authorization: Bearer`` header on every call,
returns canned JSON, and records methods, paths, headers, and bodies
for verification.

Config state is isolated per pytest process because the session
conftest points ``KISS_HOME`` at a temporary directory and
``ChannelConfig.path`` resolves ``$KISS_HOME`` lazily.

``main()`` is not exercised here: it only forwards to the shared
``channel_main`` CLI (covered by the shared CLI tests) and its body has
no branches of its own.
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

import kiss.agents.third_party_agents.notion_agent as notion_mod
from kiss.agents.third_party_agents._backend_utils import (
    ThreadedHTTPServer,
    stop_http_server,
)
from kiss.agents.third_party_agents.notion_agent import (
    NotionAgent,
    NotionChannelBackend,
    _config,
)

_TOKEN = "ntn_test_token"

# A page whose properties exercise every _prop_value branch: title
# (with a non-dict rich-text item, skipped), rich_text, select (dict
# with "name"), number (scalar), checkbox (bool), date (dict without
# "name" -> "<date>"), people (list -> "<people>").
_PAGE: dict[str, Any] = {
    "object": "page",
    "id": "page-1",
    "url": "https://notion.so/page-1",
    "archived": False,
    "parent": {"type": "workspace", "workspace": True},
    "last_edited_time": "2026-01-02T03:04:05.000Z",
    "properties": {
        "Name": {"type": "title", "title": [{"plain_text": "Project Plan"}, "junk"]},
        "Notes": {"type": "rich_text", "rich_text": [{"plain_text": "some notes"}]},
        "Status": {"type": "select", "select": {"name": "In Progress"}},
        "Points": {"type": "number", "number": 5},
        "Done": {"type": "checkbox", "checkbox": True},
        "Due": {"type": "date", "date": {"start": "2026-01-05"}},
        "Owner": {"type": "people", "people": [{"id": "u-1"}]},
    },
}

# A database-row page with no title property (title falls back to "")
# and a non-dict property entry (skipped by _page_title's isinstance).
_DB_PAGE: dict[str, Any] = {
    "object": "page",
    "id": "page-2",
    "url": "https://notion.so/page-2",
    "archived": True,
    "parent": {"type": "database_id", "database_id": "db-1"},
    "last_edited_time": "2026-01-03T00:00:00.000Z",
    "properties": {"Status": {"type": "select", "select": {"name": "Done"}}, "Junk": "notadict"},
}

_SEARCH_DB: dict[str, Any] = {
    "object": "database",
    "id": "db-1",
    "title": [{"plain_text": "Tasks"}],
    "url": "https://notion.so/db-1",
}

_DATABASE: dict[str, Any] = {
    "object": "database",
    "id": "db-1",
    "title": [{"plain_text": "Tasks"}],
    "url": "https://notion.so/db-1",
    "properties": {"Name": {"type": "title"}, "Status": {"type": "select"}},
}

# Blocks exercising every _condense_block / _plain_text branch:
# rich text, empty type value ({} -> plain_text(None) -> ""), and a
# non-dict type value.
_BLOCKS: list[dict[str, Any]] = [
    {
        "object": "block",
        "id": "blk-1",
        "type": "paragraph",
        "has_children": False,
        "paragraph": {"rich_text": [{"plain_text": "Hello "}, {"plain_text": "world"}]},
    },
    {
        "object": "block",
        "id": "blk-2",
        "type": "heading_1",
        "has_children": True,
        "heading_1": {"rich_text": [{"plain_text": "Intro"}]},
    },
    {"object": "block", "id": "blk-3", "type": "divider", "has_children": False, "divider": {}},
    {
        "object": "block",
        "id": "blk-4",
        "type": "weird",
        "has_children": False,
        "weird": "notadict",
    },
]

_APPENDED_BLOCK: dict[str, Any] = {
    "object": "block",
    "id": "blk-new",
    "type": "paragraph",
    "has_children": False,
    "paragraph": {"rich_text": [{"plain_text": "appended"}]},
}

_USERS: list[dict[str, Any]] = [
    {"object": "user", "id": "u-1", "name": "Alice", "type": "person"},
    {"object": "user", "id": "u-2", "name": "KISS Bot", "type": "bot"},
]

# One comment with created_by and one without (covers the `or {}`).
_COMMENTS: list[dict[str, Any]] = [
    {
        "object": "comment",
        "id": "c-1",
        "rich_text": [{"plain_text": "LGTM"}],
        "created_time": "2026-01-04T00:00:00.000Z",
        "created_by": {"id": "u-1"},
    },
    {"object": "comment", "id": "c-2", "rich_text": [], "created_time": ""},
]


def _list_response(results: list[dict[str, Any]], next_cursor: Any, has_more: bool) -> str:
    return json.dumps(
        {"object": "list", "results": results, "next_cursor": next_cursor, "has_more": has_more}
    )


class _NotionRequestHandler(BaseHTTPRequestHandler):
    """Emulates the Notion REST API and records requests."""

    def _authorized(self) -> bool:
        return self.headers.get("Authorization") == f"Bearer {_TOKEN}"

    def _reply(self, status: int, body: str, content_type: str = "application/json") -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _record(self, body: dict[str, Any] | None) -> None:
        cast(_NotionServer, self.server).requests.append(
            {
                "method": self.command,
                "path": self.path,
                "authorization": self.headers.get("Authorization", ""),
                "notion_version": self.headers.get("Notion-Version", ""),
                "body": body,
            }
        )

    def _read_body(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            return None

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        self._record(None)
        if not self._authorized():
            self._reply(401, json.dumps({"object": "error", "code": "unauthorized"}))
            return
        path = urlparse(self.path).path
        if path == "/v1/fail":
            self._reply(500, json.dumps({"object": "error", "code": "internal_server_error"}))
        elif path == "/v1/notjson":
            self._reply(200, "plain text", content_type="text/plain")
        elif path == "/v1/pages/page-1":
            self._reply(200, json.dumps(_PAGE))
        elif path == "/v1/blocks/page-1/children":
            self._reply(200, _list_response(_BLOCKS, "cur-2", True))
        elif path == "/v1/databases/db-1":
            self._reply(200, json.dumps(_DATABASE))
        elif path == "/v1/users":
            self._reply(200, _list_response(_USERS, None, False))
        elif path == "/v1/comments":
            self._reply(200, _list_response(_COMMENTS, None, False))
        else:
            self._reply(404, json.dumps({"object": "error", "code": "object_not_found"}))

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        body = self._read_body()
        self._record(body)
        if not self._authorized():
            self._reply(401, json.dumps({"object": "error", "code": "unauthorized"}))
            return
        path = urlparse(self.path).path
        if path == "/v1/search":
            if body and body.get("query") == "many":
                results: list[dict[str, Any]] = [
                    {
                        "object": "database",
                        "id": f"db-{i}",
                        "title": [{"plain_text": "T" * 60}],
                        "url": f"https://notion.so/db-{i}",
                    }
                    for i in range(300)
                ]
                self._reply(200, _list_response(results, None, False))
            else:
                self._reply(200, _list_response([_PAGE, _SEARCH_DB], "cur-1", True))
        elif path == "/v1/pages":
            assert body is not None
            # Notion echoes created properties with "type" and
            # "plain_text" filled in; deep-copy so the recorded request
            # body stays exactly what the client sent.
            properties = json.loads(json.dumps(body["properties"]))
            for prop in properties.values():
                if "title" in prop:
                    prop["type"] = "title"
                    for item in prop["title"]:
                        item["plain_text"] = item.get("text", {}).get("content", "")
            created = {
                "object": "page",
                "id": "page-new",
                "url": "https://notion.so/page-new",
                "archived": False,
                "parent": body["parent"],
                "last_edited_time": "2026-01-05T00:00:00.000Z",
                "properties": properties,
            }
            self._reply(200, json.dumps(created))
        elif path == "/v1/databases/db-1/query":
            self._reply(200, _list_response([_DB_PAGE], None, False))
        elif path == "/v1/comments":
            self._reply(
                200, json.dumps({"object": "comment", "id": "c-new", "discussion_id": "d-1"})
            )
        else:
            self._reply(404, json.dumps({"object": "error", "code": "object_not_found"}))

    def do_PATCH(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        body = self._read_body()
        self._record(body)
        if not self._authorized():
            self._reply(401, json.dumps({"object": "error", "code": "unauthorized"}))
            return
        path = urlparse(self.path).path
        if path == "/v1/blocks/page-1/children":
            self._reply(200, _list_response([_APPENDED_BLOCK], None, False))
        elif path == "/v1/pages/page-1":
            assert body is not None
            updated = dict(_PAGE)
            updated["archived"] = body.get("archived", False)
            self._reply(200, json.dumps(updated))
        else:
            self._reply(404, json.dumps({"object": "error", "code": "object_not_found"}))

    def log_message(self, *args: Any) -> None:  # type: ignore[override]
        pass


class _NotionServer(ThreadedHTTPServer):
    """ThreadedHTTPServer that records every request it receives."""

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, _NotionRequestHandler)
        self.requests: list[dict[str, Any]] = []


@pytest.fixture()
def notion_server():
    """Start the emulated Notion API on a free port; yield (base_url, server)."""
    server = _NotionServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    try:
        yield base_url, server
    finally:
        stop_http_server(server, thread)


@pytest.fixture()
def backend(notion_server):
    """A backend pointed at the emulated server with the valid token."""
    base_url, server = notion_server
    b = NotionChannelBackend()
    b._base_url = base_url
    b._token = _TOKEN
    return b, server


@pytest.fixture(autouse=True)
def _fresh_config():
    """Start and end every test with no persisted Notion config."""
    _config.clear()
    yield
    _config.clear()


_ALL_TOOL_NAMES = {
    "notion_search",
    "notion_get_page",
    "notion_get_block_children",
    "notion_append_paragraph",
    "notion_append_blocks",
    "notion_create_page",
    "notion_update_page",
    "notion_get_database",
    "notion_query_database",
    "notion_list_users",
    "notion_create_comment",
    "notion_get_comments",
}


def _call_every_tool(b: NotionChannelBackend) -> list[str]:
    """Invoke every backend tool once with happy-path arguments."""
    return [
        b.notion_search("x"),
        b.notion_get_page("page-1"),
        b.notion_get_block_children("page-1"),
        b.notion_append_paragraph("page-1", "hi"),
        b.notion_append_blocks("page-1", "[]"),
        b.notion_create_page("page-1", "T"),
        b.notion_update_page("page-1", archived="true"),
        b.notion_get_database("db-1"),
        b.notion_query_database("db-1"),
        b.notion_list_users(),
        b.notion_create_comment("page-1", "hi"),
        b.notion_get_comments("page-1"),
    ]


def test_agent_instantiation_unauthenticated() -> None:
    """A fresh agent is unauthenticated and exposes only the auth trio."""
    agent = NotionAgent()
    assert agent.name == "Notion Agent"
    assert agent._is_authenticated() is False
    names = [t.__name__ for t in agent._get_tools()]
    assert names == ["check_notion_auth", "authenticate_notion", "clear_notion_auth"]


def test_check_auth_unauthenticated_message() -> None:
    """check_notion_auth explains how to configure when unauthenticated."""
    agent = NotionAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    msg = tools["check_notion_auth"]()
    assert "authenticate_notion" in msg
    assert "https://www.notion.so/profile/integrations" in msg
    assert "share" in msg.lower()


def test_authenticate_persists_config_and_exposes_tools() -> None:
    """authenticate_notion persists config (0600) and unlocks backend tools."""
    agent = NotionAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    result = json.loads(tools["authenticate_notion"](f"  {_TOKEN} "))
    assert result["ok"] is True

    assert _config.path.exists()
    if sys.platform != "win32":
        assert stat.S_IMODE(_config.path.stat().st_mode) == 0o600
    saved = json.loads(_config.path.read_text(encoding="utf-8"))
    assert saved == {"token": _TOKEN}

    assert agent._is_authenticated() is True
    checked = json.loads(tools["check_notion_auth"]())
    assert checked["ok"] is True

    names = {t.__name__ for t in agent._get_tools()}
    assert _ALL_TOOL_NAMES <= names
    assert "connect" not in names  # channel protocol method, not an LLM tool

    cleared = tools["clear_notion_auth"]()
    assert "cleared" in cleared.lower()
    assert not _config.path.exists()
    assert agent._is_authenticated() is False
    assert len(agent._get_tools()) == 3


def test_authenticate_persistence_failure_returns_ok_false() -> None:
    """authenticate_notion returns ok:false and never raises when saving fails.

    The failure is forced end to end: a regular FILE occupies the path
    where the config DIRECTORY must be created, so ``_config.save``'s
    mkdir fails.  The agent must stay unauthenticated with the backend
    tools still locked.
    """
    agent = NotionAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    config_dir = _config.path.parent
    if config_dir.is_dir():
        shutil.rmtree(config_dir)
    config_dir.parent.mkdir(parents=True, exist_ok=True)
    config_dir.write_text("occupies the config directory path", encoding="utf-8")
    try:
        result = json.loads(tools["authenticate_notion"](_TOKEN))
        assert result["ok"] is False
        assert "could not save config" in result["error"]
        assert agent._is_authenticated() is False
        assert agent._backend._token == ""
        assert len(agent._get_tools()) == 3  # backend tools stay locked
    finally:
        config_dir.unlink()


def test_authenticate_rejects_empty_token() -> None:
    """authenticate_notion refuses an empty or whitespace token."""
    agent = NotionAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    assert "cannot be empty" in tools["authenticate_notion"]("   ")
    assert not _config.path.exists()


def test_new_agent_loads_persisted_config() -> None:
    """A new agent picks up previously persisted credentials."""
    _config.save({"token": _TOKEN})
    agent = NotionAgent()
    assert agent._is_authenticated() is True
    assert agent._backend._token == _TOKEN
    assert agent._backend._base_url == "https://api.notion.com/v1"


def test_tools_module_function() -> None:
    """Module-level tools() returns a non-empty tool list."""
    tools = notion_mod.tools()
    assert len(tools) >= 3
    assert all(callable(t) for t in tools)


def test_connect_without_config_fails() -> None:
    """connect() fails cleanly when no config is persisted."""
    b = NotionChannelBackend()
    assert b.connect() is False
    assert "No Notion config" in b.connection_info


def test_connect_with_config_succeeds() -> None:
    """connect() loads the persisted token into the backend."""
    _config.save({"token": _TOKEN})
    b = NotionChannelBackend()
    assert b.connect() is True
    assert b._token == _TOKEN
    assert "configured" in b.connection_info


def test_notion_search_happy_path(backend) -> None:
    """notion_search posts to /v1/search and condenses page and database hits."""
    b, server = backend
    result = json.loads(b.notion_search("plan"))
    assert result["ok"] is True
    assert result["results"] == [
        {
            "object": "page",
            "id": "page-1",
            "title": "Project Plan",
            "url": "https://notion.so/page-1",
        },
        {"object": "database", "id": "db-1", "title": "Tasks", "url": "https://notion.so/db-1"},
    ]
    assert result["next_cursor"] == "cur-1"
    assert result["has_more"] is True
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("POST", "/v1/search")
    assert req["authorization"] == f"Bearer {_TOKEN}"
    assert req["notion_version"] == "2022-06-28"
    assert req["body"] == {"query": "plan", "page_size": 20}


def test_notion_search_with_filter_and_cursor(backend) -> None:
    """filter_type and start_cursor are forwarded in the search body."""
    b, server = backend
    result = json.loads(b.notion_search("plan", filter_type="page", start_cursor="c9", page_size=5))
    assert result["ok"] is True
    assert server.requests[-1]["body"] == {
        "query": "plan",
        "page_size": 5,
        "filter": {"property": "object", "value": "page"},
        "start_cursor": "c9",
    }


def test_notion_search_invalid_filter_type(backend) -> None:
    """An unknown filter_type is rejected without any HTTP request."""
    b, server = backend
    result = json.loads(b.notion_search("x", filter_type="user"))
    assert result["ok"] is False
    assert "filter_type" in result["error"]
    assert server.requests == []


def test_notion_search_output_truncated(backend) -> None:
    """A list-heavy search response is truncated to 8000 characters."""
    b, _ = backend
    out = b.notion_search("many", page_size=100)
    assert len(out) == 8000


def test_notion_get_page_condenses_properties(backend) -> None:
    """notion_get_page condenses every property type to a plain value."""
    b, server = backend
    result = json.loads(b.notion_get_page("page-1"))
    assert result["ok"] is True
    page = result["page"]
    assert page["id"] == "page-1"
    assert page["title"] == "Project Plan"
    assert page["url"] == "https://notion.so/page-1"
    assert page["archived"] is False
    assert page["parent"] == {"type": "workspace", "workspace": True}
    assert page["last_edited_time"] == "2026-01-02T03:04:05.000Z"
    assert page["properties"] == {
        "Name": "Project Plan",
        "Notes": "some notes",
        "Status": "In Progress",
        "Points": 5,
        "Done": True,
        "Due": "<date>",
        "Owner": "<people>",
    }
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("GET", "/v1/pages/page-1")
    assert req["notion_version"] == "2022-06-28"


def test_notion_get_block_children(backend) -> None:
    """notion_get_block_children renders each block's type, id, and text."""
    b, server = backend
    result = json.loads(b.notion_get_block_children("page-1"))
    assert result["ok"] is True
    assert result["blocks"] == [
        {"id": "blk-1", "type": "paragraph", "text": "Hello world", "has_children": False},
        {"id": "blk-2", "type": "heading_1", "text": "Intro", "has_children": True},
        {"id": "blk-3", "type": "divider", "text": "", "has_children": False},
        {"id": "blk-4", "type": "weird", "text": "", "has_children": False},
    ]
    assert result["next_cursor"] == "cur-2"
    assert result["has_more"] is True
    parsed = urlparse(server.requests[-1]["path"])
    assert parsed.path == "/v1/blocks/page-1/children"
    assert parse_qs(parsed.query) == {"page_size": ["50"]}


def test_notion_get_block_children_with_cursor(backend) -> None:
    """start_cursor and page_size become query parameters."""
    b, server = backend
    result = json.loads(b.notion_get_block_children("page-1", start_cursor="c3", page_size=7))
    assert result["ok"] is True
    parsed = urlparse(server.requests[-1]["path"])
    assert parse_qs(parsed.query) == {"page_size": ["7"], "start_cursor": ["c3"]}


def test_notion_append_paragraph(backend) -> None:
    """notion_append_paragraph PATCHes a paragraph block built from text."""
    b, server = backend
    result = json.loads(b.notion_append_paragraph("page-1", "New line"))
    assert result["ok"] is True
    assert result["appended"] == [
        {"id": "blk-new", "type": "paragraph", "text": "appended", "has_children": False}
    ]
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("PATCH", "/v1/blocks/page-1/children")
    assert req["body"] == {
        "children": [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": "New line"}}]},
            }
        ]
    }


def test_notion_append_paragraph_empty_text(backend) -> None:
    """Empty paragraph text is rejected without any HTTP request."""
    b, server = backend
    result = json.loads(b.notion_append_paragraph("page-1", "   "))
    assert result == {"ok": False, "error": "text cannot be empty"}
    assert server.requests == []


def test_notion_append_blocks(backend) -> None:
    """notion_append_blocks PATCHes the caller-supplied block array."""
    b, server = backend
    children = [{"object": "block", "type": "divider", "divider": {}}]
    result = json.loads(b.notion_append_blocks("page-1", json.dumps(children)))
    assert result["ok"] is True
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("PATCH", "/v1/blocks/page-1/children")
    assert req["body"] == {"children": children}


def test_notion_append_blocks_invalid_json(backend) -> None:
    """Malformed or non-array children_json is rejected without a request."""
    b, server = backend
    result = json.loads(b.notion_append_blocks("page-1", "{not json"))
    assert result == {"ok": False, "error": "children_json is not valid JSON"}
    result = json.loads(b.notion_append_blocks("page-1", '{"a": 1}'))
    assert result == {"ok": False, "error": "children_json must be a JSON array"}
    assert server.requests == []


def test_notion_create_page_under_page(backend) -> None:
    """Page-parent creation posts a page_id parent with a title property."""
    b, server = backend
    result = json.loads(b.notion_create_page("page-1", "Child Page"))
    assert result["ok"] is True
    assert result["page"]["id"] == "page-new"
    assert result["page"]["title"] == "Child Page"
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("POST", "/v1/pages")
    assert req["body"] == {
        "parent": {"page_id": "page-1"},
        "properties": {
            "title": {"title": [{"type": "text", "text": {"content": "Child Page"}}]}
        },
    }


def test_notion_create_page_in_database(backend) -> None:
    """Database-parent creation merges properties_json over the title."""
    b, server = backend
    props = {"Status": {"select": {"name": "Done"}}}
    result = json.loads(
        b.notion_create_page(
            "db-1", "Row", parent_type="database", properties_json=json.dumps(props)
        )
    )
    assert result["ok"] is True
    req = server.requests[-1]
    assert req["body"] == {
        "parent": {"database_id": "db-1"},
        "properties": {
            "title": {"title": [{"type": "text", "text": {"content": "Row"}}]},
            "Status": {"select": {"name": "Done"}},
        },
    }


def test_notion_create_page_no_title(backend) -> None:
    """An empty title yields no title property (properties_json rules)."""
    b, server = backend
    result = json.loads(
        b.notion_create_page(
            "db-1", "", parent_type="database", properties_json='{"A": {"number": 1}}'
        )
    )
    assert result["ok"] is True
    assert server.requests[-1]["body"]["properties"] == {"A": {"number": 1}}


def test_notion_create_page_validation(backend) -> None:
    """Invalid parent_type or properties_json is rejected without a request."""
    b, server = backend
    result = json.loads(b.notion_create_page("page-1", "T", parent_type="workspace"))
    assert result == {"ok": False, "error": "parent_type must be 'page' or 'database'"}
    result = json.loads(b.notion_create_page("db-1", "T", "database", "[1]"))
    assert result == {"ok": False, "error": "properties_json must be a JSON object"}
    result = json.loads(b.notion_create_page("db-1", "T", "database", "{bad"))
    assert result == {"ok": False, "error": "properties_json is not valid JSON"}
    assert server.requests == []


def test_notion_update_page_properties_and_archive(backend) -> None:
    """notion_update_page PATCHes properties and the archived flag."""
    b, server = backend
    props = {"Status": {"select": {"name": "Done"}}}
    result = json.loads(
        b.notion_update_page("page-1", properties_json=json.dumps(props), archived="true")
    )
    assert result["ok"] is True
    assert result["page"]["archived"] is True
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("PATCH", "/v1/pages/page-1")
    assert req["body"] == {"properties": props, "archived": True}


def test_notion_update_page_unarchive_only(backend) -> None:
    """archived='false' alone sends {"archived": false}."""
    b, server = backend
    result = json.loads(b.notion_update_page("page-1", archived="false"))
    assert result["ok"] is True
    assert server.requests[-1]["body"] == {"archived": False}


def test_notion_update_page_validation(backend) -> None:
    """Bad archived values, bad JSON, and empty updates are rejected."""
    b, server = backend
    result = json.loads(b.notion_update_page("page-1", archived="yes"))
    assert result == {"ok": False, "error": "archived must be '', 'true', or 'false'"}
    result = json.loads(b.notion_update_page("page-1", properties_json="[1]"))
    assert result == {"ok": False, "error": "properties_json must be a JSON object"}
    result = json.loads(b.notion_update_page("page-1"))
    assert result["ok"] is False
    assert "nothing to update" in result["error"]
    assert server.requests == []


def test_notion_get_database(backend) -> None:
    """notion_get_database condenses the title and property schema."""
    b, server = backend
    result = json.loads(b.notion_get_database("db-1"))
    assert result["ok"] is True
    assert result["database"] == {
        "id": "db-1",
        "title": "Tasks",
        "url": "https://notion.so/db-1",
        "properties": {"Name": "title", "Status": "select"},
    }
    assert server.requests[-1]["path"] == "/v1/databases/db-1"


def test_notion_query_database(backend) -> None:
    """notion_query_database posts filter/sorts/cursor and condenses rows."""
    b, server = backend
    filter_obj = {"property": "Status", "select": {"equals": "Done"}}
    sorts = [{"property": "Status", "direction": "ascending"}]
    result = json.loads(
        b.notion_query_database(
            "db-1",
            filter_json=json.dumps(filter_obj),
            sorts_json=json.dumps(sorts),
            start_cursor="c1",
            page_size=3,
        )
    )
    assert result["ok"] is True
    assert result["results"] == [
        {
            "id": "page-2",
            "title": "",
            "url": "https://notion.so/page-2",
            "archived": True,
            "parent": {"type": "database_id", "database_id": "db-1"},
            "last_edited_time": "2026-01-03T00:00:00.000Z",
        }
    ]
    assert result["has_more"] is False
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("POST", "/v1/databases/db-1/query")
    assert req["body"] == {
        "page_size": 3,
        "filter": filter_obj,
        "sorts": sorts,
        "start_cursor": "c1",
    }


def test_notion_query_database_defaults(backend) -> None:
    """A bare query sends only page_size."""
    b, server = backend
    result = json.loads(b.notion_query_database("db-1"))
    assert result["ok"] is True
    assert server.requests[-1]["body"] == {"page_size": 20}


def test_notion_query_database_validation(backend) -> None:
    """Bad filter_json and sorts_json are rejected without a request."""
    b, server = backend
    result = json.loads(b.notion_query_database("db-1", filter_json="[1]"))
    assert result == {"ok": False, "error": "filter_json must be a JSON object"}
    result = json.loads(b.notion_query_database("db-1", filter_json="{bad"))
    assert result == {"ok": False, "error": "filter_json is not valid JSON"}
    result = json.loads(b.notion_query_database("db-1", sorts_json='{"a": 1}'))
    assert result == {"ok": False, "error": "sorts_json must be a JSON array"}
    assert server.requests == []


def test_notion_list_users(backend) -> None:
    """notion_list_users condenses user id, name, and type."""
    b, server = backend
    result = json.loads(b.notion_list_users())
    assert result["ok"] is True
    assert result["users"] == [
        {"id": "u-1", "name": "Alice", "type": "person"},
        {"id": "u-2", "name": "KISS Bot", "type": "bot"},
    ]
    parsed = urlparse(server.requests[-1]["path"])
    assert parsed.path == "/v1/users"
    assert parsed.query == ""


def test_notion_list_users_with_cursor(backend) -> None:
    """start_cursor becomes a query parameter on /v1/users."""
    b, server = backend
    result = json.loads(b.notion_list_users(start_cursor="c5"))
    assert result["ok"] is True
    parsed = urlparse(server.requests[-1]["path"])
    assert parse_qs(parsed.query) == {"start_cursor": ["c5"]}


def test_notion_create_comment(backend) -> None:
    """notion_create_comment posts a page-parent comment."""
    b, server = backend
    result = json.loads(b.notion_create_comment("page-1", "Nice work"))
    assert result == {"ok": True, "comment": {"id": "c-new", "discussion_id": "d-1"}}
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("POST", "/v1/comments")
    assert req["body"] == {
        "parent": {"page_id": "page-1"},
        "rich_text": [{"type": "text", "text": {"content": "Nice work"}}],
    }


def test_notion_create_comment_empty_text(backend) -> None:
    """Empty comment text is rejected without any HTTP request."""
    b, server = backend
    result = json.loads(b.notion_create_comment("page-1", ""))
    assert result == {"ok": False, "error": "text cannot be empty"}
    assert server.requests == []


def test_notion_get_comments(backend) -> None:
    """notion_get_comments lists comments with plain text and author id."""
    b, server = backend
    result = json.loads(b.notion_get_comments("page-1"))
    assert result["ok"] is True
    assert result["comments"] == [
        {
            "id": "c-1",
            "text": "LGTM",
            "created_time": "2026-01-04T00:00:00.000Z",
            "created_by": "u-1",
        },
        {"id": "c-2", "text": "", "created_time": "", "created_by": ""},
    ]
    parsed = urlparse(server.requests[-1]["path"])
    assert parsed.path == "/v1/comments"
    assert parse_qs(parsed.query) == {"block_id": ["page-1"]}


def test_notion_get_comments_with_cursor(backend) -> None:
    """start_cursor becomes a query parameter on /v1/comments."""
    b, server = backend
    result = json.loads(b.notion_get_comments("page-1", start_cursor="c7"))
    assert result["ok"] is True
    parsed = urlparse(server.requests[-1]["path"])
    assert parse_qs(parsed.query) == {"block_id": ["page-1"], "start_cursor": ["c7"]}


def test_path_traversal_ids_rejected_before_any_request(backend) -> None:
    """Traversal attempts in every id parameter are refused up front."""
    b, server = backend
    for attempt in (
        b.notion_get_page("../users"),
        b.notion_get_page("a\\b"),
        b.notion_get_block_children("x/../y"),
        b.notion_append_paragraph("a/b", "text"),
        b.notion_append_blocks("..", "[]"),
        b.notion_update_page("a/b", archived="true"),
        b.notion_get_database("../search"),
        b.notion_query_database("db/../db"),
        b.notion_get_comments("a\\b"),
    ):
        result = json.loads(attempt)
        assert result["ok"] is False
        assert "invalid" in result["error"]
    assert server.requests == []  # nothing ever reached the server


def test_page_id_is_encoded_as_single_path_segment(backend) -> None:
    """Special characters in ids are percent-encoded, never path-interpreted."""
    b, server = backend
    result = json.loads(b.notion_get_page("page 1%x"))
    assert result["ok"] is False  # emulator 404s the unknown id — that's fine
    assert server.requests[-1]["path"] == "/v1/pages/page%201%25x"


def test_unauthorized_token_returns_ok_false(notion_server) -> None:
    """A 401 from the API yields ok:false JSON from every tool — no exception."""
    base_url, _ = notion_server
    b = NotionChannelBackend()
    b._base_url = base_url
    b._token = "wrong-token"
    for out in _call_every_tool(b):
        result = json.loads(out)
        assert result["ok"] is False
        assert "401" in result["error"]


def test_server_error_returns_ok_false(backend) -> None:
    """A 500 from the API yields an error tuple, not an exception."""
    b, _ = backend
    data, err = b._request("GET", "/fail")
    assert data is None
    result = json.loads(err)
    assert result["ok"] is False
    assert "500" in result["error"]


def test_non_json_response_returns_ok_false(backend) -> None:
    """A non-JSON 200 body yields an error tuple, not an exception."""
    b, _ = backend
    data, err = b._request("GET", "/notjson")
    assert data is None
    assert json.loads(err) == {"ok": False, "error": "non-JSON response from Notion API"}


def test_connection_refused_returns_ok_false() -> None:
    """Every tool returns ok:false when the API is unreachable — never raises."""
    b = NotionChannelBackend()
    b._base_url = "http://127.0.0.1:9"  # discard port; nothing listens
    b._token = _TOKEN
    for out in _call_every_tool(b):
        result = json.loads(out)
        assert result["ok"] is False
        assert result["error"]
