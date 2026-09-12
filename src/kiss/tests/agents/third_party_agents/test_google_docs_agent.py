# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Google Docs channel agent.

Runs a REAL local HTTP server (stdlib ``ThreadedHTTPServer``) emulating
the Google Docs v1 and Drive v3 REST endpoints — no mocks, patches, or
fakes.  The server asserts the ``Authorization: Bearer`` header on
every call, returns canned JSON, and records requests (method, path,
query, body) for verification.

Credential state is isolated per pytest process because the session
conftest points ``KISS_HOME`` at a temporary directory and every
``_google_workspace_utils`` path helper resolves ``$KISS_HOME`` lazily.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import pytest

import kiss.agents.third_party_agents.google_docs_agent as gdocs_mod
from kiss.agents.third_party_agents._backend_utils import (
    ThreadedHTTPServer,
    stop_http_server,
)
from kiss.agents.third_party_agents._google_workspace_utils import (
    clear_google_credentials,
    fresh_access_token,
    token_path,
)
from kiss.agents.third_party_agents.google_docs_agent import (
    _SCOPES,
    _SERVICE,
    GoogleDocsAgent,
    GoogleDocsChannelBackend,
)

_TOKEN = "test-token"
_SYNTHETIC_TOKEN = "synthetic-access-token"

_AUTH_TOOL_NAMES = [
    "check_google_docs_auth",
    "authenticate_google_docs",
    "clear_google_docs_auth",
    "start_google_docs_browser_setup",
]

_BACKEND_TOOL_NAMES = {
    "gdocs_create_document",
    "gdocs_read_document",
    "gdocs_append_text",
    "gdocs_replace_text",
    "gdocs_insert_text",
    "gdocs_batch_update",
    "gdocs_list_documents",
}

# Realistic Document payload: a string element (schema noise), a
# section break, a paragraph with a non-text run in the middle, and a
# 1x2 table whose cells contain paragraphs.
_DOCUMENT = {
    "documentId": "doc123",
    "title": "Test Doc",
    "body": {
        "content": [
            "schema-noise",
            {"sectionBreak": {"sectionStyle": {}}},
            {
                "paragraph": {
                    "elements": [
                        {"textRun": {"content": "Hello "}},
                        {"pageBreak": {}},
                        {"textRun": {"content": "world.\n"}},
                    ]
                }
            },
            {
                "table": {
                    "tableRows": [
                        {
                            "tableCells": [
                                {
                                    "content": [
                                        {
                                            "paragraph": {
                                                "elements": [
                                                    {"textRun": {"content": "Cell A\n"}}
                                                ]
                                            }
                                        }
                                    ]
                                },
                                {
                                    "content": [
                                        {
                                            "paragraph": {
                                                "elements": [
                                                    {"textRun": {"content": "Cell B\n"}}
                                                ]
                                            }
                                        }
                                    ]
                                },
                            ]
                        }
                    ]
                }
            },
        ]
    },
}

_EXPECTED_TEXT = "Hello world.\nCell A\nCell B\n"

# Realistic multi-tab Document payload (includeTabsContent=true): two
# top-level tabs, the first with a nested child tab; schema noise (a
# non-dict tab, a tab without documentTab) must be skipped.
_TABBED_DOCUMENT = {
    "documentId": "tabbed",
    "title": "Tabbed Doc",
    "tabs": [
        "tab-schema-noise",
        {
            "tabProperties": {"tabId": "t1", "title": "First"},
            "documentTab": {
                "body": {
                    "content": [
                        {
                            "paragraph": {
                                "elements": [{"textRun": {"content": "Tab one.\n"}}]
                            }
                        }
                    ]
                }
            },
            "childTabs": [
                {
                    "tabProperties": {"tabId": "t1c1", "title": "Child"},
                    "documentTab": {
                        "body": {
                            "content": [
                                {
                                    "paragraph": {
                                        "elements": [
                                            {"textRun": {"content": "Child tab.\n"}}
                                        ]
                                    }
                                }
                            ]
                        }
                    },
                }
            ],
        },
        {
            "tabProperties": {"tabId": "t2", "title": "Second"},
            "documentTab": {
                "body": {
                    "content": [
                        {
                            "paragraph": {
                                "elements": [{"textRun": {"content": "Tab two.\n"}}]
                            }
                        }
                    ]
                }
            },
        },
    ],
}

_EXPECTED_TABBED_TEXT = "Tab one.\nChild tab.\nTab two.\n"

_DRIVE_FILES = [
    {
        "id": "d1",
        "name": "Doc One",
        "modifiedTime": "2026-01-01T00:00:00Z",
        "webViewLink": "https://docs.google.com/document/d/d1",
    }
]


class _DocsRequestHandler(BaseHTTPRequestHandler):
    """Emulates the Google Docs v1 + Drive v3 REST APIs and records requests."""

    def _authorized(self) -> bool:
        return self.headers.get("Authorization") == f"Bearer {_TOKEN}"

    def _reply(self, status: int, body: str, content_type: str = "application/json") -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _record(self, body: Any) -> None:
        parsed = urlparse(self.path)
        cast(_DocsServer, self.server).requests.append(
            {
                "method": self.command,
                "path": parsed.path,
                "query": parse_qs(parsed.query),
                "authorization": self.headers.get("Authorization", ""),
                "body": body,
            }
        )

    def _read_body(self) -> Any:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            return None

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        self._record(None)
        if not self._authorized():
            self._reply(401, json.dumps({"error": {"message": "Unauthorized"}}))
            return
        path = urlparse(self.path).path
        if path == "/documents/doc123":
            self._reply(200, json.dumps(_DOCUMENT))
        elif path == "/documents/tabbed":
            self._reply(200, json.dumps(_TABBED_DOCUMENT))
        elif path == "/documents/fail500":
            self._reply(500, json.dumps({"error": {"message": "Internal error"}}))
        elif path == "/drive/v3/files":
            self._reply(200, json.dumps({"files": _DRIVE_FILES}))
        elif path == "/plain":
            self._reply(200, "just text", content_type="text/plain")
        else:
            self._reply(404, json.dumps({"error": {"message": "Not found"}}))

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        body = self._read_body()
        self._record(body)
        if not self._authorized():
            self._reply(401, json.dumps({"error": {"message": "Unauthorized"}}))
            return
        path = urlparse(self.path).path
        if path == "/documents":
            self._reply(
                200,
                json.dumps({"documentId": "newdoc", "title": (body or {}).get("title", "")}),
            )
        elif path == "/documents/doc123:batchUpdate":
            replies = []
            for request in (body or {}).get("requests", []):
                if "replaceAllText" in request:
                    replies.append({"replaceAllText": {"occurrencesChanged": 3}})
                else:
                    replies.append({})
            self._reply(200, json.dumps({"documentId": "doc123", "replies": replies}))
        elif path == "/documents/emptyreply:batchUpdate":
            self._reply(200, json.dumps({"documentId": "emptyreply"}))
        elif path == "/documents/fail500:batchUpdate":
            self._reply(500, json.dumps({"error": {"message": "Internal error"}}))
        else:
            self._reply(404, json.dumps({"error": {"message": "Not found"}}))

    def log_message(self, *args: Any) -> None:  # type: ignore[override]
        pass


class _DocsServer(ThreadedHTTPServer):
    """ThreadedHTTPServer that records every request it receives."""

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, _DocsRequestHandler)
        self.requests: list[dict[str, Any]] = []


@pytest.fixture()
def docs_server():
    """Start the emulated Docs/Drive server on a free port; yield (base_url, server)."""
    server = _DocsServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base_url, server
    finally:
        stop_http_server(server, thread)


@pytest.fixture()
def backend(docs_server):
    """A backend pointed at the emulated server with a directly injected token."""
    base_url, server = docs_server
    b = GoogleDocsChannelBackend()
    b._token = _TOKEN
    b._base_url = base_url
    b._drive_base_url = f"{base_url}/drive/v3"
    return b, server


@pytest.fixture(autouse=True)
def _fresh_credentials():
    """Start and end every test with no persisted Google Docs token."""
    clear_google_credentials(_SERVICE)
    yield
    clear_google_credentials(_SERVICE)


def _write_synthetic_token() -> None:
    """Persist a synthetic, non-expiring OAuth2 user token for the service."""
    path = token_path(_SERVICE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "token": _SYNTHETIC_TOKEN,
                "refresh_token": "synthetic-refresh",
                "client_id": "synthetic-client-id",
                "client_secret": "synthetic-client-secret",
                "scopes": _SCOPES,
                "expiry": "2099-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )


def test_unauthenticated_agent_exposes_only_auth_tools() -> None:
    """A fresh agent is unauthenticated and exposes exactly the auth quartet."""
    agent = GoogleDocsAgent()
    assert agent.name == "Google Docs Agent"
    assert agent._is_authenticated() is False
    assert [t.__name__ for t in agent._get_tools()] == _AUTH_TOOL_NAMES


def test_synthetic_token_authenticates_new_agent() -> None:
    """A synthetic token.json makes a NEW agent authenticated without network."""
    _write_synthetic_token()
    agent = GoogleDocsAgent()
    assert agent._is_authenticated() is True
    creds = agent._backend._creds
    assert creds is not None
    assert creds.valid is True
    assert fresh_access_token(creds) == _SYNTHETIC_TOKEN
    names = {t.__name__ for t in agent._get_tools()}
    assert set(_AUTH_TOOL_NAMES) <= names
    assert _BACKEND_TOOL_NAMES <= names
    assert "poll_messages" not in names  # channel protocol method, not an LLM tool
    assert "connect" not in names


def test_clear_auth_removes_token_and_relocks() -> None:
    """clear_google_docs_auth deletes the token file and re-locks the agent."""
    _write_synthetic_token()
    agent = GoogleDocsAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    result = tools["clear_google_docs_auth"]()
    assert "cleared" in result.lower()
    assert not token_path(_SERVICE).exists()
    assert agent._is_authenticated() is False
    assert len(agent._get_tools()) == len(_AUTH_TOOL_NAMES)


def test_get_tools_module_function() -> None:
    """Module-level get_tools() returns the auth tools of a fresh agent."""
    tools = gdocs_mod.get_tools()
    assert [t.__name__ for t in tools] == _AUTH_TOOL_NAMES
    assert all(callable(t) for t in tools)


def test_connect_without_credentials_fails() -> None:
    """connect() fails cleanly when no token is persisted or injected."""
    b = GoogleDocsChannelBackend()
    assert b.connect() is False
    assert "No Google Docs credentials" in b.connection_info


def test_connect_with_persisted_token_succeeds() -> None:
    """connect() loads a persisted synthetic token into the backend."""
    _write_synthetic_token()
    b = GoogleDocsChannelBackend()
    assert b.connect() is True
    assert b._creds is not None
    assert "credentials loaded" in b.connection_info.lower()


def test_connect_with_injected_token_succeeds() -> None:
    """connect() accepts a directly injected bearer token without a token file."""
    b = GoogleDocsChannelBackend()
    b._token = _TOKEN
    assert b.connect() is True
    assert b._creds is None


def test_poll_messages_returns_empty(backend) -> None:
    """poll_messages returns no messages: the Docs REST API has no inbound stream."""
    b, server = backend
    messages, cursor = b.poll_messages("anything", "42", limit=5)
    assert messages == []
    assert cursor == "42"
    assert server.requests == []


def test_create_document(backend) -> None:
    """gdocs_create_document POSTs the title and returns the new document id."""
    b, server = backend
    result = json.loads(b.gdocs_create_document("My Notes"))
    assert result == {"ok": True, "document_id": "newdoc", "title": "My Notes"}
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("POST", "/documents")
    assert req["authorization"] == f"Bearer {_TOKEN}"
    assert req["body"] == {"title": "My Notes"}


def test_read_document_extracts_text_with_tables(backend) -> None:
    """gdocs_read_document handles a legacy body-only payload (no tabs array)."""
    b, server = backend
    result = json.loads(b.gdocs_read_document("doc123"))
    assert result == {"ok": True, "title": "Test Doc", "text": _EXPECTED_TEXT}
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("GET", "/documents/doc123")
    assert req["query"]["includeTabsContent"] == ["true"]
    assert req["authorization"] == f"Bearer {_TOKEN}"


def test_read_document_extracts_text_from_all_tabs(backend) -> None:
    """gdocs_read_document concatenates text from every tab, including child tabs."""
    b, server = backend
    result = json.loads(b.gdocs_read_document("tabbed"))
    assert result == {"ok": True, "title": "Tabbed Doc", "text": _EXPECTED_TABBED_TEXT}
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("GET", "/documents/tabbed")
    assert req["query"]["includeTabsContent"] == ["true"]


def test_append_text(backend) -> None:
    """gdocs_append_text posts an endOfSegmentLocation insertText batchUpdate."""
    b, server = backend
    result = json.loads(b.gdocs_append_text("doc123", "more text"))
    assert result == {"ok": True, "document_id": "doc123"}
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("POST", "/documents/doc123:batchUpdate")
    assert req["body"] == {
        "requests": [
            {"insertText": {"endOfSegmentLocation": {"segmentId": ""}, "text": "more text"}}
        ]
    }


def test_replace_text(backend) -> None:
    """gdocs_replace_text posts replaceAllText and returns occurrencesChanged."""
    b, server = backend
    result = json.loads(b.gdocs_replace_text("doc123", "old", "new", match_case=False))
    assert result == {"ok": True, "occurrences_changed": 3}
    req = server.requests[-1]
    assert req["body"] == {
        "requests": [
            {
                "replaceAllText": {
                    "replaceText": "new",
                    "containsText": {"text": "old", "matchCase": False},
                }
            }
        ]
    }


def test_replace_text_without_replies(backend) -> None:
    """A batchUpdate response with no replies yields occurrences_changed 0."""
    b, _ = backend
    result = json.loads(b.gdocs_replace_text("emptyreply", "a", "b"))
    assert result == {"ok": True, "occurrences_changed": 0}


def test_insert_text(backend) -> None:
    """gdocs_insert_text posts an insertText at the given body index."""
    b, server = backend
    result = json.loads(b.gdocs_insert_text("doc123", "hi", 7))
    assert result == {"ok": True, "document_id": "doc123"}
    req = server.requests[-1]
    assert req["body"] == {
        "requests": [{"insertText": {"location": {"index": 7}, "text": "hi"}}]
    }


def test_batch_update(backend) -> None:
    """gdocs_batch_update forwards a caller-supplied request list verbatim."""
    b, server = backend
    requests_json = '[{"deleteContentRange": {"range": {"startIndex": 1, "endIndex": 5}}}]'
    result = json.loads(b.gdocs_batch_update("doc123", requests_json))
    assert result == {"ok": True, "replies": [{}]}
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("POST", "/documents/doc123:batchUpdate")
    assert req["body"] == {"requests": json.loads(requests_json)}


def test_batch_update_rejects_bad_json(backend) -> None:
    """Malformed or non-array requests_json is refused without any request."""
    b, server = backend
    result = json.loads(b.gdocs_batch_update("doc123", "{not json"))
    assert result["ok"] is False
    assert "not valid JSON" in result["error"]
    result = json.loads(b.gdocs_batch_update("doc123", '{"insertText": {}}'))
    assert result == {"ok": False, "error": "requests_json must be a JSON array"}
    assert server.requests == []


def test_list_documents(backend) -> None:
    """gdocs_list_documents queries Drive for Docs files with the fields mask."""
    b, server = backend
    result = json.loads(b.gdocs_list_documents())
    assert result == {"ok": True, "files": _DRIVE_FILES}
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("GET", "/drive/v3/files")
    assert req["query"]["q"] == ["mimeType='application/vnd.google-apps.document'"]
    assert req["query"]["pageSize"] == ["20"]
    assert req["query"]["fields"] == ["files(id,name,modifiedTime,webViewLink)"]


def test_list_documents_escapes_query(backend) -> None:
    """Single quotes in the name filter are backslash-escaped in the Drive q."""
    b, server = backend
    result = json.loads(b.gdocs_list_documents("Bob's plan", max_results=5))
    assert result["ok"] is True
    req = server.requests[-1]
    assert req["query"]["q"] == [
        "mimeType='application/vnd.google-apps.document'"
        " and name contains 'Bob\\'s plan'"
    ]
    assert req["query"]["pageSize"] == ["5"]


def test_path_unsafe_document_ids_rejected(backend) -> None:
    """Path traversal attempts are refused up front — no HTTP request."""
    b, server = backend
    for attempt in (
        b.gdocs_read_document("../documents"),
        b.gdocs_read_document("a/b"),
        b.gdocs_read_document("a\\b"),
        b.gdocs_append_text("../x", "t"),
        b.gdocs_replace_text("../x", "a", "b"),
        b.gdocs_insert_text("../x", "t", 1),
        b.gdocs_batch_update("../x", "[]"),
    ):
        result = json.loads(attempt)
        assert result["ok"] is False
        assert "invalid document_id" in result["error"]
    assert server.requests == []


def test_document_id_is_encoded_as_single_path_segment(backend) -> None:
    """Special characters in document_id are percent-encoded, never path-split."""
    b, server = backend
    result = json.loads(b.gdocs_read_document("doc 1%2"))
    assert result["ok"] is False  # emulator 404s the unknown id — that's fine
    assert server.requests[-1]["path"] == "/documents/doc%201%252"


def test_unauthorized_token_returns_ok_false(docs_server) -> None:
    """A 401 from the server yields ok:false JSON from every tool — no exception."""
    base_url, _ = docs_server
    b = GoogleDocsChannelBackend()
    b._token = "wrong-token"
    b._base_url = base_url
    b._drive_base_url = f"{base_url}/drive/v3"
    for call in (
        lambda: b.gdocs_create_document("t"),
        lambda: b.gdocs_read_document("doc123"),
        lambda: b.gdocs_append_text("doc123", "t"),
        lambda: b.gdocs_replace_text("doc123", "a", "b"),
        lambda: b.gdocs_insert_text("doc123", "t", 1),
        lambda: b.gdocs_batch_update("doc123", "[]"),
        lambda: b.gdocs_list_documents(),
    ):
        result = json.loads(call())
        assert result["ok"] is False
        assert "401" in result["error"]


def test_server_error_returns_ok_false(backend) -> None:
    """A 500 from the server yields ok:false JSON, not an exception."""
    b, _ = backend
    result = json.loads(b.gdocs_read_document("fail500"))
    assert result["ok"] is False
    assert "500" in result["error"]
    result = json.loads(b.gdocs_append_text("fail500", "t"))
    assert result["ok"] is False
    assert "500" in result["error"]


def test_connection_refused_returns_ok_false() -> None:
    """Tools return ok:false when the server is unreachable — never raise."""
    b = GoogleDocsChannelBackend()
    b._token = _TOKEN
    b._base_url = "http://127.0.0.1:9"  # discard port; nothing listens
    b._drive_base_url = "http://127.0.0.1:9/drive/v3"
    for call in (
        lambda: b.gdocs_create_document("t"),
        lambda: b.gdocs_read_document("doc123"),
        lambda: b.gdocs_append_text("doc123", "t"),
        lambda: b.gdocs_replace_text("doc123", "a", "b"),
        lambda: b.gdocs_insert_text("doc123", "t", 1),
        lambda: b.gdocs_batch_update("doc123", "[]"),
        lambda: b.gdocs_list_documents(),
    ):
        result = json.loads(call())
        assert result["ok"] is False
        assert result["error"]


def test_request_returns_plain_text_body(backend) -> None:
    """_request falls back to the raw text body when the response is not JSON."""
    b, _ = backend
    result = b._request("GET", f"{b._base_url}/plain")
    assert result == {"ok": True, "result": "just text"}


def test_main_exits_with_no_args() -> None:
    """main() prints usage and exits when called with no arguments."""
    original_argv = sys.argv
    sys.argv = ["kiss-gdocs"]
    try:
        with pytest.raises(SystemExit):
            gdocs_mod.main()
    finally:
        sys.argv = original_argv
