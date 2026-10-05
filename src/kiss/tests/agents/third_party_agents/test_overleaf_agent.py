# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Overleaf channel agent.

Runs a REAL local HTTP server (stdlib ``ThreadedHTTPServer``) emulating
the Overleaf web routes, the socket.io v0.9 real-time handshake (with a
hand-rolled WebSocket upgrade and frame codec), and a dumb-HTTP Git
bridge serving a real bare repository — no mocks, patches, or fakes.
The server checks the ``overleaf_session2`` cookie on every request and
``x-csrf-token`` on mutating ones, and records every request for
verification.

Config state is isolated per pytest process because the session
conftest points ``KISS_HOME`` at a temporary directory and
``ChannelConfig.path`` resolves ``$KISS_HOME`` lazily.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import threading
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler
from io import BufferedIOBase
from pathlib import Path
from typing import Any

import pytest

import kiss.agents.third_party_agents.overleaf.overleaf_sea as overleaf_mod
from kiss.agents.third_party_agents._backend_utils import ThreadedHTTPServer, stop_http_server
from kiss.agents.third_party_agents._overleaf_realtime import flatten_tree
from kiss.agents.third_party_agents.overleaf.overleaf_sea import (
    OverleafAgent,
    OverleafChannelBackend,
    _config,
    _meta_content,
    _normalize_cookie,
)

_COOKIE = "s%3Aabc123.sig"
_GIT_TOKEN = "olp_gittoken"
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

_PROFILE = {"id": "u1", "email": "ada@example.com", "first_name": "Ada", "last_name": "Lovelace"}

_ROOT_FOLDER: dict[str, Any] = {
    "_id": "root",
    "name": "rootFolder",
    "docs": [{"_id": "d-main", "name": "main.tex"}, {"_id": "d-broken", "name": "broken.tex"}],
    "fileRefs": [{"_id": "f-img", "name": "logo.png"}, {"_id": "f-bib", "name": "refs.bib"}],
    "folders": [
        {
            "_id": "fo-ch",
            "name": "chapters",
            "docs": [{"_id": "d-intro", "name": "intro.tex"}],
            "fileRefs": [],
            "folders": [],
        }
    ],
}

_PROJECTS = [
    {
        "id": "p1",
        "name": "Thesis",
        "lastUpdated": "2026-09-01T00:00:00Z",
        "accessLevel": "owner",
        "source": "owner",
        "archived": False,
        "trashed": False,
        "owner": {"email": "ada@example.com", "firstName": "Ada", "lastName": "Lovelace"},
    },
    {"id": "p2", "name": "Old", "accessLevel": "readOnly", "archived": True, "trashed": False},
    {"id": "p3", "name": "Junk", "accessLevel": "owner", "archived": False, "trashed": True},
]
_TAGS = [{"_id": "t1", "name": "PhD", "color": "#43A7F0", "project_ids": ["p1", "p3"]}]

_LOG = "x" * 5000 + "END"

# (method, path) -> (status, payload); payload is JSON-able, bytes, a str (plain-text
# body, like upstream's res.sendStatus(200) -> "OK"), or None (empty body).
_ROUTES: dict[tuple[str, str], tuple[int, Any]] = {
    ("GET", "/user/personal_info"): (200, _PROFILE),
    ("GET", "/user/features"): (200, {"collaborators": -1, "gitBridge": True}),
    ("POST", "/project/new"): (200, {"project_id": "pnew"}),
    ("POST", "/project/p1/rename"): (200, "OK"),
    ("POST", "/project/pbad/rename"): (403, b"Forbidden"),
    ("POST", "/Project/p1/clone"): (200, {"project_id": "pclone", "name": "Copy"}),
    ("POST", "/Project/p1/archive"): (200, "OK"),
    ("DELETE", "/Project/p1/archive"): (200, "OK"),
    ("POST", "/project/p1/trash"): (200, "OK"),
    ("DELETE", "/project/p1/trash"): (200, "OK"),
    ("DELETE", "/Project/p1"): (200, "OK"),
    ("POST", "/Project/p1/restore"): (200, "OK"),
    ("POST", "/project/p1/settings"): (204, None),
    ("GET", "/Project/p1/download/zip"): (200, b"PK\x03\x04project"),
    ("GET", "/project/p1/entities"): (200, {"entities": [{"path": "/main.tex", "type": "doc"}]}),
    ("GET", "/project/pbad/entities"): (200, {"entities": [{"path": "/x.tex", "type": "doc"}]}),
    ("GET", "/Project/p1/file/f-img"): (200, b"\x89PNG\xff\xfe\x00"),
    ("GET", "/Project/p1/file/f-bib"): (200, "@book{k, title={Été}}".encode()),
    ("POST", "/project/p1/doc/d-main/rename"): (204, None),
    ("POST", "/project/p1/doc/d-intro/move"): (204, None),
    ("DELETE", "/project/p1/folder/fo-ch"): (204, None),
    ("POST", "/project/p1/compile"): (
        200,
        {
            "status": "success",
            "outputFiles": [
                {
                    "path": "output.pdf",
                    "type": "pdf",
                    "url": "/project/p1/build/b1/output/output.pdf",
                    "build": "b1",
                },
                {
                    "path": "output.log",
                    "type": "log",
                    "url": "/project/p1/build/b1/output/output.log",
                    "build": "b1",
                },
            ],
        },
    ),
    # A failed compile with only a log whose download 404s.
    ("POST", "/project/pbad/compile"): (
        200,
        {
            "status": "failure",
            "outputFiles": [{"path": "output.log", "type": "log", "url": "/project/pbad/gone.log"}],
        },
    ),
    ("GET", "/project/p1/build/b1/output/output.pdf"): (200, b"%PDF-1.5 fake"),
    ("GET", "/project/p1/build/b1/output/output.log"): (200, _LOG.encode()),
    ("GET", "/project/p1/wordcount"): (200, {"texcount": {"textWords": 42}}),
    ("DELETE", "/project/p1/output"): (200, "OK"),
    ("GET", "/project/p1/members"): (200, {"members": [{"_id": "u2", "email": "bob@example.com"}]}),
    ("GET", "/project/p1/invites"): (200, {"invites": [{"_id": "i1", "email": "eve@example.com"}]}),
    ("GET", "/project/pbad/members"): (200, {"members": []}),
    ("GET", "/project/pbad/invites"): (403, b"Forbidden"),
    ("POST", "/project/p1/invite"): (200, {"invite": {"_id": "i2"}}),
    ("PUT", "/project/p1/users/u2"): (204, None),
    ("DELETE", "/project/p1/users/u2"): (204, None),
    ("DELETE", "/project/p1/invite/i1"): (204, None),
    ("GET", "/project/p1/tokens"): (200, {"readOnly": "ro123", "readAndWrite": "789rw"}),
    ("GET", "/project/pbad/tokens"): (200, {}),
    ("POST", "/project/p1/leave"): (204, None),
    ("POST", "/project/p1/transfer-ownership"): (204, None),
    ("GET", "/project/p1/messages"): (200, [{"id": "m1", "content": "hi"}]),
    ("POST", "/project/p1/messages"): (204, None),
    ("GET", "/tag"): (200, _TAGS),
    ("POST", "/tag"): (200, {"_id": "t2", "name": "New"}),
    ("POST", "/tag/t1/edit"): (204, None),
    ("DELETE", "/tag/t1"): (204, None),
    ("POST", "/tag/t1/project/p1"): (204, None),
    ("DELETE", "/tag/t1/project/p1"): (204, None),
    ("GET", "/project/p1/updates"): (
        200,
        {"updates": [{"fromV": 1, "toV": 3}], "nextBeforeTimestamp": 9},
    ),
    ("GET", "/project/p1/labels"): (200, [{"id": "l1", "comment": "v1", "version": 3}]),
    ("POST", "/project/p1/labels"): (200, {"id": "l2", "comment": "draft", "version": 3}),
    ("DELETE", "/project/p1/labels/l1"): (204, None),
    ("GET", "/project/p1/diff"): (200, {"diff": [{"u": "a"}, {"i": "b"}]}),
    ("GET", "/project/p1/version/3/zip"): (200, b"PK\x03\x04v3"),
    ("POST", "/project/p1/restore_file"): (200, {"type": "doc", "id": "d-main"}),
    ("POST", "/project/p1/revert-project"): (204, None),
    ("GET", "/notifications"): (200, [{"_id": "n1", "templateKey": "notification_project_invite"}]),
}


def _dashboard_html(csrf: str) -> bytes:
    """Render the project dashboard page with the metas Overleaf embeds."""
    blob = json.dumps({"totalSize": 3, "projects": _PROJECTS}).replace('"', "&quot;")
    tags = json.dumps(_TAGS).replace('"', "&quot;")
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        f'<meta name="ol-csrfToken" content="{csrf}">'
        f'<meta name="ol-prefetchedProjectsBlob" data-type="json" content="{blob}">'
        f'<meta name="ol-tags" data-type="json" content="{tags}">'
        "</head><body>Projects</body></html>"
    ).encode()


def _find_folder(folder: dict[str, Any], folder_id: str) -> dict[str, Any]:
    """Return the folder with *folder_id* in the tree rooted at *folder*."""
    if folder["_id"] == folder_id:
        return folder
    for child in folder["folders"]:
        try:
            return _find_folder(child, folder_id)
        except LookupError:
            pass
    raise LookupError(folder_id)


def _ws_send(wfile: BufferedIOBase, opcode: int, payload: bytes) -> None:
    """Write one unmasked server WebSocket frame."""
    size = len(payload)
    if size < 126:
        header = bytes([0x80 | opcode, size])
    else:
        header = bytes([0x80 | opcode, 126]) + size.to_bytes(2, "big")
    wfile.write(header + payload)
    wfile.flush()


def _ws_recv(rfile: BufferedIOBase) -> tuple[int, bytes]:
    """Read one (masked) client WebSocket frame; return (opcode, payload)."""
    first, second = rfile.read(2)
    size = second & 0x7F
    if size == 126:
        size = int.from_bytes(rfile.read(2), "big")
    mask = rfile.read(4)
    data = rfile.read(size)
    return first & 0x0F, bytes(byte ^ mask[i % 4] for i, byte in enumerate(data))


class _OverleafServer(ThreadedHTTPServer):
    """The emulator's server, carrying the recorded traffic and behaviour switches."""

    requests: list[dict[str, Any]]
    uploads: list[dict[str, Any]]
    ws_received: list[str]
    csrf: str
    dev_csrf: bool
    login_page: bool
    rate_limited: bool
    empty_profile: bool
    features_fail: bool
    unauthorized_401: bool
    handshake_rejected: bool
    git_root: Path
    tree: dict[str, Any]
    contents: dict[str, bytes]


class _OverleafHandler(BaseHTTPRequestHandler):
    """Emulates the Overleaf web app, real-time service, and Git bridge."""

    server: Any

    def log_message(self, format: str, *args: Any) -> None:
        """Silence per-request logging."""

    def do_GET(self) -> None:
        """Handle GET."""
        self._handle("GET")

    def do_POST(self) -> None:
        """Handle POST."""
        self._handle("POST")

    def do_PUT(self) -> None:
        """Handle PUT."""
        self._handle("PUT")

    def do_DELETE(self) -> None:
        """Handle DELETE."""
        self._handle("DELETE")

    def _send(
        self, status: int, body: bytes = b"", ctype: str = "application/json", location: str = ""
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if location:
            self.send_header("Location", location)
        self.end_headers()
        self.wfile.write(body)

    def _handle(self, method: str) -> None:
        srv = self.server
        path, _, raw_query = self.path.partition("?")
        query = dict(part.split("=", 1) for part in raw_query.split("&") if "=" in part)
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        srv.requests.append(
            {
                "method": method,
                "path": path,
                "query": query,
                "headers": dict(self.headers),
                "body": body,
            }
        )
        if path.startswith("/git/"):
            return self._git(path)
        if srv.rate_limited:
            return self._send(429, b"Too Many Requests", "text/plain")
        if f"overleaf_session2={_COOKIE}" not in (self.headers.get("Cookie") or ""):
            if srv.unauthorized_401 or "json" in (self.headers.get("Accept") or ""):
                return self._send(401, b"Unauthorized", "text/plain")
            return self._send(302, b"Found", "text/plain", location="/login")
        if path == "/dev/csrf":
            if srv.dev_csrf:
                return self._send(200, srv.csrf.encode(), "text/plain")
            return self._send(404, b"Not found", "text/plain")
        if method != "GET" and self.headers.get("x-csrf-token") != srv.csrf:
            return self._send(403, b"Forbidden", "text/plain")
        if path == "/project" and method == "GET":
            html = (
                b"<html><body>Log in</body></html>" if srv.login_page else _dashboard_html(srv.csrf)
            )
            return self._send(200, html, "text/html")
        if path == "/socket.io/1/":
            if srv.handshake_rejected and srv.unauthorized_401:
                return self._send(401, b"Unauthorized", "text/plain")
            if srv.handshake_rejected:
                return self._send(302, b"Found", "text/plain", location="/login")
            if query.get("projectId") in ("p1", "pbad"):
                return self._send(
                    200, f"sid-{query['projectId']}:60:60:websocket".encode(), "text/plain"
                )
            return self._send(404, b"no such project", "text/plain")
        if path.startswith("/socket.io/1/websocket/"):
            return self._websocket(query["projectId"])
        if path == "/user/personal_info" and srv.empty_profile:
            return self._send(200, b"{}")
        if path == "/user/features" and srv.features_fail:
            return self._send(500, b"boom", "text/plain")
        if path == "/project/p1/wordcount" and query.get("file") == "notjson":
            return self._send(200, b"oops", "text/plain")
        if path in ("/project/new/upload", "/Project/p1/upload"):
            return self._upload(path, query, body)
        if path == "/project/p1/folder":
            request = json.loads(body)
            if request["name"] == "forbidden":
                return self._send(400, b"invalid folder name", "text/plain")
            folder = {"_id": f"fo-{request['name']}", "name": request["name"]}
            _find_folder(srv.tree, request["parent_folder_id"])["folders"].append(
                {**folder, "docs": [], "fileRefs": [], "folders": []}
            )
            return self._send(200, json.dumps(folder).encode())
        doc_id = path.removeprefix("/Project/p1/doc/").removesuffix("/download")
        if method == "GET" and doc_id in srv.contents:
            return self._send(200, srv.contents[doc_id], "application/octet-stream")
        status, payload = _ROUTES.get((method, path), (404, b"Not found"))
        if payload is None:
            return self._send(status)
        if isinstance(payload, str):
            return self._send(status, payload.encode(), "text/plain")
        if isinstance(payload, bytes):
            return self._send(status, payload, "application/octet-stream")
        return self._send(status, json.dumps(payload).encode())

    def _upload(self, path: str, query: dict[str, str], body: bytes) -> None:
        """Parse a multipart upload and record its fields."""
        message = BytesParser(policy=HTTP).parsebytes(
            b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body
        )
        fields: dict[str, Any] = {}
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            fields[str(name)] = {
                "filename": part.get_filename(),
                "content": part.get_payload(decode=True),
            }
        self.server.uploads.append({"path": path, "query": query, "fields": fields})
        if path == "/project/new/upload":
            return self._send(200, b'{"success": true, "project_id": "pzip"}')
        # Like Overleaf, a same-named entity in the folder is replaced in place.
        is_doc = fields["type"]["content"] == b"text/plain"
        kind, key = ("doc", "docs") if is_doc else ("file", "fileRefs")
        name = fields["qqfile"]["filename"]
        entities = _find_folder(self.server.tree, query["folder_id"])[key]
        entity = next((e for e in entities if e["name"] == name), None)
        if entity is None:
            entity = {"_id": f"{kind[0]}-{len(self.server.uploads)}", "name": name}
            entities.append(entity)
        self.server.contents[entity["_id"]] = fields["qqfile"]["content"]
        result = {"success": True, "entity_id": entity["_id"], "entity_type": kind}
        return self._send(200, json.dumps(result).encode())

    def _websocket(self, project_id: str) -> None:
        """Upgrade to WebSocket and play the socket.io joinProject exchange."""
        key = self.headers["Sec-WebSocket-Key"]
        accept = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest())
        self.wfile.write(
            b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
            b"Connection: Upgrade\r\nSec-WebSocket-Accept: " + accept + b"\r\n\r\n"
        )
        if project_id == "p1":
            join = {
                "name": "joinProjectResponse",
                "args": [{"project": {"_id": "p1", "rootFolder": [self.server.tree]}}],
            }
            frames = [
                "1::",
                "2::",
                '5:::{"name":"connectionAccepted","args":[]}',
                "5:::" + json.dumps(join),
            ]
        else:
            frames = ["1::", "7:::1+0"]
        for frame in frames:
            _ws_send(self.wfile, 1, frame.encode())
        while True:
            opcode, data = _ws_recv(self.rfile)
            if opcode == 8:
                _ws_send(self.wfile, 8, data)
                break
            self.server.ws_received.append(data.decode())
        self.close_connection = True

    def _git(self, path: str) -> None:
        """Serve the bare repo over dumb HTTP behind Basic auth."""
        expected = "Basic " + base64.b64encode(f"git:{_GIT_TOKEN}".encode()).decode()
        if self.headers.get("Authorization") != expected:
            return self._send(401, b"auth required", "text/plain")
        target = self.server.git_root / path.removeprefix("/git/p1/")
        if not path.startswith("/git/p1/") or not target.is_file():
            return self._send(404, b"Not found", "text/plain")
        return self._send(200, target.read_bytes(), "application/octet-stream")


@pytest.fixture()
def overleaf_server():
    """Start the emulated Overleaf server; yield (base_url, server)."""
    server = _OverleafServer(("127.0.0.1", 0), _OverleafHandler)
    server.requests = []
    server.uploads = []
    server.ws_received = []
    server.csrf = "tok-1"
    server.dev_csrf = True
    server.login_page = False
    server.rate_limited = False
    server.empty_profile = False
    server.features_fail = False
    server.unauthorized_401 = False
    server.handshake_rejected = False
    server.git_root = Path("/nonexistent")
    server.tree = copy.deepcopy(_ROOT_FOLDER)
    server.contents = {"d-main": b"\\documentclass{article}"}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", server
    finally:
        stop_http_server(server, thread)


@pytest.fixture()
def backend(overleaf_server):
    """A backend pointed at the emulated server with the valid cookie."""
    base_url, server = overleaf_server
    b = OverleafChannelBackend()
    b._configure(_COOKIE, _GIT_TOKEN, base_url + "/")
    return b, server


@pytest.fixture(autouse=True)
def _fresh_config():
    """Start and end every test with no persisted Overleaf config."""
    _config.clear()
    yield
    _config.clear()


def _auth_tools(agent: OverleafAgent) -> dict[str, Any]:
    return {t.__name__: t for t in agent._get_auth_tools()}


def _requests_to(server: Any, method: str, path: str) -> list[dict[str, Any]]:
    return [r for r in server.requests if r["method"] == method and r["path"] == path]


# ------------------------------------------------------------------ helpers


def test_normalize_cookie_and_meta_parsing() -> None:
    """Pasted cookies are normalised; meta tags are found and unescaped."""
    assert _normalize_cookie(f"  overleaf_session2={_COOKIE}  ") == _COOKIE
    assert _normalize_cookie(f'"overleaf_session2={_COOKIE}; GCLB=zz"') == _COOKIE
    assert _normalize_cookie(f"'{_COOKIE}'") == _COOKIE
    assert _normalize_cookie("   ") == ""
    html = _dashboard_html("abc").decode()
    assert _meta_content(html, "ol-csrfToken") == "abc"
    assert json.loads(_meta_content(html, "ol-tags") or "")[0]["name"] == "PhD"
    assert _meta_content(html, "missing") is None


def test_flatten_tree_paths() -> None:
    """flatten_tree yields docs, files, and nested folders with parent ids."""
    entries = {e["path"]: e for e in flatten_tree(_ROOT_FOLDER)}
    assert entries["main.tex"] == {
        "path": "main.tex",
        "type": "doc",
        "id": "d-main",
        "parent_folder_id": "root",
    }
    assert entries["logo.png"]["type"] == "file"
    assert entries["chapters"]["type"] == "folder"
    assert entries["chapters/intro.tex"]["parent_folder_id"] == "fo-ch"


def test_git_remote_url_forms() -> None:
    """overleaf.com hosts use git.overleaf.com; other hosts use <host>/git."""
    b = OverleafChannelBackend()
    assert b._git_remote_url("abc") == "https://git.overleaf.com/abc"
    b._configure(_COOKIE, "", "https://overleaf.com/")
    assert b._host == "https://overleaf.com"
    assert b._git_remote_url("abc") == "https://git.overleaf.com/abc"
    b._configure(_COOKIE, "", "https://latex.example.edu/")
    assert b._git_remote_url("abc") == "https://latex.example.edu/git/abc"


# ------------------------------------------------------------ auth + agent


def test_unconfigured_agent_and_instructions() -> None:
    """A fresh agent exposes only the auth trio and explains the cookie flow."""
    agent = OverleafAgent()
    assert agent.name == "Overleaf Agent"
    assert agent._is_authenticated() is False
    assert [t.__name__ for t in agent._get_tools()] == [
        "check_overleaf_auth",
        "authenticate_overleaf",
        "clear_overleaf_auth",
    ]
    msg = _auth_tools(agent)["check_overleaf_auth"]()
    assert "DevTools" in msg and "overleaf_session2" in msg and "no OAuth" in msg
    assert "authenticate_overleaf" in msg
    backend = OverleafChannelBackend()
    assert backend.connect() is False
    assert backend.connection_info == "No Overleaf config found."


def test_authenticate_verifies_saves_and_clears(overleaf_server) -> None:
    """authenticate_overleaf verifies the cookie, saves 0600 config, unlocks tools."""
    base_url, server = overleaf_server
    agent = OverleafAgent()
    tools = _auth_tools(agent)
    result = json.loads(
        tools["authenticate_overleaf"](f' "overleaf_session2={_COOKIE}" ', " gt ", base_url + "/")
    )
    assert result == {"ok": True, "message": "Overleaf configured for ada@example.com."}
    saved = json.loads(_config.path.read_text())
    assert saved == {"session_cookie": _COOKIE, "git_token": "gt", "host": base_url}
    if sys.platform != "win32":
        assert stat.S_IMODE(_config.path.stat().st_mode) == 0o600
    assert agent._is_authenticated() is True
    assert agent._backend._git_token == "gt"
    names = {t.__name__ for t in agent._get_tools()}
    assert {"overleaf_whoami", "overleaf_git_sync", "overleaf_list_projects"} <= names
    assert "connect" not in names
    checked = json.loads(tools["check_overleaf_auth"]())
    assert checked == {
        "ok": True,
        "email": "ada@example.com",
        "name": "Ada Lovelace",
        "host": base_url,
    }

    # A new agent (and the module-level tools()) loads the persisted config.
    fresh = OverleafAgent()
    assert fresh._backend._host == base_url
    assert fresh._backend.connection_info == f"Overleaf session configured for {base_url}."
    assert len(overleaf_mod.add_to_tools()) > 3

    assert "cleared" in tools["clear_overleaf_auth"]()
    assert not _config.path.exists()
    assert agent._is_authenticated() is False
    assert len(overleaf_mod.add_to_tools()) == 3


def test_authenticate_and_use_localhost_host(overleaf_server) -> None:
    """A self-hosted http://localhost:PORT instance receives the session cookie."""
    base_url, server = overleaf_server
    local_url = base_url.replace("127.0.0.1", "localhost")
    agent = OverleafAgent()
    result = json.loads(_auth_tools(agent)["authenticate_overleaf"](_COOKIE, "", local_url))
    assert result == {"ok": True, "message": "Overleaf configured for ada@example.com."}
    assert json.loads(agent._backend.overleaf_read_file("p1", "main.tex"))["ok"] is True
    assert json.loads(agent._backend.overleaf_leave_project("p1"))["ok"] is True
    assert all(f"overleaf_session2={_COOKIE}" in r["headers"]["Cookie"] for r in server.requests)


def test_authenticate_rejections(overleaf_server) -> None:
    """Bad cookies, hosts, and profiles are rejected and nothing is saved."""
    base_url, server = overleaf_server
    tools = _auth_tools(OverleafAgent())
    auth = tools["authenticate_overleaf"]
    assert "cannot be empty" in auth("  ")
    assert "http://" in auth(_COOKIE, "", "ftp://example.com")
    expired = json.loads(auth("wrong-cookie", "", base_url))
    assert expired["ok"] is False and "expired" in expired["error"]
    server.empty_profile = True
    assert "did not return a user profile" in auth(_COOKIE, "", base_url)
    assert not _config.path.exists()


def test_authenticate_save_failure(overleaf_server) -> None:
    """A config path blocked by a file yields ok:false and no authentication."""
    base_url, _ = overleaf_server
    agent = OverleafAgent()
    config_dir = _config.path.parent
    if config_dir.is_dir():
        shutil.rmtree(config_dir)
    config_dir.parent.mkdir(parents=True, exist_ok=True)
    config_dir.write_text("blocks the directory")
    try:
        result = json.loads(_auth_tools(agent)["authenticate_overleaf"](_COOKIE, "", base_url))
        assert result["ok"] is False and "could not save config" in result["error"]
        assert agent._is_authenticated() is False
    finally:
        config_dir.unlink()


def test_check_auth_reports_expired_session(overleaf_server) -> None:
    """check_overleaf_auth reports an expired cookie from a live request."""
    base_url, _ = overleaf_server
    _config.save({"session_cookie": "stale", "git_token": "", "host": base_url})
    agent = OverleafAgent()
    result = json.loads(_auth_tools(agent)["check_overleaf_auth"]())
    assert result["ok"] is False and "expired" in result["error"]


def test_main_help(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """main() wires channel_main with poll mode disabled."""
    monkeypatch.setattr(sys, "argv", ["kiss-overleaf", "--help"])
    with pytest.raises(SystemExit) as exc:
        overleaf_mod.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "kiss-overleaf" in out and "--channel" not in out


# ---------------------------------------------------------------- transport


def test_request_error_mapping(backend, refusing_port) -> None:
    """401, 302->/login, 429, other HTTP errors, bad paths, and network errors."""
    b, server = backend
    stale = OverleafChannelBackend()
    stale._configure("stale", "", b._host)
    assert "expired" in json.loads(stale.overleaf_whoami())["error"]
    assert "expired" in json.loads(stale.overleaf_list_projects())["error"]
    assert json.loads(b.overleaf_word_count("pnone"))["error"].startswith("HTTP 404")
    assert "invalid path segment" in b.overleaf_rename_project("..", "x")
    assert "non-JSON" in b.overleaf_word_count("p1", "notjson")
    server.rate_limited = True
    assert "rate limit" in b.overleaf_whoami()
    server.rate_limited = False
    server.features_fail = True
    assert "HTTP 500" in b.overleaf_whoami()
    dead = OverleafChannelBackend()
    dead._configure(_COOKIE, "", f"http://127.0.0.1:{refusing_port}")
    assert json.loads(dead.overleaf_whoami())["ok"] is False
    assert json.loads(dead.overleaf_leave_project("p1"))["ok"] is False


def test_csrf_refresh_retry_and_fallback(backend) -> None:
    """A rotated token is refreshed once; /dev/csrf falls back to the meta tag."""
    b, server = backend
    assert json.loads(b.overleaf_leave_project("p1"))["ok"] is True
    assert b._csrf == "tok-1"
    server.csrf = "tok-2"
    server.dev_csrf = False
    assert json.loads(b.overleaf_leave_project("p1"))["ok"] is True
    assert b._csrf == "tok-2"  # re-fetched from the dashboard's ol-csrfToken meta
    posts = _requests_to(server, "POST", "/project/p1/leave")
    assert [p["headers"]["x-csrf-token"] for p in posts] == ["tok-1", "tok-1", "tok-2"]
    assert all(p["headers"]["Referer"].endswith("/project") for p in posts)

    # A persistent 403 is retried exactly once and then reported.
    before = len(_requests_to(server, "POST", "/project/pbad/rename"))
    assert "HTTP 403" in b.overleaf_rename_project("pbad", "x")
    assert len(_requests_to(server, "POST", "/project/pbad/rename")) == before + 2

    # No token source at all: the request goes out without one and fails.
    server.login_page = True
    b._csrf = ""
    assert "HTTP 403" in b.overleaf_leave_project("p1")


# ---------------------------------------------------------------- projects


def test_whoami_and_list_projects(backend) -> None:
    """Profile + features; dashboard projects with tags and filters."""
    b, server = backend
    me = json.loads(b.overleaf_whoami())["result"]
    assert me["email"] == "ada@example.com" and me["features"]["gitBridge"] is True
    listed = json.loads(b.overleaf_list_projects())
    assert listed["totalSize"] == 3
    assert [p["id"] for p in listed["projects"]] == ["p1"]
    p1 = listed["projects"][0]
    assert p1["tags"] == ["PhD"] and p1["owner"] == {
        "email": "ada@example.com",
        "name": "Ada Lovelace",
    }
    everything = json.loads(b.overleaf_list_projects(include_archived=True, include_trashed=True))
    assert [p["id"] for p in everything["projects"]] == ["p1", "p2", "p3"]
    tagged = json.loads(b.overleaf_list_projects(include_trashed=True, tag="PhD"))
    assert [p["id"] for p in tagged["projects"]] == ["p1", "p3"]
    assert _requests_to(server, "GET", "/project")[0]["headers"]["Accept"] == "text/html"
    server.login_page = True
    assert "expired" in b.overleaf_list_projects()


def test_project_lifecycle_tools(backend, tmp_path: Path) -> None:
    """Create, zip upload, rename, clone, state changes, settings, zip download."""
    b, server = backend
    assert json.loads(b.overleaf_create_project("New"))["result"] == {"project_id": "pnew"}
    assert json.loads(_requests_to(server, "POST", "/project/new")[0]["body"]) == {
        "projectName": "New",
        "template": "none",
    }
    archive = tmp_path / "paper.zip"
    archive.write_bytes(b"PK\x03\x04zipdata")
    assert (
        json.loads(b.overleaf_upload_project_zip("Paper", str(archive)))["result"]["project_id"]
        == "pzip"
    )
    fields = server.uploads[-1]["fields"]
    assert fields["qqfile"] == {"filename": "Paper.zip", "content": b"PK\x03\x04zipdata"}
    assert fields["name"]["content"] == b"Paper"
    assert "could not read" in b.overleaf_upload_project_zip("X", str(tmp_path / "missing.zip"))

    # Upstream answers rename and state changes with res.sendStatus(200): body "OK".
    assert json.loads(b.overleaf_rename_project("p1", "Renamed")) == {"ok": True, "status": 200}
    assert json.loads(_requests_to(server, "POST", "/project/p1/rename")[0]["body"]) == {
        "newProjectName": "Renamed"
    }
    assert json.loads(b.overleaf_clone_project("p1", "Copy"))["result"]["project_id"] == "pclone"
    assert json.loads(_requests_to(server, "POST", "/Project/p1/clone")[0]["body"]) == {
        "projectName": "Copy"
    }
    for action in ("archive", "unarchive", "trash", "untrash", "delete", "restore"):
        assert json.loads(b.overleaf_set_project_state("p1", action)) == {
            "ok": True,
            "status": 200,
        }, action
    assert _requests_to(server, "DELETE", "/Project/p1")
    assert "HTTP 404" in b.overleaf_set_project_state("pnone", "archive")
    assert "action must be one of" in b.overleaf_set_project_state("p1", "explode")

    assert json.loads(b.overleaf_update_project_settings("p1", compiler="xelatex"))["ok"] is True
    assert json.loads(_requests_to(server, "POST", "/project/p1/settings")[0]["body"]) == {
        "compiler": "xelatex"
    }
    assert "nothing to update" in b.overleaf_update_project_settings("p1")

    out = tmp_path / "p1.zip"
    saved = json.loads(b.overleaf_download_project_zip("p1", str(out)))
    assert saved["bytes"] == len(b"PK\x03\x04project") and out.read_bytes() == b"PK\x03\x04project"
    assert "could not write" in b.overleaf_download_project_zip(
        "p1", str(tmp_path / "no" / "dir.zip")
    )
    assert "HTTP 404" in b.overleaf_download_project_zip("pnone", str(out))


# ------------------------------------------------------------------- files


def test_list_files_realtime_and_fallback(backend) -> None:
    """The socket.io tree gives ids; a real-time error falls back to /entities."""
    b, server = backend
    listed = json.loads(b.overleaf_list_files("p1"))["result"]
    assert listed["root_folder_id"] == "root"
    assert {"path": "chapters/intro.tex", "type": "doc", "id": "d-intro"} in listed["entries"]
    assert "2::" in server.ws_received  # heartbeat answered
    handshake = _requests_to(server, "GET", "/socket.io/1/")[0]
    assert handshake["query"]["projectId"] == "p1"
    upgrade = [r for r in server.requests if r["path"].startswith("/socket.io/1/websocket/sid-p1")]
    assert f"overleaf_session2={_COOKIE}" in upgrade[0]["headers"]["Cookie"]

    fallback = json.loads(b.overleaf_list_files("pbad"))["result"]
    assert "entity ids unavailable" in fallback["note"] and "7:::1+0" in fallback["note"]
    assert fallback["entities"] == [{"path": "/x.tex", "type": "doc"}]
    assert "HTTP 404" in b.overleaf_list_files("pnone")


def test_read_and_download_files(backend, tmp_path: Path) -> None:
    """Docs and UTF-8 files are read; binaries, folders, and unknown paths are refused."""
    b, _ = backend
    doc = json.loads(b.overleaf_read_file("p1", "/main.tex"))
    assert doc == {
        "ok": True,
        "path": "main.tex",
        "content": "\\documentclass{article}",
        "truncated": False,
    }
    assert json.loads(b.overleaf_read_file("p1", "refs.bib"))["content"] == "@book{k, title={Été}}"
    assert "is binary (7 bytes)" in b.overleaf_read_file("p1", "logo.png")
    assert "is a folder" in b.overleaf_read_file("p1", "chapters")
    assert "no such file" in b.overleaf_read_file("p1", "missing.tex")
    assert "no such file" in b.overleaf_read_file("p1", "  ")
    assert "HTTP 404" in b.overleaf_read_file("p1", "broken.tex")

    out = tmp_path / "logo.png"
    assert json.loads(b.overleaf_download_file("p1", "logo.png", str(out)))["bytes"] == 7
    assert out.read_bytes() == b"\x89PNG\xff\xfe\x00"
    assert "HTTP 404" in b.overleaf_download_file("p1", "broken.tex", str(out))
    assert "no such file" in b.overleaf_download_file("p1", "nope.png", str(out))


def test_write_and_upload_files(backend, tmp_path: Path) -> None:
    """Writes create missing folders and upload multipart into the right folder."""
    b, server = backend
    result = json.loads(b.overleaf_write_file("p1", "newdir/sub/x.tex", "Hello"))
    assert result["result"] == {"success": True, "entity_id": "d-1", "entity_type": "doc"}
    folder_posts = [
        json.loads(r["body"]) for r in _requests_to(server, "POST", "/project/p1/folder")
    ]
    assert folder_posts == [
        {"name": "newdir", "parent_folder_id": "root"},
        {"name": "sub", "parent_folder_id": "fo-newdir"},
    ]
    upload = server.uploads[-1]
    assert upload["query"] == {"folder_id": "fo-sub"}
    assert upload["fields"]["qqfile"] == {"filename": "x.tex", "content": b"Hello"}
    assert upload["fields"]["relativePath"]["content"] == b"null"
    assert upload["fields"]["type"]["content"] == b"text/plain"

    b.overleaf_write_file("p1", "chapters/two.tex", "2")
    assert server.uploads[-1]["query"] == {"folder_id": "fo-ch"}
    assert len(_requests_to(server, "POST", "/project/p1/folder")) == 2  # existing folder reused
    assert "must name a file" in b.overleaf_write_file("p1", "chapters/", "x")
    assert "invalid folder name" in b.overleaf_write_file("p1", "chapters/forbidden/x.tex", "x")

    local = tmp_path / "plot.pdf"
    local.write_bytes(b"%PDF plot")
    assert json.loads(b.overleaf_upload_file("p1", str(local), "figures"))["ok"] is True
    assert server.uploads[-1]["query"] == {"folder_id": "fo-figures"}
    assert server.uploads[-1]["fields"]["qqfile"] == {
        "filename": "plot.pdf",
        "content": b"%PDF plot",
    }
    assert json.loads(b.overleaf_upload_file("p1", str(local)))["ok"] is True
    assert server.uploads[-1]["query"] == {"folder_id": "root"}
    assert json.loads(b.overleaf_upload_file("p1", str(tmp_path / "gone.pdf")))["ok"] is False

    assert json.loads(b.overleaf_create_folder("p1", "figures/plots")) == {
        "ok": True,
        "folder_id": "fo-plots",
    }
    assert "7:::" in b.overleaf_create_folder("pbad", "x")


def test_write_overwrite_round_trip_and_folder_reuse(backend) -> None:
    """A rewrite replaces the doc in place, reads back, and reuses created folders."""
    b, server = backend
    first = json.loads(b.overleaf_write_file("p1", "notes/draft.tex", "one"))["result"]
    assert json.loads(b.overleaf_read_file("p1", "notes/draft.tex"))["content"] == "one"
    second = json.loads(b.overleaf_write_file("p1", "notes/draft.tex", "two"))["result"]
    assert second["entity_id"] == first["entity_id"]
    assert json.loads(b.overleaf_read_file("p1", "notes/draft.tex"))["content"] == "two"
    assert json.loads(b.overleaf_create_folder("p1", "notes")) == {
        "ok": True,
        "folder_id": "fo-notes",
    }
    assert len(_requests_to(server, "POST", "/project/p1/folder")) == 1  # created once
    assert [e["path"] for e in flatten_tree(server.tree)].count("notes/draft.tex") == 1

    # Overwriting an original doc keeps its id and serves the new content.
    assert json.loads(b.overleaf_write_file("p1", "main.tex", "new main"))["result"] == {
        "success": True,
        "entity_id": "d-main",
        "entity_type": "doc",
    }
    assert json.loads(b.overleaf_read_file("p1", "main.tex"))["content"] == "new main"


def test_read_file_caps_serialized_output(backend) -> None:
    """Unicode and escape-heavy docs are cut so the whole JSON stays within 8000 chars."""
    b, _ = backend
    for text in ("\u4e2d" * 9000, '"\\\n\x01' * 3000):
        b.overleaf_write_file("p1", "big.tex", text)
        raw = b.overleaf_read_file("p1", "big.tex")
        assert len(raw) <= 8000
        doc = json.loads(raw)
        assert doc["truncated"] is True and doc["content"]
        assert text.startswith(doc["content"])
        longer = {**doc, "content": text[: len(doc["content"]) + 1]}
        assert len(json.dumps(longer, ensure_ascii=False)) > 8000  # longest fitting prefix
    b.overleaf_write_file("p1", "small.tex", "\u4e2d" * 100)
    small = json.loads(b.overleaf_read_file("p1", "small.tex"))
    assert small["truncated"] is False and small["content"] == "\u4e2d" * 100


@pytest.mark.parametrize("unauthorized_401", [False, True])
def test_list_files_expired_handshake_skips_entities_fallback(backend, unauthorized_401) -> None:
    """Only the handshake is refused: list_files reports the expired session, not /entities."""
    b, server = backend
    server.handshake_rejected = True
    server.unauthorized_401 = unauthorized_401
    assert json.loads(b.overleaf_list_files("p1")) == {"ok": False, "error": overleaf_mod._EXPIRED}
    assert _requests_to(server, "GET", "/socket.io/1/")
    assert not _requests_to(server, "GET", "/project/p1/entities")


def test_read_file_normalises_path_and_caps_oversized_path(backend) -> None:
    """The echoed path is normalised; a path too long to echo yields a short error."""
    b, _ = backend
    doc = json.loads(b.overleaf_read_file("p1", " //main.tex/ "))
    assert doc["path"] == "main.tex" and doc["content"] == "\\documentclass{article}"
    raw = b.overleaf_read_file("p1", "/" * 8000 + "main.tex")
    assert len(raw) <= 8000 and json.loads(raw)["path"] == "main.tex"
    b.overleaf_create_folder("p1", "d" * 7990)
    b.overleaf_write_file("p1", "d" * 7990 + "/x.tex", "hello")
    raw = b.overleaf_read_file("p1", "d" * 7990 + "/x.tex")
    assert len(raw) <= 8000
    assert json.loads(raw) == {
        "ok": False,
        "error": "the file path is too long to return within the 8000-character limit",
    }


def test_read_file_truncated_flag_matches_cut(backend) -> None:
    """At the 8000-char boundary, truncated is true only when content was actually cut."""
    b, _ = backend
    b.overleaf_write_file("p1", "boundary.tex", "x" * 7929)
    raw = b.overleaf_read_file("p1", "boundary.tex")
    doc = json.loads(raw)
    assert len(raw) == 8000 and doc["truncated"] is False and doc["content"] == "x" * 7929
    b.overleaf_write_file("p1", "boundary.tex", "x" * 7930)
    raw = b.overleaf_read_file("p1", "boundary.tex")
    doc = json.loads(raw)
    assert len(raw) <= 8000 and doc["truncated"] is True
    assert doc["content"] == "x" * 7929


def test_read_empty_file_with_overlong_path_is_error(backend) -> None:
    """An empty file whose path alone overflows the limit errors instead of claiming truncation."""
    b, _ = backend
    b.overleaf_write_file("p1", "empty.tex", "")
    doc = json.loads(b.overleaf_read_file("p1", "empty.tex"))
    assert doc["ok"] is True and doc["content"] == "" and doc["truncated"] is False
    folder = "d" * 7936
    assert json.loads(b.overleaf_create_folder("p1", folder))["ok"] is True
    assert json.loads(b.overleaf_write_file("p1", f"{folder}/x.tex", ""))["ok"] is True
    assert json.loads(b.overleaf_read_file("p1", f"{folder}/x.tex")) == {
        "ok": False,
        "error": "the file path is too long to return within the 8000-character limit",
    }


@pytest.mark.parametrize("unauthorized_401", [False, True])
def test_file_tools_report_expired_session(overleaf_server, unauthorized_401, tmp_path) -> None:
    """A rejected real-time handshake (302 -> /login or 401) names authenticate_overleaf."""
    base_url, server = overleaf_server
    server.unauthorized_401 = unauthorized_401
    stale = OverleafChannelBackend()
    stale._configure("stale", "", base_url)
    local = tmp_path / "a.png"
    local.write_bytes(b"x")
    results = [
        stale.overleaf_list_files("p1"),
        stale.overleaf_read_file("p1", "main.tex"),
        stale.overleaf_download_file("p1", "main.tex", str(tmp_path / "out")),
        stale.overleaf_write_file("p1", "x.tex", "x"),
        stale.overleaf_upload_file("p1", str(local)),
        stale.overleaf_create_folder("p1", "figs"),
        stale.overleaf_rename_entity("p1", "main.tex", "y.tex"),
        stale.overleaf_move_entity("p1", "main.tex", ""),
        stale.overleaf_delete_entity("p1", "main.tex"),
    ]
    for result in results:
        assert json.loads(result) == {"ok": False, "error": overleaf_mod._EXPIRED}
    assert "authenticate_overleaf" in overleaf_mod._EXPIRED
    assert _requests_to(server, "GET", "/socket.io/1/")
    assert not any(r["path"].startswith("/socket.io/1/websocket/") for r in server.requests)


def test_rename_move_delete_entities(backend) -> None:
    """Entity tools resolve paths to typed ids and hit the right routes."""
    b, server = backend
    assert json.loads(b.overleaf_rename_entity("p1", "main.tex", "thesis.tex"))["ok"] is True
    assert json.loads(_requests_to(server, "POST", "/project/p1/doc/d-main/rename")[0]["body"]) == {
        "name": "thesis.tex"
    }
    assert json.loads(b.overleaf_move_entity("p1", "chapters/intro.tex", ""))["ok"] is True
    assert json.loads(_requests_to(server, "POST", "/project/p1/doc/d-intro/move")[0]["body"]) == {
        "folder_id": "root"
    }
    assert json.loads(b.overleaf_delete_entity("p1", "chapters"))["ok"] is True
    for result in (
        b.overleaf_rename_entity("p1", "nope", "x"),
        b.overleaf_move_entity("p1", "nope", ""),
        b.overleaf_delete_entity("p1", "nope"),
    ):
        assert "no such file" in result


# ----------------------------------------------------------------- compile


def test_compile_pdf_and_word_count(backend, tmp_path: Path) -> None:
    """Compile reports outputs and the log tail; download_pdf saves the PDF."""
    b, server = backend
    compiled = json.loads(b.overleaf_compile("p1", draft=True))
    assert compiled["status"] == "success"
    assert [f["path"] for f in compiled["outputFiles"]] == ["output.pdf", "output.log"]
    assert compiled["log_tail"].endswith("END") and len(compiled["log_tail"]) == 4000
    assert json.loads(_requests_to(server, "POST", "/project/p1/compile")[0]["body"]) == {
        "check": "silent",
        "draft": True,
        "incrementalCompilesEnabled": True,
        "rootDoc_id": "",
        "stopOnFirstError": False,
    }
    assert _requests_to(server, "POST", "/project/p1/compile")[0]["query"] == {
        "enable_pdf_caching": "true"
    }
    failed = json.loads(b.overleaf_compile("pbad"))
    assert failed["status"] == "failure" and "HTTP 404" in failed["log_tail"]
    assert "HTTP 404" in b.overleaf_compile("pnone")

    pdf = tmp_path / "out.pdf"
    assert json.loads(b.overleaf_download_pdf("p1", str(pdf)))["ok"] is True
    assert pdf.read_bytes() == b"%PDF-1.5 fake"
    assert "no PDF (status: failure)" in b.overleaf_download_pdf("pbad", str(pdf))
    assert "HTTP 404" in b.overleaf_download_pdf("pnone", str(pdf))

    assert json.loads(b.overleaf_word_count("p1"))["result"]["texcount"]["textWords"] == 42
    b.overleaf_word_count("p1", "main.tex")
    assert _requests_to(server, "GET", "/project/p1/wordcount")[-1]["query"] == {"file": "main.tex"}
    assert json.loads(b.overleaf_clear_compile_cache("p1")) == {"ok": True, "status": 200}
    assert "HTTP 404" in b.overleaf_clear_compile_cache("pnone")


# ----------------------------------------------------------- collaboration


def test_collaboration_tools(backend) -> None:
    """Members/invites, invites, privileges, removal, links, leave, transfer."""
    b, server = backend
    members = json.loads(b.overleaf_list_members("p1"))
    assert members["members"][0]["email"] == "bob@example.com"
    assert members["invites"][0]["_id"] == "i1"
    limited = json.loads(b.overleaf_list_members("pbad"))
    assert limited["invites"].startswith("unavailable:") and "HTTP 403" in limited["invites"]
    assert "HTTP 404" in b.overleaf_list_members("pnone")

    assert (
        json.loads(b.overleaf_invite_collaborator("p1", "eve@example.com", "review"))["ok"] is True
    )
    assert json.loads(_requests_to(server, "POST", "/project/p1/invite")[0]["body"]) == {
        "email": "eve@example.com",
        "privileges": "review",
    }
    assert "privileges must be one of" in b.overleaf_invite_collaborator("p1", "e@x", "owner")
    assert json.loads(b.overleaf_set_collaborator_privileges("p1", "u2", "readOnly"))["ok"] is True
    assert json.loads(_requests_to(server, "PUT", "/project/p1/users/u2")[0]["body"]) == {
        "privilegeLevel": "readOnly"
    }
    assert "privileges must be one of" in b.overleaf_set_collaborator_privileges(
        "p1", "u2", "admin"
    )
    assert json.loads(b.overleaf_remove_collaborator("p1", "u2"))["ok"] is True
    assert json.loads(b.overleaf_revoke_invite("p1", "i1"))["ok"] is True

    links = json.loads(b.overleaf_get_sharing_links("p1"))
    assert links["read_only_url"] == f"{b._host}/read/ro123"
    assert links["read_write_url"] == f"{b._host}/789rw"
    assert json.loads(b.overleaf_get_sharing_links("pbad")) == {"ok": True}
    assert "HTTP 404" in b.overleaf_get_sharing_links("pnone")

    assert json.loads(b.overleaf_leave_project("p1"))["ok"] is True
    assert json.loads(b.overleaf_transfer_ownership("p1", "u2"))["ok"] is True
    assert json.loads(
        _requests_to(server, "POST", "/project/p1/transfer-ownership")[0]["body"]
    ) == {"user_id": "u2"}


def test_chat_tags_history_notifications(backend, tmp_path: Path) -> None:
    """Chat, tag, history, label, diff, restore, revert, and notification routes."""
    b, server = backend
    assert json.loads(b.overleaf_get_chat_messages("p1"))["result"][0]["content"] == "hi"
    b.overleaf_get_chat_messages("p1", limit=5, before="123")
    assert _requests_to(server, "GET", "/project/p1/messages")[-1]["query"] == {
        "limit": "5",
        "before": "123",
    }
    assert json.loads(b.overleaf_send_chat_message("p1", "hello"))["ok"] is True
    assert "cannot be empty" in b.overleaf_send_chat_message("p1", "  ")

    assert json.loads(b.overleaf_list_tags())["result"][0]["name"] == "PhD"
    b.overleaf_create_tag("New")
    b.overleaf_create_tag("New", "#fff")
    assert [json.loads(r["body"]) for r in _requests_to(server, "POST", "/tag")] == [
        {"name": "New"},
        {"name": "New", "color": "#fff"},
    ]
    b.overleaf_edit_tag("t1", "Renamed")
    b.overleaf_edit_tag("t1", "Renamed", "#000")
    assert [json.loads(r["body"]) for r in _requests_to(server, "POST", "/tag/t1/edit")] == [
        {"name": "Renamed"},
        {"name": "Renamed", "color": "#000"},
    ]
    assert json.loads(b.overleaf_delete_tag("t1"))["ok"] is True
    assert json.loads(b.overleaf_tag_project("t1", "p1"))["ok"] is True
    assert json.loads(b.overleaf_tag_project("t1", "p1", remove=True))["ok"] is True

    assert json.loads(b.overleaf_get_history("p1"))["result"]["nextBeforeTimestamp"] == 9
    b.overleaf_get_history("p1", min_count=5, before="9")
    assert _requests_to(server, "GET", "/project/p1/updates")[-1]["query"] == {
        "min_count": "5",
        "before": "9",
    }
    assert json.loads(b.overleaf_list_labels("p1"))["result"][0]["id"] == "l1"
    assert json.loads(b.overleaf_create_label("p1", 3, "draft"))["result"]["id"] == "l2"
    assert json.loads(b.overleaf_delete_label("p1", "l1"))["ok"] is True
    assert json.loads(b.overleaf_get_diff("p1", "main.tex", 1, 3))["result"]["diff"][1] == {
        "i": "b"
    }
    assert _requests_to(server, "GET", "/project/p1/diff")[0]["query"] == {
        "pathname": "main.tex",
        "from": "1",
        "to": "3",
    }
    out = tmp_path / "v3.zip"
    assert json.loads(b.overleaf_download_version_zip("p1", 3, str(out)))["ok"] is True
    assert out.read_bytes() == b"PK\x03\x04v3"
    assert json.loads(b.overleaf_restore_file("p1", "main.tex", 3))["result"]["id"] == "d-main"
    assert json.loads(b.overleaf_revert_project("p1", 3))["ok"] is True
    assert json.loads(_requests_to(server, "POST", "/project/p1/revert-project")[0]["body"]) == {
        "version": 3
    }
    assert json.loads(b.overleaf_list_notifications())["result"][0]["_id"] == "n1"


# --------------------------------------------------------------------- git


def _run_git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture()
def git_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Give git a neutral identity and config independent of the developer's."""
    if shutil.which("git") is None:
        pytest.skip("git binary not available")
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(var, "Test")
    for var in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(var, "test@example.com")


def test_git_clone_and_sync(backend, git_env, tmp_path: Path) -> None:
    """Clone over the dumb-HTTP bridge, then commit/pull/push to the bare repo."""
    b, server = backend
    work = tmp_path / "work"
    work.mkdir()
    _run_git("init", "-b", "main", cwd=work)
    (work / "main.tex").write_text("v1")
    _run_git("add", "-A", cwd=work)
    _run_git("commit", "-m", "init", cwd=work)
    bare = tmp_path / "bare.git"
    _run_git("clone", "--bare", str(work), str(bare))
    _run_git("update-server-info", cwd=bare)
    server.git_root = bare

    clone = tmp_path / "clone"
    cloned = json.loads(b.overleaf_git_clone("p1", str(clone)))
    assert cloned["ok"] is True, cloned
    assert (clone / "main.tex").read_text() == "v1"
    assert "/git/p1" in _run_git("remote", "get-url", "origin", cwd=clone)
    assert _GIT_TOKEN not in (clone / ".git" / "config").read_text()

    _run_git("remote", "set-url", "origin", str(bare), cwd=clone)
    (clone / "main.tex").write_text("v2")
    synced = json.loads(b.overleaf_git_sync(str(clone), "Edit intro"))
    assert synced["ok"] is True, synced
    assert [s["step"] for s in synced["steps"]] == ["add", "commit", "pull", "push"]
    assert _run_git("log", "-1", "--format=%s", cwd=bare).strip() == "Edit intro"

    unchanged = json.loads(b.overleaf_git_sync(str(clone)))
    assert [s["step"] for s in unchanged["steps"]] == ["add", "pull", "push"]

    plain = tmp_path / "plain"
    plain.mkdir()
    not_repo = json.loads(b.overleaf_git_sync(str(plain)))
    assert not_repo["ok"] is False and [s["step"] for s in not_repo["steps"]] == ["add"]
    assert json.loads(b.overleaf_git_sync(str(tmp_path / "missing")))["ok"] is False


def test_git_failures_and_missing_token(backend, git_env, tmp_path: Path) -> None:
    """A wrong token fails the clone; a bad dest raises; no token is explained."""
    b, server = backend
    server.git_root = tmp_path
    b._git_token = "wrong"
    failed = json.loads(b.overleaf_git_clone("p1", str(tmp_path / "c1")))
    assert failed["ok"] is False and failed["output"]
    assert json.loads(b.overleaf_git_clone("p1", "~no_such_user_kiss_test/c2"))["ok"] is False
    b._git_token = ""
    assert "git_token not configured" in b.overleaf_git_clone("p1", str(tmp_path / "c3"))
    assert "git_token not configured" in b.overleaf_git_sync(str(tmp_path))
