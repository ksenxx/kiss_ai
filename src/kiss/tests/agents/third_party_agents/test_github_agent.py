# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the GitHub channel agent.

Runs a REAL local HTTP server (stdlib ``ThreadedHTTPServer``) emulating
the GitHub REST API endpoints — no mocks, patches, or fakes.  The
server asserts the ``Authorization: Bearer`` header on every call,
returns canned JSON (or a raw diff for the diff media type), and
records every request (method, path, headers, body) for verification.

Config state is isolated per pytest process because the session
conftest points ``KISS_HOME`` at a temporary directory and
``ChannelConfig.path`` resolves ``$KISS_HOME`` lazily.

Branch-coverage note: every branch of ``github_agent`` is reachable
end-to-end with this emulator (validation, read-only gating, HTTP
errors, connection refusal, non-JSON responses), so no branch needed a
test double.
"""

from __future__ import annotations

import base64
import json
import shutil
import stat
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import pytest

import kiss.agents.third_party_agents.github_agent as gh_mod
from kiss.agents.third_party_agents._backend_utils import (
    ThreadedHTTPServer,
    stop_http_server,
)
from kiss.agents.third_party_agents.github_agent import (
    GitHubAgent,
    GitHubChannelBackend,
    _config,
)

_TOKEN = "test-gh-token"
_DIFF_ACCEPT = "application/vnd.github.v3.diff"
_RAW_ACCEPT = "application/vnd.github.raw+json"

_REPO = {
    "full_name": "octo/hello",
    "description": None,
    "stargazers_count": 5,
    "html_url": "https://github.com/octo/hello",
    "default_branch": "main",
    "private": False,
    "watchers": 999,  # noise: must not appear in condensed output
}
_ISSUE_5 = {
    "number": 5,
    "title": "Crash on start",
    "state": "open",
    "user": {"login": "alice"},
    "created_at": "2026-01-01T00:00:00Z",
    "html_url": "https://github.com/octo/hello/issues/5",
    "body": "B" * 1500,
}
_PR_7 = {
    "number": 7,
    "title": "Add feature",
    "state": "open",
    "user": {"login": "bob"},
    "created_at": "2026-02-01T00:00:00Z",
    "html_url": "https://github.com/octo/hello/pull/7",
    "body": None,
    "head": {"ref": "feature"},
    "base": {"ref": "main"},
    "merged": False,
}
_COMMENT = {
    "user": {"login": "carol"},
    "created_at": "2026-01-02T00:00:00Z",
    "body": "C" * 1200,
}
_COMMIT = {
    "sha": "a" * 40,
    "commit": {
        "message": "fix: crash\n\nlong details",
        "author": {"name": "Alice", "date": "2026-01-03T00:00:00Z"},
    },
}
_BRANCH = {"name": "main", "commit": {"sha": "b" * 40}, "protected": True}
_CODE_MATCH = {
    "name": "main.py",
    "path": "src/main.py",
    "repository": {"full_name": "octo/hello"},
    "html_url": "https://github.com/octo/hello/blob/main/src/main.py",
}
_FILE_TEXT = "hello world\n"
_FILE_OBJ = {
    "type": "file",
    "encoding": "base64",
    "path": "README.md",
    "size": len(_FILE_TEXT),
    "content": base64.b64encode(_FILE_TEXT.encode()).decode(),
}
_DIR_LISTING = [
    {"name": "main.py", "path": "src/main.py", "type": "file", "size": 12},
    {"name": "lib", "path": "src/lib", "type": "dir", "size": 0},
]
_SYMLINK_OBJ = {"type": "symlink", "path": "link.txt", "size": 9}
_ROOT_LISTING = [
    {"name": "README.md", "path": "README.md", "type": "file", "size": 12},
    {"name": "src", "path": "src", "type": "dir", "size": 0},
]
_BIG_RAW = "R" * 9000
_BIG_OBJ = {
    "type": "file",
    "encoding": "none",
    "path": "big.bin",
    "size": len(_BIG_RAW),
    "content": "",
}
_BIGFAIL_OBJ = {
    "type": "file",
    "encoding": "none",
    "path": "bigfail.bin",
    "size": 5,
    "content": "",
}
_DIFF_7 = "diff --git a/x.py b/x.py\n+print(1)\n"
_DIFF_99 = "d" * 9000


class _GHRequestHandler(BaseHTTPRequestHandler):
    """Emulates the GitHub REST API and records requests."""

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
        cast(_GHServer, self.server).requests.append(
            {
                "method": self.command,
                "path": self.path,
                "authorization": self.headers.get("Authorization", ""),
                "accept": self.headers.get("Accept", ""),
                "api_version": self.headers.get("X-GitHub-Api-Version", ""),
                "body": body,
            }
        )

    def _route(self, body: dict[str, Any] | None) -> None:
        self._record(body)
        if not self._authorized():
            self._reply(401, json.dumps({"message": "Bad credentials"}))
            return
        path = urlparse(self.path).path
        key = (self.command, path)
        if key == ("GET", "/fail"):
            self._reply(500, json.dumps({"message": "Internal Server Error"}))
        elif key == ("GET", "/plain"):
            self._reply(200, "plain text", content_type="text/plain")
        elif key == ("GET", "/user"):
            self._reply(
                200,
                json.dumps(
                    {
                        "login": "octo",
                        "name": None,
                        "company": "ACME",
                        "html_url": "https://github.com/octo",
                        "public_repos": 3,
                    }
                ),
            )
        elif key == ("GET", "/search/repositories"):
            self._reply(200, json.dumps({"total_count": 1, "items": [_REPO]}))
        elif key == ("GET", "/search/issues"):
            self._reply(200, json.dumps({"total_count": 2, "items": [_ISSUE_5]}))
        elif key == ("GET", "/search/code"):
            self._reply(200, json.dumps({"total_count": 1, "items": [_CODE_MATCH]}))
        elif key == ("GET", "/repos/octo/hello"):
            self._reply(200, json.dumps(_REPO))
        elif key == ("GET", "/repos/octo/hello/issues"):
            self._reply(200, json.dumps([_ISSUE_5]))
        elif key == ("GET", "/repos/octo/hello/issues/5"):
            self._reply(200, json.dumps(_ISSUE_5))
        elif key == ("GET", "/repos/octo/hello/issues/5/comments"):
            self._reply(200, json.dumps([_COMMENT]))
        elif key == ("GET", "/repos/octo/hello/pulls"):
            self._reply(200, json.dumps([_PR_7]))
        elif key == ("GET", "/repos/octo/hello/pulls/7"):
            if self.headers.get("Accept") == _DIFF_ACCEPT:
                self._reply(200, _DIFF_7, content_type="application/vnd.github.v3.diff")
            else:
                self._reply(200, json.dumps(_PR_7))
        elif key == ("GET", "/repos/octo/hello/pulls/99"):
            self._reply(200, _DIFF_99, content_type="application/vnd.github.v3.diff")
        elif key == ("GET", "/repos/octo/hello/contents/README.md"):
            self._reply(200, json.dumps(_FILE_OBJ))
        elif key == ("GET", "/repos/octo/hello/contents/src"):
            self._reply(200, json.dumps(_DIR_LISTING))
        elif key == ("GET", "/repos/octo/hello/contents/link.txt"):
            self._reply(200, json.dumps(_SYMLINK_OBJ))
        elif key == ("GET", "/repos/octo/hello/contents/"):
            self._reply(200, json.dumps(_ROOT_LISTING))
        elif key == ("GET", "/repos/octo/hello/contents/big.bin"):
            if self.headers.get("Accept") == _RAW_ACCEPT:
                self._reply(200, _BIG_RAW, content_type=_RAW_ACCEPT)
            else:
                self._reply(200, json.dumps(_BIG_OBJ))
        elif key == ("GET", "/repos/octo/hello/contents/bigfail.bin"):
            if self.headers.get("Accept") == _RAW_ACCEPT:
                self._reply(500, json.dumps({"message": "Internal Server Error"}))
            else:
                self._reply(200, json.dumps(_BIGFAIL_OBJ))
        elif key == ("GET", "/repos/octo/hello/commits"):
            self._reply(200, json.dumps([_COMMIT]))
        elif key == ("GET", "/repos/octo/hello/branches"):
            self._reply(200, json.dumps([_BRANCH]))
        elif key == ("POST", "/repos/octo/hello/issues"):
            self._reply(
                201,
                json.dumps({"number": 42, "html_url": "https://github.com/octo/hello/issues/42"}),
            )
        elif key == ("POST", "/repos/octo/hello/issues/5/comments"):
            self._reply(
                201,
                json.dumps({"html_url": "https://github.com/octo/hello/issues/5#issuecomment-1"}),
            )
        elif key == ("PATCH", "/repos/octo/hello/issues/5"):
            self._reply(
                200,
                json.dumps(
                    {
                        "number": 5,
                        "state": "closed",
                        "html_url": "https://github.com/octo/hello/issues/5",
                    }
                ),
            )
        elif key == ("POST", "/repos/octo/hello/pulls"):
            self._reply(
                201,
                json.dumps({"number": 8, "html_url": "https://github.com/octo/hello/pull/8"}),
            )
        elif key == ("PUT", "/repos/octo/hello/pulls/7/merge"):
            self._reply(
                200, json.dumps({"merged": True, "message": "Pull Request successfully merged"})
            )
        else:
            self._reply(404, json.dumps({"message": "Not Found"}))

    def _read_body(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            return None

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        self._route(None)

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        self._route(self._read_body())

    def do_PATCH(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        self._route(self._read_body())

    def do_PUT(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        self._route(self._read_body())

    def log_message(self, *args: Any) -> None:  # type: ignore[override]
        pass


class _GHServer(ThreadedHTTPServer):
    """ThreadedHTTPServer that records every request it receives."""

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, _GHRequestHandler)
        self.requests: list[dict[str, Any]] = []


@pytest.fixture()
def gh_server():
    """Start the emulated GitHub REST server on a free port; yield (base_url, server)."""
    server = _GHServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base_url, server
    finally:
        stop_http_server(server, thread)


@pytest.fixture()
def backend(gh_server):
    """A backend pointed at the emulated server with the valid token."""
    base_url, server = gh_server
    b = GitHubChannelBackend()
    b._base_url = base_url
    b._token = _TOKEN
    return b, server


@pytest.fixture(autouse=True)
def _fresh_config():
    """Start and end every test with no persisted GitHub config."""
    _config.clear()
    yield
    _config.clear()


_READ_TOOLS = [
    "gh_get_file_contents",
    "gh_get_issue",
    "gh_get_me",
    "gh_get_pull_request",
    "gh_get_pull_request_diff",
    "gh_get_repository",
    "gh_list_branches",
    "gh_list_commits",
    "gh_list_issue_comments",
    "gh_list_issues",
    "gh_list_pull_requests",
    "gh_search_code",
    "gh_search_issues",
    "gh_search_repositories",
]
_WRITE_TOOLS = [
    "gh_comment_on_issue",
    "gh_create_issue",
    "gh_create_pull_request",
    "gh_merge_pull_request",
    "gh_update_issue",
]


def _call_every_tool(b: GitHubChannelBackend) -> list[str]:
    """Invoke every backend tool once with valid arguments; return the results."""
    return [
        b.gh_get_me(),
        b.gh_search_repositories("q"),
        b.gh_get_repository("octo", "hello"),
        b.gh_list_issues("octo", "hello"),
        b.gh_get_issue("octo", "hello", 5),
        b.gh_list_issue_comments("octo", "hello", 5),
        b.gh_search_issues("q"),
        b.gh_search_code("q"),
        b.gh_list_pull_requests("octo", "hello"),
        b.gh_get_pull_request("octo", "hello", 7),
        b.gh_get_pull_request_diff("octo", "hello", 7),
        b.gh_get_file_contents("octo", "hello", "README.md"),
        b.gh_list_commits("octo", "hello"),
        b.gh_list_branches("octo", "hello"),
        b.gh_create_issue("octo", "hello", "t"),
        b.gh_comment_on_issue("octo", "hello", 5, "hi"),
        b.gh_update_issue("octo", "hello", 5, state="closed"),
        b.gh_create_pull_request("octo", "hello", "t", "feature", "main"),
        b.gh_merge_pull_request("octo", "hello", 7),
    ]


# ----------------------------------------------------------------------
# Auth tools and config persistence
# ----------------------------------------------------------------------


def test_agent_instantiation_unauthenticated() -> None:
    """A fresh agent is unauthenticated and exposes only the auth trio."""
    agent = GitHubAgent()
    assert agent.name == "GitHub Agent"
    assert agent._is_authenticated() is False
    names = [t.__name__ for t in agent._get_tools()]
    assert names == ["check_github_auth", "authenticate_github", "clear_github_auth"]


def test_check_auth_unauthenticated_message() -> None:
    """check_github_auth explains how to get a token when unauthenticated."""
    agent = GitHubAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    msg = tools["check_github_auth"]()
    assert "authenticate_github" in msg
    assert "https://github.com/settings/tokens" in msg
    assert "gh auth token" in msg


def test_authenticate_persists_config_and_exposes_tools() -> None:
    """authenticate_github persists config (0600) and unlocks backend tools."""
    agent = GitHubAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    result = json.loads(tools["authenticate_github"](_TOKEN))
    assert result["ok"] is True

    assert _config.path.exists()
    if sys.platform != "win32":
        assert stat.S_IMODE(_config.path.stat().st_mode) == 0o600
    saved = json.loads(_config.path.read_text(encoding="utf-8"))
    assert saved == {"token": _TOKEN, "read_only": "false"}

    assert agent._is_authenticated() is True
    checked = json.loads(tools["check_github_auth"]())
    assert checked == {"ok": True, "read_only": False}

    names = [t.__name__ for t in agent._get_tools()]
    assert names[:3] == ["check_github_auth", "authenticate_github", "clear_github_auth"]
    assert sorted(names[3:]) == sorted(_READ_TOOLS + _WRITE_TOOLS)
    assert "connect" not in names  # channel protocol method, not an LLM tool

    cleared = tools["clear_github_auth"]()
    assert "cleared" in cleared.lower()
    assert not _config.path.exists()
    assert agent._is_authenticated() is False
    assert len(agent._get_tools()) == 3


def test_authenticate_read_only_persisted_as_string() -> None:
    """read_only=True is persisted as the string 'true' and reported by check."""
    agent = GitHubAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    json.loads(tools["authenticate_github"](_TOKEN, read_only=True))
    saved = json.loads(_config.path.read_text(encoding="utf-8"))
    assert saved == {"token": _TOKEN, "read_only": "true"}
    assert agent._backend._read_only is True
    checked = json.loads(tools["check_github_auth"]())
    assert checked == {"ok": True, "read_only": True}


def test_authenticate_persistence_failure_returns_ok_false() -> None:
    """authenticate_github reports ok:false and stays unauthenticated when saving fails.

    A regular FILE occupying the config *directory* location makes
    ``ChannelConfig.save`` raise (``mkdir`` cannot replace a file), so
    persistence fails before the backend is ever marked authenticated.
    """
    agent = GitHubAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    blocker = _config.path.parent  # .../third_party_agents/github
    if blocker.is_dir():
        shutil.rmtree(blocker)
    blocker.parent.mkdir(parents=True, exist_ok=True)
    blocker.write_text("not a directory", encoding="utf-8")
    try:
        result = json.loads(tools["authenticate_github"](_TOKEN))
        assert result["ok"] is False
        assert "failed to save GitHub config" in result["error"]
        assert agent._is_authenticated() is False
        assert agent._backend._token == ""
        assert len(agent._get_tools()) == 3  # backend tools stay locked
    finally:
        blocker.unlink()


def test_authenticate_rejects_empty_token() -> None:
    """authenticate_github refuses an empty or blank token."""
    agent = GitHubAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    assert "cannot be empty" in tools["authenticate_github"]("  ")
    assert not _config.path.exists()


def test_new_agent_loads_persisted_config() -> None:
    """A new agent picks up previously persisted credentials and read_only flag."""
    _config.save({"token": _TOKEN, "read_only": "true"})
    agent = GitHubAgent()
    assert agent._is_authenticated() is True
    assert agent._backend._token == _TOKEN
    assert agent._backend._read_only is True

    _config.save({"token": _TOKEN})  # no read_only key -> defaults to read-write
    agent2 = GitHubAgent()
    assert agent2._backend._read_only is False


def test_tools_module_function() -> None:
    """Module-level tools() returns a non-empty tool list."""
    tools = gh_mod.tools()
    assert len(tools) >= 3
    assert all(callable(t) for t in tools)


def test_connect_without_config_fails() -> None:
    """connect() fails cleanly when no config is persisted."""
    b = GitHubChannelBackend()
    assert b.connect() is False
    assert "No GitHub config" in b.connection_info


def test_connect_with_config_succeeds() -> None:
    """connect() loads persisted token and read_only mode into the backend."""
    _config.save({"token": _TOKEN, "read_only": "false"})
    b = GitHubChannelBackend()
    assert b.connect() is True
    assert b._token == _TOKEN
    assert b._read_only is False
    assert "read-write" in b.connection_info

    _config.save({"token": _TOKEN, "read_only": "true"})
    b2 = GitHubChannelBackend()
    assert b2.connect() is True
    assert b2._read_only is True
    assert "read-only" in b2.connection_info


# ----------------------------------------------------------------------
# Read tools
# ----------------------------------------------------------------------


def test_gh_get_me(backend) -> None:
    """gh_get_me hits GET /user with all three GitHub headers."""
    b, server = backend
    result = json.loads(b.gh_get_me())
    assert result == {
        "ok": True,
        "user": {
            "login": "octo",
            "name": "",
            "company": "ACME",
            "url": "https://github.com/octo",
            "public_repos": 3,
        },
    }
    req = server.requests[-1]
    assert (req["method"], urlparse(req["path"]).path) == ("GET", "/user")
    assert req["authorization"] == f"Bearer {_TOKEN}"
    assert req["accept"] == "application/vnd.github+json"
    assert req["api_version"] == "2022-11-28"


def test_gh_search_repositories(backend) -> None:
    """gh_search_repositories passes q/per_page/page and condenses repos."""
    b, server = backend
    result = json.loads(b.gh_search_repositories("language:python", per_page=7, page=2))
    assert result["ok"] is True
    assert result["total_count"] == 1
    assert result["repositories"] == [
        {
            "full_name": "octo/hello",
            "description": "",
            "stars": 5,
            "url": "https://github.com/octo/hello",
            "default_branch": "main",
            "private": False,
        }
    ]
    query = parse_qs(urlparse(server.requests[-1]["path"]).query)
    assert query == {"q": ["language:python"], "per_page": ["7"], "page": ["2"]}


def test_gh_get_repository(backend) -> None:
    """gh_get_repository hits /repos/{owner}/{repo} and condenses the repo."""
    b, server = backend
    result = json.loads(b.gh_get_repository("octo", "hello"))
    assert result["ok"] is True
    assert result["repository"]["full_name"] == "octo/hello"
    assert "watchers" not in result["repository"]
    assert urlparse(server.requests[-1]["path"]).path == "/repos/octo/hello"


def test_gh_list_issues_default(backend) -> None:
    """gh_list_issues passes state/per_page/page and omits the labels param."""
    b, server = backend
    result = json.loads(b.gh_list_issues("octo", "hello"))
    assert result["ok"] is True
    assert result["issues"] == [
        {
            "number": 5,
            "title": "Crash on start",
            "state": "open",
            "user": "alice",
            "created_at": "2026-01-01T00:00:00Z",
            "url": "https://github.com/octo/hello/issues/5",
        }
    ]
    parsed = urlparse(server.requests[-1]["path"])
    assert parsed.path == "/repos/octo/hello/issues"
    query = parse_qs(parsed.query)
    assert query == {"state": ["open"], "per_page": ["20"], "page": ["1"]}


def test_gh_list_issues_with_labels(backend) -> None:
    """gh_list_issues forwards a non-empty labels filter."""
    b, server = backend
    json.loads(b.gh_list_issues("octo", "hello", state="all", labels="bug,p1"))
    query = parse_qs(urlparse(server.requests[-1]["path"]).query)
    assert query["labels"] == ["bug,p1"]
    assert query["state"] == ["all"]


def test_gh_get_issue_includes_truncated_body(backend) -> None:
    """gh_get_issue includes the body truncated to 1000 chars."""
    b, server = backend
    result = json.loads(b.gh_get_issue("octo", "hello", 5))
    assert result["ok"] is True
    assert result["issue"]["number"] == 5
    assert result["issue"]["body"] == "B" * 1000
    assert urlparse(server.requests[-1]["path"]).path == "/repos/octo/hello/issues/5"


def test_gh_list_issue_comments(backend) -> None:
    """gh_list_issue_comments condenses comments and truncates bodies."""
    b, server = backend
    result = json.loads(b.gh_list_issue_comments("octo", "hello", 5, per_page=3))
    assert result["ok"] is True
    assert result["comments"] == [
        {"user": "carol", "created_at": "2026-01-02T00:00:00Z", "body": "C" * 1000}
    ]
    parsed = urlparse(server.requests[-1]["path"])
    assert parsed.path == "/repos/octo/hello/issues/5/comments"
    assert parse_qs(parsed.query) == {"per_page": ["3"], "page": ["1"]}


def test_gh_list_issue_comments_page_forwarded(backend) -> None:
    """gh_list_issue_comments forwards the page parameter for later pages."""
    b, server = backend
    result = json.loads(b.gh_list_issue_comments("octo", "hello", 5, per_page=3, page=4))
    assert result["ok"] is True
    query = parse_qs(urlparse(server.requests[-1]["path"]).query)
    assert query == {"per_page": ["3"], "page": ["4"]}


def test_gh_search_issues(backend) -> None:
    """gh_search_issues hits /search/issues and condenses items."""
    b, server = backend
    result = json.loads(b.gh_search_issues("is:open repo:octo/hello"))
    assert result["ok"] is True
    assert result["total_count"] == 2
    assert result["issues"][0]["number"] == 5
    assert "body" not in result["issues"][0]
    parsed = urlparse(server.requests[-1]["path"])
    assert parsed.path == "/search/issues"
    assert parse_qs(parsed.query)["q"] == ["is:open repo:octo/hello"]


def test_gh_search_code(backend) -> None:
    """gh_search_code hits /search/code and condenses matches."""
    b, server = backend
    result = json.loads(b.gh_search_code("def connect"))
    assert result["ok"] is True
    assert result["matches"] == [
        {
            "name": "main.py",
            "path": "src/main.py",
            "repository": "octo/hello",
            "url": "https://github.com/octo/hello/blob/main/src/main.py",
        }
    ]
    assert urlparse(server.requests[-1]["path"]).path == "/search/code"


def test_gh_list_pull_requests(backend) -> None:
    """gh_list_pull_requests hits /repos/{owner}/{repo}/pulls with params."""
    b, server = backend
    result = json.loads(b.gh_list_pull_requests("octo", "hello", state="closed"))
    assert result["ok"] is True
    assert result["pull_requests"][0]["number"] == 7
    assert result["pull_requests"][0]["user"] == "bob"
    parsed = urlparse(server.requests[-1]["path"])
    assert parsed.path == "/repos/octo/hello/pulls"
    assert parse_qs(parsed.query)["state"] == ["closed"]


def test_gh_get_pull_request(backend) -> None:
    """gh_get_pull_request adds head/base/merged and a body ('' when null)."""
    b, server = backend
    result = json.loads(b.gh_get_pull_request("octo", "hello", 7))
    assert result["ok"] is True
    pr = result["pull_request"]
    assert (pr["head"], pr["base"], pr["merged"]) == ("feature", "main", False)
    assert pr["body"] == ""  # null body condensed to empty string
    assert urlparse(server.requests[-1]["path"]).path == "/repos/octo/hello/pulls/7"


def test_gh_get_pull_request_diff(backend) -> None:
    """gh_get_pull_request_diff sends the diff Accept media type."""
    b, server = backend
    result = json.loads(b.gh_get_pull_request_diff("octo", "hello", 7))
    assert result == {"ok": True, "diff": _DIFF_7}
    req = server.requests[-1]
    assert urlparse(req["path"]).path == "/repos/octo/hello/pulls/7"
    assert req["accept"] == _DIFF_ACCEPT
    assert req["api_version"] == "2022-11-28"


def test_gh_get_pull_request_diff_truncated(backend) -> None:
    """A huge diff is truncated to 8000 chars."""
    b, _ = backend
    result = json.loads(b.gh_get_pull_request_diff("octo", "hello", 99))
    assert result["ok"] is True
    assert result["diff"] == "d" * 8000


def test_gh_get_file_contents_file(backend) -> None:
    """gh_get_file_contents decodes base64 file content; no ref param sent."""
    b, server = backend
    result = json.loads(b.gh_get_file_contents("octo", "hello", "README.md"))
    assert result == {
        "ok": True,
        "path": "README.md",
        "size": len(_FILE_TEXT),
        "content": _FILE_TEXT,
    }
    parsed = urlparse(server.requests[-1]["path"])
    assert parsed.path == "/repos/octo/hello/contents/README.md"
    assert parsed.query == ""


def test_gh_get_file_contents_with_ref(backend) -> None:
    """gh_get_file_contents forwards a non-empty ref as a query param."""
    b, server = backend
    result = json.loads(b.gh_get_file_contents("octo", "hello", "README.md", ref="dev"))
    assert result["ok"] is True
    assert parse_qs(urlparse(server.requests[-1]["path"]).query)["ref"] == ["dev"]


def test_gh_get_file_contents_directory(backend) -> None:
    """A directory path returns the condensed directory listing."""
    b, server = backend
    result = json.loads(b.gh_get_file_contents("octo", "hello", "src"))
    assert result == {"ok": True, "directory": _DIR_LISTING}
    assert urlparse(server.requests[-1]["path"]).path == "/repos/octo/hello/contents/src"


def test_gh_get_file_contents_non_base64_object(backend) -> None:
    """A non-base64 content object (e.g. symlink) returns its metadata."""
    b, _ = backend
    result = json.loads(b.gh_get_file_contents("octo", "hello", "link.txt"))
    assert result == {"ok": True, "path": "link.txt", "type": "symlink", "size": 9}


def test_gh_get_file_contents_empty_path_lists_repo_root(backend) -> None:
    """path='' hits /repos/{owner}/{repo}/contents/ and lists the root directory."""
    b, server = backend
    result = json.loads(b.gh_get_file_contents("octo", "hello", ""))
    assert result == {"ok": True, "directory": _ROOT_LISTING}
    assert urlparse(server.requests[-1]["path"]).path == "/repos/octo/hello/contents/"


def test_gh_get_file_contents_encoding_none_refetches_raw(backend) -> None:
    """encoding 'none' (1-100 MB file) triggers a raw re-fetch, truncated to 8000."""
    b, server = backend
    result = json.loads(b.gh_get_file_contents("octo", "hello", "big.bin"))
    assert result == {
        "ok": True,
        "path": "big.bin",
        "size": len(_BIG_RAW),
        "content": "R" * 8000,
    }
    first, second = server.requests[-2], server.requests[-1]
    assert urlparse(first["path"]).path == "/repos/octo/hello/contents/big.bin"
    assert first["accept"] == "application/vnd.github+json"
    assert urlparse(second["path"]).path == "/repos/octo/hello/contents/big.bin"
    assert second["accept"] == _RAW_ACCEPT
    assert second["api_version"] == "2022-11-28"


def test_gh_get_file_contents_encoding_none_raw_fetch_error(backend) -> None:
    """An HTTP error on the raw re-fetch yields ok:false, not an exception."""
    b, server = backend
    result = json.loads(b.gh_get_file_contents("octo", "hello", "bigfail.bin"))
    assert result["ok"] is False
    assert "500" in result["error"]
    assert server.requests[-1]["accept"] == _RAW_ACCEPT


def test_gh_list_commits_default(backend) -> None:
    """gh_list_commits condenses commits; sha/path params omitted by default."""
    b, server = backend
    result = json.loads(b.gh_list_commits("octo", "hello"))
    assert result["ok"] is True
    assert result["commits"] == [
        {
            "sha": "a" * 12,
            "message": "fix: crash",
            "author": "Alice",
            "date": "2026-01-03T00:00:00Z",
        }
    ]
    query = parse_qs(urlparse(server.requests[-1]["path"]).query)
    assert "sha" not in query
    assert "path" not in query


def test_gh_list_commits_with_sha_and_path(backend) -> None:
    """gh_list_commits forwards non-empty sha and path filters."""
    b, server = backend
    json.loads(b.gh_list_commits("octo", "hello", sha="dev", path="src/main.py"))
    query = parse_qs(urlparse(server.requests[-1]["path"]).query)
    assert query["sha"] == ["dev"]
    assert query["path"] == ["src/main.py"]


def test_gh_list_branches(backend) -> None:
    """gh_list_branches condenses branches to name/sha/protected."""
    b, server = backend
    result = json.loads(b.gh_list_branches("octo", "hello", per_page=5))
    assert result == {
        "ok": True,
        "branches": [{"name": "main", "sha": "b" * 12, "protected": True}],
    }
    parsed = urlparse(server.requests[-1]["path"])
    assert parsed.path == "/repos/octo/hello/branches"
    assert parse_qs(parsed.query) == {"per_page": ["5"], "page": ["1"]}


def test_gh_list_branches_page_forwarded(backend) -> None:
    """gh_list_branches forwards the page parameter for later pages."""
    b, server = backend
    result = json.loads(b.gh_list_branches("octo", "hello", per_page=5, page=3))
    assert result["ok"] is True
    query = parse_qs(urlparse(server.requests[-1]["path"]).query)
    assert query == {"per_page": ["5"], "page": ["3"]}


# ----------------------------------------------------------------------
# Write tools
# ----------------------------------------------------------------------


def test_gh_create_issue_full(backend) -> None:
    """gh_create_issue posts title/body/labels (comma-split, trimmed)."""
    b, server = backend
    result = json.loads(b.gh_create_issue("octo", "hello", "Bug", body="text", labels="bug, p1,"))
    assert result == {"ok": True, "number": 42, "url": "https://github.com/octo/hello/issues/42"}
    req = server.requests[-1]
    assert (req["method"], urlparse(req["path"]).path) == ("POST", "/repos/octo/hello/issues")
    assert req["body"] == {"title": "Bug", "body": "text", "labels": ["bug", "p1"]}


def test_gh_create_issue_minimal(backend) -> None:
    """gh_create_issue omits body and labels when not supplied."""
    b, server = backend
    result = json.loads(b.gh_create_issue("octo", "hello", "Bug"))
    assert result["ok"] is True
    assert server.requests[-1]["body"] == {"title": "Bug"}


def test_gh_comment_on_issue(backend) -> None:
    """gh_comment_on_issue posts the body to the comments endpoint."""
    b, server = backend
    result = json.loads(b.gh_comment_on_issue("octo", "hello", 5, "Looks good"))
    assert result["ok"] is True
    assert result["url"].endswith("#issuecomment-1")
    req = server.requests[-1]
    assert (req["method"], urlparse(req["path"]).path) == (
        "POST",
        "/repos/octo/hello/issues/5/comments",
    )
    assert req["body"] == {"body": "Looks good"}


def test_gh_update_issue_each_field(backend) -> None:
    """gh_update_issue PATCHes only the supplied fields."""
    b, server = backend
    result = json.loads(b.gh_update_issue("octo", "hello", 5, state="closed"))
    assert result == {
        "ok": True,
        "number": 5,
        "state": "closed",
        "url": "https://github.com/octo/hello/issues/5",
    }
    req = server.requests[-1]
    assert (req["method"], urlparse(req["path"]).path) == (
        "PATCH",
        "/repos/octo/hello/issues/5",
    )
    assert req["body"] == {"state": "closed"}

    json.loads(b.gh_update_issue("octo", "hello", 5, title="New title"))
    assert server.requests[-1]["body"] == {"title": "New title"}

    json.loads(b.gh_update_issue("octo", "hello", 5, body="New body"))
    assert server.requests[-1]["body"] == {"body": "New body"}

    json.loads(b.gh_update_issue("octo", "hello", 5, state="open", title="T", body="B"))
    assert server.requests[-1]["body"] == {"state": "open", "title": "T", "body": "B"}


def test_gh_update_issue_nothing_to_update(backend) -> None:
    """gh_update_issue with no fields returns ok:false without any request."""
    b, server = backend
    result = json.loads(b.gh_update_issue("octo", "hello", 5))
    assert result["ok"] is False
    assert "nothing to update" in result["error"]
    assert server.requests == []


def test_gh_create_pull_request(backend) -> None:
    """gh_create_pull_request posts title/head/base/draft (+body when given)."""
    b, server = backend
    result = json.loads(
        b.gh_create_pull_request("octo", "hello", "Add feature", "feature", "main", body="desc")
    )
    assert result == {"ok": True, "number": 8, "url": "https://github.com/octo/hello/pull/8"}
    req = server.requests[-1]
    assert (req["method"], urlparse(req["path"]).path) == ("POST", "/repos/octo/hello/pulls")
    assert req["body"] == {
        "title": "Add feature",
        "head": "feature",
        "base": "main",
        "draft": False,
        "body": "desc",
    }

    json.loads(b.gh_create_pull_request("octo", "hello", "T", "feature", "main", draft=True))
    assert server.requests[-1]["body"] == {
        "title": "T",
        "head": "feature",
        "base": "main",
        "draft": True,
    }


def test_gh_merge_pull_request(backend) -> None:
    """gh_merge_pull_request PUTs the merge_method to /pulls/{n}/merge."""
    b, server = backend
    result = json.loads(b.gh_merge_pull_request("octo", "hello", 7, merge_method="squash"))
    assert result == {"ok": True, "merged": True, "message": "Pull Request successfully merged"}
    req = server.requests[-1]
    assert (req["method"], urlparse(req["path"]).path) == (
        "PUT",
        "/repos/octo/hello/pulls/7/merge",
    )
    assert req["body"] == {"merge_method": "squash"}


def test_gh_merge_pull_request_invalid_method(backend) -> None:
    """An unknown merge_method is refused without any request."""
    b, server = backend
    result = json.loads(b.gh_merge_pull_request("octo", "hello", 7, merge_method="fast-forward"))
    assert result["ok"] is False
    assert "merge_method" in result["error"]
    assert server.requests == []


def test_read_only_mode_blocks_every_write_tool(backend) -> None:
    """In read-only mode every write tool refuses before any HTTP call."""
    b, server = backend
    b._read_only = True
    for call in (
        lambda: b.gh_create_issue("octo", "hello", "t"),
        lambda: b.gh_comment_on_issue("octo", "hello", 5, "hi"),
        lambda: b.gh_update_issue("octo", "hello", 5, state="closed"),
        lambda: b.gh_create_pull_request("octo", "hello", "t", "feature", "main"),
        lambda: b.gh_merge_pull_request("octo", "hello", 7),
    ):
        result = json.loads(call())
        assert result == {"ok": False, "error": "GitHub agent is in read-only mode"}
    assert server.requests == []  # nothing ever reached the server

    result = json.loads(b.gh_get_repository("octo", "hello"))  # reads still work
    assert result["ok"] is True


# ----------------------------------------------------------------------
# Validation and error paths
# ----------------------------------------------------------------------


def test_invalid_owner_repo_path_rejected_before_any_request(backend) -> None:
    """Traversal attempts in owner/repo/path are refused up front — no HTTP."""
    b, server = backend
    for attempt in (
        b.gh_get_repository("../evil", "hello"),
        b.gh_get_repository("octo", "a/b"),
        b.gh_get_repository("octo", "a\\b"),
        b.gh_get_repository("", "hello"),
        b.gh_list_issues("..", "hello"),
        b.gh_get_issue("octo", "..", 5),
        b.gh_list_issue_comments("o/o", "hello", 5),
        b.gh_list_pull_requests("octo", "h/h"),
        b.gh_get_pull_request("o\\o", "hello", 7),
        b.gh_get_pull_request_diff("octo", "..", 7),
        b.gh_get_file_contents("../evil", "hello", "README.md"),
        b.gh_get_file_contents("octo", "hello", "src/../../etc/passwd"),
        b.gh_get_file_contents("octo", "hello", "a\\b"),
        b.gh_list_commits("o/o", "hello"),
        b.gh_list_branches("octo", "h/h"),
        b.gh_create_issue("../evil", "hello", "t"),
        b.gh_comment_on_issue("octo", "..", 5, "hi"),
        b.gh_update_issue("o/o", "hello", 5, state="closed"),
        b.gh_create_pull_request("octo", "a\\b", "t", "h", "b"),
        b.gh_merge_pull_request("..", "hello", 7),
    ):
        result = json.loads(attempt)
        assert result["ok"] is False
        assert "invalid" in result["error"]
    assert server.requests == []  # nothing ever reached the server


def test_owner_is_encoded_as_single_path_segment(backend) -> None:
    """Special characters in owner are percent-encoded, never path-interpreted."""
    b, server = backend
    result = json.loads(b.gh_get_repository("octo cat", "hello"))
    assert result["ok"] is False  # emulator 404s the unknown repo — that's fine
    assert urlparse(server.requests[-1]["path"]).path == "/repos/octo%20cat/hello"


def test_nested_file_path_keeps_slashes(backend) -> None:
    """File paths keep '/' separators but encode other special characters."""
    b, server = backend
    json.loads(b.gh_get_file_contents("octo", "hello", "src/a b.py"))
    assert (
        urlparse(server.requests[-1]["path"]).path
        == "/repos/octo/hello/contents/src/a%20b.py"
    )


def test_unauthorized_token_returns_ok_false_for_every_tool(gh_server) -> None:
    """A 401 yields ok:false JSON from every tool — no exception."""
    base_url, _ = gh_server
    b = GitHubChannelBackend()
    b._base_url = base_url
    b._token = "wrong-token"
    for result_str in _call_every_tool(b):
        result = json.loads(result_str)
        assert result["ok"] is False
        assert "401" in result["error"]


def test_server_error_returns_ok_false(backend) -> None:
    """A 500 from the server yields ok:false JSON, not an exception."""
    b, _ = backend
    data, err = b._api("GET", "/fail")
    assert data is None
    assert "500" in err


def test_non_json_success_response_returns_text(backend) -> None:
    """A 200 with a non-JSON body is returned as raw text by _api."""
    b, _ = backend
    data, err = b._api("GET", "/plain")
    assert err == ""
    assert data == "plain text"


def test_main_without_args_prints_usage() -> None:
    """Running the module CLI with no arguments prints usage and exits 1.

    Coverage note: this exercises ``main()``'s ``channel_main`` call in
    a real subprocess, which in-process coverage cannot trace — that
    line shows as uncovered in the report despite being tested here.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "kiss.agents.third_party_agents.github_agent"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1
    assert "Usage: kiss-github" in proc.stdout


def test_connection_refused_returns_ok_false_for_every_tool() -> None:
    """Tools return ok:false when the server is unreachable — never raise."""
    b = GitHubChannelBackend()
    b._base_url = "http://127.0.0.1:9"  # discard port; nothing listens
    b._token = _TOKEN
    for result_str in _call_every_tool(b):
        result = json.loads(result_str)
        assert result["ok"] is False
        assert result["error"]
