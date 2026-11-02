"""Stored OAuth credentials go only to the server URL they were issued for (S12 F2).

Two configurations may share a server name and point at different
servers (the same name in two projects, a project override of a user
configuration); the manager keeps them apart as connections, but the
token file was chosen by name alone, so the second server received
the first one's bearer token on its very first request.  Credentials
are now kept per server name *and* URL: same-named servers have
separate files (each recording its URL), and the name-only file of
earlier versions is read by no server, so each gets its own sign-in.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import httpx
import pytest
from mcp.shared.auth import OAuthMetadata, OAuthToken

from kiss.agents.sorcar.mcp_oauth import build_provider
from kiss.agents.sorcar.mcp_servers import FileTokenStorage, MCPServerConfig, _connection_key


class _Recorder(BaseHTTPRequestHandler):
    """Answer every GET with 200 and record its path and ``Authorization`` header."""

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        self.server.seen.append((self.path, self.headers.get("Authorization", "")))  # type: ignore[attr-defined]
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: Any) -> None:  # type: ignore[override]
        pass


class _Origin(ThreadingHTTPServer):
    """One loopback origin; ``seen`` lists the requests it answered."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Recorder)
        self.seen: list[tuple[str, str]] = []
        self.thread = Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}{path}"

    def close(self) -> None:
        self.shutdown()
        self.server_close()
        self.thread.join(10)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    kiss_home = tmp_path / "kiss_home"
    monkeypatch.setenv("KISS_HOME", str(kiss_home))
    monkeypatch.delenv("KISS_MCP_CLIENT_METADATA_URL", raising=False)
    return kiss_home


@pytest.fixture
def origins() -> Iterator[tuple[_Origin, _Origin]]:
    first, second = _Origin(), _Origin()
    yield first, second
    first.close()
    second.close()


async def _refuse(*_args: Any) -> Any:
    raise AssertionError("no interactive sign-in is expected")


def _provider(cfg: MCPServerConfig) -> Any:
    return build_provider(cfg, _refuse, _refuse)


def _request(cfg: MCPServerConfig) -> None:
    """One GET to *cfg*'s URL authenticated the way an agent run's connection is."""

    async def get() -> None:
        async with httpx.AsyncClient(auth=_provider(cfg), trust_env=False) as client:
            await client.get(cfg.url)

    asyncio.run(get())


def _store(cfg: MCPServerConfig, token: str) -> None:
    """Leave *token* in the storage the provider for *cfg* uses, as a sign-in does."""
    provider = _provider(cfg)
    asyncio.run(provider.context.storage.set_tokens(
        OAuthToken(access_token=token, token_type="Bearer", expires_in=3600),
    ))


def test_same_name_other_origin_gets_no_token(
    home: Path, origins: tuple[_Origin, _Origin],
) -> None:
    first, second = origins
    cfg_first = MCPServerConfig(name="shared", transport="http", url=first.url("/mcp"))
    cfg_second = MCPServerConfig(name="shared", transport="http", url=second.url("/mcp"))
    assert _connection_key(cfg_first) != _connection_key(cfg_second)
    _store(cfg_first, "FIRST-ONLY")
    _request(cfg_second)
    assert second.seen == [("/mcp", "")]
    # The first server's own connection keeps using its token.
    _request(cfg_first)
    assert first.seen == [("/mcp", "Bearer FIRST-ONLY")]


def test_same_origin_other_path_gets_no_token(home: Path, origins: tuple[_Origin, _Origin]) -> None:
    first, _second = origins
    cfg_a = MCPServerConfig(name="shared", transport="http", url=first.url("/a/mcp"))
    cfg_b = MCPServerConfig(name="shared", transport="http", url=first.url("/b/mcp"))
    _store(cfg_a, "A-ONLY")
    _request(cfg_b)
    _request(cfg_a)
    assert first.seen == [("/b/mcp", ""), ("/a/mcp", "Bearer A-ONLY")]


def test_a_name_only_file_from_an_earlier_version_is_sent_nowhere(
    home: Path, origins: tuple[_Origin, _Origin],
) -> None:
    first, _second = origins
    cfg = MCPServerConfig(name="legacy", transport="http", url=first.url("/mcp"))
    # The token file an earlier version wrote: by name only, no URL recorded.
    legacy = FileTokenStorage("legacy")
    legacy.put("tokens", {"access_token": "OLD", "token_type": "Bearer", "expires_in": 3600})
    assert legacy.path == home / "mcp_auth" / "legacy.json"
    assert "server_url" not in json.loads(legacy.path.read_text(encoding="utf-8"))
    _request(cfg)
    assert first.seen == [("/mcp", "")]
    # The name-only storage still reads its own file as it is.
    assert asyncio.run(legacy.get_tokens()).access_token == "OLD"


def test_same_named_servers_keep_separate_files_and_both_stay_signed_in(
    home: Path, origins: tuple[_Origin, _Origin],
) -> None:
    first, second = origins
    cfg_first = MCPServerConfig(name="shared", transport="http", url=first.url("/mcp"))
    cfg_second = MCPServerConfig(name="shared", transport="http", url=second.url("/mcp"))
    _store(cfg_first, "FIRST")
    _store(cfg_second, "SECOND")
    file_first = _provider(cfg_first).context.storage.path
    file_second = _provider(cfg_second).context.storage.path
    assert file_first != file_second
    assert file_first.parent == file_second.parent == home / "mcp_auth"
    assert file_first.name.startswith("shared.") and file_first.name.endswith(".json")
    assert file_second.name.startswith("shared.") and file_second.name.endswith(".json")
    # Each file names the server its credentials are for (the digest is opaque).
    assert json.loads(file_first.read_text(encoding="utf-8"))["server_url"] == cfg_first.url
    assert json.loads(file_second.read_text(encoding="utf-8"))["server_url"] == cfg_second.url
    # The authorization-server metadata saved at sign-in stays with its server.
    _provider(cfg_first).context.storage.set_oauth_metadata(OAuthMetadata.model_validate({
        "issuer": first.url(""),
        "authorization_endpoint": first.url("/authorize"),
        "token_endpoint": first.url("/token"),
    }))
    assert _provider(cfg_first).context.oauth_metadata is not None
    assert _provider(cfg_second).context.oauth_metadata is None
    _request(cfg_first)
    _request(cfg_second)
    assert first.seen == [("/mcp", "Bearer FIRST")]
    assert second.seen == [("/mcp", "Bearer SECOND")]
    # The name-only file of earlier versions is not what either server uses.
    assert not (home / "mcp_auth" / "shared.json").exists()
