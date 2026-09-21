# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for Muse-auth on Slack, Firecrawl, and Brave Search.

SEA style, mirroring ``test_muse_auth.py``: a REAL Muse-auth daemon
subprocess plus a REAL local HTTP server (stdlib ``ThreadedHTTPServer``)
emulating the Slack / Firecrawl / Brave Search REST APIs — no mocks,
patches, or fakes.  The emulated API asserts every request arriving at
the "network" carries the REAL credential in the right header (proving
the boundary swap, including Brave's non-Bearer
``X-Subscription-Token``), while the agent-side backends only ever
hold ``muse-sgt.*`` surrogates.

Branch-coverage notes (unreachable without test doubles, so documented
instead of mocked):

* ``slack_transport.MuseWebClient`` inherits slack_sdk's
  ``files_upload_v2`` second step, which POSTs to a pre-signed
  ``files.slack.com`` URL with no credential; emulating Slack's
  two-step upload handshake end-to-end is out of scope here because it
  never touches the vault or the boundary.
* ``vault.resolve_credential``'s ``KeyError`` raise and
  ``vault.enrolled_hosts``'s missing-file branch are guarded by the
  daemon (surrogate binding is validated first, so the vault file
  exists), so they cannot fire end-to-end.
* ``daemon._boundary``'s unsafe-credential-header rejection is defense
  in depth: enrollment already rejects such headers, so a vault entry
  naming one cannot exist through daemon ops.
* ``slack_transport.MuseWebClient``'s str-body encode branch cannot
  fire through slack_sdk 3.40.1, which always encodes the body to
  bytes before building the urllib request; its ``_upload_file``
  MuseAuthError branch needs credentials cleared between upload steps
  1 and 2 (a cross-process race).
* ``authenticate_slack``'s ``client is None`` guard needs the vault
  cleared between its ``store_credentials`` and the surrogate mint (a
  cross-process race).
* ``ensure_daemon``'s wait-loop sleeps and 15s startup-failure raise
  depend on daemon spawn/shutdown timing.
"""

from __future__ import annotations

import gzip
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.third_party_agents._backend_utils import ThreadedHTTPServer, stop_http_server
from kiss.agents.third_party_agents.brave_sea import BraveSearchChannelBackend
from kiss.agents.third_party_agents.brave_sea import _config as brave_config
from kiss.agents.third_party_agents.firecrawl_sea import FirecrawlChannelBackend
from kiss.agents.third_party_agents.firecrawl_sea import _config as firecrawl_config
from kiss.agents.third_party_agents.muse_auth import __main__ as muse_cli
from kiss.agents.third_party_agents.muse_auth._common import (
    PROTOCOL_VERSION,
    muse_auth_dir,
    recv_frame,
    send_frame,
    socket_path,
    valid_service_name,
)
from kiss.agents.third_party_agents.muse_auth.client import (
    MuseAuthError,
    _checked,
    clear_credentials,
    grant,
    mint_surrogate,
    stop_daemon,
    vault_has_credentials,
)
from kiss.agents.third_party_agents.slack_sea import (
    SlackChannelBackend,
    _muse_service,
    _muse_web_client,
    _save_token,
    _token_path,
)
from kiss.tests.agents.third_party_agents.muse_test_utils import (
    auth_tools,
    setup_muse_env,
    teardown_muse_env,
    wait_daemon_stopped,
)

_REAL_SLACK_TOKEN = "xoxb-real-secret-slack"
_REAL_FIRECRAWL_KEY = "fc-real-secret-key"
_REAL_BRAVE_KEY = "brave-real-secret-key"

_GZIP_PATH = "/api/admin.analytics.getFile"


class _ApiHandler(BaseHTTPRequestHandler):
    """Emulated REST API recording every request's method/path/headers."""

    server: Any

    def _serve(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        self.server.requests.append(
            {
                "method": self.command,
                "path": self.path,
                "headers": {k: v for k, v in self.headers.items()},
                "body": body.decode("utf-8", errors="replace"),
            }
        )
        auth = next((v for k, v in self.headers.items() if k.lower() == "authorization"), "")
        if self.path == "/upload/file" and self.server.upload_redirect_to:
            self.send_response(307)
            self.send_header("Location", self.server.upload_redirect_to)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if auth.endswith("-invalid"):
            payload = json.dumps({"ok": False, "error": "invalid_auth"}).encode()
            content_type = "application/json; charset=utf-8"
        elif self.path == _GZIP_PATH:
            payload = gzip.compress(b"analytics-bytes")
            content_type = "application/gzip"
        else:
            port = self.server.server_address[1]
            upload_url = (
                self.server.upload_url_override or f"http://127.0.0.1:{port}/upload/file"
            )
            payload = json.dumps(
                {
                    "ok": True,
                    "success": True,
                    "user_id": "U1",
                    "user": "kissbot",
                    "team": "KISS",
                    "id": "crawl-1",
                    "ts": "111.222",
                    "data": {"metadata": {"title": "Page", "statusCode": 200}},
                    "channel": {"id": "C1", "name": "general"},
                    "upload_url": upload_url,
                    "file_id": "F1",
                    "files": [{"id": "F1", "title": "notes"}],
                }
            ).encode()
            content_type = "application/json; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        """Serve a GET request."""
        self._serve()

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        """Serve a POST request."""
        self._serve()

    def do_DELETE(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        """Serve a DELETE request."""
        self._serve()

    def log_message(self, *_args: Any) -> None:  # type: ignore[override]
        """Silence request logging."""


class _ApiServer(ThreadedHTTPServer):
    """ThreadedHTTPServer that records requests for verification."""

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, _ApiHandler)
        self.requests: list[dict[str, Any]] = []
        # When set, files.getUploadURLExternal replies point uploads at
        # this URL instead of this server (rogue-host upload tests).
        self.upload_url_override: str = ""
        # When set, POSTs to /upload/file answer 307 -> this URL
        # (redirect content-egress tests).
        self.upload_redirect_to: str = ""

    def header(self, name: str, index: int = -1) -> str:
        """Return a recorded request header (case-insensitive).

        Args:
            name: Header name.
            index: Which recorded request to inspect (default: last).

        Returns:
            The header value, or ``""`` when absent.
        """
        headers = self.requests[index]["headers"]
        return next((v for k, v in headers.items() if k.lower() == name.lower()), "")


@pytest.fixture()
def api_server() -> Any:
    """Run the emulated REST API on a loopback port."""
    server = _ApiServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    stop_http_server(server, thread)


@pytest.fixture()
def muse_env(isolated_kiss_home: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Enable Muse-auth inside an isolated ``KISS_HOME`` with a live daemon.

    Slack and Brave get a loopback ``extra_hosts`` policy entry (their
    real hosts are fixed); Firecrawl deliberately gets none, so its
    tests prove the enrollment-time ``hosts`` extension works.
    """
    policy = {
        "defaults": {"read": "allow", "write": "ask"},
        "services": {
            "slack": {"extra_hosts": ["127.0.0.1"]},
            "brave_search": {"extra_hosts": ["127.0.0.1"]},
        },
    }
    setup_muse_env(monkeypatch, policy)
    yield isolated_kiss_home
    teardown_muse_env()


def _slack_backend(api_server: _ApiServer) -> SlackChannelBackend:
    """Build a connected Muse-mode Slack backend against the emulator.

    Args:
        api_server: The emulated Slack API.

    Returns:
        A connected backend holding only a surrogate-token client.
    """
    backend = SlackChannelBackend()
    backend._api_base_url = f"http://127.0.0.1:{api_server.server_address[1]}/api/"
    assert backend.connect()
    return backend


def test_slack_boundary_swap_and_action_classes(muse_env: Path, api_server: _ApiServer) -> None:
    """Slack reads run unattended, writes need grants, tokens are swapped."""
    _save_token(_REAL_SLACK_TOKEN, "default")
    backend = _slack_backend(api_server)
    client = backend._client
    assert client is not None
    assert str(client.token).startswith("muse-sgt.slack.")
    assert vault_has_credentials("slack")
    # connect() ran auth.test (a read) through the boundary.
    assert api_server.requests[-1]["path"].endswith("/auth.test")
    assert api_server.header("Authorization") == f"Bearer {_REAL_SLACK_TOKEN}"

    # Reads (conversations.info, users.list) pass without grants.
    info = json.loads(backend.get_channel_info("C1"))
    assert info["ok"] is True
    users = json.loads(backend.list_users())
    assert users["ok"] is True

    # A write (chat.postMessage) is asked; the denial carries the grant
    # command and the emulated network never saw the attempt.
    before = len(api_server.requests)
    denied = json.loads(backend.post_message("C1", "hello"))
    assert denied["ok"] is False
    assert "muse_auth_denied" in denied["error"]
    assert "grant slack write" in denied["error"]
    assert len(api_server.requests) == before

    # A once-grant admits exactly one write.
    grant("slack", "write", "once")
    sent = json.loads(backend.post_message("C1", "hello"))
    assert sent["ok"] is True
    assert api_server.requests[-1]["path"].endswith("/chat.postMessage")
    assert api_server.header("Authorization") == f"Bearer {_REAL_SLACK_TOKEN}"
    assert json.loads(backend.post_message("C1", "again"))["ok"] is False

    # The surrogate never crossed the network boundary.
    for request in api_server.requests:
        assert "muse-sgt." not in json.dumps(request["headers"])


def test_slack_gzip_binary_response(muse_env: Path, api_server: _ApiServer) -> None:
    """Binary application/gzip Slack responses survive the boundary."""
    _save_token(_REAL_SLACK_TOKEN, "default")
    backend = _slack_backend(api_server)
    client = backend._client
    assert client is not None
    grant("slack", "write", "once")
    resp = client.api_call("admin.analytics.getFile", json={"type": "member"})
    assert isinstance(resp.data, bytes)
    assert gzip.decompress(resp.data) == b"analytics-bytes"


def test_slack_vault_first_and_workspace_names(muse_env: Path, api_server: _ApiServer) -> None:
    """Vault-first connects survive legacy-file deletion; names are valid."""
    _save_token(_REAL_SLACK_TOKEN, "default")
    backend = _slack_backend(api_server)
    assert backend._client is not None

    # Migration deleted the plaintext token file; the vault alone still
    # connects.
    assert not _token_path("default").exists()
    backend2 = SlackChannelBackend()
    backend2._api_base_url = backend._api_base_url
    assert backend2.connect()

    # Without vault or token file there is nothing to connect with.
    clear_credentials("slack")
    backend3 = SlackChannelBackend()
    backend3._api_base_url = backend._api_base_url
    assert not backend3.connect()
    assert "No Slack credential" in backend3._connection_info
    assert _muse_web_client("default") is None

    # Workspace-keyed service names are deterministic and valid.
    assert _muse_service("default") == "slack"
    named = _muse_service("My Team!")
    assert named.startswith("slack-my-team--")
    assert named == _muse_service("My Team!")
    assert named != _muse_service("my team!")
    assert valid_service_name(named)
    # The 64-bit digest separates labels whose slug AND 32-bit digest
    # collide (a demonstrated birthday collision with 8 hex chars).
    collide_a = _muse_service("AbcdeFgHIJklMnOPqrStuVwXyZABCdEFGhIJKlmn")
    collide_b = _muse_service("AbcDEfghIjkLMNOpqRStuvWXYzabCdefgHiJKlMn")
    assert collide_a != collide_b
    assert valid_service_name(collide_a) and valid_service_name(collide_b)

    # A non-default workspace enrolls under its own service name and
    # inherits the slack host allowlist via the service-name prefix.
    _save_token(_REAL_SLACK_TOKEN, "team2")
    backend4 = SlackChannelBackend(workspace="team2")
    backend4._api_base_url = backend._api_base_url
    assert not backend4.connect()  # 127.0.0.1 not allowed for slack-team2-...
    service = _muse_service("team2")
    assert vault_has_credentials(service)
    policy_path = muse_auth_dir() / "policy.json"
    policy = json.loads(policy_path.read_text())
    policy["services"][service] = {"extra_hosts": ["127.0.0.1"]}
    policy_path.write_text(json.dumps(policy))
    assert backend4.connect()
    assert api_server.header("Authorization") == f"Bearer {_REAL_SLACK_TOKEN}"


def test_slack_token_rotation_and_clear_via_tools(muse_env: Path, api_server: _ApiServer) -> None:
    """authenticate_slack re-enrolls rotated tokens; clear wipes the vault."""
    from kiss.agents.third_party_agents.slack_sea import SlackAgent

    agent = SlackAgent()
    assert agent._backend._client is None
    agent._backend._api_base_url = f"http://127.0.0.1:{api_server.server_address[1]}/api/"
    tools = auth_tools(agent)

    assert json.loads(tools["authenticate_slack"]("xoxb-first"))["ok"] is True
    client = agent._backend._client
    assert client is not None
    assert str(client.token).startswith("muse-sgt.slack.")
    # Muse-mode authentication never writes a plaintext token file, and
    # its auth.test validation ran through the boundary: it is audited
    # and the emulated API saw the real token exactly once so far.
    assert not _token_path("default").exists()
    audit = (muse_auth_dir() / "audit.jsonl").read_text()
    records = [json.loads(line) for line in audit.splitlines()]
    assert any(r["service"] == "slack" and r["path"].endswith("/auth.test") for r in records)
    assert len(api_server.requests) == 1
    client.auth_test()
    assert api_server.header("Authorization") == "Bearer xoxb-first"

    # Rotate: the second token must actually be used.
    assert json.loads(tools["authenticate_slack"]("xoxb-second"))["ok"] is True
    client = agent._backend._client
    assert client is not None
    client.auth_test()
    assert api_server.header("Authorization") == "Bearer xoxb-second"

    # check_slack_auth works through the boundary client.
    checked = json.loads(tools["check_slack_auth"]())
    assert checked["ok"] is True
    assert checked["team"] == "KISS"

    # An invalid token is rolled back: nothing stays enrolled.
    failed = json.loads(tools["authenticate_slack"]("xoxb-invalid"))
    assert failed["ok"] is False
    assert "invalid_auth" in failed["error"]
    assert not vault_has_credentials("slack")
    assert agent._backend._client is None
    # Re-authenticate so the clear below exercises a populated vault.
    assert json.loads(tools["authenticate_slack"]("xoxb-second"))["ok"] is True

    # Clearing wipes both the token file and the vault enrollment.
    assert "cleared" in tools["clear_slack_auth"]()
    assert not vault_has_credentials("slack")
    assert mint_surrogate("slack") is None


def test_firecrawl_selfhosted_enrollment_hosts(muse_env: Path, api_server: _ApiServer) -> None:
    """A self-hosted base_url host is enrolled with the credential."""
    base_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    firecrawl_config.save({"api_key": _REAL_FIRECRAWL_KEY, "base_url": base_url})
    backend = FirecrawlChannelBackend()
    assert backend.connect()
    assert backend._api_key.startswith("muse-sgt.firecrawl.")
    assert vault_has_credentials("firecrawl")
    # Migration scrubbed the plaintext key but kept the metadata, so
    # enrolled processes never again read a file holding the secret.
    scrubbed = json.loads(firecrawl_config.path.read_text())
    assert scrubbed == {"base_url": base_url}

    # Scrape/map/search are read-class despite being POSTs; no policy
    # extra_hosts exist for firecrawl, so reaching 127.0.0.1 proves the
    # enrollment hosts extended the allowlist.
    result = json.loads(backend.firecrawl_scrape("https://example.com"))
    assert result["ok"] is True
    assert api_server.requests[-1]["path"] == "/v2/scrape"
    assert api_server.header("Authorization") == f"Bearer {_REAL_FIRECRAWL_KEY}"

    # Starting a crawl is a write: asked, then admitted by a grant.
    denied = json.loads(backend.firecrawl_start_crawl("https://example.com"))
    assert denied["ok"] is False
    assert "grant firecrawl write" in denied["error"]
    grant("firecrawl", "write", "once")
    started = json.loads(backend.firecrawl_start_crawl("https://example.com"))
    assert started["ok"] is True
    assert started["crawl_id"] == "crawl-1"

    # Vault-first: with the legacy config gone, connects still work.
    firecrawl_config.clear()
    backend2 = FirecrawlChannelBackend()
    assert backend2.connect()
    # The base_url is no longer configured, so the default is used; the
    # enrolled host keeps the credential usable after re-configuration.
    assert backend2._base_url == "https://api.firecrawl.dev"

    clear_credentials("firecrawl")
    backend3 = FirecrawlChannelBackend()
    assert not backend3.connect()
    assert "No Firecrawl credential" in backend3._connection_info


def test_firecrawl_rotation_and_clear_via_tools(muse_env: Path, api_server: _ApiServer) -> None:
    """authenticate_firecrawl re-enrolls; clear_firecrawl_auth wipes the vault."""
    from kiss.agents.third_party_agents.firecrawl_sea import FirecrawlAgent

    base_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    agent = FirecrawlAgent()
    tools = auth_tools(agent)
    assert "Not configured" in tools["check_firecrawl_auth"]()

    assert json.loads(tools["authenticate_firecrawl"]("fc-first", base_url))["ok"] is True
    json.loads(agent._backend.firecrawl_scrape("https://example.com"))
    assert api_server.header("Authorization") == "Bearer fc-first"

    assert json.loads(tools["authenticate_firecrawl"]("fc-second", base_url))["ok"] is True
    json.loads(agent._backend.firecrawl_scrape("https://example.com"))
    assert api_server.header("Authorization") == "Bearer fc-second"

    # A cloud enrollment (no base_url) scrubs the key-only config away.
    assert json.loads(tools["authenticate_firecrawl"]("fc-cloud"))["ok"] is True
    assert not firecrawl_config.path.exists()
    assert agent._backend._base_url == "https://api.firecrawl.dev"

    assert "cleared" in tools["clear_firecrawl_auth"]()
    assert not vault_has_credentials("firecrawl")


def test_brave_header_kind_credential_swap(muse_env: Path, api_server: _ApiServer) -> None:
    """The Brave surrogate bearer becomes a real X-Subscription-Token."""
    brave_config.save({"api_key": _REAL_BRAVE_KEY})
    backend = BraveSearchChannelBackend()
    backend._base_url = f"http://127.0.0.1:{api_server.server_address[1]}/res/v1"
    assert backend.connect()
    assert backend._api_key.startswith("muse-sgt.brave_search.")
    assert vault_has_credentials("brave_search")

    result = json.loads(backend.brave_web_search("kiss framework"))
    assert result["ok"] is True
    seen = api_server.requests[-1]
    assert seen["path"].startswith("/res/v1/web/search?")
    assert api_server.header("X-Subscription-Token") == _REAL_BRAVE_KEY
    # The surrogate bearer was consumed at the boundary: no
    # Authorization header and no surrogate leaked to the network.
    assert api_server.header("Authorization") == ""
    assert "muse-sgt." not in json.dumps(seen["headers"])

    # Vault-first: the config alone being gone does not break connects.
    brave_config.clear()
    backend2 = BraveSearchChannelBackend()
    backend2._base_url = backend._base_url
    assert backend2.connect()
    assert json.loads(backend2.brave_news_search("news"))["ok"] is True

    clear_credentials("brave_search")
    backend3 = BraveSearchChannelBackend()
    assert not backend3.connect()
    assert "No Brave Search credential" in backend3._connection_info


def test_brave_rotation_and_clear_via_tools(muse_env: Path, api_server: _ApiServer) -> None:
    """authenticate_brave_search re-enrolls; clear wipes the vault."""
    from kiss.agents.third_party_agents.brave_sea import BraveSearchAgent

    agent = BraveSearchAgent()
    tools = auth_tools(agent)
    base_url = f"http://127.0.0.1:{api_server.server_address[1]}/res/v1"

    assert json.loads(tools["authenticate_brave_search"]("brave-first"))["ok"] is True
    agent._backend._base_url = base_url
    json.loads(agent._backend.brave_web_search("q"))
    assert api_server.header("X-Subscription-Token") == "brave-first"

    assert json.loads(tools["authenticate_brave_search"]("brave-second"))["ok"] is True
    agent._backend._base_url = base_url
    json.loads(agent._backend.brave_web_search("q"))
    assert api_server.header("X-Subscription-Token") == "brave-second"

    assert "cleared" in tools["clear_brave_search_auth"]()
    assert not vault_has_credentials("brave_search")


def test_slack_stale_surrogate_and_make_backend(muse_env: Path, api_server: _ApiServer) -> None:
    """Stale surrogates raise MuseAuthError; _make_backend is Muse-aware."""
    from kiss.agents.third_party_agents.slack_sea import _make_backend

    _save_token(_REAL_SLACK_TOKEN, "default")
    backend = _make_backend("default")
    client = backend._client
    assert client is not None
    assert str(client.token).startswith("muse-sgt.slack.")

    # Clearing the vault invalidates the surrogate; the transport
    # surfaces the daemon's non-denial error as MuseAuthError, and the
    # polling helpers retry it as a transient transport failure.
    clear_credentials("slack")
    with pytest.raises(MuseAuthError, match="stale surrogate"):
        client.auth_test()
    backend._client = client
    with pytest.raises(MuseAuthError, match="stale surrogate"):
        backend.poll_messages("C1", "0")

    # Without any credential (migration already deleted the plaintext),
    # poll-mode backend construction exits.
    assert not _token_path("default").exists()
    with pytest.raises(SystemExit):
        _make_backend("default")


def test_slack_upload_goes_through_boundary(muse_env: Path, api_server: _ApiServer) -> None:
    """files_upload_v2 content egress is authorized by Sentinel per hop."""
    from slack_sdk.errors import SlackRequestError

    _save_token(_REAL_SLACK_TOKEN, "default")
    backend = _slack_backend(api_server)
    grant("slack", "write", "session")

    result = json.loads(backend.upload_file("C1", "quarterly numbers", "report.txt"))
    assert result["ok"] is True
    assert result["file_id"] == "F1"
    upload = next(r for r in api_server.requests if r["path"] == "/upload/file")
    assert "quarterly numbers" in upload["body"]

    # A rogue upload URL on an unallowlisted host is refused BEFORE any
    # content leaves the machine.
    rogue = _ApiServer(("127.0.0.2", 0))
    rogue_thread = threading.Thread(target=rogue.serve_forever, daemon=True)
    rogue_thread.start()
    try:
        api_server.upload_url_override = (
            f"http://127.0.0.2:{rogue.server_address[1]}/upload/steal"
        )
        with pytest.raises(SlackRequestError, match="Failed to upload"):
            backend.upload_file("C1", "TOP SECRET CONTENT", "secret.txt")
        assert rogue.requests == []

        # An ALLOWLISTED upload URL answering 307 to an unallowlisted
        # host must not be followed: the redirect keeps the POST body,
        # so following it would ship the content off the allowlist.
        api_server.upload_url_override = ""
        api_server.upload_redirect_to = (
            f"http://127.0.0.2:{rogue.server_address[1]}/upload/stolen"
        )
        with pytest.raises(SlackRequestError, match="carry the request body"):
            backend.upload_file("C1", "TOP SECRET CONTENT", "secret.txt")
        assert rogue.requests == []
    finally:
        api_server.upload_url_override = ""
        api_server.upload_redirect_to = ""
        stop_http_server(rogue, rogue_thread)


def _start_relic_daemon() -> tuple[threading.Thread, threading.Event]:
    """Emulate a surviving pre-upgrade daemon on the real socket path.

    It answers ``status`` without a protocol field (protocol 1) and
    honors ``stop``, exactly like the previous daemon implementation.

    Returns:
        The serving thread and the event set once ``stop`` arrived.
    """
    path = socket_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(4)
    stopped = threading.Event()

    def serve_relic() -> None:
        while not stopped.is_set():
            try:
                conn, _ = server.accept()
            except OSError:
                break
            with conn:
                try:
                    request = recv_frame(conn)
                except Exception:
                    continue
                if request.get("op") == "stop":
                    stopped.set()
                    send_frame(conn, {"ok": True})
                else:
                    send_frame(conn, {"ok": True, "services": []})
        # Unlink BEFORE close: connects to an unlinked-but-bound UDS fail
        # immediately (what ensure_daemon polls for), and a late unlink can
        # no longer delete the NEW daemon's socket bound at the same path
        # after close() signalled "down".
        path.unlink(missing_ok=True)
        server.close()

    relic = threading.Thread(target=serve_relic, daemon=True)
    relic.start()
    return relic, stopped


def test_stale_pre_upgrade_daemon_is_replaced(muse_env: Path) -> None:
    """ensure_daemon stops a protocol-1 relic and spawns the current daemon."""
    from kiss.agents.third_party_agents.muse_auth import client as muse_client

    relic, stopped = _start_relic_daemon()
    assert muse_client._daemon_protocol() == 1

    muse_client.ensure_daemon()
    relic.join(timeout=10.0)
    assert stopped.is_set()
    status = muse_client._checked({"op": "status"})
    assert status["protocol"] == PROTOCOL_VERSION

    # A relic that REPLACES the verified daemon (new socket inode) is
    # caught without any cache reset: the socket identity changed, so
    # the handshake reruns and replaces the relic again.
    stop_daemon()
    wait_daemon_stopped()
    relic2, stopped2 = _start_relic_daemon()
    muse_client.ensure_daemon()
    relic2.join(timeout=10.0)
    assert stopped2.is_set()
    assert muse_client._checked({"op": "status"})["protocol"] == PROTOCOL_VERSION


def test_daemon_protocol_of_garbage_listeners(muse_env: Path) -> None:
    """Listeners answering garbage classify as protocol 0 (incompatible)."""
    from kiss.agents.third_party_agents.muse_auth import client as muse_client

    path = socket_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(4)
    replies = [{"ok": False, "error": "boom"}, {"ok": True, "protocol": "nope"}]

    def serve_garbage() -> None:
        for reply in replies:
            conn, _ = server.accept()
            with conn:
                recv_frame(conn)
                send_frame(conn, reply)
        server.close()
        path.unlink(missing_ok=True)

    thread = threading.Thread(target=serve_garbage, daemon=True)
    thread.start()
    try:
        assert muse_client._daemon_protocol() == 0  # error reply
        assert muse_client._daemon_protocol() == 0  # non-integer protocol
    finally:
        thread.join(timeout=10.0)
    assert muse_client._daemon_protocol() is None  # nothing listening


def test_slack_workspace_list_and_delete_are_vault_aware(
    muse_env: Path, capsys: pytest.CaptureFixture
) -> None:
    """Vault-only workspaces are listed and deletion clears the vault."""
    from kiss.agents.third_party_agents.slack_sea import (
        _delete_workspace,
        _list_workspaces,
    )

    _list_workspaces()
    assert "No workspaces found" in capsys.readouterr().out

    _save_token(_REAL_SLACK_TOKEN, "team2")
    backend = SlackChannelBackend(workspace="team2")
    backend.connect()  # enrolls into the vault, deleting the plaintext
    service = _muse_service("team2")
    assert vault_has_credentials(service)
    assert not _token_path("team2").exists()

    _list_workspaces()
    listing = capsys.readouterr().out
    assert "team2" in listing
    assert "vault" in listing

    # The CLI status shows workspace-keyed enrollments too.
    assert muse_cli.main(["status"]) == 0
    assert service in capsys.readouterr().out

    _delete_workspace("team2")
    assert "deleted" in capsys.readouterr().out
    assert not vault_has_credentials(service)
    with pytest.raises(SystemExit):
        _delete_workspace("team2")


def test_slack_direct_muse_auth_workspace_is_listed(
    muse_env: Path, api_server: _ApiServer, capsys: pytest.CaptureFixture
) -> None:
    """A workspace authenticated straight into the vault shows up in listings."""
    from kiss.agents.third_party_agents.slack_sea import SlackAgent, _list_workspaces

    agent = SlackAgent(workspace="team3")
    agent._backend._api_base_url = f"http://127.0.0.1:{api_server.server_address[1]}/api/"
    tools = auth_tools(agent)
    # Allow the boundary to reach the emulator for this workspace.
    policy_path = muse_auth_dir() / "policy.json"
    policy = json.loads(policy_path.read_text())
    policy["services"][_muse_service("team3")] = {"extra_hosts": ["127.0.0.1"]}
    policy_path.write_text(json.dumps(policy))

    assert json.loads(tools["authenticate_slack"]("xoxb-direct"))["ok"] is True
    assert not _token_path("team3").exists()  # vault-only, no plaintext
    _list_workspaces()
    listing = capsys.readouterr().out
    assert "team3" in listing
    assert "vault" in listing


def test_slack_transport_failure_rolls_back_enrollment(
    muse_env: Path, refusing_port: int
) -> None:
    """A Muse boundary failure during validation clears the enrollment."""
    from kiss.agents.third_party_agents.slack_sea import SlackAgent

    agent = SlackAgent()
    agent._backend._api_base_url = f"http://127.0.0.1:{refusing_port}/api/"
    tools = auth_tools(agent)
    result = json.loads(tools["authenticate_slack"]("xoxb-unvalidated-secret"))
    assert result["ok"] is False
    assert "Token validation failed" in result["error"]
    # The unvalidated token did not stay enrolled or mintable.
    assert not vault_has_credentials("slack")
    assert mint_surrogate("slack") is None
    assert agent._backend._client is None


def test_firecrawl_base_url_survives_key_removal(
    muse_env: Path, api_server: _ApiServer
) -> None:
    """Removing api_key after migration keeps the self-hosted base_url."""
    base_url = f"http://127.0.0.1:{api_server.server_address[1]}/proxy/firecrawl"
    firecrawl_config.save({"api_key": _REAL_FIRECRAWL_KEY, "base_url": base_url})
    backend = FirecrawlChannelBackend()
    assert backend.connect()

    # Finish the migration as the import CLI instructs: delete the
    # plaintext key, keeping only non-secret metadata.
    firecrawl_config.path.write_text(json.dumps({"base_url": base_url}))
    backend2 = FirecrawlChannelBackend()
    assert backend2.connect()
    assert backend2._base_url == base_url

    # Reverse-proxied read endpoints still classify as reads.
    result = json.loads(backend2.firecrawl_scrape("https://example.com"))
    assert result["ok"] is True
    assert api_server.requests[-1]["path"] == "/proxy/firecrawl/v2/scrape"
    assert api_server.header("Authorization") == f"Bearer {_REAL_FIRECRAWL_KEY}"

    # GET endpoints (crawl status) are reads too.
    status = json.loads(backend2.firecrawl_get_crawl_status("crawl-1"))
    assert status["ok"] is True
    assert api_server.requests[-1]["method"] == "GET"

    # A malformed metadata file falls back to the cloud default.
    firecrawl_config.path.write_text("[]")
    backend3 = FirecrawlChannelBackend()
    assert backend3.connect()
    assert backend3._base_url == "https://api.firecrawl.dev"


def test_unknown_service_has_no_builtin_hosts(muse_env: Path) -> None:
    """A custom-named enrollment gets an empty built-in allowlist."""
    from kiss.agents.third_party_agents.muse_auth.client import (
        MuseBoundarySession,
        bearer_surrogate,
    )

    surrogate = bearer_surrogate("customsvc", "custom-real-token")
    assert surrogate.startswith("muse-sgt.customsvc.")
    session = MuseBoundarySession("customsvc")
    resp = session.get(
        "https://example.com/data",
        headers={"Authorization": f"Bearer {surrogate}"},
    )
    assert resp.status_code == 403
    assert "not in the 'customsvc' connector's allowlist" in resp.text


def test_corrupt_vault_entry_fails_closed(muse_env: Path) -> None:
    """A corrupted vault file yields no hosts and a clear resolution error."""
    from kiss.agents.third_party_agents.muse_auth.client import (
        MuseBoundarySession,
        bearer_surrogate,
    )

    surrogate = bearer_surrogate("brave_search", _REAL_BRAVE_KEY)
    vault_file = muse_auth_dir() / "vault" / "brave_search.json"
    vault_file.write_text("{not json")
    session = MuseBoundarySession("brave_search")
    # enrolled_hosts falls back to () (the built-in host still matches,
    # so Sentinel allows the read) and credential resolution then fails
    # closed without leaking anything.
    with pytest.raises(MuseAuthError, match="credential resolution failed"):
        session.get(
            "https://api.search.brave.com/res/v1/web/search?q=x",
            headers={"Authorization": f"Bearer {surrogate}"},
            timeout=30.0,
        )


def test_store_credentials_validation(muse_env: Path) -> None:
    """The daemon rejects unsafe credential headers and enrollment hosts."""
    # A hop-by-hop/framing header can never carry a credential.
    with pytest.raises(MuseAuthError, match="invalid credential header"):
        _checked(
            {
                "op": "store_credentials",
                "service": "brave_search",
                "authorized_user_info": {"kind": "header", "header": "Host", "token": "x"},
                "scopes": [],
            }
        )
    # Hosts must be a list of plain hostnames.
    base = {
        "op": "store_credentials",
        "service": "firecrawl",
        "authorized_user_info": {"kind": "bearer", "token": "x"},
        "scopes": [],
    }
    with pytest.raises(MuseAuthError, match="hosts must be a list"):
        _checked({**base, "hosts": {"evil.example": 1}})
    with pytest.raises(MuseAuthError, match="invalid enrollment host"):
        _checked({**base, "hosts": ["evil host/path"]})
    with pytest.raises(MuseAuthError, match="invalid enrollment host"):
        _checked({**base, "hosts": [123]})
    with pytest.raises(MuseAuthError, match="at most 16"):
        _checked({**base, "hosts": [f"h{i}.example" for i in range(17)]})
    assert not vault_has_credentials("firecrawl")
    # IP literals — including IPv6 — are valid enrollment hosts.
    _checked({**base, "hosts": ["::1", "127.0.0.1", "fc.internal"]})
    assert vault_has_credentials("firecrawl")
    clear_credentials("firecrawl")


def test_cli_import_slack_and_token_services(muse_env: Path,
                                             capsys: pytest.CaptureFixture) -> None:
    """The CLI migrates slack/firecrawl/brave_search legacy credentials."""
    # slack: default workspace token file is migrated and deleted.
    assert muse_cli.main(["import", "slack"]) == 1  # no legacy token yet
    _save_token(_REAL_SLACK_TOKEN, "default")
    assert muse_cli.main(["import", "slack"]) == 0
    assert not _token_path("default").exists()
    assert vault_has_credentials("slack")

    # firecrawl: config key + self-hosted base_url host are imported.
    assert muse_cli.main(["import", "firecrawl"]) == 1  # no legacy config yet
    firecrawl_config.save({"api_key": _REAL_FIRECRAWL_KEY, "base_url": "https://fc.internal"})
    assert muse_cli.main(["import", "firecrawl"]) == 0
    assert vault_has_credentials("firecrawl")

    # brave_search: header-kind import; a blank key is refused.
    brave_config.save({"api_key": " "})
    (muse_auth_dir().parent / "third_party_agents" / "brave_search" / "config.json").write_text(
        json.dumps({"api_key": ""})
    )
    assert muse_cli.main(["import", "brave_search"]) == 1
    brave_config.save({"api_key": _REAL_BRAVE_KEY})
    assert muse_cli.main(["import", "brave_search"]) == 0
    assert vault_has_credentials("brave_search")

    # status lists all three among the enrolled services.
    assert muse_cli.main(["status"]) == 0
    status_out = capsys.readouterr().out
    for service in ("slack", "firecrawl", "brave_search"):
        assert service in status_out


def test_legacy_mode_untouched(isolated_kiss_home: Path, api_server: _ApiServer,
                               monkeypatch: pytest.MonkeyPatch) -> None:
    """With KISS_MUSE_AUTH=0 the connectors use plaintext directly."""
    from kiss.agents.third_party_agents.slack_sea import SlackAgent, _make_backend

    monkeypatch.setenv("KISS_MUSE_AUTH", "0")
    api_base_url = f"http://127.0.0.1:{api_server.server_address[1]}/api/"
    empty = SlackChannelBackend()
    assert not empty.connect()
    assert "No Slack token found" in empty._connection_info

    agent = SlackAgent()
    agent._backend._api_base_url = api_base_url
    tools = auth_tools(agent)
    assert json.loads(tools["authenticate_slack"](_REAL_SLACK_TOKEN))["ok"] is True
    assert agent._backend._client is not None
    assert agent._backend._client.token == _REAL_SLACK_TOKEN

    backend = SlackChannelBackend()
    backend._api_base_url = api_base_url
    assert backend.connect()
    assert backend._client is not None
    assert backend._client.token == _REAL_SLACK_TOKEN
    assert _make_backend("default")._client is not None

    brave_config.save({"api_key": _REAL_BRAVE_KEY})
    brave = BraveSearchChannelBackend()
    brave._base_url = f"http://127.0.0.1:{api_server.server_address[1]}/res/v1"
    assert brave.connect()
    assert brave._api_key == _REAL_BRAVE_KEY
    assert json.loads(brave.brave_web_search("q"))["ok"] is True
    assert api_server.header("X-Subscription-Token") == _REAL_BRAVE_KEY
    assert api_server.header("Authorization") == ""

    firecrawl_config.save({"api_key": _REAL_FIRECRAWL_KEY})
    crawler = FirecrawlChannelBackend()
    assert crawler.connect()
    assert crawler._api_key == _REAL_FIRECRAWL_KEY
    # No daemon socket appeared: nothing routed through Muse.
    assert not socket_path().exists()
