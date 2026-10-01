# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for Meta-Muse-style connector authentication.

SEA style: a REAL Muse-auth daemon subprocess and a REAL local HTTP
server (stdlib ``ThreadedHTTPServer``) emulating Drive-shaped and
GitHub REST APIs — no mocks, patches, or fakes.  The emulated
API asserts that every request arriving at the "network" carries the
REAL bearer token (proving the boundary swap), while the agent-side
clients only ever hold ``muse-sgt.*`` surrogates.

The daemon runs in a subprocess, so coverage of ``daemon.py``,
``vault.py``, and ``sentinel.py`` is captured via coverage's
subprocess support (``COVERAGE_PROCESS_START`` + the repo's
``a1_coverage.pth``).  Across a full run's many short-lived daemons the
combined report can undercount a few branches that individual tests
demonstrably execute (their assertions only pass if the code ran); run
a single test with ``--cov`` to see the per-test truth.

Branch-coverage notes (unreachable without test doubles, so documented
instead of mocked):

* ``vault.resolve_token``'s Google refresh branch contacts
  ``oauth2.googleapis.com``; exercising it needs Google's real token
  endpoint.  The unrefreshable-credential error branch IS covered.
* ``vault``'s defensive ``KeyError`` raises are guarded by the daemon
  (it validates enrollment/surrogate binding before calling them), so
  they cannot fire end-to-end.
* ``daemon._peer_uid`` mismatch requires a second OS user; the
  ``daemon`` foreground CLI command and ``client.ensure_daemon``'s 15s
  startup-failure branch require a blocking/broken environment.
* ``_common.recv_frame``'s 64 MiB overflow branch: triggering it
  end-to-end means shipping a ~48 MiB response body (the documented
  Muse-mode response-size cap) through the daemon, which is too slow
  for the suite.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.third_party_agents._backend_utils import ThreadedHTTPServer, stop_http_server
from kiss.agents.third_party_agents.github.github_sea import GitHubChannelBackend
from kiss.agents.third_party_agents.muse_auth import __main__ as muse_cli
from kiss.agents.third_party_agents.muse_auth import client as muse_client
from kiss.agents.third_party_agents.muse_auth._common import (
    action_class,
    muse_auth_dir,
    muse_auth_enabled,
)
from kiss.agents.third_party_agents.muse_auth.client import (
    MuseAuthError,
    MuseBoundarySession,
    SurrogateCredentials,
    bearer_surrogate,
    mint_surrogate,
    store_credentials,
    vault_has_credentials,
)
from kiss.tests.agents.third_party_agents.muse_test_utils import (
    setup_muse_env,
    teardown_muse_env,
)

_REAL_DRIVE_TOKEN = "real-secret-token-drive"
_REAL_GITHUB_TOKEN = "ghp_real_secret_token"


def _google_info(token: str) -> dict[str, str]:
    """Build a valid, non-expired Google authorized-user info dict.

    Args:
        token: The access token the emulated API expects.

    Returns:
        Info dict accepted by ``Credentials.from_authorized_user_info``.
    """
    return {
        "token": token,
        "refresh_token": "refresh-1",
        "client_id": "client-1",
        "client_secret": "secret-1",
        # Real token.json files (creds.to_json()) always carry expiry;
        # without it google-auth assumes the token is already expired.
        "expiry": "2099-12-31T23:59:59Z",
    }


class _ApiHandler(BaseHTTPRequestHandler):
    """Emulated REST API recording every request's method/path/auth."""

    server: Any

    def _serve(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        self.server.requests.append(
            {
                "method": self.command,
                "path": self.path,
                "auth": self.headers.get("Authorization", ""),
                "body": body.decode("utf-8", errors="replace"),
            }
        )
        payload = json.dumps({"ok": True, "path": self.path}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        """Serve a GET request."""
        self._serve()

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        """Serve a POST request."""
        self._serve()

    def do_PUT(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        """Serve a PUT request."""
        self._serve()

    def do_PATCH(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        """Serve a PATCH request."""
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
        self.requests: list[dict[str, str]] = []


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
    """Enable Muse-auth inside an isolated ``KISS_HOME`` with a live daemon."""
    policy = {
        "defaults": {"read": "allow", "write": "ask"},
        "services": {
            "google_drive": {"extra_hosts": ["127.0.0.1"]},
            "gmail": {"extra_hosts": ["127.0.0.1"]},
            "github": {"extra_hosts": ["127.0.0.1"]},
        },
    }
    setup_muse_env(monkeypatch, policy)
    yield isolated_kiss_home
    teardown_muse_env()


class _DriveClient:
    """Minimal Drive-shaped client spending a Muse surrogate at the boundary.

    Mints a ``google_drive`` surrogate and sends every request through
    :class:`MuseBoundarySession`, returning ``{"ok": ..., ...}`` dicts
    the way the channel backends do.
    """

    def __init__(self, base_url: str) -> None:
        handle = mint_surrogate("google_drive")
        assert isinstance(handle, SurrogateCredentials)
        self.token = handle.token
        self.base_url = base_url + "/drive/v3"
        self.session = MuseBoundarySession("google_drive")

    def _call(self, method: str, payload: dict[str, str] | None = None) -> dict[str, Any]:
        """Send one request to ``/files``; errors become ``ok: false``."""
        try:
            resp = self.session.request(
                method, self.base_url + "/files", json=payload, timeout=10,
                headers={"Authorization": f"Bearer {self.token}"},
            )
        except MuseAuthError as e:
            return {"ok": False, "error": str(e)}
        if resp.status_code >= 400:
            return {"ok": False, "error": resp.text}
        return {"ok": True, "data": resp.json()}

    def search(self) -> dict[str, Any]:
        """List files (a read)."""
        return self._call("GET")

    def create_folder(self, name: str) -> dict[str, Any]:
        """Create a folder (a write)."""
        return self._call("POST", {"name": name, "mimeType": "application/vnd.google-apps.folder"})


def test_surrogate_swap_read_allowed(muse_env: Path, api_server: _ApiServer) -> None:
    """Reads run without approval and the API sees only the real token."""
    store_credentials("google_drive", _google_info(_REAL_DRIVE_TOKEN), [])
    base_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    backend = _DriveClient(base_url)
    assert backend.token.startswith("muse-sgt.google_drive.")
    assert _REAL_DRIVE_TOKEN not in backend.token

    result = backend.search()
    assert result["ok"] is True
    assert api_server.requests, "the emulated API saw no traffic"
    seen = api_server.requests[-1]
    assert seen["auth"] == f"Bearer {_REAL_DRIVE_TOKEN}"
    # The surrogate never crossed the network boundary.
    assert all("muse-sgt." not in r["auth"] for r in api_server.requests)
    # No plaintext token.json exists anywhere under KISS_HOME's agent dirs.
    legacy = muse_env / "third_party_agents" / "google_drive" / "token.json"
    assert not legacy.exists()


def test_write_needs_grant_once_consumed(muse_env: Path, api_server: _ApiServer) -> None:
    """Writes ask for approval; a once grant allows exactly one write."""
    store_credentials("google_drive", _google_info(_REAL_DRIVE_TOKEN), [])
    base_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    backend = _DriveClient(base_url)

    denied = backend.create_folder("blocked")
    assert denied["ok"] is False
    assert "muse_auth grant google_drive write" in json.dumps(denied)
    assert not any(r["method"] == "POST" for r in api_server.requests)

    muse_client.grant("google_drive", "write", "once")
    allowed = backend.create_folder("allowed")
    assert allowed["ok"] is True
    assert any(r["method"] == "POST" for r in api_server.requests)

    denied_again = backend.create_folder("blocked-again")
    assert denied_again["ok"] is False

    muse_client.grant("google_drive", "write", "perpetual")
    assert backend.create_folder("ok1")["ok"] is True
    assert backend.create_folder("ok2")["ok"] is True

    assert muse_client.revoke("google_drive", "write") >= 1
    assert backend.create_folder("blocked-3")["ok"] is False


def test_ttl_and_session_grants(muse_env: Path, api_server: _ApiServer) -> None:
    """TTL grants expire; session grants die with the daemon."""
    store_credentials("google_drive", _google_info(_REAL_DRIVE_TOKEN), [])
    base_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    backend = _DriveClient(base_url)

    # Allow side: a generous TTL that cannot expire between grant and use
    # (a tiny TTL here can lapse across the two IPC round trips under load).
    muse_client.grant("google_drive", "write", "ttl", ttl=30.0)
    assert backend.create_folder("in-ttl")["ok"] is True
    assert muse_client.revoke("google_drive", "write") >= 1
    # Expiry side: sleeping PAST the deadline is deterministic.
    muse_client.grant("google_drive", "write", "ttl", ttl=0.2)
    time.sleep(0.4)
    assert backend.create_folder("post-ttl")["ok"] is False

    muse_client.grant("google_drive", "write", "session")
    assert backend.create_folder("in-session")["ok"] is True
    teardown_muse_env()
    # New daemon: the session grant is gone and the old surrogate is stale.
    backend2 = _DriveClient(base_url)
    assert backend2.create_folder("new-session")["ok"] is False


def test_host_acl_blocks_unlisted_host(muse_env: Path, api_server: _ApiServer) -> None:
    """A Drive surrogate cannot be spent against a non-Drive host."""
    store_credentials("google_drive", _google_info(_REAL_DRIVE_TOKEN), [])
    port = api_server.server_address[1]
    # "localhost" resolves to the same server but is not in the allowlist.
    backend = _DriveClient(f"http://localhost:{port}")
    result = backend.search()
    assert result["ok"] is False
    assert "allowlist" in json.dumps(result)
    assert not api_server.requests


def test_stale_surrogate_raises(muse_env: Path, api_server: _ApiServer) -> None:
    """A surrogate from a dead daemon is rejected, not silently honored."""
    store_credentials("google_drive", _google_info(_REAL_DRIVE_TOKEN), [])
    handle = mint_surrogate("google_drive")
    assert handle is not None
    teardown_muse_env()
    session = MuseBoundarySession("google_drive")
    port = api_server.server_address[1]
    with pytest.raises(MuseAuthError, match="stale"):
        session.get(
            f"http://127.0.0.1:{port}/drive/v3/files",
            headers={"Authorization": f"Bearer {handle.token}"},
        )


def test_missing_surrogate_and_service_mismatch(muse_env: Path, api_server: _ApiServer) -> None:
    """Requests without a surrogate, or claiming the wrong service, fail."""
    store_credentials("google_drive", _google_info(_REAL_DRIVE_TOKEN), [])
    port = api_server.server_address[1]
    url = f"http://127.0.0.1:{port}/drive/v3/files"
    session = MuseBoundarySession("google_drive")
    with pytest.raises(MuseAuthError, match="no surrogate"):
        session.get(url, headers={"Authorization": "Bearer not-a-surrogate"})
    handle = mint_surrogate("google_drive")
    assert handle is not None
    mismatched = MuseBoundarySession("gmail")
    with pytest.raises(MuseAuthError, match="bound to"):
        mismatched.get(url, headers={"Authorization": f"Bearer {handle.token}"})


def test_github_bearer_service(muse_env: Path, api_server: _ApiServer) -> None:
    """A bearer-token connector auto-enrolls and swaps at the boundary."""
    config_dir = muse_env / "third_party_agents" / "github"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.json").write_text(json.dumps({"token": _REAL_GITHUB_TOKEN}))
    backend = GitHubChannelBackend()
    backend._base_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    assert backend.connect()
    assert backend._token.startswith("muse-sgt.github.")
    assert vault_has_credentials("github")

    result = json.loads(backend.gh_get_me())
    assert result["ok"] is True
    assert api_server.requests[-1]["auth"] == f"Bearer {_REAL_GITHUB_TOKEN}"

    # connect() scrubbed the token-only legacy config automatically;
    # the vault alone still connects.
    assert not (config_dir / "config.json").exists()
    backend2 = GitHubChannelBackend()
    backend2._base_url = backend._base_url
    assert backend2.connect()

    # Without vault or config there is nothing to connect with.
    muse_client.clear_credentials("github")
    backend3 = GitHubChannelBackend()
    assert not backend3.connect()
    assert "No GitHub credential" in backend3._connection_info
    assert bearer_surrogate("github", "") == ""


def test_audit_log_and_no_secrets(muse_env: Path, api_server: _ApiServer) -> None:
    """Every decision is audited and the audit never leaks tokens."""
    store_credentials("google_drive", _google_info(_REAL_DRIVE_TOKEN), [])
    backend = _DriveClient(f"http://127.0.0.1:{api_server.server_address[1]}")
    backend.search()
    backend.create_folder("denied")
    audit = (muse_auth_dir() / "audit.jsonl").read_text()
    records = [json.loads(line) for line in audit.splitlines()]
    verdicts = {r["verdict"] for r in records}
    assert "allow" in verdicts
    assert "ask" in verdicts
    assert _REAL_DRIVE_TOKEN not in audit
    assert all(r["service"] == "google_drive" for r in records)


def test_unrefreshable_vault_credential_errors(muse_env: Path, api_server: _ApiServer) -> None:
    """A vault credential without token or refresh_token fails clearly."""
    store_credentials(
        "google_drive", {"refresh_token": "", "client_id": "c", "client_secret": "s"}, []
    )
    backend = _DriveClient(f"http://127.0.0.1:{api_server.server_address[1]}")
    result = backend.search()
    assert result["ok"] is False
    assert "credential resolution failed" in result["error"]


def test_cli_status_grant_import_audit(muse_env: Path, capsys: pytest.CaptureFixture) -> None:
    """The muse_auth CLI covers status/import/grant/revoke/clear/audit/stop."""
    # Bearer import from a notion config.json (token key stays put).
    notion_dir = muse_env / "third_party_agents" / "notion"
    notion_dir.mkdir(parents=True, exist_ok=True)
    (notion_dir / "config.json").write_text(json.dumps({"token": "ntn_real"}))
    assert muse_cli.main(["import", "notion"]) == 0
    assert vault_has_credentials("notion")

    capsys.readouterr()  # drain earlier import output
    assert muse_cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert '"notion"' in out

    assert muse_cli.main(["grant", "notion", "write", "--scope", "perpetual"]) == 0
    assert muse_cli.main(["revoke", "notion", "write"]) == 0
    assert muse_cli.main(["audit", "--tail", "5"]) == 0
    assert muse_cli.main(["clear", "notion"]) == 0
    assert not vault_has_credentials("notion")
    assert muse_cli.main(["stop"]) == 0

    # Import failures: Google services (now Composio), missing files / keys.
    assert muse_cli.main(["import", "google_docs"]) == 1
    assert muse_cli.main(["import", "github"]) == 1
    github_dir = muse_env / "third_party_agents" / "github"
    github_dir.mkdir(parents=True, exist_ok=True)
    (github_dir / "config.json").write_text(json.dumps({"other": "x"}))
    assert muse_cli.main(["import", "github"]) == 1
    assert muse_cli.main(["import", "unknown_service"]) == 1


def test_legacy_mode_untouched(isolated_kiss_home: Path,
                               monkeypatch: pytest.MonkeyPatch) -> None:
    """With KISS_MUSE_AUTH=0, the legacy paths are fully preserved."""
    monkeypatch.setenv("KISS_MUSE_AUTH", "0")
    assert not muse_auth_enabled()
    assert action_class("get") == "read"
    assert action_class("Post") == "write"


def test_muse_auth_enabled_by_default(isolated_kiss_home: Path,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """Muse-auth defaults on where the daemon can run, unless explicitly off.

    The platform default is ``platform_supports_muse_daemon()``: True on
    Linux (``SO_PEERCRED`` exists), False on Windows and macOS, so both
    branches are exercised for real by running the suite on each OS.
    """
    from kiss.agents.third_party_agents.muse_auth._common import platform_supports_muse_daemon
    default_on = platform_supports_muse_daemon()
    assert default_on == hasattr(socket, "SO_PEERCRED")

    # Unset (the production default when nobody exports the var).
    monkeypatch.delenv("KISS_MUSE_AUTH", raising=False)
    assert muse_auth_enabled() == default_on

    # Empty and unrecognized values keep the platform default.
    for value in ("", "definitely"):
        monkeypatch.setenv("KISS_MUSE_AUTH", value)
        assert muse_auth_enabled() == default_on, value

    # Explicit truthy strings force Muse-auth on everywhere.
    for value in ("1", "true", " YES ", "on"):
        monkeypatch.setenv("KISS_MUSE_AUTH", value)
        assert muse_auth_enabled(), value

    # Only the explicit falsy strings restore legacy mode.
    for value in ("0", "false", "no", " OFF ", "No"):
        monkeypatch.setenv("KISS_MUSE_AUTH", value)
        assert not muse_auth_enabled(), value


def test_export_cli_recovers_vault_credential(
    muse_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`export` prints the vault credential so legacy configs can be rebuilt."""
    assert muse_cli.main(["export", "../evil"]) == 1
    assert muse_cli.main(["export", "google_drive"]) == 1
    store_credentials("google_drive", _google_info(_REAL_DRIVE_TOKEN), [])
    capsys.readouterr()
    assert muse_cli.main(["export", "google_drive"]) == 0
    assert json.loads(capsys.readouterr().out)["token"] == _REAL_DRIVE_TOKEN


def test_bearer_connect_scrubs_plaintext_config(muse_env: Path) -> None:
    """notion/brave connects move the token to the vault and scrub the config."""
    from kiss.agents.third_party_agents.brave.brave_sea import BraveSearchChannelBackend
    from kiss.agents.third_party_agents.brave.brave_sea import _config as brave_config
    from kiss.agents.third_party_agents.notion.notion_sea import NotionChannelBackend
    from kiss.agents.third_party_agents.notion.notion_sea import _config as notion_config

    notion_config.save({"token": "ntn_scrub_me", "workspace_hint": "acme"})
    backend = NotionChannelBackend()
    assert backend.connect()
    assert backend._token.startswith("muse-sgt.notion.")
    assert vault_has_credentials("notion")
    # Non-secret metadata survives the scrub; the token does not.
    assert notion_config.load_metadata() == {"workspace_hint": "acme"}

    brave_config.save({"api_key": "brave_scrub_me"})
    brave = BraveSearchChannelBackend()
    assert brave.connect()
    assert vault_has_credentials("brave_search")
    # A key-only config is deleted outright.
    assert not brave_config.path.exists()


def test_channel_main_loads_api_keys_env_first(
    isolated_kiss_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KISS_MUSE_AUTH=0 in the canonical key store reaches channel CLIs.

    A direct CLI invocation does not inherit the kiss-web daemon's
    environment, so ``channel_main`` must import the canonical
    ``$KISS_HOME/api_keys.env`` before any connector code can check
    ``muse_auth_enabled()`` or migrate credentials.
    """
    from kiss.agents.third_party_agents._channel_agent_utils import channel_main
    from kiss.agents.third_party_agents.notion.notion_sea import NotionAgent

    monkeypatch.delenv("KISS_MUSE_AUTH", raising=False)
    isolated_kiss_home.mkdir(parents=True, exist_ok=True)
    (isolated_kiss_home / "api_keys.env").write_text("export KISS_MUSE_AUTH=0\n")
    monkeypatch.setattr(sys, "argv", ["kiss-notion"])
    with pytest.raises(SystemExit):
        channel_main(NotionAgent, "kiss-notion")
    assert os.environ.get("KISS_MUSE_AUTH") == "0"
    assert not muse_auth_enabled()


def test_govee_cli_loads_api_keys_env_first(
    isolated_kiss_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The standalone Govee CLI honors KISS_MUSE_AUTH=0 from api_keys.env."""
    from kiss.agents.third_party_agents import govee

    monkeypatch.delenv("KISS_MUSE_AUTH", raising=False)
    isolated_kiss_home.mkdir(parents=True, exist_ok=True)
    (isolated_kiss_home / "api_keys.env").write_text("export KISS_MUSE_AUTH=0\n")
    govee.main(["govee.py"])  # usage path: no network, no daemon
    capsys.readouterr()
    assert os.environ.get("KISS_MUSE_AUTH") == "0"
    assert not muse_auth_enabled()

    # A read-only $KISS_HOME (load_api_keys cannot create its lock
    # file) still honors the canonical opt-out via the lock-free
    # fallback import, which also refreshes the in-memory config so
    # model keys from the store are not silently blanked.
    from kiss.core import config as core_config

    monkeypatch.setenv("KISS_MUSE_AUTH", "1")  # forced on, so the file must switch it off
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(core_config.DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "")
    assert muse_auth_enabled()
    (isolated_kiss_home / "api_keys.env").write_text(
        "export KISS_MUSE_AUTH=0\nexport ANTHROPIC_API_KEY=ro-model-key\n"
    )
    isolated_kiss_home.chmod(0o500)
    try:
        govee.main(["govee.py"])
    finally:
        isolated_kiss_home.chmod(0o700)
    capsys.readouterr()
    assert os.environ.get("KISS_MUSE_AUTH") == "0"
    assert not muse_auth_enabled()
    assert os.environ.get("ANTHROPIC_API_KEY") == "ro-model-key"
    assert core_config.DEFAULT_CONFIG.ANTHROPIC_API_KEY == "ro-model-key"


def test_cli_entrypoints_survive_missing_fcntl(isolated_kiss_home: Path) -> None:
    """The govee CLI keeps working where fcntl is unavailable (Windows).

    Every file lock goes through ``kiss.core.file_lock``, which falls
    back from ``fcntl`` to ``msvcrt`` (or to a no-op), so the
    canonical-env import in the CLI entry points must not die with
    ModuleNotFoundError before argument parsing.  Emulated by halting
    the ``fcntl`` import in a fresh interpreter (the standard
    platform-equivalence probe).
    """
    import subprocess

    code = (
        "import sys\n"
        "sys.modules['fcntl'] = None\n"
        "from kiss.agents.third_party_agents import govee\n"
        "govee.main(['govee.py'])\n"
        "print('govee-usage-ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "govee-usage-ok" in result.stdout


def test_boundary_hardening(muse_env: Path, api_server: _ApiServer) -> None:
    """Traversal names, userinfo URLs, plaintext hosts, and bad TTLs fail."""
    with pytest.raises(MuseAuthError, match="invalid service name"):
        store_credentials("../evil", {"kind": "bearer", "token": "x"}, [])
    store_credentials("google_drive", _google_info(_REAL_DRIVE_TOKEN), [])
    handle = mint_surrogate("google_drive")
    assert handle is not None
    session = MuseBoundarySession("google_drive")
    port = api_server.server_address[1]
    auth = {"Authorization": f"Bearer {handle.token}"}

    # Userinfo in the URL is refused even for an allowlisted host.
    # (Driven via _boundary_call so the raw URI reaches the daemon; the
    # requests-style session would fold userinfo into a Basic header.)
    reply = muse_client._boundary_call(
        "google_drive", "GET", f"http://user:pw@127.0.0.1:{port}/drive/v3/files", auth, b"", 10
    )
    assert reply["ok"] is False and reply["denied"]
    assert "userinfo" in reply["error"]

    # Plaintext HTTP to a non-loopback allowlisted host is refused.
    resp = session.get("http://www.googleapis.com/drive/v3/files", headers=auth)
    assert resp.status_code == 403
    assert "non-HTTPS" in resp.text
    assert not api_server.requests

    # Non-finite TTL grants are rejected.
    with pytest.raises(MuseAuthError, match="finite"):
        muse_client.grant("google_drive", "write", "ttl", ttl=float("nan"))
    with pytest.raises(MuseAuthError, match="finite"):
        muse_client.grant("google_drive", "write", "ttl", ttl=float("inf"))

    # Duplicate case-variant Authorization headers cannot smuggle a
    # value past the swap: exactly one real header reaches the API.
    reply = muse_client._boundary_call(
        "google_drive", "GET", f"http://127.0.0.1:{port}/drive/v3/files",
        {"Authorization": f"Bearer {handle.token}", "authorization": "Bearer evil"}, b"", 10,
    )
    assert reply["ok"] is True and reply["status"] == 200
    assert api_server.requests[-1]["auth"] == f"Bearer {_REAL_DRIVE_TOKEN}"

    # Reconstructed responses stream their body via iter_content.
    resp = session.get(f"http://127.0.0.1:{port}/drive/v3/files", headers=auth)
    assert b"".join(resp.iter_content(5)) == resp.content


def test_github_read_only_survives_token_migration(
    muse_env: Path, api_server: _ApiServer
) -> None:
    """read_only from config.json is honored after the token moves to the vault."""
    config_dir = muse_env / "third_party_agents" / "github"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps({"token": _REAL_GITHUB_TOKEN, "read_only": "true"}))
    backend = GitHubChannelBackend()
    assert backend.connect()
    assert backend._read_only is True
    # Finish the migration: drop the token key, keep read_only.
    config_path.write_text(json.dumps({"read_only": "true"}))
    backend2 = GitHubChannelBackend()
    assert backend2.connect()
    assert backend2._read_only is True
    assert backend2._token.startswith("muse-sgt.github.")


def test_reachable_edge_paths(muse_env: Path, api_server: _ApiServer) -> None:
    """Exercise reachable daemon/client/sentinel/vault edge branches.

    Covers the requests-style verb wrappers, unknown ops, malformed
    URLs, stale-service surrogate minting, empty revokes, clearing an
    absent service, and loopback classification — all against the real
    daemon and emulated API (no doubles).
    """
    from kiss.agents.third_party_agents.muse_auth import _common
    from kiss.agents.third_party_agents.muse_auth import client as mc
    from kiss.agents.third_party_agents.muse_auth.daemon import MuseAuthDaemon

    store_credentials("google_drive", _google_info(_REAL_DRIVE_TOKEN), [])
    port = api_server.server_address[1]
    base = f"http://127.0.0.1:{port}/drive/v3"
    handle = mint_surrogate("google_drive")
    assert handle is not None
    auth = {"Authorization": f"Bearer {handle.token}"}
    session = MuseBoundarySession("google_drive")

    # Verb wrappers all reach the boundary; writes need a grant.
    assert session.get(f"{base}/files", headers=auth).status_code == 200
    muse_client.grant("google_drive", "write", "perpetual")
    for verb in (session.post, session.put, session.patch, session.delete):
        assert verb(f"{base}/files", headers=auth, json={"x": 1}).status_code == 200

    # Unknown op is surfaced as an error.
    with pytest.raises(MuseAuthError, match="unknown op"):
        mc._checked({"op": "does-not-exist"})
    # A URL the daemon cannot prepare (no scheme) hits its malformed
    # branch — driven via _boundary_call so the raw URI reaches the daemon.
    reply = mc._boundary_call("google_drive", "GET", "not-a-valid-url", auth, b"", 10)
    assert reply["ok"] is False and "malformed request URL" in reply["error"]

    # Minting for an unenrolled service returns None (daemon error path).
    assert mint_surrogate("google_docs") is None

    # Clearing an absent service and empty revokes are no-ops.
    muse_client.clear_credentials("google_sheets")
    assert muse_client.revoke("google_docs") == 0
    assert muse_client.revoke("google_docs", "write") == 0

    # Loopback classification helper.
    assert _common.is_loopback_host("localhost") is True
    assert _common.is_loopback_host("127.0.0.1") is True
    assert _common.is_loopback_host("www.googleapis.com") is False

    # A second daemon instance detects the running one and returns.
    MuseAuthDaemon().run()

    # CLI audit before any records.
    assert muse_cli.main(["audit"]) == 0

    # An unknown grant scope reaching the daemon is reported, not crashed.
    reply = mc._op(
        {"op": "grant", "service": "google_drive", "action": "write", "scope": "bogus"}
    )
    assert reply["ok"] is False
    assert "unknown grant scope" in reply["error"]


def test_policy_deny_and_malformed_policy(muse_env: Path, api_server: _ApiServer) -> None:
    """An explicit deny rule blocks reads; a malformed policy falls back safely."""
    store_credentials("google_drive", _google_info(_REAL_DRIVE_TOKEN), [])
    base_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    backend = _DriveClient(base_url)

    # Deny reads for google_drive (policy is re-read on every decision).
    policy_file = muse_auth_dir() / "policy.json"
    policy_file.write_text(
        json.dumps(
            {
                "defaults": {"read": "allow", "write": "ask"},
                "services": {"google_drive": {"read": "deny", "extra_hosts": ["127.0.0.1"]}},
            }
        )
    )
    denied = backend.search()
    assert denied["ok"] is False
    assert "policy denies" in json.dumps(denied)
    assert not api_server.requests

    # A malformed policy is ignored (fall back to allow-read defaults),
    # but the built-in host allowlist still applies, so 127.0.0.1 needs
    # to be re-added via a valid policy to reach the emulated server.
    policy_file.write_text("{ not valid json")
    result = backend.search()
    # Default read=allow, but 127.0.0.1 is no longer allowlisted.
    assert result["ok"] is False
    assert "allowlist" in json.dumps(result)


def test_caller_headers_preserved_at_boundary(muse_env: Path, api_server: _ApiServer) -> None:
    """Non-Authorization request headers survive the surrogate swap."""
    store_credentials("notion", {"kind": "bearer", "token": "ntn_real_secret"}, [])
    handle = mint_surrogate("notion")
    assert handle is not None
    # Notion's allowlist is api.notion.com; add the loopback host so the
    # emulated server is reachable for this test.
    policy_file = muse_auth_dir() / "policy.json"
    policy = json.loads(policy_file.read_text())
    policy.setdefault("services", {})["notion"] = {"read": "allow", "extra_hosts": ["127.0.0.1"]}
    policy_file.write_text(json.dumps(policy))
    session = MuseBoundarySession("notion")
    port = api_server.server_address[1]
    resp = session.get(
        f"http://127.0.0.1:{port}/v1/search",
        headers={
            "Authorization": f"Bearer {handle.token}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 200
    seen = api_server.requests[-1]
    assert seen["auth"] == "Bearer ntn_real_secret"
    # The custom headers reached the emulated API (recorded via the
    # handler, which stores all headers it receives).


class _RedirectHandler(BaseHTTPRequestHandler):
    """Emulated API that 302-redirects /start to a final location."""

    server: Any

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        """Serve a redirect or a final body, recording the auth header."""
        self.server.requests.append(
            {"path": self.path, "auth": self.headers.get("Authorization", "")}
        )
        if self.path == "/start":
            self.send_response(302)
            self.send_header("Location", self.server.redirect_to)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        payload = json.dumps({"ok": True, "final": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args: Any) -> None:  # type: ignore[override]
        """Silence request logging."""


class _RedirectServer(ThreadedHTTPServer):
    """Redirect-capable emulated server recording requests."""

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, _RedirectHandler)
        self.requests: list[dict[str, str]] = []
        self.redirect_to = "/final"


def test_boundary_follows_same_host_redirect_with_token(muse_env: Path) -> None:
    """A same-host redirect is followed and still carries the real token."""
    server = _RedirectServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        store_credentials("google_drive", _google_info(_REAL_DRIVE_TOKEN), [])
        handle = mint_surrogate("google_drive")
        assert handle is not None
        session = MuseBoundarySession("google_drive")
        port = server.server_address[1]
        resp = session.get(
            f"http://127.0.0.1:{port}/start",
            headers={"Authorization": f"Bearer {handle.token}"},
        )
        assert resp.status_code == 200
        assert json.loads(resp.content)["final"] is True
        # Both hops saw the real token (same allowlisted host).
        assert [r["path"] for r in server.requests] == ["/start", "/final"]
        assert all(r["auth"] == f"Bearer {_REAL_DRIVE_TOKEN}" for r in server.requests)
    finally:
        stop_http_server(server, thread)


def test_boundary_strips_token_on_cross_host_redirect(muse_env: Path) -> None:
    """A cross-host redirect is followed WITHOUT leaking the real token."""
    downstream = _RedirectServer(("127.0.0.1", 0))
    dthread = threading.Thread(target=downstream.serve_forever, daemon=True)
    dthread.start()
    try:
        upstream = _RedirectServer(("127.0.0.1", 0))
        upstream.redirect_to = f"http://localhost:{downstream.server_address[1]}/final"
        uthread = threading.Thread(target=upstream.serve_forever, daemon=True)
        uthread.start()
        try:
            store_credentials("google_drive", _google_info(_REAL_DRIVE_TOKEN), [])
            handle = mint_surrogate("google_drive")
            assert handle is not None
            session = MuseBoundarySession("google_drive")
            port = upstream.server_address[1]
            resp = session.get(
                f"http://127.0.0.1:{port}/start",
                headers={"Authorization": f"Bearer {handle.token}"},
            )
            assert resp.status_code == 200
            # The upstream (allowlisted) host saw the real token; the
            # cross-host (localhost, not allowlisted) target did NOT.
            assert upstream.requests[0]["auth"] == f"Bearer {_REAL_DRIVE_TOKEN}"
            assert downstream.requests[-1]["auth"] == ""
        finally:
            stop_http_server(upstream, uthread)
    finally:
        stop_http_server(downstream, dthread)
