"""A failed redirect-port bind leaves no cancelled session published as active.

``MCPLoginSession._replace_active`` cancels the previous session before
binding the redirect listener for the new one.  When that bind failed,
``_active`` still named the cancelled session, so ``active(name)``
returned it and ``finish_mcp_server_connect`` reported "sign-in
cancelled" for a sign-in that never started instead of "no sign-in in
progress".  The cancelled session is now unpublished before the bind.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar.mcp_oauth import MCPLoginSession, make_mcp_auth_tools
from kiss.agents.sorcar.mcp_servers import MCPServerConfig
from kiss.tests.conftest import find_free_port


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test KISS home; headless so no default browser is ever launched."""
    kiss_home = tmp_path / "kiss_home"
    kiss_home.mkdir()
    monkeypatch.setenv("KISS_HOME", str(kiss_home))
    monkeypatch.setenv("KISS_HEADLESS", "1")
    monkeypatch.delenv("KISS_MCP_CLIENT_METADATA_URL", raising=False)
    return kiss_home


@pytest.fixture(autouse=True)
def _own_redirect_port(mcp_redirect_port: int) -> Iterator[int]:
    """Own ephemeral redirect port; cancel whatever session the test left running."""
    yield mcp_redirect_port
    session = MCPLoginSession._active
    if session is not None:
        session.cancel()
        session.wait(10)
    MCPLoginSession._active = None


def test_bind_failure_unpublishes_the_cancelled_session(
    home: Path, mcp_redirect_port: int, tmp_path: Path,
) -> None:
    """After a bind failure ``active()`` is ``None`` and finish says so."""
    # Nothing listens on this port: the first session's login thread
    # fails at once with a connection error and releases the listener.
    cfg = MCPServerConfig(
        name="deadmcp", transport="http", url=f"http://127.0.0.1:{find_free_port()}/mcp",
    )
    first = MCPLoginSession.start(cfg, wait_seconds=10.0)
    assert first.wait(10.0)
    assert MCPLoginSession.active("deadmcp") is first

    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        blocker.bind(("127.0.0.1", mcp_redirect_port))
        blocker.listen(1)
        with pytest.raises(OSError):
            MCPLoginSession.start(cfg, wait_seconds=1.0)
    finally:
        blocker.close()

    assert first._cancelled
    assert MCPLoginSession.active("deadmcp") is None
    _connect, finish = make_mcp_auth_tools(str(tmp_path))
    answer = json.loads(finish("deadmcp"))
    assert answer["ok"] is False
    assert "no sign-in in progress" in answer["error"]
