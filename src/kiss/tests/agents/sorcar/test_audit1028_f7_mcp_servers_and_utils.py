"""Audit F7: ``mcp_servers`` writes go through ``atomic_write_text``; deny before suffixing.

* ``save_mcp_server`` and ``FileTokenStorage`` used to re-implement the
  atomic write of :func:`kiss.core.utils.atomic_write_text`; the shared
  helper keeps an existing ``mcp.json``'s permission bits and forces
  ``0o600`` on the token file.
* ``make_mcp_tools`` judged the ``mcp_permissions`` rules on the
  suffixed name AFTER reserving it, so a denied tool reserved a suffix
  and a deny rule on ``web_search`` let the same tool through as
  ``web_search_2`` when the plain name was taken.
* :func:`kiss.core.utils.default_route_ip` replaces two copies of the
  UDP default-route probe.
"""

from __future__ import annotations

import ipaddress
import json
import os
import stat
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar.mcp_servers import (
    FileTokenStorage,
    MCPManager,
    MCPServerConfig,
    make_mcp_tools,
    save_mcp_server,
)
from kiss.core.utils import default_route_ip

_WEB_SERVER = '''
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("web")


@mcp.tool()
def search(query: str) -> str:
    """Search the web for a query."""
    return "results for " + query


if __name__ == "__main__":
    mcp.run()
'''


@pytest.fixture
def isolated_homes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every user-level MCP location into *tmp_path*."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / ".kisshome"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "project").mkdir()
    return tmp_path


@pytest.fixture
def real_stdin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Give ``sys.stdin`` and the MCP errlog real file descriptors."""
    from mcp.client.stdio import stdio_client

    stdin_stream = open(os.devnull, encoding="utf-8")
    errlog = (tmp_path / "mcp_errlog.txt").open("w", encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stdin_stream)
    monkeypatch.setattr(sys, "stderr", errlog)
    wrapped = stdio_client.__wrapped__  # type: ignore[attr-defined]
    monkeypatch.setattr(wrapped, "__defaults__", (errlog,))
    try:
        yield
    finally:
        errlog.close()
        stdin_stream.close()


@pytest.fixture
def mcp_permission_rules() -> Iterator[Callable[[dict[str, str]], None]]:
    """Write ``mcp_permissions`` rules to the real config and restore afterwards."""
    from kiss.core.user_config import CONFIG_PATH

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    original = CONFIG_PATH.read_text() if CONFIG_PATH.exists() else None

    def _write(rules: dict[str, str]) -> None:
        CONFIG_PATH.write_text(json.dumps({"mcp_permissions": rules}), encoding="utf-8")

    yield _write
    if original is None:
        CONFIG_PATH.unlink(missing_ok=True)
    else:
        CONFIG_PATH.write_text(original)


def _save_web_server(tmp_path: Path, project: Path) -> None:
    """Register the FastMCP ``web`` stdio server in the user scope."""
    script = tmp_path / "websrv.py"
    script.write_text(_WEB_SERVER, encoding="utf-8")
    save_mcp_server(
        MCPServerConfig(
            name="web", transport="stdio", command=sys.executable, args=(str(script),),
        ),
        "user",
        str(project),
    )


def test_denied_tool_is_judged_on_its_base_name_and_reserves_no_suffix(
    isolated_homes: Path,
    tmp_path: Path,
    real_stdin: None,
    mcp_permission_rules: Callable[[dict[str, str]], None],
) -> None:
    """A deny on ``web_search`` drops the tool even when it would be suffixed."""
    project = isolated_homes / "project"
    _save_web_server(tmp_path, project)
    try:
        mcp_permission_rules({"web_search": "deny"})
        assert make_mcp_tools(str(project), taken_names={"web_search"}) == []
        mcp_permission_rules({"web_search": "allow"})
        shifted = make_mcp_tools(str(project), taken_names={"web_search"})
        assert [tool.__name__ for tool in shifted] == ["web_search_2"]
        assert "results for kiss" in shifted[0](query="kiss")
    finally:
        MCPManager.instance().shutdown()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_config_save_keeps_existing_bits_and_token_file_is_private(
    isolated_homes: Path,
) -> None:
    """``mcp.json`` keeps a deliberate chmod; the token file is always ``0o600``."""
    project = isolated_homes / "project"
    path = save_mcp_server(MCPServerConfig(name="s", command="a"), "user", str(project))
    os.chmod(path, 0o640)
    save_mcp_server(MCPServerConfig(name="t", command="b"), "user", str(project))
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert set(json.loads(path.read_text())["mcpServers"]) == {"s", "t"}

    storage = FileTokenStorage("srv")
    storage.put("client_info", {"client_id": "abc"})
    assert stat.S_IMODE(storage.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(storage.path.parent.stat().st_mode) == 0o700
    os.chmod(storage.path, 0o644)
    storage.put("client_info", {"client_id": "def"})
    assert stat.S_IMODE(storage.path.stat().st_mode) == 0o600
    assert json.loads(storage.path.read_text())["client_info"] == {"client_id": "def"}


def test_default_route_ip_is_an_ipv4_address_or_none() -> None:
    """The probe answers with a real IPv4 address (``None`` only on an offline host)."""
    ip = default_route_ip()
    if ip is not None:
        assert ipaddress.ip_address(ip).version == 4
