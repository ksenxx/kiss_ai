# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Writing an MCP server entry never destroys the config file it edits.

Audit 2026-10-30 (S9).  ``save_mcp_server`` read a config it could not
parse (a hand-edited ``$KISS_HOME/mcp.json`` with a trailing comma, a
top level that is not an object) as ``{}`` and atomically replaced it
with the one new entry, wiping every other server and the API keys in
their ``env`` and ``headers``.  The ``connect_mcp_server`` sign-in tool
took that path for any server not configured yet.  Only a missing file
counts as empty now; anything else is an error naming the file, which
is left as it was.

The config edits also locked ``<file>.lock`` next to each file, which
left an untracked ``.mcp.lock`` beside Claude Code's checked-in
``<project>/.mcp.json`` (and ``.kiss/mcp.lock``) in the user's project.
Every MCP config edit now takes the one lock ``$KISS_HOME/mcp.lock``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kiss.agents.sorcar import mcp_oauth, mcp_servers

_UNPARSABLE = {
    "trailing comma": b'{"mcpServers": {"mine": {"command": "srv", "env": {"KEY": "k"}},}}',
    "not utf-8": b'{"mcpServers": {"mine": {"command": "\xff"}}}',
    "list top level": b'[{"mcpServers": {}}]',
    "list of servers": b'{"mcpServers": [{"command": "srv"}]}',
}


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A ``KISS_HOME`` of the test's own."""
    path = tmp_path / "home"
    monkeypatch.setenv("KISS_HOME", str(path))
    return path


@pytest.mark.parametrize("content", list(_UNPARSABLE.values()), ids=list(_UNPARSABLE))
def test_an_unparsable_config_is_reported_and_left_untouched(
    home: Path, tmp_path: Path, content: bytes,
) -> None:
    config = mcp_servers.user_mcp_config_path()
    config.parent.mkdir(parents=True)
    config.write_bytes(content)
    cfg = mcp_servers.MCPServerConfig(name="notion", transport="http", url="https://x/mcp")
    with pytest.raises(ValueError, match=str(config).replace("\\", "\\\\")):
        mcp_servers.save_mcp_server(cfg, "user", str(tmp_path))
    assert config.read_bytes() == content
    # The sign-in tool's path: resolving a server not configured yet saves it.
    with pytest.raises(ValueError, match="mcp.json"):
        mcp_oauth.resolve_server("notion", "", "", str(tmp_path))
    assert config.read_bytes() == content


def test_a_missing_or_partial_config_gets_the_entry_and_keeps_the_rest(
    home: Path, tmp_path: Path,
) -> None:
    cfg = mcp_servers.MCPServerConfig(name="notion", transport="http", url="https://x/mcp")
    config = mcp_servers.save_mcp_server(cfg, "user", str(tmp_path))
    assert config == mcp_servers.user_mcp_config_path()
    assert json.loads(config.read_text(encoding="utf-8")) == {
        "mcpServers": {"notion": cfg.to_json()},
    }
    project = mcp_servers.project_mcp_config_path(str(tmp_path))
    project.parent.mkdir(parents=True)
    project.write_text('{"other": 1}', encoding="utf-8")
    assert mcp_servers.save_mcp_server(cfg, "project", str(tmp_path)) == project
    assert json.loads(project.read_text(encoding="utf-8")) == {
        "other": 1, "mcpServers": {"notion": cfg.to_json()},
    }


def test_config_edits_leave_no_lock_file_in_the_project(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    claude = mcp_servers.claude_project_mcp_config_path(str(project))
    claude.write_text(json.dumps({"mcpServers": {"gh": {"command": "gh-mcp"}}}), "utf-8")
    cfg = mcp_servers.MCPServerConfig(name="gh", transport="stdio", command="gh-mcp")
    mcp_servers.save_mcp_server(cfg, "project", str(project))
    assert mcp_servers.remove_mcp_server("gh", str(project)) == [
        claude, mcp_servers.project_mcp_config_path(str(project)),
    ]
    assert mcp_servers.load_mcp_servers(str(project)) == {}
    assert sorted(p.name for p in project.rglob("*.lock")) == []
    assert (home / "mcp.lock").is_file()
