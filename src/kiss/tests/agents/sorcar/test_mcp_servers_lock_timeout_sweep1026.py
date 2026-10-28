# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The MCP config and token-file locks wait for a wedged holder only so long.

``save_mcp_server``, ``remove_mcp_server``, ``FileTokenStorage.put`` and
``.clear`` take the cross-process ``flock`` next to the file they
rewrite.  Without a bound
a holder stopped in a debugger or on a stuck filesystem parked the tool
thread forever; they now wait ``mcp_servers._LOCK_TIMEOUT_S`` and raise
``TimeoutError``, like ``user_config`` and ``machines``.  The tests hold
the real lock file from this process (a second open file description
conflicts with the first under ``flock``) and lower the bound so the
wait is short.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import pytest

from kiss.agents.sorcar import mcp_servers
from kiss.core.file_lock import lock_exclusive, unlock


@pytest.fixture
def quick_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_servers, "_LOCK_TIMEOUT_S", 0.3)


def _assert_times_out(action: Callable[[], object], lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as held:
        lock_exclusive(held)
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="still locked"):
            action()
        elapsed = time.monotonic() - started
        unlock(held)
    assert 0.3 <= elapsed < 3.0, f"waited {elapsed:.2f}s"


def test_save_mcp_server_gives_up_on_a_wedged_project_lock(
    tmp_path: Path,
    quick_timeout: None,
) -> None:
    cfg = mcp_servers.MCPServerConfig(name="demo", transport="http", url="http://x/")
    lock_path = mcp_servers.project_mcp_config_path(str(tmp_path)).with_suffix(".lock")
    save = partial(mcp_servers.save_mcp_server, cfg, "project", str(tmp_path))
    _assert_times_out(save, lock_path)
    assert not mcp_servers.project_mcp_config_path(str(tmp_path)).exists()
    # Released: the same save goes through at once.
    saved = mcp_servers.save_mcp_server(cfg, "project", str(tmp_path))
    assert "demo" in mcp_servers.load_mcp_servers(str(tmp_path))
    assert saved == mcp_servers.project_mcp_config_path(str(tmp_path))


def test_remove_mcp_server_gives_up_on_a_wedged_project_lock(
    tmp_path: Path,
    quick_timeout: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home"))
    cfg = mcp_servers.MCPServerConfig(name="demo", transport="http", url="http://x/")
    mcp_servers.save_mcp_server(cfg, "project", str(tmp_path))
    config_path = mcp_servers.project_mcp_config_path(str(tmp_path))
    remove = partial(mcp_servers.remove_mcp_server, "demo", str(tmp_path))
    _assert_times_out(remove, config_path.with_suffix(".lock"))
    assert "demo" in mcp_servers.load_mcp_servers(str(tmp_path)), "timed out: untouched"
    # Released: the same removal goes through at once.
    assert remove() == [config_path]
    assert mcp_servers.load_mcp_servers(str(tmp_path)) == {}


def test_token_storage_put_and_clear_give_up_on_a_wedged_lock(
    tmp_path: Path,
    quick_timeout: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    store = mcp_servers.FileTokenStorage("sweep1026")
    _assert_times_out(partial(store.put, "client_info", {"client_id": "c"}), store._lock_path)
    assert not store.path.exists()
    store.put("client_info", {"client_id": "c"})
    assert store._read() == {"client_info": {"client_id": "c"}}
    _assert_times_out(store.clear, store._lock_path)
    assert store.path.exists(), "the timed-out clear must not have deleted the file"
    assert store.clear() is True
