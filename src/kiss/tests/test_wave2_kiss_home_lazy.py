# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""$KISS_HOME must be resolved lazily by every consumer.

``$KISS_HOME``-or-``~/.kiss`` used to be resolved independently five
times across the codebase, and two of the copies (``vscode_config``,
``web_server``) froze the answer at import time — so a ``KISS_HOME``
exported *after* those modules were imported (exactly what the test
conftest does) was silently ignored.  All five sites now route through
the canonical lazy resolver :func:`kiss.core.config.kiss_home`; these
tests set ``KISS_HOME`` **after** importing every consumer and assert
each one observes the new location.
"""

import json
from pathlib import Path

import pytest

import kiss.agents.sorcar.persistence as persistence
import kiss.agents.third_party_agents._channel_agent_utils as channel_utils
import kiss.agents.vscode.user_assets as user_assets
import kiss.agents.vscode.vscode_config as vscode_config
import kiss.agents.vscode.web_server as web_server
from kiss.core.config import kiss_home


@pytest.fixture()
def fresh_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point KISS_HOME at a new temp dir *after* all modules are imported.

    Also clears any ``CONFIG_DIR``/``CONFIG_PATH`` pins that an earlier
    test in the same process may have left in ``vscode_config``'s
    module dict, and any web_server override slots.
    """
    for name in ("CONFIG_DIR", "CONFIG_PATH"):
        if name in vars(vscode_config):
            monkeypatch.delattr(vscode_config, name, raising=False)
    monkeypatch.setattr(web_server, "_KISS_HOME", None)
    monkeypatch.setattr(web_server, "_TLS_DIR", None)
    monkeypatch.setattr(web_server, "_CLOUDFLARED_PIDFILE", None)
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    return tmp_path


def test_canonical_resolver_is_lazy(fresh_home: Path) -> None:
    """kiss_home() re-reads the environment on every call."""
    assert kiss_home() == fresh_home


def test_user_assets_resolves_lazily(fresh_home: Path) -> None:
    assert user_assets.kiss_home_dir() == fresh_home


def test_persistence_resolves_lazily(fresh_home: Path) -> None:
    assert persistence._default_kiss_dir() == fresh_home


def test_channel_agent_utils_resolves_lazily(fresh_home: Path) -> None:
    assert channel_utils._kiss_home() == fresh_home


def test_vscode_config_resolves_lazily(fresh_home: Path) -> None:
    """CONFIG_DIR/CONFIG_PATH attributes and accessors track KISS_HOME."""
    assert vscode_config.CONFIG_DIR == fresh_home
    assert vscode_config.CONFIG_PATH == fresh_home / "config.json"
    assert vscode_config._config_dir() == fresh_home
    assert vscode_config._config_path() == fresh_home / "config.json"


def test_vscode_config_save_load_roundtrip(fresh_home: Path) -> None:
    """save_config writes into the post-import KISS_HOME and load reads it."""
    vscode_config.save_config({"max_budget": 42})
    on_disk = json.loads((fresh_home / "config.json").read_text())
    assert on_disk["max_budget"] == 42
    assert vscode_config.load_config()["max_budget"] == 42


def test_vscode_config_test_override_pin_wins(fresh_home: Path) -> None:
    """An explicit CONFIG_DIR/CONFIG_PATH assignment shadows lazy lookup."""
    pin = fresh_home / "pinned"
    vscode_config.CONFIG_DIR = pin
    try:
        assert vscode_config._config_dir() == pin
        assert vscode_config._config_path() == pin / "config.json"
    finally:
        del vscode_config.CONFIG_DIR
    assert vscode_config._config_dir() == fresh_home


def test_web_server_paths_resolve_lazily(fresh_home: Path) -> None:
    assert web_server._kiss_home_dir() == fresh_home
    assert web_server._tls_dir() == fresh_home / "tls"
    assert web_server._url_file_path() == fresh_home / "remote-url.json"
    assert web_server._URL_FILE == fresh_home / "remote-url.json"
    assert web_server._default_uds_path() == fresh_home / "sorcar.sock"
    assert web_server._cloudflared_pidfile() == fresh_home / "cloudflared.pid"


def test_web_server_url_file_override_reaches_production_consumers(
    fresh_home: Path,
) -> None:
    """A temporary ``_URL_FILE`` pin must redirect the real server.

    The lazy PEP 562 attribute is often pinned by tests.  Merely making
    attribute reads return the pin is insufficient: the production
    constructor resolves its default through ``_url_file_path``.
    """
    pinned = fresh_home / "pinned" / "remote.json"
    web_server._URL_FILE = pinned
    try:
        assert web_server._url_file_path() == pinned
        assert web_server.RemoteAccessServer()._url_file == pinned
    finally:
        del web_server._URL_FILE
    assert web_server._url_file_path() == fresh_home / "remote-url.json"
