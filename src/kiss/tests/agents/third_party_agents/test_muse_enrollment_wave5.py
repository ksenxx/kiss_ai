# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the ``discord_server`` fixture is imported from test_discord_oauth)
"""A Muse enrollment lands its vault credential and its ``config.json`` metadata together.

Mattermost, BlueBubbles, LINE, Zalo, Twitch, Nextcloud Talk and Discord stored
the credential in the vault first and wrote the non-secret metadata (server
origin, webhook secret, OA id, client id, Discord auth mode) second, outside
any lock.  So:

* a metadata write that failed after the store (here: the config lock cannot
  be opened, or the directory is read-only) left the NEW credential next to the
  OLD metadata, and the next backend sent the token to a server its origin
  binding refuses, or used the wrong Discord auth mode;
* two enrollments running at once could interleave A-store, B-store, B-write,
  A-write, leaving A's server paired with B's credential.

``enroll_credential`` holds the config lock across both steps, writes the
metadata first and, if anything fails or a Stop lands after that, asks the
daemon for the vault's credential generation: unchanged puts the old file
back, changed (the store landed, its reply lost) keeps the new metadata.
Every test enrolls through the real connector code against the loopback
emulators of ``test_muse_auth_messaging`` / ``test_discord_oauth`` and a live
Muse daemon, then checks what a fresh backend actually sends.  The interleaving
is forced with thread tracers, not timing.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import threading
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from kiss.agents.third_party_agents._channel_agent_utils import (
    config_file_lock,
    write_private_file,
)
from kiss.agents.third_party_agents.discord.discord_sea import DiscordAgent
from kiss.agents.third_party_agents.discord.discord_sea import _config as discord_config
from kiss.agents.third_party_agents.mattermost.mattermost_sea import (
    MattermostAgent,
    MattermostChannelBackend,
)
from kiss.agents.third_party_agents.mattermost.mattermost_sea import _config as mm_config
from kiss.agents.third_party_agents.mattermost.mattermost_sea import (
    _muse_authenticate as mm_authenticate,
)
from kiss.agents.third_party_agents.muse_auth._common import muse_auth_dir
from kiss.agents.third_party_agents.muse_auth.client import store_credentials
from kiss.tests.agents.third_party_agents.muse_test_utils import auth_tools, loopback_policy
from kiss.tests.agents.third_party_agents.recording_http import serve_recording
from kiss.tests.agents.third_party_agents.test_discord_oauth import (  # noqa: F401
    _BOT_TOKEN,
    _USER_TOKEN,
    _approve,
    _DiscordServer,
    _start,
    discord_server,
)
from kiss.tests.agents.third_party_agents.test_muse_auth_messaging import (
    _REAL_MM_TOKEN,
    _MessagingApiHandler,
    _MessagingApiServer,
)
from kiss.tests.conftest import IS_WINDOWS

# Mattermost is origin-bound at enrollment; the Discord emulator needs a
# loopback ``extra_hosts`` entry (as in test_discord_oauth).
MUSE_POLICY = loopback_policy("discord")

_VAULT_REFUSED = "invalid credential token value"

# The generator behind the ``config_file_lock`` context manager: its frame
# starts when a thread asks for the lock.
_LOCK_CODE = cast(types.FunctionType, inspect.unwrap(config_file_lock)).__code__


@pytest.fixture()
def api_server() -> Iterator[_MessagingApiServer]:
    """The Mattermost server the first enrollment points at."""
    yield from serve_recording(_MessagingApiHandler, _MessagingApiServer)


@pytest.fixture()
def other_server() -> Iterator[_MessagingApiServer]:
    """The Mattermost server the second enrollment points at."""
    yield from serve_recording(_MessagingApiHandler, _MessagingApiServer)


def _block_config_lock(config_path: Path) -> Path:
    """Put a directory where *config_path*'s lock file goes, so opening the lock fails.

    Unlike a ``chmod``, this also fails for root.

    Returns:
        The directory, for the caller to remove.
    """
    lock = config_path.parent / (config_path.name + ".lock")
    lock.unlink(missing_ok=True)
    lock.mkdir()
    return lock


def _rotate_mattermost(tools: dict[str, Any], server: _MessagingApiServer) -> dict[str, Any]:
    """Enroll the working token for *server* through the agent's tool."""
    port = server.server_address[1]
    answer = tools["authenticate_mattermost"]("127.0.0.1", _REAL_MM_TOKEN, port=port, scheme="http")
    return dict(json.loads(answer))


def _assert_fresh_backend_reaches(server: _MessagingApiServer) -> None:
    """A new backend (as each cron tick builds) authenticates against *server*."""
    fresh = MattermostChannelBackend()
    assert fresh.connect() is True, fresh._connection_info
    assert server.header("Authorization") == f"Bearer {_REAL_MM_TOKEN}"


def test_unopenable_config_lock_leaves_the_enrollment_working(
    muse_env: Path, api_server: _MessagingApiServer, other_server: _MessagingApiServer
) -> None:
    """The review's reproduction: the failure comes before the vault changes.

    The old order stored the rotation's credential (bound to the other
    server), then failed to write the metadata: the next backend sent it to
    the enrolled server, whose origin the credential no longer matched.
    """
    tools = auth_tools(MattermostAgent())
    assert _rotate_mattermost(tools, api_server)["ok"] is True
    enrolled = mm_config.path.read_text()
    lock = _block_config_lock(mm_config.path)
    try:
        result = _rotate_mattermost(tools, other_server)
    finally:
        lock.rmdir()
    assert result["ok"] is False
    assert mm_config.path.read_text() == enrolled
    _assert_fresh_backend_reaches(api_server)


@pytest.mark.skipif(
    IS_WINDOWS or os.geteuid() == 0, reason="directory modes bind neither root nor Windows"
)
def test_unwritable_config_dir_leaves_the_enrollment_working(
    muse_env: Path, api_server: _MessagingApiServer, other_server: _MessagingApiServer
) -> None:
    """The metadata write itself fails (the lock was taken): the vault is still untouched."""
    tools = auth_tools(MattermostAgent())
    assert _rotate_mattermost(tools, api_server)["ok"] is True
    enrolled = mm_config.path.read_text()
    channel_dir = mm_config.path.parent
    mode = channel_dir.stat().st_mode
    channel_dir.chmod(0o500)
    try:
        result = _rotate_mattermost(tools, other_server)
    finally:
        channel_dir.chmod(mode)
    assert result["ok"] is False
    assert mm_config.path.read_text() == enrolled
    _assert_fresh_backend_reaches(api_server)


def test_first_enrollment_the_vault_refuses_leaves_no_config(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """With nothing enrolled before, a refused store removes the metadata it wrote."""
    tools = auth_tools(MattermostAgent())
    port = api_server.server_address[1]
    answer = tools["authenticate_mattermost"]("127.0.0.1", "mm  spaced", port=port, scheme="http")
    result = json.loads(answer)
    assert result["ok"] is False
    assert _VAULT_REFUSED in result["error"]
    assert not mm_config.path.exists()


def _swap_tracer(tracer: Any) -> Any:
    """Install *tracer* on this thread; return the one it replaces (coverage's, if measured)."""
    previous = sys.gettrace()
    sys.settrace(tracer)
    return previous


class _StopAsConfigWritten:
    """Tracer: raise a Stop as the first private-file write of *path* returns."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.fired = False

    def __call__(self, frame: Any, event: str, arg: Any) -> Any:
        """Global tracer: trace ``write_private_file`` frames writing *path*, until fired."""
        if self.fired or frame.f_code is not write_private_file.__code__:
            return None
        return self.local if frame.f_locals.get("path") == self.path else None

    def local(self, frame: Any, event: str, arg: Any) -> Any:
        """Local tracer: raise ``KeyboardInterrupt`` on the write's return."""
        if event == "return" and not self.fired:
            self.fired = True
            raise KeyboardInterrupt("injected stop")
        return self.local


class _StopAsStoreReturns:
    """Tracer: raise a Stop as ``store_credentials`` returns, its store already committed."""

    def __init__(self) -> None:
        self.fired = False

    def __call__(self, frame: Any, event: str, arg: Any) -> Any:
        """Global tracer: trace ``store_credentials`` frames only, until fired."""
        if self.fired or frame.f_code is not store_credentials.__code__:
            return None
        return self.local

    def local(self, frame: Any, event: str, arg: Any) -> Any:
        """Local tracer: raise ``KeyboardInterrupt`` on a successful return."""
        if event == "return" and arg is True and not self.fired:
            self.fired = True
            raise KeyboardInterrupt("injected stop")
        return self.local


def test_stop_right_after_the_metadata_write_puts_the_old_file_back(
    muse_env: Path, api_server: _MessagingApiServer, other_server: _MessagingApiServer
) -> None:
    """A Stop between writing the new metadata and storing its credential undoes the write."""
    tools = auth_tools(MattermostAgent())
    assert _rotate_mattermost(tools, api_server)["ok"] is True
    enrolled = mm_config.path.read_text()
    previous = _swap_tracer(_StopAsConfigWritten(mm_config.path))
    try:
        with pytest.raises(KeyboardInterrupt):
            _rotate_mattermost(tools, other_server)
    finally:
        sys.settrace(previous)
    assert mm_config.path.read_text() == enrolled
    _assert_fresh_backend_reaches(api_server)


def test_stop_after_the_store_landed_keeps_the_new_metadata(
    muse_env: Path, api_server: _MessagingApiServer, other_server: _MessagingApiServer
) -> None:
    """A Stop that only cost the store's reply leaves the landed enrollment whole.

    The vault already holds the new credential, so putting the old
    metadata back (or never writing the new one, the store-first order)
    would pair it with the old server.
    """
    tools = auth_tools(MattermostAgent())
    assert _rotate_mattermost(tools, api_server)["ok"] is True
    previous = _swap_tracer(_StopAsStoreReturns())
    try:
        with pytest.raises(KeyboardInterrupt):
            _rotate_mattermost(tools, other_server)
    finally:
        sys.settrace(previous)
    assert json.loads(mm_config.path.read_text())["port"] == str(other_server.server_address[1])
    _assert_fresh_backend_reaches(other_server)


class _HideMuseDirOnStoreFailure:
    """Tracer: as a failing ``store_credentials`` returns, put a file where the Muse dir is.

    The daemon then cannot be reached or respawned, so the enrollment
    cannot learn the vault's generation.
    """

    def __init__(self) -> None:
        self.muse_dir = muse_auth_dir()
        self.backup = self.muse_dir.with_name("muse_auth.hidden")
        self.fired = False

    def __call__(self, frame: Any, event: str, arg: Any) -> Any:
        """Global tracer: trace ``store_credentials`` frames only, until fired."""
        if self.fired or frame.f_code is not store_credentials.__code__:
            return None
        return self.local

    def local(self, frame: Any, event: str, arg: Any) -> Any:
        """Local tracer: hide the directory once the store has raised."""
        if event == "exception" and not self.fired:
            self.fired = True
            self.muse_dir.rename(self.backup)
            self.muse_dir.write_text("")
        return self.local

    def restore(self) -> None:
        """Put the Muse directory back."""
        if self.fired:
            self.muse_dir.unlink()
            self.backup.rename(self.muse_dir)


def test_vault_that_cannot_be_asked_counts_as_unchanged(
    muse_env: Path, api_server: _MessagingApiServer, other_server: _MessagingApiServer
) -> None:
    """With the daemon unreachable after a failed store, the old metadata is put back."""
    tools = auth_tools(MattermostAgent())
    assert _rotate_mattermost(tools, api_server)["ok"] is True
    enrolled = mm_config.path.read_text()
    hide = _HideMuseDirOnStoreFailure()
    previous = _swap_tracer(hide)
    try:
        port = other_server.server_address[1]
        answer = tools["authenticate_mattermost"](
            "127.0.0.1", "mm  spaced", port=port, scheme="http"
        )
    finally:
        sys.settrace(previous)
        hide.restore()
    assert hide.fired
    assert json.loads(answer)["ok"] is False
    assert mm_config.path.read_text() == enrolled
    _assert_fresh_backend_reaches(api_server)


class _HoldAfterStore:
    """Thread tracer: hold the thread once, as its ``store_credentials`` call returns."""

    def __init__(self) -> None:
        self.held = threading.Event()
        self.release = threading.Event()
        self.fired = False

    def __call__(self, frame: Any, event: str, arg: Any) -> Any:
        """Global tracer: trace ``store_credentials`` frames only, until fired."""
        if self.fired or frame.f_code is not store_credentials.__code__:
            return None
        return self.local

    def local(self, frame: Any, event: str, arg: Any) -> Any:
        """Local tracer: hold the thread on the store's return."""
        if event == "return" and not self.fired:
            self.fired = True
            self.held.set()
            self.release.wait(30)
        return self.local


class _SignalLockBeforeStore:
    """Thread tracer: set *event* if the thread takes the config lock before its store."""

    def __init__(self, event: threading.Event) -> None:
        self.event = event
        self.stored = False

    def __call__(self, frame: Any, event: str, arg: Any) -> Any:
        """Watch call events only."""
        if event == "call":
            if frame.f_code is store_credentials.__code__:
                self.stored = True
            elif frame.f_code is _LOCK_CODE and not self.stored:
                self.event.set()
        return None


def _enroll_traced(
    tracer: Any,
    server: _MessagingApiServer,
    results: dict[str, dict[str, Any]],
    key: str,
    done: threading.Event,
) -> None:
    """Thread body: enroll the working token for *server* under *tracer*."""
    previous = sys.gettrace()  # coverage's, when measured
    sys.settrace(tracer)
    try:
        port = server.server_address[1]
        backend = MattermostChannelBackend()
        answer = mm_authenticate(backend, "127.0.0.1", _REAL_MM_TOKEN, port, "http")
        results[key] = json.loads(answer)
    finally:
        sys.settrace(previous)
        done.set()


def test_concurrent_enrollments_never_mix_server_and_credential(
    muse_env: Path, api_server: _MessagingApiServer, other_server: _MessagingApiServer
) -> None:
    """The review's interleaving: A holds right after its vault store while B enrolls.

    The old code let B finish (store, then metadata) while A was held, then
    A wrote A's server next to B's credential.  Now A holds the config lock
    across its whole enrollment, so B, once it reaches that lock, waits for
    A and lands both its halves last.
    """
    results: dict[str, dict[str, Any]] = {}
    hold_a = _HoldAfterStore()
    a_done = threading.Event()
    first = threading.Thread(target=_enroll_traced, args=(hold_a, api_server, results, "a", a_done))
    first.start()
    try:
        assert hold_a.held.wait(30)
        # Set when B finishes, or when B reaches the config lock before its
        # own store: from then on nothing B does can interleave with A.
        b_progress = threading.Event()
        second = threading.Thread(
            target=_enroll_traced,
            args=(_SignalLockBeforeStore(b_progress), other_server, results, "b", b_progress),
        )
        second.start()
        assert b_progress.wait(30)
    finally:
        hold_a.release.set()
    first.join(30)
    second.join(30)
    assert a_done.is_set() and not second.is_alive()
    assert results["a"]["ok"] is True and results["b"]["ok"] is True, results
    assert json.loads(mm_config.path.read_text())["port"] == str(other_server.server_address[1])
    _assert_fresh_backend_reaches(other_server)


def test_discord_bot_token_whose_metadata_cannot_land_keeps_the_sign_in(
    discord_server: _DiscordServer, muse_env: Path
) -> None:
    """A bot enrollment that cannot write ``config.json`` keeps the user sign-in whole.

    The old order stored the bot credential, then failed to clear the
    ``auth_mode: user`` marker: the next backend ran in user mode with the
    bot's credential.
    """
    tools = auth_tools(DiscordAgent())
    _approve(_start(tools))
    assert json.loads(tools["finish_discord_auth"]())["ok"] is True
    lock = _block_config_lock(discord_config.path)
    try:
        result = json.loads(tools["authenticate_discord"](bot_token=_BOT_TOKEN))
    finally:
        lock.rmdir()
    assert result["ok"] is False
    assert json.loads(discord_config.path.read_text()) == {"auth_mode": "user"}
    fresh = DiscordAgent()._backend
    assert fresh._user_auth
    assert json.loads(fresh.list_guilds())["ok"] is True
    assert discord_server.auth_headers[-1] == f"Bearer {_USER_TOKEN}"


def test_discord_sign_in_whose_marker_cannot_land_keeps_the_bot(
    discord_server: _DiscordServer, muse_env: Path
) -> None:
    """A user sign-in that cannot write its ``auth_mode`` marker keeps the bot token whole.

    The old order stored the user grant, then failed to write the marker:
    the next backend ran in bot mode with the user's credential.
    """
    tools = auth_tools(DiscordAgent())
    assert json.loads(tools["authenticate_discord"](bot_token=_BOT_TOKEN))["ok"] is True
    assert not discord_config.path.exists()
    started = _start(tools)
    lock = _block_config_lock(discord_config.path)
    try:
        _approve(started)
        result = json.loads(tools["finish_discord_auth"]())
    finally:
        lock.rmdir()
    assert result["ok"] is False
    assert not discord_config.path.exists()
    fresh = DiscordAgent()._backend
    assert not fresh._user_auth
    assert json.loads(fresh.list_guilds())["ok"] is True
    assert discord_server.auth_headers[-1] == f"Bot {_BOT_TOKEN}"
