# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end checks for the shared conftest fixtures added by the W8 audit.

``refusing_port`` replaces the bind/close/reuse-the-number pattern that
several "unreachable server" tests used; the tests here prove the two
properties those callers rely on: connects are refused for the whole test,
and no other socket can take the port meanwhile. ``isolated_kiss_home``
replaces five identical per-module fixtures; the test proves channel
configs really land under the returned directory.
"""

from __future__ import annotations

import errno
import socket
import time
from pathlib import Path

import pytest

from kiss.agents.third_party_agents._channel_agent_utils import ChannelConfig
from kiss.core.config import kiss_home
from kiss.tests.conftest import IS_WINDOWS


def test_refusing_port_refuses_connections_and_stays_reserved(refusing_port: int) -> None:
    """Connecting is refused at once and the port cannot be bound by anyone else.

    The refusal must be an RST, not a timeout: callers use the port as an
    "unreachable server" and assert on fast failure paths.  POSIX kernels
    answer the SYN with RST at once; the Windows TCP stack retransmits the
    SYN twice, about a second apart, before it surfaces WSAECONNREFUSED,
    so the refusal takes 2-2.5 s there whatever state the port is in.
    """
    started = time.monotonic()
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("127.0.0.1", refusing_port), timeout=5)
    assert time.monotonic() - started < (4.0 if IS_WINDOWS else 2.0)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as other, pytest.raises(OSError) as info:
        other.bind(("127.0.0.1", refusing_port))
    assert info.value.errno == errno.EADDRINUSE


def test_isolated_kiss_home_redirects_channel_configs(isolated_kiss_home: Path) -> None:
    """kiss_home() and a ~/.kiss-based ChannelConfig both follow the fixture."""
    assert kiss_home() == isolated_kiss_home
    config = ChannelConfig(Path.home() / ".kiss" / "third_party_agents" / "w8audit", ())
    assert config.path == isolated_kiss_home / "third_party_agents" / "w8audit" / "config.json"
    config.save({"k": "v"})
    assert config.load() == {"k": "v"}
    assert (isolated_kiss_home / "third_party_agents" / "w8audit" / "config.json").is_file()
