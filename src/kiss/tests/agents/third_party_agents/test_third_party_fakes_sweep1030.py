# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The loopback service emulators the channel-agent tests share.

Each HTTP emulator answers requests on one handler thread per connection,
so two requests that touch the same state must not see each other
half-way.  ``Pause`` (a ``sys.settrace`` hook, nothing replaced) and the
Composio emulator's own status-poll hold put the second request inside
the first one's window for sure.

A second interleaving of the Composio fix, two concurrent DELETEs of one
connected account, has no hook to hold the first DELETE between its
lookup and its delete; it is covered by the same lock and not
reproduced here.

The IRC listener helper is served by the test thread itself; its
failure report for a client that hangs up early is pinned last.
"""

from __future__ import annotations

import socket
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from typing import Any

import pytest
import requests

from kiss.tests.agents.third_party_agents.composio_test_utils import (
    API,
    API_KEY,
    FakeComposioServer,
)
from kiss.tests.agents.third_party_agents.irc_test_utils import IRCServer, read_lines
from kiss.tests.agents.third_party_agents.recording_http import recording_server
from kiss.tests.agents.third_party_agents.test_muse_connect_flows import (
    _AuthHandler,
    _AuthServer,
)
from kiss.tests.server.pause import Pause

_HELD_KEY = "held-composio-key"


def test_composio_status_poll_held_across_a_delete_answers_404(
    composio: FakeComposioServer,
) -> None:
    """A status poll that waits while its account is deleted answers 404.

    The poll used to find the account, wait, then index the accounts
    table again: the ``KeyError`` killed the handler and the client saw
    the connection drop with no answer.
    """
    composio.api_keys.add(_HELD_KEY)
    composio.accounts["ca_gone"] = "ACTIVE"
    held, release = composio.hold_status_polls(_HELD_KEY)
    url = f"{composio.base_url}{API}/connected_accounts/ca_gone"
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            poll = pool.submit(requests.get, url, headers={"x-api-key": _HELD_KEY}, timeout=30)
            assert held.wait(10), "the status poll never reached its hold"
            deleted = requests.delete(url, headers={"x-api-key": API_KEY}, timeout=10)
            assert deleted.status_code == 200
        finally:
            release.set()
        answer = poll.result(timeout=30)
    assert answer.status_code == 404
    assert answer.json()["error"]["message"] == "connected account not found"
    again = requests.delete(url, headers={"x-api-key": API_KEY}, timeout=10)
    assert again.status_code == 404
    assert composio.deleted == ["ca_gone"]
    assert "ca_gone" not in composio.accounts


def _any_call(frame_locals: dict[str, Any]) -> bool:
    """Match every call of the paused code object."""
    return True


def test_matrix_registration_reads_its_own_request_body() -> None:
    """A request recorded while a registration is in flight does not replace its body.

    The registration handler used to parse ``server.requests[-1]``, the
    most recently recorded request of ANY handler thread, so a revoke
    landing in between turned a valid registration into a 400.
    """
    metadata = {"client_name": "KISS", "client_uri": "https://kiss.example"}
    with (
        recording_server(_AuthHandler, _AuthServer) as server,
        ThreadPoolExecutor(max_workers=1) as pool,
        Pause(_AuthHandler._register_matrix_client.__code__, _any_call) as pause,
    ):
        registration = pool.submit(
            requests.post, f"{server.base_url}/mas/oauth2/registration", json=metadata, timeout=30
        )
        assert pause.paused.wait(10), "the registration never reached its handler"
        revoked = requests.post(
            f"{server.base_url}/mas/oauth2/revoke", data={"token": "t"}, timeout=10
        )
        assert revoked.status_code == 200
        pause.release()
        answer = registration.result(timeout=30)
    assert answer.status_code == 201, answer.text
    assert answer.json()["client_id"] == "mas-client-1"
    assert server.matrix_registrations == [metadata]


def test_irc_read_lines_names_a_client_that_hung_up() -> None:
    """``read_lines`` fails with the lines read so far when the client closes first."""
    with closing(IRCServer()) as server:
        with socket.create_connection(("127.0.0.1", server.port), timeout=5) as client:
            client.sendall(b"NICK bot\r\n")
        with (
            server.accept() as conn,
            pytest.raises(AssertionError, match=r"closed before 'USER '; got \['NICK bot'\]"),
        ):
            read_lines(conn, "USER ")
