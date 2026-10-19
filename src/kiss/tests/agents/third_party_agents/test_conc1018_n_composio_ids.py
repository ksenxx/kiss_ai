# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""``FakeComposioServer.next_id`` is called from the per-request handler
threads of a ``ThreadingHTTPServer``; an unguarded ``+= 1`` could hand two
concurrent creates the same id (finding conc1018 N1-F1)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import requests

from kiss.tests.agents.third_party_agents.composio_test_utils import (
    API,
    API_KEY,
    FakeComposioServer,
)
from kiss.tests.agents.third_party_agents.recording_http import serving

_CREATES = 64


def _create_auth_config(base_url: str) -> str:
    response = requests.post(
        f"{base_url}{API}/auth_configs",
        json={"toolkit": {"slug": "gmail"}},
        headers={"x-api-key": API_KEY},
        timeout=30,
    )
    assert response.status_code == 201
    return str(response.json()["auth_config"]["id"])


def test_concurrent_creates_get_distinct_ids() -> None:
    """Many auth-config POSTs fired at once all receive different ``ac_<n>`` ids."""
    with serving(FakeComposioServer()) as server, ThreadPoolExecutor(max_workers=16) as pool:
        ids = list(pool.map(_create_auth_config, [server.base_url] * _CREATES))
    assert len(set(ids)) == _CREATES
    assert sorted(ids, key=lambda s: int(s.split("_")[1])) == [
        f"ac_{n}" for n in range(1, _CREATES + 1)
    ]
    assert len(server.auth_configs) == _CREATES
