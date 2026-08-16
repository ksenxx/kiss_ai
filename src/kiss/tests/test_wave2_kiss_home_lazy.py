# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `fresh_home` module fixture is imported from
#   kiss.tests.server.test_wave2_kiss_home_lazy and is intentionally
#   shadowed by the test parameter of the same name)


"""Tests that stay behind from ``kiss.tests.test_wave2_kiss_home_lazy``
(now ``kiss.tests.server.test_wave2_kiss_home_lazy``): they assert on assets outside
kiss.core/kiss.agents.sorcar/kiss.server (e.g. the real
chat.html/main.js media content), so they keep their original
location while the server-only majority of the file moved to
tests/server.
"""


from pathlib import Path

import kiss.agents.third_party_agents._channel_agent_utils as channel_utils
from kiss.tests.server.test_wave2_kiss_home_lazy import fresh_home  # noqa: F401


def test_channel_agent_utils_resolves_lazily(fresh_home: Path) -> None:
    """ChannelConfig paths under ~/.kiss are rebased onto $KISS_HOME lazily."""
    cfg = channel_utils.ChannelConfig(
        Path.home() / ".kiss" / "third_party_agents" / "lazytest", (),
    )
    assert cfg.path == (
        fresh_home / "third_party_agents" / "lazytest" / "config.json"
    )
