# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `fresh_home` module fixture is imported from
#   kiss.tests.server.test_wave2_kiss_home_lazy and is intentionally
#   shadowed by the test parameter of the same name)


"""KISS_HOME laziness tests for the channel-agent utilities.

Split from ``kiss.tests.server.test_wave2_kiss_home_lazy``: these tests
depend on ``kiss.agents.third_party_agents._channel_agent_utils``, so
they live in tests/agents/third_party_agents while the server-only
majority of the original file lives in tests/server.
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
