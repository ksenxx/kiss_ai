# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests locking the channel-agent / chat-session contract.

Verifies the contract enforced by the
"third-party agents do not inherit SorcarAgent" cleanup:

1. No channel agent class is a subclass of :class:`SorcarAgent` (and so
   not of its ``ChatSorcarAgent`` subclass either) — third-party agents
   are plain :class:`BaseChannelAgent` carriers whose tasks always
   execute on the kiss-web daemon through :func:`kiss.server.sorcar.run`.
2. ``channel_main()``'s parser rejects every chat-session CLI flag
   (``-n/--new``, ``-c/--chat-id``, ``-l/--list-chat-id``) — they no
   longer exist anywhere in the project's CLI surface.
"""

from __future__ import annotations

import sys

import pytest

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    channel_main,
)
from kiss.agents.third_party_agents.slack.slack_sea import SlackAgent
from kiss.tests.agents.third_party_agents.channel_agent_registry import (
    AGENT_IDS,
    ALL_CHANNEL_AGENTS,
    get_agent_class,
)


@pytest.mark.parametrize("class_name,module_path", ALL_CHANNEL_AGENTS, ids=AGENT_IDS)
def test_channel_agent_does_not_subclass_sorcar_agent(
    class_name: str,
    module_path: str,
) -> None:
    """No channel agent class inherits from ``SorcarAgent``.

    Channel agents are plain :class:`BaseChannelAgent` carriers; every
    task they run is submitted to the kiss-web daemon through
    :func:`kiss.server.sorcar.run`, so the local instance needs none of
    the ``SorcarAgent`` execution machinery.
    """
    cls = get_agent_class(class_name, module_path)
    assert not issubclass(cls, SorcarAgent), f"{class_name} still inherits SorcarAgent"
    assert issubclass(cls, BaseChannelAgent)


@pytest.mark.parametrize(
    "flag",
    [
        "-n",
        "--new",
        "-c",
        "--chat-id",
        "-l",
        "--list-chat-id",
    ],
)
def test_channel_main_rejects_chat_session_flag(
    flag: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``channel_main`` no longer exposes any chat-session CLI flag.

    Argparse exits with status 2 on an unrecognized flag (after
    printing a "unrecognized arguments" diagnostic to stderr).  Both
    are asserted so the contract is locked end-to-end.  The parser runs
    before any agent is built, so the real ``SlackAgent`` class is passed.
    """
    monkeypatch.setattr(sys, "argv", ["test-cli", flag, "-t", "noop"])
    with pytest.raises(SystemExit) as exc_info:
        channel_main(SlackAgent, "kiss-slack")
    assert exc_info.value.code == 2
    assert "unrecognized" in capsys.readouterr().err
