# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Behavioural contract shared by every channel agent.

For each of the agents in :data:`ALL_CHANNEL_AGENTS`:

1. A fresh agent in an empty KISS home is unauthenticated and its tool
   list is exactly its auth tools (backend tools stay locked); the
   module's SEA ``tools()`` appends exactly those tools to the run's
   toolset.
2. The cron gateway (``gateway_command``) schedules a poll tick only for
   channels with a poll backend, and refuses the others.
3. The ``kiss-<channel>`` console script runs the module's ``main``,
   which prints the usage line and exits 1 when given no arguments.
"""

from __future__ import annotations

import importlib
import sys
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from kiss.agents.sorcar.cron_agent import _channel_cli_name, gateway_command
from kiss.agents.sorcar.sea_commands import load_sea
from kiss.tests.agents.third_party_agents.channel_agent_registry import (
    AGENT_IDS,
    ALL_CHANNEL_AGENTS,
    EXPECTED_AUTH_TOOLS,
    get_agent_class,
)

GATEWAY_CHANNELS = {
    "bluebubbles",
    "dingtalk",
    "discord",
    "email",
    "feishu",
    "googlechat",
    "irc",
    "line",
    "matrix",
    "mattermost",
    "msteams",
    "nextcloud",
    "ntfy",
    "phone",
    "qq",
    "signal",
    "simplex",
    "slack",
    "sms",
    "synology",
    "telegram",
    "webhook",
    "weixin",
    "whatsapp",
    "zalo",
}
"""Channels whose SEA module has a poll backend, so a gateway tick can be scheduled."""

HIDDEN_CHANNELS = {"a2a", "oai"}
"""SEA folders that are not listed as channels (``available_channels()`` skips them)."""


def _tool_names(tools: list) -> list[str]:
    return [t.__name__ for t in tools]


@pytest.mark.parametrize("class_name,module_path", ALL_CHANNEL_AGENTS, ids=AGENT_IDS)
def test_fresh_agent_exposes_only_auth_tools(
    class_name: str, module_path: str, isolated_kiss_home: object
) -> None:
    """An agent in an empty KISS home is unauthenticated and offers only its auth tools."""
    agent = get_agent_class(class_name, module_path)()
    assert agent._is_authenticated() is False
    auth_tools = agent._get_auth_tools()
    assert auth_tools and all(callable(t) for t in auth_tools)
    assert _tool_names(agent._get_tools()) == _tool_names(auth_tools)
    assert _tool_names(auth_tools) == EXPECTED_AUTH_TOOLS[class_name]


def _marker_tool() -> str:
    """A tool the run already has before the SEA adds the channel's."""
    return "marker"


@pytest.mark.parametrize("class_name,module_path", ALL_CHANNEL_AGENTS, ids=AGENT_IDS)
def test_sea_tools_append_auth_tools(
    class_name: str, module_path: str, isolated_kiss_home: object
) -> None:
    """The module's SEA ``tools()`` keeps the run's tools and appends exactly the auth tools.

    This is the boundary the daemon uses (``sea_path`` dispatch), so it
    also covers the SEAs ``available_channels()`` hides (A2A, OAI).
    """
    module = importlib.import_module(module_path)
    assert module.__file__
    sea = load_sea(Path(module.__file__))
    assert _tool_names(sea.tools([])) == EXPECTED_AUTH_TOOLS[class_name]
    tools = sea.tools([_marker_tool])
    assert tools[0] is _marker_tool
    assert _tool_names(tools[1:]) == EXPECTED_AUTH_TOOLS[class_name]


@pytest.mark.parametrize("class_name,module_path", ALL_CHANNEL_AGENTS, ids=AGENT_IDS)
def test_gateway_command_follows_poll_support(class_name: str, module_path: str) -> None:
    """The cron gateway builds a tick command only for channels with a poll backend."""
    channel = module_path.rsplit(".", 2)[1]
    command = gateway_command(channel, "123", pairing=False)
    if channel in GATEWAY_CHANNELS:
        assert command == f"{_channel_cli_name(channel)} --channel=123 --quiet"
    elif channel in HIDDEN_CHANNELS:
        assert command == f"error: unknown channel {channel!r}"
    else:
        assert command == f"error: channel {channel!r} has no gateway (poll) mode"


@pytest.mark.parametrize("class_name,module_path", ALL_CHANNEL_AGENTS, ids=AGENT_IDS)
def test_console_script_runs_module_main(
    class_name: str,
    module_path: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every agent module is wired to a ``kiss-*`` console script whose main() prints usage.

    ``main()`` is run in-process with no arguments: it prints the
    ``Usage: kiss-<channel>`` line and exits 1.
    """
    scripts = {e.value: e for e in entry_points(group="console_scripts")}
    entry = scripts[f"{module_path}:main"]
    assert entry.name.startswith("kiss-")
    main = entry.load()
    assert main is importlib.import_module(module_path).main
    monkeypatch.setattr(sys, "argv", [entry.name])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1
    assert f"Usage: {entry.name} " in capsys.readouterr().out
