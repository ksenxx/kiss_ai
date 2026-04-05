"""Integration tests for BaseChannelAgent and channel_main().

Verifies that:
1. All 23 channel agents inherit from BaseChannelAgent
2. BaseChannelAgent._get_tools() properly delegates to _get_auth_tools()
   and conditionally includes backend tools based on _is_authenticated()
3. channel_main() handles CLI parsing for both daemon and non-daemon modes
"""

from __future__ import annotations

import importlib
import sys

import pytest

from kiss.channels._channel_agent_utils import BaseChannelAgent, channel_main

# All agent (class_name, module_path) pairs
ALL_AGENTS = [
    ("BlueBubblesAgent", "kiss.channels.bluebubbles_agent"),
    ("DiscordAgent", "kiss.channels.discord_agent"),
    ("FeishuAgent", "kiss.channels.feishu_agent"),
    ("GmailAgent", "kiss.channels.gmail_agent"),
    ("GoogleChatAgent", "kiss.channels.googlechat_agent"),
    ("IMessageAgent", "kiss.channels.imessage_agent"),
    ("IRCAgent", "kiss.channels.irc_agent"),
    ("LineAgent", "kiss.channels.line_agent"),
    ("MatrixAgent", "kiss.channels.matrix_agent"),
    ("MattermostAgent", "kiss.channels.mattermost_agent"),
    ("MSTeamsAgent", "kiss.channels.msteams_agent"),
    ("NextcloudTalkAgent", "kiss.channels.nextcloud_talk_agent"),
    ("NostrAgent", "kiss.channels.nostr_agent"),
    ("PhoneControlAgent", "kiss.channels.phone_control_agent"),
    ("SignalAgent", "kiss.channels.signal_agent"),
    ("SlackAgent", "kiss.channels.slack_agent"),
    ("SMSAgent", "kiss.channels.sms_agent"),
    ("SynologyChatAgent", "kiss.channels.synology_chat_agent"),
    ("TelegramAgent", "kiss.channels.telegram_agent"),
    ("TlonAgent", "kiss.channels.tlon_agent"),
    ("TwitchAgent", "kiss.channels.twitch_agent"),
    ("WhatsAppAgent", "kiss.channels.whatsapp_agent"),
    ("ZaloAgent", "kiss.channels.zalo_agent"),
]


def _get_agent_class(module_path: str, class_name: str) -> type:
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


@pytest.mark.parametrize(
    "class_name,module_path", ALL_AGENTS, ids=[a[0] for a in ALL_AGENTS]
)
def test_all_agents_inherit_base_channel_agent(
    class_name: str, module_path: str
) -> None:
    """Every channel agent class inherits from BaseChannelAgent."""
    cls = _get_agent_class(module_path, class_name)
    assert issubclass(cls, BaseChannelAgent)


@pytest.mark.parametrize(
    "class_name,module_path", ALL_AGENTS, ids=[a[0] for a in ALL_AGENTS]
)
def test_all_agents_have_is_authenticated(
    class_name: str, module_path: str
) -> None:
    """Every channel agent overrides _is_authenticated()."""
    cls = _get_agent_class(module_path, class_name)
    # _is_authenticated should be defined on the agent class, not just inherited
    assert "_is_authenticated" in cls.__dict__


@pytest.mark.parametrize(
    "class_name,module_path", ALL_AGENTS, ids=[a[0] for a in ALL_AGENTS]
)
def test_all_agents_have_get_auth_tools(
    class_name: str, module_path: str
) -> None:
    """Every channel agent overrides _get_auth_tools()."""
    cls = _get_agent_class(module_path, class_name)
    assert "_get_auth_tools" in cls.__dict__


@pytest.mark.parametrize(
    "class_name,module_path", ALL_AGENTS, ids=[a[0] for a in ALL_AGENTS]
)
def test_no_inline_get_tools(class_name: str, module_path: str) -> None:
    """No agent defines its own _get_tools() — inherited from BaseChannelAgent."""
    cls = _get_agent_class(module_path, class_name)
    assert "_get_tools" not in cls.__dict__


@pytest.mark.parametrize(
    "class_name,module_path", ALL_AGENTS, ids=[a[0] for a in ALL_AGENTS]
)
def test_agent_instantiation(class_name: str, module_path: str) -> None:
    """Every agent can be instantiated and has a _backend attribute."""
    cls = _get_agent_class(module_path, class_name)
    agent = cls()
    assert hasattr(agent, "_backend")


@pytest.mark.parametrize(
    "class_name,module_path", ALL_AGENTS, ids=[a[0] for a in ALL_AGENTS]
)
def test_is_authenticated_returns_bool(
    class_name: str, module_path: str
) -> None:
    """_is_authenticated() returns a bool for freshly created agents."""
    cls = _get_agent_class(module_path, class_name)
    agent = cls()
    result = agent._is_authenticated()
    assert isinstance(result, bool)


@pytest.mark.parametrize(
    "class_name,module_path", ALL_AGENTS, ids=[a[0] for a in ALL_AGENTS]
)
def test_get_auth_tools_returns_list(
    class_name: str, module_path: str
) -> None:
    """_get_auth_tools() returns a non-empty list of callables."""
    cls = _get_agent_class(module_path, class_name)
    agent = cls()
    auth_tools = agent._get_auth_tools()
    assert isinstance(auth_tools, list)
    assert len(auth_tools) >= 2  # at least check + authenticate
    assert all(callable(t) for t in auth_tools)


@pytest.mark.parametrize(
    "class_name,module_path", ALL_AGENTS, ids=[a[0] for a in ALL_AGENTS]
)
def test_get_tools_includes_auth_tools(
    class_name: str, module_path: str
) -> None:
    """_get_tools() includes auth tools from _get_auth_tools()."""
    cls = _get_agent_class(module_path, class_name)
    agent = cls()
    all_tools = agent._get_tools()
    auth_tools = agent._get_auth_tools()
    auth_names = {t.__name__ for t in auth_tools}
    all_names = {t.__name__ for t in all_tools}
    assert auth_names.issubset(all_names)


@pytest.mark.parametrize(
    "class_name,module_path", ALL_AGENTS, ids=[a[0] for a in ALL_AGENTS]
)
def test_unauthenticated_agent_no_backend_tools(
    class_name: str, module_path: str
) -> None:
    """A freshly created (unauthenticated) agent doesn't include backend tools."""
    cls = _get_agent_class(module_path, class_name)
    agent = cls()
    if agent._is_authenticated():
        pytest.skip("Agent auto-authenticates from stored config")
    all_tools = agent._get_tools()
    all_names = {t.__name__ for t in all_tools}
    backend_methods = agent._backend.get_tool_methods()
    backend_names = {m.__name__ for m in backend_methods}
    # No backend-specific tool should be in the unauthenticated tool list
    assert all_names.isdisjoint(backend_names)


def test_channel_main_no_args_exits(capsys: pytest.CaptureFixture[str]) -> None:
    """channel_main() with no CLI args prints usage and exits."""
    from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent

    class FakeAgent(BaseChannelAgent, StatefulSorcarAgent):
        pass

    original_argv = sys.argv[:]
    try:
        sys.argv = ["test-cli"]
        with pytest.raises(SystemExit) as exc_info:
            channel_main(FakeAgent, "kiss-test")
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "kiss-test" in captured.out
    finally:
        sys.argv = original_argv


def test_channel_main_usage_includes_daemon_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """channel_main() usage includes [--daemon] when make_daemon_backend is set."""
    from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent

    class FakeAgent(BaseChannelAgent, StatefulSorcarAgent):
        pass

    original_argv = sys.argv[:]
    try:
        sys.argv = ["test-cli"]
        with pytest.raises(SystemExit):
            channel_main(
                FakeAgent,
                "kiss-test",
                channel_name="Test",
                make_daemon_backend=lambda: None,
            )
        captured = capsys.readouterr()
        assert "--daemon" in captured.out
    finally:
        sys.argv = original_argv


def test_channel_main_usage_no_daemon_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """channel_main() usage omits [--daemon] when make_daemon_backend is None."""
    from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent

    class FakeAgent(BaseChannelAgent, StatefulSorcarAgent):
        pass

    original_argv = sys.argv[:]
    try:
        sys.argv = ["test-cli"]
        with pytest.raises(SystemExit):
            channel_main(FakeAgent, "kiss-test")
        captured = capsys.readouterr()
        assert "--daemon" not in captured.out
    finally:
        sys.argv = original_argv


# Modules that have _make_daemon_backend
_DAEMON_MODULES = [
    "kiss.channels.bluebubbles_agent",
    "kiss.channels.discord_agent",
    "kiss.channels.feishu_agent",
    "kiss.channels.googlechat_agent",
    "kiss.channels.irc_agent",
    "kiss.channels.line_agent",
    "kiss.channels.matrix_agent",
    "kiss.channels.mattermost_agent",
    "kiss.channels.msteams_agent",
    "kiss.channels.nextcloud_talk_agent",
    "kiss.channels.phone_control_agent",
    "kiss.channels.signal_agent",
    "kiss.channels.slack_agent",
    "kiss.channels.sms_agent",
    "kiss.channels.synology_chat_agent",
    "kiss.channels.telegram_agent",
    "kiss.channels.zalo_agent",
]

# Modules without daemon support
_NO_DAEMON_MODULES = [
    "kiss.channels.gmail_agent",
    "kiss.channels.imessage_agent",
    "kiss.channels.nostr_agent",
    "kiss.channels.tlon_agent",
    "kiss.channels.twitch_agent",
    "kiss.channels.whatsapp_agent",
]


@pytest.mark.parametrize("module_path", _DAEMON_MODULES)
def test_daemon_modules_have_make_daemon_backend(module_path: str) -> None:
    """Modules with daemon support expose a _make_daemon_backend() function."""
    mod = importlib.import_module(module_path)
    assert hasattr(mod, "_make_daemon_backend")
    assert callable(mod._make_daemon_backend)


@pytest.mark.parametrize("module_path", _NO_DAEMON_MODULES)
def test_non_daemon_modules_have_no_make_daemon_backend(module_path: str) -> None:
    """Modules without daemon support don't expose _make_daemon_backend()."""
    mod = importlib.import_module(module_path)
    assert not hasattr(mod, "_make_daemon_backend")


@pytest.mark.parametrize("module_path", _DAEMON_MODULES + _NO_DAEMON_MODULES)
def test_all_modules_have_main(module_path: str) -> None:
    """All channel agent modules expose a main() function."""
    mod = importlib.import_module(module_path)
    assert hasattr(mod, "main")
    assert callable(mod.main)
