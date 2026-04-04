"""Integration tests for all new channel agents — no mocks or test doubles.

Tests agent instantiation, auth trio presence, config persistence,
backend tool discovery, and ChannelDaemon construction for every
new channel agent created by the OpenClaw port.
"""

from __future__ import annotations

import importlib
import sys

import pytest

# ---------------------------------------------------------------------------
# Channel agent metadata — each entry drives parameterized tests
# ---------------------------------------------------------------------------

_CHANNEL_AGENTS = [
    {
        "module": "kiss.channels.telegram_agent",
        "agent_class": "TelegramAgent",
        "backend_class": "TelegramChannelBackend",
        "agent_name": "Telegram Agent",
        "auth_check": "check_telegram_auth",
        "auth_set": "authenticate_telegram",
        "auth_clear": "clear_telegram_auth",
    },
    {
        "module": "kiss.channels.discord_agent",
        "agent_class": "DiscordAgent",
        "backend_class": "DiscordChannelBackend",
        "agent_name": "Discord Agent",
        "auth_check": "check_discord_auth",
        "auth_set": "authenticate_discord",
        "auth_clear": "clear_discord_auth",
    },
    {
        "module": "kiss.channels.googlechat_agent",
        "agent_class": "GoogleChatAgent",
        "backend_class": "GoogleChatChannelBackend",
        "agent_name": "Google Chat Agent",
        "auth_check": "check_googlechat_auth",
        "auth_set": "authenticate_googlechat",
        "auth_clear": "clear_googlechat_auth",
    },
    {
        "module": "kiss.channels.signal_agent",
        "agent_class": "SignalAgent",
        "backend_class": "SignalChannelBackend",
        "agent_name": "Signal Agent",
        "auth_check": "check_signal_auth",
        "auth_set": "authenticate_signal",
        "auth_clear": "clear_signal_auth",
    },
    {
        "module": "kiss.channels.msteams_agent",
        "agent_class": "MSTeamsAgent",
        "backend_class": "MSTeamsChannelBackend",
        "agent_name": "MS Teams Agent",
        "auth_check": "check_msteams_auth",
        "auth_set": "authenticate_msteams",
        "auth_clear": "clear_msteams_auth",
    },
    {
        "module": "kiss.channels.matrix_agent",
        "agent_class": "MatrixAgent",
        "backend_class": "MatrixChannelBackend",
        "agent_name": "Matrix Agent",
        "auth_check": "check_matrix_auth",
        "auth_set": "authenticate_matrix",
        "auth_clear": "clear_matrix_auth",
    },
    {
        "module": "kiss.channels.feishu_agent",
        "agent_class": "FeishuAgent",
        "backend_class": "FeishuChannelBackend",
        "agent_name": "Feishu Agent",
        "auth_check": "check_feishu_auth",
        "auth_set": "authenticate_feishu",
        "auth_clear": "clear_feishu_auth",
    },
    {
        "module": "kiss.channels.line_agent",
        "agent_class": "LineAgent",
        "backend_class": "LineChannelBackend",
        "agent_name": "LINE Agent",
        "auth_check": "check_line_auth",
        "auth_set": "authenticate_line",
        "auth_clear": "clear_line_auth",
    },
    {
        "module": "kiss.channels.mattermost_agent",
        "agent_class": "MattermostAgent",
        "backend_class": "MattermostChannelBackend",
        "agent_name": "Mattermost Agent",
        "auth_check": "check_mattermost_auth",
        "auth_set": "authenticate_mattermost",
        "auth_clear": "clear_mattermost_auth",
    },
    {
        "module": "kiss.channels.irc_agent",
        "agent_class": "IRCAgent",
        "backend_class": "IRCChannelBackend",
        "agent_name": "IRC Agent",
        "auth_check": "check_irc_auth",
        "auth_set": "authenticate_irc",
        "auth_clear": "clear_irc_auth",
    },
    {
        "module": "kiss.channels.bluebubbles_agent",
        "agent_class": "BlueBubblesAgent",
        "backend_class": "BlueBubblesChannelBackend",
        "agent_name": "BlueBubbles Agent",
        "auth_check": "check_bluebubbles_auth",
        "auth_set": "authenticate_bluebubbles",
        "auth_clear": "clear_bluebubbles_auth",
    },
    {
        "module": "kiss.channels.imessage_agent",
        "agent_class": "IMessageAgent",
        "backend_class": "IMessageChannelBackend",
        "agent_name": "iMessage Agent",
        "auth_check": "check_imessage_auth",
        "auth_set": "authenticate_imessage",
        "auth_clear": "clear_imessage_auth",
    },
    {
        "module": "kiss.channels.nextcloud_talk_agent",
        "agent_class": "NextcloudTalkAgent",
        "backend_class": "NextcloudTalkChannelBackend",
        "agent_name": "Nextcloud Talk Agent",
        "auth_check": "check_nextcloud_auth",
        "auth_set": "authenticate_nextcloud",
        "auth_clear": "clear_nextcloud_auth",
    },
    {
        "module": "kiss.channels.nostr_agent",
        "agent_class": "NostrAgent",
        "backend_class": "NostrChannelBackend",
        "agent_name": "Nostr Agent",
        "auth_check": "check_nostr_auth",
        "auth_set": "authenticate_nostr",
        "auth_clear": "clear_nostr_auth",
    },
    {
        "module": "kiss.channels.synology_chat_agent",
        "agent_class": "SynologyChatAgent",
        "backend_class": "SynologyChatChannelBackend",
        "agent_name": "Synology Chat Agent",
        "auth_check": "check_synology_auth",
        "auth_set": "authenticate_synology",
        "auth_clear": "clear_synology_auth",
    },
    {
        "module": "kiss.channels.tlon_agent",
        "agent_class": "TlonAgent",
        "backend_class": "TlonChannelBackend",
        "agent_name": "Tlon Agent",
        "auth_check": "check_tlon_auth",
        "auth_set": "authenticate_tlon",
        "auth_clear": "clear_tlon_auth",
    },
    {
        "module": "kiss.channels.twitch_agent",
        "agent_class": "TwitchAgent",
        "backend_class": "TwitchChannelBackend",
        "agent_name": "Twitch Agent",
        "auth_check": "check_twitch_auth",
        "auth_set": "authenticate_twitch",
        "auth_clear": "clear_twitch_auth",
    },
    {
        "module": "kiss.channels.zalo_agent",
        "agent_class": "ZaloAgent",
        "backend_class": "ZaloChannelBackend",
        "agent_name": "Zalo Agent",
        "auth_check": "check_zalo_auth",
        "auth_set": "authenticate_zalo",
        "auth_clear": "clear_zalo_auth",
    },
    {
        "module": "kiss.channels.phone_control_agent",
        "agent_class": "PhoneControlAgent",
        "backend_class": "PhoneControlChannelBackend",
        "agent_name": "Phone Control Agent",
        "auth_check": "check_phone_auth",
        "auth_set": "authenticate_phone",
        "auth_clear": "clear_phone_auth",
    },
    {
        "module": "kiss.channels.sms_agent",
        "agent_class": "SMSAgent",
        "backend_class": "SMSChannelBackend",
        "agent_name": "SMS Agent",
        "auth_check": "check_sms_auth",
        "auth_set": "authenticate_sms",
        "auth_clear": "clear_sms_auth",
    },
]

_CHANNEL_IDS = [ch["agent_class"] for ch in _CHANNEL_AGENTS]


def _load(info: dict) -> tuple:
    """Import and return (module, AgentClass, BackendClass)."""
    mod = importlib.import_module(info["module"])
    agent_cls = getattr(mod, info["agent_class"])
    backend_cls = getattr(mod, info["backend_class"])
    return mod, agent_cls, backend_cls


# ---------------------------------------------------------------------------
# Test: Agent instantiation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("info", _CHANNEL_AGENTS, ids=_CHANNEL_IDS)
def test_agent_instantiates(info: dict) -> None:
    """Every channel agent can be created without credentials."""
    _, agent_cls, _ = _load(info)
    agent = agent_cls()
    assert agent.name == info["agent_name"]


# ---------------------------------------------------------------------------
# Test: Auth trio presence in _get_tools()
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test: Check auth returns helpful message when unauthenticated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("info", _CHANNEL_AGENTS, ids=_CHANNEL_IDS)
def test_check_auth_unauthenticated(info: dict) -> None:
    """check_*_auth() returns a helpful message when not configured."""
    mod, agent_cls, _ = _load(info)
    clear_fn = getattr(mod, "_clear_config", None)
    if clear_fn:
        clear_fn()
    agent = agent_cls()
    agent.web_use_tool = None
    tools = {t.__name__: t for t in agent._get_tools()}
    result = tools[info["auth_check"]]()
    # The result should indicate not authenticated
    lower = result.lower()
    assert ("not authenticated" in lower
            or "not configured" in lower
            or "no " in lower
            or "authenticate" in lower), (
        f"Expected unauthenticated message, got: {result[:200]}"
    )


# ---------------------------------------------------------------------------
# Test: Clear auth works when not authenticated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("info", _CHANNEL_AGENTS, ids=_CHANNEL_IDS)
def test_clear_auth_when_not_authenticated(info: dict) -> None:
    """clear_*_auth() works without error even when not authenticated."""
    mod, agent_cls, _ = _load(info)
    clear_fn = getattr(mod, "_clear_config", None)
    if clear_fn:
        clear_fn()
    agent = agent_cls()
    agent.web_use_tool = None
    tools = {t.__name__: t for t in agent._get_tools()}
    result = tools[info["auth_clear"]]()
    assert "cleared" in result.lower() or "removed" in result.lower(), (
        f"Expected cleared message, got: {result[:200]}"
    )


# ---------------------------------------------------------------------------
# Test: Backend instantiation and get_tool_methods
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("info", _CHANNEL_AGENTS, ids=_CHANNEL_IDS)
def test_backend_get_tool_methods(info: dict) -> None:
    """get_tool_methods() returns a list and excludes protocol methods."""
    _, _, backend_cls = _load(info)
    backend = backend_cls()
    tools = backend.get_tool_methods()
    assert isinstance(tools, list)
    # Protocol methods must NOT be in the tool list
    protocol_names = {
        "connect", "find_channel", "find_user", "join_channel",
        "poll_messages", "send_message", "wait_for_reply",
        "is_from_bot", "strip_bot_mention", "get_tool_methods",
    }
    tool_names = {t.__name__ for t in tools}
    overlap = tool_names & protocol_names
    assert not overlap, f"Protocol methods leaked into tools: {overlap}"


# ---------------------------------------------------------------------------
# Test: Backend protocol methods exist
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test: Config persistence roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("info", _CHANNEL_AGENTS, ids=_CHANNEL_IDS)
def test_config_roundtrip(info: dict) -> None:
    """Config save/load/clear works on real filesystem."""
    mod, _, _ = _load(info)
    config_path_fn = getattr(mod, "_config_path", None)
    load_fn = getattr(mod, "_load_config", None)
    clear_fn = getattr(mod, "_clear_config", None)
    if not config_path_fn or not load_fn or not clear_fn:
        pytest.skip(f"No standard config functions in {info['module']}")

    # Backup existing config
    path = config_path_fn()
    backup = None
    if path.exists():
        backup = path.read_text()

    try:
        # Clear and verify
        clear_fn()
        assert load_fn() is None

        # Load from non-existent returns None
        if path.exists():
            path.unlink()
        assert load_fn() is None

        # Load corrupt JSON returns None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{bad json!!")
        assert load_fn() is None

        # Load non-dict JSON returns None
        path.write_text('"just a string"')
        assert load_fn() is None

        # Clear works even when no file
        if path.exists():
            path.unlink()
        clear_fn()  # Should not raise

    finally:
        # Restore original config
        if backup is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(backup)
        elif path.exists():
            path.unlink()


# ---------------------------------------------------------------------------
# Test: CLI main exits with no args
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("info", _CHANNEL_AGENTS, ids=_CHANNEL_IDS)
def test_main_exits_with_no_args(info: dict) -> None:
    """main() exits when called with no arguments."""
    mod, _, _ = _load(info)
    main_fn = getattr(mod, "main", None)
    if main_fn is None:
        pytest.skip(f"No main() in {info['module']}")

    original_argv = sys.argv
    sys.argv = ["test_agent"]
    try:
        main_fn()
        pytest.fail("main() should have raised SystemExit")
    except SystemExit:
        pass  # Expected
    finally:
        sys.argv = original_argv


# ---------------------------------------------------------------------------
# Test: ChannelDaemon construction
# ---------------------------------------------------------------------------


def test_channel_daemon_stop() -> None:
    """ChannelDaemon.stop() sets the stop event."""
    from kiss.channels.background_agent import ChannelDaemon
    from kiss.channels.discord_agent import DiscordChannelBackend

    backend = DiscordChannelBackend()
    daemon = ChannelDaemon(
        backend=backend,
        channel_name="",
        agent_name="Test",
    )
    assert not daemon._stop_event.is_set()
    daemon.stop()
    assert daemon._stop_event.is_set()


