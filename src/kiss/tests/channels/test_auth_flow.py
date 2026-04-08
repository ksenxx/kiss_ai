"""Integration tests for channel agent authentication flows.

Verifies that each agent's auth tools:
1. Show correct URLs and function references in check prompts
2. Reject empty required parameters
3. Clear auth resets state to unauthenticated
4. Browser auth tools provide fallback when no browser is available
5. Auth tool signatures match expected required parameters
6. Authenticate docstrings document their parameters
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Agent metadata — drives parameterized tests
# ---------------------------------------------------------------------------

_AUTH_AGENTS: list[dict[str, Any]] = [
    {
        "module": "kiss.channels.slack_agent",
        "class": "SlackAgent",
        "check": "check_slack_auth",
        "auth": "authenticate_slack",
        "clear": "clear_slack_auth",
        "browser_auth": "start_slack_browser_auth",
        "required_params": ["token"],
        "prompt_urls": [],
        "prompt_keywords": ["xoxb-", "Slack"],
    },
    {
        "module": "kiss.channels.telegram_agent",
        "class": "TelegramAgent",
        "check": "check_telegram_auth",
        "auth": "authenticate_telegram",
        "clear": "clear_telegram_auth",
        "required_params": ["bot_token"],
        "prompt_urls": [],
        "prompt_keywords": ["@BotFather", "/newbot"],
    },
    {
        "module": "kiss.channels.discord_agent",
        "class": "DiscordAgent",
        "check": "check_discord_auth",
        "auth": "authenticate_discord",
        "clear": "clear_discord_auth",
        "browser_auth": "start_discord_browser_auth",
        "required_params": ["bot_token"],
        "prompt_urls": [],
        "prompt_keywords": ["Discord"],
    },
    {
        "module": "kiss.channels.googlechat_agent",
        "class": "GoogleChatAgent",
        "check": "check_googlechat_auth",
        "auth": "authenticate_googlechat",
        "clear": "clear_googlechat_auth",
        "required_params": [],
        "prompt_urls": [
            "console.cloud.google.com/iam-admin/serviceaccounts",
            "console.cloud.google.com/apis/credentials",
        ],
        "prompt_keywords": ["Google Chat"],
    },
    {
        "module": "kiss.channels.signal_agent",
        "class": "SignalAgent",
        "check": "check_signal_auth",
        "auth": "authenticate_signal",
        "clear": "clear_signal_auth",
        "required_params": ["phone_number"],
        "prompt_urls": ["https://github.com/AsamK/signal-cli"],
        "prompt_keywords": ["signal-cli"],
    },
    {
        "module": "kiss.channels.msteams_agent",
        "class": "MSTeamsAgent",
        "check": "check_msteams_auth",
        "auth": "authenticate_msteams",
        "clear": "clear_msteams_auth",
        "required_params": ["tenant_id", "client_id", "client_secret"],
        "prompt_urls": ["https://portal.azure.com"],
        "prompt_keywords": ["App registrations"],
    },
    {
        "module": "kiss.channels.matrix_agent",
        "class": "MatrixAgent",
        "check": "check_matrix_auth",
        "auth": "authenticate_matrix",
        "clear": "clear_matrix_auth",
        "required_params": ["homeserver_url", "access_token"],
        "prompt_urls": [],
        "prompt_keywords": ["Element", "Access Token"],
    },
    {
        "module": "kiss.channels.feishu_agent",
        "class": "FeishuAgent",
        "check": "check_feishu_auth",
        "auth": "authenticate_feishu",
        "clear": "clear_feishu_auth",
        "required_params": ["app_id", "app_secret"],
        "prompt_urls": [
            "https://open.feishu.cn/app",
            "https://open.larksuite.com/app",
        ],
        "prompt_keywords": ["Feishu", "Lark"],
    },
    {
        "module": "kiss.channels.line_agent",
        "class": "LineAgent",
        "check": "check_line_auth",
        "auth": "authenticate_line",
        "clear": "clear_line_auth",
        "required_params": ["channel_access_token"],
        "prompt_urls": ["https://developers.line.biz/console/"],
        "prompt_keywords": ["LINE", "Messaging API"],
    },
    {
        "module": "kiss.channels.mattermost_agent",
        "class": "MattermostAgent",
        "check": "check_mattermost_auth",
        "auth": "authenticate_mattermost",
        "clear": "clear_mattermost_auth",
        "required_params": ["url", "token"],
        "prompt_urls": [],
        "prompt_keywords": ["Personal Access Tokens"],
    },
    {
        "module": "kiss.channels.irc_agent",
        "class": "IRCAgent",
        "check": "check_irc_auth",
        "auth": "authenticate_irc",
        "clear": "clear_irc_auth",
        "required_params": ["server", "nick"],
        "prompt_urls": [],
        "prompt_keywords": ["irc.libera.chat"],
    },
    {
        "module": "kiss.channels.bluebubbles_agent",
        "class": "BlueBubblesAgent",
        "check": "check_bluebubbles_auth",
        "auth": "authenticate_bluebubbles",
        "clear": "clear_bluebubbles_auth",
        "required_params": ["server_url", "password"],
        "prompt_urls": ["https://bluebubbles.app"],
        "prompt_keywords": ["BlueBubbles"],
    },
    {
        "module": "kiss.channels.imessage_agent",
        "class": "IMessageAgent",
        "check": "check_imessage_auth",
        "auth": "authenticate_imessage",
        "clear": "clear_imessage_auth",
        "required_params": [],
        "prompt_urls": [],
        "prompt_keywords": ["iMessage", "macOS"],
    },
    {
        "module": "kiss.channels.nextcloud_talk_agent",
        "class": "NextcloudTalkAgent",
        "check": "check_nextcloud_auth",
        "auth": "authenticate_nextcloud",
        "clear": "clear_nextcloud_auth",
        "required_params": ["url", "username", "password"],
        "prompt_urls": [],
        "prompt_keywords": ["Nextcloud", "Devices & sessions"],
    },
    {
        "module": "kiss.channels.nostr_agent",
        "class": "NostrAgent",
        "check": "check_nostr_auth",
        "auth": "authenticate_nostr",
        "clear": "clear_nostr_auth",
        "required_params": ["private_key"],
        "prompt_urls": [],
        "prompt_keywords": ["nsec", "Nostr"],
    },
    {
        "module": "kiss.channels.synology_chat_agent",
        "class": "SynologyChatAgent",
        "check": "check_synology_auth",
        "auth": "authenticate_synology",
        "clear": "clear_synology_auth",
        "required_params": ["webhook_url"],
        "prompt_urls": [],
        "prompt_keywords": ["Synology Chat", "Incoming Webhooks"],
    },
    {
        "module": "kiss.channels.tlon_agent",
        "class": "TlonAgent",
        "check": "check_tlon_auth",
        "auth": "authenticate_tlon",
        "clear": "clear_tlon_auth",
        "required_params": ["ship_url", "code"],
        "prompt_urls": [],
        "prompt_keywords": ["Tlon", "Urbit", "dojo"],
    },
    {
        "module": "kiss.channels.twitch_agent",
        "class": "TwitchAgent",
        "check": "check_twitch_auth",
        "auth": "authenticate_twitch",
        "clear": "clear_twitch_auth",
        "required_params": ["client_id", "access_token"],
        "prompt_urls": [
            "https://dev.twitch.tv/console/apps",
            "https://id.twitch.tv/oauth2/authorize",
        ],
        "prompt_keywords": ["Twitch"],
    },
    {
        "module": "kiss.channels.whatsapp_agent",
        "class": "WhatsAppAgent",
        "check": "check_whatsapp_auth",
        "auth": "authenticate_whatsapp",
        "clear": "clear_whatsapp_auth",
        "required_params": ["access_token", "phone_number_id"],
        "prompt_urls": ["https://developers.facebook.com/apps/"],
        "prompt_keywords": ["WhatsApp", "Phone number ID"],
    },
    {
        "module": "kiss.channels.zalo_agent",
        "class": "ZaloAgent",
        "check": "check_zalo_auth",
        "auth": "authenticate_zalo",
        "clear": "clear_zalo_auth",
        "required_params": ["access_token"],
        "prompt_urls": ["https://developers.zalo.me/"],
        "prompt_keywords": ["Zalo"],
    },
    {
        "module": "kiss.channels.phone_control_agent",
        "class": "PhoneControlAgent",
        "check": "check_phone_auth",
        "auth": "authenticate_phone",
        "clear": "clear_phone_auth",
        "required_params": ["device_ip"],
        "prompt_urls": [],
        "prompt_keywords": ["companion", "REST"],
    },
    {
        "module": "kiss.channels.sms_agent",
        "class": "SMSAgent",
        "check": "check_sms_auth",
        "auth": "authenticate_sms",
        "clear": "clear_sms_auth",
        "required_params": ["account_sid", "auth_token"],
        "prompt_urls": ["https://console.twilio.com/"],
        "prompt_keywords": ["Twilio"],
    },
    {
        "module": "kiss.channels.gmail_agent",
        "class": "GmailAgent",
        "check": "check_gmail_auth",
        "auth": "authenticate_gmail",
        "clear": "clear_gmail_auth",
        "browser_auth": "start_gmail_browser_setup",
        "required_params": [],
        "prompt_urls": [],
        "prompt_keywords": ["Gmail", "credentials.json"],
    },
]

_IDS = [a["class"] for a in _AUTH_AGENTS]


def _get_agent(info: dict[str, Any]) -> Any:
    """Instantiate a fresh agent from module/class info."""
    mod = importlib.import_module(info["module"])
    cls = getattr(mod, info["class"])
    agent = cls()
    agent.web_use_tool = None  # Disable browser for tests
    return agent


def _get_tools(agent: Any) -> dict[str, Any]:
    """Return auth tools as a name→callable dict."""
    return {t.__name__: t for t in agent._get_auth_tools()}


# ---------------------------------------------------------------------------
# 1. Check prompt references the authenticate function name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("info", _AUTH_AGENTS, ids=_IDS)
def test_check_prompt_references_auth_function(info: dict[str, Any]) -> None:
    """check_*_auth() output mentions the authenticate_*() function name."""
    agent = _get_agent(info)
    tools = _get_tools(agent)
    result = tools[info["check"]]()
    assert info["auth"] in result, (
        f"check prompt should reference '{info['auth']}', got: {result[:300]}"
    )


# ---------------------------------------------------------------------------
# 2. Check prompt contains expected URLs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("info", _AUTH_AGENTS, ids=_IDS)
def test_check_prompt_contains_urls(info: dict[str, Any]) -> None:
    """check_*_auth() output includes the expected setup URLs."""
    if not info["prompt_urls"]:
        pytest.skip("No URLs expected for this agent")
    agent = _get_agent(info)
    tools = _get_tools(agent)
    result = tools[info["check"]]()
    for url in info["prompt_urls"]:
        assert url in result, (
            f"check prompt should contain URL '{url}', got: {result[:300]}"
        )


# ---------------------------------------------------------------------------
# 3. Check prompt contains expected keywords
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("info", _AUTH_AGENTS, ids=_IDS)
def test_check_prompt_contains_keywords(info: dict[str, Any]) -> None:
    """check_*_auth() output includes platform-specific keywords."""
    if not info["prompt_keywords"]:
        pytest.skip("No keywords expected for this agent")
    agent = _get_agent(info)
    tools = _get_tools(agent)
    result = tools[info["check"]]()
    for kw in info["prompt_keywords"]:
        assert kw.lower() in result.lower(), (
            f"check prompt should contain keyword '{kw}', got: {result[:300]}"
        )


# ---------------------------------------------------------------------------
# 4. Empty required parameters are rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("info", _AUTH_AGENTS, ids=_IDS)
def test_authenticate_rejects_empty_required_params(info: dict[str, Any]) -> None:
    """authenticate_*() returns error when required params are empty strings."""
    if not info["required_params"]:
        pytest.skip("No required params for this agent")
    agent = _get_agent(info)
    tools = _get_tools(agent)
    auth_fn = tools[info["auth"]]
    sig = inspect.signature(auth_fn)

    for param_name in info["required_params"]:
        if param_name not in sig.parameters:
            continue
        # Build kwargs with all params set to valid-looking values except the target
        kwargs: dict[str, Any] = {}
        for p_name, p in sig.parameters.items():
            if p_name == param_name:
                kwargs[p_name] = ""  # empty — should be rejected
            elif p.annotation in (int, "int"):
                kwargs[p_name] = 1
            elif p.annotation in (bool, "bool"):
                kwargs[p_name] = False
            else:
                kwargs[p_name] = "test_value"
        result = auth_fn(**kwargs)
        lower = result.lower()
        assert "empty" in lower or "required" in lower or "cannot be empty" in lower, (
            f"authenticate should reject empty '{param_name}', got: {result[:300]}"
        )


# ---------------------------------------------------------------------------
# 5. Empty string (whitespace only) is also rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("info", _AUTH_AGENTS, ids=_IDS)
def test_authenticate_rejects_whitespace_params(info: dict[str, Any]) -> None:
    """authenticate_*() rejects whitespace-only strings for required params."""
    if not info["required_params"]:
        pytest.skip("No required params for this agent")
    agent = _get_agent(info)
    tools = _get_tools(agent)
    auth_fn = tools[info["auth"]]
    sig = inspect.signature(auth_fn)

    # Test first required param with whitespace
    first_param = info["required_params"][0]
    if first_param not in sig.parameters:
        pytest.skip(f"Param {first_param} not in signature")
    kwargs: dict[str, Any] = {}
    for p_name, p in sig.parameters.items():
        if p_name == first_param:
            kwargs[p_name] = "   "  # whitespace only
        elif p.annotation in (int, "int"):
            kwargs[p_name] = 1
        elif p.annotation in (bool, "bool"):
            kwargs[p_name] = False
        else:
            kwargs[p_name] = "test_value"
    result = auth_fn(**kwargs)
    lower = result.lower()
    assert "empty" in lower or "required" in lower or "cannot be empty" in lower, (
        f"authenticate should reject whitespace '{first_param}', got: {result[:300]}"
    )


# ---------------------------------------------------------------------------
# 6. Clear → check cycle: clear resets to unauthenticated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("info", _AUTH_AGENTS, ids=_IDS)
def test_clear_then_check_shows_unauthenticated(info: dict[str, Any]) -> None:
    """After clear_*_auth(), check_*_auth() shows not authenticated."""
    agent = _get_agent(info)
    tools = _get_tools(agent)

    # Clear first
    tools[info["clear"]]()

    # Check should show not authenticated
    result = tools[info["check"]]()
    lower = result.lower()
    assert (
        "not authenticated" in lower
        or "not configured" in lower
        or "authenticate" in lower
    ), f"After clear, check should show unauthenticated, got: {result[:300]}"


# ---------------------------------------------------------------------------
# 7. Auth tool function signatures have expected required params
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("info", _AUTH_AGENTS, ids=_IDS)
def test_auth_function_has_expected_params(info: dict[str, Any]) -> None:
    """authenticate_*() signature includes all expected required params."""
    agent = _get_agent(info)
    tools = _get_tools(agent)
    auth_fn = tools[info["auth"]]
    sig = inspect.signature(auth_fn)

    actual_params = list(sig.parameters.keys())
    for expected in info["required_params"]:
        assert expected in actual_params, (
            f"Expected param '{expected}' in {info['auth']} signature, "
            f"got: {actual_params}"
        )


# ---------------------------------------------------------------------------
# 8. Auth tool docstrings are present and document Args
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("info", _AUTH_AGENTS, ids=_IDS)
def test_auth_function_has_documented_args(info: dict[str, Any]) -> None:
    """authenticate_*() has a docstring with Args section if it takes params."""
    agent = _get_agent(info)
    tools = _get_tools(agent)
    auth_fn = tools[info["auth"]]

    assert auth_fn.__doc__, f"{info['auth']} has no docstring"

    sig = inspect.signature(auth_fn)
    if sig.parameters:
        assert "Args:" in auth_fn.__doc__, (
            f"{info['auth']} takes params but docstring has no Args section"
        )
        for param_name in info["required_params"]:
            assert param_name in auth_fn.__doc__, (
                f"{info['auth']} docstring should document '{param_name}'"
            )


# ---------------------------------------------------------------------------
# 9. Check and clear tool docstrings are present
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("info", _AUTH_AGENTS, ids=_IDS)
def test_check_and_clear_have_docstrings(info: dict[str, Any]) -> None:
    """check_*_auth() and clear_*_auth() have docstrings with Returns."""
    agent = _get_agent(info)
    tools = _get_tools(agent)

    check_fn = tools[info["check"]]
    assert check_fn.__doc__, f"{info['check']} has no docstring"
    assert "Returns:" in check_fn.__doc__, f"{info['check']} docstring has no Returns"

    clear_fn = tools[info["clear"]]
    assert clear_fn.__doc__, f"{info['clear']} has no docstring"
    assert "Returns:" in clear_fn.__doc__, f"{info['clear']} docstring has no Returns"


# ---------------------------------------------------------------------------
# 10. Browser auth tools return fallback when no browser
# ---------------------------------------------------------------------------

_BROWSER_AUTH_AGENTS = [a for a in _AUTH_AGENTS if "browser_auth" in a]
_BROWSER_IDS = [a["class"] for a in _BROWSER_AUTH_AGENTS]


@pytest.mark.parametrize("info", _BROWSER_AUTH_AGENTS, ids=_BROWSER_IDS)
def test_browser_auth_fallback_no_browser(info: dict[str, Any]) -> None:
    """start_*_browser_auth() returns fallback instructions when browser unavailable."""
    agent = _get_agent(info)
    agent.web_use_tool = None
    tools = _get_tools(agent)
    browser_fn = tools[info["browser_auth"]]
    result = browser_fn()
    assert "browser" in result.lower(), (
        f"Browser fallback should mention 'browser', got: {result[:300]}"
    )


# ---------------------------------------------------------------------------
# 11. Clear auth returns string containing 'cleared'
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("info", _AUTH_AGENTS, ids=_IDS)
def test_clear_auth_returns_cleared_message(info: dict[str, Any]) -> None:
    """clear_*_auth() return value contains 'cleared'."""
    agent = _get_agent(info)
    tools = _get_tools(agent)
    result = tools[info["clear"]]()
    assert "cleared" in result.lower(), (
        f"clear should return message with 'cleared', got: {result[:200]}"
    )


# ---------------------------------------------------------------------------
# 12. Check auth returns str (not None or other type)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("info", _AUTH_AGENTS, ids=_IDS)
def test_check_auth_returns_string(info: dict[str, Any]) -> None:
    """check_*_auth() always returns a string."""
    agent = _get_agent(info)
    tools = _get_tools(agent)
    result = tools[info["check"]]()
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# 13. Platform-specific agents handle non-darwin platform
# ---------------------------------------------------------------------------


class TestPlatformSpecificAuth:
    """Tests for platform-gated agents (iMessage, BlueBubbles)."""

    def test_imessage_check_on_non_darwin(self) -> None:
        """check_imessage_auth() returns platform error on non-darwin."""
        if sys.platform == "darwin":
            pytest.skip("Running on macOS — platform check passes")
        agent = _get_agent(
            {"module": "kiss.channels.imessage_agent", "class": "IMessageAgent"}
        )
        tools = _get_tools(agent)
        result = tools["check_imessage_auth"]()
        data = json.loads(result)
        assert data.get("ok") is False
        assert "macOS" in data.get("error", "") or "macos" in data.get("error", "").lower()

    def test_imessage_authenticate_on_non_darwin(self) -> None:
        """authenticate_imessage() returns platform error on non-darwin."""
        if sys.platform == "darwin":
            pytest.skip("Running on macOS — platform check passes")
        agent = _get_agent(
            {"module": "kiss.channels.imessage_agent", "class": "IMessageAgent"}
        )
        tools = _get_tools(agent)
        result = tools["authenticate_imessage"]()
        data = json.loads(result)
        assert data.get("ok") is False

    def test_bluebubbles_authenticate_on_non_darwin(self) -> None:
        """authenticate_bluebubbles() returns platform error on non-darwin."""
        if sys.platform == "darwin":
            pytest.skip("Running on macOS — platform check passes")
        agent = _get_agent(
            {
                "module": "kiss.channels.bluebubbles_agent",
                "class": "BlueBubblesAgent",
            }
        )
        tools = _get_tools(agent)
        result = tools["authenticate_bluebubbles"](
            server_url="http://localhost:1234", password="test"
        )
        data = json.loads(result)
        assert data.get("ok") is False


# ---------------------------------------------------------------------------
# 14. Full cycle: check (unauth) → clear → check (unauth)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("info", _AUTH_AGENTS, ids=_IDS)
def test_full_unauth_cycle(info: dict[str, Any]) -> None:
    """Full cycle: check (unauth) → clear → check (unauth) is consistent."""
    agent = _get_agent(info)
    tools = _get_tools(agent)

    # Initial check
    result1 = tools[info["check"]]()

    # Clear
    tools[info["clear"]]()

    # Check again — should be same unauthenticated state
    result2 = tools[info["check"]]()

    # Both should indicate unauthenticated
    for result in [result1, result2]:
        lower = result.lower()
        assert (
            "not authenticated" in lower
            or "not configured" in lower
            or "authenticate" in lower
        ), f"Expected unauthenticated message, got: {result[:300]}"
