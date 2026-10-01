# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for channel agent authentication flows.

For every channel agent in ``_AUTH_AGENTS``, ``authenticate_*()``
rejects empty and whitespace-only required parameters and its signature
carries every expected required parameter.  Platform-gated agents
(iMessage, BlueBubbles) report the platform error off macOS.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from typing import Any

import pytest

_AUTH_AGENTS: list[dict[str, Any]] = [
    {
        "module": "kiss.agents.third_party_agents.slack.slack_sea",
        "class": "SlackAgent",
        "auth": "authenticate_slack",
        # authenticate_slack() with no arguments starts the PKCE sign-in.
        "required_params": [],
    },
    {
        "module": "kiss.agents.third_party_agents.telegram.telegram_sea",
        "class": "TelegramAgent",
        "auth": "authenticate_telegram",
        "required_params": ["bot_token"],
    },
    {
        "module": "kiss.agents.third_party_agents.discord.discord_sea",
        "class": "DiscordAgent",
        "auth": "authenticate_discord",
        # authenticate_discord() with no arguments starts OAuth sign-in.
        "required_params": [],
    },
    {
        "module": "kiss.agents.third_party_agents.googlechat.googlechat_sea",
        "class": "GoogleChatAgent",
        "auth": "authenticate_googlechat",
        "required_params": [],
    },
    {
        "module": "kiss.agents.third_party_agents.signal.signal_sea",
        "class": "SignalAgent",
        "auth": "authenticate_signal",
        # phone_number is optional: without it ``signal-cli link`` runs and
        # the user scans the QR code from their phone.
        "required_params": [],
    },
    {
        "module": "kiss.agents.third_party_agents.msteams.msteams_sea",
        "class": "MSTeamsAgent",
        "auth": "authenticate_msteams",
        # Device code sign-in with the KISS app; tenant_id is optional.
        "required_params": [],
    },
    {
        "module": "kiss.agents.third_party_agents.matrix.matrix_sea",
        "class": "MatrixAgent",
        "auth": "authenticate_matrix",
        # access_token is optional: without it the OAuth 2.0 device
        # authorisation sign-in (browser consent) is used.
        "required_params": ["homeserver_url"],
    },
    {
        "module": "kiss.agents.third_party_agents.feishu.feishu_sea",
        "class": "FeishuAgent",
        "auth": "authenticate_feishu",
        "required_params": ["app_id", "app_secret"],
    },
    {
        "module": "kiss.agents.third_party_agents.line.line_sea",
        "class": "LineAgent",
        "auth": "authenticate_line",
        "required_params": ["channel_access_token"],
    },
    {
        "module": "kiss.agents.third_party_agents.mattermost.mattermost_sea",
        "class": "MattermostAgent",
        "auth": "authenticate_mattermost",
        "required_params": ["url", "token"],
    },
    {
        "module": "kiss.agents.third_party_agents.irc.irc_sea",
        "class": "IRCAgent",
        "auth": "authenticate_irc",
        "required_params": ["server", "nick"],
    },
    {
        "module": "kiss.agents.third_party_agents.bluebubbles.bluebubbles_sea",
        "class": "BlueBubblesAgent",
        "auth": "authenticate_bluebubbles",
        "required_params": ["server_url", "password"],
        "macos_only": True,
    },
    {
        "module": "kiss.agents.third_party_agents.imessage.imessage_sea",
        "class": "IMessageAgent",
        "auth": "authenticate_imessage",
        "required_params": [],
        "macos_only": True,
    },
    {
        "module": "kiss.agents.third_party_agents.nextcloud.nextcloud_sea",
        "class": "NextcloudTalkAgent",
        "auth": "authenticate_nextcloud",
        # username/password are optional: with only the URL, Login Flow
        # v2 lets the user sign in and grant access in their browser.
        "required_params": ["url"],
    },
    {
        "module": "kiss.agents.third_party_agents.nostr.nostr_sea",
        "class": "NostrAgent",
        "auth": "authenticate_nostr",
        "required_params": ["private_key"],
    },
    {
        "module": "kiss.agents.third_party_agents.synology.synology_sea",
        "class": "SynologyChatAgent",
        "auth": "authenticate_synology",
        "required_params": ["webhook_url"],
    },
    {
        "module": "kiss.agents.third_party_agents.tlon.tlon_sea",
        "class": "TlonAgent",
        "auth": "authenticate_tlon",
        "required_params": ["ship_url", "code"],
    },
    {
        "module": "kiss.agents.third_party_agents.twitch.twitch_sea",
        "class": "TwitchAgent",
        "auth": "authenticate_twitch",
        # access_token is optional: without it the device code grant
        # (browser consent) is used.
        "required_params": ["client_id"],
    },
    {
        # QR-paired personal WhatsApp (whatsapp-mcp bridge): authenticate
        # takes no credentials — it clones and builds the bridge.
        "module": "kiss.agents.third_party_agents.whatsapp.whatsapp_sea",
        "class": "WhatsAppAgent",
        "auth": "authenticate_whatsapp",
        "required_params": [],
    },
    {
        "module": "kiss.agents.third_party_agents.zalo.zalo_sea",
        "class": "ZaloAgent",
        "auth": "authenticate_zalo",
        "required_params": ["access_token"],
    },
    {
        "module": "kiss.agents.third_party_agents.phone.phone_sea",
        "class": "PhoneControlAgent",
        "auth": "authenticate_phone",
        "required_params": ["device_ip"],
    },
    {
        "module": "kiss.agents.third_party_agents.sms.sms_sea",
        "class": "SMSAgent",
        "auth": "authenticate_sms",
        "required_params": ["account_sid", "auth_token"],
    },
    {
        "module": "kiss.agents.third_party_agents.gmail.gmail_sea",
        "class": "GmailAgent",
        "auth": "authenticate_gmail",
        "required_params": [],
    },
]

_IDS = [a["class"] for a in _AUTH_AGENTS]


def _get_agent(info: dict[str, Any]) -> Any:
    """Instantiate a fresh agent from module/class info."""
    mod = importlib.import_module(info["module"])
    cls = getattr(mod, info["class"])
    agent = cls()
    return agent


def _get_tools(agent: Any) -> dict[str, Any]:
    """Return auth tools as a name→callable dict."""
    return {t.__name__: t for t in agent._get_auth_tools()}


@pytest.mark.parametrize("info", _AUTH_AGENTS, ids=_IDS)
def test_authenticate_rejects_empty_required_params(info: dict[str, Any]) -> None:
    """authenticate_*() returns error when required params are empty strings."""
    if info.get("macos_only") and sys.platform != "darwin":
        pytest.skip("macOS-only agent")
    if not info["required_params"]:
        pytest.skip("No required params for this agent")
    agent = _get_agent(info)
    tools = _get_tools(agent)
    auth_fn = tools[info["auth"]]
    sig = inspect.signature(auth_fn)

    for param_name in info["required_params"]:
        if param_name not in sig.parameters:
            continue
        kwargs: dict[str, Any] = {}
        for p_name, p in sig.parameters.items():
            if p_name == param_name:
                kwargs[p_name] = ""
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


@pytest.mark.parametrize("info", _AUTH_AGENTS, ids=_IDS)
def test_authenticate_rejects_whitespace_params(info: dict[str, Any]) -> None:
    """authenticate_*() rejects whitespace-only strings for required params."""
    if info.get("macos_only") and sys.platform != "darwin":
        pytest.skip("macOS-only agent")
    if not info["required_params"]:
        pytest.skip("No required params for this agent")
    agent = _get_agent(info)
    tools = _get_tools(agent)
    auth_fn = tools[info["auth"]]
    sig = inspect.signature(auth_fn)

    first_param = info["required_params"][0]
    if first_param not in sig.parameters:
        pytest.skip(f"Param {first_param} not in signature")
    kwargs: dict[str, Any] = {}
    for p_name, p in sig.parameters.items():
        if p_name == first_param:
            kwargs[p_name] = "   "
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


class TestPlatformSpecificAuth:
    """Tests for platform-gated agents (iMessage, BlueBubbles)."""

    def test_imessage_check_on_non_darwin(self) -> None:
        """check_imessage_auth() returns platform error on non-darwin."""
        if sys.platform == "darwin":
            pytest.skip("Running on macOS — platform check passes")
        agent = _get_agent(
            {
                "module": "kiss.agents.third_party_agents.imessage.imessage_sea",
                "class": "IMessageAgent",
            }
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
            {
                "module": "kiss.agents.third_party_agents.imessage.imessage_sea",
                "class": "IMessageAgent",
            }
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
                "module": "kiss.agents.third_party_agents.bluebubbles.bluebubbles_sea",
                "class": "BlueBubblesAgent",
            }
        )
        tools = _get_tools(agent)
        result = tools["authenticate_bluebubbles"](
            server_url="http://localhost:1234", password="test"
        )
        data = json.loads(result)
        assert data.get("ok") is False


