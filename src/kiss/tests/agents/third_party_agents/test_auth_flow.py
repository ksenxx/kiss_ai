# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for channel agent authentication flows.

For every channel agent whose ``authenticate_*()`` takes credentials,
the tool rejects empty and whitespace-only required parameters (calling
it with the parameter by name also pins the signature).  Platform-gated
agents (iMessage, BlueBubbles) report the platform error off macOS.
"""

from __future__ import annotations

import inspect
import json
import sys
from typing import Any

import pytest

from kiss.tests.agents.third_party_agents.channel_agent_registry import (
    ALL_CHANNEL_AGENTS,
    get_agent_class,
)
from kiss.tests.agents.third_party_agents.muse_test_utils import auth_tools

_MODULES = dict(ALL_CHANNEL_AGENTS)

_CREDENTIAL_AGENTS: dict[str, tuple[str, list[str]]] = {
    "TelegramAgent": ("authenticate_telegram", ["bot_token"]),
    # access_token is optional: without it the OAuth 2.0 device
    # authorisation sign-in (browser consent) is used.
    "MatrixAgent": ("authenticate_matrix", ["homeserver_url"]),
    "FeishuAgent": ("authenticate_feishu", ["app_id", "app_secret"]),
    "LineAgent": ("authenticate_line", ["channel_access_token"]),
    "MattermostAgent": ("authenticate_mattermost", ["url", "token"]),
    "IRCAgent": ("authenticate_irc", ["server", "nick"]),
    "BlueBubblesAgent": ("authenticate_bluebubbles", ["server_url", "password"]),
    # username/password are optional: with only the URL, Login Flow
    # v2 lets the user sign in and grant access in their browser.
    "NextcloudTalkAgent": ("authenticate_nextcloud", ["url"]),
    "NostrAgent": ("authenticate_nostr", ["private_key"]),
    "SynologyChatAgent": ("authenticate_synology", ["webhook_url"]),
    "TlonAgent": ("authenticate_tlon", ["ship_url", "code"]),
    # access_token is optional: without it the device code grant
    # (browser consent) is used.
    "TwitchAgent": ("authenticate_twitch", ["client_id"]),
    "ZaloAgent": ("authenticate_zalo", ["access_token"]),
    "PhoneControlAgent": ("authenticate_phone", ["device_ip"]),
    "SMSAgent": ("authenticate_sms", ["account_sid", "auth_token"]),
}
"""Agent class -> its ``authenticate_*`` tool and the parameters it refuses blank.

Agents absent here (Slack, Discord, Google Chat, Signal, MS Teams,
iMessage, WhatsApp, Gmail) start a browser or device sign-in with no
arguments.
"""


def _agent(class_name: str) -> Any:
    """Instantiate a fresh channel agent by class name."""
    return get_agent_class(class_name, _MODULES[class_name])()


@pytest.mark.parametrize("blank", ["", "   "], ids=["empty", "whitespace"])
@pytest.mark.parametrize("class_name", list(_CREDENTIAL_AGENTS))
def test_authenticate_rejects_blank_required_params(class_name: str, blank: str) -> None:
    """authenticate_*() returns an error for every empty or whitespace-only required param."""
    if class_name == "BlueBubblesAgent" and sys.platform != "darwin":
        pytest.skip("macOS-only agent")
    tool_name, required_params = _CREDENTIAL_AGENTS[class_name]
    auth_fn = auth_tools(_agent(class_name))[tool_name]
    sig = inspect.signature(auth_fn)

    for blank_param in required_params:
        kwargs: dict[str, Any] = {}
        for p_name, p in sig.parameters.items():
            if p_name == blank_param:
                kwargs[p_name] = blank
            elif p.annotation in (int, "int"):
                kwargs[p_name] = 1
            elif p.annotation in (bool, "bool"):
                kwargs[p_name] = False
            else:
                kwargs[p_name] = "test_value"
        result = auth_fn(**kwargs)
        lower = result.lower()
        assert "empty" in lower or "required" in lower, (
            f"authenticate should reject {blank!r} for '{blank_param}', got: {result[:300]}"
        )


class TestPlatformSpecificAuth:
    """Tests for platform-gated agents (iMessage, BlueBubbles)."""

    def test_imessage_check_on_non_darwin(self) -> None:
        """check_imessage_auth() returns platform error on non-darwin."""
        if sys.platform == "darwin":
            pytest.skip("Running on macOS — platform check passes")
        result = auth_tools(_agent("IMessageAgent"))["check_imessage_auth"]()
        data = json.loads(result)
        assert data.get("ok") is False
        assert "macOS" in data.get("error", "") or "macos" in data.get("error", "").lower()

    def test_imessage_authenticate_on_non_darwin(self) -> None:
        """authenticate_imessage() returns platform error on non-darwin."""
        if sys.platform == "darwin":
            pytest.skip("Running on macOS — platform check passes")
        result = auth_tools(_agent("IMessageAgent"))["authenticate_imessage"]()
        data = json.loads(result)
        assert data.get("ok") is False

    def test_bluebubbles_authenticate_on_non_darwin(self) -> None:
        """authenticate_bluebubbles() returns platform error on non-darwin."""
        if sys.platform == "darwin":
            pytest.skip("Running on macOS — platform check passes")
        result = auth_tools(_agent("BlueBubblesAgent"))["authenticate_bluebubbles"](
            server_url="http://localhost:1234", password="test"
        )
        data = json.loads(result)
        assert data.get("ok") is False
