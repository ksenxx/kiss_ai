# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the Discord and Slack token paste-back auth contracts.

Discord and Slack authenticate with a static bot token created by hand
in a developer portal (discord.com/developers/applications,
api.slack.com/apps) that sits behind the user's own login.  Unlike the
Google channels there is no loopback consent server, no redirect URL to
replay, and no finish_* tool — the paste-back artifact is the token
itself.  These tests pin that contract in the channel prompts and in
the check/start auth-tool texts: check first, drive the portal only
while it loads without a login wall, never ask for the user's password
or 2FA code, and on any failure hand off via ask_user_question() so the
user creates the app in their OWN browser and pastes the token back.
"""

from __future__ import annotations

import re

import pytest

from kiss.agents.third_party_agents._channel_agent_utils import BaseChannelAgent
from kiss.agents.third_party_agents.discord_sea import DiscordAgent
from kiss.agents.third_party_agents.slack_sea import SlackAgent

_CASES: list[tuple[type[BaseChannelAgent], str, str, str, str]] = [
    (
        DiscordAgent,
        "discord",
        "Discord",
        "https://discord.com/developers/applications",
        "authenticate_discord(bot_token=",
    ),
    (
        SlackAgent,
        "slack",
        "Slack",
        "https://api.slack.com/apps",
        "authenticate_slack(token=",
    ),
]
_IDS = [c[1] for c in _CASES]


def _unauthenticated(agent_cls: type[BaseChannelAgent]) -> BaseChannelAgent:
    """Instantiate an agent and strip any credentials the host machine has.

    The config directories are resolved from the real home directory at
    import time, so a developer machine may construct an authenticated
    agent; the not-authenticated branch of the check tool is what these
    tests exercise.

    Args:
        agent_cls: DiscordAgent or SlackAgent.

    Returns:
        An agent whose backend holds no credential.
    """
    agent = agent_cls()
    if agent_cls is DiscordAgent:
        agent._backend._bot_token = ""  # type: ignore[attr-defined]
    else:
        agent._backend._client = None  # type: ignore[attr-defined]
    return agent


@pytest.mark.parametrize("agent_cls,service,label,portal,auth_call", _CASES, ids=_IDS)
def test_prompt_describes_token_paste_back_hand_off(
    agent_cls: type[BaseChannelAgent],
    service: str,
    label: str,
    portal: str,
    auth_call: str,
) -> None:
    """Both prompts carry the full token paste-back hand-off."""
    prompt = agent_cls.channel_system_prompt
    assert f"## {label} Authentication" in prompt
    # Check-first rule: never redo setup over a valid token.
    assert f"check_{service}_auth()" in prompt
    assert "never re-run setup over a valid token" in prompt
    # The portal sits behind the user's login; the agent must not
    # collect their credentials.
    assert portal in prompt
    assert "password or 2FA code" in prompt
    # Browser driving is best-effort only; on failure, no retry loop.
    assert f"start_{service}_browser_auth()" in prompt
    assert "do not retry it or relaunch the browser" in prompt
    # The hand-off: user creates the app in their OWN browser and
    # pastes the token back; the agent stores it via authenticate_*.
    assert "ask_user_question()" in prompt
    assert "OWN browser" in prompt
    assert auth_call in prompt
    # Verify at the end.
    assert prompt.rstrip().endswith(f"verifying with check_{service}_auth().")


@pytest.mark.parametrize("agent_cls,service,label,portal,auth_call", _CASES, ids=_IDS)
def test_prompt_has_no_stale_or_google_only_wording(
    agent_cls: type[BaseChannelAgent],
    service: str,
    label: str,
    portal: str,
    auth_call: str,
) -> None:
    """The old browser mandate and Google-only OAuth concepts are gone."""
    prompt = agent_cls.channel_system_prompt
    # Pre-fix wording that caused headless retry loops.
    assert "stuck on login or captcha" not in prompt
    assert "autonomously" not in prompt
    assert "computer use" not in prompt
    # A hardcoded model name has no place in a channel prompt.
    assert "claude" not in prompt.lower()
    # These channels have no OAuth consent server: naming the Google
    # flow's artifacts would reference tools that do not exist here.
    assert "finish_" not in prompt
    assert "curl" not in prompt
    assert "consent_required" not in prompt
    assert "redirect URL" not in prompt


@pytest.mark.parametrize("agent_cls,service,label,portal,auth_call", _CASES, ids=_IDS)
def test_prompt_references_only_real_tools(
    agent_cls: type[BaseChannelAgent],
    service: str,
    label: str,
    portal: str,
    auth_call: str,
) -> None:
    """Every tool the prompt names must exist among the agent's auth tools."""
    tool_names = {tool.__name__ for tool in agent_cls()._get_auth_tools()}
    referenced = set(re.findall(r"([a-z_]+)\(", agent_cls.channel_system_prompt))
    missing = referenced - tool_names - {"ask_user_question"}
    assert not missing, f"{label} prompt names unknown tools: {sorted(missing)}"


@pytest.mark.parametrize("agent_cls,service,label,portal,auth_call", _CASES, ids=_IDS)
def test_check_tool_text_offers_the_hand_off(
    agent_cls: type[BaseChannelAgent],
    service: str,
    label: str,
    portal: str,
    auth_call: str,
) -> None:
    """The not-authenticated check text agrees with the prompt's hand-off."""
    agent = _unauthenticated(agent_cls)
    tools = {tool.__name__: tool for tool in agent._get_auth_tools()}
    text = tools[f"check_{service}_auth"]()
    assert f"Not authenticated with {label}" in text
    assert f"start_{service}_browser_auth()" in text
    assert "ask_user_question" in text
    assert portal in text
    assert "OWN browser" in text
    assert "paste back" in text
    assert "password or 2FA code" in text
    assert "autonomously" not in text


@pytest.mark.parametrize("agent_cls,service,label,portal,auth_call", _CASES, ids=_IDS)
def test_start_tool_text_offers_the_hand_off(
    agent_cls: type[BaseChannelAgent],
    service: str,
    label: str,
    portal: str,
    auth_call: str,
) -> None:
    """start_*_browser_auth() keeps browser steps but falls back to paste-back."""
    agent = _unauthenticated(agent_cls)
    tools = {tool.__name__: tool for tool in agent._get_auth_tools()}
    start = tools[f"start_{service}_browser_auth"]
    text = start()
    assert portal in text
    assert "browser" in text.lower()
    assert "ask_user_question" in text
    assert "OWN browser" in text
    assert "paste back" in text
    # The docstring steers the agent too: it must forbid credential
    # collection and retry loops.
    doc = start.__doc__ or ""
    assert "do not retry" in doc
    assert "password or 2FA code" in doc
    assert "ask_user_question()" in doc
    assert "autonomously" not in doc
