# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the Google channel agents' authentication system prompts.

Each ``channel_system_prompt`` is appended verbatim to every task
dispatched to its channel agent, so its wording directly steers the
subagent's sign-in behaviour.  These tests pin the Composio Connect
Link contract shared by the Gmail, Google Drive, Google Chat, Google
Calendar, Google Docs, and Google Sheets prompts: check first, then
authenticate, the user opens the link in their OWN browser, then
finish; the agent never drives the Google sign-in page itself or asks
for the user's password.
"""

from __future__ import annotations

import re

import pytest

from kiss.agents.third_party_agents._channel_agent_utils import BaseChannelAgent
from kiss.agents.third_party_agents.gcal.gcal_sea import GoogleCalendarAgent
from kiss.agents.third_party_agents.gdocs.gdocs_sea import GoogleDocsAgent
from kiss.agents.third_party_agents.gdrive.gdrive_sea import GoogleDriveAgent
from kiss.agents.third_party_agents.gmail.gmail_sea import GmailAgent
from kiss.agents.third_party_agents.googlechat.googlechat_sea import GoogleChatAgent
from kiss.agents.third_party_agents.gsheets.gsheets_sea import GoogleSheetsAgent

_CASES = [
    (GmailAgent, "gmail", "Gmail"),
    (GoogleDriveAgent, "google_drive", "Google Drive"),
    (GoogleChatAgent, "googlechat", "Google Chat"),
    (GoogleCalendarAgent, "google_calendar", "Google Calendar"),
    (GoogleDocsAgent, "google_docs", "Google Docs"),
    (GoogleSheetsAgent, "google_sheets", "Google Sheets"),
]


@pytest.mark.parametrize("agent_cls,service,label", _CASES, ids=[c[1] for c in _CASES])
def test_prompt_describes_connect_link_hand_off(
    agent_cls: type[BaseChannelAgent], service: str, label: str
) -> None:
    """Every Google auth prompt carries the check -> authenticate -> finish hand-off."""
    prompt = agent_cls.channel_system_prompt
    assert f"## {label} Authentication" in prompt
    assert "Composio" in prompt
    check = prompt.index(f"check_{service}_auth()")
    authenticate = prompt.index(f"authenticate_{service}(")
    finish = prompt.index(f"finish_{service}_auth()")
    assert check < authenticate < finish
    assert "Connect Link" in prompt
    assert "ask_user_question()" in prompt
    assert "OWN browser" in prompt
    assert "Never open the link in your built-in browser" in prompt
    assert "password or 2FA code" in prompt
    assert "'pending'" in prompt


@pytest.mark.parametrize("agent_cls,service,label", _CASES, ids=[c[1] for c in _CASES])
def test_prompt_references_only_real_tools(
    agent_cls: type[BaseChannelAgent], service: str, label: str
) -> None:
    """Every tool the prompt names must exist among the agent's auth tools."""
    tool_names = {tool.__name__ for tool in agent_cls()._get_auth_tools()}
    referenced = set(re.findall(r"([a-z_]+)\(", agent_cls.channel_system_prompt))
    missing = referenced - tool_names - {"ask_user_question"}
    assert not missing, f"{label} prompt names unknown tools: {sorted(missing)}"


def test_googlechat_prompt_covers_auth_config_and_service_account() -> None:
    """Google Chat needs a custom auth config, and keeps its Chat bot option."""
    prompt = GoogleChatAgent.channel_system_prompt
    assert "KISS_COMPOSIO_AUTH_CONFIG_GOOGLECHAT" in prompt
    assert "authenticate_googlechat_service_account()" in prompt
