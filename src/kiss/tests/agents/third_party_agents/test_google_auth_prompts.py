# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the Google channel agents' authentication system prompts.

Each ``channel_system_prompt`` is appended verbatim to every task
dispatched to its channel agent, so its wording directly steers the
subagent's OAuth behaviour.  These tests pin the paste-back consent
hand-off contract shared by the Gmail, Google Drive, Google Chat,
Google Calendar, Google Docs, and Google Sheets prompts: the tool opens
the consent page in the user's default browser when it can, consent is
approved by the user in their OWN browser (the URL is always shown so
they can open it by hand) and a redirect URL pasted back from another
device is replayed locally — the agent must never drive
accounts.google.com itself or ask for the user's password.
"""

from __future__ import annotations

import pytest

from kiss.agents.third_party_agents._channel_agent_utils import BaseChannelAgent
from kiss.agents.third_party_agents.gcal_sea import GoogleCalendarAgent
from kiss.agents.third_party_agents.gdocs_sea import GoogleDocsAgent
from kiss.agents.third_party_agents.gdrive_sea import GoogleDriveAgent
from kiss.agents.third_party_agents.gmail_sea import GmailAgent
from kiss.agents.third_party_agents.googlechat_sea import GoogleChatAgent
from kiss.agents.third_party_agents.gsheets_sea import GoogleSheetsAgent

_CASES = [
    (GmailAgent, "gmail", "Gmail"),
    (GoogleDriveAgent, "google_drive", "Google Drive"),
    (GoogleChatAgent, "googlechat", "Google Chat"),
    (GoogleCalendarAgent, "google_calendar", "Google Calendar"),
    (GoogleDocsAgent, "google_docs", "Google Docs"),
    (GoogleSheetsAgent, "google_sheets", "Google Sheets"),
]


@pytest.mark.parametrize(
    "agent_cls,service,label", _CASES, ids=[c[1] for c in _CASES]
)
def test_prompt_describes_paste_back_hand_off(
    agent_cls: type[BaseChannelAgent], service: str, label: str
) -> None:
    """Every Google auth prompt carries the full paste-back hand-off."""
    prompt = agent_cls.channel_system_prompt
    assert f"## {label} Authentication" in prompt
    # Check-first rule: never re-run OAuth over valid credentials.
    assert f"check_{service}_auth()" in prompt
    assert "never start an OAuth flow over valid credentials" in prompt
    # The consent hand-off: the tool opens the page in the user's default
    # browser when it can; the agent always shows the URL, and a redirect
    # URL pasted back from another device is replayed locally.
    assert "'consent_required'" in prompt
    assert "user's default browser" in prompt
    assert "'browser_opened'" in prompt
    assert "ALWAYS call ask_user_question() with the full auth_url" in prompt
    assert "do NOT open the auth_url or any accounts.google.com" in prompt
    assert "password or 2FA code" in prompt
    assert "ask_user_question()" in prompt
    assert "paste back" in prompt
    assert "curl -s '<pasted redirect URL>'" in prompt
    assert f"finish_{service}_auth()" in prompt
    # No browser retries: switch to the hand-off on the first failure.
    assert "do not retry it or relaunch the browser" in prompt


@pytest.mark.parametrize(
    "agent_cls,service,label", _CASES, ids=[c[1] for c in _CASES]
)
def test_prompt_references_only_real_tools(
    agent_cls: type[BaseChannelAgent], service: str, label: str
) -> None:
    """Every tool the prompt names must exist among the agent's auth tools."""
    import re

    tool_names = {tool.__name__ for tool in agent_cls()._get_auth_tools()}
    referenced = set(re.findall(r"([a-z_]+)\(", agent_cls.channel_system_prompt))
    missing = referenced - tool_names - {"ask_user_question"}
    assert not missing, f"{label} prompt names unknown tools: {sorted(missing)}"


def test_googlechat_prompt_covers_service_account_option() -> None:
    """Google Chat auth also supports service accounts; the prompt must say so."""
    prompt = GoogleChatAgent.channel_system_prompt
    assert "service_account_json_path" in prompt
    assert "service_account.json" in prompt
