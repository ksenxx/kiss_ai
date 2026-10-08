# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the shared Google sign-in tools.

Drives ``make_google_auth_tools`` through the real Composio SDK against
a real local Composio API emulator (``composio_test_utils``) — no mocks
or patches.  The session conftest points ``KISS_HOME`` at a temporary
directory; an autouse fixture forgets the connection and any saved API
key around every test.
"""

from __future__ import annotations

import json
import stat

import pytest

from kiss.agents.third_party_agents import _composio_google as composio_google
from kiss.agents.third_party_agents._google_workspace_utils import (
    google_auth_prompt,
    make_google_auth_tools,
)
from kiss.agents.third_party_agents.gcal.gcal_sea import _SERVICE, GoogleCalendarAgent
from kiss.tests.agents.third_party_agents.composio_test_utils import (
    API_KEY,
    connect,
    reset_state,
)
from kiss.tests.conftest import IS_WINDOWS

pytestmark = pytest.mark.usefixtures("isolated_kiss_home")


def _tools(calls: list[str]) -> dict:
    """Build the Calendar auth tools recording each on_connected call in *calls*."""
    agent = GoogleCalendarAgent()

    def on_connected() -> bool:
        calls.append("connected")
        return True

    tools = make_google_auth_tools(agent, _SERVICE, "Google Calendar", on_connected)
    return {t.__name__: t for t in tools}


def test_finish_reports_a_failing_backend(composio) -> None:
    """A finished connection whose first API call fails is reported as an error."""
    agent = GoogleCalendarAgent()
    agent._backend._connection_info = "Calendar auth failed: 401"
    tools = {
        t.__name__: t
        for t in make_google_auth_tools(agent, _SERVICE, "Google Calendar", lambda: False)
    }
    started = json.loads(tools["authenticate_google_calendar"]())
    composio.accounts[started["verification_uri"].rsplit("/", 1)[1]] = "ACTIVE"
    result = json.loads(tools["finish_google_calendar_auth"]())
    assert result["ok"] is False
    assert result["error"].endswith("first API call failed: Calendar auth failed: 401")


def test_tool_names_and_docstrings() -> None:
    """The four tools carry per-service names and real docstrings."""
    tools = _tools([])
    assert list(tools) == [
        "check_google_calendar_auth",
        "authenticate_google_calendar",
        "clear_google_calendar_auth",
        "finish_google_calendar_auth",
    ]
    assert all("Google Calendar" in (t.__doc__ or "") for t in tools.values())
    assert "api_key" in (tools["authenticate_google_calendar"].__doc__ or "")


def test_check_without_api_key_asks_for_one(monkeypatch) -> None:
    """With no Composio key configured, check explains how to supply one."""
    monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
    msg = _tools([])["check_google_calendar_auth"]()
    assert "Not authenticated with Google Calendar" in msg
    assert "authenticate_google_calendar(api_key='...')" in msg


def test_check_with_api_key_skips_key_hint(composio) -> None:
    """With a key configured, check only points at authenticate/finish."""
    msg = _tools([])["check_google_calendar_auth"]()
    assert "finish_google_calendar_auth()" in msg
    assert "No Composio API key" not in msg


def test_authenticate_without_any_key_fails(monkeypatch) -> None:
    """authenticate reports the missing key instead of raising."""
    monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
    result = json.loads(_tools([])["authenticate_google_calendar"]())
    assert result["ok"] is False
    assert "No Composio API key" in result["error"]


def test_authenticate_saves_api_key_then_links(composio, monkeypatch) -> None:
    """An api_key argument is saved (0600) and used for the Connect Link."""
    monkeypatch.delenv("COMPOSIO_API_KEY")
    result = json.loads(_tools([])["authenticate_google_calendar"](api_key=f" {API_KEY} "))
    assert result["status"] == "consent_required"
    assert result["verification_uri"].startswith("https://connect.composio.dev/link/")
    assert composio_google.composio_api_key() == API_KEY
    mode = composio_google._api_key_path().stat().st_mode
    assert IS_WINDOWS or stat.S_IMODE(mode) == 0o600  # NTFS has no mode bits


def test_finish_connects_and_calls_on_connected(composio) -> None:
    """finish records an approved connection and wires the backend once."""
    calls: list[str] = []
    tools = _tools(calls)
    started = json.loads(tools["authenticate_google_calendar"]())
    composio.accounts[started["verification_uri"].rsplit("/", 1)[1]] = "ACTIVE"
    result = json.loads(tools["finish_google_calendar_auth"]())
    assert result["ok"] is True
    assert calls == ["connected"]
    assert json.loads(tools["check_google_calendar_auth"]()) == {
        "ok": True,
        "message": "Google Calendar is connected.",
    }


def test_finish_pending_and_failed_do_not_call_on_connected(composio) -> None:
    """Unapproved or failed sign-ins leave the agent unconnected."""
    calls: list[str] = []
    tools = _tools(calls)
    started = json.loads(tools["authenticate_google_calendar"]())
    pending = json.loads(tools["finish_google_calendar_auth"]())
    assert pending["status"] == "pending"
    composio.accounts[started["verification_uri"].rsplit("/", 1)[1]] = "FAILED"
    failed = json.loads(tools["finish_google_calendar_auth"]())
    assert failed["ok"] is False
    assert "FAILED" in failed["error"]
    assert calls == []


def test_finish_without_authenticate_explains(composio) -> None:
    """finish before authenticate says to call authenticate first."""
    result = json.loads(_tools([])["finish_google_calendar_auth"]())
    assert result["ok"] is False
    assert "authenticate_google_calendar()" in result["error"]


def test_clear_deletes_connection(composio) -> None:
    """clear deletes the Composio account and forgets it locally."""
    account = connect(composio, _SERVICE)
    assert _tools([])["clear_google_calendar_auth"]() == "Google Calendar connection cleared."
    assert composio.deleted == [account]
    assert composio_google.connected_account_id(_SERVICE) == ""


def test_sidebar_probe_follows_the_composio_state(composio) -> None:
    """The sidebar probe reflects the Composio connection, not the vault.

    Without a recorded connected account the channel is unauthenticated
    even when a stale ``google_calendar`` vault enrollment exists;
    connecting through Composio makes it authenticated.
    """
    from kiss.agents.third_party_agents import auth_status

    assert not composio_google._state_path(_SERVICE).exists()
    for enrolled in (None, set(), {"google_calendar"}):
        status = auth_status.channel_status("gcal", enrolled)
        assert (status["authenticated"], status["error"]) == (False, "")
    connect(composio, _SERVICE)
    assert auth_status.channel_status("gcal", {"google_calendar"})["authenticated"] is True
    reset_state(_SERVICE)
    assert auth_status.channel_status("gcal", set())["authenticated"] is False


def test_google_auth_prompt_names_the_service_tools() -> None:
    """The shared prompt section names the service's own tools in order."""
    prompt = google_auth_prompt("gmail", "Gmail")
    assert prompt.startswith("\n\n## Gmail Authentication\n")
    assert prompt.index("check_gmail_auth()") < prompt.index("authenticate_gmail(")
    assert prompt.index("authenticate_gmail(") < prompt.index("finish_gmail_auth()")
