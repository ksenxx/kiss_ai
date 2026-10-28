# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sign-in lifecycle shared by the Composio-backed Google channel agents.

Calendar, Docs, Drive and Sheets wire ``make_google_auth_tools`` the
same way; one parametrised test per lifecycle step pins each agent's
wiring (display name, auth tool names, backend tool names, SEA ``tools``
method and CLI usage line) against a real local Composio API emulator
(``composio_test_utils``) -- no mocks or patches.  The per-agent test
modules cover the REST tools behind the unlocked backend.

The service ids and tool names in ``_AGENTS`` are literals, deliberately
not read from the implementation modules, so a wrong ``_SERVICE`` in one
agent fails here instead of being used to generate its own expectation.

State is isolated per test: ``isolated_kiss_home`` points ``KISS_HOME`` at a
fresh temporary directory, so a connection recorded by one test never
leaks into the next.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

import kiss.agents.third_party_agents.gcal.gcal_sea as gcal_mod
import kiss.agents.third_party_agents.gdocs.gdocs_sea as gdocs_mod
import kiss.agents.third_party_agents.gdrive.gdrive_sea as gdrive_mod
import kiss.agents.third_party_agents.gsheets.gsheets_sea as gsheets_mod
from kiss.tests.agents.third_party_agents.composio_test_utils import FakeComposioServer, connect
from kiss.tests.agents.third_party_agents.muse_test_utils import tools_by_name


@dataclass(frozen=True)
class _GoogleAgent:
    """One Composio-backed Google agent and the names its wiring must expose."""

    agent: type
    backend: type
    service: str
    label: str
    auth_tools: list[str]
    backend_tools: frozenset[str]


pytestmark = pytest.mark.usefixtures("isolated_kiss_home")

_AGENTS = [
    _GoogleAgent(
        gcal_mod.GoogleCalendarAgent,
        gcal_mod.GoogleCalendarChannelBackend,
        "google_calendar",
        "Google Calendar",
        [
            "check_google_calendar_auth",
            "authenticate_google_calendar",
            "clear_google_calendar_auth",
            "finish_google_calendar_auth",
        ],
        frozenset(
            {
                "gcal_list_calendars",
                "gcal_list_events",
                "gcal_get_event",
                "gcal_create_event",
                "gcal_update_event",
                "gcal_delete_event",
                "gcal_quick_add",
            }
        ),
    ),
    _GoogleAgent(
        gdocs_mod.GoogleDocsAgent,
        gdocs_mod.GoogleDocsChannelBackend,
        "google_docs",
        "Google Docs",
        [
            "check_google_docs_auth",
            "authenticate_google_docs",
            "clear_google_docs_auth",
            "finish_google_docs_auth",
        ],
        frozenset(
            {
                "gdocs_create_document",
                "gdocs_read_document",
                "gdocs_append_text",
                "gdocs_replace_text",
                "gdocs_insert_text",
                "gdocs_batch_update",
                "gdocs_list_documents",
            }
        ),
    ),
    _GoogleAgent(
        gdrive_mod.GoogleDriveAgent,
        gdrive_mod.GoogleDriveChannelBackend,
        "google_drive",
        "Google Drive",
        [
            "check_google_drive_auth",
            "authenticate_google_drive",
            "clear_google_drive_auth",
            "finish_google_drive_auth",
        ],
        frozenset(
            {
                "gdrive_search_files",
                "gdrive_get_file",
                "gdrive_read_file",
                "gdrive_download_file",
                "gdrive_upload_file",
                "gdrive_create_folder",
                "gdrive_share_file",
                "gdrive_move_file",
                "gdrive_trash_file",
            }
        ),
    ),
    _GoogleAgent(
        gsheets_mod.GoogleSheetsAgent,
        gsheets_mod.GoogleSheetsChannelBackend,
        "google_sheets",
        "Google Sheets",
        [
            "check_google_sheets_auth",
            "authenticate_google_sheets",
            "clear_google_sheets_auth",
            "finish_google_sheets_auth",
        ],
        frozenset(
            {
                "gsheets_create_spreadsheet",
                "gsheets_get_info",
                "gsheets_get_values",
                "gsheets_update_values",
                "gsheets_append_values",
                "gsheets_clear_values",
                "gsheets_add_sheet",
                "gsheets_batch_update",
                "gsheets_list_spreadsheets",
            }
        ),
    ),
]
_IDS = [a.service for a in _AGENTS]


@pytest.mark.parametrize("spec", _AGENTS, ids=_IDS)
def test_check_auth_unauthenticated_explains_setup(
    spec: _GoogleAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """check_*_auth names the authenticate tool and the missing Composio key."""
    monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
    msg = tools_by_name(spec.agent())[f"check_{spec.service}_auth"]()
    assert f"Not authenticated with {spec.label}" in msg
    assert f"authenticate_{spec.service}" in msg
    assert "No Composio API key" in msg


@pytest.mark.parametrize("spec", _AGENTS, ids=_IDS)
def test_connect_link_flow_authenticates_agent(
    spec: _GoogleAgent, composio: FakeComposioServer
) -> None:
    """authenticate -> user approves -> finish unlocks the backend tools."""
    agent = spec.agent()
    assert agent.name == f"{spec.label} Agent"
    tools = tools_by_name(agent)
    started = json.loads(tools[f"authenticate_{spec.service}"]())
    assert started["status"] == "consent_required"
    composio.accounts[started["verification_uri"].rsplit("/", 1)[1]] = "ACTIVE"
    assert json.loads(tools[f"finish_{spec.service}_auth"]())["ok"] is True
    assert agent._is_authenticated() is True
    assert agent._backend.connection_info == f"{spec.label} connected through Composio."
    names = set(tools_by_name(agent))
    assert set(spec.auth_tools) <= names
    assert spec.backend_tools <= names
    # Channel protocol methods are not LLM tools.
    assert "connect" not in names
    assert "poll_messages" not in names
    assert json.loads(tools_by_name(agent)[f"check_{spec.service}_auth"]())["ok"] is True


@pytest.mark.parametrize("spec", _AGENTS, ids=_IDS)
def test_clear_auth_removes_connection_and_relocks_tools(
    spec: _GoogleAgent, composio: FakeComposioServer
) -> None:
    """clear_*_auth deletes the Composio connection and re-locks the backend tools."""
    account = connect(composio, spec.service)
    agent = spec.agent()
    result = tools_by_name(agent)[f"clear_{spec.service}_auth"]()
    assert "cleared" in result.lower()
    assert composio.deleted == [account]
    assert agent._is_authenticated() is False
    # The full ordered list, not the de-duplicated keys: a tool listed
    # twice would hide behind the mapping.
    assert [t.__name__ for t in agent._get_tools()] == spec.auth_tools


@pytest.mark.parametrize("spec", _AGENTS, ids=_IDS)
def test_backend_connect_without_connection_fails(spec: _GoogleAgent) -> None:
    """connect() fails cleanly when no Composio connection is recorded."""
    backend = spec.backend()
    assert backend.connect() is False
    assert "not connected" in backend.connection_info


@pytest.mark.parametrize("spec", _AGENTS, ids=_IDS)
def test_backend_connect_with_composio_connection_succeeds(
    spec: _GoogleAgent, composio: FakeComposioServer
) -> None:
    """connect() succeeds once a Composio connection is recorded."""
    connect(composio, spec.service)
    backend = spec.backend()
    assert backend.connect() is True
    assert backend.connection_info == f"{spec.label} connected through Composio."
