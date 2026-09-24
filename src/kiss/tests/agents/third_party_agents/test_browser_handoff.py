# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the default-browser authentication hand-off.

Every connector's sign-in ends on a page only the user may complete.
The agents now open that page in the user's default browser when the
process has one and always return the URL (and code) for a manual
sign-in.  These tests run the real code paths with a scripted browser
installed through ``$BROWSER`` (the ``webbrowser`` convention) that
records every URL it is asked to open, so nothing is mocked: the
launcher really forks the browser process, the consent sessions really
start, the Google loopback server really listens.

Not covered here, and why:

* ``sys.platform == "darwin"`` / ``"win32"`` openers (``open``,
  ``os.startfile``): platform-specific; only reachable on those hosts.
* Opening with no ``$BROWSER`` set on a machine with a display: it would
  launch the real ``xdg-open`` of the test host.
"""

from __future__ import annotations

import json
import re
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.third_party_agents import _browser_handoff
from kiss.agents.third_party_agents._browser_handoff import (
    _launch_commands,
    browser_handoff_note,
    open_in_default_browser,
    portal_handoff,
)
from kiss.agents.third_party_agents._device_auth import (
    ConsentSession,
    consent_instructions,
    consent_required,
)
from kiss.agents.third_party_agents._google_workspace_utils import (
    CLOUD_CONSOLE_URL,
    google_consent_steps,
    remote_oauth_instructions,
    start_google_consent,
)
from kiss.agents.third_party_agents.brave_sea import BraveSearchAgent
from kiss.agents.third_party_agents.discord_sea import DiscordAgent
from kiss.agents.third_party_agents.gcal_sea import GoogleCalendarAgent
from kiss.agents.third_party_agents.gmail_sea import GmailAgent
from kiss.agents.third_party_agents.googlechat_sea import GoogleChatAgent
from kiss.agents.third_party_agents.signal_sea import SignalAgent
from kiss.agents.third_party_agents.slack_sea import SlackAgent
from kiss.agents.third_party_agents.telegram_sea import TelegramAgent
from kiss.agents.third_party_agents.whatsapp_sea import _qr_handoff
from kiss.tests.agents.third_party_agents.test_muse_connect_flows import _FAKE_SIGNAL_CLI
from kiss.tests.conftest import IS_WINDOWS, install_cli_script

_FAKE_BROWSER = """#!/usr/bin/env python3
# Scripted stand-in for the user's browser: records the URLs it is asked
# to open (one JSON line per launch) and exits like a real opener would.
import json
import os
import sys
import time

here = os.path.dirname(os.path.abspath(sys.argv[0]))
with open(os.path.join(here, "browser-log"), "a") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\\n")
if os.path.exists(os.path.join(here, "refuse")):
    sys.exit(1)
if os.path.exists(os.path.join(here, "linger")):
    time.sleep(5)
"""

_DUMMY_CLIENT_SECRETS = {
    "installed": {
        "client_id": "test-client-id.apps.googleusercontent.com",
        "client_secret": "test-secret",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost"],
    }
}


@pytest.fixture()
def fake_browser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install the recording browser as ``$BROWSER`` on a non-headless host.

    Returns:
        The directory holding the script, its ``browser-log`` and the
        ``refuse`` / ``linger`` behaviour markers.
    """
    home = tmp_path / "browser"
    home.mkdir()
    script = home / "browser"
    install_cli_script(script, _FAKE_BROWSER)
    launcher = script.with_name("browser.cmd") if IS_WINDOWS else script
    monkeypatch.setenv("BROWSER", str(launcher))
    monkeypatch.setenv("KISS_HEADLESS", "0")
    return home


def _opened(home: Path) -> list[list[str]]:
    """Return the argv of every browser launch recorded so far."""
    log = home / "browser-log"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def _unique_url(path: str = "signin") -> str:
    """A URL no earlier test opened, so the reopen guard cannot interfere."""
    return f"https://example.test/{path}/{uuid.uuid4()}"


def _auth_tools(agent: Any) -> dict[str, Any]:
    return {tool.__name__: tool for tool in agent._get_auth_tools()}


# --------------------------------------------------------------------------
# open_in_default_browser
# --------------------------------------------------------------------------


def test_opens_url_with_browser_env_and_reports_success(fake_browser: Path) -> None:
    """``$BROWSER`` is run with the URL appended; a clean exit means opened."""
    url = _unique_url()
    assert open_in_default_browser(url) is True
    assert _opened(fake_browser) == [[url]]


def test_browser_env_placeholder_and_fallback_list_are_honoured(
    fake_browser: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``%s`` receives the URL; ``os.pathsep`` entries are tried in order."""
    import os

    launcher = os.environ["BROWSER"]
    monkeypatch.setenv("BROWSER", f'/no/such/browser{os.pathsep}"{launcher}" --new-tab=%s')
    url = _unique_url("placeholder")
    assert _launch_commands(url) == [["/no/such/browser", url], [launcher, f"--new-tab={url}"]]
    assert open_in_default_browser(url) is True
    assert _opened(fake_browser) == [[f"--new-tab={url}"]]


def test_malformed_browser_env_or_url_never_raises(
    fake_browser: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken ``$BROWSER`` or URL counts as not opened instead of failing the tool."""
    assert open_in_default_browser("http://[") is False
    monkeypatch.setenv("BROWSER", '"')
    assert _launch_commands("https://x.test/") == []
    assert open_in_default_browser(_unique_url("malformed")) is False
    monkeypatch.setenv("BROWSER", " ")
    assert _launch_commands("https://x.test/")[0][0] in ("open", "xdg-open")
    import os

    monkeypatch.setenv("BROWSER", f"{os.pathsep}/bin/x{os.pathsep}")
    assert _launch_commands("https://x.test/") == [["/bin/x", "https://x.test/"]]
    assert _opened(fake_browser) == []


def test_concurrent_callers_open_one_tab(fake_browser: Path) -> None:
    """Two threads asking for the same page at once launch the browser once."""
    import threading

    (fake_browser / "linger").touch()
    url = _unique_url("concurrent")
    results: list[bool] = []
    threads = [
        threading.Thread(target=lambda: results.append(open_in_default_browser(url)))
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive(), "open_in_default_browser did not return"
    assert results == [True, True]
    assert _opened(fake_browser) == [[url]]


def test_headless_and_disallowed_schemes_never_launch(
    fake_browser: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No launch when headless, and never for schemes that are not a web page."""
    assert open_in_default_browser("sgnl://linkdevice?uuid=1") is False
    assert open_in_default_browser("javascript:alert(1)") is False
    assert open_in_default_browser("not a url") is False
    monkeypatch.setenv("KISS_HEADLESS", "1")
    assert open_in_default_browser(_unique_url("headless")) is False
    assert _opened(fake_browser) == []


def test_missing_or_failing_opener_reports_not_opened(
    fake_browser: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unknown command or an opener that exits non-zero yields False."""
    url = _unique_url("failing")
    (fake_browser / "refuse").touch()
    assert open_in_default_browser(url) is False
    assert _opened(fake_browser) == [[url]]
    # A failed attempt is not remembered: the next attempt launches again.
    (fake_browser / "refuse").unlink()
    assert open_in_default_browser(url) is True
    assert _opened(fake_browser) == [[url], [url]]
    monkeypatch.setenv("BROWSER", str(fake_browser / "does-not-exist"))
    assert open_in_default_browser(_unique_url("missing")) is False


def test_lingering_opener_counts_as_opened(fake_browser: Path) -> None:
    """A ``$BROWSER`` that is the browser itself outlives the grace period."""
    (fake_browser / "linger").touch()
    url = _unique_url("linger")
    assert open_in_default_browser(url) is True
    assert _opened(fake_browser) == [[url]]


def test_recently_opened_url_is_not_opened_twice(fake_browser: Path) -> None:
    """Repeated check_*_auth() calls reuse the tab opened moments ago."""
    url = _unique_url("twice")
    assert open_in_default_browser(url) is True
    assert open_in_default_browser(url) is True
    assert _opened(fake_browser) == [[url]]
    # A different URL is a different page and opens.
    other = _unique_url("twice")
    assert open_in_default_browser(other) is True
    assert _opened(fake_browser) == [[url], [other]]


def test_default_command_without_browser_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``$BROWSER`` the platform opener is used (xdg-open here)."""
    monkeypatch.delenv("BROWSER", raising=False)
    import sys

    expected = "open" if sys.platform == "darwin" else "xdg-open"
    assert _launch_commands("https://x.test/") == [[expected, "https://x.test/"]]


def test_handoff_notes_always_carry_the_url() -> None:
    """Both outcomes tell the agent to show the URL via ask_user_question()."""
    url = "https://portal.test/apps"
    opened = browser_handoff_note(url, True)
    assert opened.startswith(url) and "default browser" in opened
    assert "ask_user_question()" in opened and "if no browser window appeared" in opened
    closed = browser_handoff_note(url, False)
    assert url in closed and "OWN browser" in closed and "ask_user_question()" in closed
    assert "No browser could be opened" in closed


def test_portal_handoff_opens_and_describes(fake_browser: Path) -> None:
    """portal_handoff() opens the portal and returns the opened-note."""
    url = _unique_url("portal")
    assert portal_handoff(url).startswith(f"{url} has just been opened")
    assert _opened(fake_browser) == [[url]]


# --------------------------------------------------------------------------
# Connect-style consent sessions (device flow / Nextcloud) and Signal
# --------------------------------------------------------------------------


def _session(uri: str, code: str = "", prefilled: bool = False) -> ConsentSession:
    session = ConsentSession("demo", lifetime=600.0, interval=5.0)
    session.verification_uri = uri
    session.user_code = code
    session.code_prefilled = prefilled
    return session


def test_consent_required_opens_verification_page_and_keeps_code(fake_browser: Path) -> None:
    """The device-code page opens in the browser; URL and code are still returned."""
    uri = _unique_url("device")
    answer = consent_required("demo", "Demo", _session(uri, "WDJB-MJHT"))
    assert answer["browser_opened"] is True
    assert answer["verification_uri"] == uri and answer["user_code"] == "WDJB-MJHT"
    text = answer["instructions"]
    assert "has just been opened in the user's default browser" in text
    assert "ask_user_question()" in text and uri in text and "OWN browser" in text
    assert "enter the code WDJB-MJHT" in text and "finish_demo_auth()" in text
    assert "built-in browser" in text and "password or 2FA code" in text
    assert _opened(fake_browser) == [[uri]]


def test_consent_required_headless_still_hands_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a browser the answer says so and hands the URL + code to the user."""
    monkeypatch.setenv("KISS_HEADLESS", "1")
    uri = _unique_url("device-headless")
    answer = consent_required("demo", "Demo", _session(uri, "CODE-1", prefilled=True))
    assert answer["browser_opened"] is False
    text = answer["instructions"]
    assert "No browser could be opened from this machine" in text
    assert uri in text and "confirm the code shown is CODE-1" in text
    # No code at all (Nextcloud Login Flow v2): no code step.
    plain = consent_instructions("demo", "Demo", _session(uri))
    assert "code" not in plain.split("Steps:")[1].split("approve")[0]


def test_signal_link_opens_black_on_white_qr_page(
    isolated_kiss_home: Path, fake_browser: Path, tmp_path: Path
) -> None:
    """Signal opens its QR page (a file:// URL it wrote) in the user's browser."""
    cli = tmp_path / "signal-cli"
    install_cli_script(cli, _FAKE_SIGNAL_CLI)
    tools = _auth_tools(SignalAgent())
    try:
        started = json.loads(tools["authenticate_signal"](signal_cli_path=str(cli)))
        assert started["status"] == "consent_required"
        assert started["browser_opened"] is True
        page = Path(started["qr_page"])
        assert _opened(fake_browser) == [[page.as_uri()]]
        assert f"The QR page {page} has just been opened" in started["instructions"]
        assert started["qr_text"] in started["instructions"]
    finally:
        ConsentSession.cancel_active("signal")


def test_whatsapp_qr_handoff_both_outcomes(
    fake_browser: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The WhatsApp QR page opens when possible; otherwise show_browser() is the fallback."""
    page = tmp_path / "qr.html"
    page.write_text("<html></html>")
    opened = _qr_handoff(page)
    assert opened["browser_opened"] is True and opened["qr_page"] == str(page)
    assert "has just been opened" in opened["message"]
    assert "wait_for_whatsapp_pairing()" in opened["message"]
    assert _opened(fake_browser) == [[page.as_uri()]]
    monkeypatch.setenv("KISS_HEADLESS", "1")
    closed = _qr_handoff(page)
    assert closed["browser_opened"] is False
    assert "show_browser()" in closed["message"] and f"file://{page}" in closed["message"]


# --------------------------------------------------------------------------
# Google consent (loopback server) and Cloud Console setup
# --------------------------------------------------------------------------


def _write_client_secrets(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_DUMMY_CLIENT_SECRETS))


def _forge_redirect(auth_url: str) -> None:
    """Play a browser that lands on the loopback server with a bogus code."""
    match = re.search(r"redirect_uri=http%3A%2F%2Flocalhost%3A(\d+)", auth_url)
    assert match is not None
    with urllib.request.urlopen(
        f"http://localhost:{match.group(1)}/?state=bogus&code=bogus", timeout=10
    ) as resp:
        assert resp.status == 200


def test_google_consent_opens_auth_url_and_finishes_locally(
    isolated_kiss_home: Path, fake_browser: Path
) -> None:
    """authenticate_google_calendar() opens Google's consent page and returns at once."""
    from kiss.agents.third_party_agents._google_workspace_utils import credentials_path

    _write_client_secrets(credentials_path("google_calendar"))
    tools = _auth_tools(GoogleCalendarAgent())
    started = json.loads(tools["authenticate_google_calendar"]())
    assert started["status"] == "consent_required" and started["browser_opened"] is True
    assert started["auth_url"].startswith("https://accounts.google.com/o/oauth2/auth")
    assert _opened(fake_browser) == [[started["auth_url"]]]
    text = started["instructions"]
    assert "has just been opened in the user's default browser" in text
    assert "completes by itself" in text and "Only if a URL was pasted back" in text
    assert started["auth_url"] in text and "finish_google_calendar_auth()" in text
    _forge_redirect(started["auth_url"])
    finished = json.loads(tools["finish_google_calendar_auth"]())
    assert finished["ok"] is False and "state" in finished["error"].lower()


def test_google_consent_headless_and_missing_or_broken_credentials(
    isolated_kiss_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Headless: the URL is handed over for paste-back; bad/missing credentials.json."""
    from kiss.agents.third_party_agents._google_workspace_utils import credentials_path

    monkeypatch.setenv("KISS_HEADLESS", "1")
    assert start_google_consent("google_docs", "Google Docs", ["scope"]) is None
    _write_client_secrets(credentials_path("google_docs"))
    answer = start_google_consent("google_docs", "Google Docs", ["scope"])
    assert answer is not None
    started = json.loads(answer)
    assert started["browser_opened"] is False
    assert "No browser could be opened" in started["instructions"]
    assert "paste back" in started["instructions"]
    _forge_redirect(started["auth_url"])
    from kiss.agents.third_party_agents._google_workspace_utils import RemoteOAuthSession

    creds, status = RemoteOAuthSession.finish("google_docs", ["scope"])
    assert creds is None and status != "pending"
    credentials_path("google_docs").write_text("not json")
    answer = start_google_consent("google_docs", "Google Docs", ["scope"])
    assert answer is not None
    broken = json.loads(answer)
    assert broken["ok"] is False and "may be malformed" in broken["error"]
    # The instruction builder's opened branch, standalone.
    note = remote_oauth_instructions("google_docs", "Google Docs", "https://a.test/", True)
    assert "completes by itself" in note and "https://a.test/" in note
    assert "ALWAYS call ask_user_question() with the full auth_url" in google_consent_steps("g")


def test_gmail_and_googlechat_use_the_shared_consent(
    isolated_kiss_home: Path, fake_browser: Path
) -> None:
    """The two hand-written Google agents open the consent page the same way."""
    from kiss.agents.third_party_agents import gmail_sea, googlechat_sea

    _write_client_secrets(gmail_sea._credentials_path())
    gmail_tools = _auth_tools(GmailAgent())
    started = json.loads(gmail_tools["authenticate_gmail"]())
    assert started["status"] == "consent_required" and started["browser_opened"] is True
    _forge_redirect(started["auth_url"])
    assert json.loads(gmail_tools["finish_gmail_auth"]())["ok"] is False

    _write_client_secrets(googlechat_sea._credentials_path())
    chat_tools = _auth_tools(GoogleChatAgent())
    started = json.loads(chat_tools["authenticate_googlechat"]())
    assert started["status"] == "consent_required" and started["browser_opened"] is True
    _forge_redirect(started["auth_url"])
    assert json.loads(chat_tools["finish_googlechat_auth"]())["ok"] is False
    assert len(_opened(fake_browser)) == 2


def test_cloud_console_setup_tools_open_the_console(
    isolated_kiss_home: Path, fake_browser: Path
) -> None:
    """start_*_browser_setup() and the Google Chat check open Cloud Console for the user."""
    gmail_tools = _auth_tools(GmailAgent())
    missing = gmail_tools["authenticate_gmail"]()
    assert "credentials.json not found" in missing and "start_gmail_browser_setup()" in missing
    setup = gmail_tools["start_gmail_browser_setup"]()
    assert f"{CLOUD_CONSOLE_URL} has just been opened" in setup
    assert "Desktop app" in setup and "authenticate_gmail()" in setup
    assert "autonomously" not in setup and "go_to_url" not in setup
    generic = _auth_tools(GoogleCalendarAgent())["start_google_calendar_browser_setup"]()
    assert "has just been opened" in generic and "authenticate_google_calendar()" in generic
    chat_tools = _auth_tools(GoogleChatAgent())
    check = chat_tools["check_googlechat_auth"]()
    assert "Not authenticated with Google Chat" in check and "has just been opened" in check
    # One console tab for all three: the reopen guard collapses them.
    assert _opened(fake_browser) == [[CLOUD_CONSOLE_URL]]


# --------------------------------------------------------------------------
# Token / API-key portals
# --------------------------------------------------------------------------


def test_token_agents_open_their_developer_portals(
    isolated_kiss_home: Path, fake_browser: Path
) -> None:
    """check_*_auth() / start_*_browser_auth() open the portal and keep the paste-back."""
    slack = SlackAgent()
    slack._backend._client = None
    slack_tools = _auth_tools(slack)
    check = slack_tools["check_slack_auth"]()
    assert "Not authenticated with Slack" in check and "start_slack_browser_auth()" in check
    assert "user's default browser" in check and "paste back" in check
    start = slack_tools["start_slack_browser_auth"]()
    assert "https://api.slack.com/apps has just been opened" in start
    assert "OWN browser" in start and "paste back" in start and "go_to_url" not in start

    discord = DiscordAgent()
    discord._backend._bot_token = ""
    start = _auth_tools(discord)["start_discord_browser_auth"]()
    assert "https://discord.com/developers/applications has just been opened" in start
    assert "Reset Token" in start and "paste back" in start

    brave = BraveSearchAgent()
    brave._backend._api_key = ""
    check = _auth_tools(brave)["check_brave_search_auth"]()
    assert "Not configured for Brave Search" in check
    assert "https://api-dashboard.search.brave.com/ has just been opened" in check

    telegram = TelegramAgent()
    telegram._backend._bot = None
    check = _auth_tools(telegram)["check_telegram_auth"]()
    assert "@BotFather" in check and "https://t.me/BotFather has just been opened" in check

    assert [launch[0] for launch in _opened(fake_browser)] == [
        "https://api.slack.com/apps",
        "https://discord.com/developers/applications",
        "https://api-dashboard.search.brave.com/",
        "https://t.me/BotFather",
    ]


def test_device_flow_prerequisite_portals_open(
    isolated_kiss_home: Path, fake_browser: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an OAuth app ID, the registration portal opens for the user."""
    from kiss.agents.third_party_agents.github_sea import GitHubAgent
    from kiss.agents.third_party_agents.msteams_sea import MSTeamsAgent
    from kiss.agents.third_party_agents.twitch_sea import TwitchAgent

    monkeypatch.delenv("KISS_GITHUB_CLIENT_ID", raising=False)
    missing = json.loads(_auth_tools(GitHubAgent())["authenticate_github"]())
    assert missing["ok"] is False
    assert "https://github.com/settings/applications/new has just been opened" in missing["error"]
    twitch = _auth_tools(TwitchAgent())["authenticate_twitch"]("")
    assert twitch.startswith("client_id cannot be empty.")
    assert "https://dev.twitch.tv/console/apps has just been opened" in twitch
    teams = _auth_tools(MSTeamsAgent())["authenticate_msteams"]("", "")
    assert teams.startswith("tenant_id cannot be empty.")
    assert "portal.azure.com" in teams and "has just been opened" in teams
    assert len(_opened(fake_browser)) == 3


def test_prompts_describe_the_default_browser_hand_off() -> None:
    """The channel prompts tell the agent the page is opened for the user and to show the URL."""
    for agent_cls in (SlackAgent, DiscordAgent):
        prompt = agent_cls.channel_system_prompt
        assert "user's default browser" in prompt and "if it did not open by itself" in prompt
        assert "Do not drive the portal" in prompt
    from kiss.agents.third_party_agents.github_sea import GitHubAgent

    prompt = GitHubAgent.channel_system_prompt
    assert "tries by itself to open that URL in the user's default browser" in prompt
    assert "'browser_opened'" in prompt and "ALWAYS call ask_user_question()" in prompt
    assert "'browser_opened'" in SignalAgent.channel_system_prompt or (
        "default browser" in SignalAgent.channel_system_prompt
    )
    assert _browser_handoff.__doc__ is not None
