"""Deferred Muse wiring of the channel SEAs: thread safety and first-use behaviour.

Regression tests for the 2026-10-12 review of the lazy Muse wiring
(``tmp/review-1012-lazy-muse.md``).  Every Muse-aware channel agent
builds its backend without touching the daemon (so ``sea lint`` and
``sea docs`` stay side-effect free) and wires the vault surrogate on
the first request.  These tests run the real daemon against recording
HTTP fakes and prove that:

- N threads firing the first request at once all cross the daemon
  boundary with one shared surrogate (``audit.jsonl`` holds one allow
  record per request, the fake server sees the real credential on every
  request and never a surrogate), and a later request from the same
  backend reuses the wiring;
- a wiring failure fails closed on every retry (nothing ever leaves on
  the direct transport without a credential);
- first-use paths that bypassed the wiring (Telegram's direct poll and
  typing calls, Mattermost's typing indicator, Nextcloud's sign-out
  revocation, the status tools) now wire first.
"""

from __future__ import annotations

import base64
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.third_party_agents.muse_auth._common import muse_auth_dir
from kiss.agents.third_party_agents.muse_auth.client import (
    MuseAuthError,
    grant,
    store_credentials,
    vault_has_credentials,
)
from kiss.core.file_lock import exclusive_file_lock
from kiss.tests.agents.third_party_agents.muse_test_utils import (
    auth_tools,
    loopback_policy,
    tools_by_name,
)
from kiss.tests.agents.third_party_agents.recording_http import (
    RecordingHandler,
    RecordingServer,
    serve_recording,
)

_REAL = "real-secret-credential"
_MS_ACCESS = "graph-access-token"
_THREADS = 16
_SETTLE_SECONDS = 0.5
"""Best-effort pause for the threads to reach the wiring guard (see ``_fire_first_requests``)."""
_SERVICES = (
    "homeassistant",
    "brave_search",
    "notion",
    "github",
    "firecrawl",
    "msteams",
    "twitch",
    "discord",
    "zalo",
    "telegram",
    "mattermost",
    "line",
    "bluebubbles",
    "nextcloud",
    "synology",
    "slack",
    "ntfy",
)
# One JSON body every connector under test accepts for its first read.
_BODY: dict[str, Any] = {
    "ok": True,
    "success": True,
    "error": 0,
    "id": "B1",
    "username": "kissbot",
    "data": {"name": "kiss", "oa_id": "1", "displayName": "kiss"},
    "value": [],
    "ocs": {"meta": {"statuscode": 200}, "data": []},
    "web": {"results": []},
}


class _Handler(RecordingHandler):
    """Record every request; answer with ``_BODY`` (or a token for the OAuth endpoint)."""

    server: Any

    def _serve(self) -> None:
        body = self._read_raw()
        path = self.record_request(
            body.decode("utf-8", errors="replace"), headers=dict(self.headers.items())
        )["path"]
        if path.endswith("/oauth2/v2.0/token"):
            payload: dict[str, Any] = {"access_token": _MS_ACCESS, "expires_in": 3600}
        elif path.endswith("/getMe"):
            payload = {"ok": True, "result": {"id": 7, "username": "kissbot", "first_name": "k"}}
        elif path.endswith("/getUpdates"):
            payload = {"ok": True, "result": []}
        else:
            payload = _BODY
        self._json(200, payload)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _serve  # noqa: N815


@pytest.fixture()
def server() -> Any:
    """Run the recording fake on a loopback port."""
    yield from serve_recording(_Handler)


# Every service may reach loopback.
MUSE_POLICY = loopback_policy(*_SERVICES)


def _audit(service: str, verdict: str) -> list[dict[str, Any]]:
    """Return the Sentinel audit records of *service* with *verdict*."""
    path = muse_auth_dir() / "audit.jsonl"
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [r for r in records if r["service"] == service and r["verdict"] == verdict]


def _api_requests(server: RecordingServer) -> list[dict[str, Any]]:
    """Return the recorded connector requests (the daemon's OAuth token exchange excluded)."""
    return [r for r in server.requests if not r["path"].endswith("/oauth2/v2.0/token")]


def _carried(request: dict[str, Any], where: str, key: str, value: str) -> bool:
    """Return whether the recorded *request* carried the real credential."""
    if where == "header":
        headers = {k.lower(): v for k, v in request["headers"].items()}
        return headers.get(key.lower()) == value
    if where == "query":
        return bool(request["query"].get(key) == [value])
    return value in str(request["path"])


def _after_barrier(
    barrier: threading.Barrier, request: Any, agent: Any, results: queue.Queue[Any]
) -> None:
    """Wait for every thread, then issue the request; queue its result or exception."""
    try:
        barrier.wait()
        results.put(request(agent))
    except Exception as e:  # noqa: BLE001 - the test inspects every outcome
        results.put(e)


def _fire_first_requests(request: Any, agent: Any) -> list[Any]:
    """Fire *request* from ``_THREADS`` threads at once as the backend's very first request.

    ``spawn.lock`` is held while the threads start, so the first wiring
    (which spawns the daemon under that lock) cannot finish before the
    lock is released; with unguarded wiring each thread then enrols and
    mints on its own.

    The ``_SETTLE_SECONDS`` pause is best effort, not a guarantee: nothing
    in the backend signals "I have reached the wiring guard", so the pause
    is what gives the other threads time to get there while the lock is
    held.  On a loaded machine a late thread simply arrives after the
    wiring is done and takes the "reuses the wiring" path, so the test
    then checks less (a sequential wiring) but never fails spuriously.

    Args:
        request: ``request(agent)`` issues one read through the backend.
        agent: A freshly built, still unwired channel agent.

    Returns:
        Every thread's result (or the exception it raised).

    Raises:
        AssertionError: A thread did not finish within the deadline.  The
            workers are daemon threads, so a request stuck in the backend
            fails the test instead of hanging it, and cannot keep the
            interpreter alive at exit either (an executor's worker threads
            are joined at shutdown; daemon threads are not).
    """
    barrier = threading.Barrier(_THREADS, timeout=30)
    results: queue.Queue[Any] = queue.Queue()
    with exclusive_file_lock(muse_auth_dir() / "spawn.lock"):
        for _ in range(_THREADS):
            threading.Thread(
                target=_after_barrier, args=(barrier, request, agent, results), daemon=True
            ).start()
        time.sleep(_SETTLE_SECONDS)
    deadline = time.monotonic() + 120
    outcomes: list[Any] = []
    while len(outcomes) < _THREADS:
        try:
            outcomes.append(results.get(timeout=max(0.0, deadline - time.monotonic())))
        except queue.Empty:
            missing = _THREADS - len(outcomes)
            raise AssertionError(f"{missing} of {_THREADS} first requests never returned") from None
    return outcomes


# ------------------------------------------------------------ per-SEA setups


def _homeassistant(server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> Any:
    from kiss.agents.third_party_agents.homeassistant.homeassistant_sea import (
        HomeAssistantAgent,
        _config,
    )

    _config.save({"base_url": server.base_url, "token": _REAL})
    return HomeAssistantAgent()


def _homeassistant_request(agent: Any) -> Any:
    return json.loads(agent._backend._request("GET", "/api/states"))


def _brave(server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> Any:
    from kiss.agents.third_party_agents.brave.brave_sea import BraveSearchAgent, _config

    _config.save({"api_key": _REAL})
    agent = BraveSearchAgent()
    agent._backend._base_url = server.base_url
    return agent


def _brave_request(agent: Any) -> Any:
    return agent._backend._get("/web/search", {"q": "kiss"})


def _notion(server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> Any:
    from kiss.agents.third_party_agents.notion.notion_sea import NotionAgent, _config

    _config.save({"token": _REAL})
    agent = NotionAgent()
    agent._backend._base_url = server.base_url + "/v1"
    return agent


def _notion_request(agent: Any) -> Any:
    data, error = agent._backend._request("GET", "/users/me")
    assert error == "", error
    return data


def _github(server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> Any:
    from kiss.agents.third_party_agents.github.github_sea import GitHubAgent, _config

    _config.save({"token": _REAL})
    agent = GitHubAgent()
    agent._backend._base_url = server.base_url
    return agent


def _github_request(agent: Any) -> Any:
    data, error = agent._backend._api("GET", "/user")
    assert error == "", error
    return data


def _firecrawl(server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> Any:
    from kiss.agents.third_party_agents.firecrawl.firecrawl_sea import FirecrawlAgent, _config

    _config.save({"api_key": _REAL, "base_url": server.base_url})
    return FirecrawlAgent()


def _firecrawl_request(agent: Any) -> Any:
    return agent._backend._request("GET", "/v2/team/credit-usage")


def _msteams(server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> Any:
    from kiss.agents.third_party_agents.msteams.msteams_sea import MSTeamsAgent

    store_credentials(
        "msteams",
        {
            "kind": "oauth2_client_credentials",
            "token_url": f"{server.base_url}/tenant/oauth2/v2.0/token",
            "client_id": "client-id",
            "client_secret": _REAL,
            "token_scope": "https://graph.microsoft.com/.default",
        },
        [],
    )
    agent = MSTeamsAgent()
    agent._backend._graph_base = server.base_url + "/v1.0"
    return agent


def _msteams_request(agent: Any) -> Any:
    result = agent._backend._get("/me")
    assert result.get("ok") is not False, result
    return result


def _twitch(server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> Any:
    from kiss.agents.third_party_agents.twitch.twitch_sea import TwitchAgent, _config

    _config.save({"client_id": "client-id", "access_token": _REAL})
    agent = TwitchAgent()
    agent._backend._helix_base = server.base_url + "/helix"
    return agent


def _twitch_request(agent: Any) -> Any:
    result = agent._backend._get("/users")
    assert result.get("ok") is not False, result
    return result


def _discord(server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> Any:
    from kiss.agents.third_party_agents.discord.discord_sea import DiscordAgent, _config

    _config.save({"bot_token": _REAL})
    agent = DiscordAgent()
    agent._backend._api_base = server.base_url
    return agent


def _discord_request(agent: Any) -> Any:
    result = agent._backend._get("/users/@me")
    assert result.get("id") == "B1", result
    return result


def _zalo(server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> Any:
    from kiss.agents.third_party_agents.zalo.zalo_sea import ZaloAgent, _config

    monkeypatch.setenv("ZALO_API_BASE", server.base_url + "/v2.0/oa")
    _config.save({"access_token": _REAL, "oa_id": "1"})
    return ZaloAgent()


def _zalo_request(agent: Any) -> Any:
    result = json.loads(agent._backend.get_oa_info())
    assert result["ok"] is True, result
    return result


def _telegram(server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> Any:
    from kiss.agents.third_party_agents.telegram.telegram_sea import TelegramAgent

    store_credentials("telegram", {"kind": "path", "token": _REAL}, [])
    agent = TelegramAgent()
    agent._backend._api_base = server.base_url
    return agent


def _telegram_request(agent: Any) -> Any:
    return agent._backend.poll_messages("", "0")


def _mattermost(server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> Any:
    from kiss.agents.third_party_agents.mattermost.mattermost_sea import MattermostAgent, _config

    port = server.server_address[1]
    _config.save({"url": "127.0.0.1", "port": str(port), "scheme": "http", "token": _REAL})
    return MattermostAgent()


def _mattermost_request(agent: Any) -> Any:
    result = json.loads(agent._backend.get_user("me"))
    assert result["ok"] is True, result
    return result


def _line(server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> Any:
    from kiss.agents.third_party_agents.line.line_sea import LineAgent, _config

    monkeypatch.setenv("LINE_API_BASE", server.base_url)
    _config.save({"channel_access_token": _REAL, "channel_secret": "inbound-secret"})
    return LineAgent()


def _line_request(agent: Any) -> Any:
    return agent._backend._api.get_profile("U1")


def _bluebubbles(server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> Any:
    from kiss.agents.third_party_agents.bluebubbles.bluebubbles_sea import (
        BlueBubblesAgent,
        _config,
    )

    _config.save({"server_url": server.base_url, "password": _REAL})
    return BlueBubblesAgent()


def _bluebubbles_request(agent: Any) -> Any:
    # The tools are macOS-only; the channel poll is the portable read.
    return agent._backend.poll_messages("chat1", "")


def _nextcloud(server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> Any:
    from kiss.agents.third_party_agents.nextcloud.nextcloud_sea import NextcloudTalkAgent, _config

    _config.save({"url": server.base_url, "username": "alice", "password": _REAL})
    return NextcloudTalkAgent()


def _nextcloud_request(agent: Any) -> Any:
    result = json.loads(agent._backend.list_rooms())
    assert result["ok"] is True, result
    return result


def _synology(server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> Any:
    from kiss.agents.third_party_agents.synology.synology_sea import SynologyChatAgent, _config

    _config.save({"webhook_url": f"{server.base_url}/webapi/entry.cgi?token={_REAL}"})
    # The incoming webhook is a write: let every thread's POST through.
    grant("synology", "write", "perpetual")
    return SynologyChatAgent()


def _synology_request(agent: Any) -> Any:
    agent._backend.send_message("", "hello")


def _slack(server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> Any:
    from kiss.agents.third_party_agents.slack.slack_sea import SlackAgent, _save_token

    monkeypatch.setenv("KISS_SLACK_BASE_URL", server.base_url)
    _save_token(_REAL, "default")
    agent = SlackAgent()
    agent._backend._api_base_url = server.base_url + "/api/"
    return agent


def _slack_request(agent: Any) -> Any:
    client = agent._backend._client
    assert client is not None
    return client.auth_test().data


_BASIC = "Basic " + base64.b64encode(f"alice:{_REAL}".encode()).decode()
_CASES = [
    pytest.param("homeassistant", _homeassistant, _homeassistant_request,
                 ("header", "Authorization", f"Bearer {_REAL}"), id="homeassistant"),
    pytest.param("brave_search", _brave, _brave_request,
                 ("header", "X-Subscription-Token", _REAL), id="brave"),
    pytest.param("notion", _notion, _notion_request,
                 ("header", "Authorization", f"Bearer {_REAL}"), id="notion"),
    pytest.param("github", _github, _github_request,
                 ("header", "Authorization", f"Bearer {_REAL}"), id="github"),
    pytest.param("firecrawl", _firecrawl, _firecrawl_request,
                 ("header", "Authorization", f"Bearer {_REAL}"), id="firecrawl"),
    pytest.param("msteams", _msteams, _msteams_request,
                 ("header", "Authorization", f"Bearer {_MS_ACCESS}"), id="msteams"),
    pytest.param("twitch", _twitch, _twitch_request,
                 ("header", "Authorization", f"Bearer {_REAL}"), id="twitch"),
    pytest.param("discord", _discord, _discord_request,
                 ("header", "Authorization", f"Bot {_REAL}"), id="discord"),
    pytest.param("zalo", _zalo, _zalo_request, ("header", "access_token", _REAL), id="zalo"),
    pytest.param("telegram", _telegram, _telegram_request,
                 ("path", "", f"/bot{_REAL}/getUpdates"), id="telegram"),
    pytest.param("mattermost", _mattermost, _mattermost_request,
                 ("header", "Authorization", f"Bearer {_REAL}"), id="mattermost"),
    pytest.param("line", _line, _line_request,
                 ("header", "Authorization", f"Bearer {_REAL}"), id="line"),
    pytest.param("bluebubbles", _bluebubbles, _bluebubbles_request,
                 ("query", "password", _REAL), id="bluebubbles"),
    pytest.param("nextcloud", _nextcloud, _nextcloud_request,
                 ("header", "Authorization", _BASIC), id="nextcloud"),
    pytest.param("synology", _synology, _synology_request,
                 ("query", "token", _REAL), id="synology"),
    pytest.param("slack", _slack, _slack_request,
                 ("header", "Authorization", f"Bearer {_REAL}"), id="slack"),
]  # fmt: skip


@pytest.mark.parametrize(("service", "build", "issue", "evidence"), _CASES)
def test_concurrent_first_requests_share_one_wiring(
    muse_env: Path,
    server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    service: str,
    build: Any,
    issue: Any,
    evidence: tuple[str, str, str],
) -> None:
    """N simultaneous first requests all cross the boundary with the one surrogate.

    A fresh agent exposes its tools without a daemon; the first request
    wires the surrogate once, under a lock, and every concurrent request
    waits for it.  Each request then shows up as one Sentinel allow
    record and reaches the fake server with the real credential; a
    second enrolment (an unguarded re-wiring) would invalidate the
    surrogate the other threads hold and fail their requests, so all N
    succeeding proves the daemon was wired once.
    """
    agent = build(server, monkeypatch)
    assert agent._is_authenticated() is True
    assert muse_auth_dir().joinpath("audit.jsonl").exists() is False
    assert server.requests == []

    results = _fire_first_requests(issue, agent)

    failures = [r for r in results if isinstance(r, Exception)]
    assert failures == [], failures
    allowed = _audit(service, "allow")
    assert len(allowed) == _THREADS, allowed
    requests = _api_requests(server)
    assert len(requests) == _THREADS
    where, key, value = evidence
    missing = [r for r in requests if not _carried(r, where, key, value)]
    assert missing == [], missing
    assert "muse-sgt." not in json.dumps(server.requests)
    assert vault_has_credentials(service)
    # The daemon exchanged the MS Teams client credentials once for all N.
    assert len(server.requests) - len(requests) == (1 if service == "msteams" else 0)

    # A later request (a cached tool retry) reuses the wiring: one more
    # audited request, no new enrolment.
    issue(agent)
    assert len(_audit(service, "allow")) == _THREADS + 1
    assert len(_api_requests(server)) == _THREADS + 1
    assert _carried(server.requests[-1], where, key, value)


def test_ntfy_concurrent_first_publishes_all_cross_the_boundary(
    muse_env: Path, server: RecordingServer
) -> None:
    """ntfy's real token, loaded from config at construction, never leaves directly.

    The policy asks before writes, so an ungranted publish is denied at
    the daemon: N concurrent first publishes must all be denied there
    (N "ask" records, zero requests at the server).  A publish that went
    out on the direct transport with the configured token would bypass
    Sentinel and reach the server.
    """
    from kiss.agents.third_party_agents.ntfy.ntfy_sea import NtfyAgent, _config

    _config.save({"topic": "t1", "server": server.base_url, "token": _REAL})
    agent = NtfyAgent()
    assert agent._backend._token == _REAL and agent._backend._lazy_muse

    results = _fire_first_requests(_ntfy_publish, agent)

    assert all(isinstance(r, RuntimeError) and "403" in str(r) for r in results), results
    assert server.requests == []
    assert len(_audit("ntfy", "ask")) == _THREADS
    backend = agent._backend
    assert backend._muse and backend._token.startswith("muse-sgt.ntfy.")
    assert "token" not in json.loads(_config.path.read_text())

    # A granted publish from the same (cached) backend reaches the server
    # with the real token swapped in at the boundary.
    grant("ntfy", "write", "once")
    _ntfy_publish(agent)
    assert len(server.requests) == 1
    assert _carried(server.requests[0], "header", "Authorization", f"Bearer {_REAL}")
    assert "muse-sgt." not in json.dumps(server.requests)


def _ntfy_publish(agent: Any) -> None:
    agent._backend.send_message("", "hello")


def test_ntfy_failed_wiring_fails_closed_on_every_retry(
    muse_env: Path, server: RecordingServer
) -> None:
    """A rejected token leaves ntfy tokenless and every later request refused.

    The first publish runs the deferred wiring, which the daemon
    rejects; the retry (and a poll) must raise the remembered error
    instead of going out tokenless on the direct transport, until a
    new token is configured through the auth tool.
    """
    from kiss.agents.third_party_agents.ntfy.ntfy_sea import NtfyAgent, _config

    _config.save({"topic": "t1", "server": server.base_url, "token": "bad\ntoken"})
    agent = NtfyAgent()
    tools = tools_by_name(agent)
    assert "publish_notification" in tools

    first = json.loads(tools["publish_notification"]("x"))
    assert first["ok"] is False and "wiring failed" in first["error"]
    retry = json.loads(tools["publish_notification"]("x"))
    assert retry["ok"] is False and "wiring failed" in retry["error"]
    with pytest.raises(MuseAuthError, match="wiring failed"):
        agent._backend._fetch_messages("t1", "", 10)
    assert server.requests == []
    assert agent._backend._token == "" and not agent._backend._muse
    assert agent._is_authenticated() is False
    assert json.loads(auth_tools(agent)["check_ntfy_auth"]())["ok"] is False

    # Configuring a good token through the auth tool recovers the backend.
    fixed = json.loads(auth_tools(agent)["authenticate_ntfy"]("t1", server.base_url, _REAL))
    assert fixed["ok"] is True, fixed
    grant("ntfy", "write", "once")
    sent = json.loads(tools["publish_notification"]("x"))
    assert sent["ok"] is True, sent
    assert len(server.requests) == 1
    assert _carried(server.requests[0], "header", "Authorization", f"Bearer {_REAL}")


def test_discord_failed_wiring_fails_closed_on_every_retry(
    muse_env: Path, server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A property-wired backend refuses every request after a rejected enrolment."""
    from kiss.agents.third_party_agents.discord.discord_sea import DiscordAgent, _config

    _config.save({"bot_token": "bad\ntoken"})
    agent = DiscordAgent()
    agent._backend._api_base = server.base_url
    tools = tools_by_name(agent)

    for _ in range(2):
        failed = json.loads(tools["list_guilds"]())
        assert failed["ok"] is False and "invalid credential token" in failed["error"]
    assert server.requests == []
    assert agent._backend._token == "" and not agent._backend._muse
    assert not vault_has_credentials("discord")


def test_telegram_fresh_agent_poll_and_typing_cross_the_boundary(
    muse_env: Path, server: RecordingServer
) -> None:
    """The first poll of a fresh agent wires before binding the direct transport."""
    from kiss.agents.third_party_agents.telegram.telegram_sea import TelegramAgent

    store_credentials("telegram", {"kind": "path", "token": _REAL}, [])
    agent = TelegramAgent()
    agent._backend._api_base = server.base_url
    assert agent._backend._bot_obj is None

    assert agent._backend.poll_messages("", "0") == ([], "0")
    assert len(server.requests) == 1
    assert server.requests[0]["path"] == f"/bot{_REAL}/getUpdates"
    assert len(_audit("telegram", "allow")) == 1

    fresh = TelegramAgent()
    fresh._backend._api_base = server.base_url
    fresh._backend.send_typing("12")
    assert server.requests[-1]["path"] == f"/bot{_REAL}/sendChatAction"
    assert "muse-sgt." not in json.dumps(server.requests)

    # Without any credential the fresh agent's poll stays silent and
    # sends nothing on the direct transport.
    from kiss.agents.third_party_agents.muse_auth.client import clear_credentials

    clear_credentials("telegram")
    unwired = TelegramAgent()
    unwired._backend._api_base = server.base_url
    assert unwired._backend.poll_messages("", "0") == ([], "0")
    unwired._backend.send_typing("12")
    assert len(server.requests) == 2


def test_mattermost_fresh_agent_first_typing_indicator_is_sent(
    muse_env: Path, server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The typing indicator of a fresh agent wires first instead of skipping."""
    agent = _mattermost(server, monkeypatch)
    assert agent._backend._driver_obj is None

    agent._backend.send_typing("C1", "root")

    assert len(server.requests) == 1
    assert server.requests[0]["path"] == "/api/v4/users/me/typing"
    assert json.loads(server.requests[0]["body"]) == {"channel_id": "C1", "parent_id": "root"}
    assert _carried(server.requests[0], "header", "Authorization", f"Bearer {_REAL}")


def test_nextcloud_fresh_agent_sign_out_revokes_remote_app_password(
    muse_env: Path, server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clearing a Login Flow password on a fresh agent revokes it on the server."""
    from kiss.agents.third_party_agents.nextcloud.nextcloud_sea import _config

    agent = _nextcloud(server, monkeypatch)
    meta = json.loads(_config.path.read_text())
    meta["login_flow"] = "true"
    _config.path.write_text(json.dumps(meta))
    grant("nextcloud", "write", "once")

    result = auth_tools(agent)["clear_nextcloud_auth"]()

    assert "revoked" in result.lower(), result
    assert [r["method"] for r in server.requests] == ["DELETE"]
    assert server.requests[0]["path"] == "/ocs/v2.php/core/apppassword"
    assert _carried(server.requests[0], "header", "Authorization", _BASIC)
    assert not vault_has_credentials("nextcloud")
    assert not _config.path.exists()


def test_status_tools_report_a_configured_fresh_agent(
    muse_env: Path, server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Status tools of fresh agents report the stored configuration, not a sign-in prompt."""
    checks: list[tuple[Any, str, dict[str, Any]]] = [
        (_discord, "check_discord_auth", {"username": "kissbot"}),
        (_msteams, "check_msteams_auth", {"ok": True}),
        (_zalo, "check_zalo_auth", {"name": "kiss"}),
        (_homeassistant, "check_homeassistant_auth", {"base_url": server.base_url}),
        (_firecrawl, "check_firecrawl_auth", {"base_url": server.base_url}),
        (_telegram, "check_telegram_auth", {"username": "kissbot"}),
        (_mattermost, "check_mattermost_auth", {"ok": True}),
    ]
    for build, name, expected in checks:
        agent = build(server, monkeypatch)
        before = len(server.requests)
        status = json.loads(auth_tools(agent)[name]())
        assert status["ok"] is True, (name, status)
        for key, value in expected.items():
            assert status[key] == value, (name, status)
        if name in ("check_homeassistant_auth", "check_firecrawl_auth"):
            # The endpoint comes from config metadata: no request needed.
            assert len(server.requests) == before
        else:
            assert len(server.requests) > before
            assert "muse-sgt." not in json.dumps(server.requests[before:])
