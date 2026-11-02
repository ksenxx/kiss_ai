"""Two concurrent refresh grants with one consumed refresh token: exactly one is accepted.

``SlackApiServer`` answers each request on its own thread.  Before the
fake serialized its grants, handler A could validate the old refresh
token, handler B validate the same token, and both rotate it: two ``ok``
replies for one one-use token, which the sequential contract the Slack
sign-in tests rely on (``test_slack_pkce_auth.py::
test_second_client_reuses_token_rotated_by_the_first``) forbids.

The race window is scheduled, not timed: a ``threading.settrace`` hook on
the handler threads holds the first handler that reaches ``_issue`` until
the second one overlaps it, that is until it has entered ``answer`` while
the fake's grant lock is held (it can then validate only after the first
handler has rotated the token) or until it has reached ``_issue`` itself
(a fake without the lock lets it validate the token the first handler is
about to rotate, and both redeem).  Merely entering ``answer`` would not
do: a lockless fake could still rotate the token before the second
handler validated it, and the test would pass against the very race.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

from kiss.tests.agents.third_party_agents.recording_http import recording_server
from kiss.tests.agents.third_party_agents.slack_oauth_test_utils import (
    CLIENT_ID,
    SlackApiHandler,
    SlackApiServer,
    SlackOAuthState,
)


class _HoldFirstIssue:
    """Handler-thread tracer: the first ``_issue`` waits until a second handler overlaps it."""

    def __init__(self, oauth: SlackOAuthState) -> None:
        # ``None`` for a fake that does not serialize its grants: its second
        # handler is not kept out and reaches ``_issue`` too.
        self.grant_lock = getattr(oauth, "_lock", None)
        self.answers = 0
        self.issues = 0
        self.overlap = threading.Event()
        # Whether the first ``_issue`` saw the overlap before its wait ran out.
        self.overlapped = False
        self._counting = threading.Lock()

    def __call__(self, frame: Any, event: str, arg: Any) -> Any:
        """React to the ``call`` events of ``answer`` and ``_issue`` only."""
        if event != "call":
            return None
        if frame.f_code is SlackOAuthState.answer.__code__:
            with self._counting:
                self.answers += 1
                self._note_overlap()
        elif frame.f_code is SlackOAuthState._issue.__code__:
            with self._counting:
                self.issues += 1
                first = self.issues == 1
                self._note_overlap()
            if first:
                self.overlapped = self.overlap.wait(10)
        return None

    def _note_overlap(self) -> None:
        """Flag the overlap once the second handler is in ``answer`` behind the held grant
        lock, or has itself reached ``_issue``; called at both handlers' events under
        ``_counting``, so the order in which they arrive does not matter."""
        excluded = self.grant_lock is not None and self.grant_lock.locked()
        if self.issues == 2 or (self.answers == 2 and excluded):
            self.overlap.set()


def test_concurrent_refreshes_redeem_the_token_once() -> None:
    """Of two overlapping refreshes of the same token, one succeeds and one is invalid."""
    with recording_server(SlackApiHandler, SlackApiServer) as server:
        server.oauth.refresh_token = "one-use-token"
        hold = _HoldFirstIssue(server.oauth)
        url = server.base_url + "/api/oauth.v2.access"
        form = {
            "client_id": CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": "one-use-token",
        }
        previous = threading.gettrace()
        threading.settrace(hold)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(requests.post, url, data=form, timeout=15) for _ in range(2)]
                replies = [future.result(timeout=20).json() for future in futures]
        finally:
            threading.settrace(previous)
    assert hold.overlapped, "the two handlers did not overlap"
    assert len(server.oauth.forms) == 2
    assert sorted(reply["ok"] for reply in replies) == [False, True], replies
    assert [reply["error"] for reply in replies if not reply["ok"]] == ["invalid_refresh_token"]
    assert server.oauth.issued == 1
