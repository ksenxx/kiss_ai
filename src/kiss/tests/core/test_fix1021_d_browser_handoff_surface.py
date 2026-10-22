# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``open_for_user`` reports where the page really is (audit 2026-10-21, D-F1).

Two task threads hand the same sign-in page to the user while a Browser
tab opener is registered.  The first caller's launch drags on past the
reopen guard (``_REOPEN_AFTER``, lowered here) and then fails; the
second caller, arriving once that claim is stale, takes the page over
and its retry of the opener succeeds.  The first caller then falls back
to the default browser, meets the second caller's retry in flight and
must share it: the page is open in the Browser tab, so it must report
:data:`BROWSER_TAB`, never :data:`DEFAULT_BROWSER` (the old F1 bug) and
never :data:`NOT_OPENED` (the headless daemon's bug: it gave up before
looking at the claim at all).

The stale launch is also the only way the first caller's own release
can meet a claim it does not own: releasing the second caller's claim
would send the first caller off to launch a default browser that does
not exist and report nothing opened, so the owner check in the release
is exercised too.

Only the public ``open_for_user`` is driven, with a real opener object
whose two calls are ordered by events.  A waiter that takes over right
after the first launch's release would exercise the same fall-back
path, but its ordering against the releasing thread's own fall-back is
decided inside ``open_for_user`` with no hook to pin it, so the stale
launch is the interleaving this test fixes.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Iterator

import pytest

from kiss.core import browser_handoff
from kiss.core.browser_handoff import BROWSER_TAB, open_for_user, set_browser_tab_opener

_STEP_TIMEOUT = 10.0
_REOPEN_AFTER = 1.0


class _RetriedOpener:
    """Opener whose first launch outlives the reopen guard and fails; the second succeeds."""

    def __init__(self) -> None:
        self.calls = 0
        self.first_called_at = 0.0
        self.first_called = threading.Event()
        self.second_called = threading.Event()

    def __call__(self, url: str) -> bool:
        self.calls += 1
        if self.calls == 1:
            self.first_called_at = time.monotonic()
            self.first_called.set()
            # The launch drags on until the next caller has taken the
            # page over, then fails: the first caller falls back while
            # that retry is in flight.
            assert self.second_called.wait(_STEP_TIMEOUT), "no caller took the stale launch over"
            return False
        self.second_called.set()
        return True


@pytest.fixture()
def opener(monkeypatch: pytest.MonkeyPatch) -> Iterator[_RetriedOpener]:
    """Register the retried opener; the reopen guard is short and no default browser exists."""
    retried = _RetriedOpener()
    set_browser_tab_opener(retried)
    monkeypatch.setattr(browser_handoff, "_REOPEN_AFTER", _REOPEN_AFTER)
    monkeypatch.setenv("BROWSER", "/no/such/browser-fix1021")
    yield retried
    set_browser_tab_opener(None)
    retried.second_called.set()


def _open(url: str, results: dict[str, str], name: str) -> None:
    results[name] = open_for_user(url)


def _wait_until_stale(claimed_before: float) -> None:
    """Block until a claim recorded before *claimed_before* has outlived the reopen guard."""
    while time.monotonic() - claimed_before < browser_handoff._REOPEN_AFTER:
        time.sleep(0.01)


@pytest.mark.parametrize("headless", ["0", "1"])
def test_first_caller_reports_the_browser_tab_its_successor_opened(
    opener: _RetriedOpener, monkeypatch: pytest.MonkeyPatch, headless: str
) -> None:
    """Headless or not, the failed launch's caller reports the retry that replaced it."""
    monkeypatch.setenv("KISS_HEADLESS", headless)
    url = f"https://example.test/signin/{uuid.uuid4()}"
    results: dict[str, str] = {}
    first = threading.Thread(target=_open, args=(url, results, "first"), daemon=True)
    second = threading.Thread(target=_open, args=(url, results, "second"), daemon=True)
    started = time.monotonic()
    first.start()
    assert opener.first_called.wait(_STEP_TIMEOUT)
    # The second caller must not queue behind a launch that outlived the
    # reopen guard: it takes the page over and retries the opener.
    _wait_until_stale(opener.first_called_at)
    second.start()
    for thread in (first, second):
        thread.join(timeout=_STEP_TIMEOUT)
        assert not thread.is_alive(), "open_for_user() did not return"
    assert results == {"first": BROWSER_TAB, "second": BROWSER_TAB}
    assert opener.calls == 2
    # The first caller's release left the second caller's settled claim
    # alone: the page counts as open and is not opened a third time.
    assert open_for_user(url) == BROWSER_TAB
    assert opener.calls == 2
    assert time.monotonic() - started < browser_handoff._SETTLE_TIMEOUT / 4
