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
from kiss.core.browser_handoff import (
    BROWSER_TAB,
    NOT_OPENED,
    open_for_user,
    set_browser_tab_opener,
)

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


class _PendingRetryOpener:
    """Opener whose first launch fails on cue and whose retry stays pending until released."""

    def __init__(self, retry_opens: bool) -> None:
        self.retry_opens = retry_opens
        self.callers: list[str] = []
        self.first_called = threading.Event()
        self.fail_first = threading.Event()
        self.retry_called = threading.Event()
        self.finish_retry = threading.Event()

    def __call__(self, url: str) -> bool:
        self.callers.append(threading.current_thread().name)
        if not self.first_called.is_set():
            self.first_called.set()
            self.fail_first.wait(_STEP_TIMEOUT)
            return False
        if self.retry_called.is_set():
            return False  # a third launch: the assertions on ``callers`` report it
        self.retry_called.set()
        self.finish_retry.wait(_STEP_TIMEOUT)
        return self.retry_opens


@pytest.mark.parametrize("retry_opens", [False, True])
def test_waiter_outlasted_by_a_pending_retry_reports_nothing_opened(
    monkeypatch: pytest.MonkeyPatch, retry_opens: bool
) -> None:
    """A wait that runs out during a replacement launch reports no page (audit C1-F1).

    Two callers wait for the first launch; it fails, one of them takes
    the page over, and its retry is still pending when the other's wait
    runs out.  That waiter used to report the retry's surface although
    no launch had succeeded, and none might ever.  The pending retry
    stays its launching thread's: the waiter opens no second copy, and
    a retry that succeeds is still settled by its owner.
    """
    settle = 1.2
    opener = _PendingRetryOpener(retry_opens)
    set_browser_tab_opener(opener)
    monkeypatch.setattr(browser_handoff, "_REOPEN_AFTER", settle)
    monkeypatch.setattr(browser_handoff, "_SETTLE_TIMEOUT", settle)
    monkeypatch.setenv("KISS_HEADLESS", "1")
    url = f"https://example.test/signin/{uuid.uuid4()}"
    results: dict[str, str] = {}
    threads = {
        name: threading.Thread(target=_open, args=(url, results, name), name=name, daemon=True)
        for name in "ABC"
    }
    try:
        threads["A"].start()
        assert opener.first_called.wait(_STEP_TIMEOUT)
        threads["B"].start()
        threads["C"].start()
        # Both waiters are waiting well before the retry is claimed, so
        # neither finds that claim stale when its own wait runs out.
        time.sleep(settle * 0.75)
        opener.fail_first.set()
        assert opener.retry_called.wait(_STEP_TIMEOUT), "no waiter retried the failed launch"
        retrier = opener.callers[1]
        waiter = "C" if retrier == "B" else "B"
        threads[waiter].join(timeout=_STEP_TIMEOUT)
        # The retry is still pending: the page is open nowhere.
        assert results.get(waiter) == NOT_OPENED
        opener.finish_retry.set()
        for thread in threads.values():
            thread.join(timeout=_STEP_TIMEOUT)
            assert not thread.is_alive(), "open_for_user() did not return"
        if retry_opens:
            assert results[retrier] == BROWSER_TAB
            assert open_for_user(url) == BROWSER_TAB  # settled by its owner: not opened again
        else:
            assert results == dict.fromkeys("ABC", NOT_OPENED)
        assert opener.callers == ["A", retrier]
    finally:
        opener.fail_first.set()
        opener.finish_retry.set()
        set_browser_tab_opener(None)


def _opens(url: str) -> bool:
    """A Browser-tab opener whose every launch succeeds."""
    return True


def test_settled_claims_past_the_reopen_guard_are_forgotten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A settled hand-off stays on record only until the reopen guard has passed.

    Every sign-in URL carries a fresh state parameter, so a record that
    only ever replaced the one URL being claimed again kept every page
    the daemon had opened in its lifetime.
    """
    set_browser_tab_opener(_opens)
    monkeypatch.setattr(browser_handoff, "_REOPEN_AFTER", _REOPEN_AFTER / 10)
    browser_handoff._recent.clear()
    urls = [f"https://example.test/signin/{uuid.uuid4()}" for _ in range(3)]
    try:
        assert open_for_user(urls[0]) == BROWSER_TAB
        assert open_for_user(urls[1]) == BROWSER_TAB
        _wait_until_stale(browser_handoff._recent[urls[1]].opened_at)
        assert open_for_user(urls[2]) == BROWSER_TAB
    finally:
        set_browser_tab_opener(None)
    assert set(browser_handoff._recent) == {urls[2]}
