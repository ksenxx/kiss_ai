"""``kiss.core.thread_join``: a timed ``Thread.join`` that ends at its deadline.

The interpreter defect behind the module (python/cpython#161712) is a
race between two reads of the clock a few nanoseconds apart, so no test
can provoke it on demand.  It was measured instead, on a Windows machine
running the whole suite in parallel: of four probe processes that joined
a parked thread with ``join(0.1)``, each had joins that did not return
(29 of 31,660), while the same loops over ``time.sleep``, ``Event.wait``
and the Win32 waits had none.

These tests pin what the replacement promises on every platform, and
that Windows, where the defect bites, runs it as ``Thread.join``.

Coverage: ``install_bounded_join`` assigns on Windows only, so a platform
reaches one side of its ``if``; the other side needs another platform.
"""

import subprocess
import sys
import threading
import time

import pytest

from kiss.core.thread_join import bounded_join


def _parked() -> tuple[threading.Thread, threading.Event]:
    """Start a thread that lives until the returned event is set."""
    release = threading.Event()
    thread = threading.Thread(target=release.wait, args=(120,), daemon=True)
    thread.start()
    return thread, release


def test_a_timed_join_of_a_live_thread_ends_at_its_deadline() -> None:
    thread, release = _parked()
    try:
        started = time.monotonic()
        bounded_join(thread, 0.3)
        waited = time.monotonic() - started
        assert thread.is_alive()
        assert 0.3 <= waited < 30
    finally:
        release.set()
        thread.join()


@pytest.mark.parametrize("timeout", [0, -5])
def test_a_join_with_no_time_left_only_looks(timeout: float) -> None:
    thread, release = _parked()
    try:
        started = time.monotonic()
        bounded_join(thread, timeout)
        assert time.monotonic() - started < 30
        assert thread.is_alive()
    finally:
        release.set()
        thread.join()


def test_a_timed_join_returns_when_the_thread_ends() -> None:
    thread = threading.Thread(target=time.sleep, args=(0.2,), daemon=True)
    thread.start()
    started = time.monotonic()
    bounded_join(thread, 120)
    assert not thread.is_alive()
    assert time.monotonic() - started < 60


def test_a_timed_join_of_a_finished_thread_returns_at_once() -> None:
    thread = threading.Thread(target=time.sleep, args=(0,), daemon=True)
    thread.start()
    thread.join()
    started = time.monotonic()
    bounded_join(thread, 120)
    assert time.monotonic() - started < 60


def test_a_join_without_a_timeout_waits_for_the_thread() -> None:
    thread = threading.Thread(target=time.sleep, args=(0.2,), daemon=True)
    thread.start()
    bounded_join(thread)
    assert not thread.is_alive()


def test_a_timed_join_refuses_what_thread_join_refuses() -> None:
    with pytest.raises(RuntimeError, match="before it is started"):
        bounded_join(threading.Thread(target=time.sleep, args=(0,)), 1)
    with pytest.raises(RuntimeError, match="current thread"):
        bounded_join(threading.current_thread(), 1)


@pytest.mark.parametrize(
    ("timeout", "error"),
    [
        (float("nan"), ValueError),
        (float("inf"), OverflowError),
        (1e300, OverflowError),
        (10**30, OverflowError),
        ("1", TypeError),
    ],
)
def test_a_timeout_the_interpreter_cannot_use_is_refused(
    timeout: float, error: type[Exception]
) -> None:
    """The interpreter's join raises these at once; a NaN must not turn into an endless wait."""
    thread, release = _parked()
    try:
        with pytest.raises(error):
            bounded_join(thread, timeout)
        assert thread.is_alive()
    finally:
        release.set()
        thread.join()


def test_windows_runs_the_bounded_join_and_other_platforms_keep_theirs() -> None:
    assert (threading.Thread.join is bounded_join) is (sys.platform == "win32")


def test_a_command_with_captured_output_ends_at_its_timeout() -> None:
    """``Popen.communicate(timeout=...)`` waits with ``Thread.join`` on Windows."""
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            capture_output=True,
            timeout=0.5,
        )
    assert time.monotonic() - started < 60
