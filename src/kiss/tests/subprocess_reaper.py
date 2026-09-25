# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest plugin that terminates subprocesses tests leave running.

Why
---
A test run once left 529 detached ``muse_auth.daemon`` processes behind:
tests started a daemon through the product's ``subprocess.Popen(...,
start_new_session=True)`` and never stopped it, so every daemon outlived
its test, the pytest process, and eventually the worktree it was started
from.  This plugin makes that impossible to repeat without anyone having
to remember a finalizer.

How
---
Every :class:`subprocess.Popen` constructed after this module is imported
is recorded by wrapping ``Popen.__init__`` (``subprocess.run``,
``check_output``, and asyncio's subprocess transports all build a
``Popen``, so they are covered too).  Each process is attributed to the
innermost *bucket* open at the time it starts:

* the **test** being set up, run or torn down -- swept in
  :func:`pytest_runtest_teardown` after the test's own fixtures and
  ``tearDown`` have run, so a daemon a test forgets is gone before the
  next test starts.  Only processes *nothing references any more* are
  terminated there (see :func:`_sweep_test`): one a shared fixture's
  object started on first use and still holds is that object's to stop,
  and is deferred to the end of the run;
* a **class/module/package/session-scoped fixture** (``setUpClass`` and
  ``setUpModule`` are such fixtures in pytest) while it is being set up
  -- swept by a finalizer that runs right after the fixture's own
  teardown, so a shared server lives exactly as long as its fixture;
* the **process** for anything else (conftest import, collection) --
  swept in :func:`pytest_sessionfinish`; an ``atexit`` sweep of every
  bucket is the last resort for a run cut short by ``pytest.exit`` or
  Ctrl-C and for anything a background thread started after that.

A sweep sends ``SIGTERM`` to everything still alive, waits up to
:data:`_TERM_GRACE_SECONDS`, then ``SIGKILL``s the stubborn ones.  A child
that leads its own process group (``start_new_session=True``,
``process_group=0``) is signalled as a group and counts as alive while
*any* member of the group exists, so its descendants die with it even
when the leader itself has already exited (the double-fork daemon
pattern); on Windows :func:`kiss.core.processes.kill_process_group`
terminates the child's Job Object or process tree instead (``taskkill
/F`` knows no graceful phase there).  Each sweep that had to terminate
something reports a :class:`LeakedSubprocessWarning` naming the test or
fixture and the commands, so the leak can be fixed at its source.

Effects and limits
------------------
A recorded ``Popen`` is referenced until its bucket is swept, so a pipe
the test neither reads nor closes stays open until the test ends
(``subprocess.run`` and ``with Popen(...)`` close theirs).  Processes
created without a ``Popen`` (``os.fork``, ``os.posix_spawn``,
:mod:`multiprocessing`) are not tracked, and a non-leader child's own
children cannot be found from the parent alone.
"""

from __future__ import annotations

import atexit
import functools
import gc
import os
import signal
import subprocess
import sys
import threading
import time
import types
import warnings
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import pytest

from kiss.core.processes import IS_WINDOWS, kill_process_group

_TERM_GRACE_SECONDS = 5.0
_KILL_GRACE_SECONDS = 3.0
_POLL_INTERVAL_SECONDS = 0.02
_FORCE_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)


@dataclass(slots=True)  # slots: ``gc.get_referrers`` then names the entry itself, not a dict
class _Tracked:
    proc: subprocess.Popen[Any]
    # Whether the child leads its own process group (decided once, right
    # after it starts, because a dead leader can no longer be asked).
    leads_group: bool


Bucket = list[_Tracked]

# Innermost bucket last.  The bottom bucket belongs to the process.
_stack: list[Bucket] = [[]]
_stack_lock = threading.Lock()
_BUCKET_KEY: pytest.StashKey[Bucket] = pytest.StashKey()
_original_popen_init = subprocess.Popen.__init__


class LeakedSubprocessWarning(pytest.PytestWarning):
    """A test, fixture or session left subprocesses running and they were terminated."""


def _leads_own_group(proc: subprocess.Popen[Any]) -> bool:
    """Return whether *proc* should be signalled as a whole process group.

    Windows children are always handled by :func:`kill_process_group`
    (Job Object or ``taskkill /T``), which reaches their descendants.
    On POSIX only a child that leads its own group (``start_new_session``
    or ``process_group=0``) is safe to ``killpg``: every other child
    shares *our* group.
    """
    if IS_WINDOWS:  # pragma: no cover — Windows-only branch
        return True
    try:
        return os.getpgid(proc.pid) == proc.pid
    except ProcessLookupError:  # pragma: no cover — the child died before we could ask
        return False


@functools.wraps(_original_popen_init)
def _tracking_popen_init(self: subprocess.Popen[Any], *args: Any, **kwargs: Any) -> None:
    _original_popen_init(self, *args, **kwargs)
    entry = _Tracked(self, _leads_own_group(self))
    with _stack_lock:
        _stack[-1].append(entry)


# Installed at import rather than in ``pytest_configure``: the root
# conftest that registers this plugin is imported *before* configure, and
# a process a conftest starts at import time must be tracked too.  The
# wrapper stays for the life of the process; restoring it would blind an
# outer session whenever a test runs ``pytest.main`` in-process.
subprocess.Popen.__init__ = _tracking_popen_init  # type: ignore[method-assign]


def _open_bucket() -> Bucket:
    bucket: Bucket = []
    with _stack_lock:
        _stack.append(bucket)
    return bucket


def _close_bucket(bucket: Bucket) -> None:
    # Buckets nest strictly: a fixture's setup happens inside a test's
    # setup phase, so the one being closed is always the innermost.
    with _stack_lock:
        assert _stack.pop() is bucket


def _group_exists(pid: int) -> bool:
    if IS_WINDOWS:  # pragma: no cover — Windows descendants die with the leader's Job Object
        return False
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover — a member we may not signal (setuid)
        return True
    return True


def _alive(entry: _Tracked) -> bool:
    """Return whether the child, or any member of the group it leads, still exists."""
    if entry.proc.poll() is None:
        return True
    return entry.leads_group and _group_exists(entry.proc.pid)


def _signal(entry: _Tracked, force: bool) -> None:
    sig = _FORCE_SIGNAL if force else signal.SIGTERM
    try:
        if entry.leads_group:
            kill_process_group(entry.proc.pid, sig)
        else:
            entry.proc.send_signal(sig)
    except ProcessLookupError:  # pragma: no cover — exited since the liveness check
        pass
    except OSError:  # pragma: no cover — tree kill refused (Windows, EPERM member)
        # ``send_signal`` is a no-op for an already reaped child and
        # raises ``ProcessLookupError`` for one that died meanwhile.
        try:
            entry.proc.send_signal(sig)
        except ProcessLookupError:
            pass


def _wait_all(entries: list[_Tracked], grace: float) -> None:
    deadline = time.monotonic() + grace
    for entry in entries:
        while _alive(entry) and time.monotonic() < deadline:
            time.sleep(_POLL_INTERVAL_SECONDS)


def _describe(entry: _Tracked) -> str:
    args = str(entry.proc.args)
    if len(args) > 200:
        args = args[:197] + "..."
    suffix = " (still running)" if _alive(entry) else ""
    return f"pid {entry.proc.pid}: {args}{suffix}"


def terminate_leftovers(bucket: Bucket) -> list[str]:
    """Terminate every process in *bucket* that is still running and empty it.

    Args:
        bucket: Processes recorded for one test, fixture or the process.

    Returns:
        A description of each process that had to be terminated (empty
        when everything had already exited on its own).
    """
    alive = [entry for entry in bucket if _alive(entry)]
    bucket.clear()
    if not alive:
        return []
    for entry in alive:
        _signal(entry, force=False)
    _wait_all(alive, _TERM_GRACE_SECONDS)
    stubborn = [entry for entry in alive if _alive(entry)]
    for entry in stubborn:
        _signal(entry, force=True)
    _wait_all(stubborn, _KILL_GRACE_SECONDS)
    return [_describe(entry) for entry in alive]


def _report(owner: str, terminated: list[str]) -> str:
    return f"{owner} left {len(terminated)} subprocess(es) running; terminated: " + "; ".join(
        terminated
    )


def _warn_about(owner: str, terminated: list[str]) -> None:
    if not terminated:
        return
    # A test that leaves ``simplefilter("error")`` behind must not turn
    # this report into a teardown error; pytest's per-test recorder still
    # receives the warning because only the filters are overridden here.
    with warnings.catch_warnings():
        warnings.simplefilter("always", LeakedSubprocessWarning)
        warnings.warn(_report(owner, terminated), LeakedSubprocessWarning, stacklevel=2)


def _sweep_fixture(bucket: Bucket, owner: str) -> None:
    _warn_about(owner, terminate_leftovers(bucket))


def _owned(entry: _Tracked) -> bool:
    """Return whether anything besides this plugin still references the ``Popen``.

    Frames are ignored: by teardown the only running frames that can hold
    the object are this plugin's own.
    """
    return any(
        referrer is not entry and not isinstance(referrer, types.FrameType)
        for referrer in gc.get_referrers(entry.proc)
    )


def _sweep_test(bucket: Bucket, owner: str) -> None:
    """Terminate what the test left running *and dropped*; defer what is still owned.

    A ``Popen`` some live object still holds -- a browser driver a
    module-scoped tool started on first use, a server kept in a module
    global -- belongs to that owner and moves to the enclosing bucket, to
    be swept when the run ends if the owner never stops it.  A ``Popen``
    nothing references any more is a leak and is terminated now.  The
    collection first frees the test's garbage (a ``TestCase`` instance in
    a reference cycle, an asyncio transport, ...) so that a process only
    such garbage referenced counts as dropped.
    """
    alive = [entry for entry in bucket if _alive(entry)]
    bucket.clear()
    if not alive:
        return
    gc.collect()
    owned: Bucket = []
    dropped: Bucket = []
    for entry in alive:
        (owned if _owned(entry) else dropped).append(entry)
    with _stack_lock:
        _stack[-1].extend(owned)
    _warn_about(owner, terminate_leftovers(dropped))


def _sweep_buckets(buckets: list[Bucket]) -> list[str]:
    with _stack_lock:
        leftovers = [entry for bucket in buckets for entry in bucket]
        for bucket in buckets:
            bucket.clear()
    return terminate_leftovers(leftovers)


def _sweep_at_exit() -> None:
    """Last resort: everything still tracked when the interpreter exits.

    Covers a process a background thread started after the session
    ended, and the test bucket of a run cut short by ``pytest.exit`` or
    Ctrl-C, whose teardown phase never ran.
    """
    terminated = _sweep_buckets(_stack[:])
    if terminated:
        print(_report("subprocess_reaper: the process", terminated), file=sys.stderr)


atexit.register(_sweep_at_exit)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item: pytest.Item) -> Generator[None, object, object]:
    """Open the test's bucket before any of its fixtures run.

    Args:
        item: The test about to be set up.

    Returns:
        Generator required by the pytest hookwrapper protocol.
    """
    item.stash[_BUCKET_KEY] = _open_bucket()
    return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item: pytest.Item) -> Generator[None, object, object]:
    """Terminate what the test left running once all its finalizers ran.

    Args:
        item: The test being torn down.

    Returns:
        Generator required by the pytest hookwrapper protocol.
    """
    try:
        return (yield)
    finally:
        bucket = item.stash.setdefault(_BUCKET_KEY, [])
        _close_bucket(bucket)
        _sweep_test(bucket, f"test {item.nodeid}")


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(
    fixturedef: pytest.FixtureDef[Any], request: pytest.FixtureRequest
) -> Generator[None, object, object]:
    """Give a class/module/package/session fixture its own bucket while it sets up.

    Processes it starts are terminated by a finalizer registered *before*
    the fixture's own teardown, so (finalizers run last-in-first-out)
    they are swept right after the fixture has had its chance to stop
    them cleanly.  Function-scoped fixtures belong to the test's bucket.

    Args:
        fixturedef: The fixture being set up.
        request: The request for it, used to register the finalizer.

    Returns:
        Generator required by the pytest hookwrapper protocol.
    """
    if fixturedef.scope == "function":
        return (yield)
    bucket = _open_bucket()
    owner = f"{fixturedef.scope}-scoped fixture {fixturedef.argname!r} ({fixturedef.baseid})"
    request.addfinalizer(functools.partial(_sweep_fixture, bucket, owner))
    try:
        return (yield)
    finally:
        _close_bucket(bucket)


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Terminate every process started outside tests that is still alive at the end.

    Args:
        session: The finishing session; its terminal writer reports what
            had to be terminated (warnings recorded this late are not
            shown in the summary).
    """
    # Only the bottom bucket: a nested in-process ``pytest.main`` must
    # not kill what the enclosing test or fixture is still using.
    terminated = _sweep_buckets(_stack[:1])
    if terminated:
        writer = session.config.get_terminal_writer()
        writer.line("")
        writer.line(_report("subprocess_reaper: the session", terminated), yellow=True)
