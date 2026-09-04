# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-03: ``DockerManager`` torn reads of ``self.container``
racing ``close()``.

``self.container`` is nulled by ``close()`` (under the lifecycle
lock), but the command paths read the attribute MORE THAN ONCE without
that lock: ``Bash`` checked ``self.container is None`` and then read
``self.container.id`` again; ``_bash_streaming`` and ``_kill_exec``
asserted on one read and dereferenced another; ``get_host_port``
checked and then called ``self.container.reload()``.  A ``close()``
that lands between the two reads turned an orderly "no container is
open" refusal into an ``AttributeError`` — or, on the timed-out-exec
reaper path, an ``AssertionError`` that killed the reaper daemon
thread, leaving the timed-out command running for the rest of the
container's life (the exact outcome the reaper exists to prevent).

Fix under test: every method takes ONE snapshot of ``self.container``
and uses only the snapshot afterwards; ``_kill_exec`` treats a
concurrently-closed container as "nothing left to kill" (the kill
target died with the container) instead of asserting.

Reproduction notes:

* ``test_kill_exec_after_concurrent_close_is_a_noop`` is the
  deterministic state-based reproduction of the reaper interleaving —
  the reaper's loop observed a live container, ``close()`` completed,
  then the reaper invoked the kill.  It fails with ``AssertionError``
  on the unfixed code.
* The ``Bash``/``get_host_port`` windows are a few instructions wide,
  so per the audit protocol they were confirmed on the unfixed code by
  temporarily inserting a ``time.sleep(0.08)`` between the check and
  the second attribute read (``AttributeError: 'NoneType' object has
  no attribute 'id'`` / ``... 'reload'``); the temporary sleeps were
  removed after confirmation.  The remaining ``_race_delay()`` hook in
  the fixed methods (a production no-op) lets
  ``test_bash_racing_close_never_tears`` hold every worker inside the
  historical window while ``close()`` runs, proving the snapshot makes
  the interleaving harmless.

All tests run against the real Docker daemon (skipped when none is
available) with real threads — no mocks of the code under test.

Unreachable-without-doubles branches of the modified code, documented
per the testing policy instead of being mocked: ``_kill_exec``'s
``except Exception`` (the container vanishing between the snapshot and
``exec_run``) and the reaper's ``exec_inspect`` failure branch require
the daemon to drop the container mid-call.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator

import docker
import pytest

from kiss.agents.sorcar.docker_manager import DockerManager
from kiss.core.kiss_error import KISSError

IMAGE = "python:3.11-slim"


def _docker_available() -> bool:
    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _docker_available(), reason="Docker daemon is not running"),
]


@pytest.fixture
def cleanup() -> Iterator[None]:
    """Snapshot existing containers; force-remove anything new afterwards."""
    client = docker.from_env()
    before = {c.id for c in client.containers.list(all=True)}
    try:
        yield
    finally:
        for container in client.containers.list(all=True):
            if container.id not in before:
                try:
                    container.remove(force=True)
                except Exception:
                    pass


def test_kill_exec_after_concurrent_close_is_a_noop(cleanup: None) -> None:
    """The reaper's kill must survive a close() that won the race.

    The timed-out-exec reaper checks ``self.container is None`` at the
    top of each poll and calls ``_kill_exec`` afterwards; a ``close()``
    completing in between handed ``_kill_exec`` a manager whose
    container is gone.  On the unfixed code this asserted (killing the
    reaper thread); it must simply do nothing.
    """
    mgr = DockerManager(IMAGE)
    mgr.open()
    mgr.close()
    assert mgr.container is None
    # Unfixed code: AssertionError.  Fixed code: silent no-op — the
    # tagged process tree died with the container.
    mgr._kill_exec("deadbeef" * 4)


def test_bash_racing_close_never_tears(cleanup: None) -> None:
    """Concurrent Bash/get_host_port vs close(): orderly outcomes only.

    Every worker enters its method, passes the container check, and —
    thanks to ``KISS_RACE_DELAY`` — parks inside the historical
    check-to-use window while the main thread runs ``close()`` to
    completion.  Acceptable outcomes are a successful result, the
    documented ``KISSError``, or a docker-daemon error for the removed
    container; an ``AttributeError``/``AssertionError`` means a torn
    read of ``self.container``.
    """
    mgr = DockerManager(IMAGE)
    mgr.open()
    barrier = threading.Barrier(5)
    outcomes: list[BaseException | str] = []
    outcomes_lock = threading.Lock()

    def record(value: BaseException | str) -> None:
        with outcomes_lock:
            outcomes.append(value)

    def run_bash() -> None:
        barrier.wait(30)
        try:
            record(mgr.Bash("echo hi", "say hi", timeout_seconds=30))
        except BaseException as exc:  # noqa: BLE001 — recorded for assertions
            record(exc)

    def run_port() -> None:
        barrier.wait(30)
        try:
            record(str(mgr.get_host_port(80)))
        except BaseException as exc:  # noqa: BLE001 — recorded for assertions
            record(exc)

    threads = [threading.Thread(target=run_bash, daemon=True) for _ in range(3)]
    threads.append(threading.Thread(target=run_port, daemon=True))
    os.environ["KISS_RACE_DELAY"] = "0.1"
    try:
        for t in threads:
            t.start()
        barrier.wait(30)
        mgr.close()
        for t in threads:
            t.join(timeout=120)
            assert not t.is_alive()
    finally:
        os.environ.pop("KISS_RACE_DELAY", None)

    torn = [
        o for o in outcomes
        if isinstance(o, (AttributeError, AssertionError))
    ]
    assert not torn, f"torn container read under concurrent close(): {torn!r}"
    assert len(outcomes) == 4
    assert mgr.container is None


def test_bash_and_get_host_port_after_close_raise_kiss_error(
    cleanup: None,
) -> None:
    """The closed-manager branch stays an orderly KISSError refusal."""
    mgr = DockerManager(IMAGE)
    mgr.open()
    try:
        assert "hi" in mgr.Bash("echo hi", "say hi", timeout_seconds=30)
    finally:
        mgr.close()
    with pytest.raises(KISSError):
        mgr.Bash("echo hi", "say hi", timeout_seconds=30)
    with pytest.raises(KISSError):
        mgr.get_host_port(80)


def test_streaming_and_ports_snapshot_paths_still_work(cleanup: None) -> None:
    """Streaming Bash and a real port mapping work through the snapshots.

    Covers the non-torn branches of the modified methods end to end:
    the streaming path (shared token-tagged exec creation), a mapped
    port, and an unmapped port.
    """
    mgr = DockerManager(IMAGE, ports={80: 0})
    chunks: list[str] = []
    mgr.stream_callback = chunks.append
    mgr.open()
    try:
        out = mgr.Bash("echo streamed", "stream a line", timeout_seconds=30)
        assert "streamed" in out
        assert "streamed" in "".join(chunks)
        port = mgr.get_host_port(80)
        assert isinstance(port, int) and port > 0
        assert mgr.get_host_port(4321) is None
    finally:
        mgr.close()
