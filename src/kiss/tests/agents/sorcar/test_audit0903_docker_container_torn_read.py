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
(since removed as unused) checked and then called
``self.container.reload()``.  A ``close()``
that lands between the two reads turned an orderly "no container is
open" refusal into an ``AttributeError`` — or, on the timed-out-exec
reaper path, an ``AssertionError`` that killed the reaper daemon
thread, leaving the timed-out command running for the rest of the
container's life (the exact outcome the reaper exists to prevent).

Fix under test: every method takes ONE snapshot of ``self.container``
and uses only the snapshot afterwards; ``_kill_exec`` is given the
reaper's own snapshot and treats a container removed meanwhile as
"nothing left to kill" (the kill target died with the container)
instead of asserting.

Reproduction notes:

* ``test_kill_exec_after_concurrent_close_is_a_noop`` is the
  deterministic state-based reproduction of the reaper interleaving —
  the reaper took its container snapshot, ``close()`` completed, then
  the reaper invoked the kill on the snapshot.  It failed with
  ``AssertionError`` on the unfixed code.
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

Unreachable-without-doubles branch of the modified code, documented
per the testing policy instead of being mocked: the reaper's
``exec_inspect`` failure branch requires the daemon to drop the
container mid-call.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator

import docker
import pytest

from kiss.agents.sorcar.docker_manager import DockerManager
from kiss.core.kiss_error import KISSError
from kiss.tests.agents.sorcar.docker_test_containers import (
    IMAGE,
    image_container_ids,
    is_docker_available,
    remove_new_image_containers,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not is_docker_available(), reason="Docker daemon is not running"),
]


@pytest.fixture
def cleanup() -> Iterator[None]:
    """Snapshot this process's containers; force-remove the new ones afterwards."""
    client = docker.from_env()
    before = image_container_ids(client)
    try:
        yield
    finally:
        remove_new_image_containers(client, before)


def test_kill_exec_after_concurrent_close_is_a_noop(cleanup: None) -> None:
    """The reaper's kill must survive a close() that won the race.

    The timed-out-exec reaper sweeps the container its run was started
    in; a ``close()`` completing before the sweep hands ``_kill_exec``
    a container that is gone.  On the unfixed code this asserted
    (killing the reaper thread); it must simply do nothing.
    """
    mgr = DockerManager(IMAGE)
    mgr.open()
    container = mgr.container
    assert container is not None
    mgr.close()
    assert mgr.container is None
    # Unfixed code: AssertionError.  Fixed code: silent no-op — the
    # tagged process tree died with the container.
    mgr._kill_exec("deadbeef" * 4, container)


def test_bash_racing_close_never_tears(cleanup: None) -> None:
    """Concurrent Bash vs close(): orderly outcomes only.

    Every worker enters ``Bash``, passes the container check, and —
    thanks to ``KISS_RACE_DELAY`` — parks inside the historical
    check-to-use window while the main thread runs ``close()`` to
    completion.  Acceptable outcomes are a successful result, the
    documented ``KISSError``, or a docker-daemon error for the removed
    container; an ``AttributeError``/``AssertionError`` means a torn
    read of ``self.container``.
    """
    mgr = DockerManager(IMAGE)
    mgr.open()
    barrier = threading.Barrier(4)
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

    threads = [threading.Thread(target=run_bash, daemon=True) for _ in range(3)]
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
    assert len(outcomes) == 3
    assert mgr.container is None


def test_bash_after_close_raises_kiss_error(cleanup: None) -> None:
    """The closed-manager branch stays an orderly KISSError refusal."""
    mgr = DockerManager(IMAGE)
    mgr.open()
    try:
        assert "hi" in mgr.Bash("echo hi", "say hi", timeout_seconds=30)
    finally:
        mgr.close()
    with pytest.raises(KISSError):
        mgr.Bash("echo hi", "say hi", timeout_seconds=30)


def test_streaming_snapshot_path_still_works(cleanup: None) -> None:
    """Streaming Bash works through the snapshot.

    Covers the non-torn branch of the streaming path (shared
    token-tagged exec creation) end to end.
    """
    mgr = DockerManager(IMAGE)
    chunks: list[str] = []
    mgr.stream_callback = chunks.append
    mgr.open()
    try:
        out = mgr.Bash("echo streamed", "stream a line", timeout_seconds=30)
        assert "streamed" in out
        assert "streamed" in "".join(chunks)
    finally:
        mgr.close()
