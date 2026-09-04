# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Review fixes #4 and #6: ``DockerManager`` lifecycle under concurrency.

* **#4** Two concurrent ``open()`` calls on one manager both passed the
  ``container is not None`` guard while the field was still ``None``
  (a Docker start takes long enough), so both started a container and
  created a shared-volume temp dir while the manager kept only one id:
  one container and one directory leaked for good.  A per-manager
  lifecycle lock now serialises guard, temp dir, start, publication
  and ``close()``.
* **#6** ``_remove_shared_volume_dir`` cleared ``host_shared_path``
  BEFORE ``rmtree``, so a failed removal left an untraceable directory
  behind and no later ``close()`` could retry.  The field is now
  cleared only after the directory is gone.

Both run against the real Docker daemon (skipped when none is running)
with real threads; every container and directory is removed afterwards.
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
def cleanup() -> Iterator[set[str | None]]:
    """Snapshot existing containers; force-remove anything new afterwards."""
    client = docker.from_env()
    before: set[str | None] = {c.id for c in client.containers.list(all=True)}
    try:
        yield before
    finally:
        for container in client.containers.list(all=True):
            if container.id not in before:
                try:
                    container.remove(force=True)
                except Exception:
                    pass


def test_concurrent_opens_start_exactly_one_container(cleanup: set[str]) -> None:
    """Of two simultaneous ``open()`` calls one succeeds and one is refused."""
    client = docker.from_env()
    mgr = DockerManager(IMAGE)
    barrier = threading.Barrier(2)
    outcomes: list[BaseException | None] = [None, None]
    dirs_seen: list[str | None] = [None, None]

    def opener(slot: int) -> None:
        barrier.wait(30)
        try:
            mgr.open()
        except BaseException as exc:  # noqa: BLE001 — recorded for the assertions
            outcomes[slot] = exc
        dirs_seen[slot] = mgr.host_shared_path

    threads = [threading.Thread(target=opener, args=(i,)) for i in range(2)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=180)
        assert all(not t.is_alive() for t in threads)

        errors = [e for e in outcomes if e is not None]
        assert len(errors) == 1, f"expected exactly one refusal, got {outcomes!r}"
        assert isinstance(errors[0], KISSError)
        assert mgr.container is not None
        started = {c.id for c in client.containers.list(all=True)} - cleanup
        assert started == {mgr.container.id}, (
            f"containers started: {started}, tracked: {mgr.container.id}"
        )
        shared = mgr.host_shared_path
        assert shared is not None and os.path.isdir(shared)
        # No second temp dir was created for the refused call.
        assert set(d for d in dirs_seen if d) == {shared}
        assert mgr.Bash("echo one", "probe").strip() == "one"
    finally:
        mgr.close()
    assert mgr.container is None
    assert mgr.host_shared_path is None
    assert not os.path.exists(shared)


def test_concurrent_close_and_open_never_orphan(cleanup: set[str]) -> None:
    """``close()`` racing ``open()`` on an open manager leaves one
    consistent state: either the re-open won (one container) or it was
    refused; never two containers."""
    client = docker.from_env()
    mgr = DockerManager(IMAGE)
    mgr.open()
    first = mgr.container
    assert first is not None
    barrier = threading.Barrier(2)
    open_error: list[BaseException | None] = [None]

    def closer() -> None:
        barrier.wait(30)
        mgr.close()

    def reopener() -> None:
        barrier.wait(30)
        try:
            mgr.open()
        except BaseException as exc:  # noqa: BLE001
            open_error[0] = exc

    threads = [threading.Thread(target=closer), threading.Thread(target=reopener)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=180)
        alive = {c.id for c in client.containers.list(all=True)} - cleanup
        if open_error[0] is None:
            # open ran after close: exactly the new container remains.
            assert mgr.container is not None
            assert alive == {mgr.container.id}
        else:
            # open ran before close and was refused; close removed the first.
            assert isinstance(open_error[0], KISSError)
            assert mgr.container is None
            assert alive == set()
    finally:
        mgr.close()


def test_failed_shared_dir_removal_is_retried_by_next_close(cleanup: set[str]) -> None:
    """A directory ``rmtree`` cannot delete stays tracked and is removed
    by the next ``close()`` once it is deletable."""
    mgr = DockerManager(IMAGE)
    mgr.open()
    shared = mgr.host_shared_path
    assert shared is not None
    locked = os.path.join(shared, "locked")
    os.mkdir(locked)
    with open(os.path.join(locked, "f"), "w", encoding="utf-8") as fh:
        fh.write("x")
    os.chmod(locked, 0o000)  # rmtree cannot list/remove the entry
    try:
        mgr.close()  # must not raise
        assert mgr.container is None
        assert os.path.isdir(shared), "expected the rmtree to have failed"
        assert mgr.host_shared_path == shared, "leaked directory lost track"
        with pytest.raises(KISSError, match="shared volume"):
            mgr.open()  # still undeletable: refused, nothing new started
        assert mgr.container is None
        assert mgr.host_shared_path == shared
    finally:
        os.chmod(locked, 0o700)
    # Deletable again: open() clears the leftover first, then starts.
    mgr.open()
    assert not os.path.exists(shared)
    assert mgr.container is not None
    fresh = mgr.host_shared_path
    assert fresh is not None and fresh != shared and os.path.isdir(fresh)
    mgr.close()
    assert mgr.host_shared_path is None
    assert not os.path.exists(fresh)


def test_close_retries_removal_of_a_tracked_leftover(cleanup: set[str]) -> None:
    """A second ``close()`` removes the directory the first one could not."""
    mgr = DockerManager(IMAGE)
    mgr.open()
    shared = mgr.host_shared_path
    assert shared is not None
    locked = os.path.join(shared, "locked")
    os.mkdir(locked)
    os.chmod(locked, 0o000)
    try:
        mgr.close()
        assert mgr.host_shared_path == shared
    finally:
        os.chmod(locked, 0o700)
    mgr.close()  # no container, but the leftover is retried
    assert mgr.host_shared_path is None
    assert not os.path.exists(shared)


def test_remove_shared_dir_when_already_absent_clears_field() -> None:
    """A tracked directory that vanished externally is simply forgotten."""
    mgr = DockerManager(IMAGE)
    mgr.host_shared_path = os.path.join("/nonexistent", "audit0902-fix")
    mgr._remove_shared_volume_dir()
    assert mgr.host_shared_path is None
    mgr._remove_shared_volume_dir()  # no path: no-op
    assert mgr.host_shared_path is None
