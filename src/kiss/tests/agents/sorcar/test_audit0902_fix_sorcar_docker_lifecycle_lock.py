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
import time
from collections.abc import Iterator

import docker
import pytest

from kiss.agents.sorcar.docker_manager import _SWEEP_TIMEOUT_S, DockerManager
from kiss.core import tool_interrupt
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
def cleanup() -> Iterator[set[str]]:
    """Snapshot this process's containers; force-remove the new ones afterwards."""
    client = docker.from_env()
    before = image_container_ids(client)
    try:
        yield before
    finally:
        remove_new_image_containers(client, before)


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
        started = image_container_ids(client) - cleanup
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
        alive = image_container_ids(client) - cleanup
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
    assert mgr.container is not None
    mgr.container.kill()  # already gone: the container cannot empty the volume itself
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
    assert mgr.container is not None
    mgr.container.kill()  # already gone: the container cannot empty the volume itself
    try:
        mgr.close()
        assert mgr.host_shared_path == shared
    finally:
        os.chmod(locked, 0o700)
    mgr.close()  # no container, but the leftover is retried
    assert mgr.host_shared_path is None
    assert not os.path.exists(shared)


#: A ``find`` that says it has started, is at work while it polls for the test's go-ahead,
#: and only then deletes.
_GATED_FIND = (
    "#!/bin/sh\\ntouch /testbed/sweeping\\nwhile [ ! -e /tmp/gate ]; do sleep 0.2; done\\n"
    "exec /usr/bin/find \"$@\"\\n"
)


@pytest.mark.parametrize("case", ["ordinary", "stopped", "outlasting"])
def test_close_sweeps_root_owned_files_whatever_the_sweep_takes(
    cleanup: set[str], case: str,
) -> None:
    """Root-owned files are gone after ``close()``: under a set Stop, and past the bound.

    Review r1 (R1): a sweep bounded in total was cut while it still worked,
    and what it left is root-owned on the host, out of ``rmtree``'s reach.
    Review r3 (R1): so did a sweep that the stopped tool call of the closing
    thread never started.  In the last case the container's ``find`` is busy
    past the bound (it polls for this test's go-ahead), and the sweep's line
    every 5 s tells ``close()`` that it is at work.
    """
    mgr = DockerManager(IMAGE)
    mgr.open()
    owned, shared = mgr.container, mgr.host_shared_path
    assert owned is not None and shared is not None
    fill = "mkdir -m 700 /testbed/owned && touch /testbed/owned/f"
    if case == "outlasting":
        fill += f" && printf '{_GATED_FIND}' > /usr/local/bin/find"
        fill += " && chmod +x /usr/local/bin/find"
    mgr.Bash(fill, "fill")
    assert os.path.isdir(os.path.join(shared, "owned"))
    if case == "outlasting":
        closer = threading.Thread(target=mgr.close, daemon=True)
        closer.start()
        while not os.path.exists(os.path.join(shared, "sweeping")):
            assert closer.is_alive(), "close() returned without sweeping"
            closer.join(0.1)
        closer.join(_SWEEP_TIMEOUT_S + 5)
        assert closer.is_alive(), "close() gave up on a sweep that was at work"
        owned.exec_run(["touch", "/tmp/gate"])
        closer.join(60)
        assert not closer.is_alive(), "close() did not end with its sweep"
    else:
        call = tool_interrupt.new_tool_call("close")
        tool_interrupt.register_tool_call(call)
        try:
            if case == "stopped":
                call.event.set()
            mgr.close()
        finally:
            tool_interrupt.unregister_tool_call(call)
    assert mgr.host_shared_path is None and not os.path.exists(shared)
    assert image_container_ids(docker.from_env()) == cleanup


#: Makes the container's bash run in a root that has no ``/proc`` (the image's ``sh`` and
#: ``sleep`` with their libraries), where ``find`` sits in one call for good.
_NO_PROC = (
    "mkdir -p /jail/dev && cp --parents /bin/sh /bin/sleep"
    " $(ldd /bin/sh /bin/sleep | grep -o '/lib[^ ]*' | sort -u) /jail"
    " && printf '#!/bin/sh\\nexec sleep 100000\\n' > /jail/bin/find && chmod +x /jail/bin/find"
    " && printf '#!/bin/sh\\nexec chroot /jail /bin/sh \"$@\"\\n' > /jail/bash"
    " && chmod +x /jail/bash && mv /jail/bash /bin/bash"
)


@pytest.mark.parametrize("case", ["frozen", "stuck", "unobserved"])
def test_close_gives_up_on_a_sweep_whose_stream_fell_silent(cleanup: set[str], case: str) -> None:
    """The container is removed and the lifecycle lock released while the sweep gets nowhere.

    Audit 2026-11-10 (S4-F4): ``close()`` waited for the sweep's stream with
    no timeout, the lifecycle lock held.  The container's ``find`` is replaced
    by one that leaves a marker and sits in one call for good.  Frozen: the
    container is paused once the marker is there, so the daemon has accepted
    the sweep and its stream stays silent.  Stuck: the container lives; the
    sweep's line every 5 s was printed whatever ``find`` did, and ``close()``
    waited for ever.  Unobserved (L11-F9): the sweep runs where it cannot read
    what ``find`` does, and printed its line every 5 s for that reason.
    """
    mgr = DockerManager(IMAGE)
    mgr.open()
    owned, shared = mgr.container, mgr.host_shared_path
    assert owned is not None and shared is not None
    script = "#!/bin/sh\\ntouch /testbed/sweeping\\nexec sleep 100000\\n"
    hang = f"printf '{script}' > /usr/local/bin/find && chmod +x /usr/local/bin/find"
    mgr.Bash(_NO_PROC if case == "unobserved" else hang, "make the sweep hang")
    if case == "unobserved":  # the loop of the sweep needs its sleep, and no /proc
        jailed = mgr.Bash("[ ! -e /proc ] && sleep 0 >/dev/null 2>&1 && echo jailed", "check")
        assert jailed == "jailed\n", jailed
    closer = threading.Thread(target=mgr.close, daemon=True)
    started = time.monotonic()
    closer.start()
    while case != "unobserved" and not os.path.exists(os.path.join(shared, "sweeping")):
        assert closer.is_alive(), "close() returned without sweeping"
        closer.join(0.1)
    if case == "frozen":
        owned.pause()
    closer.join(_SWEEP_TIMEOUT_S + 60)
    elapsed = time.monotonic() - started
    assert not closer.is_alive(), "close() is still waiting for the sweep's stream"
    assert elapsed >= _SWEEP_TIMEOUT_S, f"the stream was not silent: close() took {elapsed:.1f}s"
    assert image_container_ids(docker.from_env()) == cleanup
    assert mgr.container is None and mgr.container_name is None
    assert mgr.host_shared_path is None and not os.path.exists(shared)
    mgr.close()  # this thread takes the lifecycle lock the closer released


def test_close_removes_the_anonymous_volumes_of_its_container(cleanup: set[str]) -> None:
    """A volume the daemon made for an image's ``VOLUME`` goes with the container."""
    client = docker.from_env()
    image = f"kiss-test-volume:pid{os.getpid()}"
    seed = client.containers.create(IMAGE)
    seed.commit(*image.split(":"), changes=["VOLUME /data"])
    seed.remove()
    mgr = DockerManager(image, mount_shared_volume=False)
    try:
        mgr.open()
        assert mgr.container is not None
        mounts = client.api.inspect_container(mgr.container.id or "")["Mounts"]
        names = [mount["Name"] for mount in mounts if mount["Type"] == "volume"]
        mgr.close()
        left = [v for name in names for v in client.volumes.list(filters={"name": name})]
        for volume in left:
            volume.remove()
    finally:
        mgr.close()
        client.images.remove(image)
    assert names and not left, f"anonymous volumes left by close(): {left}"


def test_remove_shared_dir_when_already_absent_clears_field() -> None:
    """A tracked directory that vanished externally is simply forgotten."""
    mgr = DockerManager(IMAGE)
    mgr.host_shared_path = os.path.join("/nonexistent", "audit0902-fix")
    mgr._remove_shared_volume_dir()
    assert mgr.host_shared_path is None
    mgr._remove_shared_volume_dir()  # no path: no-op
    assert mgr.host_shared_path is None
