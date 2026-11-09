# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-02 (sorcar-agents): ``DockerManager.open()`` lifecycle.

Two container-lifecycle defects, both reproduced against a real docker
daemon (the tests skip when none is running):

* **Double start.**  A second ``open()`` on an already-open manager
  created a SECOND container and a second shared-volume temp dir and
  simply overwrote ``self.container`` / ``self.host_shared_path``:
  the first container kept running forever (``close()`` only knows
  the newest one) and its temp dir was never removed.
* **Failed start leaks the shared volume dir.**  ``open()`` creates
  the host temp dir BEFORE ``containers.run``; when the daemon rejects
  the container (here: a relative working directory) the exception
  propagated with the temp dir left on disk and recorded on the
  manager, so ``close()`` — which returns early when no container is
  open — never cleaned it up either.
* **A retried close deleted a mounted directory.**  A ``close()``
  whose stop and removal failed kept the container's name and the
  shared directory; the next ``close()`` retried the removal and then
  deleted the directory whether or not the retry succeeded, and
  ``open()`` deleted it too before starting a replacement container.
  The running container lost the files it still mounted.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import docker
import docker.errors
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
def manager() -> Iterator[DockerManager]:
    """A manager whose containers are all removed after the test."""
    mgr = DockerManager(IMAGE)
    client = docker.from_env()
    before = image_container_ids(client)
    try:
        yield mgr
    finally:
        mgr.close()
        remove_new_image_containers(client, before)


def test_second_open_is_refused_and_keeps_first_container(
    manager: DockerManager,
) -> None:
    """``open()`` on an open manager must not start a second container."""
    client = docker.from_env()
    before = image_container_ids(client)
    manager.open()
    first = manager.container
    first_dir = manager.host_shared_path
    assert first is not None and first_dir is not None
    first_id = first.id
    with pytest.raises(KISSError):
        manager.open()
    assert manager.container is first, "the first container was orphaned"
    assert manager.host_shared_path == first_dir
    started = image_container_ids(client) - before
    assert started == {first_id}, f"extra containers started: {started}"
    assert manager.Bash("echo still-open", "probe").strip() == "still-open"
    manager.close()
    assert manager.container is None
    assert not os.path.exists(first_dir)


def test_close_without_shared_volume_and_unremovable_dir() -> None:
    """``close()`` copes with no shared volume and with an unremovable one."""
    client = docker.from_env()
    before = image_container_ids(client)
    try:
        plain = DockerManager(IMAGE, mount_shared_volume=False)
        plain.open()
        assert plain.host_shared_path is None
        plain.close()
        assert plain.container is None and plain.host_shared_path is None

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
            # The leaked directory stays tracked so a later close() can
            # retry it (review fix #6).
            assert mgr.host_shared_path == shared
        finally:
            os.chmod(locked, 0o700)
        mgr.close()
        assert mgr.host_shared_path is None
        assert not os.path.exists(shared)
    finally:
        remove_new_image_containers(client, before)


def test_close_keeps_the_shared_dir_while_the_container_cannot_be_removed() -> None:
    """The directory a container still mounts survives every retry until the container is gone.

    After ``open()`` the SDK handle and the manager's client are pointed
    at a port nothing listens on, so the stop, the remove and the by-name
    retry of each ``close()`` fail with a real connection refused while
    the container runs on with the directory mounted.  The failed-start
    branch of ``open()`` (a discard failing right after a refused start)
    needs a daemon that fails a remove right after refusing a start, which
    no real daemon arranges; it keeps the directory by the same rule.
    """
    client = docker.from_env()
    before = image_container_ids(client)
    manager = DockerManager(IMAGE)
    manager.open()
    owned, shared = manager.container, manager.host_shared_path
    assert owned is not None and shared is not None
    with open(os.path.join(shared, "important.txt"), "w", encoding="utf-8") as fh:
        fh.write("mounted\n")
    unreachable = docker.DockerClient(
        base_url="tcp://127.0.0.1:1", version=client.api.api_version,
    )
    owned.client = manager.client = unreachable
    try:
        manager.close()
        assert manager.container is None and manager.container_name == owned.name
        manager.close()  # the retry fails again: the directory must stay mounted
        assert manager.container_name == owned.name
        assert manager.host_shared_path == shared, "a mounted directory was deleted"
        with pytest.raises(KISSError, match="could not be removed"):
            manager.open()
        assert manager.container_name == owned.name and manager.host_shared_path == shared
        running = client.containers.get(owned.id or "")
        assert running.exec_run("cat /testbed/important.txt").output == b"mounted\n"
    finally:
        manager.client = client  # the daemon is reachable again
        manager.close()
        remove_new_image_containers(client, before)
    assert manager.container_name is None and manager.host_shared_path is None
    assert not os.path.exists(shared)
    with pytest.raises(docker.errors.NotFound):
        client.containers.get(owned.id or "")


def test_failed_open_removes_shared_volume_dir(manager: DockerManager) -> None:
    """A rejected ``containers.run`` must not leak the host temp dir."""
    manager.workdir = "relative"  # not absolute -> daemon rejects create
    with pytest.raises(docker.errors.APIError, match="absolute path"):
        manager.open()
    assert manager.container is None
    assert manager.host_shared_path is None, (
        f"shared volume dir leaked: {manager.host_shared_path}"
    )
    # The manager is reusable once the cause is fixed.
    manager.workdir = "/"
    manager.open()
    assert manager.host_shared_path is not None
    assert os.path.isdir(manager.host_shared_path)
    manager.close()
    assert not os.path.exists(manager.host_shared_path or "/nonexistent")
