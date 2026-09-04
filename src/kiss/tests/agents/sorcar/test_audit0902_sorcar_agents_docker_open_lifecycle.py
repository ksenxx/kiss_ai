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
  the container (here: an invalid port binding) the exception
  propagated with the temp dir left on disk and recorded on the
  manager, so ``close()`` — which returns early when no container is
  open — never cleaned it up either.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import docker
import docker.errors
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
def manager() -> Iterator[DockerManager]:
    """A manager whose containers are all removed after the test."""
    mgr = DockerManager(IMAGE)
    client = docker.from_env()
    before = {c.id for c in client.containers.list(all=True)}
    try:
        yield mgr
    finally:
        mgr.close()
        for container in client.containers.list(all=True):
            if container.id not in before:
                try:
                    container.remove(force=True)
                except Exception:
                    pass


def test_second_open_is_refused_and_keeps_first_container(
    manager: DockerManager,
) -> None:
    """``open()`` on an open manager must not start a second container."""
    client = docker.from_env()
    before = {c.id for c in client.containers.list(all=True)}
    manager.open()
    first = manager.container
    first_dir = manager.host_shared_path
    assert first is not None and first_dir is not None
    with pytest.raises(KISSError):
        manager.open()
    assert manager.container is first, "the first container was orphaned"
    assert manager.host_shared_path == first_dir
    started = {c.id for c in client.containers.list(all=True)} - before
    assert started == {first.id}, f"extra containers started: {started}"
    assert manager.Bash("echo still-open", "probe").strip() == "still-open"
    manager.close()
    assert manager.container is None
    assert not os.path.exists(first_dir)


def test_close_without_shared_volume_and_unremovable_dir() -> None:
    """``close()`` copes with no shared volume and with an unremovable one."""
    client = docker.from_env()
    before = {c.id for c in client.containers.list(all=True)}
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
        for container in client.containers.list(all=True):
            if container.id not in before:
                try:
                    container.remove(force=True)
                except Exception:
                    pass


def test_failed_open_removes_shared_volume_dir(manager: DockerManager) -> None:
    """A rejected ``containers.run`` must not leak the host temp dir."""
    manager.ports = {8000: 70000}  # invalid host port -> daemon rejects create
    with pytest.raises(docker.errors.APIError):
        manager.open()
    assert manager.container is None
    assert manager.host_shared_path is None, (
        f"shared volume dir leaked: {manager.host_shared_path}"
    )
    # The manager is reusable once the cause is fixed.
    manager.ports = None
    manager.open()
    assert manager.host_shared_path is not None
    assert os.path.isdir(manager.host_shared_path)
    manager.close()
    assert not os.path.exists(manager.host_shared_path or "/nonexistent")
