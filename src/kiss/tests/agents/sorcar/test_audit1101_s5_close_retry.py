# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 (S5 F6): a close whose stop failed does not forget a running container.

``DockerManager._close_locked`` caught a failed ``stop()``, then a
failed plain ``remove()`` (the daemon refuses to remove a running
container), and still cleared the handle and the name, deleted the
shared directory and printed ``Container closed successfully``: the
container ran on, owned by nobody, and a second ``close()`` had nothing
to act on.  Ownership is now cleared only once the container is gone:
a refused plain remove falls back to the force remove the failure path
of ``open()`` uses (``_discard_container``), which keeps the name on
the manager when it fails too, so a later ``close()`` retries by name.

The container and the manager are real; the one injection is a
``stop()`` of the docker SDK handle raising once, since a real daemon
cannot be asked to fail one stop RPC.  The force-remove failure branch
(the name kept for a later close) needs that second RPC to fail as
well, which no real daemon arranges: it is not covered here.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import docker
import docker.errors
import pytest

from kiss.agents.sorcar.docker_manager import DockerManager
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


def test_a_close_whose_stop_failed_still_removes_the_container(
    cleanup: None, capsys: pytest.CaptureFixture[str],
) -> None:
    """The container is force-removed, the shared directory deleted and nothing left behind."""
    manager = DockerManager(IMAGE)
    manager.open()
    owned = manager.container
    assert owned is not None and manager.host_shared_path is not None
    shared = manager.host_shared_path
    real_stop = owned.stop

    def failed_stop(*args: Any, **kwargs: Any) -> None:
        raise OSError("one transient stop RPC failure")

    owned.stop = failed_stop  # type: ignore[method-assign]
    try:
        capsys.readouterr()
        manager.close()
    finally:
        owned.stop = real_stop  # type: ignore[method-assign]
    out = capsys.readouterr().out
    assert "Failed to stop container" in out and "Failed to remove container" in out, out
    with pytest.raises(docker.errors.NotFound):
        owned.reload()
    assert manager.container is None and manager.container_name is None
    assert manager.host_shared_path is None and not os.path.exists(shared)
    assert out.rstrip().endswith("Container closed successfully"), out
    # A second close has nothing to do and says so.
    manager.close()
    assert "No container to close." in capsys.readouterr().out
