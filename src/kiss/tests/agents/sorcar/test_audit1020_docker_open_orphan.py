# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-20 (B5/F1): a failed ``start()`` must not orphan the container.

``DockerManager.open()`` used docker-py's ``containers.run()``, which is
``create()`` + ``start()`` and hands the handle back only when both
succeeded.  When ``start()`` raised (an image without ``/bin/bash``, a bad
bind mount, a Stop injected into the task thread mid-call) the container
the daemon had already created stayed behind in ``Created`` (or
``running``) state with nothing referencing it: ``close()`` printed "No
container to close." and only a manual prune removed it.  ``open()`` now
creates and starts in two steps and force-removes the container when the
start fails.

The test runs against the real daemon (skipped when none answers) with an
image that has no bash, so the daemon itself refuses the start.
"""

from __future__ import annotations

import docker
import docker.errors
import pytest

from kiss.agents.sorcar.docker_manager import DockerManager
from kiss.tests.agents.sorcar.docker_test_containers import (
    image_container_ids,
    is_docker_available,
    remove_new_image_containers,
)

NO_BASH_IMAGE = "alpine:latest"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not is_docker_available(), reason="Docker daemon is not running"),
]


def test_failed_start_removes_the_created_container() -> None:
    """A container whose ``start()`` the daemon refuses is removed, not orphaned."""
    client = docker.from_env()
    before = image_container_ids(client)
    manager = DockerManager(NO_BASH_IMAGE, mount_shared_volume=True)
    try:
        with pytest.raises(docker.errors.APIError, match="/bin/bash"):
            manager.open()
        assert manager.container is None
        assert manager.host_shared_path is None
        leaked = image_container_ids(client) - before
        assert not leaked, f"containers orphaned by the failed open(): {leaked}"
        manager.close()  # nothing to close, must not raise
        assert image_container_ids(client) - before == set()
    finally:
        manager.close()
        remove_new_image_containers(client, before)
