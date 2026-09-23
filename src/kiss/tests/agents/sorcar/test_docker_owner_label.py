# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``DockerManager.open()`` labels its containers with the starting process.

Regression for the 2026-09-23 full-suite run: the Docker tests of the main
pytest set and of the ``slow`` set ran in concurrent pytest processes, and
the test helpers identified "their" containers by image alone.  One
process's teardown force-removed another process's running container
(``test_docker_run_commands_parallel``: ``No such container`` on exec) and
the lifecycle tests counted the other process's containers as leaks
(``extra containers started``).

The manager now stamps every container it starts with ``kiss.owner=host:pid``
and the helpers filter by that label, so this test checks the contract end
to end across two real processes: a container started by a child Python
process is invisible to this process's bookkeeping, while this process's
own container is counted and removed.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator

import docker
import docker.errors
import pytest

from kiss.agents.sorcar.docker_manager import OWNER_LABEL, DockerManager, owner_label_value
from kiss.tests.agents.sorcar.docker_test_containers import (
    IMAGE,
    image_container_ids,
    remove_new_image_containers,
)


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

#: Starts a container from a second process and prints its id; the
#: container is left running so the parent can inspect it.
_CHILD_SCRIPT = f"""
from kiss.agents.sorcar.docker_manager import DockerManager
mgr = DockerManager({IMAGE!r}, mount_shared_volume=False)
mgr.open()
print(mgr.container.id)
"""


@pytest.fixture
def foreign_container_id() -> Iterator[str]:
    """The id of a container that another Python process started and owns."""
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT], capture_output=True, text=True,
        timeout=300, check=True,
    )
    container_id = proc.stdout.strip().splitlines()[-1]
    try:
        yield container_id
    finally:
        try:
            docker.from_env().api.remove_container(container_id, force=True)
        except Exception:
            pass


def test_open_labels_container_with_this_process(foreign_container_id: str) -> None:
    """A container this process opens carries ``kiss.owner=host:pid``; a
    container another process opened carries that process's value."""
    client = docker.from_env()
    before = image_container_ids(client)
    mgr = DockerManager(IMAGE, mount_shared_volume=False)
    try:
        mgr.open()
        assert mgr.container is not None
        own = client.containers.get(mgr.container.id or "")
        assert own.labels[OWNER_LABEL] == owner_label_value()
        foreign = client.containers.get(foreign_container_id)
        foreign_owner = foreign.labels[OWNER_LABEL]
        assert foreign_owner != owner_label_value()
        assert foreign_owner.rsplit(":", 1)[0] == owner_label_value().rsplit(":", 1)[0]
        assert foreign_owner.rsplit(":", 1)[1].isdigit()
    finally:
        mgr.close()
        remove_new_image_containers(client, before)


def test_helpers_ignore_containers_of_other_processes(foreign_container_id: str) -> None:
    """``image_container_ids`` counts only this process's containers and
    ``remove_new_image_containers`` leaves the foreign container running."""
    client = docker.from_env()
    before = image_container_ids(client)
    assert foreign_container_id not in before
    mgr = DockerManager(IMAGE, mount_shared_volume=False)
    try:
        mgr.open()
        assert mgr.container is not None and mgr.container.id
        own_id = mgr.container.id
        started = image_container_ids(client) - before
        assert started == {own_id}
        # The helper, not mgr.close(), must remove the container this
        # process started and nothing else.
        remove_new_image_containers(client, before)
        assert own_id not in image_container_ids(client)
        with pytest.raises(docker.errors.NotFound):
            client.containers.get(own_id)
        assert client.containers.get(foreign_container_id).status == "running"
    finally:
        # Failure-safe cleanup that does not depend on the helper under
        # test; a no-op after the helper already removed the container.
        mgr.close()
